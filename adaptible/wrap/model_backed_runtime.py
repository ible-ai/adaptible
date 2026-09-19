"""A runtime whose answers come from a real model, not a hardcoded string.

``wrap_test.FakeRuntime`` stipulates the outcome: it returns "Sydney" for the
base handle and "Canberra" for any adapter, so a repair "succeeds" without a
single weight changing. That leaves the part the product is actually made of --
train a LoRA, apply it, and see the answer change -- untested in the loop.

This runtime generates with a real tiny Transformers model and applies the real
adapter directory the worker just wrote, so acceptance is decided by what the
weights do. The model is randomly initialised, so it cannot read evidence or
emit JSON; those structured helper calls stay scripted, exactly as they are in
``FakeRuntime``. What becomes real is every call whose answer the loop is
trying to change.
"""

import json
import re
from pathlib import Path

import httpx

from adaptible.wrap.thinking import completion_details

# Scripted helper answers, keyed by the fixture below. Extraction, judging and
# paraphrasing need a competent model; a two-layer random one cannot do them.
FACT = {
    "question": "What is the largest city in Morocco?",
    "answer": "Casablanca",
    "document": "Casablanca is the largest city in Morocco.",
    "variants": (
        "Which city is the largest in Morocco?",
        "Name Morocco's largest city.",
        "What is Morocco's biggest city?",
    ),
}

CONTROLS = {
    "France": "Paris.",
    "12 times": "144.",
    "days": "7.",
}

# Every short answer the scripted grounding helper may be asked to confirm:
# the repaired fact plus each control's expected answer.
KNOWN_NAMES = (FACT["answer"], "Paris", "144", "7")


def _framed(answer, thinking):
    """Give a scripted answer a completed thought when the caller wants one."""
    if not thinking or "</think>" in answer:
        return answer
    return f"<think>\nRecalling the answer.\n</think>\n\n{answer}"


def scripted_answer(prompt, response_format=None, thinking=False):
    """Helper answers a toy model cannot produce. ``None`` means generate.

    Extraction, evidence checking and paraphrasing need a competent model. The
    loop's own decisions are not scripted: only these inputs to them are.
    """
    schema = (response_format or {}).get("json_schema", {}).get("name")
    if "<reference_material>" in prompt:
        # The grounded reasoning draft: a competent frozen base writes this from
        # the evidence. Training on it, and every check after, stays real.
        return _framed(FACT["answer"], True)
    if schema == "evidence_check":
        return json.dumps(dict(answers_question=True, supported=True))
    if schema == "question_variants":
        return json.dumps(dict(questions=list(FACT["variants"])))
    if prompt.startswith("Reference sentences:"):
        reference = prompt.split("\nQuestion:", 1)[0]
        for line in reference.splitlines()[1:]:
            index, sentence = line.split(": ", 1)
            for name in KNOWN_NAMES:
                if name in sentence:
                    return json.dumps(dict(name=name, sentence_index=int(index)))
        return json.dumps(dict(name="", sentence_index=-1))
    if prompt.startswith("Reference text:"):
        reference = prompt.split("\nQuestion:", 1)[0]
        for name in KNOWN_NAMES:
            if name in reference:
                return json.dumps(
                    dict(quote=reference.removeprefix("Reference text: "), name=name)
                )
        return ""
    if prompt.startswith("Reference note:"):
        return FACT["answer"] + "."
    for marker, answer in CONTROLS.items():
        if marker in prompt:
            return _framed(answer, thinking)
    return None


class TinyGenerator:
    """Greedy generation from a tiny checkpoint, with optional LoRA applied."""

    def __init__(self, blob, *, max_tokens=24, gguf_file=None):
        self.blob = Path(blob)
        self.gguf_file = gguf_file
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._adapters = {}

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            options = {} if self.gguf_file is None else {"gguf_file": self.gguf_file}
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.blob, local_files_only=True, **options
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.blob, local_files_only=True, dtype=torch.float32, **options
            ).eval()
        return self._model, self._tokenizer

    def forget(self, adapter):
        self._adapters.pop(str(adapter), None)

    def _fresh_base(self):
        import torch
        from transformers import AutoModelForCausalLM

        options = {} if self.gguf_file is None else {"gguf_file": self.gguf_file}
        return AutoModelForCausalLM.from_pretrained(
            self.blob, local_files_only=True, dtype=torch.float32, **options
        ).eval()

    def _model_for(self, adapter):
        base, _ = self._load()
        if not adapter:
            return base
        key = str(adapter)
        if key not in self._adapters:
            from peft import PeftModel

            # PeftModel.from_pretrained injects LoRA layers into the model it is
            # given. Reusing one base would stack every adapter onto it and
            # corrupt both the adapted and the unadapted answers.
            self._adapters[key] = PeftModel.from_pretrained(
                self._fresh_base(), key
            ).eval()
        return self._adapters[key]

    def generate(self, question, adapter=None):
        """Greedy-decode one answer, so a run is reproducible."""
        import torch

        _, tokenizer = self._load()
        model = self._model_for(adapter)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][ids.shape[1] :], skip_special_tokens=True
        ).strip()


class ModelBackedRuntime:
    """Serves a real tiny model; swaps in the real adapter the trainer wrote."""

    name = "local-model"
    native = False
    url = "http://runtime"
    max_tokens = 24
    # The non-thinking codepath: a random model cannot emit valid reasoning.
    architecture = "qwen2"

    def __init__(self, blob, *, learning_rate=1e-3):
        self.blob = Path(blob)
        self.learning_rate = learning_rate
        self.active = None
        self.suspended = False
        self.staged = {}
        self.generated = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._respond))
        self.generator = TinyGenerator(self.blob, max_tokens=self.max_tokens)

    # ---- real generation -------------------------------------------------

    def generate(self, question, handle):
        adapter = Path(self.staged.get(handle, handle)) / "adapter" if handle else None
        text = self.generator.generate(question, adapter)
        self.generated.append((question, handle, text))
        return text

    # ---- runtime contract ------------------------------------------------

    def training_options(self):
        # A random tiny model needs a rate suited to it; the production default
        # moves it too little to learn inside the 64-update bound.
        return {"learning_rate": self.learning_rate}

    def payload(self, body, handle=None, frozen=False):
        return {**body, "model": "base" if frozen else handle or self.active or "base"}

    async def complete(self, messages, handle=None, frozen=False, **kwargs):
        assert not self.suspended, "Generation attempted before serving resumed"
        prompt = messages[-1]["content"]
        raw = scripted_answer(
            prompt, kwargs.get("response_format"), kwargs.get("thinking", False)
        )
        if raw is None:
            raw = self.generate(prompt, None if frozen else handle or self.active)
        # Match Runtime.complete: callers read reasoning and framing from here,
        # and the whole thinking path depends on it being populated.
        result = completion_details(dict(content=raw), "stop")
        details = kwargs.get("details")
        if details is not None:
            details.update(result)
        return result["content"]

    async def stage(self, directory):
        self.suspended = False
        handle = str(directory)
        self.staged[handle] = str(directory)
        self.generator.forget(Path(directory) / "adapter")
        return handle

    async def restore(self, handle):
        self.suspended = False
        self.active = handle

    async def release_for_training(self):
        self.suspended = True

    async def discard(self, handle):
        self.generator.forget(Path(self.staged.get(handle, handle)) / "adapter")

    async def close(self):
        await self.client.aclose()

    # ---- client-facing proxy --------------------------------------------

    def _respond(self, request):
        body = json.loads(request.content)
        handle = body.get("model")
        question = body["messages"][-1]["content"]
        text = self.generate(question, None if handle == "base" else handle)
        if body.get("stream"):
            parts = [
                dict(model=handle, choices=[dict(delta=dict(content=piece))])
                for piece in (text[:2], text[2:])
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
                + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json=dict(
                model=handle,
                choices=[dict(message=dict(content=text), finish_reason="stop")],
                message=dict(content=text),
            ),
        )


class FakeProcess:
    """The part of ``asyncio.subprocess.Process`` the wrapper actually uses."""

    _next_pid = -424242

    def __init__(self, output=b""):
        self.returncode = None
        self.output = output
        FakeProcess._next_pid -= 1
        self.pid = FakeProcess._next_pid

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    async def communicate(self, _input=None):
        self.returncode = 0
        return self.output, None


class SpawnInterceptor:
    """Stand in for one executable's child processes, passing everything else.

    ``runtime.asyncio`` is the asyncio module itself, so a patch here is global.
    The training worker must still really be spawned, so only commands whose
    argv[0] matches are intercepted.
    """

    def __init__(self, executables, output=b""):
        self.executables = {str(e) for e in executables}
        self.output = output
        self.commands = []
        self.processes = []
        self._patches = []

    def matches(self, args):
        return bool(args) and str(args[0]) in self.executables

    def start(self):
        import asyncio
        import os
        from unittest import mock

        real_spawn = asyncio.create_subprocess_exec
        real_killpg = os.killpg

        async def spawn(*args, **kwargs):
            if not self.matches(args):
                return await real_spawn(*args, **kwargs)
            self.commands.append([str(a) for a in args])
            process = FakeProcess(self.output)
            self.processes.append(process)
            return process

        def killpg(pid, signal_number):
            # Never signal a real process group on behalf of a fake pid; pid 0
            # would target this test runner's own group.
            if any(process.pid == pid for process in self.processes):
                return None
            return real_killpg(pid, signal_number)

        self._patches = [
            mock.patch.object(asyncio, "create_subprocess_exec", side_effect=spawn),
            mock.patch.object(os, "killpg", side_effect=killpg),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def stop(self):
        for patch in self._patches:
            patch.stop()
        self._patches = []


class LlamaCppHarness:
    """Stands in for ``llama-server`` so the real ``LlamaCpp`` runtime can run.

    Intercepts exactly two things: the child process it spawns, and the socket
    it talks to. Everything else -- the command line, the readiness poll, the
    ``lora`` scale in each request, stage/restore/discard, stopping the server
    before training -- is the shipped code.
    """

    def __init__(self, runtime, generator):
        self.runtime = runtime
        self.generator = generator
        self.launches = []
        self.stopped = 0
        self._patch = None

    def adapter_for(self, handle):
        """The PEFT directory beside the exported ``adapter.gguf``."""
        return Path(handle).parent / "adapter" if handle else None

    def _respond(self, request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        question = body["messages"][-1]["content"]
        text = scripted_answer(question, body.get("response_format"))
        if text is None:
            # llama.cpp applies a loaded adapter only at the scale it is given.
            scales = body.get("lora") or []
            applied = bool(scales) and bool(scales[0].get("scale"))
            text = self.generator.generate(
                question, self.adapter_for(self.runtime.loaded) if applied else None
            )
        if body.get("stream"):
            parts = [
                dict(choices=[dict(delta=dict(content=piece))])
                for piece in (text[:2], text[2:])
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
                + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json=dict(
                choices=[dict(message=dict(content=text), finish_reason="stop")],
            ),
        )

    def __enter__(self):
        self._patch = SpawnInterceptor([self.runtime.executable]).start()
        self.launches = self._patch.commands
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._respond)
        )
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def _openai_response(text, stream=False):
    """One OpenAI chat completion, streamed or not."""
    if stream:
        parts = [
            dict(choices=[dict(delta=dict(content=piece))])
            for piece in (text[:2], text[2:])
        ]
        return httpx.Response(
            200,
            text="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
            + "data: [DONE]\n\n",
        )
    return httpx.Response(
        200,
        json=dict(choices=[dict(message=dict(content=text), finish_reason="stop")]),
    )


class VLLMHarness:
    """Stands in for a managed ``vllm serve`` process.

    The child process, its process group and the socket are intercepted. Adapter
    registration, the private handle, ``/v1/load_lora_adapter`` and selection by
    model name are the shipped code.
    """

    def __init__(self, runtime, generator):
        self.runtime = runtime
        self.generator = generator
        self.loaded = {}
        self.unloaded = []
        self._spawn = None

    def _respond(self, request):
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={})
        if path == "/v1/models":
            models = [{"id": self.runtime.name}] + [
                {"id": name} for name in self.loaded
            ]
            return httpx.Response(200, json={"data": models})
        body = json.loads(request.content)
        if path == "/v1/load_lora_adapter":
            self.loaded[body["lora_name"]] = Path(body["lora_path"])
            return httpx.Response(200, json={})
        if path == "/v1/unload_lora_adapter":
            self.unloaded.append(body["lora_name"])
            self.loaded.pop(body["lora_name"], None)
            return httpx.Response(200, json={})
        question = body["messages"][-1]["content"]
        text = scripted_answer(question, body.get("response_format"))
        if text is None:
            text = self.generator.generate(question, self.loaded.get(body["model"]))
        return _openai_response(text, body.get("stream"))

    def __enter__(self):
        self._spawn = SpawnInterceptor([self.runtime.executable]).start()
        self.commands = self._spawn.commands
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._respond)
        )
        return self

    def __exit__(self, *exc):
        self._spawn.stop()
        return False


class LMStudioHarness:
    """Stands in for the LM Studio server and its ``lms`` CLI.

    Fusion is not intercepted: ``stage`` really runs the GGUF fusion worker, and
    a loaded model here is served from the fused file it produced.
    """

    def __init__(self, runtime, base_gguf):
        self.runtime = runtime
        self.base_gguf = Path(base_gguf)
        self.models = {}
        self.instances = {}
        self.loads = []
        self.unloads = []
        self._generators = {}
        self._spawn = None
        self._next_instance = 0

    def _generator_for(self, path):
        key = str(path)
        if key not in self._generators:
            self._generators[key] = TinyGenerator(
                Path(path).parent, gguf_file=Path(path).name
            )
        return self._generators[key]

    def _respond(self, request):
        path = request.url.path
        if path == "/api/v1/models":
            return httpx.Response(
                200, json={"models": [{"key": key} for key in self.models]}
            )
        body = json.loads(request.content)
        if path == "/api/v1/models/unload":
            self.unloads.append(body["instance_id"])
            self.instances.pop(body["instance_id"], None)
            return httpx.Response(200, json={})
        question = body["messages"][-1]["content"]
        text = scripted_answer(question, body.get("response_format"))
        if text is None:
            served = self.instances[body["model"]]
            text = self._generator_for(served).generate(question)
        return _openai_response(text, body.get("stream"))

    def _import(self, command):
        """Register what ``lms import --user-repo <repo> <path>`` would create."""
        path = Path(command[2])
        repo = command[command.index("--user-repo") + 1]
        self.models[repo] = path
        return f"Symbolic link created at  {path}\n".encode()

    def __enter__(self):
        self._spawn = SpawnInterceptor([self.runtime.executable])
        original = self._spawn.matches

        def matches(args):
            if original(args) and len(args) > 2 and args[1] == "import":
                self._spawn.output = self._import([str(a) for a in args])
                return True
            return original(args)

        self._spawn.matches = matches
        self._spawn.start()
        self.commands = self._spawn.commands
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._respond)
        )
        # Models load through LM Studio's SDK API -- the only one that takes a
        # KV cache type -- and are then served over REST, which is what this
        # harness answers.
        self.runtime._load_instance = self._load_instance
        return self

    async def _load_instance(self, key):
        self._next_instance += 1
        instance = f"instance-{self._next_instance}"
        self.instances[instance] = self.models[key]
        self.loads.append(key)
        return instance

    def __exit__(self, *exc):
        self._spawn.stop()
        return False


class OllamaHarness:
    """Stands in for a running Ollama service and its CLI.

    Private tags, the Modelfile written for ``ollama create``, unloading through
    ``/api/ps`` plus ``keep_alive=0``, and tag deletion are the shipped code.
    """

    def __init__(self, runtime, generator, *, modelfile="FROM tiny\n", template=""):
        self.runtime = runtime
        self.generator = generator
        self.modelfile = modelfile
        self.template = template
        self.tags = {}
        # Parameters per model name. The wrapper pins its sampler on a tag it
        # creates for itself (Ollama's house repeat_penalty/top_k/top_p are not
        # the experiment's and cannot be sent per request), so the stub has to
        # model create-then-show rather than report one fixed Modelfile.
        self.parameters = {}
        self.resident = set()
        self.deleted = []
        self._spawn = None

    def adapter_for(self, tag):
        directory = self.tags.get(tag)
        return Path(directory) / "adapter" if directory else None

    def _create(self, command):
        """Register the tag ``ollama create <name> -f <Modelfile>`` would make."""
        name = command[2]
        modelfile = Path(command[command.index("-f") + 1])
        adapter = re.search(r'^ADAPTER\s+"?(.+?)"?\s*$', modelfile.read_text(), re.M)
        self.tags[name] = Path(adapter[1]).parent if adapter else None
        return b"success\n"

    def _respond(self, request):
        path = request.url.path
        if path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": name} for name in sorted(self.resident)]}
            )
        if path == "/api/delete":
            body = json.loads(request.content)
            self.deleted.append(body["model"])
            self.tags.pop(body["model"], None)
            return httpx.Response(200, json={})
        body = json.loads(request.content)
        if path == "/api/show":
            name = body.get("model", "")
            return httpx.Response(
                200,
                json={
                    "modelfile": self.modelfile,
                    "template": self.template,
                    "system": "",
                    "details": {},
                    "model_info": {},
                    "parameters": self.parameters.get(name, ""),
                },
            )
        if path == "/api/create":
            self.parameters[body["model"]] = "\n".join(
                f"{key} {value}" for key, value in body["parameters"].items()
            )
            self.tags.setdefault(body["model"], None)
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/generate":
            # An empty prompt with keep_alive=0 is how the wrapper unloads.
            if not body.get("prompt") and body.get("keep_alive") == 0:
                self.resident.discard(body["model"])
                return httpx.Response(200, json={"response": "", "done": True})
            text = self.generator.generate(body["prompt"], None)
            return httpx.Response(200, json={"response": text, "done": True})
        question = body["messages"][-1]["content"]
        text = scripted_answer(question, body.get("response_format"))
        if text is None:
            model = body["model"]
            self.resident.add(
                model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
            )
            text = self.generator.generate(question, self.adapter_for(model))
        return _openai_response(text, body.get("stream"))

    def __enter__(self):
        from unittest import mock

        self._which = mock.patch(
            "adaptible.wrap.runtime.shutil.which", return_value="/usr/bin/ollama"
        )
        self._which.start()
        self._spawn = SpawnInterceptor(["/usr/bin/ollama"])
        original = self._spawn.matches

        def matches(args):
            if original(args) and len(args) > 2 and args[1] == "create":
                self._spawn.output = self._create([str(a) for a in args])
                return True
            return original(args)

        self._spawn.matches = matches
        self._spawn.start()
        self.commands = self._spawn.commands
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._respond)
        )
        return self

    def __exit__(self, *exc):
        self._spawn.stop()
        self._which.stop()
        return False
