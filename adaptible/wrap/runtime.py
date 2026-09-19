"""Small runtime-specific hooks: discover, generate, stage, activate, discard."""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sys
from pathlib import Path

import httpx

from . import config
from .ollama import remove_fused_file, track_fused_file
from .prompt_format import OLLAMA_QWEN3_FORMAT, OLLAMA_QWEN3_TEMPLATE_SHA256
from .thinking import completion_details, generation_mode, thinking_complete

logger = logging.getLogger(__name__)

_COMMAND_SHUTDOWN_TIMEOUT = 3.0


async def run_command(*args, env=None):
    process = None
    creation = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *map(str, args),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    )
    try:
        # Cancellation during process creation must not lose the child handle.
        process = await asyncio.shield(creation)
        output, _ = await process.communicate()
    except BaseException:
        if process is None:
            try:
                process = await creation
            except (Exception, asyncio.CancelledError):
                logger.debug("No child to reap: process creation did not complete.")
        if process is not None:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    logger.debug("Child already exited before terminate().")
            # Drain output while reaping: wait() alone can hang with a full pipe.
            cleanup = asyncio.create_task(process.communicate())
            try:
                await asyncio.wait_for(
                    asyncio.shield(cleanup), _COMMAND_SHUTDOWN_TIMEOUT
                )
            except TimeoutError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        logger.debug("Child already exited before kill().")
                await cleanup
        raise
    if process.returncode:
        raise RuntimeError(output.decode(errors="replace")[-2000:])
    return output.decode(errors="replace")


def fingerprint(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# The original decodes with mlx_lm's `make_sampler(temp=...)`: temperature and
# nothing else. Every llama.cpp-derived server ships its own house defaults
# instead -- repeat_penalty 1.1 and a truncated tail -- and applies them to
# every request. Measured on Ollama against identical f32 weights: 2311
# characters returned where the original returned 2247, and repeat_penalty 1.0
# reproduced the original's exactly. A repetition penalty biases the argmax, so
# it changes greedy output; top_k/top_p/min_p do not bite under greedy decoding
# but truncate the distribution a correction is sampled from at temperature
# 0.7, which is where candidates come from.
NEUTRAL_SAMPLING = {
    "repeat_penalty": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
}


# llama.cpp's Metal backend computes at reduced precision in three independent
# places by default, whatever precision the GGUF is stored in:
#
#   - prompt prefill: `kernel_mul_mm_f32_f32` stages both operands as `half` for
#     any matrix product over more than 8 tokens (ggml-metal mul_mm.metal).
#     `-ub 8` keeps every prefill chunk on the true-f32 matrix-vector kernel;
#   - the KV cache is stored at f16 unless told otherwise;
#   - flash attention, on by default on Metal, accumulates at reduced precision.
#
# Measured on one model served both ways from bit-identical weights, the
# next-token log-probabilities differ from MLX's by 3.7e-4 at llama.cpp's
# defaults and by 1.4e-6 with all three pinned -- f32 rounding. Each one alone
# leaves the gap at 1e-4 or worse, because the other two are still there, so
# all three are needed. A difference of 1e-4 is enough to flip a greedy argmax
# on a near-tie late in a long generation, which is how three ggml runtimes
# came to diverge from the original at the same character of the same answer.
LLAMA_CPP_FULL_PRECISION = (
    "--ubatch-size",
    "8",
    "--cache-type-k",
    "f32",
    "--cache-type-v",
    "f32",
    "--flash-attn",
    "off",
)


class Runtime:
    native = False

    def __init__(self, name, directory, max_tokens=768):
        self.name, self.directory, self.max_tokens = name, Path(directory), max_tokens
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active = None
        self.architecture = None
        # Set by detect_reasoning(): whether this model reasons on every
        # request with no control to disable it.
        self.always_reasons = False
        self.serving_template_digest = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5))

    def payload(self, body, *, handle=None, frozen=False):
        raise NotImplementedError

    async def prepare_payload(self, body, *, frozen=False):
        """Prepare public routing while the caller holds the generation gate."""
        return self.payload(body, frozen=frozen)

    def normalize_payload(self, body, *, path="/v1/chat/completions"):
        """Translate verified runtime controls after recording the client's mode."""
        return body

    def generation_timeout(self, thinking=False, max_tokens=None):
        """Allow time to generate the budget that was actually granted.

        Deriving this from the declared mode instead of ``max_tokens`` times out
        on any model that reasons without being asked to: the wrapper only marks
        a request as thinking for Qwen3, while a distilled reasoning model on
        another architecture reasons on every call, including the bounded
        structured ones. Native CPU reasoning runs at a few tokens per second.
        """
        seconds = min(1800, max(180, (max_tokens or self.max_tokens) / 2 + 60))
        return httpx.Timeout(seconds, connect=5)

    async def complete(
        self,
        messages,
        *,
        handle=None,
        frozen=False,
        temperature=0,
        seed=0,
        response_format=None,
        thinking=False,
        details=None,
        max_tokens=None,
    ):
        # A caller that knows its request needs more room than a served turn
        # says so; everything else keeps the serving budget.
        budget = max_tokens or self.max_tokens
        body = self.payload(
            dict(
                messages=messages,
                stream=False,
                temperature=temperature,
                seed=seed,
                max_tokens=budget,
            ),
            handle=handle,
            frozen=frozen,
        )
        if response_format is not None:
            body["response_format"] = response_format
        if self.architecture == "qwen3":
            # Actual re-asks preserve the interaction's mode; structured
            # evidence helpers use the explicit nonthinking default.
            body["reasoning_effort"] = "medium" if thinking else "none"
            if not self.native:
                body["chat_template_kwargs"] = {"enable_thinking": thinking}
        body = self.normalize_payload(body)
        response = await self.client.post(
            self.url + "/v1/chat/completions",
            json=body,
            timeout=self.generation_timeout(thinking, budget),
        )
        response.raise_for_status()

        choice = response.json()["choices"][0]
        result = completion_details(choice["message"], choice.get("finish_reason"))
        if details is not None:
            details.update(result)
        return result["content"]

    async def detect_reasoning(self):
        """Ask once whether this model reasons without being told to.

        Qwen3 takes an explicit control, so it is never probed. Anything else
        may still be a distilled reasoning model, and the only reliable signal
        is whether a plain request comes back carrying a completed thought.
        A failed probe leaves the flag false: an unreasoning model is the safe
        assumption, and repair simply uses the non-reasoning path.
        """
        if self.architecture == "qwen3":
            return False
        details = {}
        try:
            await self.complete(
                [dict(role="user", content="Reply with the word ready.")],
                frozen=True,
                details=details,
            )
        except Exception:
            logger.warning("Could not probe reasoning behaviour", exc_info=True)
            return False
        self.always_reasons = thinking_complete(details)
        logger.info("Model reasons unconditionally: %s", self.always_reasons)
        return self.always_reasons

    async def restore(self, handle):
        self.active = handle

    def training_options(self):
        return {}

    async def release_for_training(self):
        """Release serving weights before the dense training worker starts."""
        raise NotImplementedError

    async def close(self):
        await self.client.aclose()


class Ollama(Runtime):
    native = True

    def __init__(
        self, name, directory, url=config.OLLAMA_URL, context_size=None, **kwargs
    ):
        super().__init__(name, directory, **kwargs)
        if context_size is not None and (
            isinstance(context_size, bool)
            or not isinstance(context_size, int)
            or context_size < 128
        ):
            raise ValueError("Ollama context size must be an integer of at least 128.")
        self.context_size = context_size
        self.base_handle = name
        self.context_record = self.directory / "ollama-context.json"
        self.context_create_task = None
        self.url = url.rstrip("/")
        self.executable = shutil.which("ollama")
        if not self.executable:
            raise ValueError(
                "Ollama is required. Start its local service and put ollama on PATH."
            )
        self.env = {**os.environ, "OLLAMA_HOST": self.url}
        self.prefix = (
            "adaptible-"
            + hashlib.sha256(str(self.directory.resolve()).encode()).hexdigest()[:12]
            + "-"
        )

    async def discover(self):
        # Reconcile a context tag left by an interrupted previous process.
        await self._cleanup_context()
        response = await self.client.post(
            self.url + "/api/show", json={"model": self.name}
        )
        response.raise_for_status()
        info = response.json()
        self.modelfile = info["modelfile"]
        self.template = info.get("template", "")
        self.system = info.get("system", "")
        prompt_settings = dict(template=self.template, system=self.system)
        if re.search(r"^MESSAGE\s", self.modelfile, re.M):
            # MESSAGE may contain quoted multiline text. Pin the complete
            # serialized Modelfile rather than guessing its parsing here.
            prompt_settings["configured_history"] = self.modelfile
        self.serving_template_digest = hashlib.sha256(
            json.dumps(
                prompt_settings,
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if re.search(r"^ADAPTER\s", self.modelfile, re.M):
            raise ValueError(
                "Start with an installed base model, not a model with an existing adapter."
            )
        match = re.search(r"^FROM\s+(.+)$", self.modelfile, re.M)
        if not match:
            raise ValueError("Ollama did not report a local model file.")
        self.blob = Path(match[1].strip().strip('"')).resolve(strict=True)
        if not self.blob.is_file():
            raise ValueError("Ollama's model file must be accessible on this machine.")
        # Unconditional: the sampling parameters have to be pinned whether or
        # not a context size was asked for, or a wrapper started without
        # --context-size silently serves under Ollama's house sampler.
        await self._configure_context(info)
        return self.blob

    @staticmethod
    def _parameter_lines(info):
        return [
            line.strip()
            for line in info.get("parameters", "").splitlines()
            if line.strip()
        ]

    @classmethod
    def _parameter_map(cls, info):
        values = {}
        for line in cls._parameter_lines(info):
            key, _, value = line.partition(" ")
            values.setdefault(key, []).append(value.strip())
        return values

    @staticmethod
    def _context_parameters(info):
        lines = info.get("parameters", "").splitlines()
        contexts = [
            line.split()[1:] for line in lines if line.split()[:1] == ["num_ctx"]
        ]
        context = (
            int(contexts[0][0])
            if len(contexts) == 1 and len(contexts[0]) == 1
            else None
        )
        other = sorted(
            line.strip()
            for line in lines
            if line.strip() and line.split()[0] != "num_ctx"
        )
        return context, other

    def _serving_parameters(self):
        """What the wrapper pins on the model it serves through.

        The original decodes with mlx_lm's `make_sampler(temp=...)`, which is
        temperature and nothing else: no repetition penalty and no truncation.
        Ollama applies its own house defaults to every request instead, and
        they change the output. Measured on the same greedy prompt against the
        identical f32 weights, Ollama returned 2311 characters where the
        original returned 2247, and pinning `repeat_penalty` to 1.0 reproduced
        the original's 2247 exactly, character for character. `top_k` and
        `top_p` do not show up under greedy decoding but truncate the
        distribution a correction is sampled from at temperature 0.7, which is
        where candidates come from.

        These cannot be sent per request: neither `repeat_penalty`, `top_k`,
        `frequency_penalty` nor a nested `options` object reaches the sampler
        through Ollama's OpenAI-compatible route (all four measured, all four
        ignored). The model the wrapper creates for itself is the only place
        they can be pinned.
        """
        desired = dict(NEUTRAL_SAMPLING)
        # Ollama runs ggml's Metal kernels, whose matrix-matrix path stages
        # prefill operands at f16 for any batch over 8 tokens (see
        # `LLAMA_CPP_FULL_PRECISION`). A batch of 8 keeps prefill on the f32
        # kernel. Flash attention and the KV cache type are the other two
        # reduced-precision defaults, but Ollama only sets those per *server*
        # (`OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`), and its KV cache
        # offers f16, q8_0 and q4_0 -- no f32 -- so this is the one of the
        # three a model tag can carry.
        desired["num_batch"] = 8
        if self.context_size is not None:
            desired["num_ctx"] = self.context_size
        return desired

    @staticmethod
    def _satisfies(current, desired):
        for key, value in desired.items():
            got = current.get(key)
            if got is None or len(got) != 1:
                return False
            try:
                if float(got[0]) != float(value):
                    return False
            except ValueError:
                return False
        return True

    async def _configure_context(self, original):
        desired = self._serving_parameters()
        if self._satisfies(self._parameter_map(original), desired):
            return
        handle = (
            self.prefix
            + "context-"
            + (str(self.context_size) if self.context_size is not None else "default")
        )
        if handle == self.name:
            raise ValueError(
                "The requested original model aliases this wrapper's context tag."
            )
        # /api/create inherits existing layers by digest; no weight file upload,
        # checkpoint conversion, or modification of the original model occurs.
        record = dict(model=handle, original=self.name, context_size=self.context_size)
        pending = self.context_record.with_suffix(".json.pending")
        pending.write_text(json.dumps(record))
        pending.replace(self.context_record)
        try:
            self.context_create_task = asyncio.create_task(
                self.client.post(
                    self.url + "/api/create",
                    json={
                        "model": handle,
                        "from": self.name,
                        "parameters": dict(desired),
                        "stream": False,
                    },
                )
            )
            response = await asyncio.shield(self.context_create_task)
            response.raise_for_status()
            if response.json().get("status") != "success":
                raise RuntimeError(
                    "Ollama did not finish creating the private context tag."
                )
            response = await self.client.post(
                self.url + "/api/show", json={"model": handle}
            )
            response.raise_for_status()
            info = response.json()
            # Everything the wrapper did not ask to change must come through
            # untouched; what it did ask for must actually have taken.
            preserved = sorted(
                line
                for line in self._parameter_lines(info)
                if line.partition(" ")[0] not in desired
            )
            inherited = sorted(
                line
                for line in self._parameter_lines(original)
                if line.partition(" ")[0] not in desired
            )
            source = re.search(r"^FROM\s+(.+)$", info.get("modelfile", ""), re.M)
            if (
                not self._satisfies(self._parameter_map(info), desired)
                or preserved != inherited
                or source is None
                or Path(source[1].strip().strip('"')).resolve() != self.blob
                or any(
                    info.get(key, default) != original.get(key, default)
                    for key, default in (
                        ("template", ""),
                        ("system", ""),
                        ("messages", []),
                    )
                )
            ):
                raise ValueError(
                    "Ollama's private serving tag changed its base/prompt settings or did not honor the pinned parameters."
                )
            self.modelfile = info["modelfile"]
            self.base_handle = handle
        except BaseException:
            await self._cleanup_context()
            raise

    async def _cleanup_context(self):
        if not self.context_record.exists():
            return
        record = json.loads(self.context_record.read_text())
        handle = record.get("model", "")
        if not re.fullmatch(
            re.escape(self.prefix) + r"context-(\d+|default)", handle
        ) or handle in (self.name, self.active):
            raise ValueError("Refusing to remove an unowned Ollama context tag.")
        task = self.context_create_task
        if task is not None and not task.done():
            # Do not delete before an in-flight metadata creation finishes.
            # On timeout the ownership record remains for later reconciliation.
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30)
            except TimeoutError as exc:
                raise RuntimeError(
                    "Ollama context creation is still pending; ownership retained for cleanup."
                ) from exc
            except Exception:
                logger.warning(
                    "Ollama context creation failed; continuing with tag removal.",
                    exc_info=True,
                )
        await self._delete_private_tag(handle)
        self.context_record.unlink(missing_ok=True)
        self.base_handle = self.name

    async def close(self):
        if self.client.is_closed:
            return
        try:
            await self._cleanup_context()
        finally:
            await super().close()

    def training_options(self):
        if self.architecture != "qwen3":
            return {}

        if re.search(r"^MESSAGE\s", self.modelfile, re.M):
            raise ValueError(
                "Ollama Qwen3 training does not yet support Modelfile MESSAGE history. "
                "Its extra conversation prefix must be reproduced before training."
            )
        digest = hashlib.sha256(self.template.encode()).hexdigest()
        if digest != OLLAMA_QWEN3_TEMPLATE_SHA256:
            raise ValueError(
                "This Ollama Qwen3 template is not supported for training. "
                "Chat remains available; repair requires a verified native prompt "
                "format rather than assuming the GGUF's HF template matches."
            )
        return dict(
            prompt_format=OLLAMA_QWEN3_FORMAT,
            template_sha256=digest,
            system=self.system,
        )

    def payload(self, body, *, handle=None, frozen=False):
        return {
            **body,
            "model": (
                self.base_handle
                if frozen
                else (handle or self.active or self.base_handle)
            ),
        }

    def normalize_payload(self, body, *, path="/v1/chat/completions"):
        """Map stock Qwen3's requested effort levels to its binary thinking.

        Ollama 0.12 rejects level strings for Qwen3. Its thinking-capable model
        default is boolean true, set before rendering the stock /think prefix.
        Other models/templates and invalid or conflicting controls pass through.

        A model that reasons on every turn is asked for its thought explicitly.
        Ollama's native API drops the reasoning unless ``think`` is set, so the
        wrapper would mark such a turn as thinking, receive no thought, record
        every turn incomplete, and refuse to accept feedback on any of them.
        """
        if path.startswith("/api/") and self.always_reasons and "think" not in body:
            body = {**body, "think": True}
        if (
            self.architecture != "qwen3"
            or not isinstance(getattr(self, "template", None), str)
            or hashlib.sha256(self.template.encode()).hexdigest()
            != OLLAMA_QWEN3_TEMPLATE_SHA256
            or path not in ("/v1/chat/completions", "/api/chat", "/api/generate")
        ):
            return body
        mode = generation_mode(
            body,
            self.architecture,
            native=True,
            path=path,
            always_reasons=self.always_reasons,
        )
        if mode.get("error") or not mode.get("thinking"):
            return body
        copied = dict(body)
        if path.startswith("/api/"):
            if copied.get("think") in ("low", "medium", "high"):
                copied["think"] = True
        else:
            if copied.get("reasoning_effort") in ("low", "medium", "high"):
                copied.pop("reasoning_effort")
            reasoning = copied.get("reasoning")
            if isinstance(reasoning, dict) and reasoning.get("effort") in (
                "low",
                "medium",
                "high",
            ):
                copied.pop("reasoning")
        return copied

    async def stage(self, directory):
        directory = Path(directory)
        name = self.prefix + directory.name
        if self.architecture == "qwen3":
            return await self._stage_fused(directory, name)
        # Pin the base file, while preserving its template, system, and parameters.
        modelfile = directory / "Modelfile"
        modelfile.write_text(
            self.modelfile + f'\nADAPTER "{(directory / "adapter.gguf").resolve()}"\n'
        )
        await run_command(
            self.executable, "create", name, "-f", modelfile, env=self.env
        )
        return name

    async def _stage_fused(self, directory, name):
        # Ollama's Qwen3 runner may support inference while rejecting LoRA.
        # Fuse into a derived serving file; train only the cumulative adapter.

        merged = directory / "merged.gguf"
        if merged.is_symlink() or merged.resolve() == self.blob.resolve():
            raise ValueError(
                "A derived Ollama model must not replace or alias its base."
            )
        track_fused_file(self.directory, name, merged)
        try:
            if not merged.exists():
                try:
                    await run_command(
                        sys.executable,
                        "-m",
                        "adaptible.wrap.gguf_fusion",
                        str(self.blob),
                        str(directory / "adapter.gguf"),
                        str(merged),
                    )
                finally:
                    merged.with_suffix(".gguf.partial").unlink(missing_ok=True)
            modelfile = directory / "Modelfile"
            modelfile.write_text(
                re.sub(
                    r"^FROM[^\n]*",
                    lambda _: f'FROM "{merged.resolve()}"',
                    self.modelfile,
                    count=1,
                    flags=re.M,
                )
            )
            await run_command(
                self.executable, "create", name, "-f", modelfile, env=self.env
            )
            return name
        except BaseException:
            # Creation may have succeeded just before a response or CLI failure.
            # No handle reached the controller, so clean up this name here.
            try:
                await self._delete_private_tag(name)
            except Exception:
                logger.exception("Could not remove failed private Ollama tag %s", name)
            finally:
                remove_fused_file(self.directory, name)
            raise

    async def _delete_private_tag(self, handle):
        if not handle.startswith(self.prefix) or handle == self.active:
            return
        await self._unload((handle,))
        response = await self.client.request(
            "DELETE", self.url + "/api/delete", json={"model": handle}
        )
        if response.status_code != 404:
            response.raise_for_status()

    async def discard(self, handle):
        if handle and handle.startswith(self.prefix) and handle != self.active:
            if self.architecture == "qwen3":

                await self._delete_private_tag(handle)
                remove_fused_file(self.directory, handle)
                return
            await self._unload((handle,))
            await run_command(self.executable, "rm", handle, env=self.env)

    async def release_for_training(self):
        await self._unload((self.base_handle, self.active))

    async def _unload(self, models):
        # Empty generation with keep_alive=0 unloads weights, preserving the tag.
        # Callers supply only the selected base or this wrapper's own adapters.
        names = {
            name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
            for name in models
            if name
        }

        async def loaded():
            response = await self.client.get(self.url + "/api/ps")
            response.raise_for_status()
            return {model["name"] for model in response.json()["models"]} & names

        for name in await loaded():
            response = await self.client.post(
                self.url + "/api/generate",
                json=dict(model=name, prompt="", stream=False, keep_alive=0),
            )
            response.raise_for_status()
        # Ollama may acknowledge the request before its runner exits.
        for _ in range(60):
            if not await loaded():
                return
            await asyncio.sleep(0.25)
        raise RuntimeError("Ollama did not release this wrapper's serving weights.")


class LlamaCpp(Runtime):
    """Owns only its child server; never restarts an unrelated running service."""

    def __init__(self, model, directory, executable=None, context_size=4096, **kwargs):
        super().__init__(Path(model).stem, directory, **kwargs)
        self.blob = Path(model).expanduser().resolve(strict=True)
        self.executable = executable or shutil.which("llama-server")
        if not self.executable:
            raise ValueError(
                "Put llama-server on PATH, or pass --llama-server /path/to/llama-server."
            )
        self.context_size = context_size
        self.process = None
        self.log = None
        self.loaded = None
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    async def discover(self):
        return self.blob

    async def launch(self, handle=None):
        await self.stop()
        args = [
            self.executable,
            "-m",
            str(self.blob),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--alias",
            self.name,
            "-c",
            str(self.context_size),
            "--parallel",
            "1",
            *LLAMA_CPP_FULL_PRECISION,
        ]
        if handle:
            args += ["--lora", handle, "--lora-init-without-apply"]
        self.log = (self.directory / "llama-server.log").open("ab")
        self.process = await asyncio.create_subprocess_exec(
            *args, stdout=self.log, stderr=self.log
        )
        self.loaded = handle
        last_error = None
        attempts = round(config.READY_TIMEOUT_SECONDS / config.READY_POLL_SECONDS)
        for _ in range(attempts):
            if self.process.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited; see {self.directory / 'llama-server.log'}"
                )
            try:
                r = await self.client.get(self.url + "/health", timeout=1)
                if r.status_code == 200:
                    return
                last_error = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.debug("llama-server not ready yet: %s", last_error)
            await asyncio.sleep(config.READY_POLL_SECONDS)
        raise RuntimeError(
            "llama-server did not become ready within "
            f"{config.READY_TIMEOUT_SECONDS:g} seconds"
            + (f"; last error: {last_error}" if last_error else "")
            + f". See {self.directory / 'llama-server.log'}."
        )

    def payload(self, body, *, handle=None, frozen=False):
        desired = None if frozen else (handle or self.active)
        if desired and desired != self.loaded:
            raise RuntimeError(
                "Requested adapter is not loaded in the managed runtime."
            )
        return {
            # llama-server reuses the KV cache of whatever prompt its slot held
            # last, so a reused prefix is evaluated in a different batch shape
            # from a cold one and the logits differ in the last bits. Greedy
            # decoding absorbs that; sampling does not, and the same seed and
            # prompt returned two different corrections depending only on what
            # had been asked before. The original re-runs the whole prompt on
            # every call and has no such state, so the wrapper turns the cache
            # off. A caller that sets `cache_prompt` itself still wins.
            "cache_prompt": False,
            **NEUTRAL_SAMPLING,
            **body,
            "model": self.name,
            # An empty list restores llama.cpp's configured adapter scales; it
            # does not disable a loaded adapter. Explicit zero also makes the
            # server invalidate cached tokens from the previously adapted slot.
            "lora": [{"id": 0, "scale": int(bool(desired))}] if self.loaded else [],
        }

    async def stage(self, directory):
        handle = str((Path(directory) / "adapter.gguf").resolve())
        await self.launch(handle)
        return handle

    async def restore(self, handle):
        if self.process is None or self.loaded != handle:
            await self.launch(handle)
        self.active = handle

    async def discard(self, handle):
        pass

    async def release_for_training(self):
        await self.stop()

    async def stop(self):
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 15)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    async def close(self):
        await self.stop()
        await super().close()
