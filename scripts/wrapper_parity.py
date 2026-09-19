# ruff: noqa: T201  -- a comparison script reports to the terminal by design.
"""Check that a wrapper reproduces scripts/cycles_mlx.py token for token.

One correction, run twice: once by the original on MLX and once through a
wrapper runtime. Each run generates greedily for the experiment's four prompts
for geo_001, trains on one fixed correction, and generates again. The wrapper
trains with its own PyTorch trainer and serves the result through its own
runtime, starting from the LoRA initialisation MLX drew (the original's draw is
unseeded, so without this two MLX runs would not agree with each other).
`compare` then checks every generation, thought and answer, for equality.

What a match covers: the wrapper's prompt building and pinned runtime settings
(greedy decoding), its training example, optimizer, stop rule and step count,
and the adapter it exports and serves. It does not cover sampling a candidate
or the keep decision; the correction is the fixed text below.

Everything runs at float32. The checkpoint is stored as float16 (its
config.json says bfloat16), and widening f16 to f32 is exact. At f16, MLX and
llama.cpp kernels choose different tokens within a few hundred characters.

Apple Silicon, about 10 GB of free memory, one step at a time:

    python scripts/wrapper_parity.py prepare --out DIR \\
        --convert ~/llama.cpp/convert_hf_to_gguf.py
    python scripts/wrapper_parity.py mlx --out DIR
    python scripts/wrapper_parity.py llama-cpp --out DIR --executable llama-server
    python scripts/wrapper_parity.py lm-studio --out DIR --executable ~/.lmstudio/bin/lms
    python scripts/wrapper_parity.py vllm --out DIR --executable vllm
    python scripts/wrapper_parity.py ollama --out DIR --model dsr1-f32 \\
        --url http://127.0.0.1:11434
    python scripts/wrapper_parity.py compare --out DIR

`prepare` writes DIR/Modelfile for Ollama; run `ollama create dsr1-f32 -f
DIR/Modelfile` before the ollama step. On Apple Silicon, vLLM means
vllm-metal, which computes with mlx_lm, so a vLLM match there checks the
wrapper rather than vLLM's own kernels.
"""

import argparse
import ast
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts" / "cycles_mlx.py"
SOURCE = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"  # StatefulLLM's default
ITEM = "geo_001"
INIT_SEED = 0
MAX_TOKENS = 2048  # adaptible.llm.MAX_TOKENS, what the experiment scores with
CONTEXT = 8192
RUNTIMES = ("llama-cpp", "lm-studio", "vllm", "ollama")

# The correction both sides train on, as the experiment's model returns it:
# its chat template opens <think>, so the text starts inside the thought.
CANDIDATE = (
    "Okay, the user is asking which city is the capital of Australia. The "
    "reference note says the correct answer is Canberra. Many people assume "
    "Sydney because it is the largest city, but the seat of the federal "
    "government is Canberra.\n"
    "</think>\n\n"
    "The capital of Australia is Canberra. Sydney is the largest city, but "
    "Canberra has been the seat of the federal government since 1927. "
    "Melbourne held that role until the new capital was ready."
)

# The Hugging Face chat template in Ollama's syntax, <think> prefilled as the
# original's template does. Ollama's stock deepseek-r1 template differs.
MODELFILE = '''FROM {gguf}
TEMPLATE """{{{{- if .System }}}}{{{{ .System }}}}{{{{ end }}}}
{{{{- range .Messages }}}}
{{{{- if eq .Role "user" }}}}<｜User｜>{{{{ .Content }}}}
{{{{- else if eq .Role "assistant" }}}}<｜Assistant｜>{{{{ .Content }}}}<｜end▁of▁sentence｜>
{{{{- end }}}}
{{{{- end }}}}<｜Assistant｜><think>
"""
PARAMETER stop <｜begin▁of▁sentence｜>
PARAMETER stop <｜end▁of▁sentence｜>
PARAMETER stop <｜User｜>
PARAMETER stop <｜Assistant｜>
'''


# --------------------------------------------------------------------------
# the original, read from its source


def original():
    """cycles_mlx.py's definitions and model arguments, without running it.

    The script builds a model at import, so what it defines is lifted with
    `ast`, as experiment_equivalence_test does. A copy here would only show
    that the copy agrees with itself. `LR` and `MAX_STEPS` read the
    environment in the script, so they get an empty one: the committed
    defaults are the reference.
    """
    from adaptible import InteractionHistory
    from adaptible.eval.harness import contains_key_terms
    from adaptible.revise import make_revision_training_example

    functions = ("closed", "answer_of", "example_from")
    constants = ("PARA", "LR", "MAX_STEPS")
    tree = ast.parse(ORIGINAL.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in constants for t in node.targets)
        )
    ]
    namespace = dict(
        os=types.SimpleNamespace(environ={}),
        re=__import__("re"),
        contains_key_terms=contains_key_terms,
        InteractionHistory=InteractionHistory,
        make_revision_training_example=make_revision_training_example,
    )
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(ORIGINAL), "exec"),
        namespace,
    )
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "StatefulLLM"
    )
    namespace["model_arguments"] = {
        k.arg: ast.literal_eval(k.value)
        for k in call.keywords
        if k.arg in ("num_lora_layers", "lora_parameters")
    }
    return namespace


def item():
    from adaptible.eval import generate_default_dataset

    return {i.id: i for i in generate_default_dataset()}[ITEM]


def prompts(namespace):
    return [item().question] + namespace["PARA"][ITEM]


def parts(text):
    """(thought, answer) as the experiment reads them.

    `answer_of` takes the text after the last </think> and `example_from` the
    thought before the first. A leading <think> is framing: the original's
    template prefills it, and a runtime may hand it back.
    """
    pieces = text.split("</think>")
    thought = pieces[0].strip()
    if thought.startswith("<think>"):
        thought = thought[len("<think>") :].strip()
    return thought, pieces[-1].strip()


# --------------------------------------------------------------------------
# prepare


def prepare(out, source, convert):
    """Widen the checkpoint to f32 as a HF directory and, optionally, a GGUF."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not Path(source).exists():
        source = snapshot_download(
            source, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"]
        )
    source = Path(source)
    hf = out / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    tensors, stored = {}, set()
    # One tensor at a time, so the source is never resident beside its copy.
    # `._*` are macOS metadata files on non-APFS disks, not checkpoints.
    for path in sorted(
        p for p in source.glob("*.safetensors") if not p.name.startswith("._")
    ):
        with safe_open(str(path), framework="pt") as reader:
            for name in reader.keys():  # noqa: SIM118 -- safe_open is not a dict
                value = reader.get_tensor(name)
                stored.add(str(value.dtype))
                wide = value.to(torch.float32) if value.is_floating_point() else value
                if not torch.equal(wide.to(value.dtype), value):
                    raise SystemExit(f"{name}: widening to f32 is not exact")
                tensors[name] = wide.contiguous()
    stored = sorted(stored)
    save_file(tensors, str(hf / "model.safetensors"), metadata={"format": "pt"})
    count = len(tensors)
    del tensors
    for path in source.iterdir():
        if (
            path.suffix in (".json", ".txt", ".model")
            and path.name != "model.safetensors.index.json"
        ):
            shutil.copy(path, hf / path.name)
    config = json.loads((hf / "config.json").read_text())
    config["torch_dtype"] = "float32"
    config.pop("quantization", None)
    (hf / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{hf}: {count} tensors widened from {stored} to float32")

    gguf = out / "f32.gguf"
    if convert:
        subprocess.run(
            [
                sys.executable,
                str(Path(convert).expanduser()),
                str(hf),
                "--outtype",
                "f32",
                "--outfile",
                str(gguf),
            ],
            check=True,
        )
        print(f"{gguf}: written")
    else:
        print(
            f"skipped the GGUF; pass --convert PATH/convert_hf_to_gguf.py to write {gguf}"
        )
    (out / "Modelfile").write_text(MODELFILE.format(gguf=gguf))
    print(f"{out / 'Modelfile'}: for `ollama create <name> -f {out / 'Modelfile'}`")


# --------------------------------------------------------------------------
# the original on MLX


def to_peft(mlx_tensors):
    """MLX LoRA tensors in PEFT's names and layout.

    MLX computes x @ A_m @ B_m * scale with A_m (in, r) and B_m (r, out); PEFT
    holds A_p (r, in) and B_p (out, r). The two agree when each is transposed
    and PEFT's alpha / r equals MLX's scale, which train.py sets (80 / 8).
    """
    import numpy as np

    return {
        f"base_model.model.model.{stem.removeprefix('model.')}.lora_{kind.upper()}.weight": np.ascontiguousarray(
            np.asarray(value).T
        )
        for name, value in mlx_tensors.items()
        for stem, _, kind in [name.rpartition(".lora_")]
    }


def run_mlx(out):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file

    import adaptible
    from adaptible.eval.harness import VERIFY_LOSS_FLOOR
    from adaptible.revise import collate_training_examples

    namespace = original()
    mx.random.seed(INIT_SEED)
    model = adaptible.StatefulLLM(
        model_name=str(out / "hf"),
        model_path=None,
        learning_rate=namespace["LR"],
        **namespace["model_arguments"],
    )
    dtypes = {str(v.dtype) for _, v in tree_flatten(model._model.parameters())}
    if dtypes != {"mlx.core.float32"}:
        raise SystemExit(
            f"expected an all-f32 model, got {sorted(dtypes)}; run prepare"
        )

    initial = dict(tree_flatten(model._model.trainable_parameters()))
    (out / "init").mkdir(exist_ok=True)
    save_file(to_peft(initial), str(out / "init" / "adapter_model.safetensors"))

    def generate():
        return [
            model.generate_response(q, use_history=False, max_tokens=MAX_TOKENS) or ""
            for q in prompts(namespace)
        ]

    base = generate()
    namespace["tok"] = model._tokenizer
    example, target = namespace["example_from"](item(), CANDIDATE)
    stats = model.train_on_examples(
        collate_training_examples([example], model._tokenizer),
        [],
        loss_target=VERIFY_LOSS_FLOOR,
        max_steps=namespace["MAX_STEPS"],
        rehearsal_weight=0.0,
        rehearsal_margin=0.05,
    )
    record = dict(
        base=base,
        adapted=generate(),
        target=target,
        steps=stats.steps,
        initial_loss=float(stats.initial_loss),
        final_loss=float(stats.final_loss),
    )
    (out / "mlx.json").write_text(json.dumps(record, indent=2))
    print(f"mlx: {stats.steps} steps, final loss {record['final_loss']:.6f}")


# --------------------------------------------------------------------------
# a wrapper


class FixedCandidate:
    """A runtime stand-in that returns CANDIDATE, so the wrapper's own
    flagship_candidate shapes the training target from it."""

    async def complete(self, messages, *, details=None, **options):
        from adaptible.wrap.thinking import completion_details

        reasoning, _, answer = CANDIDATE.partition("</think>")
        detail = completion_details(
            {"content": answer.lstrip(), "reasoning_content": reasoning},
            finish_reason="stop",
        )
        details.update(detail)
        return detail["content"]


def build_runtime(version, out, directory, args):
    from adaptible.wrap.lmstudio import LMStudio
    from adaptible.wrap.runtime import LlamaCpp, Ollama
    from adaptible.wrap.vllm import VLLM

    common = dict(max_tokens=MAX_TOKENS, context_size=CONTEXT)
    gguf = str(out / "f32.gguf")
    if version == "ollama":
        return Ollama(args.model, directory, url=args.url, **common)
    if version == "llama-cpp":
        return LlamaCpp(gguf, directory, executable=args.executable, **common)
    if version == "lm-studio":
        return LMStudio(gguf, directory, executable=args.executable, **common)
    return VLLM(str(out / "hf"), directory, executable=args.executable, **common)


async def generate(runtime, tokenizer, questions):
    """Greedy turns as the wrapper scores them under --flagship-recipe: cut
    where the original's loop breakers stop, thought and answer joined the way
    the original returns them."""
    from adaptible.wrap.loop_breaker import complete_as_original

    texts = []
    for q in questions:
        details = {}
        content = await complete_as_original(
            runtime,
            tokenizer,
            [dict(role="user", content=q)],
            lines=True,
            details=details,
            temperature=0,
            seed=0,
            max_tokens=MAX_TOKENS,
        )
        reasoning = details.get("reasoning") or ""
        closed = details.get("finish_reason") != "loop"
        texts.append(
            f"{reasoning}</think>{content}"
            if reasoning and closed
            else reasoning or content
        )
    return texts


async def run_wrapper(version, out, args):
    from adaptible.wrap.model_source import read_architecture
    from adaptible.wrap.repair import _FLAGSHIP_MAX_STEPS, Controller, Trainer
    from adaptible.wrap.tokenizer import load_tokenizer

    if not (out / "init" / "adapter_model.safetensors").exists():
        raise SystemExit("run the mlx step first: it writes the LoRA initialisation")
    namespace = original()
    question = item().question
    directory = Path(tempfile.mkdtemp(prefix=f"parity-{version}-", dir=out))
    runtime = build_runtime(version, out, directory, args)
    handle = None
    try:
        blob = await runtime.discover()
        runtime.architecture = read_architecture(blob)

        stand_in = types.SimpleNamespace(
            runtime=FixedCandidate(), flagship_candidate_temperature=0
        )
        candidate = await Controller.flagship_candidate(
            stand_in,
            question,
            f"the correct answer is {item().correct_answer}.",
            item().correct_answer,
        )
        if candidate is None:
            raise SystemExit("the wrapper rejected the fixed correction")
        # Trained before anything is served, so the model is never in memory
        # twice. The job is the one repair.py's flagship branch sends.
        stats = await Trainer().train(
            blob,
            [dict(role="user", content=question)],
            candidate["target"],
            directory / "candidate",
            str(out / "init"),
            examples=[],
            training_options={
                **runtime.training_options(),
                "thinking": True,
                "reasoning_prefix": candidate["reasoning_prefix"],
                "thinking_training": "flagship_rationale",
            },
            max_total_steps=_FLAGSHIP_MAX_STEPS,
            stop_rule="experiment",
        )

        await runtime.restore(None)
        await runtime.detect_reasoning()
        tokenizer = load_tokenizer(blob)
        base = await generate(runtime, tokenizer, prompts(namespace))
        handle = await runtime.stage(directory / "candidate")
        await runtime.restore(handle)
        adapted = await generate(runtime, tokenizer, prompts(namespace))
    finally:
        if handle:
            runtime.active = None
            try:
                await runtime.discard(handle)
            except (
                Exception
            ) as error:  # noqa: BLE001 -- reported, the run's result stands
                print(f"could not remove the staged adapter {handle}: {error}")
        await runtime.close()
        # The candidate and any fused GGUF (7 GB at f32) are only this run's.
        shutil.rmtree(directory, ignore_errors=True)
    record = dict(
        base=base,
        adapted=adapted,
        target=candidate["target"],
        steps=stats["steps"],
        initial_loss=stats["initial_loss"],
        final_loss=stats["final_loss"],
    )
    (out / f"{version}.json").write_text(json.dumps(record, indent=2))
    print(f"{version}: {stats['steps']} steps, final loss {stats['final_loss']:.6f}")


# --------------------------------------------------------------------------
# compare


def first_difference(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
        min(len(left), len(right)),
    )


def compare(out):
    """Every generation against MLX's; returns the number of mismatches."""
    reference = json.loads((out / "mlx.json").read_text())
    mismatches = 0
    found = False
    for version in RUNTIMES:
        path = out / f"{version}.json"
        if not path.exists():
            continue
        found = True
        record = json.loads(path.read_text())
        rows = []
        for phase in ("base", "adapted"):
            for i, (want, got) in enumerate(
                zip(reference[phase], record[phase], strict=True)
            ):
                for half, a, b in zip(
                    ("thought", "answer"), parts(want), parts(got), strict=True
                ):
                    if a != b:
                        at = first_difference(a, b)
                        rows.append(
                            f"  {phase}:{i} {half} differs at char {at}: {a[at:at + 40]!r} vs {b[at:at + 40]!r}"
                        )
        same = 2 * len(reference["base"]) - len({r.split()[0] for r in rows})
        gap = f"{abs(record['final_loss'] - reference['final_loss']):.1e}"
        print(
            f"{version:10} {same}/{2 * len(reference['base'])} generations identical | "
            f"steps {record['steps']} vs {reference['steps']} | "
            f"target {'same' if record['target'] == reference['target'] else 'DIFFERENT'} | "
            f"final loss gap {gap}"
        )
        for row in rows:
            print(row)
        mismatches += (
            len(rows)
            + (record["steps"] != reference["steps"])
            + (record["target"] != reference["target"])
        )
    if not found:
        raise SystemExit(f"no runtime results in {out}")
    return mismatches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=("prepare", "mlx", *RUNTIMES, "compare"))
    parser.add_argument("--out", type=Path, required=True, help="working directory")
    parser.add_argument(
        "--source", default=SOURCE, help="prepare: checkpoint directory or Hub id"
    )
    parser.add_argument("--convert", help="prepare: llama.cpp's convert_hf_to_gguf.py")
    parser.add_argument("--executable", help="llama-server, lms or vllm")
    parser.add_argument("--model", help="ollama: the model created from DIR/Modelfile")
    parser.add_argument(
        "--url", default="http://127.0.0.1:11434", help="ollama: server URL"
    )
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "prepare":
        prepare(out, args.source, args.convert)
    elif args.step == "mlx":
        run_mlx(out)
    elif args.step == "compare":
        return 1 if compare(out) else 0
    else:
        if args.step == "ollama" and not args.model:
            parser.error("ollama needs --model")
        if args.step != "ollama" and not args.executable:
            parser.error(f"{args.step} needs --executable")
        asyncio.run(run_wrapper(args.step, out, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
