"""Offline training worker. Started and reaped by the wrapper; never serves."""

import json
import os
import sys
from pathlib import Path

# Sibling modules defer torch/transformers to call time, so importing them here
# still leaves enforce_offline() the first thing that runs before any of those
# libraries load.
from .gguf_adapter import export_adapter
from .model_source import declared_precision, fingerprint_base, inspect_base
from .tokenizer import load_tokenizer
from .training_budget import fit_masked_target
from .training_examples import encode_examples
from .training_resume import budget, load_manifest, restore_optimizer, save_resume

# Enforce the no-second-checkpoint contract, including child library lookups.
# Applied when training starts rather than at import, so that importing this
# module does not reconfigure an unrelated process.
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def checkpoint_dtype(source, source_options):
    """The precision the checkpoint declares; float32 when it declares none.

    A GGUF source carries its config inside the file and transformers
    dequantises it on load, so there is no `config.json` to read and float32
    is both the dequantised precision and the safe answer.
    """
    import torch

    if source_options.get("gguf_file"):
        return torch.float32
    # The weights, not `config.json`. The checkpoint this port is validated
    # against declares `bfloat16` in its config and stores `F16` in all 339 of
    # its tensors; trusting the config trains in a precision the model was
    # never stored in, which is the defect this function exists to prevent.
    declared = declared_precision(source)
    if declared is None:
        try:
            declared = json.loads((Path(source) / "config.json").read_text()).get(
                "torch_dtype"
            )
        except (OSError, ValueError):
            declared = None
    resolved = getattr(torch, declared, None) if isinstance(declared, str) else None
    return resolved if isinstance(resolved, torch.dtype) else torch.float32


def enforce_offline():
    """Pins Hugging Face libraries offline before any of them are imported."""
    os.environ.update(_OFFLINE_ENVIRONMENT)


def train(job):
    enforce_offline()
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    # Subclasses torch.optim.Optimizer, so it cannot defer torch itself.
    from .training_optimizer import training_optimizer

    blob, out = Path(job["blob"]).expanduser().resolve(strict=True), Path(job["out"])

    maximum = budget(job)
    source_sha256 = fingerprint_base(blob)
    resume = load_manifest(job, source_sha256)
    thinking = bool((job.get("training_options") or {}).get("thinking", False))
    architecture = inspect_base(blob)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                path=str(blob),
                kind="hf" if blob.is_dir() else "gguf",
                sha256=source_sha256,
            )
        )
    )
    if blob.is_dir():
        source, source_options = blob, {}
    else:
        # Explicit gguf_file selects its embedded config over parent config.json.
        source, source_options = blob.parent, {"gguf_file": blob.name}
    tokenizer = load_tokenizer(blob)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # Train in the checkpoint's own precision, never the accelerator's
    # preference. Choosing bf16 because an accelerator is present trains a
    # different adapter from the same example: bf16 keeps 7 mantissa bits
    # against f32's 23. Measured on one cycle of the flagship recipe, the
    # step-0 answer-token loss -- which is the *base* model's, since LoRA's B
    # is zero-initialised and contributes nothing to the first forward -- came
    # back 0.07825317 here against the original's 0.07735140 on identical
    # tokens and an identical mask, and the resulting adapter produced
    # different post-training answers on 3 of 4 prompts. This is the trainer's
    # half of the same defect as vLLM's unpinned `--dtype`.
    dtype = checkpoint_dtype(source, source_options)
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, dtype=dtype, **source_options
    )
    if model.config.model_type != architecture:
        raise ValueError("Loaded model architecture does not match its source.")
    model.to(device)
    torch.manual_seed(0)
    config = LoraConfig(
        r=8,
        lora_alpha=80,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        layers_to_transform=list(
            range(
                max(0, model.config.num_hidden_layers - 8),
                model.config.num_hidden_layers,
            )
        ),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    weights_from = (
        (Path(job["resume_from"]) / "adapter") if resume else job.get("previous")
    )
    if weights_from:
        result = set_peft_model_state_dict(
            model, load_file(str(Path(weights_from) / "adapter_model.safetensors"))
        )
        if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
            raise ValueError("Saved adapter does not match this model.")

    inputs = encode_examples(tokenizer, job, architecture, device=device)

    options = job.get("training_options") or {}
    optimizer = training_optimizer(
        (p for p in model.parameters() if p.requires_grad),
        thinking=bool(options.get("thinking", False)),
        learning_rate=options.get("learning_rate"),
    )
    if resume:
        restore_optimizer(optimizer, job["resume_from"], resume)
    prior_steps = resume["total_steps"] if resume else 0

    stats = fit_masked_target(
        model,
        inputs["input_ids"],
        inputs["labels"],
        optimizer,
        attention_mask=inputs["attention_mask"],
        # The experiment's stop rule is on the answer tokens, not the whole
        # target: "loss over the whole target, stop rule on the answer tokens
        # ... stop when answer-token loss < 0.15" (results README). Its own
        # candidates.csv records candidates that took a single step at loss
        # 0.10, so an early stop is the experiment's behaviour, not a defect.
        stop_labels=inputs.get("stop_labels"),
        # The experiment's loop updates first and then tests the pre-update
        # loss; the wrapper's default tests before updating. Only the flagship
        # recipe asks for the experiment's, and it asks explicitly.
        stop_after_update=job.get("stop_rule") == "experiment",
        **(
            dict(max_steps=maximum - prior_steps, stop_at_target=not bool(resume))
            if thinking
            else {}
        ),
    )
    stats["examples"] = len(inputs["input_ids"])
    stats["optimizer"] = type(optimizer).__name__
    stats["learning_rate"] = optimizer.param_groups[0]["lr"]
    stats["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
    model.save_pretrained(out / "adapter")
    if not blob.is_dir():
        export_adapter(
            out / "adapter",
            out / "adapter.gguf",
            architecture=architecture,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
        )
    if thinking:
        stats["total_steps"] = prior_steps + stats["steps"]
        stats["resume_from"] = job.get("resume_from")
        save_resume(job, source_sha256, optimizer, stats["total_steps"])
    (out / "stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    train(json.loads(Path(sys.argv[1]).read_text()))
