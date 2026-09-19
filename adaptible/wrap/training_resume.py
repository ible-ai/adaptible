"""Validated continuation of one candidate's bounded optimizer trajectory."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_identity(job, source_sha256):
    """Bind the data, source and accepted parent, excluding output/budget fields."""
    parent = job.get("previous")
    identity = {
        k: v
        for k, v in job.items()
        if k not in {"out", "resume_from", "max_total_steps"}
    }
    identity["blob"] = str(Path(job["blob"]).expanduser().resolve(strict=True))
    identity["source_sha256"] = source_sha256
    if parent:
        identity["previous"] = str(Path(parent).resolve(strict=True))
        identity["parent_sha256"] = digest(Path(parent) / "adapter_model.safetensors")
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def budget(job):
    thinking = bool((job.get("training_options") or {}).get("thinking", False))
    if job.get("resume_from") and not thinking:
        raise ValueError("Only thinking candidates support optimizer continuation.")
    maximum = job.get("max_total_steps", 64)
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= 64
    ):
        raise ValueError("max_total_steps must be an integer from 1 to 64.")
    return maximum


def load_manifest(job, source_sha256):
    """Check identity and complete artifacts before loading any candidate tensor."""
    maximum = budget(job)
    if not job.get("resume_from"):
        return None
    directory = Path(job["resume_from"]).expanduser().resolve(strict=True)
    if directory == Path(job["out"]).expanduser().resolve():
        raise ValueError("Continuation must write to a new candidate directory.")
    manifest = json.loads((directory / "resume.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Invalid continuation manifest.")
    total = manifest.get("total_steps")
    if (
        manifest.get("schema_version") != 1
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not 1 <= total < maximum <= 64
    ):
        raise ValueError("Invalid or exhausted continuation step budget.")
    if manifest.get("identity") != training_identity(job, source_sha256):
        raise ValueError(
            "Continuation source, parent, training data or options changed."
        )
    for name, field in (
        ("adapter/adapter_model.safetensors", "adapter_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
    ):
        if digest(directory / name) != manifest.get(field):
            raise ValueError("Continuation artifact digest mismatch.")
    return manifest


def restore_optimizer(optimizer, directory, manifest):
    """Load only tensors/basic containers and validate every moment before use."""
    import torch

    state = torch.load(
        Path(directory) / "optimizer.pt", map_location="cpu", weights_only=True
    )
    if (
        manifest.get("optimizer") != type(optimizer).__name__
        or type(optimizer).__name__ != "MLXAdamW"
    ):
        raise ValueError("Continuation optimizer type changed.")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Invalid optimizer state format.")
    if not isinstance(state["state"], dict):
        raise ValueError("Invalid optimizer parameter state.")
    groups = state["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer parameter groups changed.")
    seen = set()
    for saved, current in zip(groups, optimizer.param_groups, strict=False):
        if not isinstance(saved, dict) or set(saved) != set(current):
            raise ValueError("Optimizer group fields changed.")
        if any(saved[k] != current[k] for k in current if k != "params"):
            raise ValueError("Optimizer hyperparameters changed.")
        if not isinstance(saved["params"], list) or len(saved["params"]) != len(
            current["params"]
        ):
            raise ValueError("Optimizer parameter count changed.")
        for key, parameter in zip(saved["params"], current["params"], strict=False):
            if not isinstance(key, int) or isinstance(key, bool) or key in seen:
                raise ValueError("Invalid optimizer parameter identity.")
            seen.add(key)
            moments = state["state"].get(key)
            if not isinstance(moments, dict) or set(moments) != {"m", "v"}:
                raise ValueError("Missing or invalid optimizer moments.")
            for name, value in moments.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or not torch.isfinite(value).all()
                    or (name == "v" and (value < 0).any())
                ):
                    raise ValueError("Invalid optimizer moment tensor.")
    if set(state["state"]) != seen:
        raise ValueError("Unexpected optimizer parameter state.")
    # PyTorch relocates state to each parameter's device, retaining its dtype.
    optimizer.load_state_dict(state)


def save_resume(job, source_sha256, optimizer, total_steps):
    """Publish the manifest last, after complete adapter and optimizer writes."""
    import torch

    directory = Path(job["out"])
    temporary = directory / "optimizer.pt.tmp"
    torch.save(optimizer.state_dict(), temporary)
    temporary.replace(directory / "optimizer.pt")
    manifest = dict(
        schema_version=1,
        identity=training_identity(job, source_sha256),
        total_steps=total_steps,
        optimizer=type(optimizer).__name__,
        adapter_sha256=digest(directory / "adapter/adapter_model.safetensors"),
        optimizer_sha256=digest(directory / "optimizer.pt"),
    )
    temporary = directory / "resume.json.tmp"
    temporary.write_text(json.dumps(manifest))
    temporary.replace(directory / "resume.json")
