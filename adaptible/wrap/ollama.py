"""Ownership bookkeeping for locally derived Ollama serving models."""

import hashlib
import json
from pathlib import Path


def _manifest(directory):
    directory = Path(directory)
    path = directory / "ollama-derived.json"
    return path, json.loads(path.read_text()) if path.exists() else {}


def _save(path, values):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(values))
    temporary.replace(path)


def track_fused_file(directory, handle, path):
    directory, path = Path(directory), Path(path)
    if path.name != "merged.gguf" or not path.resolve().is_relative_to(
        (directory / "adapters").resolve()
    ):
        raise ValueError(
            "Derived Ollama models must stay in this wrapper's adapter directory."
        )
    manifest, values = _manifest(directory)
    values[handle] = str(path.absolute())
    _save(manifest, values)


def remove_fused_file(directory, handle):
    """Remove only a recorded private artifact, never follow a replaced symlink."""
    directory = Path(directory)
    prefix = (
        "adaptible-"
        + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
        + "-"
    )
    handle = handle.removesuffix(":latest")
    if not handle.startswith(prefix):
        return False
    manifest, values = _manifest(directory)
    recorded = values.get(handle)
    if recorded is None:
        return False
    path = Path(recorded)
    if (
        path.name != "merged.gguf"
        or path.is_symlink()
        or not path.resolve().is_relative_to((directory / "adapters").resolve())
    ):
        return False
    path.unlink(missing_ok=True)
    path.with_suffix(".gguf.partial").unlink(missing_ok=True)
    values.pop(handle)
    _save(manifest, values)
    return True
