"""Inspect and identify an existing local base; never resolve a Hub download."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from .gguf_adapter import (
    SUPPORTED_ARCHITECTURES,
    inspect_model,
)
from .gguf_adapter import (
    read_architecture as read_gguf_architecture,
)

# safetensors dtype tags, in the header each file carries.
_SAFETENSORS_DTYPES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
}


def declared_precision(source):
    """The precision a checkpoint is actually stored in, or None.

    Read from the safetensors header rather than `config.json`, because the
    two disagree and the weights are the ones that get loaded. The checkpoint
    this port is validated against says `"torch_dtype": "bfloat16"` in its
    config while all 339 of its tensors are `F16` -- so anything trusting the
    config runs the model at a precision it was never stored in, and the
    loaders (mlx_lm, and transformers when no dtype is passed) go by the
    tensors. f16 and bf16 are not interchangeable: 10 mantissa bits against 7,
    5 exponent bits against 8, and neither contains the other.
    """
    source = Path(source)
    files = sorted(source.glob("*.safetensors"))
    found = set()
    for path in files:
        # AppleDouble sidecars on exFAT match the glob and are not checkpoints.
        if path.name.startswith("._"):
            continue
        try:
            with path.open("rb") as handle:
                length = int.from_bytes(handle.read(8), "little")
                header = json.loads(handle.read(length))
        except (OSError, ValueError):
            return None
        found.update(
            _SAFETENSORS_DTYPES.get(entry.get("dtype"))
            for name, entry in header.items()
            if name != "__metadata__" and isinstance(entry, dict)
        )
    found.discard(None)
    # A mixed checkpoint has no single precision to honour; say so rather than
    # picking one of them.
    return found.pop() if len(found) == 1 else None


def _check_hf_weights(source):
    if (source / "model.safetensors").is_file():
        return [source / "model.safetensors"]
    index = source / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError(
            "The local model directory needs model.safetensors or its shard index."
        )
    weight_map = json.loads(index.read_text()).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            "The safetensors shard index must contain a nonempty weight_map."
        )
    for name in weight_map.values():
        if (
            not isinstance(name, str)
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).suffix != ".safetensors"
        ):
            raise ValueError(
                "Safetensors shard paths must stay within the model directory."
            )
        if not (source / name).is_file():
            raise ValueError(f"Missing local safetensors shard: {name}")
    return [source / name for name in set(weight_map.values())]


def read_architecture(source):
    """Identify a local base model without deciding whether repair supports it.

    Serving is a proxy and works for any model the runtime can already run, so
    startup uses this. Only the training path applies :func:`inspect_base`.
    """
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_file():
        return read_gguf_architecture(source)
    if not source.is_dir():
        raise ValueError("Provide an existing GGUF file or local model directory.")
    return json.loads((source / "config.json").read_text()).get("model_type")


def repairable(source):
    """Whether repair can train this base, and why not when it cannot."""
    try:
        inspect_base(source)
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def inspect_base(source):
    """As :func:`read_architecture`, but refuses a base repair cannot train.

    Raises:
        ValueError: The architecture, quantisation or tokenizer files rule out
            local training.
    """
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_file():
        return inspect_model(source)
    if not source.is_dir():
        raise ValueError("Provide an existing GGUF file or local model directory.")
    config = json.loads((source / "config.json").read_text())

    architecture = config.get("model_type")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "Repair supports dense Qwen2/Qwen2.5, Qwen3 and Llama-family text models."
        )
    if config.get("quantization_config"):
        raise ValueError(
            "The local Hugging Face training path requires unquantized weights. "
            "Quantized GGUF is supported through the GGUF runtimes; no alternate "
            "checkpoint will be downloaded."
        )
    _check_hf_weights(source)
    if not (source / "tokenizer.json").is_file() and not (
        (source / "vocab.json").is_file() and (source / "merges.txt").is_file()
    ):
        raise ValueError("The local model directory must include its tokenizer files.")
    return architecture


def fingerprint_base(source):
    """Pin GGUF bytes or all local HF weight/config/tokenizer file contents."""
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_file():
        with source.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    suffixes = {".json", ".safetensors", ".model", ".txt", ".tiktoken", ".jinja"}
    files = {p for p in source.rglob("*") if p.is_file() and p.suffix in suffixes}
    # An indexed shard or template directory may itself be a symlink into
    # external storage. rglob does not recurse through such directories.
    files.update(_check_hf_weights(source))
    files.update(p for p in source.glob("chat_templates/*.jinja") if p.is_file())
    if not files:
        raise ValueError("No local model files found to fingerprint.")
    digest = hashlib.sha256(b"adaptible-local-hf-v1\0")
    for path in sorted(files):
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()
