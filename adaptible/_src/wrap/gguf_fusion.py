"""Fuse adapted matrices into a derived GGUF without altering base weights."""

import shutil
import sys
from pathlib import Path

from .gguf_adapter import SUPPORTED_ARCHITECTURES


def fused_dtype(tensor_type):
    """The precision to write a fused tensor at: the source's own.

    An f32 source stays f32 and an f16 source stays f16, so fusion never makes
    one runtime serve a different precision from the others. A *quantized*
    source has no unquantized precision to match; f16 is kept there because
    fusion must produce an unquantized tensor anyway and a second quantization
    pass could erase a small learned update. bf16 is widened to f32, which is
    exact, because numpy has no native bfloat16.
    """
    import gguf
    import numpy as np

    if tensor_type == gguf.GGMLQuantizationType.F32:
        return np.dtype("float32")
    if tensor_type == gguf.GGMLQuantizationType.BF16:
        return np.dtype("float32")
    return np.dtype("float16")


def fuse_gguf(base, adapter, destination):
    """Merge one tensor at a time; preserve unmodified quantized bytes and metadata."""
    import gguf
    import numpy as np

    base, adapter, destination = map(Path, (base, adapter, destination))
    if destination.resolve() in (base.resolve(), adapter.resolve()):
        raise ValueError("A fused model must not overwrite its base or adapter.")
    source, delta = gguf.GGUFReader(str(base)), gguf.GGUFReader(str(adapter))

    architecture = source.fields["general.architecture"].contents()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "GGUF fusion supports dense Qwen2/Qwen2.5, Qwen3 and Llama-family GGUF."
        )
    if delta.fields["general.architecture"].contents() != architecture:
        raise ValueError("Adapter architecture does not match the base.")
    alpha = float(delta.fields["adapter.lora.alpha"].contents())
    pairs = {}
    for tensor in delta.tensors:
        name, suffix = tensor.name.rsplit(".lora_", 1)
        if suffix not in ("a", "b"):
            raise ValueError(f"Unsupported adapter tensor: {tensor.name}")
        pairs.setdefault(name, {})[suffix] = tensor
    if not pairs or any(set(pair) != {"a", "b"} for pair in pairs.values()):
        raise ValueError("Adapter must contain complete LoRA pairs.")
    base_names = {tensor.name for tensor in source.tensors}
    if not set(pairs) <= base_names:
        raise ValueError("Adapter contains tensors absent from the base.")
    for tensor in source.tensors:
        if tensor.name in pairs:
            a, b = (pairs[tensor.name][key].data for key in ("a", "b"))
            shape = tuple(reversed(tensor.shape.tolist()))
            if (
                a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != b.shape[1]
                or (b.shape[0], a.shape[1]) != shape
            ):
                raise ValueError(f"Adapter dimensions do not match {tensor.name}.")
    # Changed tensors are written at the source's own precision. Hardcoding
    # f16 here made LM Studio the only runtime serving the adapter at a
    # different precision from every other one -- measured on an f32
    # checkpoint, where llama.cpp, Ollama and vLLM all served f32 and this
    # path served f16 -- which is precisely what a five-way comparison on
    # identical weights exists to rule out. Unchanged tensors keep their
    # original data type, as before.
    required = sum(
        (
            t.n_elements * fused_dtype(t.tensor_type).itemsize
            if t.name in pairs
            else t.n_bytes
        )
        for t in source.tensors
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + 64 * 1024**2:
        raise ValueError(
            f"GGUF fusion needs {required / 1024**3:.2f} GiB free for a local derived GGUF."
        )
    temporary = destination.with_suffix(".gguf.partial")
    # Do not let GGUFWriter's wb open follow a stale symlink or hardlink.
    # Exclusive creation also makes an interrupted prior worker explicit.
    with temporary.open("xb"):
        pass
    writer = gguf.GGUFWriter(str(temporary), architecture, endianess=source.endianess)
    for name, field in source.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        writer.add_key_value(
            name,
            field.contents(),
            field.types[0],
            field.types[1] if len(field.types) > 1 else None,
        )
    for tensor in source.tensors:
        if tensor.name in pairs:
            dtype = fused_dtype(tensor.tensor_type)
            writer.add_tensor_info(
                tensor.name,
                tuple(reversed(tensor.shape.tolist())),
                dtype,
                tensor.n_elements * dtype.itemsize,
            )
        else:
            writer.add_tensor_info(
                tensor.name,
                tensor.data.shape,
                tensor.data.dtype,
                tensor.n_bytes,
                raw_dtype=tensor.tensor_type,
            )
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in source.tensors:
            if tensor.name in pairs:
                a, b = (
                    pairs[tensor.name][key].data.astype(np.float32)
                    for key in ("a", "b")
                )
                values = gguf.dequantize(tensor.data, tensor.tensor_type).astype(
                    np.float32
                )
                values += (b @ a) * (alpha / a.shape[0])
                dtype = fused_dtype(tensor.tensor_type)
                if not np.isfinite(values).all() or (
                    dtype == np.float16
                    and np.max(np.abs(values)) > np.finfo(np.float16).max
                ):
                    raise ValueError(f"Non-finite fused weights in {tensor.name}.")
                writer.write_tensor_data(values.astype(dtype))
                del values, a, b
            else:
                writer.write_tensor_data(tensor.data)
        writer.close()
        temporary.replace(destination)
    finally:
        writer.close()
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":

    fuse_gguf(*sys.argv[1:])
