"""Adapter-only GGUF export for dense text models.

Qwen2/Qwen2.5 and Qwen3 store attention projections identically in Transformers
and llama.cpp, so their LoRA matrices need only tensor-name mapping. Llama-family
GGUFs (which also carry Mistral) interleave the rotary pairs of attn_q/attn_k, so
an adapter trained in Transformers space must be permuted the same way before it
is added to those tensors. ``wrap_gguf_roundtrip_test`` checks every architecture
here numerically against its PEFT reference.
"""

import json
import re
from pathlib import Path

SUPPORTED_ARCHITECTURES = frozenset({"qwen2", "qwen3", "llama", "mistral"})

# GGUF architectures whose attn_q/attn_k rows are rotary-interleaved.
PERMUTED_ARCHITECTURES = frozenset({"llama"})


def read_architecture(path: Path):
    """Report a GGUF's architecture without judging whether it can be trained."""
    from gguf import GGUFReader

    return GGUFReader(str(path)).fields["general.architecture"].contents()


def inspect_model(path: Path):
    """As :func:`read_architecture`, but refuses one this exporter cannot adapt.

    Raises:
        ValueError: The architecture has no verified adapter export.
    """
    architecture = read_architecture(path)
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "Repair supports dense Qwen2/Qwen2.5, Qwen3 and Llama-family text "
            f"GGUF models; this file is {architecture!r}."
        )
    return architecture


def permute_lora_b(weights, n_head, n_head_kv=None):
    """Interleave rotary pairs in a LoRA ``B`` matrix, as llama.cpp stores them.

    ``B`` has shape ``(out_features, rank)``; the permutation reorders rows, so
    it applies unchanged. ``A`` spans input features and is left alone. Because
    the permutation only reorders rows it distributes over addition, which is
    what lets a permuted delta be added to an already-permuted base tensor.
    """
    import numpy as np

    weights = np.asarray(weights)
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    if weights.shape[0] % (n_head * 2):
        raise ValueError("LoRA B rows do not divide into rotary pairs per head.")
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


def export_adapter(
    directory: Path,
    destination: Path,
    *,
    architecture="qwen2",
    num_attention_heads=None,
    num_key_value_heads=None,
):
    """Write a LoRA adapter as a GGUF the runtimes can apply to their base.

    Args:
        directory: A PEFT adapter directory.
        destination: Path of the ``.gguf`` adapter to write.
        architecture: The base model's GGUF architecture.
        num_attention_heads: Required for architectures in
            ``PERMUTED_ARCHITECTURES``, which need the rotary interleave.
        num_key_value_heads: As above, for grouped-query attention.

    Raises:
        ValueError: The architecture is unsupported, the adapter is not ordinary
            uniform-rank LoRA, or head counts are missing where they are needed.
    """

    import gguf
    from safetensors.numpy import load_file

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported adapter architecture: {architecture!r}")
    config = json.loads((directory / "adapter_config.json").read_text())
    if (
        config.get("use_dora")
        or config.get("use_rslora")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
    ):
        raise ValueError("This exporter requires ordinary uniform-rank LoRA.")
    mapping = dict(
        q_proj="attn_q",
        k_proj="attn_k",
        v_proj="attn_v",
        o_proj="attn_output",
        gate_proj="ffn_gate",
        up_proj="ffn_up",
        down_proj="ffn_down",
    )
    permuted = architecture in PERMUTED_ARCHITECTURES
    if permuted and not num_attention_heads:
        raise ValueError(
            f"{architecture!r} adapters need the base model's head counts to "
            "match llama.cpp's rotary interleave."
        )
    # Dense Qwen3 adds q/k RMS norms, which stay frozen. Its seven adapted
    # linear projections have Qwen2's layout; no RoPE row permutation applies.
    writer = gguf.GGUFWriter(str(destination), architecture)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(config["lora_alpha"]))
    pairs = {}
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    for name, value in tensors.items():
        m = re.fullmatch(
            r"base_model.model.model.layers.(\d+).(?:self_attn|mlp).(\w+).lora_([AB]).weight",
            name,
        )
        if not m or m[2] not in mapping:
            raise ValueError(f"Unsupported adapter tensor: {name}")
        target = f"blk.{m[1]}.{mapping[m[2]]}.weight"
        pairs.setdefault(target, set()).add(m[3])
        value = value.astype("float32")
        if permuted and m[3] == "B" and m[2] in ("q_proj", "k_proj"):
            value = permute_lora_b(
                value,
                num_attention_heads,
                num_key_value_heads if m[2] == "k_proj" else None,
            )
        writer.add_tensor(target + ".lora_" + m[3].lower(), value)
    if not pairs or any(p != {"A", "B"} for p in pairs.values()):
        raise ValueError("Adapter is empty or has incomplete LoRA pairs.")
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
