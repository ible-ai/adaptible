"""Numeric proof that an exported GGUF adapter means the same thing as its PEFT source.

The wrapper trains a LoRA in Transformers space, exports it to GGUF, and the
GGUF runtimes fuse it into their base. Nothing downstream re-checks that those
three representations agree, and they do not agree by default: Llama-family
GGUFs interleave the rotary pairs of ``attn_q``/``attn_k``, so an adapter added
without the same interleave lands in the wrong rows.

This compares logits from ``base + PEFT adapter`` against ``fuse(base, exported)``
reloaded, for every architecture the wrapper accepts. Any new architecture must
pass here before it is added to ``SUPPORTED_ARCHITECTURES``.

The comparison is *relative to the adapter's own effect*. An absolute tolerance
is useless: a barely-trained adapter makes every error small, so a wrong
permutation slips through. ``test_the_comparison_detects_a_wrong_permutation``
holds that property in place.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from adaptible.wrap.gguf_adapter import (
    PERMUTED_ARCHITECTURES,
    SUPPORTED_ARCHITECTURES,
    export_adapter,
)
from adaptible.wrap.gguf_fusion import fuse_gguf
from adaptible.wrap.train import train
from adaptible.wrap import tiny_models as tiny

# A correct fusion differs from its PEFT reference only by float16 rounding,
# which measures ~4e-4 of the adapter's effect. A mis-permuted one measures
# ~1e-2. Anything between is a real defect, not noise.
MAX_RELATIVE_ERROR = 0.005

# How much worse a wrong permutation must look than a correct one. An absolute
# floor would be tuned to whatever adapter one fixture happens to train; the
# ratio between the two measurements is the property that must hold.
MIN_PERMUTATION_SIGNAL = 10

PROMPT = torch.tensor([[1, 5, 9, 14, 20, 27]])

# LoRA B starts at zero and moves little on a random model, so the honest
# signal is tiny. Scaling B makes the adapter's effect dominate f16 rounding
# without changing what is being compared: B is linear in the fused delta.
_AMPLIFY = 50


def _logits(model):
    # Newer Transformers place a loaded model on MPS/CUDA by default.
    with torch.no_grad():
        return model(PROMPT.to(model.device)).logits.cpu()


class AdapterRoundTripTest(unittest.TestCase):
    """base+PEFT vs fuse(base, exported) for each supported architecture."""

    def measure(self, architecture, root):
        """Returns the fusion error as a fraction of the adapter's own effect."""
        from peft import PeftModel
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM

        base = tiny.write_gguf_checkpoint(root / "base.gguf", architecture)
        out = root / "candidate"
        train(tiny.training_job(base, out))

        saved = out / "adapter" / "adapter_model.safetensors"
        tensors = load_file(str(saved))
        save_file(
            {
                name: (value * _AMPLIFY if "lora_B" in name else value).contiguous()
                for name, value in tensors.items()
            },
            str(saved),
        )

        config = tiny.tiny_config(architecture)
        gguf_architecture = tiny.ARCHITECTURES[architecture]["gguf"].lower()
        export_adapter(
            out / "adapter",
            out / "adapter.gguf",
            architecture=gguf_architecture,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )

        reference = AutoModelForCausalLM.from_pretrained(
            root, gguf_file="base.gguf", local_files_only=True, dtype=torch.float32
        ).eval()
        plain = _logits(reference)
        adapted = _logits(PeftModel.from_pretrained(reference, str(out / "adapter")))

        fuse_gguf(base, out / "adapter.gguf", root / "fused.gguf")
        fused = _logits(
            AutoModelForCausalLM.from_pretrained(
                root,
                gguf_file="fused.gguf",
                local_files_only=True,
                dtype=torch.float32,
            ).eval()
        )

        effect = (adapted - plain).abs().max().item()
        self.assertGreater(effect, 0.01, "the adapter must actually change the output")
        return (adapted - fused).abs().max().item() / effect

    def test_every_supported_architecture_fuses_to_its_peft_reference(self):
        exportable = sorted(
            name
            for name, spec in tiny.ARCHITECTURES.items()
            if spec["supported"] and spec["gguf"]
        )
        self.assertTrue(exportable)
        for architecture in exportable:
            with (
                self.subTest(architecture=architecture),
                tempfile.TemporaryDirectory() as directory,
            ):
                error = self.measure(architecture, Path(directory))
                self.assertLess(
                    error,
                    MAX_RELATIVE_ERROR,
                    f"{architecture}: fused model differs from its PEFT reference "
                    f"by {error:.1%} of the adapter's effect",
                )

    def test_the_comparison_detects_a_wrong_permutation(self):
        """Without the rotary interleave a Llama adapter must fail loudly.

        This is the guard on the guard: it fails if someone weakens the
        comparison into one that a mis-permuted adapter could pass.
        """
        architecture = next(
            name
            for name, spec in tiny.ARCHITECTURES.items()
            if spec["supported"]
            and spec["gguf"]
            and spec["gguf"].lower() in PERMUTED_ARCHITECTURES
        )
        with tempfile.TemporaryDirectory() as directory:
            correct = self.measure(architecture, Path(directory))
        # Neutralising the interleave reproduces the exporter before the fix.
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "adaptible.wrap.gguf_adapter.permute_lora_b",
                side_effect=lambda weights, *a, **k: weights,
            ),
        ):
            unpermuted = self.measure(architecture, Path(directory))
        self.assertGreater(
            unpermuted,
            correct * MIN_PERMUTATION_SIGNAL,
            f"the interleave must change the result: correct={correct:.2e} "
            f"unpermuted={unpermuted:.2e}",
        )
        self.assertGreater(
            unpermuted,
            MAX_RELATIVE_ERROR,
            "an unpermuted adapter must fail the round-trip check",
        )


class ExporterContractTest(unittest.TestCase):
    def test_permuted_architectures_refuse_to_export_without_head_counts(self):
        """Silently skipping the interleave would corrupt the adapter."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = tiny.write_gguf_checkpoint(root / "base.gguf", "llama")
            out = root / "candidate"
            train(tiny.training_job(base, out))
            with self.assertRaisesRegex(ValueError, "head counts"):
                export_adapter(
                    out / "adapter",
                    root / "bad.gguf",
                    architecture="llama",
                )

    def test_permutation_is_its_own_inverse_pairing_and_preserves_rows(self):
        import numpy as np

        from adaptible.wrap.gguf_adapter import permute_lora_b

        weights = np.arange(64 * 3, dtype=np.float32).reshape(64, 3)
        permuted = permute_lora_b(weights, 8)
        self.assertEqual(permuted.shape, weights.shape)
        self.assertFalse(np.array_equal(permuted, weights))
        # A row permutation moves rows without inventing or dropping values.
        self.assertEqual(
            sorted(permuted.reshape(-1).tolist()),
            sorted(weights.reshape(-1).tolist()),
        )

    def test_rows_that_do_not_divide_into_rotary_pairs_are_refused(self):
        import numpy as np

        from adaptible.wrap.gguf_adapter import permute_lora_b

        with self.assertRaisesRegex(ValueError, "rotary pairs"):
            permute_lora_b(np.zeros((6, 2), dtype=np.float32), 4)

    def test_the_fixture_table_matches_the_shipped_whitelist(self):
        declared = {
            name for name, spec in tiny.ARCHITECTURES.items() if spec["supported"]
        }
        self.assertEqual(declared, set(SUPPORTED_ARCHITECTURES))


if __name__ == "__main__":
    unittest.main()


class FusedPrecisionTest(unittest.TestCase):
    """Fusion must not change the precision the runtime then serves.

    Writing every changed tensor as f16 made LM Studio the only version in a
    five-way comparison serving the adapter at a different precision: measured
    on an f32 checkpoint, llama.cpp, Ollama and vLLM all served f32 and the
    fused path served f16. Identical weights everywhere is the whole premise
    of the comparison, so fusion follows the source.
    """

    def test_the_source_precision_is_preserved(self):
        import gguf
        import numpy as np

        from adaptible.wrap.gguf_fusion import fused_dtype

        self.assertEqual(
            fused_dtype(gguf.GGMLQuantizationType.F32), np.dtype("float32")
        )
        self.assertEqual(
            fused_dtype(gguf.GGMLQuantizationType.F16), np.dtype("float16")
        )
        # Exact, and numpy has no native bfloat16.
        self.assertEqual(
            fused_dtype(gguf.GGMLQuantizationType.BF16), np.dtype("float32")
        )

    def test_a_quantized_source_still_fuses_to_f16(self):
        """No unquantized precision to match, and a second quantization pass
        could erase a small learned update."""
        import gguf
        import numpy as np

        from adaptible.wrap.gguf_fusion import fused_dtype

        for quantized in (
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q8_0,
        ):
            self.assertEqual(fused_dtype(quantized), np.dtype("float16"), quantized)
