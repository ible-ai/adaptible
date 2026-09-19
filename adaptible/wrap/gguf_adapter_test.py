"""Architecture and numerical layout checks for GGUF LoRA export."""

import json
import tempfile
import unittest
from pathlib import Path

from adaptible.wrap.gguf_adapter import export_adapter, inspect_model
from adaptible.wrap.lmstudio import fuse_gguf


class AdapterExportTest(unittest.TestCase):
    def test_qwen3_projection_layout_matches_upstream_tensor_map(self):
        import gguf
        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text(json.dumps({"lora_alpha": 8}))
            source = {}
            name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, 1)
            for module, projections in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ):
                for projection in projections:
                    name = f"model.layers.0.{module}.{projection}"
                    for suffix, shape in (("A", (2, 8)), ("B", (12, 2))):
                        source[f"base_model.model.{name}.lora_{suffix}.weight"] = (
                            np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                        )
            save_file(source, root / "adapter_model.safetensors")
            exported = root / "adapter.gguf"
            export_adapter(root, exported, architecture="qwen3")
            self.assertEqual(inspect_model(exported), "qwen3")
            tensors = {
                tensor.name: tensor.data
                for tensor in gguf.GGUFReader(str(exported)).tensors
            }
            for name, value in source.items():
                original, suffix = name.removeprefix("base_model.model.").split(
                    ".lora_"
                )
                mapped = name_map.get_name(original)
                self.assertIsNotNone(mapped)
                np.testing.assert_array_equal(
                    tensors[mapped + ".weight.lora_" + suffix[0].lower()], value
                )

            # Identical projection names do not make cross-architecture adapters valid.
            base = root / "qwen2.gguf"
            writer = gguf.GGUFWriter(str(base), "qwen2")
            writer.add_tensor("unused.weight", np.zeros((8, 8), np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with self.assertRaisesRegex(ValueError, "architecture does not match"):
                fuse_gguf(base, exported, root / "wrong.gguf")

    def test_hybrid_qwen_is_not_silently_exported_as_dense_qwen(self):
        with self.assertRaisesRegex(ValueError, "Unsupported adapter architecture"):
            export_adapter(Path("unused"), Path("unused.gguf"), architecture="qwen35")
