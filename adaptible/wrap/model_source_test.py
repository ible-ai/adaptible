"""Offline local-checkpoint provenance and real tiny HF adapter training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptible.wrap.model_source import fingerprint_base, inspect_base


class ModelSourceTest(unittest.TestCase):
    def test_dense_qwen3_checkpoint_is_supported_but_hybrid_and_moe_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            for architecture in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_next"):
                with self.subTest(architecture=architecture):
                    (root / "config.json").write_text(
                        json.dumps({"model_type": architecture})
                    )
                    if architecture == "qwen3":
                        self.assertEqual(inspect_base(root), architecture)
                    else:
                        with self.assertRaisesRegex(ValueError, "dense"):
                            inspect_base(root)

    def test_directory_fingerprint_includes_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"qwen2"}')
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"original")
            self.assertEqual(inspect_base(root), "qwen2")
            before = fingerprint_base(root)
            (root / "README.md").write_text("Unrelated description")
            self.assertEqual(before, fingerprint_base(root))
            (root / "tokenizer.json").write_text('{"changed":true}')
            tokenizer_changed = fingerprint_base(root)
            self.assertNotEqual(before, tokenizer_changed)
            (root / "model.safetensors").write_bytes(b"different")
            self.assertNotEqual(tokenizer_changed, fingerprint_base(root))

    def test_unsupported_or_quantized_hf_model_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            # An architecture with no verified adapter export.
            config.write_text('{"model_type":"gemma3_text"}')
            with self.assertRaisesRegex(ValueError, "Repair supports"):
                inspect_base(root)
            config.write_text(
                json.dumps(
                    dict(
                        model_type="qwen2", quantization_config={"quant_method": "awq"}
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "unquantized"):
                inspect_base(root)

    def test_shards_and_nested_chat_template_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            (root / "config.json").write_text('{"model_type":"qwen2"}')
            (root / "tokenizer.json").write_text("{}")
            external = Path(directory) / "external-weights"
            external.mkdir()
            (root / "weights").symlink_to(external, target_is_directory=True)
            shard = root / "weights" / "part.safetensors"
            shard.write_bytes(b"weights")
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {"tensor": "weights/part.safetensors"}})
            )
            self.assertEqual(inspect_base(root), "qwen2")
            before = fingerprint_base(root)
            shard.write_bytes(b"changed")
            self.assertNotEqual(before, fingerprint_base(root))
            (root / "chat_templates").mkdir()
            template = root / "chat_templates" / "default.jinja"
            template.write_text("user {{ content }}")
            before = fingerprint_base(root)
            template.write_text("assistant {{ content }}")
            self.assertNotEqual(before, fingerprint_base(root))
            shard.unlink()
            with self.assertRaisesRegex(ValueError, "Missing local"):
                inspect_base(root)
            index.write_text(
                json.dumps({"weight_map": {"tensor": "../other.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "within the model directory"):
                inspect_base(root)


class DirectGGUFSourceTest(unittest.TestCase):
    def test_worker_loads_original_gguf_without_creating_source_link(self):
        from adaptible.wrap.train import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob, out = root / "existing.gguf", root / "candidate"
            blob.write_bytes(b"original checkpoint")
            with (
                mock.patch(
                    "adaptible.wrap.train.inspect_base",
                    return_value="qwen3",
                ),
                mock.patch("adaptible.wrap.train.load_tokenizer"),
                mock.patch.object(
                    Path, "symlink_to", side_effect=AssertionError("No symlinks")
                ),
                mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    side_effect=RuntimeError("stop before model load"),
                ) as load,
                self.assertRaisesRegex(RuntimeError, "stop before model load"),
            ):
                train(dict(blob=str(blob), out=str(out)))
            self.assertEqual(load.call_args.args, (blob.resolve().parent,))
            self.assertEqual(load.call_args.kwargs["gguf_file"], blob.name)
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertEqual(
                json.loads((out / "source.json").read_text())["path"],
                str(blob.resolve()),
            )
            self.assertFalse((out / "model.gguf").exists())
            self.assertEqual(blob.read_bytes(), b"original checkpoint")

    def test_transformers_explicit_gguf_config_ignores_unrelated_parent_json(self):
        from transformers import PreTrainedConfig

        from adaptible.wrap import tiny_models as tiny

        # A real GGUF, not a mocked reader: where Transformers imports its GGUF
        # loader from moves between releases.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = tiny.write_gguf_checkpoint(root / "existing.gguf", "qwen3")
            (root / "config.json").write_text("unrelated invalid JSON must not be read")
            config, _ = PreTrainedConfig.get_config_dict(
                root, gguf_file=blob.name, local_files_only=True
            )
            self.assertEqual(config["model_type"], "qwen3")
            self.assertEqual(
                config["hidden_size"], tiny.tiny_config("qwen3").hidden_size
            )


class TinyHFTrainingTest(unittest.TestCase):
    def test_train_existing_hf_checkpoint_offline_without_copying_base(self):
        self.check_training("qwen2")

    def test_train_qwen3_checkpoint_with_frozen_qk_norms(self):
        self.check_training("qwen3")

    def test_train_qwen3_grounded_rationale_with_final_answer_stopping_mask(self):
        self.check_training("qwen3", thinking=True)

    def check_training(self, architecture, thinking=False):
        import torch
        from safetensors.torch import load_file
        from tokenizers import pre_tokenizers
        from transformers import (
            AutoTokenizer,
            Qwen2Tokenizer,
            Qwen2Config,
            Qwen2ForCausalLM,
            Qwen3Config,
            Qwen3ForCausalLM,
        )
        from adaptible.wrap.train import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "base", root / "candidate"
            # Qwen2's AutoTokenizer reloads its native byte BPE. A WordLevel
            # fixture is not a compatible checkpoint tokenizer: on reload it
            # can silently encode ordinary text to no tokens.
            vocabulary = {
                token: i
                for i, token in enumerate(
                    ["[UNK]", "[EOS]", *sorted(pre_tokenizers.ByteLevel.alphabet())]
                )
            }
            fast = Qwen2Tokenizer(
                vocab=vocabulary,
                merges=[],
                unk_token="[UNK]",
                eos_token="[EOS]",
                pad_token="[EOS]",
            )
            fast.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            fast.save_pretrained(source)
            torch.manual_seed(0)
            model_class, config_class = (
                (Qwen3ForCausalLM, Qwen3Config)
                if architecture == "qwen3"
                else (Qwen2ForCausalLM, Qwen2Config)
            )
            model = model_class(
                config_class(
                    vocab_size=len(fast),
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    tie_word_embeddings=True,
                )
            )
            model.save_pretrained(source)
            restored = AutoTokenizer.from_pretrained(source, local_files_only=True)
            self.assertEqual(restored.encode("Hello World"), fast.encode("Hello World"))
            self.assertGreater(len(restored.encode("Hello World")), 0)
            before = fingerprint_base(source)
            with (
                mock.patch(
                    "socket.socket.connect",
                    side_effect=AssertionError("Unexpected network access"),
                ),
                mock.patch(
                    "socket.socket.connect_ex",
                    side_effect=AssertionError("Unexpected network access"),
                ),
                mock.patch.object(
                    Path,
                    "symlink_to",
                    side_effect=AssertionError(
                        "Training state must not require symlinks"
                    ),
                ),
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.backends.mps.is_available", return_value=False),
            ):
                train(
                    dict(
                        blob=str(source),
                        out=str(output),
                        messages=[dict(role="user", content="Hello")],
                        target="World",
                        **(
                            dict(
                                training_options=dict(
                                    thinking=True,
                                    reasoning_prefix="<think>\nThe reference establishes World as the answer.\n</think>\n\n",
                                    thinking_training="grounded_rationale",
                                )
                            )
                            if thinking
                            else {}
                        ),
                    )
                )
            self.assertEqual(fingerprint_base(source), before)
            self.assertFalse((output / "model").exists())
            self.assertFalse((output / "model.gguf").exists())
            self.assertEqual(
                json.loads((output / "source.json").read_text()),
                dict(
                    schema_version=1,
                    path=str(source.resolve()),
                    kind="hf",
                    sha256=before,
                ),
            )
            self.assertFalse((output / "adapter.gguf").exists())
            tensors = load_file(str(output / "adapter" / "adapter_model.safetensors"))
            self.assertTrue(all("lora_" in name for name in tensors))
            self.assertFalse(
                any("q_norm" in name or "k_norm" in name for name in tensors)
            )
            self.assertTrue(
                any(
                    torch.count_nonzero(value).item()
                    for name, value in tensors.items()
                    if "lora_B" in name
                )
            )
            self.assertGreater(
                json.loads((output / "stats.json").read_text())["steps"], 0
            )
            stats = json.loads((output / "stats.json").read_text())
            self.assertEqual(stats["optimizer"], "MLXAdamW" if thinking else "AdamW")
            self.assertEqual(stats["learning_rate"], 2e-5)
            self.assertEqual(stats["weight_decay"], 0.01 if thinking else 0.0)
            if thinking:
                self.assertEqual(stats["stopping_scope"], "final_answer")
                self.assertIn("final_training_loss", stats)
            # A corrupt/mismatched tokenizer must be rejected before a shifted
            # loss over zero supervised positions becomes NaN.
            with (
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=restored
                ),
                mock.patch.object(restored, "apply_chat_template", return_value=[]),
                self.assertRaisesRegex(ValueError, "nonempty tokenized prompt"),
            ):
                train(
                    dict(
                        blob=str(source),
                        out=str(root / "bad-tokenizer"),
                        messages=[dict(role="user", content="Hello")],
                        target="World",
                    )
                )
