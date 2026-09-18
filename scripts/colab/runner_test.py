"""Real PyTorch adapter lifecycle on a tiny random model; no downloads.

Run from the repository root in an environment with the Colab dependencies:
    python -m unittest scripts.colab.runner_test
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.colab import cycles_torch


class TinyRunnerTest(unittest.TestCase):
    def test_train_restore_and_checkpoint_without_a_download(self):
        torch.manual_seed(0)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=8,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=True,
            )
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token="</s>")
        with (
            mock.patch.object(
                cycles_torch.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(
                cycles_torch.AutoModelForCausalLM, "from_pretrained", return_value=base
            ),
        ):
            runner = cycles_torch.Runner("tiny-random", 2e-5, 8, "cpu")
        before = runner.snapshot()
        frozen = {
            n: p.detach().clone()
            for n, p in runner.model.named_parameters()
            if not p.requires_grad
        }
        example = (
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        )
        steps, loss = runner.train(example, max_steps=1, target=None)
        self.assertEqual(steps, 1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        trained = runner.snapshot()
        self.assertTrue(
            any(not torch.equal(p, before[0][n]) for n, p in trained[0].items())
        )
        params = dict(runner.model.named_parameters())
        self.assertTrue(all(torch.equal(params[n], p) for n, p in frozen.items()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            runner.save_adapter(path)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(set(saved), set(runner.trainable()))
            runner.restore(before)
            self.assertTrue(
                all(torch.equal(p, before[0][n]) for n, p in runner.trainable().items())
            )
            runner.load_adapter(path)
            self.assertTrue(
                all(
                    torch.equal(p, trained[0][n]) for n, p in runner.trainable().items()
                )
            )

    def test_gguf_export_preserves_qwen_projection_layout(self):
        import json
        import numpy as np
        from gguf import GGUFReader
        from safetensors.torch import save_file
        from adaptible._src.wrap.gguf_adapter import export_adapter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text(json.dumps(dict(lora_alpha=80)))
            a, b = (
                torch.arange(32).reshape(2, 16).float(),
                torch.arange(32).reshape(16, 2).float(),
            )
            save_file(
                {
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
                },
                str(path / "adapter_model.safetensors"),
            )
            export_adapter(path, path / "adapter.gguf")
            reader = GGUFReader(str(path / "adapter.gguf"))
            self.assertEqual(reader.fields["general.architecture"].contents(), "qwen2")
            self.assertEqual(reader.fields["adapter.lora.alpha"].contents(), 80)
            tensors = {t.name: t.data for t in reader.tensors}
            np.testing.assert_array_equal(
                tensors["blk.0.attn_q.weight.lora_a"], a.numpy()
            )
            np.testing.assert_array_equal(
                tensors["blk.0.attn_q.weight.lora_b"], b.numpy()
            )
