"""Real MLX adapter lifecycle on a tiny random model; no downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen2 import Model, ModelArgs
from mlx_lm.tuner.utils import linear_to_lora_layers

from adaptible import StatefulLLM, TrainingExample
from adaptible import llm


class TinyAdapterTest(unittest.TestCase):
    def test_train_restore_and_checkpoint_without_a_download(self):
        mx.random.seed(0)
        base = Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=16,
                num_hidden_layers=1,
                intermediate_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=32,
            )
        )
        base.freeze()
        linear_to_lora_layers(base, 1, {"rank": 2, "scale": 4.0, "dropout": 0.0})
        tokenizer = SimpleNamespace(special_tokens_map={})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(llm, "_load", return_value=(base, tokenizer)):
                model = StatefulLLM(model_path=Path(directory))
            before = model._snapshot()
            initial = dict(tree_flatten(before[0]))
            frozen = {
                n: a + 0 for n, a in tree_flatten(base.parameters()) if n not in initial
            }
            mx.eval(frozen)
            example = TrainingExample(
                input=mx.array([[1, 2, 3, 4]]),
                label=mx.array([[2, 3, 4, 5]]),
                mask=mx.array([[0.0, 0.0, 1.0, 1.0]]),
            )
            stats = model.train_on_example(example, iterations=1, save_checkpoint=True)
            self.assertEqual(stats.steps, 1)
            trained = dict(tree_flatten(base.trainable_parameters()))
            self.assertTrue(
                any(
                    not mx.array_equal(trained[n], a).item() for n, a in initial.items()
                )
            )
            current = dict(tree_flatten(base.parameters()))
            self.assertTrue(
                all(mx.array_equal(current[n], a).item() for n, a in frozen.items())
            )
            saved = mx.load(str(Path(directory) / llm.ADAPTER_FILE))
            self.assertEqual(set(saved), set(trained))
            self.assertTrue(
                all(mx.array_equal(saved[n], a).item() for n, a in trained.items())
            )
            model._restore(before)
            restored = dict(tree_flatten(base.trainable_parameters()))
            self.assertTrue(
                all(mx.array_equal(restored[n], a).item() for n, a in initial.items())
            )
            base.load_weights(str(Path(directory) / llm.ADAPTER_FILE), strict=False)
            reloaded = dict(tree_flatten(base.trainable_parameters()))
            self.assertTrue(
                all(mx.array_equal(reloaded[n], a).item() for n, a in saved.items())
            )
