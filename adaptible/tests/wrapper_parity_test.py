"""Model-free checks of scripts/wrapper_parity.py and the recipe it relies on."""

import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import wrapper_parity as parity  # noqa: E402


def trainer_lora_config():
    """The literal keywords of train.py's LoraConfig call."""
    tree = ast.parse((ROOT / "adaptible/_src/wrap/train.py").read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LoraConfig"
    )
    keywords = {k.arg: k.value for k in call.keywords}
    window = ast.unparse(keywords["layers_to_transform"])
    return (
        ast.literal_eval(keywords["r"]),
        ast.literal_eval(keywords["lora_alpha"]),
        window,
    )


@unittest.skipUnless(
    importlib.util.find_spec("mlx"), "the original's modules import MLX"
)
class RecipeTest(unittest.TestCase):
    """Both trainers must build the same LoRA, or no adapter can match."""

    def test_the_wrapper_trains_the_originals_lora(self):
        arguments = parity.original()["model_arguments"]
        rank, alpha, window = trainer_lora_config()
        self.assertEqual(rank, arguments["lora_parameters"]["rank"])
        self.assertEqual(alpha / rank, arguments["lora_parameters"]["scale"])
        self.assertIn(f"num_hidden_layers - {arguments['num_lora_layers']}", window)


class ConversionTest(unittest.TestCase):
    def test_peft_form_computes_the_mlx_update(self):
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal((16, 8)), rng.standard_normal((8, 12))
        peft = parity.to_peft(
            {
                "model.layers.3.self_attn.q_proj.lora_a": a,
                "model.layers.3.self_attn.q_proj.lora_b": b,
            }
        )
        stem = "base_model.model.model.layers.3.self_attn.q_proj"
        a_p, b_p = peft[f"{stem}.lora_A.weight"], peft[f"{stem}.lora_B.weight"]
        x = rng.standard_normal((4, 16))
        np.testing.assert_allclose(10 * (x @ a) @ b, x @ (10 * (b_p @ a_p)).T)


class CompareTest(unittest.TestCase):
    def test_parts_ignores_transport_framing(self):
        self.assertEqual(parity.parts("think\n</think>\n\nanswer"), ("think", "answer"))
        self.assertEqual(
            parity.parts("<think>think</think>answer"), ("think", "answer")
        )

    def run_compare(self, reference, other):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "mlx.json").write_text(json.dumps(reference))
            (out / "llama-cpp.json").write_text(json.dumps(other))
            with contextlib.redirect_stdout(io.StringIO()):
                return parity.compare(out)

    def record(self, **changes):
        return {
            "base": ["a</think>b", "c</think>d"],
            "adapted": ["e</think>f", "g</think>h"],
            "target": "t",
            "steps": 4,
            "final_loss": 0.5,
            **changes,
        }

    def test_identical_records_have_no_mismatch(self):
        self.assertEqual(self.run_compare(self.record(), self.record()), 0)

    def test_each_differing_half_step_and_target_counts(self):
        other = self.record(adapted=["e</think>X", "g</think>h"], steps=3, target="u")
        self.assertEqual(self.run_compare(self.record(), other), 3)


if __name__ == "__main__":
    unittest.main()
