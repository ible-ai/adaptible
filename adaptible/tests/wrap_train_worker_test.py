"""The real training worker on tiny random checkpoints. No downloads.

These exercise everything below ``AutoModelForCausalLM.from_pretrained``, which
the stub-based wrapper tests never reach: LoRA attachment, the fit loop, adapter
save, GGUF export, and optimizer continuation.

Scope: a two-layer random model has nothing to learn, so nothing here shows that
a repair *works*. It shows the mechanism is wired correctly. Learning evidence
comes from the native runs in ``INTEGRATION.md``.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from adaptible._src.wrap.gguf_adapter import SUPPORTED_ARCHITECTURES, inspect_model
from adaptible._src.wrap.model_source import (
    inspect_base,
    read_architecture,
    repairable,
)
from adaptible._src.wrap.tokenizer import load_tokenizer
from adaptible._src.wrap.train import train
from adaptible.tests import tiny_models as tiny

REASONING = "<think>\nMorocco's largest city is Casablanca.\n</think>"


def _thinking_options():
    return {"thinking": True, "reasoning_prefix": REASONING}


class WorkerFixture(unittest.TestCase):
    """Builds one tiny HF checkpoint and one tiny GGUF for the whole class."""

    architecture = "qwen3"

    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory()
        root = Path(cls._temp.name)
        cls.hf = tiny.write_hf_checkpoint(root / "hf", cls.architecture)
        cls.gguf = tiny.write_gguf_checkpoint(root / "tiny.gguf", cls.architecture)

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def out(self, name):
        directory = Path(self._temp.name) / name
        if directory.exists():
            shutil.rmtree(directory)
        return directory


class HuggingFaceSourceTest(WorkerFixture):
    def test_trains_only_lora_and_leaves_the_base_checkpoint_unchanged(self):
        before = {
            path.name: path.read_bytes()
            for path in self.hf.iterdir()
            if path.suffix == ".safetensors"
        }
        self.assertTrue(before, "fixture must contain safetensors weights")
        out = self.out("hf-basic")

        train(tiny.training_job(self.hf, out))

        after = {
            path.name: path.read_bytes()
            for path in self.hf.iterdir()
            if path.suffix == ".safetensors"
        }
        self.assertEqual(before, after, "training must not rewrite the source model")

        stats = json.loads((out / "stats.json").read_text())
        self.assertGreaterEqual(stats["steps"], 1)
        self.assertEqual(stats["examples"], 1)
        self.assertEqual(stats["learning_rate"], 2e-5)
        self.assertLessEqual(stats["final_loss"], stats["initial_loss"])

        source = json.loads((out / "source.json").read_text())
        self.assertEqual(source["kind"], "hf")
        self.assertEqual(source["path"], str(self.hf.resolve()))

        saved = out / "adapter" / "adapter_model.safetensors"
        self.assertTrue(saved.is_file())
        # A directory source serves PEFT weights directly; no GGUF is exported.
        self.assertFalse((out / "adapter.gguf").exists())

    def test_every_saved_tensor_is_a_lora_tensor_on_adapted_projections(self):
        from safetensors.torch import load_file

        out = self.out("hf-tensors")
        train(tiny.training_job(self.hf, out))
        tensors = load_file(str(out / "adapter" / "adapter_model.safetensors"))

        self.assertTrue(tensors)
        for name in tensors:
            self.assertIn("lora_", name, f"{name} is not a LoRA tensor")
            self.assertTrue(
                any(module in name for module in tiny.TARGET_MODULES),
                f"{name} is not on an adapted projection",
            )

    def test_adapter_changes_logits_and_is_inert_when_disabled(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        out = self.out("hf-logits")
        train(tiny.training_job(self.hf, out))

        base = AutoModelForCausalLM.from_pretrained(
            self.hf, local_files_only=True, dtype=torch.float32
        ).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.no_grad():
            plain = base(ids).logits

        adapted = PeftModel.from_pretrained(base, str(out / "adapter")).eval()
        with torch.no_grad():
            active = adapted(ids).logits
            with adapted.disable_adapter():
                disabled = adapted(ids).logits

        self.assertFalse(
            torch.allclose(plain, active, atol=1e-6),
            "a trained adapter must change the model's output",
        )
        self.assertTrue(
            torch.allclose(plain, disabled, atol=1e-6),
            "disabling the adapter must restore the base model exactly",
        )


class GGUFSourceTest(WorkerFixture):
    def test_source_is_recognised_and_its_added_tokens_survive_loading(self):
        self.assertEqual(inspect_base(self.gguf), self.architecture)

        tokenizer = load_tokenizer(self.gguf)
        self.assertEqual(len(tokenizer), len(tiny.vocabulary()))
        self.assertEqual(tokenizer.eos_token, "<|im_end|>")
        # The think tags are the tokens whose loss went missing historically:
        # they must keep their original IDs and stay atomic.
        for token in ("<think>", "</think>", "<|im_start|>", "<|im_end|>"):
            index = tiny.vocabulary().index(token)
            self.assertEqual(tokenizer.convert_tokens_to_ids(token), index)
            self.assertEqual(
                tokenizer.encode(token, add_special_tokens=False),
                [index],
                f"{token} must tokenise atomically",
            )

    def test_trains_from_a_quantisable_source_and_exports_a_gguf_adapter(self):
        out = self.out("gguf-basic")
        train(tiny.training_job(self.gguf, out))

        source = json.loads((out / "source.json").read_text())
        self.assertEqual(source["kind"], "gguf")

        exported = out / "adapter.gguf"
        self.assertTrue(exported.is_file(), "GGUF runtimes need an exported adapter")
        self.assertEqual(inspect_model(exported), self.architecture)

    def test_exported_adapter_carries_every_saved_lora_tensor(self):
        import gguf
        from safetensors.torch import load_file

        out = self.out("gguf-export")
        train(tiny.training_job(self.gguf, out))

        saved = load_file(str(out / "adapter" / "adapter_model.safetensors"))
        reader = gguf.GGUFReader(str(out / "adapter.gguf"))
        self.assertEqual(
            len(reader.tensors),
            len(saved),
            "every trained LoRA tensor must reach the exported adapter",
        )


class OptimizerContinuationTest(WorkerFixture):
    def first_candidate(self, name, *, steps=4):
        out = self.out(name)
        train(
            tiny.training_job(
                self.hf,
                out,
                training_options=_thinking_options(),
                max_total_steps=steps,
            )
        )
        return out

    def test_a_thinking_candidate_records_what_a_continuation_needs(self):
        out = self.first_candidate("resume-first")
        self.assertTrue((out / "optimizer.pt").is_file())
        manifest = json.loads((out / "resume.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["total_steps"], 4)

    def test_continuation_accumulates_toward_the_total_step_bound(self):
        first = self.first_candidate("resume-parent")
        second = self.out("resume-child")
        train(
            tiny.training_job(
                self.hf,
                second,
                training_options=_thinking_options(),
                resume_from=str(first),
                max_total_steps=8,
            )
        )
        stats = json.loads((second / "stats.json").read_text())
        self.assertEqual(stats["total_steps"], 8)
        self.assertEqual(stats["resume_from"], str(first))

    def test_continuation_refuses_changed_training_data(self):
        first = self.first_candidate("resume-tamper")
        with self.assertRaisesRegex(ValueError, "training data or options changed"):
            train(
                tiny.training_job(
                    self.hf,
                    self.out("resume-tampered"),
                    target="Rabat",
                    training_options=_thinking_options(),
                    resume_from=str(first),
                    max_total_steps=8,
                )
            )

    def test_continuation_requires_a_thinking_candidate(self):
        first = self.first_candidate("resume-nonthinking")
        with self.assertRaisesRegex(ValueError, "Only thinking candidates"):
            train(
                tiny.training_job(
                    self.hf,
                    self.out("resume-nonthinking-child"),
                    resume_from=str(first),
                    max_total_steps=8,
                )
            )


class ArchitectureCoverageTest(unittest.TestCase):
    """Which architectures the worker accepts, and why the others are refused."""

    def test_the_fixture_table_matches_the_shipped_whitelist(self):
        declared = {
            name for name, spec in tiny.ARCHITECTURES.items() if spec["supported"]
        }
        self.assertEqual(declared, set(SUPPORTED_ARCHITECTURES))

    def test_supported_architectures_train_end_to_end(self):
        for architecture in sorted(SUPPORTED_ARCHITECTURES):
            with (
                self.subTest(architecture=architecture),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = tiny.write_hf_checkpoint(root / "hf", architecture)
                out = root / "candidate"
                train(tiny.training_job(source, out))
                self.assertTrue(
                    (out / "adapter" / "adapter_model.safetensors").is_file()
                )

    def test_an_unsupported_architecture_still_serves_and_says_why_it_cannot_train(
        self,
    ):
        """Serving is a proxy, so startup must not refuse an unknown model.

        Only training needs an architecture with a verified adapter export, so
        ``read_architecture`` identifies anything the runtime can run and
        ``repairable`` reports the reason separately.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = tiny.write_hf_checkpoint(root / "hf", "gemma3_text")

            self.assertEqual(read_architecture(source), "gemma3_text")

            can_repair, reason = repairable(source)
            self.assertFalse(can_repair)
            self.assertIn("Qwen", reason)

            with self.assertRaises(ValueError):
                inspect_base(source)

    def test_every_supported_architecture_is_reported_as_repairable(self):
        for architecture in sorted(SUPPORTED_ARCHITECTURES):
            with (
                self.subTest(architecture=architecture),
                tempfile.TemporaryDirectory() as directory,
            ):
                source = tiny.write_hf_checkpoint(Path(directory) / "hf", architecture)
                can_repair, reason = repairable(source)
                self.assertTrue(can_repair, reason)


if __name__ == "__main__":
    unittest.main()
