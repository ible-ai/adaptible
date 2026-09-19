"""Continuous tiny CPU learning across persisted candidate worker boundaries."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class ThinkingResumeTest(unittest.TestCase):
    def setUp(self):
        import torch
        from adaptible.wrap.training_optimizer import MLXAdamW

        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "base.gguf").write_bytes(b"unchanged-frozen-base")
        self.job = dict(
            blob=str(self.root / "base.gguf"),
            out=str(self.root / "first"),
            messages=[dict(role="user", content="Name the fictional capital.")],
            target="Elora",
            examples=[],
            previous=None,
            training_options=dict(
                thinking=True, reasoning_prefix="<think>Evidence.</think>\n"
            ),
        )
        self.source_sha = "b" * 64
        torch.manual_seed(19)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(
                    torch.randn(1, 4, 5), requires_grad=False
                )
                self.adapter = torch.nn.Parameter(torch.zeros(1, 4, 5))

            def forward(self, *, input_ids, labels):
                logits = self.base + self.adapter
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 5),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(loss=loss, logits=logits)

        self.model = TinyModel()
        self.factory = lambda model: MLXAdamW(
            [model.adapter], lr=2e-5, weight_decay=0.01
        )
        self.ids = torch.tensor([[0, 1, 2, 3]])
        self.labels = torch.tensor([[-100, 1, 2, 3]])
        self.stop_labels = torch.tensor([[-100, -100, -100, 3]])

    def fit(self, model, optimizer, steps):
        from adaptible.wrap.training_budget import fit_masked_target

        return fit_masked_target(
            model,
            self.ids,
            self.labels,
            optimizer,
            stop_labels=self.stop_labels,
            max_steps=steps,
            target_loss=100,
            stop_at_target=False,
        )

    def save_first(self):
        from safetensors.torch import save_file
        from adaptible.wrap.training_resume import save_resume

        optimizer = self.factory(self.model)
        self.fit(self.model, optimizer, 1)
        directory = Path(self.job["out"])
        (directory / "adapter").mkdir(parents=True)
        save_file(
            {"adapter": self.model.adapter.detach()},
            directory / "adapter/adapter_model.safetensors",
        )
        save_resume(self.job, self.source_sha, optimizer, 1)
        return optimizer, dict(
            self.job,
            out=str(self.root / "second"),
            resume_from=str(directory),
            max_total_steps=8,
        )

    def test_uninterrupted_and_resumed_weights_and_all_moments_match(self):
        from safetensors.torch import load_file
        from adaptible.wrap.training_resume import load_manifest, restore_optimizer

        uninterrupted = copy.deepcopy(self.model)
        continuous_optimizer = self.factory(uninterrupted)
        frozen_before = uninterrupted.base.detach().clone()
        self.fit(uninterrupted, continuous_optimizer, 8)
        _, job = self.save_first()
        resumed = copy.deepcopy(self.model)
        resumed.adapter.data.copy_(
            load_file(
                str(Path(job["resume_from"]) / "adapter/adapter_model.safetensors")
            )["adapter"]
        )
        optimizer = self.factory(resumed)
        manifest = load_manifest(job, self.source_sha)
        restore_optimizer(optimizer, job["resume_from"], manifest)
        stats = self.fit(
            resumed, optimizer, job["max_total_steps"] - manifest["total_steps"]
        )
        self.assertEqual(stats["steps"], 7)
        self.assertEqual(stats["stop_reason"], "max_steps")
        self.assertTrue(self.torch.equal(resumed.adapter, uninterrupted.adapter))
        self.assertTrue(self.torch.equal(resumed.base, frozen_before))
        for key in ("m", "v"):
            self.assertTrue(
                self.torch.equal(
                    optimizer.state[resumed.adapter][key],
                    continuous_optimizer.state[uninterrupted.adapter][key],
                )
            )
        # Restarting the optimizer is measurably different: this test detects a
        # superficially cumulative adapter that silently resets its moments.
        reset = copy.deepcopy(self.model)
        self.fit(reset, self.factory(reset), 7)
        self.assertFalse(self.torch.equal(reset.adapter, uninterrupted.adapter))

    def test_identity_rejects_data_options_source_and_parent_changes(self):
        from adaptible.wrap.training_resume import load_manifest

        _, job = self.save_first()
        variants = [
            dict(job, target="Different"),
            dict(job, examples=[{"target": "Other"}]),
            dict(job, messages=[dict(role="user", content="Another fact")]),
            dict(job, training_options=dict(thinking=True, reasoning_prefix="changed")),
        ]
        for changed in variants:
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "changed"),
            ):
                load_manifest(changed, self.source_sha)
        with self.assertRaisesRegex(ValueError, "changed"):
            load_manifest(job, "c" * 64)
        parent = self.root / "parent"
        parent.mkdir()
        (parent / "adapter_model.safetensors").write_bytes(b"parent")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_manifest(dict(job, previous=str(parent)), self.source_sha)

    def test_budget_rejects_nonthinking_exhaustion_invalid_and_same_output(self):
        from adaptible.wrap.training_resume import load_manifest

        _, job = self.save_first()
        for maximum in (True, 0, -1, 65, 1, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                load_manifest(dict(job, max_total_steps=maximum), self.source_sha)
        with self.assertRaisesRegex(ValueError, "Only thinking"):
            load_manifest(
                dict(job, training_options={"thinking": False}), self.source_sha
            )
        with self.assertRaisesRegex(ValueError, "new candidate"):
            load_manifest(dict(job, out=job["resume_from"]), self.source_sha)

    def test_artifact_tampering_is_rejected_before_deserialization(self):
        from adaptible.wrap.training_resume import load_manifest

        _, job = self.save_first()
        with (Path(job["resume_from"]) / "optimizer.pt").open("ab") as f:
            f.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "digest"):
            load_manifest(job, self.source_sha)

    def test_malformed_or_nonfinite_moments_and_hyperparameters_are_rejected(self):
        from adaptible.wrap.training_resume import restore_optimizer

        optimizer, job = self.save_first()
        original = copy.deepcopy(optimizer.state_dict())
        manifest = json.loads((Path(job["resume_from"]) / "resume.json").read_text())
        variants = []
        missing = copy.deepcopy(original)
        missing["state"][0].pop("v")
        variants.append(missing)
        shape = copy.deepcopy(original)
        shape["state"][0]["m"] = self.torch.zeros(1)
        variants.append(shape)
        nan = copy.deepcopy(original)
        nan["state"][0]["m"][0, 0, 0] = float("nan")
        variants.append(nan)
        negative = copy.deepcopy(original)
        negative["state"][0]["v"][0, 0, 0] = -1
        variants.append(negative)
        lr = copy.deepcopy(original)
        lr["param_groups"][0]["lr"] = 0.5
        variants.append(lr)
        for state in variants:
            self.torch.save(state, Path(job["resume_from"]) / "optimizer.pt")
            with self.subTest(state=state), self.assertRaises(ValueError):
                restore_optimizer(
                    self.factory(self.model), job["resume_from"], manifest
                )


if __name__ == "__main__":
    unittest.main()
