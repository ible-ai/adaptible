"""Actual tiny CPU LoRA learning under the wrapper's bounded loss budget.

No tokenizer, checkpoint, network or serving process is needed. The frozen
embedding and output weights stand in for a base model; only a rank-two update
can learn the two target transitions.
"""

from types import SimpleNamespace
import unittest


class WrapperTrainingBudgetTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        torch.manual_seed(8)

        class TinyLoraLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.base_head = torch.nn.Linear(4, 8, bias=False)
                self.embedding.requires_grad_(False)
                self.base_head.requires_grad_(False)
                self.lora_A = torch.nn.Parameter(torch.randn(4, 2) * 0.1)
                self.lora_B = torch.nn.Parameter(torch.zeros(2, 8))

            def forward(self, *, input_ids, labels):
                hidden = self.embedding(input_ids[:, :-1])
                logits = self.base_head(hidden) + hidden @ self.lora_A @ self.lora_B
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100
                )
                return SimpleNamespace(loss=loss)

        self.model = TinyLoraLM()
        self.ids = torch.tensor([[1, 2, 3, 4]])
        self.labels = torch.tensor([[-100, -100, 3, 4]])
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=0.2,
            weight_decay=0,
        )
        self.before = {
            name: value.detach().clone()
            for name, value in self.model.named_parameters()
        }

    def fit(self, **kwargs):
        from adaptible._src.wrap.training_budget import fit_masked_target

        return fit_masked_target(
            self.model, self.ids, self.labels, self.optimizer, **kwargs
        )

    def test_real_low_rank_learning_stops_at_target_and_preserves_base(self):
        stats = self.fit(max_steps=64, target_loss=0.15)
        self.assertGreater(stats["steps"], 0)
        self.assertLess(stats["steps"], 64)
        self.assertEqual(stats["stop_reason"], "target_loss")
        self.assertGreater(stats["initial_loss"], 0.15)
        self.assertLessEqual(stats["final_loss"], 0.15)
        self.assertAlmostEqual(stats["loss"], stats["final_loss"])
        for name, value in self.model.named_parameters():
            if not value.requires_grad:
                self.assertTrue(self.torch.equal(value, self.before[name]), name)
        self.assertFalse(self.torch.equal(self.model.lora_B, self.before["lora_B"]))

    def test_unreachable_target_uses_hard_cap_and_reports_post_update_loss(self):
        stats = self.fit(max_steps=2, target_loss=1e-12)
        self.assertEqual(stats["steps"], 2)
        self.assertEqual(stats["stop_reason"], "max_steps")
        final = self.model(input_ids=self.ids, labels=self.labels).loss.item()
        self.assertAlmostEqual(stats["final_loss"], final, places=6)
        self.assertLess(stats["final_loss"], stats["initial_loss"])
        self.assertGreaterEqual(stats["elapsed_seconds"], 0)

    def test_already_learned_target_takes_no_optimizer_step(self):
        initial = self.model(input_ids=self.ids, labels=self.labels).loss.item()
        stats = self.fit(max_steps=64, target_loss=initial + 0.01)
        self.assertEqual(stats["steps"], 0)
        self.assertEqual(stats["stop_reason"], "target_loss")
        for name, value in self.model.named_parameters():
            self.assertTrue(self.torch.equal(value, self.before[name]), name)

    def test_nonfinite_objective_does_not_modify_adapter(self):
        self.labels[:] = -100  # No target tokens: cross entropy is undefined.
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.fit(max_steps=64, target_loss=0.15)
        for name, value in self.model.named_parameters():
            self.assertTrue(self.torch.equal(value, self.before[name]), name)

    def test_padded_training_supplies_attention_mask_on_every_forward(self):
        original_forward = self.model.forward
        self.ids = self.torch.tensor([[1, 2, 3, 4, 0, 0]])
        self.labels = self.torch.tensor([[-100, -100, 3, 4, -100, -100]])
        mask = self.torch.tensor([[1, 1, 1, 1, 0, 0]])
        observed = []

        def forward(*, input_ids, labels, attention_mask):
            self.assertTrue(self.torch.equal(attention_mask, mask))
            self.assertTrue((labels[attention_mask == 0] == -100).all())
            observed.append(True)
            return original_forward(input_ids=input_ids, labels=labels)

        self.model.forward = forward
        stats = self.fit(max_steps=2, target_loss=1e-12, attention_mask=mask)
        self.assertEqual(stats["steps"], 2)
        self.assertEqual(len(observed), 3)  # Includes final post-update loss.
        self.assertLess(stats["final_loss"], stats["initial_loss"])


class ThinkingStoppingLossTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

        class IndependentTokenLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 4, 5))
                self.calls = 0

            def forward(self, *, input_ids, labels):
                self.calls += 1
                loss = torch.nn.functional.cross_entropy(
                    self.logits[:, :-1].reshape(-1, 5),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(loss=loss, logits=self.logits)

        self.model = IndependentTokenLM()
        self.ids = torch.tensor([[0, 1, 2, 3]])
        self.labels = torch.tensor([[-100, 1, 2, 3]])
        self.stop_labels = torch.tensor([[-100, -100, -100, 3]])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.2)

    def fit(self, **kwargs):
        from adaptible._src.wrap.training_budget import fit_masked_target

        return fit_masked_target(
            self.model,
            self.ids,
            self.labels,
            self.optimizer,
            stop_labels=self.stop_labels,
            **kwargs,
        )

    def test_rationale_gets_gradients_but_stopping_loss_only_measures_answer(self):
        with self.torch.no_grad():
            self.model.logits[0, 2, 3] = 1.0
        before = self.model.logits.detach().clone()
        stats = self.fit(max_steps=1, target_loss=1e-8)
        self.assertEqual(stats["steps"], 1)
        self.assertEqual(self.model.calls, 2)  # No extra answer-loss forward.
        self.assertFalse(self.torch.equal(before[0, 0], self.model.logits[0, 0]))
        self.assertFalse(self.torch.equal(before[0, 1], self.model.logits[0, 1]))
        self.assertFalse(self.torch.equal(before[0, 2], self.model.logits[0, 2]))
        self.assertTrue(self.torch.equal(before[0, 3], self.model.logits[0, 3]))
        expected = self.torch.nn.functional.cross_entropy(
            self.model.logits[:, 2], self.stop_labels[:, 3]
        ).item()
        self.assertAlmostEqual(stats["final_loss"], expected, places=6)
        self.assertNotAlmostEqual(stats["final_loss"], stats["final_training_loss"])
        self.assertEqual(stats["stopping_scope"], "final_answer")

    def test_easy_answer_still_trains_rationale_once_before_stopping(self):
        with self.torch.no_grad():
            self.model.logits[0, 2, 3] = 12.0
        before = self.model.logits.detach().clone()
        stats = self.fit(max_steps=4, target_loss=0.15)
        self.assertEqual(stats["steps"], 1)
        self.assertGreater(stats["initial_training_loss"], 0.15)
        self.assertLess(stats["initial_loss"], 0.15)
        self.assertFalse(self.torch.equal(before[0, 0], self.model.logits[0, 0]))
        self.assertFalse(self.torch.equal(before[0, 1], self.model.logits[0, 1]))
        self.assertLess(stats["final_training_loss"], stats["initial_training_loss"])
        self.assertEqual(stats["stop_reason"], "target_loss")

    def test_stopping_mask_cannot_target_different_or_unsupervised_tokens(self):
        for labels in (
            [[-100, -100, -100, 4]],
            [[0, -100, -100, 3]],
            [[-100, -100, -100, -100]],
        ):
            with self.subTest(labels=labels):
                self.stop_labels = self.torch.tensor(labels)
                with self.assertRaises(ValueError):
                    self.fit(max_steps=1)
        self.assertEqual(self.model.calls, 0)


if __name__ == "__main__":
    unittest.main()


class FlagshipStoppingCriterionTest(unittest.TestCase):
    """The experiment's stop rule is on the answer tokens.

    "loss over the whole target, stop rule on the answer tokens ... stop when
    answer-token loss < 0.15" (results/self-repair-cycles-2026-09-11-mlx).
    Its candidates.csv records candidates that took a single step at loss
    0.10, so an early stop is the experiment's behaviour. A previous change
    here replaced it with a whole-target stop, which was an infidelity
    introduced while trying to remove one.
    """

    def test_the_stop_rule_is_on_the_answer_tokens(self):
        import inspect

        from adaptible._src.wrap import train

        source = inspect.getsource(train)
        self.assertIn('stop_labels=inputs.get("stop_labels")', source)
        self.assertNotIn("flagship_rationale", source)
