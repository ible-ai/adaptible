"""Model-free checks that failed repairs cannot leak temporary adapter weights."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from adaptible import InteractionHistory, StatefulLLM
from adaptible import llm


class RepairRollbackTest(unittest.TestCase):
    def setUp(self):
        self.llm = StatefulLLM.__new__(StatefulLLM)
        self.llm._lock = threading.RLock()
        self.llm._base_adapter = "frozen"
        self.llm._model_path = None
        self.llm._tokenizer = object()
        self.llm._loss_target = 0.15
        self.llm._max_train_steps = 4
        self.weights = "previously accepted"
        self.llm._snapshot = lambda: self.weights
        self.llm._restore = self.restore
        self.llm.extract = mock.Mock(return_value="Canberra")
        self.llm._sample = mock.Mock(return_value="reasoning</think>Canberra.")
        self.llm.judge = mock.Mock(return_value=True)
        self.llm.rephrase = mock.Mock(return_value=["Which city?"])
        self.llm._score_prompts = mock.Mock(
            side_effect=[(0, "✗✗"), (1, "✓✗"), (2, "✓✓")]
        )
        self.llm._controls_answer = mock.Mock(return_value=True)
        self.llm.train_on_examples = mock.Mock(side_effect=self.train)
        self.interaction = InteractionHistory(idx=0, user_input="Capital of Australia?")
        for name in ("make_revision_training_example", "collate_training_examples"):
            patcher = mock.patch.object(llm, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def restore(self, weights):
        self.weights = weights

    def train(self, *args, **kwargs):
        self.weights += " + candidate"
        return SimpleNamespace(steps=1, final_loss=0.1)

    def repair(self):
        return self.llm.repair(self.interaction, "The capital is Canberra.")

    def test_sampling_failure_restores_previously_accepted_weights(self):
        self.llm._sample.side_effect = RuntimeError("sampling failed")
        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.repair()
        self.assertEqual(self.weights, "previously accepted")

    def test_training_failure_discards_partial_update(self):
        def fail(*args, **kwargs):
            self.weights = "partial update"
            raise RuntimeError("training failed")

        self.llm.train_on_examples.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.repair()
        self.assertEqual(self.weights, "previously accepted")

    def test_control_failure_discards_unvalidated_update(self):
        self.llm._controls_answer.side_effect = RuntimeError("control failed")
        with self.assertRaisesRegex(RuntimeError, "control failed"):
            self.repair()
        self.assertEqual(self.weights, "previously accepted")

    def test_second_candidate_failure_preserves_first_accepted_update(self):
        self.llm._score_prompts.side_effect = [
            (0, "✗✗"),
            (1, "✓✗"),
            RuntimeError("scoring failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "scoring failed"):
            self.repair()
        self.assertEqual(self.weights, "previously accepted + candidate")

    def test_success_retains_accepted_updates(self):
        self.assertTrue(self.repair())
        self.assertEqual(self.weights, "previously accepted + candidate + candidate")

    def test_rejection_restores_previous_weights(self):
        self.llm._controls_answer.return_value = False
        self.assertFalse(self.repair())
        self.assertEqual(self.weights, "previously accepted")

    def test_judge_failure_restores_weights_used_to_generate_answers(self):
        self.llm.generate_response = mock.Mock(
            return_value="reasoning</think>Canberra."
        )
        self.llm.judge.side_effect = RuntimeError("judge failed")
        with self.assertRaisesRegex(RuntimeError, "judge failed"):
            StatefulLLM._score_prompts(self.llm, "Capital?", ["Capital?"], "Canberra")
        self.assertEqual(self.weights, "previously accepted")
