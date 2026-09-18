"""Behavioral rejection may resume thinking training, within one total budget."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from adaptible._src.wrap.repair import Controller
from adaptible._src.wrap.store import Store
from adaptible._src.wrap.thinking import completion_details
from adaptible.tests.wrap_test import FakeRuntime


class RetryTrainer:
    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []
        self.totals = {}
        self.fail_resume = False

    async def train(self, blob, messages, target, directory, previous=None, **options):
        assert self.runtime.suspended
        resume = options.get("resume_from")
        if resume is not None and self.fail_resume:
            raise RuntimeError("resume failed")
        total = options.get("max_total_steps", 1)
        old_total = self.totals[str(resume)] if resume is not None else 0
        self.calls.append(dict(directory=directory, previous=previous, **options))
        directory.mkdir(parents=True)
        (directory / "resume.json").write_text("{}")
        self.totals[str(directory)] = total
        return dict(steps=total - old_total, total_steps=total)


class ThinkingRetryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime()
        self.store = Store(Path(self.temp.name), "retry-test")
        self.trainer = RetryTrainer(self.runtime)
        self.controller = Controller(self.runtime, self.store, trainer=self.trainer)
        self.messages = [dict(role="user", content="What is the capital of Australia?")]
        self.controller.find_reference = AsyncMock(
            return_value=("Canberra is the capital of Australia.", "Canberra", "")
        )
        self.controller.prompts = AsyncMock(return_value=[self.messages])
        self.controller.validate_prompts = AsyncMock(return_value=[self.messages])

        async def judge(q, response, expected, **kwargs):
            return response.rstrip(".") == expected.rstrip(".")

        self.controller.judge = judge
        self.learn_at = 2
        self.break_control = False

        async def complete(messages, handle=None, **options):
            assert not self.runtime.suspended
            text = messages[-1]["content"]
            steps = self.trainer.totals.get(str(handle), 0)
            if "12 times" in text:
                response = "wrong" if handle and self.break_control else "144"
            elif "France" in text:
                response = "Paris"
            elif "days" in text:
                response = "7"
            else:
                response = "Canberra" if steps >= self.learn_at else "Sydney"
            if "details" in options:
                options["details"].update(
                    completion_details(
                        dict(content=response, reasoning="Compare facts."), "stop"
                    )
                )
            return response

        self.runtime.complete = complete
        self.draft = patch(
            "adaptible._src.wrap.repair.generate_thinking_draft",
            AsyncMock(
                return_value=dict(
                    target="Canberra",
                    reasoning_prefix="<think>Compare facts.</think>\n",
                )
            ),
        )
        self.draft.start()

    async def asyncTearDown(self):
        self.draft.stop()
        await self.controller.close()
        self.temp.cleanup()

    async def repair(self, thinking=True):
        idx = self.store.record(
            self.messages, "Sydney", generation_mode=dict(thinking=thinking)
        )
        return await self.controller.repair(
            next(r for r in self.store.rows() if r["id"] == idx)
        )

    async def test_rejected_checkpoint_resumes_then_first_behavioral_pass_stops(self):
        parent = Path(self.temp.name).resolve() / "accepted-parent"
        parent.mkdir()
        self.store.set("accepted", dict(directory=str(parent)))
        self.runtime.active = str(parent)
        status, _ = await self.repair()
        self.assertEqual(status, "kept")
        first, second = self.trainer.calls
        self.assertNotIn("resume_from", first)
        self.assertEqual(second["resume_from"], first["directory"])
        self.assertEqual(second["max_total_steps"], 2)
        self.assertEqual(second["previous"], first["previous"])
        self.assertEqual(first["previous"], str(parent / "adapter"))
        self.assertEqual(
            self.store.get("accepted")["directory"], str(second["directory"].resolve())
        )
        self.assertEqual(self.runtime.active, str(second["directory"]))
        self.assertEqual(len(self.store.repairs()), 1)
        decisions = [
            json.loads((c["directory"] / "validation.json").read_text())["decision"]
            for c in self.trainer.calls
        ]
        self.assertEqual(decisions, ["rejected", "kept"])

    async def test_never_learning_stops_at_64_and_preserves_rejections(self):
        self.learn_at = 100
        status, _ = await self.repair()
        self.assertEqual(status, "rejected")
        self.assertEqual(list(self.trainer.totals.values()), [1, 2, 4, 8, 16, 32, 64])
        self.assertIsNone(self.store.get("accepted"))
        self.assertIsNone(self.runtime.active)
        self.assertTrue(
            all(
                (c["directory"] / "validation.json").exists()
                for c in self.trainer.calls
            )
        )

    async def test_the_first_candidate_leaves_room_to_continue(self):
        """The opening rung must be bounded, not the whole allowance.

        Without an explicit first bound the worker falls back to the full 64
        updates, ``total_steps`` equals the cap, and the continuation path can
        never run -- so a thinking repair gets exactly one attempt. A fake
        trainer that defaults the bound to 1 hides this; the real one defaults
        it to 64.
        """
        self.learn_at = 100
        await self.repair()
        budgets = [call.get("max_total_steps") for call in self.trainer.calls]
        self.assertEqual(budgets[0], 1, "the first candidate must be bounded")
        self.assertTrue(
            all(isinstance(budget, int) for budget in budgets),
            "every candidate needs an explicit bound, first included",
        )
        self.assertEqual(budgets, [1, 2, 4, 8, 16, 32, 64])

    async def test_control_regression_does_not_trigger_more_training(self):
        self.learn_at = 1
        self.break_control = True
        status, _ = await self.repair()
        self.assertEqual(status, "rejected")
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertIsNone(self.runtime.active)

    async def test_resume_failure_restores_current_serving_state(self):
        parent = str(Path(self.temp.name).resolve() / "accepted-parent")
        self.store.set("accepted", dict(directory=parent))
        self.runtime.active = parent
        self.trainer.fail_resume = True
        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            await self.repair()
        self.assertEqual(self.runtime.active, parent)
        self.assertFalse(self.runtime.suspended)
        self.assertEqual(self.store.get("accepted"), dict(directory=parent))

    async def test_nonthinking_rejection_does_not_resume(self):
        status, _ = await self.repair(thinking=False)
        self.assertEqual(status, "rejected")
        self.assertEqual(len(self.trainer.calls), 1)


if __name__ == "__main__":
    unittest.main()
