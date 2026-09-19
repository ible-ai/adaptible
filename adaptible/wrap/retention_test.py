"""Durable repair retention and per-prompt forgetting checks, without a model."""

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from adaptible.wrap.retention import evaluate_retention, retention_regressions
from adaptible.wrap.store import Store


def repair_record(idx=1):
    messages = [dict(role="user", content="What is Australia's capital?")]
    return dict(
        interaction_idx=idx,
        question=messages[0]["content"],
        messages=messages,
        expected="Canberra",
        note="Australia's capital is Canberra.",
        prompts=[messages],
        heldout_prompts=[[dict(role="user", content="Name the Australian capital.")]],
        evidence=dict(kind="web", sources=[dict(url="https://example.org/capital")]),
    )


class RepairLedgerTest(unittest.TestCase):
    def test_serving_template_identity_survives_restart_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "model")
            store.bind_serving_template("original")
            store.close()
            store = Store(Path(directory), "model")
            try:
                store.bind_serving_template("original")
                with self.assertRaisesRegex(
                    ValueError, "template or system prompt changed"
                ):
                    store.bind_serving_template("changed")
                self.assertEqual(store.get("serving_template_digest"), "original")
            finally:
                store.close()

    def test_unverified_existing_adapter_is_not_rebound_to_a_new_template(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "model")
            try:
                accepted = dict(directory="existing-adapter")
                store.set("accepted", accepted)
                with self.assertRaisesRegex(ValueError, "no serving-template identity"):
                    store.bind_serving_template("new")
                self.assertEqual(store.get("accepted"), accepted)
                self.assertIsNone(store.get("serving_template_digest"))
            finally:
                store.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name), "retention-test")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_acceptance_survives_crash_before_outcome_and_later_feedback_edit(self):
        record = repair_record()
        idx = self.store.record(record["messages"], "Sydney")
        self.assertEqual(idx, record["interaction_idx"])
        self.store.feedback(idx, True)
        self.store.outcome(idx, "reviewing", "")
        accepted = dict(directory="/already-written/adapter")
        self.store.accept_repair(accepted, record)
        # Simulate shutdown before Controller can write the 'kept' outcome.
        self.store.close()
        self.store = Store(Path(self.temp.name), "retention-test")
        self.assertEqual(self.store.get("accepted"), accepted)
        self.assertEqual(self.store.repairs(), [record])
        self.assertEqual(self.store.pending()[0]["id"], idx)
        self.store.feedback(idx, False)
        self.assertEqual(self.store.rows()[0]["status"], "new")
        self.assertEqual(self.store.repairs(), [record])

    def test_ledger_failure_rolls_back_adapter_pointer(self):
        original = dict(directory="/old/adapter")
        self.store.set("accepted", original)
        self.store.db.executescript(
            """CREATE TRIGGER fail_repair BEFORE INSERT ON accepted_repairs
            BEGIN SELECT RAISE(ABORT, 'simulated failure'); END;"""
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.accept_repair(dict(directory="/new/adapter"), repair_record())
        self.assertEqual(self.store.get("accepted"), original)
        self.assertEqual(self.store.repairs(), [])

    def test_subsequent_acceptance_keeps_other_repairs_and_replaces_same_item(self):
        first, second = repair_record(), repair_record(2)
        self.store.accept_repair(dict(directory="/one"), first)
        self.store.accept_repair(dict(directory="/two"), second)
        updated = {**first, "note": "Updated source evidence."}
        self.store.accept_repair(dict(directory="/three"), updated)
        self.assertEqual(self.store.repairs(), [updated, second])
        self.assertEqual(self.store.get("accepted"), dict(directory="/three"))


class RetentionGuardTest(unittest.IsolatedAsyncioTestCase):
    async def test_checks_all_recorded_and_heldout_prompts_without_reference_input(
        self,
    ):
        record = repair_record()
        runtime = AsyncMock()
        runtime.complete.side_effect = ["Canberra", "Canberra"]
        judge = AsyncMock(return_value=True)
        results = await evaluate_retention(
            [record], runtime, judge, handle="candidate-adapter"
        )
        self.assertEqual([item["kind"] for item in results], ["repair", "heldout"])
        self.assertTrue(all(item["passed"] for item in results))
        self.assertEqual(
            [call.args[0] for call in runtime.complete.await_args_list],
            record["prompts"] + record["heldout_prompts"],
        )
        for call in runtime.complete.await_args_list:
            self.assertEqual(call.kwargs, {"handle": "candidate-adapter"})
            self.assertNotIn("Canberra", str(call.args[0]))

    async def test_equal_total_cannot_hide_loss_of_previously_passing_prompt(self):
        runtime = AsyncMock()
        runtime.complete.side_effect = ["Canberra", "Sydney", "Sydney", "Canberra"]
        judge = AsyncMock(
            side_effect=lambda q, response, expected, **kwargs: response == expected
        )
        before = await evaluate_retention([repair_record()], runtime, judge)
        after = await evaluate_retention(
            [repair_record()], runtime, judge, handle="candidate"
        )
        self.assertEqual(sum(r["passed"] for r in before), 1)
        self.assertEqual(sum(r["passed"] for r in after), 1)
        self.assertEqual(retention_regressions(before, after), [before[0]])

    async def test_missing_or_changed_checks_fail_closed_but_old_failures_do_not_veto(
        self,
    ):
        runtime = AsyncMock()
        runtime.complete.side_effect = ["Canberra", "Sydney"]
        judge = AsyncMock(
            side_effect=lambda q, response, expected, **kwargs: response == expected
        )
        before = await evaluate_retention([repair_record()], runtime, judge)
        self.assertEqual(retention_regressions(before, before[:1]), [])
        self.assertEqual(retention_regressions(before, before[1:]), [before[0]])
        changed = copy.deepcopy(before)
        changed[0]["messages"] = [dict(role="user", content="Different question?")]
        self.assertEqual(retention_regressions(before, changed), [before[0]])

    async def test_record_without_generated_paraphrases_checks_original(self):
        record = repair_record()
        record.pop("prompts")
        record.pop("heldout_prompts")
        runtime = AsyncMock()
        runtime.complete.return_value = "Canberra"
        judge = AsyncMock(return_value=True)
        results = await evaluate_retention([record], runtime, judge)
        runtime.complete.assert_awaited_once_with(record["messages"], handle=None)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
