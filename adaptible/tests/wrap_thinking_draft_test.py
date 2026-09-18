"""Grounded reasoning targets require real complete drafts and independent judging."""

import copy
import unittest
from unittest.mock import AsyncMock

from adaptible._src.wrap.thinking import completion_details
from adaptible._src.wrap.thinking_draft import generate_thinking_draft


class ThinkingDraftTest(unittest.IsolatedAsyncioTestCase):
    async def draft(self, message, stop="stop", *, accepted=True, messages=None):
        runtime = AsyncMock()
        data = completion_details(message, stop)

        async def complete(sent, **options):
            options["details"].update(data)
            return data["content"]

        runtime.complete.side_effect = complete
        judge = AsyncMock(return_value=accepted)
        source_messages = messages or [
            dict(role="user", content="Which station opened first?")
        ]
        before = copy.deepcopy(source_messages)
        self.audit = {}
        result = await generate_thinking_draft(
            runtime,
            judge,
            source_messages,
            "The archival record says Arbor station opened in 1901, before Birch in 1910.",
            "JUDGE_ONLY_EXPECTED_SENTINEL",
            audit=self.audit,
        )
        self.assertEqual(source_messages, before)
        return result, runtime, judge

    async def test_complete_grounded_draft_uses_same_base_without_expected_leak(self):
        reasoning = "The archive dates Arbor earlier than Birch."
        result, runtime, judge = await self.draft(
            dict(content="Arbor.", reasoning=reasoning)
        )
        self.assertEqual(result["target"], "Arbor.")
        self.assertIn(reasoning, result["reasoning_prefix"])
        self.assertIn("</think>", result["reasoning_prefix"])
        self.assertEqual(self.audit["verdict"], "passed")
        self.assertEqual(self.audit["generation"]["reasoning"], reasoning)
        self.assertIn("Arbor station", self.audit["evidence"])
        runtime.complete.assert_awaited_once()
        args, options = runtime.complete.call_args
        self.assertTrue(options["frozen"])
        self.assertTrue(options["thinking"])
        self.assertNotIn("response_format", options)
        prompt = "\n".join(m["content"] for m in args[0])
        self.assertIn("Which station opened first?", prompt)
        self.assertIn("Arbor station opened in 1901", prompt)
        self.assertIn("final answer must contain only the", prompt)
        self.assertIn("Keep all explanation in your thinking", prompt)
        self.assertNotIn("JUDGE_ONLY_EXPECTED_SENTINEL", prompt)
        judge.assert_awaited_once_with(
            "Which station opened first?",
            "Arbor.",
            "JUDGE_ONLY_EXPECTED_SENTINEL",
            note="The archival record says Arbor station opened in 1901, before Birch in 1910.",
        )

    async def test_unsupported_final_answer_rejected_even_with_complete_reasoning(self):
        result, runtime, judge = await self.draft(
            dict(content="Birch.", reasoning="I favor the later date."), accepted=False
        )
        self.assertIsNone(result)
        self.assertEqual(self.audit["verdict"], "final_answer_not_grounded")
        runtime.complete.assert_awaited_once()
        judge.assert_awaited_once()

    async def test_truncated_rationale_never_becomes_a_training_target(self):
        for message in (
            dict(content="", reasoning="I am comparing the dates"),
            dict(content="Arbor.", reasoning="Arbor is earlier."),
        ):
            with self.subTest(message=message):
                result, runtime, judge = await self.draft(message, "length")
                self.assertIsNone(result)
                self.assertEqual(self.audit["verdict"], "truncated")
                self.assertEqual(self.audit["generation"]["finish_reason"], "length")
                runtime.complete.assert_awaited_once()
                judge.assert_not_awaited()

    async def test_no_empty_or_unclosed_thinking_fallback(self):
        for message in (
            dict(content="Arbor."),
            dict(content="<think>\n\n</think>\nArbor."),
            dict(content="<think>Arbor opened earlier"),
        ):
            with self.subTest(message=message):
                result, runtime, judge = await self.draft(message)
                self.assertIsNone(result)
                runtime.complete.assert_awaited_once()
                judge.assert_not_awaited()

    async def test_original_system_and_conversation_are_preserved_for_sampling(self):
        messages = [
            dict(role="system", content="Use English."),
            dict(role="user", content="We are discussing railway history."),
            dict(role="assistant", content="Understood."),
            dict(role="user", content="Which station opened first?"),
        ]
        _, runtime, _ = await self.draft(
            dict(content="Arbor.", reasoning="The earlier recorded year is 1901."),
            messages=messages,
        )
        sent = runtime.complete.call_args.args[0]
        self.assertEqual(len(sent), len(messages))
        self.assertTrue(sent[0]["content"].startswith("Use English."))
        self.assertEqual(sent[1:3], messages[1:3])

    async def test_missing_question_or_evidence_rejected_before_generation(self):
        runtime, judge = AsyncMock(), AsyncMock()
        for messages, note in (
            ([], "source"),
            ([dict(role="assistant", content="x")], "source"),
            ([dict(role="user", content="question")], ""),
        ):
            with self.subTest(messages=messages, note=note):
                with self.assertRaises(ValueError):
                    await generate_thinking_draft(
                        runtime, judge, messages, note, "answer"
                    )
        runtime.complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
