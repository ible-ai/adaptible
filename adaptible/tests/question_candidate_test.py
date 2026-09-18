"""Disjoint synthetic synonym candidates and hard routing boundaries."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, AsyncMock

from adaptible._src.wrap.question_candidate import candidate_pair
from adaptible._src.wrap.scope_router import ScopeRouter


def messages(q):
    return [dict(role="user", content=q)]


def repair(q):
    return dict(
        interaction_idx=8,
        question=q,
        prompts=[messages(q)],
        messages=messages(q),
        generation_mode={"thinking": True},
        expected="SECRET_ANSWER",
        note="SECRET_REFERENCE",
    )


class QuestionCandidateTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        async def completed(*args, details, **kwargs):
            details.update(
                complete=True, framing_valid=True, finish_reason="stop", reasoning=""
            )
            return DEFAULT

        self.runtime = SimpleNamespace(
            complete=AsyncMock(return_value='{"same_fact":true}', side_effect=completed)
        )
        self.router = ScopeRouter(self.runtime)
        self.mode = {"thinking": True}

    async def route(self, a, b):
        return await self.router.choose(messages(b), [repair(a)], self.mode)

    async def test_new_synonyms_reach_classifier_without_reference_or_answer(self):
        pairs = (
            (
                "Who composed the music for Winter Ember?",
                "Who is the composer for Winter Ember?",
            ),
            (
                "Which company operates Juniper Beacon?",
                "Which organization runs Juniper Beacon?",
            ),
            (
                "How long does a Falcon Loop shuttle take?",
                "What is the duration of a Falcon Loop shuttle trip?",
            ),
            (
                "Where is the Quiet Lantern festival held?",
                "What is the location of the Quiet Lantern festival?",
            ),
            (
                "What material is the Meridian Capsule made of?",
                "What substance forms the Meridian Capsule?",
            ),
            (
                "How much does entry to Cloud Gallery cost?",
                "What is the admission fee at Cloud Gallery?",
            ),
        )
        for a, b in pairs:
            with self.subTest(a=a):
                self.assertTrue(candidate_pair(a, b))
                result = await self.route(a, b)
                self.assertEqual(result["scope"], 8)
                call = self.runtime.complete.call_args
                self.assertTrue(call.kwargs["frozen"])
                self.assertNotIn("SECRET", json.dumps(call.args))
                payload = json.loads(call.args[0][-1]["content"])
                self.assertEqual(payload, dict(question_a=a, question_b=b))
                self.assertEqual(result["classifier"]["input"], payload)
        self.assertEqual(self.runtime.complete.await_count, 6)

    async def test_structural_candidates_reach_completed_classifier(self):
        pairs = (
            ("Which bureau maintains the Silver Acorn Trust?", "The Silver Acorn Trust is maintained by which bureau?"),
            ("In which district is the Misty Sparrow Studio situated?", "Which district contains the Misty Sparrow Studio?"),
            ("What is the rental charge for the Bronze Kite Pass?", "How much does it cost to rent the Bronze Kite Pass?"),
            ("What is the Cloud Rune sculpture carved from?", "Which material is used to carve the Cloud Rune sculpture?"),
            ("What is the elapsed duration of a Blue Cormorant voyage?", "How long does a Blue Cormorant voyage last?"),
        )
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                result = await self.route(a, b)
                self.assertEqual(result["scope"], 8)
                self.assertEqual(result["classifier"]["verdict"], "same_fact")
                self.assertNotIn("SECRET", json.dumps(self.runtime.complete.call_args.args))
        self.assertEqual(self.runtime.complete.await_count, len(pairs))

    async def test_same_entity_different_property_is_still_a_model_decision(self):
        self.runtime.complete.return_value = '{"same_fact":false}'
        result = await self.route(
            "Who directs Winter Ember?", "Who finances Winter Ember?"
        )
        self.assertIsNone(result["scope"])
        self.assertEqual(result["classifier"]["verdict"], "different_fact")
        self.runtime.complete.assert_awaited_once()

    async def test_hard_subject_direction_time_and_clause_guards_skip_classifier(self):
        for a, b in (
            ("Who mentors Vera Vale?", "Whom does Vera Vale mentor?"),
            ("Who follows Vera Vale?", "Who is followed by Vera Vale?"),
            ("Who employs Vera Vale?", "Who is directly employed by Vera Vale?"),
            ("What is the total charge for the Bronze Kite Pass?", "How much does it cost to purchase the Bronze Kite Pass?"),
            ("Who mentors Vera Vale?", "Who does Vera Vale mentor?"),
            (
                "Who directed Winter Ember in 2015?",
                "Who directed Winter Ember in 2025?",
            ),
            ("Who directs Winter Ember?", "Who directs Winter Ember and Silver Cloud?"),
            ("Who directs Winter Ember?", "Who directs Silver Cloud?"),
            ("Who directs Winter Ember?", "Who directs that film?"),
            (
                "Who directs Winter Ember?",
                "Who directs Winter Ember? Also name the producer.",
            ),
            ("Who directs Winter Ember?", "Who directed Winter Ember, and when?"),
            ("Who mentors Vera Vale?", "Who mentors Vera Valean?"),
            (
                "How much is entry to Cloud Gallery?",
                "How much is entry to Cloud Gallery per month?",
            ),
            (
                "Who is the designer of Juniper Beacon?",
                "Who was the designer of Juniper Beacon?",
            ),
        ):
            with self.subTest(a=a, b=b):
                self.assertIsNone((await self.route(a, b))["scope"])
        self.runtime.complete.assert_not_awaited()

    async def test_passive_equivalent_patient_question_can_reach_classifier(self):
        a = "Whom does Vera Vale mentor?"
        b = "Who is mentored by Vera Vale?"
        self.assertEqual((await self.route(a, b))["scope"], 8)
        self.runtime.complete.assert_awaited_once()

    async def test_parsed_authority_and_mode_context_are_unchanged(self):
        a = "What is the capital of Elaria?"
        b = "What country has Elaria as its capital?"
        self.assertIsNone((await self.route(a, b))["scope"])
        a = "Who directs Winter Ember?"
        b = "Who is the director of Winter Ember?"
        self.assertIsNone(
            (await self.router.choose(messages(b), [repair(a)], {"thinking": False}))[
                "scope"
            ]
        )
        altered = [dict(role="system", content="Different context"), *messages(b)]
        self.assertIsNone(
            (await self.router.choose(altered, [repair(a)], self.mode))["scope"]
        )
        self.runtime.complete.assert_not_awaited()

    async def test_unknown_category_requires_classifier_verdict(self):
        self.runtime.complete.return_value = '{"same_fact":false}'
        a = "What is special about Juniper Beacon?"
        b = "What is notable about Juniper Beacon?"
        self.assertTrue(candidate_pair(a, b))
        result = await self.route(a, b)
        self.assertIsNone(result["scope"])
        self.assertEqual(result["classifier"]["verdict"], "different_fact")
        self.runtime.complete.assert_awaited_once()

    def test_missing_name_abstains(self):
        for a, b in (
            ("Who wrote the screenplay?", "Who is the screenwriter?"),
            ("Who wrote winter ember?", "Who composed winter ember?"),
        ):
            self.assertFalse(candidate_pair(a, b))


if __name__ == "__main__":
    unittest.main()
