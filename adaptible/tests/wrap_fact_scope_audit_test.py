"""Independent conservative scope audit with fictional, nonbenchmark entities."""

import unittest
from unittest.mock import AsyncMock

from adaptible._src.wrap.fact_scope import equivalent, parse_question
from adaptible._src.wrap.scope_router import ScopeRouter


def messages(question):
    return [dict(role="user", content=question)]


def repair(question, *, idx=1, variants=()):
    return dict(
        interaction_idx=idx,
        question=question,
        messages=messages(question),
        prompts=[messages(question), *map(messages, variants)],
        generation_mode={"thinking": True},
    )


class FactGrammarAuditTest(unittest.TestCase):
    def test_complete_same_property_forms_match(self):
        for left, right in (
            (
                "What is the largest city in Velora?",
                "Name Velora's largest city by population.",
            ),
            (
                "What is the largest city in Velora?",
                "Which city in Velora has the most inhabitants?",
            ),
            (
                "What is the capital of North Aster?",
                "Identify North Aster's capital city.",
            ),
            (
                "What is the legislative capital of Velora?",
                "Which city serves as the legislative capital of Velora?",
            ),
            ("What currency does Velora use?", "Name the currency used in Velora."),
        ):
            with self.subTest(left=left, right=right):
                self.assertTrue(equivalent(parse_question(left), parse_question(right)))

    def test_area_date_negation_and_additional_facts_do_not_match(self):
        original = parse_question("What is the largest city in Velora?")
        for question in (
            "What is the largest city by area in Velora?",
            "What is the largest city in Velora in 1910?",
            "What was the largest city in Velora?",
            "What is not the largest city in Velora?",
            "What is the second largest city in Velora?",
            "What is the largest city and the currency in Velora?",
            "What is the largest city in Velora? Also name its airport.",
            "What is the largest city in Velora and why?",
            "What is the largest city in Velora without suburbs?",
            "What is the largest city in Velora's largest province?",
        ):
            with self.subTest(question=question):
                self.assertFalse(equivalent(original, parse_question(question)))

    def test_capital_qualifiers_and_other_relationships_stay_distinct(self):
        for left, right in (
            (
                "What is the legislative capital of Velora?",
                "What is the judicial capital of Velora?",
            ),
            (
                "What is the official capital of Velora?",
                "What is the capital of Velora?",
            ),
            ("What is the capital of Velora?", "What is the currency of Velora?"),
            ("What is the largest city in Velora?", "What is the capital of Velora?"),
            (
                "What is the capital of North Aster?",
                "What is the capital of Aster North?",
            ),
        ):
            with self.subTest(left=left, right=right):
                self.assertFalse(
                    equivalent(parse_question(left), parse_question(right))
                )

    def test_nominal_near_collisions_never_become_adjectival_aliases(self):
        for first, second in (
            ("Aster", "Asteria"),
            ("Aster", "Asterian"),
            ("North Aster", "South Aster"),
            ("Velora", "Veloro"),
        ):
            with self.subTest(first=first, second=second):
                left = parse_question(f"What is the largest city in {first}?")
                right = parse_question(f"What is the largest city in {second}?")
                self.assertFalse(equivalent(left, right))


class FactRouterAuthorityAuditTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = type("Runtime", (), {})()
        self.runtime.complete = AsyncMock(return_value='{"same_fact":true}')
        self.router = ScopeRouter(self.runtime)

    async def route(self, question, repairs):
        return await self.router.choose(messages(question), repairs, {"thinking": True})

    async def test_unparsed_variants_cannot_override_original_property(self):
        original = "What is the capital of Velora?"
        for variant in (
            "Which city was formerly the capital of Velora?",
            "Name Velora's capital and its currency.",
            "What country has Velora as its capital?",
        ):
            with self.subTest(variant=variant):
                decision = await self.route(
                    variant, [repair(original, variants=[variant])]
                )
                self.assertIsNone(decision["scope"], decision)
        self.runtime.complete.assert_not_awaited()

    async def test_parsed_conflicting_variant_cannot_expand_original_scope(self):
        decision = await self.route(
            "What is the largest city in Velora?",
            [
                repair(
                    "What is the capital of Velora?",
                    variants=["What is the largest city in Velora?"],
                )
            ],
        )
        self.assertIsNone(decision["scope"], decision)
        self.runtime.complete.assert_not_awaited()

    async def test_reversed_roles_cannot_reach_overconfident_classifier(self):
        decision = await self.route(
            "What country has Velora as its capital?",
            [repair("What is the capital of Velora?")],
        )
        self.assertIsNone(decision["scope"], decision)
        self.runtime.complete.assert_not_awaited()

    async def test_ambiguous_adjective_between_two_nominal_scopes_abstains(self):
        decision = await self.route(
            "Which Asterian city is largest?",
            [
                repair("What is the largest city in Aster?", idx=1),
                repair("What is the largest city in Asteria?", idx=2),
            ],
        )
        self.assertIsNone(decision["scope"], decision)
        self.runtime.complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
