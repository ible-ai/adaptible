"""Finite grammar coverage and routing boundaries, using invented subjects."""

import unittest
from unittest.mock import AsyncMock

from adaptible.wrap.fact_scope import equivalent, parse_question
from adaptible.wrap.scope_router import ScopeRouter


def messages(text):
    return [{"role": "user", "content": text}]


def repair(question, **extra):
    return dict(
        interaction_idx=4,
        question=question,
        messages=messages(question),
        prompts=[messages(question)],
        generation_mode={"thinking": True},
        **extra,
    )


class FactScopeTest(unittest.TestCase):
    def assertEquivalent(self, left, right):
        a, b = parse_question(left), parse_question(right)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertTrue(equivalent(a, b), (a, b))

    def test_population_forms_and_productive_adjective_position(self):
        for place, adjective in (
            ("Arborland", "Arborlandian"),
            ("Veloro", "Veloran"),
            ("Tavira", "Tavirese"),
            ("North Arborland", "North Arborlandian"),
        ):
            original = (
                f"What is the largest city in {place}? Reply with only the city name."
            )
            for wording in (
                f"Which {adjective} city is the largest?",
                f"Which {adjective} city has the greatest population?",
                f"Which {adjective} city has the most residents?",
                f"Which city in {place} is largest by population?",
                f"Which city in {place} has the most inhabitants?",
                f"Which city in {place} has the greatest population?",
                f"Name {place}'s biggest city.",
                f"What is the most populous city in {place}?",
                f"Which city is biggest in {place}?",
            ):
                with self.subTest(wording=wording):
                    self.assertEquivalent(original, wording)

    def test_capital_and_currency_forms_preserve_property(self):
        for qualifier in (
            "",
            "legislative ",
            "administrative ",
            "judicial ",
            "official ",
            "constitutional ",
        ):
            self.assertEquivalent(
                f"What is the {qualifier}capital of Veloria?",
                f"Which city serves as the {qualifier}capital of Veloria?",
            )
            self.assertEquivalent(
                f"What is the {qualifier}capital of Veloria?",
                f"Which city serves as Veloria's {qualifier}capital?",
            )
        for wording in (
            "Which currency does Veloria use?",
            "What currency is used in Veloria?",
            "Name Veloria's currency.",
            "Identify the currency used in Veloria.",
        ):
            self.assertEquivalent("What is the currency of Veloria?", wording)
        self.assertFalse(
            equivalent(
                parse_question("What is the legislative capital of Veloria?"),
                parse_question("What is the administrative capital of Veloria?"),
            )
        )
        self.assertFalse(
            equivalent(
                parse_question("What is the capital of Veloria?"),
                parse_question("What is the currency of Veloria?"),
            )
        )

    def test_composed_relative_clauses_work_across_three_properties(self):
        for original, variants in (
            (
                "What is the legislative capital of Sablemere?",
                (
                    "Identify the city that serves as the legislative capital of Sablemere.",
                    "Name the city which is designated as Sablemere's legislative capital.",
                    "Which Sablemerian city is designated as the legislative capital?",
                    "Identify the Sablemerian city that serves as the legislative capital.",
                ),
            ),
            (
                "What is the largest city in Sablemere?",
                (
                    "Name the city that is most populous in Sablemere.",
                    "Identify the city which has the most residents in Sablemere.",
                    "Name the Sablemerian city that has the greatest population.",
                ),
            ),
            (
                "What is the currency of Merivale?",
                (
                    "Identify the currency that is used in Merivale.",
                    "What is the currency which is used in Merivale?",
                ),
            ),
        ):
            for variant in variants:
                with self.subTest(variant=variant):
                    self.assertEquivalent(original, variant)

    def test_relative_and_designated_forms_do_not_erase_roles_or_modifiers(self):
        original = parse_question("What is the legislative capital of Sablemere?")
        for text in (
            "Identify the city that was the legislative capital of Sablemere.",
            "Which Sablemerian city will be designated as the legislative capital?",
            "Which Sablemerian city is not designated as the legislative capital?",
            "Which Sablemerian city is designated as the legislative capital in 1900?",
            "Which Sablemerian city is designated as the administrative capital?",
            "Which Sablemerian city is designated as the seat of government?",
            "Which country is designated as Sablemere's legislative capital?",
            "Name the country that has Sablemere as its legislative capital.",
            "Identify the city that is the legislative capital of Republic of Sablemere.",
            "Identify the city that serves as the legislative capital of Sablemere and name its mayor.",
        ):
            with self.subTest(text=text):
                self.assertFalse(equivalent(original, parse_question(text)))

    def test_nominal_names_are_never_merged_by_adjective_spelling(self):
        for left, right in (
            ("Arborland", "Arborlandian"),
            ("Veloro", "Veloran"),
            ("North Arborland", "South Arborland"),
        ):
            a = parse_question(f"What is the largest city in {left}?")
            b = parse_question(f"What is the largest city in {right}?")
            self.assertFalse(equivalent(a, b))
        self.assertFalse(
            equivalent(
                parse_question("What is the largest city in Aro?"),
                parse_question("Which Aroian city is largest?"),
            )
        )

    def test_unconsumed_modifiers_multifact_negation_and_prose_are_not_parsed(self):
        for text in (
            "What is the largest city in Arborland by area?",
            "What is the largest city in Arborland in 1990?",
            "What is not the largest city in Arborland?",
            "What is the second largest city in Arborland?",
            "Which former Arborlandian city is largest?",
            "Which not-Arborlandian city is largest?",
            "What is the capital of former-Veloria?",
            "What is the capital of non-Veloria?",
            "What is the largest city and capital of Arborland?",
            "What is the currency of Arborland and Veloria?",
            "What is the currency of Arborland excluding imports?",
            "What is the capital of Arborland? Explain why it changed.",
            "Tell me everything about Arborland.",
            "Which country contains Arborland as its largest city?",
            "Arborland has a city. Which is largest?",
            "Which city in Arborland has more inhabitants?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_question(text))


class FactScopeRouterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = type("Runtime", (), {})()
        self.runtime.complete = AsyncMock(
            side_effect=AssertionError("Grammar must not ask model")
        )
        self.router = ScopeRouter(self.runtime)
        self.mode = {"thinking": True}

    async def test_fast_route_is_latest_cumulative_scope_without_classifier(self):
        original = (
            "What is the largest city in Arborland? Reply only with the city name."
        )
        entry = repair(original, expected="DO_NOT_SEND", note="DO_NOT_SEND")
        for q in (
            "Which Arborlandian city is largest?",
            "Which city in Arborland is largest by population?",
        ):
            result = await self.router.choose(messages(q), [entry], self.mode)
            self.assertEqual(
                result,
                dict(scope=4, reason="scope_match", matcher="question_grammar_v1"),
            )
        self.runtime.complete.assert_not_awaited()

    async def test_mode_and_exact_context_still_gate_grammar(self):
        entry = repair("What is the currency of Veloria?")
        prefix = [{"role": "system", "content": "Use plain answers."}]
        entry["messages"] = prefix + entry["messages"]
        query = messages("Which currency does Veloria use?")
        for context, mode, reason in (
            ([], self.mode, "unsupported_context"),
            (prefix, {"thinking": False}, "mode_mismatch"),
        ):
            result = await self.router.choose(context + query, [entry], mode)
            self.assertEqual(result["reason"], reason)
            self.assertIsNone(result["scope"])
        self.assertEqual(
            (await self.router.choose(prefix + query, [entry], self.mode))["scope"], 4
        )
        self.runtime.complete.assert_not_awaited()

    async def test_conflicting_validated_variant_cannot_expand_authoritative_original(
        self,
    ):
        entry = repair("What is the legislative capital of Veloria?")
        entry["prompts"].extend(
            [
                messages("What is the currency of Veloria?"),
                messages("What is the administrative capital of Veloria?"),
                messages("What is the legislative capital of Beloria?"),
                messages("Which city was formerly the legislative capital of Veloria?"),
            ]
        )
        for q in (
            "What is the currency of Veloria?",
            "What is the administrative capital of Veloria?",
            "What is the legislative capital of Beloria?",
            "Which city was formerly the legislative capital of Veloria?",
        ):
            self.assertIsNone(
                (await self.router.choose(messages(q), [entry], self.mode))["scope"]
            )
        self.runtime.complete.assert_not_awaited()

    async def test_unparsed_inverse_cannot_override_directed_relation(self):
        entry = repair("What is the capital of Veloria?")
        q = "What country has Veloria as its capital?"
        self.assertIsNone(
            (await self.router.choose(messages(q), [entry], self.mode))["scope"]
        )
        self.runtime.complete.assert_not_awaited()

    async def test_nominal_near_collision_does_not_reach_classifier(self):
        entry = repair("What is the largest city in Arborland?")
        q = "What is the largest city in Arborlandian?"
        self.assertIsNone(
            (await self.router.choose(messages(q), [entry], self.mode))["scope"]
        )
        self.runtime.complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
