"""Capital-source guards: no model or network is required."""

import unittest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from adaptible.wrap.repair import Controller
from adaptible.wrap.source_property_guard import capital_source_check

Q = "What is the capital of Veloria? Reply with only the name, without explanation."


class CapitalSourceGuardTest(unittest.TestCase):
    def test_current_forms_preserve_candidate_and_owner_roles(self):
        for text in (
            "Cedar is the national capital of Veloria.",
            "Veloria's political capital is Cedar.",
            "The capital of Veloria is Cedar, its political center.",
            "Veloria: capital Cedar, population 400000.",
            "Cedar, the capital of Veloria, was founded in 1650.",
            "Since 1965, Cedar has been the capital of Veloria.",
            "Cedar (pronounced [se-dar]) is the capital city of Veloria, an island nation.",
            "A government move cemented Cedar's status as the political capital of Veloria.",
        ):
            with self.subTest(text=text):
                self.assertTrue(capital_source_check(Q, text)["eligible"])
                self.assertTrue(capital_source_check(Q, text, "Cedar")["eligible"])
                self.assertFalse(capital_source_check(Q, text, "Veloria")["eligible"])

    def test_irrelevant_historical_negative_and_wrong_jurisdiction_reject(self):
        cases = (
            (
                "Cedar Council is one of seven administrative regions of Veloria.",
                "Cedar Council",
            ),
            (
                "It is the administrative capital of both Cedar Municipal Council and the entire Cedar Region.",
                "Cedar Municipal Council",
            ),
            ("Old Harbor was Veloria's first capital city.", "Old Harbor"),
            ("The name Old Harbor means the house of peace.", "Old Harbor"),
            (
                "The most populous island is Old Harbor, home to the city of the same name.",
                "Old Harbor",
            ),
            ("The capitol of Veloria used to be Old Harbor.", "Old Harbor"),
            ("Cedar is the capital of West Veloria Region.", "Cedar"),
            ("Cedar is the capital of Veloria Region.", "Cedar"),
            ("Cedar is not the capital of Veloria.", "Cedar"),
            ("Cedar may become the capital of Veloria.", "Cedar"),
            (
                "Old Harbor was formerly the capital of Veloria; today Cedar is the capital of Veloria.",
                "Old Harbor",
            ),
        )
        for text, name in cases:
            with self.subTest(text=text):
                self.assertFalse(capital_source_check(Q, text, name)["eligible"])

    def test_partial_names_cannot_satisfy_complete_value_or_owner_roles(self):
        for text, name in (
            ("North Cedar is the capital of Veloria.", "Cedar"),
            (
                "A move cemented North Cedar's status as the capital of Veloria.",
                "Cedar",
            ),
            ("New Veloria's capital is Cedar.", "Cedar"),
            ("Cedar is the capital of New Veloria.", "Cedar"),
        ):
            with self.subTest(text=text):
                self.assertFalse(capital_source_check(Q, text, name)["eligible"])

    def test_designation_events_abstain_instead_of_assuming_still_current(self):
        for text in (
            "Cedar was declared as the capital of Veloria in 1974.",
            "On October 7, 2006, Cedar was officially inaugurated as the capital of Veloria.",
        ):
            result = capital_source_check(Q, text, "Cedar")
            self.assertFalse(result["eligible"])
            self.assertEqual(
                result["reason"], "historical_designation_not_current_evidence"
            )

    def test_qualifiers_dated_questions_and_real_conflicts_are_preserved(self):
        q = "What is the legislative capital of Veloria?"
        self.assertTrue(
            capital_source_check(
                q, "Cedar is the legislative capital of Veloria.", "Cedar"
            )["eligible"]
        )
        self.assertFalse(
            capital_source_check(
                q, "Cedar is the administrative capital of Veloria.", "Cedar"
            )["eligible"]
        )
        for q in (
            "What was the capital of Veloria in 1900?",
            "What is the first capital of Veloria?",
        ):
            result = capital_source_check(
                q, "Cedar is the capital of Veloria.", "Cedar"
            )
            self.assertFalse(result["eligible"])
            self.assertEqual(result["reason"], "unsupported_capital_question")
        # Both actual current claims stay eligible; consensus must still reject
        # their disagreement, rather than this guard choosing a desired answer.
        for name in ("Cedar", "Birch"):
            self.assertTrue(
                capital_source_check(Q, f"{name} is the capital of Veloria.", name)[
                    "eligible"
                ]
            )

    def test_jurisdiction_prefixes_are_distinct_not_optional_aliases(self):
        for asked, offered in (
            ("Arden", "Republic of Arden"),
            ("Republic of Arden", "Democratic Republic of Arden"),
            ("Democratic Republic of Arden", "Republic of Arden"),
            ("Republic of Arden", "Fictional Republic of Arden"),
        ):
            with self.subTest(asked=asked, offered=offered):
                q = f"What is the capital of {asked}?"
                text = f"Linden is the capital of {offered}."
                self.assertFalse(capital_source_check(q, text, "Linden")["eligible"])
        q = "What is the capital of Republic of Arden?"
        text = "Linden is the capital of Republic of Arden."
        self.assertTrue(capital_source_check(q, text, "Linden")["eligible"])

    def test_government_seat_does_not_establish_capital_or_constitutional_capital(self):
        for q in (
            "What is the capital of Orvane?",
            "What is the constitutional capital of Orvane?",
        ):
            result = capital_source_check(
                q, "Linden is the seat of government of Orvane.", "Linden"
            )
            self.assertFalse(result["eligible"])
        q = "What is the constitutional capital of Orvane?"
        self.assertFalse(
            capital_source_check(
                q, "Linden is the administrative capital of Orvane.", "Linden"
            )["eligible"]
        )
        self.assertTrue(
            capital_source_check(
                q, "Linden is the constitutional capital of Orvane.", "Linden"
            )["eligible"]
        )

    def test_unrelated_question_family_keeps_existing_path(self):
        result = capital_source_check(
            "Who wrote Silver Lake?", "Mira wrote Silver Lake.", "Mira"
        )
        self.assertFalse(result["applicable"])
        self.assertTrue(result["eligible"])


class CapitalGroundingIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(complete=AsyncMock())
        self.controller = Controller(self.runtime, None)
        self.controller.supported = AsyncMock(return_value=True)

    async def test_irrelevant_sentences_never_reach_overconfident_model(self):
        for text in (
            "Cedar Council is an administrative region of Veloria.",
            "Old Harbor was Veloria's first capital city.",
            "The name Old Harbor means the house of peace.",
            "The most populous island is Old Harbor.",
        ):
            self.assertEqual(await self.controller.grounded_source(Q, text), {})
            self.assertFalse(
                self.controller.last_source_diagnostics["sentences"][0]["eligible"]
            )
        self.runtime.complete.assert_not_awaited()
        self.controller.supported.assert_not_awaited()

    async def test_selected_owner_or_partial_value_name_cannot_pass(self):
        for text, name in (
            ("Cedar is the capital of Veloria.", "Veloria"),
            ("North Cedar is the capital of Veloria.", "Cedar"),
        ):
            self.runtime.complete.return_value = json.dumps(
                dict(name=name, sentence_index=0)
            )
            self.assertEqual(await self.controller.grounded_source(Q, text), {})
            self.assertFalse(
                self.controller.last_source_diagnostics["selection"]["eligible"]
            )
        self.controller.supported.assert_not_awaited()

    async def test_current_claims_are_copied_and_true_conflict_is_preserved(self):
        sources = [
            dict(
                url=f"https://{site}.example/fact",
                text=f"{name} is the capital of Veloria.",
            )
            for site, name in (("first", "Cedar"), ("second", "Birch"))
        ]
        self.controller.reference_search = SimpleNamespace(
            search=AsyncMock(return_value=sources)
        )
        self.controller.store = SimpleNamespace(references=Mock())
        self.runtime.complete.side_effect = [
            json.dumps(dict(name=name, sentence_index=0)) for name in ("Cedar", "Birch")
        ]
        note, name, reason = await self.controller.find_reference(
            dict(id=1, note=""), Q
        )
        self.assertEqual((note, name), ("", ""))
        self.assertIn("conflicting answers", reason)
        self.assertEqual(self.controller.supported.await_count, 2)
        for call in self.runtime.complete.call_args_list:
            properties = call.kwargs["response_format"]["json_schema"]["schema"][
                "properties"
            ]
            self.assertNotIn("enum", properties["name"])


if __name__ == "__main__":
    unittest.main()


class ObservedSourceWordingTest(unittest.TestCase):
    """Sentences a live run actually retrieved and the guard used to reject.

    Every one states the current capital plainly. Losing them made 15 of 16
    reviews skip for want of evidence, so the repair loop could not run at all.
    """

    SRI_LANKA = (
        "What is the legislative capital of Sri Lanka? Reply with only the name."
    )
    TANZANIA = "What is the capital of Tanzania? Reply with only the name."
    BENIN = "What is the official capital of Benin? Reply with only the name."

    def assertEligible(self, question, sentence):
        result = capital_source_check(question, sentence)
        self.assertTrue(result["eligible"], f"{result['reason']}: {sentence}")

    def assertRejected(self, question, sentence, reason=None):
        result = capital_source_check(question, sentence)
        self.assertFalse(result["eligible"], sentence)
        if reason:
            self.assertEqual(result["reason"], reason)

    def test_a_currency_marker_before_the_qualifier_is_not_a_blocker(self):
        self.assertEligible(
            self.SRI_LANKA,
            "The current legislative capital of Sri Lanka is Sri Jayawardenepura "
            "Kotte and the judicial capital is Colombo.",
        )

    def test_coordinated_qualifiers_still_carry_the_requested_one(self):
        self.assertEligible(
            self.SRI_LANKA,
            "Sri Jayawardenepura Kotte is the administrative and legislative "
            "capital of Sri Lanka.",
        )
        self.assertEligible(
            self.TANZANIA,
            "Dodoma is the official political and administrative capital of Tanzania.",
        )

    def test_a_qualifier_the_question_did_not_ask_for_is_still_required(self):
        # Coordination must not let any qualifier stand in for another.
        self.assertRejected(
            self.SRI_LANKA,
            "Colombo is the judicial and commercial capital of Sri Lanka.",
        )

    def test_negation_about_another_property_does_not_void_the_claim(self):
        self.assertEligible(
            self.BENIN,
            "Despite not being the largest city in the country (Cotonou holds "
            "that title), Porto-Novo is the official capital of Benin.",
        )

    def test_negation_of_the_capital_claim_itself_still_rejects(self):
        self.assertRejected(
            self.TANZANIA,
            "Dar es Salaam is not the capital of Tanzania.",
            "negated_claim",
        )

    def test_historical_and_designation_wording_is_still_refused(self):
        self.assertRejected(
            self.TANZANIA, "Dar es Salaam was the former capital of Tanzania."
        )
        self.assertRejected(
            "What is the capital of Palau? Reply with only the name.",
            "Koror City used to be the capital of Palau.",
        )
        self.assertRejected(
            "What is the capital of Palau? Reply with only the name.",
            "On October 7, 2006, Ngerulmud was officially inaugurated as the "
            "capital of Palau.",
            "historical_designation_not_current_evidence",
        )
