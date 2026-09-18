"""Adversarial contracts for evidence and whole-answer repair judging.

The semantic verifier is stubbed deliberately: negative decisions must prevent
acceptance, while necessary lexical guards must work even when it returns true.
Native probes separately assess the verifier's fallible semantic decisions.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from adaptible._src.wrap.repair import (
    Controller,
    merge_name_variants,
    normalized,
    read_retry_budget,
)
from adaptible._src.wrap.store import Store
from adaptible.tests.wrap_test import FakeRuntime, FakeTrainer

QUESTION = "What is the largest city in Morocco?"
REFERENCE = "Casablanca is the largest city in Morocco."


class GroundingQualityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = type("Runtime", (), {})()
        self.runtime.complete = AsyncMock()
        self.controller = Controller(self.runtime, None)
        self.controller.supported = AsyncMock(return_value=True)

    async def test_ambiguous_bracket_suffix_is_rejected_instead_of_changing_answer(
        self,
    ):
        for q, name, note in (
            (
                QUESTION,
                "Casablanca[a]",
                "Casablanca[a] is the largest city in Morocco.",
            ),
            ("Which label is printed?", "B[12]", "The printed label is B[12]."),
        ):
            with self.subTest(name=name):
                self.runtime.complete.reset_mock()
                self.runtime.complete.return_value = json.dumps(
                    dict(name=name, sentence_index=0)
                )
                self.assertEqual(await self.controller.grounded(q, note), {})
                self.runtime.complete.assert_awaited_once()

    async def test_apostrophe_inside_a_name_is_preserved(self):
        note = "The librarian is O'Connor."
        self.runtime.complete.return_value = json.dumps(
            dict(name="O'Connor", sentence_index=0)
        )
        result = await self.controller.grounded("Who is the librarian?", note)
        self.assertEqual(result, dict(name="O'Connor", quote=note))

    def test_punctuation_variants_agree_without_dropping_meaningful_words(self):
        self.assertEqual(normalized("Porto-Novo"), normalized("Porto Novo"))
        self.assertEqual(normalized("Porto–Novo"), normalized("Porto Novo"))
        self.assertNotEqual(normalized("Kansas City"), normalized("Kansas"))
        self.assertNotEqual(normalized("Mexico City"), normalized("Mexico"))
        self.assertNotEqual(normalized("-7"), normalized("7"))
        self.assertNotEqual(normalized("1990-1991"), normalized("1990 1991"))

    async def test_negative_answer_cannot_pass_as_positive_even_if_model_agrees(self):
        self.controller.grounded = AsyncMock(return_value=dict(name="-7", quote="-7."))
        self.assertFalse(
            await self.controller.judge(
                "How many days are in a week?",
                "-7.",
                "7",
                note="There are 7 days in a week.",
            )
        )

    async def test_literal_subsentence_cannot_drop_negating_reference_context(self):
        response = "Canberra is in Europe."
        note = "Canberra is the capital of Australia. It is false that Canberra is in Europe."
        self.controller.grounded = AsyncMock(
            return_value=dict(name="Canberra", quote=response)
        )
        self.assertFalse(
            await self.controller.judge(
                "What is the capital of Australia?", response, "Canberra", note=note
            )
        )

    async def test_copied_caption_is_not_evidence_that_city_answers_question(self):
        caption = "The 12th-century Koutoubia Mosque in Marrakesh."
        self.runtime.complete.return_value = json.dumps(
            dict(name="Marrakesh", sentence_index=0)
        )
        self.assertEqual(await self.controller.grounded(QUESTION, caption), {})
        self.controller.supported.assert_not_awaited()
        self.runtime.complete.assert_not_awaited()

    async def test_wrong_relation_cannot_vote_even_if_classifier_would_agree(self):
        for text in (
            "Ash is the capital of Eloria.",
            "Big city: Birch (why?",
            "Fir is not the largest city in Eloria.",
        ):
            with self.subTest(text=text):
                self.runtime.complete.return_value = json.dumps(
                    dict(name="Ash", sentence_index=0)
                )
                self.assertEqual(
                    await self.controller.grounded(
                        "Which city is largest in Eloria?", text
                    ),
                    {},
                )
                self.runtime.complete.assert_not_awaited()
                self.controller.supported.assert_not_awaited()

    async def test_irrelevant_capital_span_cannot_overshadow_ranked_statement(self):
        ranked = "Birch is certainly the largest city in the country."
        text = ranked + " Ash is the capital of Eloria."
        self.runtime.complete.return_value = json.dumps(
            dict(name="Birch", sentence_index=0)
        )
        self.assertEqual(
            await self.controller.grounded("Which city is largest in Eloria?", text),
            dict(name="Birch", quote=ranked),
        )
        actual = self.runtime.complete.call_args.args[0][-1]["content"]
        self.assertIn(ranked, actual)
        self.assertNotIn("Ash", actual)
        self.controller.supported.assert_awaited_once_with(
            "Which city is largest in Eloria?", "Birch", ranked
        )

    async def test_source_abstention_is_not_resampled_into_an_answer(self):
        caption = "An illustration of the largest city in the region."
        self.runtime.complete.side_effect = [
            json.dumps(dict(name="", sentence_index=-1)),
            json.dumps(dict(name="Marrakesh", sentence_index=0)),
        ]

        self.assertEqual(await self.controller.grounded(QUESTION, caption), {})
        self.runtime.complete.assert_awaited_once()
        self.controller.supported.assert_not_awaited()

    async def test_correct_quote_does_not_validate_wrong_entity_inside_it(self):
        self.runtime.complete.return_value = json.dumps(
            dict(name="Morocco", sentence_index=0)
        )

        async def supported(question, response, note):
            return response != "Morocco"

        self.controller.supported.side_effect = supported

        self.assertEqual(await self.controller.grounded(QUESTION, REFERENCE), {})
        self.controller.supported.assert_any_await(QUESTION, "Morocco", REFERENCE)
        self.runtime.complete.assert_awaited_once()

    async def test_relevance_check_receives_the_exact_evidence_retained_for_drafting(
        self,
    ):
        quote = "Casablanca is the largest city in Morocco."
        context = "The country has many cities. " + quote + " Rabat is its capital."
        self.runtime.complete.return_value = json.dumps(
            dict(name="Casablanca", sentence_index=0)
        )

        self.assertEqual(
            await self.controller.grounded(QUESTION, context),
            dict(quote=quote, name="Casablanca"),
        )
        self.controller.supported.assert_awaited_once_with(
            QUESTION, "Casablanca", quote
        )

    async def test_source_schema_answers_before_selecting_an_existing_sentence(self):
        self.runtime.complete.return_value = json.dumps(
            dict(name="Casablanca", sentence_index=0)
        )
        result = await self.controller.grounded(QUESTION, REFERENCE)
        self.assertEqual(result, dict(name="Casablanca", quote=REFERENCE))
        schema = self.runtime.complete.call_args.kwargs["response_format"][
            "json_schema"
        ]["schema"]
        self.assertEqual(list(schema["properties"]), ["name", "sentence_index"])
        self.assertNotIn("enum", schema["properties"]["name"])
        self.assertEqual(schema["properties"]["sentence_index"]["enum"], [-1, 0])

    async def test_invalid_indices_and_names_outside_selected_sentence_fail_closed(
        self,
    ):
        text = (
            "The largest city is in Morocco. Casablanca is the largest city in Morocco."
        )
        for selection in (
            dict(name="Casablanca", sentence_index=0),
            dict(name="Casablanca", sentence_index=2),
            dict(name="Casablanca", sentence_index=-2),
            dict(name="Casablanca", sentence_index=True),
            dict(name="Casablanca", sentence_index="1"),
            dict(name="Atlantis", sentence_index=1),
            dict(name=[], sentence_index=1),
        ):
            with self.subTest(selection=selection):
                self.runtime.complete.return_value = json.dumps(selection)
                self.controller.supported.reset_mock()
                self.assertEqual(await self.controller.grounded(QUESTION, text), {})
                self.controller.supported.assert_not_awaited()

    async def test_later_contradiction_is_visible_to_answer_extractor(self):
        prefix = "Casablanca is the largest city in Morocco."
        suffix = " That was incorrect; Marrakesh is actually the largest city."
        response = prefix + suffix

        async def extract(messages, **kwargs):
            text = messages[-1]["content"].split("\nQuestion:", 1)[0]
            text = text.removeprefix("Reference text: ")
            if "That was incorrect" in text:
                return json.dumps(dict(quote=suffix.strip(), name="Marrakesh"))
            return json.dumps(dict(quote=prefix, name="Casablanca"))

        self.runtime.complete.side_effect = extract

        self.assertFalse(await self.controller.judge(QUESTION, response, "Casablanca"))
        prompts = [
            call.args[0][-1]["content"]
            for call in self.runtime.complete.await_args_list
        ]
        self.assertTrue(any(response in prompt for prompt in prompts))

    async def test_unsupported_continuation_cannot_pass_on_correct_first_sentence(self):
        response = REFERENCE + " It is also the capital of Morocco."
        self.controller.grounded = AsyncMock(
            return_value=dict(quote=REFERENCE, name="Casablanca")
        )
        self.controller.supported.return_value = False

        self.assertFalse(
            await self.controller.judge(
                QUESTION, response, "Casablanca", note=REFERENCE
            )
        )

    async def test_positive_model_verdict_cannot_excuse_unsupported_continuation(self):
        response = REFERENCE + " It is also the capital of Morocco."
        self.controller.grounded = AsyncMock(
            return_value=dict(quote=REFERENCE, name="Casablanca")
        )
        self.controller.supported.return_value = True

        self.assertFalse(
            await self.controller.judge(
                QUESTION, response, "Casablanca", note=REFERENCE
            )
        )

    async def test_supported_complete_answer_passes_without_expected_name_hint(self):
        self.runtime.complete.return_value = json.dumps(
            dict(quote=REFERENCE, name="Casablanca")
        )

        self.assertTrue(
            await self.controller.judge(
                QUESTION, REFERENCE, "Casablanca", note=REFERENCE
            )
        )
        for call in self.runtime.complete.await_args_list:
            schema = call.kwargs["response_format"]["json_schema"]["schema"]
            self.assertNotIn("enum", schema["properties"]["name"])

    async def test_expected_answer_never_enters_extractor_prompt_or_retry_schema(self):
        hidden_expected = "unmentioned-expectation-sentinel"
        self.runtime.complete.return_value = "not JSON"

        self.assertEqual(
            await self.controller.grounded(
                QUESTION, REFERENCE, expected=hidden_expected
            ),
            {},
        )
        self.assertEqual(self.runtime.complete.await_count, 2)
        for call in self.runtime.complete.await_args_list:
            self.assertNotIn(hidden_expected, json.dumps(call.args))
            self.assertNotIn(hidden_expected, json.dumps(call.kwargs))
            schema = call.kwargs["response_format"]["json_schema"]["schema"]
            self.assertNotIn("enum", schema["properties"]["name"])


class SupportedBoundaryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = type("Runtime", (), {})()
        self.runtime.complete = AsyncMock()
        self.controller = Controller(self.runtime, None)

    async def test_malformed_json_and_non_object_results_fail_closed(self):
        for output in (
            "not JSON",
            "{",
            "[]",
            '[{"answers_question":true,"supported":true}]',
            "null",
            "true",
            "false",
            "1",
            '"true"',
        ):
            with self.subTest(output=output):
                self.runtime.complete.reset_mock()
                self.runtime.complete.return_value = output
                self.assertFalse(
                    await self.controller.supported(QUESTION, "Casablanca", REFERENCE)
                )
                self.runtime.complete.assert_awaited_once()

    async def test_only_two_explicit_true_booleans_accept(self):
        decisions = (
            {},
            dict(answers_question=True),
            dict(supported=True),
            dict(answers_question=False, supported=True),
            dict(answers_question=True, supported=False),
            dict(answers_question=1, supported=True),
            dict(answers_question=True, supported=1),
            dict(answers_question="true", supported=True),
            dict(answers_question=True, supported="true"),
            dict(answers_question=[], supported=True),
            dict(answers_question=True, supported={}),
            dict(answers_question=None, supported=True),
        )
        for decision in decisions:
            with self.subTest(decision=decision):
                self.runtime.complete.return_value = json.dumps(decision)
                self.assertFalse(
                    await self.controller.supported(QUESTION, "Casablanca", REFERENCE)
                )
        self.runtime.complete.return_value = json.dumps(
            dict(answers_question=True, supported=True)
        )
        self.assertTrue(
            await self.controller.supported(QUESTION, "Casablanca", REFERENCE)
        )

    async def test_verifier_uses_frozen_model_and_boolean_only_schema(self):
        self.runtime.complete.return_value = json.dumps(
            dict(answers_question=True, supported=True)
        )
        self.assertTrue(
            await self.controller.supported(QUESTION, "Casablanca", REFERENCE)
        )
        call = self.runtime.complete.call_args
        self.assertIs(call.kwargs["frozen"], True)
        schema = call.kwargs["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"],
            dict(answers_question=dict(type="boolean"), supported=dict(type="boolean")),
        )
        self.assertNotIn("enum", json.dumps(schema))
        self.assertNotIn("Casablanca", json.dumps(schema))


class QuestionVariantTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = type("Runtime", (), {})()
        self.runtime.complete = AsyncMock()
        self.controller = Controller(self.runtime, None)
        self.original = [dict(role="user", content=QUESTION)]

    async def test_structured_variants_preserve_original_and_allow_imperatives(self):
        variants = [
            "Which Moroccan city is the largest?",
            "Name the largest city in Morocco.",
            "Morocco has which city as its largest?",
        ]
        self.runtime.complete.return_value = json.dumps(dict(questions=variants))
        result = await self.controller.prompts(QUESTION, self.original)
        self.assertIs(result[0], self.original)
        self.assertEqual([messages[0]["content"] for messages in result[1:]], variants)
        call = self.runtime.complete.call_args
        self.assertIs(call.kwargs["frozen"], True)
        schema = call.kwargs["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["questions"]["minItems"], 3)
        self.assertEqual(schema["properties"]["questions"]["maxItems"], 3)
        self.assertEqual(call.args[0][-1]["content"], QUESTION)
        self.assertNotIn("Casablanca", json.dumps(call.args))
        self.assertNotIn("Casablanca", json.dumps(call.kwargs))

    async def test_duplicate_and_cosmetic_variants_do_not_inflate_coverage(self):
        variant = "Name the largest city in Morocco."
        self.runtime.complete.return_value = json.dumps(
            dict(
                questions=[QUESTION.upper().replace("?", "!"), variant, variant.upper()]
            )
        )
        result = await self.controller.prompts(QUESTION, self.original)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1][0]["content"], variant)

    async def test_malformed_structure_does_not_fabricate_paraphrases(self):
        for output in (
            "not JSON",
            "null",
            "[]",
            "{}",
            '{"questions":"question"}',
            '{"questions":[null,1,{}]}',
        ):
            with self.subTest(output=output):
                self.runtime.complete.return_value = output
                self.assertEqual(
                    await self.controller.prompts(QUESTION, self.original),
                    [self.original],
                )

    async def test_changed_meaning_or_unsupported_variant_is_excluded(self):
        same = [dict(role="user", content="Name the largest city in Morocco.")]
        changed = [dict(role="user", content="Name Morocco's capital city.")]
        unsupported = [dict(role="user", content="Name the oldest city in Morocco.")]
        self.controller.grounded = AsyncMock(
            # Even a classifier that falsely approves every variant cannot
            # change a recognized question's subject or requested property.
            return_value=dict(name="Casablanca", quote=REFERENCE)
        )
        result = await self.controller.validate_prompts(
            [self.original, same, changed, unsupported],
            note=REFERENCE,
            expected="Casablanca",
        )
        self.assertEqual(result, [self.original, same])
        self.controller.grounded.assert_awaited_once_with(same[0]["content"], REFERENCE)

    async def test_generated_answer_hint_is_not_used_as_a_question(self):
        leaked = [
            dict(role="user", content="Is Casablanca the largest city in Morocco?")
        ]
        self.controller.grounded = AsyncMock(
            return_value=dict(name="Casablanca", quote=REFERENCE)
        )
        result = await self.controller.validate_prompts(
            [self.original, leaked], note=REFERENCE, expected="Casablanca"
        )
        self.assertEqual(result, [self.original])
        self.controller.grounded.assert_not_awaited()


class CorrectionSupersessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_updated_reference_can_correct_the_same_accepted_interaction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "supersession-test")
            runtime = FakeRuntime()
            trainer = FakeTrainer()
            trainer.runtime = runtime
            controller = Controller(runtime, store, trainer=trainer, web_search=False)
            question = "What is the capital of Australia?"
            messages = [dict(role="user", content=question)]
            idx = store.record(messages, "Melbourne.")
            previous = dict(directory=str(Path(directory) / "old-adapter"))
            store.accept_repair(
                previous,
                dict(
                    interaction_idx=idx,
                    question=question,
                    messages=messages,
                    prompts=[messages],
                    expected="Sydney",
                    note="Sydney is the capital of Australia.",
                ),
            )
            runtime.active = previous["directory"]
            original = runtime.complete

            async def complete(messages, *, handle=None, **kwargs):
                if messages[-1]["content"] in (
                    question,
                    "Which city is Australia's capital?",
                ):
                    return "Canberra." if handle else "Sydney."
                return await original(messages, handle=handle, **kwargs)

            runtime.complete = complete
            store.feedback(idx, True, "Canberra is the capital of Australia.")
            try:
                status, _ = await controller.repair(store.pending()[0])
                self.assertEqual(status, "kept")
                self.assertTrue(trainer.calls)
                self.assertNotEqual(store.get("accepted"), previous)
                records = store.repairs()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["interaction_idx"], idx)
                self.assertEqual(records[0]["expected"], "Canberra")
            finally:
                await controller.close()


if __name__ == "__main__":
    unittest.main()


class NameVariantCorroborationTest(unittest.TestCase):
    """Sources that agree must not be read as a conflict.

    A live run retrieved three sources for Tanzania's capital -- two saying
    "Dodoma", one "Dodoma City" -- and skipped the review for conflicting
    answers, discarding evidence that agreed.
    """

    def merged(self, answers):
        return {
            name: len(sources) for name, sources in merge_name_variants(answers).items()
        }

    def test_a_trailing_word_is_not_a_different_answer(self):
        self.assertEqual(
            self.merged({"dodoma": ["a", "b"], "dodoma city": ["c"]}), {"dodoma": 3}
        )

    def test_a_leading_word_is_not_a_different_answer(self):
        self.assertEqual(
            self.merged({"porto novo": ["a"], "city of porto novo": ["b"]}),
            {"porto novo": 2},
        )

    def test_genuinely_different_answers_stay_in_conflict(self):
        self.assertEqual(
            self.merged({"dodoma": ["a"], "dar es salaam": ["b"]}),
            {"dodoma": 1, "dar es salaam": 1},
        )

    def test_a_shared_word_alone_does_not_merge(self):
        # "Kansas City" and "Panama City" share a word but name two places.
        self.assertEqual(
            self.merged({"kansas city": ["a"], "panama city": ["b"]}),
            {"kansas city": 1, "panama city": 1},
        )

    def test_the_best_supported_variant_represents_the_group(self):
        merged = merge_name_variants({"dodoma city": ["a"], "dodoma": ["b", "c", "d"]})
        self.assertEqual(list(merged), ["dodoma"])
        self.assertEqual(len(merged["dodoma"]), 4)


class SuppliedNoteTest(unittest.IsolatedAsyncioTestCase):
    """A note attached to a correction is an explicit reference, not web prose.

    The flagship loop (``scripts/cycles_mlx.py``) attaches the reference note
    "the correct answer is X." and reads the correction from it. That note is
    not prose and carries no relation for the source guards to check, so
    routing it through the retrieved-prose path rejects the experiment's own
    reference before the model is ever asked.
    """

    QUESTION = "What is the capital of Turkey?"
    NOTE = "the correct answer is Ankara."

    def setUp(self):
        self.runtime = type("Runtime", (), {})()
        self.runtime.complete = AsyncMock()
        self.controller = Controller(self.runtime, None)
        self.controller.supported = AsyncMock(return_value=True)
        self.controller.store = SimpleNamespace(references=Mock())

    async def test_a_note_stating_the_answer_is_read(self):
        self.runtime.complete.return_value = json.dumps(
            dict(quote=self.NOTE, name="Ankara")
        )
        note, name, _ = await self.controller.find_reference(
            dict(id=1, note=self.NOTE), self.QUESTION
        )
        self.assertEqual((note, name), (self.NOTE, "Ankara"))

    async def test_a_note_the_model_cannot_read_yields_no_answer(self):
        # The reader still has to find the answer in the note: an unrelated
        # note must not become a correction.
        self.runtime.complete.return_value = json.dumps(dict(quote="", name=""))
        _, name, reason = await self.controller.find_reference(
            dict(id=1, note="Tulips are flowers."), self.QUESTION
        )
        self.assertEqual(name, "")
        self.assertIn("could not read", reason)

    async def test_a_name_absent_from_the_note_is_refused(self):
        # Invention is still blocked: the name must occur in the quote.
        self.runtime.complete.return_value = json.dumps(
            dict(quote=self.NOTE, name="Istanbul")
        )
        _, name, _ = await self.controller.find_reference(
            dict(id=1, note=self.NOTE), self.QUESTION
        )
        self.assertEqual(name, "")

    async def test_stored_documents_keep_the_guarded_prose_path(self):
        # Only the note changes. A document naming a different property must
        # still be refused by the source guard, before any extraction.
        self.controller.lookup = Mock(
            return_value="Istanbul is the largest city in Turkey."
        )
        _, name, _ = await self.controller.find_reference(
            dict(id=1, note=""), self.QUESTION
        )
        self.assertEqual(name, "")
        self.runtime.complete.assert_not_awaited()

    async def test_variants_are_validated_with_the_reference_reader(self):
        # A generated rewording is checked against the same evidence, so it has
        # to be read the same way the evidence was read. Sending the variant
        # through the prose path would drop every rephrasing of a noted repair.
        row = dict(id=1, note=self.NOTE)
        self.controller.read_reference = AsyncMock(
            return_value=dict(name="Ankara", quote=self.NOTE)
        )
        self.controller.grounded = AsyncMock(return_value={})
        original = [dict(role="user", content=self.QUESTION)]
        variant = [dict(role="user", content="Name the capital city of Turkey.")]
        result = await self.controller.validate_prompts(
            [original, variant],
            note=self.NOTE,
            expected="Ankara",
            reader=self.controller.reader(row),
        )
        self.assertEqual(result, [original, variant])
        self.controller.grounded.assert_not_awaited()

    async def test_quote_is_constrained_to_the_reference_sentences(self):
        # A model that disagrees with the reference will otherwise write its
        # own quote: "the correct answer is The Sun." came back as "The correct
        # answer is Earth.", and unconstrained it answers "Proxima Centauri"
        # with a matching invented sentence. Copying is enforced by the schema.
        note = "The correct answer is The Sun. It is a star."
        self.runtime.complete.return_value = json.dumps(
            dict(quote="The correct answer is The Sun.", name="The Sun")
        )
        result = await self.controller.read_reference(
            "What is the nearest star to Earth?", note
        )
        self.assertEqual(result["name"], "The Sun")
        schema = self.runtime.complete.await_args.kwargs["response_format"][
            "json_schema"
        ]["schema"]
        self.assertEqual(
            schema["properties"]["quote"]["enum"],
            ["The correct answer is The Sun.", "It is a star.", ""],
        )
        # The answer itself is never offered as a choice.
        self.assertNotIn("enum", schema["properties"]["name"])

    async def test_a_long_reference_is_not_turned_into_a_huge_grammar(self):
        long_reference = " ".join(f"Sentence number {i}." for i in range(80))
        self.runtime.complete.return_value = json.dumps(dict(quote="", name=""))
        await self.controller.read_reference(self.QUESTION, long_reference)
        schema = self.runtime.complete.await_args.kwargs["response_format"][
            "json_schema"
        ]["schema"]
        self.assertNotIn("enum", schema["properties"]["quote"])

    async def test_a_read_that_runs_out_of_tokens_is_retried_with_more(self):
        # The first attempt spends a serving turn's whole budget reasoning and
        # returns nothing. Reporting that as an unreadable reference loses a
        # correction the model can make; the retry gets room to finish.
        first = dict(finish_reason="length", reasoning="thinking " * 500, content="")
        second = json.dumps(dict(quote=self.NOTE, name="Ankara"))

        async def complete(messages, **kwargs):
            if self.runtime.complete.await_count == 1:
                kwargs["details"].update(first)
                return ""
            kwargs["details"].update(finish_reason="stop")
            return second

        self.runtime.complete.side_effect = complete
        result = await self.controller.read_reference(self.QUESTION, self.NOTE)
        self.assertEqual(result["name"], "Ankara")
        budgets = [
            c.kwargs.get("max_tokens") for c in self.runtime.complete.await_args_list
        ]
        self.assertIsNone(budgets[0])
        self.assertEqual(budgets[1], 8192)

    async def test_the_retry_budget_stays_inside_the_served_context(self):
        # vLLM refuses a request whose max_tokens exceeds the context window
        # with a 400 rather than truncating, which failed the whole repair.
        self.runtime.context_size = 4096
        self.runtime.max_tokens = 1024
        self.assertEqual(read_retry_budget(self.runtime), 2048)
        self.runtime.context_size = 65536
        self.assertEqual(read_retry_budget(self.runtime), 8192)
        self.runtime.context_size = 512
        self.assertEqual(read_retry_budget(self.runtime), 1024)
        del self.runtime.context_size
        self.assertEqual(read_retry_budget(self.runtime), 8192)

    async def test_a_contrary_prior_gets_the_flagship_number_of_tries(self):
        # The greedy read answers from the model's prior instead of the note;
        # sampled reads are what recover it. The flagship reader takes four.
        wrong = json.dumps(dict(quote=self.NOTE, name="Istanbul"))
        right = json.dumps(dict(quote=self.NOTE, name="Ankara"))
        self.runtime.complete.side_effect = [wrong, wrong, wrong, right]
        result = await self.controller.read_reference(self.QUESTION, self.NOTE)
        self.assertEqual(result["name"], "Ankara")
        self.assertEqual(self.runtime.complete.await_count, 4)

    async def test_judging_an_answer_keeps_its_two_attempts(self):
        # Judging is not resampling an opinion: the retry pins the quote.
        self.runtime.complete.return_value = json.dumps(dict(quote="", name=""))
        await self.controller.read_reference(
            self.QUESTION, "Ankara is the capital.", expected="Ankara"
        )
        self.assertLessEqual(self.runtime.complete.await_count, 2)
