"""Routing guards are exercised with an intentionally overconfident classifier."""

import asyncio
import importlib.util
import json
import unittest

import httpx
from unittest.mock import ANY, DEFAULT, AsyncMock, patch

from adaptible._src.wrap.scope_router import ScopeRouter


def messages(question):
    return [{"role": "user", "content": question}]


def repair(question="What is the largest city in Eloria?", idx=7, **kwargs):
    return dict(
        interaction_idx=idx,
        question=question,
        prompts=[question],
        messages=messages(question),
        generation_mode={"thinking": True},
        **kwargs,
    )


class ScopeRouterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Exercise the classifier fallback independently of the grammar fast path.
        # FactScopeRouterTest verifies the integrated grammar and its safety gates.
        grammar = patch(
            "adaptible._src.wrap.scope_router._grammar_match", return_value=False
        )
        grammar.start()
        self.addCleanup(grammar.stop)
        self.runtime = type("Runtime", (), {})()

        async def completed(*args, details, **kwargs):
            details.update(
                complete=True, framing_valid=True, finish_reason="stop", reasoning=""
            )
            return DEFAULT

        self.runtime.complete = AsyncMock(
            return_value='{"same_fact":true}', side_effect=completed
        )
        self.router = ScopeRouter(self.runtime)
        self.mode = {"thinking": True}
        self.repairs = [repair()]

    async def choose(self, q):
        return await self.router.choose(messages(q), self.repairs, self.mode)

    async def test_exact_and_unseen_paraphrase_use_same_scope(self):
        self.assertEqual(
            await self.choose("What is the largest city in Eloria?"),
            dict(scope=7, reason="exact"),
        )
        self.runtime.complete.assert_not_awaited()
        self.assertEqual(
            await self.choose("Name Eloria's biggest city."),
            dict(scope=7, reason="scope_match", classifier=ANY),
        )
        self.assertEqual(
            await self.choose("Which city in Eloria is biggest?"),
            dict(scope=7, reason="scope_match", classifier=ANY),
        )
        self.assertEqual(
            await self.choose("Which city in Eloria has the most inhabitants?"),
            dict(scope=7, reason="scope_match", classifier=ANY),
        )

    async def test_wrong_subject_relation_qualifier_never_reaches_lying_classifier(
        self,
    ):
        for q in (
            "What is the largest city in Beloria?",
            "What is the capital of Eloria?",
            "What is the second largest city in Eloria?",
            "What is not the largest city in Eloria?",
            "What was the largest city in Eloria in 1900?",
            "What is the largest city by area in Eloria?",
            "What is the smallest city in Eloria?",
            "What is the largest city in North Eloria?",
            "What is the largest country in Eloria?",
            "What is the largest city and capital in Eloria?",
        ):
            with self.subTest(q=q):
                self.assertIsNone((await self.choose(q))["scope"])
        self.runtime.complete.assert_not_awaited()

    async def test_shared_geographic_word_is_not_subject_equivalence(self):
        self.repairs = [repair("What is the capital of South Meridia?")]
        self.assertIsNone(
            (await self.choose("Name the capital of North Meridia."))["scope"]
        )
        self.runtime.complete.assert_not_awaited()

    async def test_productive_adjectives_only_shortlist_for_semantic_check(self):
        for place, adjective in (
            ("Arborland", "Arborlandian"),
            ("Veloro", "Veloran"),
            ("Tavira", "Tavirese"),
        ):
            with self.subTest(place=place):
                router = ScopeRouter(self.runtime)
                question = f"Which {adjective} city is largest?"
                scopes = [repair(f"What is the largest city in {place}?")]
                self.runtime.complete.return_value = '{"same_fact":false}'
                self.assertEqual(
                    await router.choose(messages(question), scopes, self.mode),
                    dict(scope=None, reason="router_abstained", classifier=ANY),
                )
                self.runtime.complete.return_value = '{"same_fact":true}'
                self.assertEqual(
                    await ScopeRouter(self.runtime).choose(
                        messages(question), scopes, self.mode
                    ),
                    dict(scope=7, reason="scope_match", classifier=ANY),
                )
        self.assertEqual(self.runtime.complete.await_count, 6)

    async def test_adjective_matching_preserves_subject_and_scope_boundaries(self):
        self.repairs = [repair("What is the largest city in North Arborland?")]
        for question in (
            "Which South Arborlandian city is largest?",
            "Which North Arborvillean city is largest?",
            "Which North Arborlandiaese city is largest?",
            "Which North Arborlandian city is the capital?",
            "Which North Arborlandian city is largest by area?",
        ):
            with self.subTest(question=question):
                self.assertIsNone((await self.choose(question))["scope"])
        self.repairs = [repair("What is the largest city in Aro?")]
        self.assertIsNone((await self.choose("Which Aroian city is largest?"))["scope"])
        self.runtime.complete.assert_not_awaited()

    async def test_population_wordings_share_shortlist_but_area_does_not(self):
        self.repairs = [repair("What is the largest city in Arborland?")]
        for question in (
            "Which city in Arborland is largest by population?",
            "Which city in Arborland is most populous?",
            "Which city in Arborland has the most inhabitants?",
        ):
            self.assertEqual((await self.choose(question))["scope"], 7)
        self.assertEqual(self.runtime.complete.await_count, 3)
        self.assertIsNone(
            (await self.choose("Which city in Arborland is largest by area?"))["scope"]
        )
        self.assertEqual(self.runtime.complete.await_count, 3)

    async def test_classifier_only_receives_questions_and_uses_frozen_base(self):
        self.repairs[0].update(
            expected="SECRET_ANSWER",
            note="SECRET_EVIDENCE",
            evidence={"secret": "SECRET_SOURCE"},
            response="SECRET_RESPONSE",
            heldout_prompts=["SECRET_EXTERNAL_TEST"],
        )
        await self.choose("Which city in Eloria is biggest?")
        args, kwargs = self.runtime.complete.call_args
        payload = json.dumps(args)
        self.assertNotIn("SECRET", payload)
        self.assertTrue(kwargs["frozen"])
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(
            kwargs["response_format"]["json_schema"]["schema"]["properties"],
            {"same_fact": {"type": "boolean"}},
        )

    async def test_malformed_and_false_results_abstain(self):
        for value in (
            "null",
            "[]",
            "true",
            '{"same_fact":"true"}',
            '{"same_fact":1}',
            '{"same_fact":true,"extra":false}',
            '{"same_fact":false}',
            "bad json",
        ):
            router = ScopeRouter(self.runtime)
            self.runtime.complete.return_value = value
            self.assertEqual(
                await router.choose(
                    messages("Which city in Eloria is biggest?"),
                    self.repairs,
                    self.mode,
                ),
                dict(scope=None, reason="router_abstained", classifier=ANY),
            )

    async def test_mode_mismatch_prevents_even_exact_route(self):
        self.assertEqual(
            await self.router.choose(
                messages(self.repairs[0]["question"]), self.repairs, {"thinking": False}
            ),
            dict(scope=None, reason="mode_mismatch"),
        )
        self.runtime.complete.assert_not_awaited()

    async def test_context_prefix_must_match_exactly_and_never_enters_classifier(self):
        prefix = [
            {"role": "system", "content": "SECRET_SYSTEM"},
            {"role": "user", "content": "Earlier context"},
            {"role": "assistant", "content": "SECRET_OLD_ANSWER"},
        ]
        self.repairs[0]["messages"] = prefix + messages(self.repairs[0]["question"])
        q = messages("Which city in Eloria is biggest?")
        for context in ([], prefix[:-1], [{"role": "system", "content": "Different"}]):
            self.assertIsNone(
                (await self.router.choose(context + q, self.repairs, self.mode))[
                    "scope"
                ]
            )
        result = await self.router.choose(prefix + q, self.repairs, self.mode)
        self.assertEqual(result["scope"], 7)
        self.assertNotIn("SECRET", json.dumps(self.runtime.complete.call_args.args))

    async def test_tools_images_and_ambiguous_followup_abstain(self):
        for value in (
            [{"role": "user", "content": [{"type": "text", "text": "Eloria"}]}],
            [{"role": "tool", "content": "Eloria"}],
            [{"role": "user", "content": "Eloria", "tool_calls": []}],
        ):
            self.assertEqual(
                (await self.router.choose(value, self.repairs, self.mode))["reason"],
                "unsupported_context",
            )
        self.assertIsNone((await self.choose("And its largest city?"))["scope"])
        self.runtime.complete.assert_not_awaited()

    async def test_cache_tracks_scope_changes_and_is_bounded(self):
        self.router = ScopeRouter(self.runtime, cache_size=1)
        q = "Which city in Eloria is biggest?"
        await self.choose(q)
        await self.choose(q)
        self.assertEqual(self.runtime.complete.await_count, 1)
        self.repairs[0]["interaction_idx"] = 8
        self.assertEqual((await self.choose(q))["scope"], 8)
        self.assertEqual(self.runtime.complete.await_count, 2)
        self.assertEqual(len(self.router.cache), 1)
        self.repairs.clear()
        self.assertIsNone((await self.choose(q))["scope"])

    async def test_real_message_variants_and_output_suffix_do_not_hide_scope(self):
        self.repairs[0]["question"] += " Reply with only the city name."
        self.repairs[0]["prompts"] = [messages("Identify the biggest city in Eloria.")]
        result = await self.choose("Identify the biggest city in Eloria.")
        self.assertEqual(result, dict(scope=7, reason="exact"))
        self.assertEqual(
            (await self.choose("Which city in Eloria is biggest?"))["scope"], 7
        )
        self.assertIsNone(
            (await self.choose("Name a populated city in Eloria."))["scope"]
        )
        self.assertIsNone(
            (await self.choose("Name a populous city in Eloria."))["scope"]
        )

    async def test_legacy_nonthinking_mode_matches_explicit_false(self):
        self.repairs[0]["generation_mode"] = {}
        self.assertEqual(
            (
                await self.router.choose(
                    messages(self.repairs[0]["question"]),
                    self.repairs,
                    {"thinking": False},
                )
            )["scope"],
            7,
        )

    async def test_signed_numeric_subjects_are_distinct(self):
        self.repairs = [repair("What is the square of -7?")]
        self.assertIsNone((await self.choose("Give the square of 7."))["scope"])
        self.runtime.complete.assert_not_awaited()

    async def test_transport_error_abstains(self):
        self.runtime.complete.side_effect = httpx.ConnectError("offline")
        self.assertEqual(
            (await self.choose("Which city in Eloria is biggest?"))["reason"],
            "router_abstained",
        )

    async def test_classifier_diagnostics_distinguish_false_malformed_and_transport(
        self,
    ):
        q = messages("Which city in Eloria is biggest?")
        for raw, verdict in (
            ('{"same_fact":true}', "same_fact"),
            ('{"same_fact":false}', "different_fact"),
            ('{"same_fact":"true"}', "malformed"),
            ("broken", "malformed"),
        ):
            self.runtime.complete.return_value = raw
            result = await ScopeRouter(self.runtime).choose(q, self.repairs, self.mode)
            self.assertEqual(result["classifier"]["verdict"], verdict)
            self.assertEqual(result["classifier"]["raw_output"], raw)
            self.assertEqual(
                result["classifier"]["input"]["question_b"], q[0]["content"]
            )
        self.runtime.complete.side_effect = httpx.ConnectError(
            "PRIVATE_TRANSPORT_DETAIL"
        )
        result = await ScopeRouter(self.runtime).choose(q, self.repairs, self.mode)
        self.assertEqual(result["classifier"]["verdict"], "transport_error")
        self.assertEqual(result["classifier"]["error_type"], "ConnectError")
        self.assertNotIn("PRIVATE_TRANSPORT_DETAIL", json.dumps(result))

    async def test_classifier_diagnostics_keep_observed_generation_details(self):
        async def complete(*args, details, **kwargs):
            details.update(
                finish_reason="stop", reasoning="", framing_valid=True, complete=True
            )
            return '{"same_fact":false}'

        self.runtime.complete.side_effect = complete
        result = await self.choose("Which city in Eloria is biggest?")
        self.assertEqual(result["classifier"]["generation"]["finish_reason"], "stop")
        self.assertEqual(result["classifier"]["generation"]["reasoning"], "")

    async def test_valid_json_requires_complete_generation_and_valid_framing(self):
        for generation in (
            {},
            dict(complete=False, framing_valid=True, finish_reason="length"),
            dict(complete=True, framing_valid=True, finish_reason="length"),
            dict(complete=True, framing_valid=False, finish_reason="stop"),
            dict(complete=True, framing_valid=True, finish_reason=None),
            dict(complete=True, framing_valid=True, finish_reason="tool_calls"),
            dict(complete="true", framing_valid=True, finish_reason="stop"),
            dict(complete=True, framing_valid=1, finish_reason="stop"),
        ):
            with self.subTest(generation=generation):

                async def complete(*args, details, **kwargs):
                    details.update(generation)
                    return '{"same_fact":true}'

                self.runtime.complete.side_effect = complete
                result = await ScopeRouter(self.runtime).choose(
                    messages("Which city in Eloria is biggest?"),
                    self.repairs,
                    self.mode,
                )
                self.assertIsNone(result["scope"])
                self.assertEqual(
                    result["classifier"]["raw_output"], '{"same_fact":true}'
                )
                self.assertEqual(
                    result["classifier"]["verdict"], "incomplete_generation"
                )

    async def test_supported_normal_stop_reasons_can_authorize_complete_json(self):
        for finish in ("stop", "eos", "end_turn"):

            async def complete(*args, details, **kwargs):
                details.update(complete=True, framing_valid=True, finish_reason=finish)
                return '{"same_fact":true}'

            self.runtime.complete.side_effect = complete
            result = await ScopeRouter(self.runtime).choose(
                messages("Which city in Eloria is biggest?"), self.repairs, self.mode
            )
            self.assertEqual(result["scope"], 7)

    async def test_ambiguous_scopes_abstain_and_cancellation_propagates(self):
        self.repairs.append(repair("Which city in Eloria is largest?", idx=8))
        self.assertEqual(
            (await self.choose("Which city in Eloria is biggest?"))["reason"],
            "router_abstained",
        )
        self.runtime.complete.assert_not_awaited()
        self.repairs.pop()
        self.runtime.complete.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.choose("Which city in Eloria is biggest?")

    async def test_repeated_scope_routes_latest_id_and_keeps_validated_variants(self):
        self.repairs[0]["prompts"] = [messages("Identify the largest city in Eloria.")]
        later = repair("WHAT IS THE LARGEST CITY IN ELORIA?", idx=19)
        later["prompts"] = [messages("Name the biggest city in Eloria.")]
        # Input order must not determine which accepted interaction is reported.
        self.repairs.insert(0, later)
        for q in (
            self.repairs[1]["question"],
            "Identify the largest city in Eloria.",
            "Name the biggest city in Eloria.",
        ):
            self.assertEqual(await self.choose(q), dict(scope=19, reason="exact"))
        self.runtime.complete.assert_not_awaited()
        result = await self.choose("Which city in Eloria has the most inhabitants?")
        self.assertEqual(result, dict(scope=19, reason="scope_match", classifier=ANY))
        payload = json.loads(self.runtime.complete.call_args.args[0][-1]["content"])
        self.assertIn(
            "Identify the largest city in Eloria.",
            result["classifier"]["scope_questions"],
        )
        self.assertIn(
            "Name the biggest city in Eloria.", result["classifier"]["scope_questions"]
        )

    async def test_repeated_original_does_not_union_other_mode_or_context(self):
        other_mode = repair(idx=19)
        other_mode["generation_mode"] = {"thinking": False}
        other_mode["prompts"] = [messages("Name the biggest city in Eloria.")]
        other_context = repair(idx=20)
        other_context["messages"].insert(
            0, {"role": "system", "content": "Different context"}
        )
        other_context["prompts"] = [messages("Identify the biggest city in Eloria.")]
        self.repairs.extend([other_mode, other_context])
        self.assertEqual((await self.choose(self.repairs[0]["question"]))["scope"], 7)
        result = await self.choose("Name the biggest city in Eloria.")
        self.assertEqual(result, dict(scope=7, reason="scope_match", classifier=ANY))
        payload = json.loads(self.runtime.complete.call_args.args[0][-1]["content"])
        self.assertNotIn(
            "Name the biggest city in Eloria.", result["classifier"]["scope_questions"]
        )
        self.assertNotIn(
            "Identify the biggest city in Eloria.",
            result["classifier"]["scope_questions"],
        )


if __name__ == "__main__":
    unittest.main()


class SelectiveReasonerModeTest(unittest.IsolatedAsyncioTestCase):
    """A model that decides per turn whether to reason keeps one scope.

    The repair is recorded with whatever that answer did; the identical re-ask
    is routed with whatever the request declared. Splitting those into separate
    mode partitions serves the frozen base to both, so a correction trains, is
    accepted, and is then never actually used.
    """

    INFERRED = ["model_reasons_unconditionally"]

    def setUp(self):
        self.router = ScopeRouter(None)

    async def choose(self, repair_mode, request_mode):
        entry = repair(idx=5)
        entry["generation_mode"] = repair_mode
        return await self.router.choose(
            messages("What is the largest city in Eloria?"), [entry], request_mode
        )

    async def test_a_thoughtless_repair_still_serves_a_thinking_request(self):
        result = await self.choose(
            dict(thinking=False, source=self.INFERRED + ["turn_without_reasoning"]),
            dict(thinking=True, source=self.INFERRED),
        )
        self.assertEqual(result["scope"], 5)

    async def test_a_thinking_repair_still_serves_a_thoughtless_request(self):
        result = await self.choose(
            dict(thinking=True, source=self.INFERRED),
            dict(thinking=False, source=self.INFERRED + ["turn_without_reasoning"]),
        )
        self.assertEqual(result["scope"], 5)

    async def test_a_controllable_model_still_separates_its_modes(self):
        # Where the request really does carry a thinking control, a repair
        # trained in the other mode must not be served to it.
        result = await self.choose(dict(thinking=True), dict(thinking=False))
        self.assertIsNone(result["scope"])
        self.assertEqual(result["reason"], "mode_mismatch")


class FlagshipRecipeRoutingTest(unittest.IsolatedAsyncioTestCase):
    """Reproducing cycles_mlx.py means the weights answer every prompt.

    The scope router exists so a served wrapper cannot leak one correction into
    unrelated requests. The experiment has no such gate: training updates the
    weights and every prompt is answered by them, so a paraphrase answering
    correctly is the model generalizing. Routing would measure the router.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from adaptible._src.wrap.repair import Controller
        from adaptible._src.wrap.store import Store
        from adaptible.tests.wrap_test import FakeRuntime

        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime()
        self.runtime.active = "adapter-handle"
        self.store = Store(Path(self.temp.name), "recipe-test")
        self.store.set("accepted", dict(directory="/tmp/x"))
        self.controller = Controller(
            self.runtime, self.store, web_search=False, flagship_recipe=True
        )
        self.strict = Controller(self.runtime, self.store, web_search=False)
        self.messages = [dict(role="user", content="Something entirely unrelated.")]

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    async def test_every_prompt_is_answered_by_the_accepted_weights(self):
        self.controller.store.repairs = lambda: [dict(interaction_idx=7, prompts=[])]
        route = await self.controller.route(self.messages, {})
        self.assertEqual(route["scope"], 7)
        self.assertEqual(route["reason"], "flagship_recipe_serves_all")
        self.assertEqual(route["policy"], "flagship_recipe_v1")

    async def test_the_served_default_still_routes_by_scope(self):
        self.strict.store.repairs = lambda: [dict(interaction_idx=7, prompts=[])]
        self.strict.scope_router.choose = AsyncMock(
            return_value=dict(scope=None, reason="no_scope")
        )
        route = await self.strict.route(self.messages, {})
        self.assertIsNone(route["scope"])
        self.assertEqual(route["policy"], "correction_scoped_v1")
        self.strict.scope_router.choose.assert_awaited_once()


class FlagshipTrainingExampleTest(unittest.IsolatedAsyncioTestCase):
    """The trained pair must be the one cycles_mlx.example_from builds.

    The experiment samples a correction from the hinted prompt and trains that
    candidate's own reasoning together with the first two sentences of its
    answer, as a single example. Training a bare extracted name, or adding
    generated phrasings, is a different and easier experiment.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from adaptible._src.wrap.repair import Controller
        from adaptible._src.wrap.store import Store
        from adaptible.tests.wrap_test import FakeRuntime

        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime()
        self.store = Store(Path(self.temp.name), "flagship-example")
        self.controller = Controller(
            self.runtime, self.store, web_search=False, flagship_recipe=True
        )
        self.sent = []

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def reply(self, reasoning, content, complete=True):
        async def complete_fn(messages, **kwargs):
            self.sent.append((messages, kwargs))
            kwargs["details"].update(
                content=content,
                reasoning=reasoning,
                reasoning_prefix=f"<think>\n{reasoning}\n</think>\n\n",
                finish_reason="stop",
                framing_valid=True,
                complete=complete,
            )
            return content

        self.runtime.complete = complete_fn

    async def test_candidate_reasoning_and_two_sentences_are_trained(self):
        self.reply(
            "The note says Ankara.",
            "Ankara is the capital. It has been since 1923. A third sentence.",
        )
        example = await self.controller.flagship_candidate(
            "What is the capital of Turkey?", "the correct answer is Ankara.", "Ankara"
        )
        self.assertEqual(
            example["target"], "Ankara is the capital. It has been since 1923."
        )
        self.assertIn("The note says Ankara.", example["reasoning_prefix"])
        prompt = self.sent[0][0][-1]["content"]
        self.assertEqual(
            prompt,
            "What is the capital of Turkey?\n\n(Reference note: the correct answer is Ankara.)",
        )

    async def test_a_candidate_missing_the_answer_is_not_used(self):
        self.reply("Thinking.", "Istanbul is the capital of Turkey.")
        example = await self.controller.flagship_candidate(
            "What is the capital of Turkey?", "the correct answer is Ankara.", "Ankara"
        )
        self.assertIsNone(example)

    async def test_an_unfinished_thought_is_not_used(self):
        self.reply("Thinking.", "Ankara.", complete=False)
        example = await self.controller.flagship_candidate(
            "What is the capital of Turkey?", "the correct answer is Ankara.", "Ankara"
        )
        self.assertIsNone(example)

    async def test_a_runaway_first_sentence_is_not_used(self):
        self.reply("Thinking.", "Ankara " + "and more " * 60 + ". Second.")
        example = await self.controller.flagship_candidate(
            "What is the capital of Turkey?", "the correct answer is Ankara.", "Ankara"
        )
        self.assertIsNone(example)


class FlagshipStepBudgetTest(unittest.IsolatedAsyncioTestCase):
    """The experiment trains at most MAX_STEPS=4 toward a 0.15 answer loss.

    The wrapper's own budget is a doubling ladder starting at one optimizer
    step, which is enough to nudge a bare extracted name but not to learn a
    sentence and its rationale: the first faithful run trained one step and was
    rejected for no_reask_improvement while still answering "Istanbul".
    """

    def test_the_experiment_budget_is_four_steps(self):
        from adaptible._src.wrap.repair import (
            _FLAGSHIP_MAX_STEPS,
            INITIAL_THINKING_STEPS,
        )

        self.assertEqual(_FLAGSHIP_MAX_STEPS, 4)
        self.assertNotEqual(_FLAGSHIP_MAX_STEPS, INITIAL_THINKING_STEPS)

    @unittest.skipUnless(
        importlib.util.find_spec("mlx"), "the experiment's constants import MLX"
    )
    def test_the_worker_stops_at_the_experiment_loss_floor(self):
        import inspect

        from adaptible._src.eval.harness import VERIFY_LOSS_FLOOR
        from adaptible._src.wrap.training_budget import fit_masked_target

        default = inspect.signature(fit_masked_target).parameters["target_loss"].default
        self.assertEqual(default, VERIFY_LOSS_FLOOR)


class FlagshipCandidateCountTest(unittest.IsolatedAsyncioTestCase):
    """The experiment tries K=2 candidates per cycle, each from fresh seeds.

    cycles_mlx.py trains each candidate from the restored snapshot and keeps
    the first that raises the item's score; a rejected candidate is discarded
    and the next tried, never continued. Drawing one candidate gives the item
    half the attempts the experiment gives it.
    """

    def test_the_experiment_draws_two_candidates(self):
        from adaptible._src.wrap.repair import (
            _FLAGSHIP_CANDIDATES,
            _FLAGSHIP_SAMPLES,
        )

        self.assertEqual(_FLAGSHIP_CANDIDATES, 2)
        self.assertEqual(_FLAGSHIP_SAMPLES, 6)

    async def test_each_candidate_uses_a_distinct_seed_range(self):
        import tempfile
        from pathlib import Path

        from adaptible._src.wrap.repair import _FLAGSHIP_SEED, Controller
        from adaptible._src.wrap.store import Store
        from adaptible.tests.wrap_test import FakeRuntime

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "seeds")
            try:
                controller = Controller(
                    FakeRuntime(), store, web_search=False, flagship_recipe=True
                )
                seeds = []

                async def complete(messages, **kwargs):
                    seeds.append(kwargs["seed"])
                    kwargs["details"].update(
                        content="", reasoning="", reasoning_prefix="", complete=False
                    )
                    return ""

                controller.runtime.complete = complete
                await controller.flagship_candidate("q", "note", "X", start=0)
                await controller.flagship_candidate("q", "note", "X", start=6)
                self.assertEqual(seeds[0], _FLAGSHIP_SEED)
                self.assertEqual(seeds[6], _FLAGSHIP_SEED + 6)
                self.assertEqual(len(set(seeds)), len(seeds))
            finally:
                store.close()


class FlagshipReaskSetTest(unittest.IsolatedAsyncioTestCase):
    """The keep decision must use the caller's phrasings and n > best_n alone."""

    def test_supplied_prompts_are_stored_and_returned(self):
        import tempfile
        from pathlib import Path

        from adaptible._src.wrap.store import Store

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "reask")
            try:
                idx = store.record([dict(role="user", content="Q")], "A")
                store.feedback(idx, True, "a note", reask_prompts=["P1", "P2", "P3"])
                row = store.pending()[0]
                self.assertEqual(row["reask_prompts"], ["P1", "P2", "P3"])
            finally:
                store.close()

    def test_the_keep_rule_is_score_alone_under_the_recipe(self):
        import inspect

        from adaptible._src.wrap import repair as repair_module

        source = inspect.getsource(repair_module)
        self.assertIn("controls_ok = score > best and (", source)
        self.assertIn("self.flagship_recipe\n", source)


class FlagshipJudgeTest(unittest.IsolatedAsyncioTestCase):
    """The keep decision must use cycles_mlx.ok, not the wrapper's strict judge.

    The wrapper's note-mode judge admits only the bare extracted name or a
    verbatim span of the reference, so "The capital of Morocco is Rabat." --
    an answer the experiment counts -- scored as a miss. That pinned the
    item's internal score at zero and made `n > best_n` unable to fire no
    matter how good the candidate was.
    """

    def controller(self, **kwargs):
        import tempfile
        from pathlib import Path

        from adaptible._src.wrap.repair import Controller
        from adaptible._src.wrap.store import Store
        from adaptible.tests.wrap_test import FakeRuntime

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        store = Store(Path(self.temp.name), "judge")
        self.addCleanup(store.close)
        return Controller(FakeRuntime(), store, web_search=False, **kwargs)

    async def test_a_prose_answer_naming_the_term_passes(self):
        c = self.controller(flagship_recipe=True)
        self.assertTrue(
            await c.judge(
                "What is the capital of Morocco?",
                "The capital of Morocco is Rabat. It is the administrative capital.",
                "Rabat",
                note="the correct answer is Rabat.",
            )
        )

    async def test_a_known_wrong_entity_still_fails(self):
        c = self.controller(flagship_recipe=True)
        self.assertFalse(
            await c.judge(
                "What is the nearest star to Earth?",
                "The nearest star is the Sun, though Proxima Centauri is closest to it.",
                "Sun",
                note="the correct answer is The Sun.",
                wrong=("Proxima", "Alpha Centauri"),
            )
        )

    async def test_an_answer_without_the_term_fails(self):
        c = self.controller(flagship_recipe=True)
        self.assertFalse(
            await c.judge(
                "What is the capital of Morocco?",
                "The capital of Morocco is Casablanca.",
                "Rabat",
                note="the correct answer is Rabat.",
            )
        )
