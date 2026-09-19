"""Thinking mode, final-only evaluation, and masked trace learning contracts."""

import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from adaptible.wrap.app import create_app
from adaptible.wrap.prompt_format import (
    render_ollama_qwen3,
    OLLAMA_QWEN3_FORMAT,
    OLLAMA_QWEN3_TEMPLATE_SHA256,
)
from adaptible.wrap.repair import Controller
from adaptible.wrap.retention import evaluate_retention, retention_regressions
from adaptible.wrap.runtime import Runtime
from adaptible.wrap.store import Store
from adaptible.wrap.thinking import (
    completion_details,
    generation_mode,
    thinking_complete,
)
from adaptible.wrap.training_examples import encode_examples
from adaptible.wrap.wrap_test import FakeRuntime, FakeTrainer


class ThinkingFormatTest(unittest.TestCase):
    def test_modes_default_true_explicit_false_and_conflicts(self):
        self.assertTrue(generation_mode({}, "qwen3")["thinking"])
        self.assertEqual(generation_mode({}, "qwen2"), {})
        self.assertFalse(
            generation_mode({"think": False}, "qwen3", native=True, path="/api/chat")[
                "thinking"
            ]
        )
        for body in (
            {"reasoning_effort": "none"},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ):
            self.assertFalse(generation_mode(body, "qwen3")["thinking"])
        self.assertIn(
            "error",
            generation_mode({"think": True, "reasoning_effort": "none"}, "qwen3"),
        )
        self.assertIn("error", generation_mode({"raw": True}, "qwen3"))

    def test_reasoning_fields_never_become_final_answer(self):
        for key in ("reasoning_content", "reasoning", "thinking"):
            details = completion_details(
                {"content": "Sydney", key: "The correct answer may be Canberra."},
                "stop",
            )
            self.assertEqual(details["content"], "Sydney")
            self.assertTrue(thinking_complete(details))
            self.assertIn("Canberra", details["reasoning"])
        self.assertEqual(
            completion_details(
                {"content": "<think>Canberra</think>\n\nSydney"}, "stop"
            )["content"],
            "Sydney",
        )
        self.assertFalse(
            thinking_complete(
                completion_details({"content": "<think>Canberra"}, "length")
            )
        )
        self.assertFalse(
            thinking_complete(completion_details({"content": "Canberra"}, "stop"))
        )
        self.assertFalse(
            thinking_complete(
                completion_details(
                    {"content": "Canberra", "reasoning": "trace"}, "length"
                )
            )
        )
        for text in (
            "<think>x</think><think>unfinished",
            "<think><think>x</think>A",
            "</think><think>x",
            "preamble<think>x</think>A",
        ):
            self.assertFalse(
                thinking_complete(completion_details({"content": text}, "stop"))
            )

    def test_stock_ollama_omits_prior_thoughts_like_native_template(self):
        final = [
            dict(role="assistant", content="Old answer"),
            dict(role="user", content="Next?"),
        ]
        expected = render_ollama_qwen3(final, thinking=True)
        for previous in (
            dict(role="assistant", content="<think>Old thought</think>\n\nOld answer"),
            dict(
                role="assistant", content="Old answer", reasoning_content="Old thought"
            ),
            dict(role="assistant", content="Old answer", thinking="Old thought"),
        ):
            messages = [previous, final[-1]]
            before = json.dumps(messages)
            self.assertEqual(render_ollama_qwen3(messages, thinking=True), expected)
            self.assertEqual(json.dumps(messages), before)
        self.assertIn("Next? /think", expected)
        self.assertNotIn("/no_think", expected)
        self.assertTrue(expected.endswith("<|im_start|>assistant\n"))

    def test_grounded_rationale_and_final_loss_with_answer_only_stopping_mask(self):
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import Qwen2Tokenizer

        tokenizer = Qwen2Tokenizer(
            vocab={
                v: i for i, v in enumerate(["<eos>", *sorted(ByteLevel.alphabet())])
            },
            merges=[],
            eos_token="<eos>",
            pad_token="<eos>",
        )
        template = "{% for m in messages %}{{m.role + ':' + m.content + '\\n'}}{% endfor %}{{ 'assistant:' }}{% if enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
        messages = [dict(role="user", content="Original?")]
        other = [dict(role="user", content="Variant?")]
        trace = "<think>\nA grounded corrected rationale supporting the answer.\n</think>\n\n"
        for native, prefilled in ((False, False), (False, True), (True, False)):
            tokenizer.chat_template = template + (
                "{{ '<think>\\n' }}" if prefilled else ""
            )
            options = dict(thinking=True, reasoning_prefix=trace)
            if native:
                options.update(
                    prompt_format=OLLAMA_QWEN3_FORMAT,
                    template_sha256=OLLAMA_QWEN3_TEMPLATE_SHA256,
                )
            job = dict(
                messages=messages,
                target="Correct",
                training_options=options,
                examples=[
                    dict(messages=other, target="Correct", reasoning_prefix=trace)
                ],
            )
            batch = encode_examples(tokenizer, job, "qwen3")
            target = tokenizer.encode("Correct<eos>", add_special_tokens=False)
            for tokens, labels, stop_labels in zip(
                batch["input_ids"], batch["labels"], batch["stop_labels"]
            ):
                active = stop_labels != -100
                self.assertEqual(stop_labels[active].tolist(), target)
                first = active.nonzero()[0].item()
                self.assertIn(trace, tokenizer.decode(tokens[:first]))
                self.assertEqual(tokenizer.decode(tokens[:first]).count("<think>"), 1)
                self.assertEqual(stop_labels[:first].tolist(), [-100] * first)
                supervised = tokenizer.decode(tokens[labels != -100])
                self.assertIn("grounded corrected rationale", supervised)
                self.assertIn("Correct", supervised)
                self.assertTrue((labels != -100).sum() > active.sum())
                # Causal shift trains the first final-answer token from the last
                # trace-boundary token; it never trains any reasoning token.
                self.assertEqual(
                    stop_labels[1:][stop_labels[1:] != -100].tolist(), target
                )
            broken = {**job, "examples": [dict(messages=other, target="Correct")]}
            with self.assertRaisesRegex(
                ValueError, "completed nonempty reasoning prefix"
            ):
                encode_examples(tokenizer, broken, "qwen3")

    def test_old_store_migrates_without_changing_legacy_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory), "identity")
            old = store.record([dict(role="user", content="Old")], "Answer")
            new = store.record(
                [dict(role="user", content="New")],
                "Answer",
                generation_mode={"thinking": True},
            )
            store.close()
            store = Store(Path(directory), "identity")
            try:
                self.assertEqual(store.rows()[0]["generation_mode"], {})
                self.assertTrue(store.rows()[1]["generation_mode"]["thinking"])
            finally:
                store.close()


class ReasoningRuntime(FakeRuntime):
    architecture = "qwen3"

    def __init__(self):
        super().__init__()
        self.trace_calls = []
        self.incomplete = False
        self.strip_candidate_thought = False

    async def complete(self, messages, handle=None, frozen=False, **kwargs):
        thinking = kwargs.pop("thinking", False)
        details = kwargs.pop("details", None)
        self.trace_calls.append((messages, thinking, frozen, handle))
        response = await super().complete(
            messages, handle=handle, frozen=frozen, **kwargs
        )
        if thinking and frozen:
            response = "Canberra."
        if details is not None:
            trace = (
                ""
                if self.strip_candidate_thought and handle
                else "A real prior thought about this question."
            )
            details.update(
                completion_details(
                    dict(content=response, reasoning=trace),
                    "length" if self.incomplete else "stop",
                )
            )
        return response


class ThinkingControllerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = ReasoningRuntime()
        self.store = Store(Path(self.temp.name), "thinking-test")
        self.trainer = FakeTrainer()
        self.trainer.train = AsyncMock(wraps=self.trainer.train)
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            documents={"Australia": "Canberra is the capital of Australia."},
            web_search=False,
        )
        self.messages = [dict(role="user", content="What is the capital of Australia?")]

    async def asyncTearDown(self):
        await self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    async def repair(self):
        idx = self.store.record(
            self.messages, "Sydney", generation_mode={"thinking": True}
        )
        return await self.controller.repair(self.store.rows()[-1])

    async def test_thinking_repair_conditions_on_trace_and_pins_mode(self):
        status, _ = await self.repair()
        self.assertEqual(status, "kept")
        kwargs = self.trainer.train.await_args.kwargs
        self.assertTrue(kwargs["training_options"]["thinking"])
        self.assertIn("</think>", kwargs["training_options"]["reasoning_prefix"])
        self.assertEqual(len(kwargs["examples"]), 2)
        self.assertTrue(all("reasoning_prefix" in item for item in kwargs["examples"]))
        ledger = self.store.repairs()[0]
        self.assertTrue(ledger["generation_mode"]["thinking"])
        heldout = ledger["heldout_prompts"][0]
        self.assertNotIn(heldout, [e["messages"] for e in kwargs["examples"]])
        self.assertTrue(
            all(
                thinking
                for messages, thinking, frozen, handle in self.runtime.trace_calls
                if handle
            )
        )
        self.assertTrue(
            any(
                thinking and frozen
                for messages, thinking, frozen, handle in self.runtime.trace_calls
            )
        )
        self.assertTrue(
            any(
                not thinking and frozen
                for messages, thinking, frozen, handle in self.runtime.trace_calls
            )
        )
        diagnostics = json.loads(
            next(
                (Path(self.temp.name) / "adapters").glob("*/validation.json")
            ).read_text()
        )
        self.assertTrue(diagnostics["generation_mode"]["thinking"])
        self.assertTrue(
            all(
                x["generation"]["reasoning"]
                for x in diagnostics["reasks"]["after_results"]
            )
        )
        audit = json.loads(
            next(
                (Path(self.temp.name) / "reviews").glob("*/thinking_drafts.json")
            ).read_text()
        )
        self.assertEqual(len(audit["drafts"]), 3)
        self.assertTrue(all(item["verdict"] == "passed" for item in audit["drafts"]))

    async def test_incomplete_pretraining_reasoning_skips_without_training(self):
        self.runtime.incomplete = True
        status, reason = await self.repair()
        self.assertEqual(status, "skipped")
        self.assertIn("complete grounded reasoning correction", reason)
        self.trainer.train.assert_not_awaited()
        audit = json.loads(
            next(
                (Path(self.temp.name) / "reviews").glob("*/thinking_drafts.json")
            ).read_text()
        )
        self.assertEqual(len(audit["drafts"]), 1)
        self.assertEqual(audit["drafts"][0]["verdict"], "truncated")
        self.assertTrue(audit["drafts"][0]["generation"]["reasoning"])

    async def test_candidate_cannot_pass_by_silently_disabling_thinking(self):
        self.runtime.strip_candidate_thought = True
        status, _ = await self.repair()
        self.assertEqual(status, "rejected")
        self.assertIsNone(self.store.get("accepted"))

    async def test_retention_uses_each_saved_repair_mode(self):
        records = [
            dict(
                interaction_idx=i,
                question="Question",
                messages=self.messages,
                expected="Sydney",
                generation_mode=mode,
            )
            for i, mode in enumerate(({}, {"thinking": True}), 1)
        ]
        judge = AsyncMock(return_value=True)
        results = await evaluate_retention(records, self.runtime, judge)
        self.assertEqual([x["thinking"] for x in results], [False, True])
        self.runtime.strip_candidate_thought = True
        candidate = await evaluate_retention(
            records, self.runtime, judge, handle="candidate"
        )
        self.assertEqual(
            [x["interaction_idx"] for x in retention_regressions(results, candidate)],
            [2],
        )


class ThinkingRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_separates_fields_and_sets_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime("base", directory, max_tokens=2048)
            runtime.architecture = "qwen3"
            runtime.url = "http://fake"
            runtime.payload = lambda body, **kwargs: body
            sent = []

            def respond(request):
                sent.append(json.loads(request.content))
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": "Sydney",
                                    "reasoning": "Canberra considered",
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )

            await runtime.client.aclose()
            runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            try:
                details = {}
                result = await runtime.complete(
                    [dict(role="user", content="Capital?")],
                    thinking=True,
                    details=details,
                )
                self.assertEqual(result, "Sydney")
                self.assertTrue(thinking_complete(details))
                self.assertEqual(sent[-1]["reasoning_effort"], "medium")
                self.assertTrue(sent[-1]["chat_template_kwargs"]["enable_thinking"])
                self.assertGreater(runtime.generation_timeout(True).read, 600)
                await runtime.complete([dict(role="user", content="Helper?")])
                self.assertEqual(sent[-1]["reasoning_effort"], "none")
            finally:
                await runtime.close()


class ThinkingAPITest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime()
        self.runtime.architecture = "qwen3"
        self.store = Store(Path(self.temp.name), "api-thinking")
        self.controller = Controller(
            self.runtime, self.store, trainer=FakeTrainer(), web_search=False
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )
        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond)
        )
        self.reasoning = "Consider Canberra, then conclude Sydney."
        self.content = "Sydney"
        self.finish_reason = "stop"
        self.inline = False
        self.sent = []

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    def respond(self, request):
        body = json.loads(request.content)
        self.sent.append(body)
        native = request.url.path.startswith("/api/")
        field = "thinking" if native else "reasoning"
        content = (
            f"<think>{self.reasoning}</think>\n\n{self.content}"
            if self.inline
            else self.content
        )
        message = dict(
            content=content, **({} if self.inline else {field: self.reasoning})
        )
        if body.get("stream"):
            if native:
                parts = [
                    dict(model="base", message=message, done=False),
                    dict(
                        model="base",
                        message=dict(content=""),
                        done=True,
                        done_reason=self.finish_reason,
                    ),
                ]
                return httpx.Response(
                    200, text="".join(json.dumps(p) + "\n" for p in parts)
                )
            if self.inline:
                chunks = [dict(content=content[:4]), dict(content=content[4:])]
            else:
                chunks = [dict(reasoning=self.reasoning), dict(content=content)]
            parts = [
                dict(model="base", choices=[dict(delta=delta)]) for delta in chunks
            ]
            parts.append(
                dict(
                    model="base",
                    choices=[dict(delta={}, finish_reason=self.finish_reason)],
                )
            )
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
                + "data: [DONE]\n\n",
            )
        if native:
            return httpx.Response(
                200,
                json=dict(
                    model="base",
                    message=message,
                    done=True,
                    done_reason=self.finish_reason,
                ),
            )
        return httpx.Response(
            200,
            json=dict(
                model="base",
                choices=[dict(message=message, finish_reason=self.finish_reason)],
            ),
        )

    async def test_terminal_stream_and_complete_preserve_thoughts_and_final_only_history(
        self,
    ):
        for native in (True, False):
            self.runtime.native = native
            for endpoint in ("/interact", "/stream_interact"):
                for inline in (True, False):
                    self.inline = inline
                    await self.client.post("/new_chat")
                    response = await self.client.post(
                        endpoint, json=dict(prompt="Capital?")
                    )
                    self.assertEqual(response.status_code, 200)
                    rendered = (
                        response.json()["response"]
                        if endpoint == "/interact"
                        else response.text
                    )
                    self.assertIn(self.reasoning, rendered)
                    self.assertTrue(rendered.endswith("Sydney"))
                    self.assertEqual(rendered.count("<think>"), 1)
                    self.assertEqual(rendered.count("</think>"), 1)
                    row = self.store.rows()[-1]
                    self.assertEqual(row["response"], "Sydney")
                    self.assertEqual(
                        row["response_details"]["reasoning"], self.reasoning
                    )
                    self.assertTrue(row["generation_mode"]["thinking"])
                    self.assertEqual(row["status"], "new")
                    await self.client.post("/interact", json=dict(prompt="Next?"))
                    history = self.sent[-1]["messages"]
                    self.assertEqual(
                        history[1], dict(role="assistant", content="Sydney")
                    )

    async def test_public_native_modes_and_reasoning_payload_remain_intact(self):
        self.runtime.native = True
        for endpoint, request in (
            ("/api/chat", dict(messages=[dict(role="user", content="Capital?")])),
            ("/api/generate", dict(prompt="Capital?")),
        ):
            for stream in (True, False):
                body = dict(model="local-model", think=True, stream=stream, **request)
                response = await self.client.post(endpoint, json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.sent[-1], {**body, "model": "base"})
                self.assertIn(self.reasoning, response.text)
                self.assertTrue(self.store.rows()[-1]["generation_mode"]["thinking"])
                self.assertEqual(self.store.rows()[-1]["response"], "Sydney")

    async def test_truncated_thoughts_are_visible_but_never_committed_as_history(self):
        self.content = ""
        self.finish_reason = "length"
        for endpoint in ("/interact", "/stream_interact"):
            await self.client.post("/new_chat")
            response = await self.client.post(endpoint, json=dict(prompt="Incomplete?"))
            rendered = (
                response.json()["response"]
                if endpoint == "/interact"
                else response.text
            )
            self.assertIn(self.reasoning, rendered)
            self.assertIn("<think>", rendered)
            self.assertNotIn("</think>", rendered)
            row = self.store.rows()[-1]
            self.assertEqual(row["status"], "incomplete")
            feedback = await self.client.post(
                "/feedback", json=dict(interaction_idx=row["id"], thumbs="down")
            )
            self.assertEqual(feedback.status_code, 409)
            await self.client.post(endpoint, json=dict(prompt="New question"))
            self.assertEqual(len(self.sent[-1]["messages"]), 1)

    async def test_mode_and_reasoning_details_survive_reopening_store(self):
        await self.client.post("/interact", json=dict(prompt="Capital?"))
        reopened = Store(Path(self.temp.name), "api-thinking")
        try:
            row = reopened.rows()[-1]
            self.assertTrue(row["generation_mode"]["thinking"])
            self.assertEqual(row["response_details"]["reasoning"], self.reasoning)
            self.assertEqual(row["response"], "Sydney")
        finally:
            reopened.close()


class GenerationTimeoutTest(unittest.TestCase):
    """The wait must follow the granted budget, not the declared mode.

    A distilled reasoning model on a non-Qwen3 architecture reasons on every
    call while the wrapper marks none of them as thinking. Deriving the timeout
    from the mode gave such a model 180s to produce 2048 tokens, and the review
    died on ReadTimeout before it ever trained.
    """

    def runtime(self, max_tokens):
        from adaptible.wrap.runtime import Runtime

        instance = Runtime.__new__(Runtime)
        instance.max_tokens = max_tokens
        return instance

    def test_a_large_budget_is_allowed_time_even_when_not_flagged_thinking(self):
        self.assertGreater(self.runtime(2048).generation_timeout().read, 600)

    def test_the_declared_mode_does_not_shorten_the_wait(self):
        for max_tokens in (160, 768, 2048, 8192):
            with self.subTest(max_tokens=max_tokens):
                instance = self.runtime(max_tokens)
                self.assertEqual(
                    instance.generation_timeout(False).read,
                    instance.generation_timeout(True).read,
                )

    def test_short_budgets_keep_a_floor_and_long_ones_stay_bounded(self):
        self.assertEqual(self.runtime(160).generation_timeout().read, 180)
        self.assertEqual(self.runtime(100000).generation_timeout().read, 1800)
