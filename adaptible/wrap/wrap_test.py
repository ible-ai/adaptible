"""Wrapper API and repair contracts, with no pretrained checkpoint."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from adaptible.wrap.app import create_app
from adaptible.wrap.repair import Controller, answer
from adaptible.wrap.store import Store


class FakeRuntime:
    name = "local-model"
    native = True
    url = "http://runtime"
    max_tokens = 64
    blob = Path("existing-model.gguf")

    def __init__(self):
        self.active = None
        self.break_control = False
        self.suspended = False
        self.sent = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))

    def payload(self, body, handle=None, frozen=False):
        return {**body, "model": "base" if frozen else handle or self.active or "base"}

    def respond(self, request):
        body = json.loads(request.content)
        self.sent.append(body)
        text = "Sydney" if body["model"] == "base" else "Canberra"
        if body.get("stream"):
            if request.url.path.startswith("/api/"):
                parts = [
                    dict(model=body["model"], message=dict(content=t), done=False)
                    for t in (text[:2], text[2:])
                ]
                parts.append(
                    dict(model=body["model"], message=dict(content=""), done=True)
                )
                return httpx.Response(
                    200, text="".join(json.dumps(p) + "\n" for p in parts)
                )
            parts = [
                dict(model=body["model"], choices=[dict(delta=dict(content=t))])
                for t in (text[:2], text[2:])
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(p) + "\n\n" for p in parts)
                + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json=dict(
                model=body["model"],
                choices=[dict(message=dict(content=text))],
                message=dict(content=text),
            ),
        )

    async def complete(self, messages, handle=None, frozen=False, **kwargs):
        assert not self.suspended, "Generation attempted before serving resumed"
        prompt = messages[-1]["content"]
        if (
            kwargs.get("response_format", {}).get("json_schema", {}).get("name")
            == "evidence_check"
        ):
            return json.dumps(dict(answers_question=True, supported=True))
        if prompt.startswith("Reference sentences:"):
            reference = prompt.split("\nQuestion:", 1)[0]
            for line in reference.splitlines()[1:]:
                index, sentence = line.split(": ", 1)
                for name in ("Canberra", "Sydney", "Paris", "London", "144", "7"):
                    if name in sentence:
                        return json.dumps(dict(name=name, sentence_index=int(index)))
            return json.dumps(dict(name="", sentence_index=-1))
        if prompt.startswith("Reference text:"):
            reference = prompt.split("\nQuestion:", 1)[0]
            for name in ("Canberra", "Sydney", "Paris", "London", "144", "7"):
                if name in reference:
                    return json.dumps(
                        dict(
                            quote=reference.removeprefix("Reference text: "), name=name
                        )
                    )
            return ""
        if (
            kwargs.get("response_format", {}).get("json_schema", {}).get("name")
            == "question_variants"
        ):
            return json.dumps(
                dict(
                    questions=[
                        "Which city is Australia's capital?",
                        "Name the capital city of Australia.",
                        "Which city serves as Australia's capital?",
                    ]
                )
            )
        if prompt.startswith("Reference note:"):
            return "Canberra."
        if "12 times" in prompt:
            return "144."
        if "France" in prompt:
            return "London." if handle and self.break_control else "Paris."
        if "days" in prompt:
            return "7."
        return "Canberra." if handle or self.active else "Sydney."

    async def stage(self, directory):
        self.suspended = False
        return str(directory)

    async def restore(self, handle):
        self.suspended = False
        self.active = handle

    async def release_for_training(self):
        self.suspended = True

    async def discard(self, handle):
        pass

    async def close(self):
        await self.client.aclose()


class FakeTrainer:
    def __init__(self):
        self.calls = []
        self.examples = []
        self.budgets = []
        self.fail = False
        self.runtime = None

    async def train(
        self,
        blob,
        messages,
        target,
        directory,
        previous=None,
        *,
        examples=None,
        training_options=None,
        resume_from=None,
        max_total_steps=None,
    ):
        self.budgets.append(max_total_steps)
        if self.runtime:
            assert self.runtime.suspended, "Serving weights still resident at training"
        self.calls.append((blob, messages, target, previous))
        self.examples.append(examples or [])
        if self.fail:
            raise RuntimeError("injected training failure")
        directory.mkdir(parents=True)
        return dict(steps=1, loss=0.1)


class WrapperTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime()
        self.store = Store(Path(self.temp.name), "test-identity")
        self.trainer = FakeTrainer()
        self.trainer.runtime = self.runtime
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            idle_seconds=60,
            documents={"Australia": "Canberra is the capital of Australia."},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    async def flag(self):
        response = await self.client.post(
            "/interact", json={"prompt": "What is the capital of Australia?"}
        )
        self.assertEqual(response.status_code, 200)
        idx = response.json()["interaction_idx"]
        response = await self.client.post(
            "/feedback", json=dict(interaction_idx=idx, thumbs="down")
        )
        self.assertEqual(response.status_code, 200)
        return idx

    async def test_later_repair_cannot_erase_previously_accepted_answer(self):
        prior_question = "What was the earlier answer?"
        prior_messages = [dict(role="user", content=prior_question)]
        prior_id = self.store.record(prior_messages, "Paris.")
        prior = dict(directory=str(Path(self.temp.name) / "prior"))
        self.store.accept_repair(
            prior,
            dict(
                interaction_idx=prior_id,
                question=prior_question,
                messages=prior_messages,
                prompts=[prior_messages],
                expected="Paris",
                note="The earlier answer was Paris.",
            ),
        )
        self.runtime.active = prior["directory"]
        original = self.runtime.complete

        async def complete(messages, *, handle=None, **kwargs):
            if messages == prior_messages:
                return "London." if handle else "Paris."
            # The new question is wrong under the current adapter, so the
            # candidate really improves it while forgetting the earlier repair.
            if messages[-1]["content"] in (
                "What is the capital of Australia?",
                "Which city is Australia's capital?",
            ):
                return "Canberra." if handle else "Sydney."
            return await original(messages, handle=handle, **kwargs)

        self.runtime.complete = complete
        idx = await self.flag()
        await self.client.post("/trigger_review")
        await self.client.get("/sync")
        row = next(r for r in self.store.rows() if r["id"] == idx)
        self.assertEqual(row["status"], "rejected")
        self.assertTrue(self.trainer.calls)
        self.assertEqual(self.store.get("accepted"), prior)
        self.assertEqual(self.runtime.active, prior["directory"])
        self.assertEqual(len(self.store.repairs()), 1)
        diagnostic = self.validation_diagnostic()
        self.assertIn("prior_repair_regression", diagnostic["failure_categories"])
        self.assertEqual(diagnostic["retention"]["before"][0]["response"], "Paris.")
        self.assertEqual(diagnostic["retention"]["after"][0]["response"], "London.")
        self.assertIn(f"interaction {prior_id}", row["reason"])

    def validation_diagnostic(self):
        files = list((Path(self.temp.name) / "adapters").glob("*/validation.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    async def test_validation_records_no_improvement_without_extra_calls(self):
        original = self.runtime.complete
        calls = []

        async def complete(messages, *, handle=None, **kwargs):
            response = await original(messages, handle=handle, **kwargs)
            if handle and response == "Canberra.":
                response = "Sydney."
            calls.append((messages, handle, response))
            return response

        self.runtime.complete = complete
        await self.flag()
        await self.client.post("/trigger_review")
        result = (await self.client.get("/sync")).json()
        diagnostic = self.validation_diagnostic()
        self.assertEqual(diagnostic["failure_categories"], ["no_reask_improvement"])
        self.assertIn("no re-ask improvement", result["reviews"][0]["reason"])
        self.assertEqual(diagnostic["controls"]["status"], "not_run")
        self.assertEqual(diagnostic["retention"]["status"], "not_run")
        for row in diagnostic["reasks"]["after_results"]:
            self.assertEqual(row["response"], "Sydney.")
            self.assertFalse(row["passed"])
            self.assertEqual(row["expected"], "Canberra")
            self.assertEqual(
                sum(
                    messages == row["messages"] and handle is not None
                    for messages, handle, response in calls
                ),
                1,
            )

    async def test_validation_identifies_lost_current_reask_despite_total_gain(self):
        original = self.runtime.complete
        first = "What is the capital of Australia?"

        async def complete(messages, *, handle=None, **kwargs):
            if messages[-1]["content"] == first and not kwargs.get("frozen"):
                return "Sydney." if handle else "Canberra."
            return await original(messages, handle=handle, **kwargs)

        self.runtime.complete = complete
        await self.flag()
        await self.client.post("/trigger_review")
        result = (await self.client.get("/sync")).json()
        diagnostic = self.validation_diagnostic()
        # The candidate gains overall while making the flagged answer wrong,
        # so both faults are reported and neither alone would keep it.
        self.assertEqual(
            diagnostic["failure_categories"],
            ["flagged_answer_unfixed", "current_reask_regression"],
        )
        self.assertGreater(
            sum(diagnostic["reasks"]["after"]), sum(diagnostic["reasks"]["before"])
        )
        self.assertIn(
            "lost previously passing current re-ask", result["reviews"][0]["reason"]
        )
        self.assertIsNone(self.runtime.active)

    async def test_kept_validation_preserves_heldout_training_boundary(self):
        await self.flag()
        await self.client.post("/trigger_review")
        await self.client.get("/sync")
        diagnostic = self.validation_diagnostic()
        self.assertEqual(diagnostic["decision"], "kept")
        self.assertEqual(diagnostic["failure_categories"], [])
        self.assertEqual(diagnostic["controls"]["status"], "passed")
        self.assertEqual(diagnostic["retention"]["status"], "passed")
        self.assertEqual(len(diagnostic["reasks"]["after_results"]), 4)
        heldout = diagnostic["reasks"]["after_results"][-1]["messages"]
        self.assertNotIn(
            heldout, [example["messages"] for example in self.trainer.examples[0]]
        )

    async def test_api_preserves_options_and_records_stream(self):
        response = await self.client.post(
            "/v1/chat/completions",
            json=dict(
                model="local-model",
                messages=[dict(role="user", content="Hi")],
                stream=True,
                temperature=0.25,
                stop=["!"],
                tools=[dict(type="function", function=dict(name="lookup"))],
            ),
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('"model": "local-model"', response.text)
        self.assertNotIn('"model": "base"', response.text)
        self.assertEqual(self.runtime.sent[-1]["temperature"], 0.25)
        self.assertEqual(self.runtime.sent[-1]["stop"], ["!"])
        self.assertEqual(
            self.runtime.sent[-1]["tools"][0]["function"]["name"], "lookup"
        )
        self.assertEqual(self.store.rows()[0]["response"], "Sydney")
        self.assertEqual(
            str(self.store.rows()[0]["id"]), response.headers["X-Interaction-Idx"]
        )

    async def test_tool_call_response_with_null_content_is_forwarded(self):
        tools = [
            dict(
                id="call_1",
                type="function",
                function=dict(name="lookup", arguments='{"city":"Sydney"}'),
            )
        ]
        result = dict(
            model="base",
            choices=[
                dict(
                    index=0,
                    finish_reason="tool_calls",
                    message=dict(role="assistant", content=None, tool_calls=tools),
                )
            ],
        )
        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=result)
            )
        )
        response = await self.client.post(
            "/v1/chat/completions",
            json=dict(
                model="local-model",
                messages=[dict(role="user", content="Look up Sydney")],
            ),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["choices"][0]["message"]["content"])
        self.assertEqual(data["choices"][0]["message"]["tool_calls"], tools)
        self.assertEqual(data["model"], "local-model")
        self.assertEqual(self.store.rows()[0]["response"], "")
        self.assertEqual(self.store.rows()[0]["status"], "new")

    async def test_truncated_stream_cannot_be_used_as_training_feedback(self):
        for path in ("/v1/chat/completions", "/api/chat"):
            with self.subTest(path=path):
                native = path.startswith("/api/")
                chunk = (
                    dict(model="base", message=dict(content="partial"), done=False)
                    if native
                    else dict(
                        model="base", choices=[dict(delta=dict(content="partial"))]
                    )
                )
                wire = (
                    json.dumps(chunk) + "\n"
                    if native
                    else "data: " + json.dumps(chunk) + "\n\n"
                )
                await self.runtime.client.aclose()
                self.runtime.client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, text=wire)
                    )
                )
                response = await self.client.post(
                    path,
                    json=dict(
                        model="local-model",
                        stream=True,
                        messages=[dict(role="user", content="Question")],
                    ),
                )
                idx = int(response.headers["X-Interaction-Idx"])
                row = next(r for r in self.store.rows() if r["id"] == idx)
                self.assertEqual(row["response"], "partial")
                self.assertEqual(row["status"], "incomplete")
                self.assertFalse(self.controller.gate.locked())
                feedback = await self.client.post(
                    "/feedback", json=dict(interaction_idx=idx, thumbs="down")
                )
                self.assertEqual(feedback.status_code, 409)
                self.assertFalse(self.store.pending())

    async def test_feedback_train_accept_and_restart(self):
        idx = await self.flag()
        await self.client.post("/trigger_review")
        response = await self.client.get("/sync")
        self.assertEqual(response.json()["reviews"][0]["status"], "kept")
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertEqual(self.store.rows()[0]["status"], "kept")
        accepted = self.store.get("accepted")
        self.assertEqual(len(self.trainer.examples[0]), 2)
        ledger = self.store.repairs()[0]
        self.assertEqual(len(ledger["heldout_prompts"]), 1)
        self.assertNotIn(
            ledger["heldout_prompts"][0],
            [e["messages"] for e in self.trainer.examples[0]],
        )
        response = await self.client.post(
            "/interact",
            json=dict(prompt="What is the capital of Australia?", use_history=False),
        )
        self.assertEqual(response.json()["response"], "Canberra")
        self.assertNotEqual(response.headers["X-Adaptible-Adapter"], "base")
        ambiguous = await self.client.post(
            "/interact", json=dict(prompt="Capital?", use_history=False)
        )
        self.assertEqual(ambiguous.headers["X-Adaptible-Adapter"], "base")
        # Reopen only our store, and load the accepted adapter into a fresh runtime.
        store = Store(Path(self.temp.name), "test-identity")
        runtime = FakeRuntime()
        controller = Controller(runtime, store, trainer=FakeTrainer())
        try:
            await controller.start()
            self.assertEqual(runtime.active, accepted["directory"])
            self.assertEqual(store.rows()[0]["id"], idx)
        finally:
            await controller.close()

    async def check_retirement(self, fail):
        previous = str(Path(self.temp.name).resolve() / "previous")
        self.runtime.active = previous
        self.store.set("accepted", dict(directory=previous))

        async def discard(handle):
            self.assertEqual(handle, previous)
            self.assertNotEqual(self.runtime.active, previous)
            self.assertEqual(
                self.store.get("accepted")["directory"],
                str(Path(self.runtime.active).resolve()),
            )
            if fail:
                raise RuntimeError("injected retirement failure")

        self.runtime.discard = mock.AsyncMock(side_effect=discard)
        # This tests commit/retirement ordering independently of model grading.
        with mock.patch.object(
            self.controller,
            "checks",
            new=mock.AsyncMock(side_effect=[[False, False], [True, True]]),
        ):
            await self.flag()
            await self.client.post("/trigger_review")
            result = (await self.client.get("/sync")).json()
        self.assertEqual(result["reviews"][0]["status"], "kept")
        self.assertIsNone(self.controller.failed)
        self.runtime.discard.assert_awaited_once_with(previous)
        self.assertEqual(self.trainer.calls[0][-1], str(Path(previous) / "adapter"))
        self.assertNotEqual(self.store.get("accepted")["directory"], previous)

    async def test_previous_adapter_is_retired_only_after_durable_acceptance(self):
        await self.check_retirement(False)

    async def test_retirement_failure_does_not_undo_an_accepted_adapter(self):
        with self.assertLogs("adaptible.wrap.repair", level="WARNING"):
            await self.check_retirement(True)

    async def test_control_regression_rejects_candidate_and_restores(self):
        self.runtime.break_control = True
        await self.flag()
        await self.client.post("/trigger_review")
        result = (await self.client.get("/sync")).json()
        self.assertEqual(result["reviews"][0]["status"], "rejected")
        self.assertIsNone(self.runtime.active)
        self.assertFalse(self.runtime.suspended)
        self.assertIsNone(self.store.get("accepted"))
        self.assertEqual(len(self.trainer.calls), 1)
        diagnostic = self.validation_diagnostic()
        self.assertEqual(diagnostic["failure_categories"], ["control_regression"])
        self.assertEqual(diagnostic["controls"]["status"], "failed_short_circuit")
        self.assertEqual(diagnostic["controls"]["after"][-1]["response"], "London.")
        self.assertEqual(diagnostic["controls"]["after"][-1]["expected"], "Paris")
        self.assertIn("capital of France", result["reviews"][0]["reason"])

    async def test_unload_failure_skips_training_and_restores_serving(self):
        async def fail_unload():
            self.runtime.suspended = True
            raise RuntimeError("injected unload failure")

        self.runtime.release_for_training = fail_unload
        await self.flag()
        await self.client.post("/trigger_review")
        result = (await self.client.get("/sync")).json()
        self.assertEqual(result["reviews"][0]["status"], "failed")
        self.assertFalse(self.trainer.calls)
        self.assertFalse(self.runtime.suspended)
        response = await self.client.post("/interact", json=dict(prompt="Hello"))
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.store.get("accepted"))

    async def test_training_failure_is_reported_and_does_not_activate(self):
        self.trainer.fail = True
        await self.flag()
        await self.client.post("/trigger_review")
        result = (await self.client.get("/sync")).json()
        self.assertEqual(result["reviews"][0]["status"], "failed")
        self.assertIn("injected training failure", result["reviews"][0]["reason"])
        self.assertIsNone(self.runtime.active)
        self.assertFalse(self.runtime.suspended)

    async def test_shutdown_cancels_training_without_restarting_serving(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_training(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.trainer.train = blocked_training
        self.runtime.restore = mock.AsyncMock()
        self.runtime.close = mock.AsyncMock(wraps=self.runtime.close)
        self.store.close = mock.Mock(wraps=self.store.close)
        idx = await self.flag()
        await self.client.post("/trigger_review")
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(self.controller.close(), 1)
        self.assertTrue(cancelled.is_set())
        self.runtime.restore.assert_not_awaited()
        self.runtime.close.assert_awaited_once()
        self.store.close.assert_called_once()
        self.assertFalse(self.controller.gate.locked())
        # The interrupted review is durable and can be retried on next startup.
        reopened = Store(Path(self.temp.name), "test-identity")
        try:
            self.assertEqual([row["id"] for row in reopened.pending()], [idx])
        finally:
            reopened.close()

    async def test_idle_feedback_needs_no_manual_review(self):
        self.controller.idle_seconds = 0
        await self.flag()
        await asyncio.wait_for(self.controller.task, 2)
        self.assertEqual(self.store.rows()[0]["status"], "kept")

    async def test_queued_feedback_uses_latest_rating_and_reference(self):
        for thumbs in ("up", "down"):
            with self.subTest(thumbs=thumbs):
                started, release = asyncio.Event(), asyncio.Event()
                seen = []
                ids = [
                    self.store.record([dict(role="user", content="Question")], "Answer")
                    for _ in range(2)
                ]
                for idx in ids:
                    self.store.feedback(idx, True, "original note")

                async def review(row):
                    seen.append((row["id"], row["note"]))
                    if row["id"] == ids[0]:
                        started.set()
                        await release.wait()
                    return "skipped", "scheduler fixture"

                with mock.patch.object(self.controller, "repair", new=review):
                    self.controller.schedule(immediate=True)
                    await asyncio.wait_for(started.wait(), 1)
                    response = await self.client.post(
                        "/feedback",
                        json=dict(
                            interaction_idx=ids[1], thumbs=thumbs, note="updated note"
                        ),
                    )
                    self.assertEqual(response.status_code, 200)
                    release.set()
                    await asyncio.wait_for(self.controller.sync(), 1)
                expected = [(ids[0], "original note")]
                if thumbs == "down":
                    expected.append((ids[1], "updated note"))
                self.assertEqual(seen, expected)

    async def test_status_remains_responsive_while_model_is_busy(self):
        async with self.controller.gate:
            response = await asyncio.wait_for(self.client.get("/status"), 0.5)
        self.assertEqual(response.status_code, 200)

    async def test_new_chat_isolates_cli_history(self):
        await self.client.post("/interact", json=dict(prompt="First"))
        await self.client.post("/interact", json=dict(prompt="Second"))
        self.assertEqual(len(self.runtime.sent[-1]["messages"]), 3)
        await self.client.post("/new_chat")
        await self.client.post("/interact", json=dict(prompt="Third"))
        self.assertEqual(len(self.runtime.sent[-1]["messages"]), 1)
        self.assertEqual(len(self.store.rows()), 3)

    async def test_qwen3_terminal_requests_enable_thinking(self):
        self.runtime.architecture = "qwen3"
        for native in (True, False):
            self.runtime.native = native
            for endpoint in ("/interact", "/stream_interact"):
                with self.subTest(native=native, endpoint=endpoint):
                    response = await self.client.post(
                        endpoint, json=dict(prompt="Hello", use_history=False)
                    )
                    self.assertEqual(response.status_code, 200)
                    sent = self.runtime.sent[-1]
                    self.assertEqual(sent["reasoning_effort"], "medium")
                    if native:
                        self.assertNotIn("chat_template_kwargs", sent)
                    else:
                        self.assertEqual(
                            sent["chat_template_kwargs"], {"enable_thinking": True}
                        )
                    self.assertEqual(self.store.rows()[-1]["response"], "Sydney")

    async def test_qwen2_terminal_generation_settings_unchanged(self):
        self.runtime.architecture = "qwen2"
        for native in (True, False):
            self.runtime.native = native
            for endpoint in ("/interact", "/stream_interact"):
                with self.subTest(native=native, endpoint=endpoint):
                    response = await self.client.post(
                        endpoint, json=dict(prompt="Hello", use_history=False)
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("reasoning_effort", self.runtime.sent[-1])
                    self.assertNotIn("chat_template_kwargs", self.runtime.sent[-1])

    async def test_qwen3_public_chat_preserves_explicit_and_default_modes(self):
        self.runtime.architecture = "qwen3"
        for native in (True, False):
            self.runtime.native = native
            for settings in (
                {},
                dict(
                    reasoning_effort="high",
                    chat_template_kwargs={"enable_thinking": True},
                ),
            ):
                for stream in (False, True):
                    with self.subTest(native=native, settings=settings, stream=stream):
                        body = dict(
                            model="local-model",
                            messages=[dict(role="user", content="Hello")],
                            stream=stream,
                            **settings,
                        )
                        response = await self.client.post(
                            "/v1/chat/completions", json=body
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(
                            self.runtime.sent[-1], {**body, "model": "base"}
                        )

    async def test_qwen3_native_chat_preserves_explicit_thinking(self):
        self.runtime.architecture = "qwen3"
        for endpoint, content in (
            ("/api/chat", dict(messages=[dict(role="user", content="Hello")])),
            ("/api/generate", dict(prompt="Hello")),
        ):
            with self.subTest(endpoint=endpoint):
                body = dict(model="local-model", stream=False, think=True, **content)
                response = await self.client.post(endpoint, json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.runtime.sent[-1], {**body, "model": "base"})

    async def test_other_model_is_not_forwarded(self):
        response = await self.client.post(
            "/v1/chat/completions",
            json=dict(model="other", messages=[dict(role="user", content="Hi")]),
        )
        self.assertEqual(response.status_code, 404)
        self.assertFalse(self.runtime.sent)

    async def test_wrong_model_cannot_resume_a_state_directory(self):
        with self.assertRaisesRegex(ValueError, "different model"):
            Store(Path(self.temp.name), "different-identity")

    def test_unclosed_reasoning_is_not_an_answer(self):
        self.assertEqual(answer("<think>Canberra"), "")
        self.assertEqual(answer("thinking</think>Canberra"), "Canberra")

    async def test_native_stream_uses_public_model_and_records_answer(self):
        response = await self.client.post(
            "/api/chat",
            json=dict(
                model="local-model", messages=[dict(role="user", content="Hello")]
            ),
        )
        self.assertEqual(response.status_code, 200)
        chunks = [json.loads(line) for line in response.text.splitlines()]
        self.assertTrue(all(c["model"] == "local-model" for c in chunks))
        self.assertEqual(self.store.rows()[0]["response"], "Sydney")

    async def test_fabricated_quote_is_not_grounding(self):
        original = self.runtime.complete

        async def fabricate(*args, **kwargs):
            return json.dumps(dict(quote="Canberra is on Mars.", name="Canberra"))

        self.runtime.complete = fabricate
        try:
            self.assertEqual(
                await self.controller.grounded(
                    "Capital?",
                    "Canberra is the capital of Australia.",
                    expected="Canberra",
                ),
                {},
            )
        finally:
            self.runtime.complete = original

    async def test_feedback_during_incomplete_stream_is_rejected(self):
        idx = self.store.record([dict(role="user", content="Question")], "")
        self.store.outcome(idx, "streaming", "")
        response = await self.client.post(
            "/feedback", json=dict(interaction_idx=idx, thumbs="down")
        )
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.store.pending())

    async def test_feedback_on_a_turn_that_ran_out_of_tokens_is_accepted(self):
        # cycles_mlx.py scores a truncated generation as a miss and still goes
        # on to sample and train. Refusing feedback here meant an item the
        # model answered badly enough to run away could never be repaired.
        idx = self.store.record([dict(role="user", content="Question")], "")
        with self.store.db:
            self.store.db.execute(
                "UPDATE interactions SET status=?, response_details=? WHERE id=?",
                (
                    "incomplete",
                    json.dumps(dict(finish_reason="length", content="a bad answer")),
                    idx,
                ),
            )
        response = await self.client.post(
            "/feedback", json=dict(interaction_idx=idx, thumbs="down", note="a note")
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.store.pending())

    async def test_import_wrapper_does_not_load_mlx_or_torch(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import adaptible.wrap.app; assert 'mlx' not in sys.modules; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class RuntimeMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_ollama_unloads_only_selected_base_and_active_adapter(self):
        from adaptible.wrap.runtime import Ollama

        loaded = {"base:latest", "private-adapter:latest", "unrelated:latest"}
        unloaded = []

        def respond(request):
            if request.url.path == "/api/ps":
                return httpx.Response(
                    200, json={"models": [{"name": n} for n in loaded]}
                )
            self.assertEqual(request.url.path, "/api/generate")
            body = json.loads(request.content)
            self.assertEqual(body["keep_alive"], 0)
            self.assertEqual(body["prompt"], "")
            unloaded.append(body["model"])
            loaded.remove(body["model"])
            return httpx.Response(200, json={"done": True})

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "adaptible.wrap.runtime.shutil.which", return_value="ollama"
            ),
        ):
            runtime = Ollama("base", Path(directory))
            await runtime.client.aclose()
            runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            runtime.active = "private-adapter"
            try:
                await runtime.release_for_training()
                self.assertEqual(
                    set(unloaded), {"base:latest", "private-adapter:latest"}
                )
                self.assertEqual(loaded, {"unrelated:latest"})
                self.assertEqual(runtime.active, "private-adapter")
                self.assertEqual(runtime.payload({})["model"], "private-adapter")
                candidate = runtime.prefix + "candidate"
                loaded.update({candidate + ":latest", "base:latest"})
                runtime.active = candidate
                await runtime.discard(candidate)
                self.assertIn(candidate + ":latest", loaded)
                runtime.active = "private-adapter"

                async def remove(*args, **kwargs):
                    self.assertEqual(args[1:], ("rm", candidate))
                    self.assertNotIn(candidate + ":latest", loaded)

                with mock.patch("adaptible.wrap.runtime.run_command", new=remove):
                    await runtime.discard(candidate)
                self.assertEqual(loaded, {"base:latest", "unrelated:latest"})
            finally:
                await runtime.close()

    async def test_llama_cpp_stops_child_and_reloads_previous_adapter(self):
        from adaptible.wrap.runtime import LlamaCpp

        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "base.gguf"
            blob.touch()
            runtime = LlamaCpp(blob, Path(directory), executable="llama-server")
            child = mock.Mock(returncode=None, wait=mock.AsyncMock())
            runtime.process = child
            runtime.loaded = runtime.active = "previous.gguf"
            runtime.launch = mock.AsyncMock()
            try:
                await runtime.release_for_training()
                child.terminate.assert_called_once()
                child.wait.assert_awaited_once()
                self.assertIsNone(runtime.process)
                await runtime.restore("previous.gguf")
                runtime.launch.assert_awaited_once_with("previous.gguf")
                self.assertEqual(runtime.active, "previous.gguf")
            finally:
                await runtime.close()


class TrainerPrecisionTest(unittest.TestCase):
    """The trainer must train in the checkpoint's precision, not the device's.

    `dtype = torch.bfloat16 if device != "cpu" else torch.float32` picked bf16
    whenever an accelerator was present, so the wrapper trained a different
    adapter from the same example: bf16 keeps 7 mantissa bits against f32's 23.
    On one cycle of the flagship recipe the step-0 answer-token loss -- the
    base model's, since LoRA's B is zero-initialised and contributes nothing to
    the first forward -- was 0.07825317 against the original's 0.07735140 on
    identical tokens and an identical mask, and the adapter that came out of it
    gave different post-training answers on 3 of 4 prompts. Same defect as
    vLLM's unpinned `--dtype`, in the trainer.
    """

    def test_an_f32_checkpoint_trains_in_f32(self):
        import torch

        from adaptible.wrap.train import checkpoint_dtype

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
            self.assertIs(checkpoint_dtype(source, {}), torch.float32)

    def test_a_bf16_checkpoint_is_honoured_too(self):
        import torch

        from adaptible.wrap.train import checkpoint_dtype

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
            self.assertIs(checkpoint_dtype(source, {}), torch.bfloat16)

    def test_gguf_and_undeclared_sources_fall_back_to_f32_not_bf16(self):
        import torch

        from adaptible.wrap.train import checkpoint_dtype

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            self.assertIs(checkpoint_dtype(source, {"gguf_file": "m.gguf"}), torch.float32)
            self.assertIs(checkpoint_dtype(source, {}), torch.float32)
            (source / "config.json").write_text(json.dumps({"torch_dtype": "nonsense"}))
            self.assertIs(checkpoint_dtype(source, {}), torch.float32)
