"""Stock binary-thinking compatibility on real wrapper routes, without a model."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import httpx

from adaptible._src.wrap.app import create_app
from adaptible._src.wrap.prompt_format import OLLAMA_QWEN3_TEMPLATE_SHA256
from adaptible._src.wrap.repair import Controller
from adaptible._src.wrap.runtime import Ollama
from adaptible._src.wrap.store import Store

TEMPLATE = (Path(__file__).parent / "fixtures/ollama_qwen3_template.txt").read_text()


class OllamaThinkingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        with mock.patch(
            "adaptible._src.wrap.runtime.shutil.which", return_value="ollama"
        ):
            self.runtime = Ollama(
                "test-model", self.root / "state", url="http://native.fixture"
            )
        self.runtime.architecture = "qwen3"
        self.runtime.template = TEMPLATE
        self.runtime.base_handle = "owned-context-8192"
        self.assertEqual(
            hashlib.sha256(TEMPLATE.encode()).hexdigest(), OLLAMA_QWEN3_TEMPLATE_SHA256
        )
        self.requests = []
        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond)
        )
        self.store = Store(self.root / "history", "fixture")
        self.controller = Controller(self.runtime, self.store, web_search=False)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper.fixture",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if body.get("stream"):
            chunks = [
                dict(delta=dict(reasoning="Consider evidence."), finish_reason=None),
                dict(delta=dict(content="Rabat"), finish_reason="stop"),
            ]
            return httpx.Response(
                200,
                text="".join(
                    "data: "
                    + json.dumps(dict(model=body["model"], choices=[c]))
                    + "\n\n"
                    for c in chunks
                )
                + "data: [DONE]\n\n",
            )
        if request.url.path == "/api/generate":
            return httpx.Response(
                200,
                json=dict(
                    model=body["model"],
                    response="Rabat",
                    thinking="Consider evidence.",
                    done=True,
                    done_reason="stop",
                ),
            )
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json=dict(
                    model=body["model"],
                    message=dict(
                        role="assistant", content="Rabat", thinking="Consider evidence."
                    ),
                    done=True,
                    done_reason="stop",
                ),
            )
        return httpx.Response(
            200,
            json=dict(
                model=body["model"],
                choices=[
                    dict(
                        message=dict(
                            role="assistant",
                            content="Rabat",
                            reasoning="Consider evidence.",
                        ),
                        finish_reason="stop",
                    )
                ],
            ),
        )

    async def test_flat_and_nested_levels_translate_without_input_mutation(self):
        for effort in ("low", "medium", "high"):
            for options in (
                {"reasoning_effort": effort},
                {"reasoning": {"effort": effort}},
                {"reasoning_effort": effort, "reasoning": {"effort": "high"}},
            ):
                body = dict(
                    messages=[dict(role="user", content="A question")],
                    temperature=0.7,
                    seed=14,
                    **options,
                )
                before = copy.deepcopy(body)
                translated = self.runtime.normalize_payload(body)
                self.assertNotIn("reasoning_effort", translated)
                self.assertNotIn("reasoning", translated)
                self.assertEqual(body, before)
                self.assertEqual(translated["temperature"], 0.7)
                self.assertEqual(translated["seed"], 14)

    async def test_native_levels_translate_to_true_but_booleans_stay(self):
        for path in ("/api/chat", "/api/generate"):
            for value in ("low", "medium", "high", True, False):
                body = {"think": value}
                translated = self.runtime.normalize_payload(body, path=path)
                self.assertIs(
                    translated["think"], value if isinstance(value, bool) else True
                )
                self.assertEqual(body, {"think": value})

    async def test_none_errors_conflicts_and_unknown_formats_are_untouched(self):
        cases = [
            {"reasoning_effort": "none"},
            {"reasoning": {"effort": "none"}},
            {"reasoning_effort": "high", "reasoning": {"effort": "none"}},
            {"reasoning_effort": "invalid"},
            {"reasoning": {"effort": "high", "extra": True}},
            {"reasoning_effort": "high", "think": False},
            {
                "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": True},
            },
            {"reasoning_effort": "high", "raw": True},
            {"reasoning_effort": "high", "template": "custom"},
        ]
        for body in cases:
            self.assertEqual(self.runtime.normalize_payload(body), body)
        self.assertEqual(
            self.runtime.normalize_payload(
                {"think": "high", "reasoning_effort": "none"}, path="/api/chat"
            ),
            {"think": "high", "reasoning_effort": "none"},
        )

    async def test_other_architectures_templates_and_routes_are_untouched(self):
        body = {"reasoning_effort": "high"}
        self.runtime.architecture = "qwen2"
        self.assertEqual(self.runtime.normalize_payload(body), body)
        self.runtime.architecture = "qwen3"
        self.runtime.template = "custom"
        self.assertEqual(self.runtime.normalize_payload(body), body)
        self.runtime.template = TEMPLATE
        self.assertEqual(self.runtime.normalize_payload(body, path="/unexpected"), body)

    async def test_internal_thinking_normalizes_after_controls_are_inserted(self):
        details = {}
        await self.runtime.complete(
            [dict(role="user", content="A question")], thinking=True, details=details
        )
        self.assertNotIn("reasoning_effort", self.requests[-1][1])
        self.assertTrue(details["complete"])
        self.assertTrue(details["reasoning"])
        await self.runtime.complete(
            [dict(role="user", content="A question")], thinking=False
        )
        self.assertEqual(self.requests[-1][1]["reasoning_effort"], "none")

    async def test_public_api_keeps_original_mode_before_normalizing(self):
        body = dict(
            model="test-model",
            messages=[dict(role="user", content="A question")],
            reasoning_effort="high",
            stream=False,
        )
        response = await self.client.post("/v1/chat/completions", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("reasoning_effort", self.requests[-1][1])
        row = self.store.rows()[-1]
        self.assertTrue(row["generation_mode"]["thinking"])
        self.assertEqual(row["generation_mode"]["source"], ["reasoning_effort"])

    async def test_both_terminal_paths_use_compatible_thinking(self):
        for path in ("/interact", "/stream_interact"):
            response = await self.client.post(
                path, json=dict(prompt="A question", use_history=False)
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("Rabat", response.text)
            self.assertNotIn("reasoning_effort", self.requests[-1][1])
            self.assertTrue(self.store.rows()[-1]["generation_mode"]["thinking"])

    async def test_both_native_routes_preserve_recorded_level_as_thinking(self):
        for path, inputs in (
            ("/api/chat", {"messages": [dict(role="user", content="A question")]}),
            ("/api/generate", {"prompt": "A question"}),
        ):
            response = await self.client.post(
                path,
                json=dict(model="test-model", think="high", stream=False, **inputs),
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIs(self.requests[-1][1]["think"], True)
            self.assertTrue(self.store.rows()[-1]["generation_mode"]["thinking"])


class UnconditionalReasonerThinkTest(unittest.IsolatedAsyncioTestCase):
    """A model that always reasons must be asked for its thought.

    Ollama's native API returns the reasoning only when ``think`` is set.
    Without it the wrapper marks the turn as thinking, receives an empty
    thought, records the turn incomplete, and then refuses feedback on it --
    so no interaction on this runtime can ever be corrected.
    """

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        with mock.patch(
            "adaptible._src.wrap.runtime.shutil.which", return_value="ollama"
        ):
            self.runtime = Ollama(
                "deepseek-r1", Path(self.temp.name) / "state", url="http://fixture"
            )
        self.runtime.architecture = "qwen2"
        self.runtime.template = "{{ .Prompt }}"

    def tearDown(self):
        self.temp.cleanup()

    def test_native_request_asks_for_the_thought(self):
        self.runtime.always_reasons = True
        body = dict(model="deepseek-r1", messages=[])
        self.assertIs(
            self.runtime.normalize_payload(body, path="/api/chat")["think"], True
        )

    def test_a_model_that_does_not_always_reason_is_unchanged(self):
        self.runtime.always_reasons = False
        body = dict(model="deepseek-r1", messages=[])
        self.assertEqual(self.runtime.normalize_payload(body, path="/api/chat"), body)

    def test_an_explicit_client_control_is_not_overridden(self):
        self.runtime.always_reasons = True
        body = dict(model="deepseek-r1", messages=[], think=False)
        self.assertIs(
            self.runtime.normalize_payload(body, path="/api/chat")["think"], False
        )

    def test_the_openai_route_is_not_given_a_native_control(self):
        # think is a native field; the OpenAI endpoint rejects it.
        self.runtime.always_reasons = True
        body = dict(model="deepseek-r1", messages=[])
        self.assertNotIn(
            "think",
            self.runtime.normalize_payload(body, path="/v1/chat/completions"),
        )


class SelectiveReasonerTurnTest(unittest.IsolatedAsyncioTestCase):
    """A model that reasons on some prompts stays correctable on the rest.

    ``always_reasons`` is inferred from one startup probe, but the same
    DeepSeek-R1 distill returns a thought for "reply with the word ready" and
    none for "what is the capital of Turkey?". Recording the thoughtless turn
    as incomplete makes it unflaggable, and an unflaggable turn can never be
    corrected -- on any runtime, not just Ollama.
    """

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        with mock.patch(
            "adaptible._src.wrap.runtime.shutil.which", return_value="ollama"
        ):
            self.runtime = Ollama(
                "deepseek-r1", self.root / "state", url="http://native.fixture"
            )
        self.runtime.architecture = "qwen2"
        self.runtime.template = "{{ .Prompt }}"
        self.runtime.always_reasons = True
        self.reasoning = ""
        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond)
        )
        self.store = Store(self.root / "history", "fixture")
        self.controller = Controller(self.runtime, self.store, web_search=False)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper.fixture",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    def respond(self, request):
        message = dict(role="assistant", content="The capital of Turkey is Istanbul.")
        if self.reasoning:
            message["reasoning"] = self.reasoning
        return httpx.Response(
            200,
            json=dict(
                id="1",
                model="deepseek-r1",
                choices=[dict(index=0, message=message, finish_reason="stop")],
            ),
        )

    async def ask(self):
        response = await self.client.post(
            "/interact",
            json=dict(prompt="What is the capital of Turkey?", use_history=False),
        )
        response.raise_for_status()
        idx = response.json()["interaction_idx"]
        row = self.store.db.execute(
            "SELECT status, generation_mode FROM interactions WHERE id=?", (idx,)
        ).fetchone()
        return idx, row[0], json.loads(row[1])

    async def test_a_turn_without_a_thought_is_complete_and_flaggable(self):
        idx, status, mode = await self.ask()
        self.assertEqual(status, "new")
        self.assertIs(mode["thinking"], False)
        self.assertIn("turn_without_reasoning", mode["source"])
        feedback = await self.client.post(
            "/feedback", json=dict(interaction_idx=idx, thumbs="down", note="a note")
        )
        self.assertEqual(feedback.status_code, 200)

    async def test_a_turn_with_a_thought_is_still_a_thinking_turn(self):
        self.reasoning = "Okay, the user asks about Turkey."
        _, status, mode = await self.ask()
        self.assertEqual(status, "new")
        self.assertIs(mode["thinking"], True)
        self.assertNotIn("turn_without_reasoning", mode["source"])
