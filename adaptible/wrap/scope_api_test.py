"""HTTP deployment contracts with fake generation, real controller and SQLite.

The router boundary is mocked here; scope.py has independent matching tests.
Dummy adapter files establish identity only, never native learning evidence.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from adaptible.wrap.app import create_app
from adaptible.wrap.repair import Controller
from adaptible.wrap.store import Store

SYSTEM = "Use the public catalog."
FIRST = "When did Ember Observatory open?"
SECOND = "Where is Juniper Archive?"
OTHER = "Who founded Larch Museum?"


class BoundaryRouter:
    def __init__(self):
        self.calls = []

    async def choose(self, messages, repairs, mode):
        self.calls.append((messages, repairs, mode))
        if mode.get("error") or not mode.get("thinking"):
            return dict(scope=None, reason="mode_mismatch")
        for repair in repairs:
            if messages == repair["messages"]:
                return dict(scope=repair["interaction_idx"], reason="exact")
        return dict(scope=None, reason="no_scope")


class DifferentialRuntime:
    name = "fixture"
    native = True
    architecture = "qwen3"
    url = "http://native.fixture"
    max_tokens = 128

    def __init__(self):
        self.active = None
        self.sent = []
        self.prepared = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))

    async def stage(self, directory):
        return "handle:" + Path(directory).name

    async def restore(self, handle):
        self.active = handle

    async def prepare_payload(self, body, *, frozen=False):
        self.prepared.append((frozen, self.active))
        return {**body, "model": "base" if frozen else self.active}

    async def close(self):
        await self.client.aclose()

    def respond(self, request):
        body = json.loads(request.content)
        self.sent.append((request.url.path, body))
        question = body.get("prompt") or body["messages"][-1]["content"]
        base = body["model"] == "base"
        # The adapted runtime deliberately forgets the unrelated answer.
        answer = (
            {FIRST: "1912", SECOND: "East Wing", OTHER: "Ada Reed"}
            if base
            else {FIRST: "1917", SECOND: "West Wing", OTHER: "Mira Stone"}
        )[question]
        thinking = "I considered the catalog and chose the answer."
        native = request.url.path.startswith("/api/")
        generate = request.url.path == "/api/generate"
        if not body.get("stream"):
            if native:
                result = dict(model=body["model"], done=True, done_reason="stop")
                result.update(
                    dict(response=answer, thinking=thinking)
                    if generate
                    else dict(message=dict(content=answer, thinking=thinking))
                )
            else:
                result = dict(
                    model=body["model"],
                    choices=[
                        dict(
                            message=dict(content=answer, reasoning_content=thinking),
                            finish_reason="stop",
                        )
                    ],
                )
            return httpx.Response(200, json=result)
        if native:
            chunks = (
                [dict(thinking=thinking), dict(response=answer)]
                if generate
                else [
                    dict(message=dict(thinking=thinking)),
                    dict(message=dict(content=answer)),
                ]
            )
            chunks.append(dict(done=True, done_reason="stop"))
            return httpx.Response(
                200,
                text="".join(
                    json.dumps(dict(model=body["model"], **chunk)) + "\n"
                    for chunk in chunks
                ),
            )
        chunks = [
            dict(delta=dict(reasoning_content=thinking)),
            dict(delta=dict(content=answer)),
            dict(delta={}, finish_reason="stop"),
        ]
        return httpx.Response(
            200,
            text="".join(
                "data: "
                + json.dumps(dict(model=body["model"], choices=[chunk]))
                + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n",
        )


class ScopeAPIContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.store = Store(self.directory, "fixture-identity")
        self.runtime = DifferentialRuntime()
        self.controller = Controller(self.runtime, self.store, web_search=False)
        self.controller.scope_router = BoundaryRouter()
        self.first = self.seed("first", FIRST)
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper.fixture",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    def seed(self, name, question, *, system=None, include_digest=True):
        messages = ([dict(role="system", content=system)] if system else []) + [
            dict(role="user", content=question)
        ]
        idx = self.store.record(messages, "old", generation_mode={"thinking": True})
        directory = self.directory / name
        (directory / "adapter").mkdir(parents=True)
        data = ("fake weights for " + name).encode()
        (directory / "adapter/adapter_model.safetensors").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        accepted = dict(directory=str(directory))
        if include_digest:
            accepted["sha256"] = digest
        self.store.accept_repair(
            accepted,
            dict(
                interaction_idx=idx,
                messages=messages,
                question=question,
                generation_mode={"thinking": True},
                prompts=[messages],
            ),
        )
        return dict(id=idx, directory=directory, sha256=digest)

    async def send(self, path, question, *, stream=False, system=None, extra=None):
        native = path.startswith("/api/")
        body = dict(model="fixture", stream=stream, temperature=0.3, seed=19)
        body.update(
            dict(think=True, options=dict(num_predict=128))
            if native
            else dict(reasoning_effort="high", max_tokens=128)
        )
        if path == "/api/generate":
            body["prompt"] = question
            if system:
                body["system"] = system
        else:
            body["messages"] = (
                [dict(role="system", content=system)] if system else []
            ) + [dict(role="user", content=question)]
        body.update(extra or {})
        response = await self.client.post(path, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        forwarded = self.runtime.sent[-1][1]
        self.assertEqual(
            {k: v for k, v in forwarded.items() if k != "model"},
            {k: v for k, v in body.items() if k != "model"},
        )
        row = next(
            row
            for row in self.store.rows()
            if row["id"] == int(response.headers["X-Interaction-Idx"])
        )
        route = row["response_details"]["adaptation"]
        self.assertEqual(route["adapter"], response.headers["X-Adaptible-Adapter"])
        self.assertEqual(
            str(route["scope"]) if route["scope"] is not None else None,
            response.headers.get("X-Adaptible-Scope"),
        )
        self.assertEqual(row["status"], "new")
        self.assertTrue(row["response_details"]["complete"])
        self.assertTrue(row["response_details"]["reasoning"])
        self.assertEqual(route["policy"], "correction_scoped_v1")
        self.assertNotIn("handle:", response.text)
        return row, route

    async def test_every_public_protocol_routes_and_preserves_request_body(self):
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            for stream in (False, True):
                for question, expected, adapter in (
                    (FIRST, "1917", "first"),
                    (OTHER, "Ada Reed", "base"),
                ):
                    with self.subTest(path=path, stream=stream, question=question):
                        row, route = await self.send(path, question, stream=stream)
                        self.assertEqual(row["response"], expected)
                        self.assertEqual(route["adapter"], adapter)
                        self.assertEqual(
                            route["adapter_sha256"],
                            self.first["sha256"] if adapter != "base" else None,
                        )
                        self.assertEqual(
                            self.runtime.prepared[-1][0], adapter == "base"
                        )

    async def test_latest_cumulative_handle_serves_old_scope_after_restart(self):
        second = self.seed("second", SECOND)
        await self.runtime.restore(await self.runtime.stage(second["directory"]))
        for question, scope in ((FIRST, self.first["id"]), (SECOND, second["id"])):
            _, route = await self.send("/v1/chat/completions", question)
            self.assertEqual(
                (route["adapter"], route["scope"], route["adapter_sha256"]),
                ("second", scope, second["sha256"]),
            )
            self.assertEqual(self.runtime.sent[-1][1]["model"], "handle:second")
        await self.client.aclose()
        await self.controller.close()
        self.store = Store(self.directory, "fixture-identity")
        self.runtime = DifferentialRuntime()
        self.controller = Controller(self.runtime, self.store, web_search=False)
        self.controller.scope_router = BoundaryRouter()
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper.fixture",
        )
        row, route = await self.send("/api/generate", FIRST, stream=True)
        self.assertEqual(
            (row["response"], route["adapter"], route["scope"]),
            ("1917", "second", self.first["id"]),
        )

    async def test_system_context_must_match_on_chat_and_native_generate(self):
        scoped = self.seed("system", FIRST, system=SYSTEM)
        await self.runtime.restore(await self.runtime.stage(scoped["directory"]))
        for path in ("/v1/chat/completions", "/api/chat", "/api/generate"):
            for system, adapter in (
                (SYSTEM, "system"),
                ("Use another catalog.", "base"),
            ):
                row, route = await self.send(path, FIRST, system=system)
                self.assertEqual(route["adapter"], adapter)
                self.assertEqual(
                    row["messages"][0], dict(role="system", content=system)
                )

    async def test_cli_nonstream_and_stream_use_routing_and_keep_thinking(self):
        for path in ("/interact", "/stream_interact"):
            for question, expected, adapter in (
                (FIRST, "1917", "first"),
                (OTHER, "Ada Reed", "base"),
            ):
                response = await self.client.post(
                    path, json=dict(prompt=question, use_history=False)
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["X-Adaptible-Adapter"], adapter)
                text = (
                    response.json()["response"]
                    if path == "/interact"
                    else response.text
                )
                self.assertIn("<think>", text)
                self.assertTrue(text.endswith(expected), text)

    async def test_unsupported_context_falls_back_without_consulting_router(self):
        count = len(self.controller.scope_router.calls)
        row, route = await self.send("/api/generate", FIRST, extra={"raw": True})
        self.assertEqual(row["response"], "1912")
        self.assertEqual(
            (route["adapter"], route["reason"]), ("base", "unsupported_context")
        )
        self.assertEqual(len(self.controller.scope_router.calls), count)

    async def test_old_state_computes_actual_adapter_digest_before_reporting_it(self):
        older = self.seed("without-digest", SECOND, include_digest=False)
        await self.runtime.restore(await self.runtime.stage(older["directory"]))
        _, route = await self.send("/v1/chat/completions", SECOND)
        self.assertEqual(route["adapter_sha256"], older["sha256"])
        self.assertEqual(self.store.get("accepted")["sha256"], older["sha256"])


if __name__ == "__main__":
    unittest.main()
