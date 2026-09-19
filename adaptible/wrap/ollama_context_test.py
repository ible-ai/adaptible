"""Owned Ollama context metadata, preserving original weights and settings."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import httpx

from adaptible.wrap.runtime import Ollama


class OllamaContextTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.blob = self.root / "original.gguf"
        self.blob.write_bytes(b"original weights")
        self.original = dict(
            modelfile=f'FROM "{self.blob}"\nTEMPLATE "native template"\nSYSTEM "Original system"\nPARAMETER num_ctx 4096\n',
            template="native template",
            system="Original system",
            messages=[],
            parameters='num_ctx 4096\nstop "first"\nstop "second"\ntemperature 0.6',
        )
        self.tags = {"original": copy.deepcopy(self.original)}
        self.loaded, self.requests = set(), []
        self.corrupt = None
        self.create_started = self.create_release = None
        self.runtime = await self.new_runtime()

    async def new_runtime(self, context=8192):
        with mock.patch(
            "adaptible.wrap.runtime.shutil.which", return_value="ollama"
        ):
            runtime = Ollama(
                "original", self.root / "state", url="http://fake", context_size=context
            )
        await runtime.client.aclose()
        runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))
        self.addAsyncCleanup(runtime.close)
        return runtime

    @staticmethod
    def applied(parameters, requested):
        """What Ollama reports back after a create that sets ``requested``.

        Overridden keys are replaced, everything else is inherited -- which is
        what the wrapper then verifies, so the fixture has to model it rather
        than patch one line of text.
        """
        kept = [
            line
            for line in parameters.splitlines()
            if line.strip() and line.split()[0] not in requested
        ]
        return "\n".join(kept + [f"{key} {value}" for key, value in requested.items()])

    async def respond(self, request):
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, body))
        name = body.get("model", "").removesuffix(":latest")
        if request.url.path == "/api/show":
            return (
                httpx.Response(200, json=self.tags[name])
                if name in self.tags
                else httpx.Response(404)
            )
        if request.url.path == "/api/create":
            self.assertEqual(set(body), {"model", "from", "parameters", "stream"})
            self.assertEqual(body["from"], "original")
            self.assertFalse(body["stream"])
            # The wrapper pins the sampler as well as the context. Ollama's
            # house defaults (repeat_penalty 1.1, top_k 40, top_p 0.9) are not
            # the original's -- mlx_lm's sampler is temperature and nothing
            # else -- and none of them can be sent per request through the
            # OpenAI-compatible route, so they are pinned on the tag.
            sampling = dict(body["parameters"])
            context = sampling.pop("num_ctx")
            # Batch 8 keeps ggml's prefill off the f16-staged matrix kernel.
            self.assertEqual(sampling.pop("num_batch"), 8)
            self.assertEqual(
                sampling,
                {"repeat_penalty": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0},
            )
            if self.create_started:
                self.create_started.set()
                await self.create_release.wait()
            info = copy.deepcopy(self.original)
            info["parameters"] = self.applied(info["parameters"], body["parameters"])
            info["modelfile"] = info["modelfile"].replace(
                "num_ctx 4096", f"num_ctx {context}"
            )
            if self.corrupt:
                self.corrupt(info)
            self.tags[name] = info
            return httpx.Response(200, json={"status": "success"})
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": n + ":latest"} for n in self.loaded]}
            )
        if request.url.path == "/api/generate":
            self.assertEqual(body["keep_alive"], 0)
            self.loaded.discard(name)
            return httpx.Response(200, json={"done": True})
        if request.url.path == "/api/delete":
            self.assertNotEqual(name, "original")
            existed = self.tags.pop(name, None)
            return httpx.Response(200 if existed else 404, json={})
        self.fail(f"Unexpected API request: {request.url}")

    async def test_context_derivative_reuses_base_and_preserves_all_other_settings(
        self,
    ):
        self.assertEqual(await self.runtime.discover(), self.blob)
        handle = self.runtime.base_handle
        self.assertNotEqual(handle, self.runtime.name)
        self.assertEqual(self.runtime.name, "original")
        self.assertIn("PARAMETER num_ctx 8192", self.runtime.modelfile)
        body = {"model": "client-name", "options": {"num_ctx": 1024}, "messages": []}
        before = copy.deepcopy(body)
        self.assertEqual(self.runtime.payload(body)["model"], handle)
        self.assertEqual((await self.runtime.prepare_payload(body))["model"], handle)
        self.assertEqual(body, before)
        self.runtime.active = self.runtime.prefix + "accepted"
        self.assertEqual(self.runtime.payload(body)["model"], self.runtime.active)
        self.assertEqual(self.runtime.payload(body, frozen=True)["model"], handle)
        self.assertEqual(
            self.runtime.payload(body, handle="candidate")["model"], "candidate"
        )
        self.assertEqual(self.tags["original"], self.original)
        self.assertEqual(self.blob.read_bytes(), b"original weights")
        self.assertEqual(
            list((self.root / "state").iterdir()), [self.runtime.context_record]
        )
        self.tags[self.runtime.active] = {}
        self.loaded = {"original", handle, self.runtime.active, "unrelated"}
        await self.runtime.release_for_training()
        self.assertEqual(self.loaded, {"original", "unrelated"})
        await self.runtime.close()
        self.assertNotIn(handle, self.tags)
        self.assertIn(self.runtime.active, self.tags)
        self.assertFalse(self.runtime.context_record.exists())

    async def test_already_explicit_matching_settings_need_no_derivative(self):
        """A model that already serves the wrapper's settings is left alone.

        Matching the context is no longer enough on its own: a tag that still
        carries Ollama's house sampler needs a derivative however right its
        num_ctx is, because the sampler is what changes the output.
        """
        self.original["parameters"] = (
            "num_ctx 4096\nstop \"first\"\nstop \"second\"\n"
            "repeat_penalty 1.0\ntop_k 0\ntop_p 1.0\nmin_p 0.0\nnum_batch 8"
        )
        self.tags["original"] = copy.deepcopy(self.original)
        runtime = await self.new_runtime(4096)
        await runtime.discover()
        self.assertEqual(runtime.base_handle, "original")
        self.assertFalse(any(path == "/api/create" for _, path, _ in self.requests))

    async def test_matching_context_but_house_sampler_still_gets_a_derivative(self):
        """The defect this pinning exists for.

        Ollama's default repeat_penalty of 1.1 changed greedy output: on the
        same prompt and the identical f32 weights it returned 2311 characters
        where the original returned 2247, and 1.0 reproduced the original's
        exactly. A wrapper that skipped the derivative because num_ctx already
        matched would serve under that penalty and silently disagree with the
        thing it is meant to reproduce.
        """
        runtime = await self.new_runtime(4096)
        await runtime.discover()
        self.assertNotEqual(runtime.base_handle, "original")
        create = [body for _, path, body in self.requests if path == "/api/create"]
        self.assertEqual(len(create), 1)
        self.assertEqual(create[0]["parameters"]["repeat_penalty"], 1.0)

    async def test_implicit_context_gets_explicit_derivative(self):
        self.original["parameters"] = self.original["parameters"].replace(
            "num_ctx 4096\n", ""
        )
        self.tags["original"] = copy.deepcopy(self.original)
        # The fixture's `applied` now reports every requested parameter back,
        # num_ctx included, so nothing extra has to be simulated here.
        await self.runtime.discover()
        self.assertNotEqual(self.runtime.base_handle, "original")

    async def test_unhonored_context_is_rejected_and_private_tag_removed(self):
        self.corrupt = lambda info: info.update(parameters=self.original["parameters"])
        with self.assertRaisesRegex(ValueError, "did not honor the pinned parameters"):
            await self.runtime.discover()
        self.assertEqual(set(self.tags), {"original"})
        self.assertFalse(self.runtime.context_record.exists())

    async def test_changed_template_is_rejected_without_mutating_original(self):
        self.corrupt = lambda info: info.update(template="different")
        with self.assertRaisesRegex(ValueError, "base/prompt"):
            await self.runtime.discover()
        self.assertEqual(self.tags, {"original": self.original})

    async def test_restart_reconciles_owned_context_tag_then_recreates(self):
        await self.runtime.discover()
        handle = self.runtime.base_handle
        await self.runtime.client.aclose()  # Simulate crash without normal cleanup.
        restarted = await self.new_runtime()
        await restarted.discover()
        self.assertEqual(restarted.base_handle, handle)
        self.assertEqual(sum(path == "/api/create" for _, path, _ in self.requests), 2)
        self.assertTrue(
            any(
                path == "/api/delete" and body["model"] == handle
                for _, path, body in self.requests
            )
        )

    async def test_cancel_waits_for_metadata_creation_then_cleans_tag(self):
        self.create_started, self.create_release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(self.runtime.discover())
        await self.create_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.create_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.tags, {"original": self.original})
        self.assertFalse(self.runtime.context_record.exists())

    async def test_tampered_ownership_record_never_deletes_user_tag(self):
        self.runtime.context_record.write_text(json.dumps({"model": "original"}))
        with self.assertRaisesRegex(ValueError, "unowned"):
            await self.runtime.discover()
        self.assertEqual(self.requests, [])
        self.runtime.context_record.unlink()

    async def test_cli_passes_requested_context_to_ollama(self):
        from adaptible.wrap.__main__ import parser, serve

        args = parser().parse_args(
            [
                "ollama",
                "original",
                "--context-size",
                "8192",
                "--state-dir",
                str(self.root / "cli"),
            ]
        )
        runtime = mock.Mock(
            discover=mock.AsyncMock(side_effect=RuntimeError("stop before models")),
            close=mock.AsyncMock(),
        )
        with mock.patch(
            "adaptible.wrap.__main__.Ollama", return_value=runtime
        ) as constructor:
            with self.assertRaisesRegex(RuntimeError, "stop before models"):
                await serve(args)
        self.assertEqual(constructor.call_args.kwargs["context_size"], 8192)
        runtime.close.assert_awaited_once()
