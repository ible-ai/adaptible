"""Real tiny GGUF arithmetic plus mocked native Ollama lifecycle."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import httpx
import numpy as np

from adaptible.wrap.gguf_fusion import fuse_gguf
from adaptible.wrap.ollama import remove_fused_file
from adaptible.wrap.runtime import Ollama


def finish(writer):
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class OllamaFusionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.base = self.root / "original.gguf"
        writer = gguf.GGUFWriter(str(self.base), "qwen3")
        writer.add_tensor("blk.0.attn_q.weight", np.zeros((4, 32), np.float32))
        writer.add_tensor("blk.0.attn_q_norm.weight", np.ones(32, np.float32))
        finish(writer)
        self.before = self.base.read_bytes()
        self.tags, self.loaded, self.commands = set(), set(), []
        self.fail_fusion = self.fail_create = self.cancel_create = False
        self.runtime = await self.new_runtime()
        self.patch = patch(
            "adaptible.wrap.runtime.run_command", side_effect=self.command
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def new_runtime(self):
        with patch("adaptible.wrap.runtime.shutil.which", return_value="ollama"):
            runtime = Ollama("original-tag", self.state, url="http://fake")
        await runtime.client.aclose()
        runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))
        self.addAsyncCleanup(runtime.close)
        runtime.architecture = "qwen3"
        runtime.blob = self.base
        runtime.modelfile = f'FROM "{self.base}"\nTEMPLATE "{{{{ .Messages }}}}"\nSYSTEM "Be precise."\nPARAMETER temperature 0.6\n'
        return runtime

    def candidate(self, name):
        path = self.state / "adapters" / name
        path.mkdir(parents=True)
        writer = gguf.GGUFWriter(str(path / "adapter.gguf"), "qwen3")
        writer.add_float32("adapter.lora.alpha", 4)
        writer.add_tensor(
            "blk.0.attn_q.weight.lora_a", np.full((2, 32), 0.1, np.float32)
        )
        writer.add_tensor(
            "blk.0.attn_q.weight.lora_b", np.full((4, 2), 0.2, np.float32)
        )
        finish(writer)
        return path

    async def command(self, *args, **kwargs):
        self.commands.append(args)
        if args[1:3] == ("-m", "adaptible.wrap.gguf_fusion"):
            if self.fail_fusion:
                Path(args[-1]).with_suffix(".gguf.partial").write_bytes(b"incomplete")
                raise RuntimeError("failed fusion")
            fuse_gguf(*args[3:])
        elif args[1] == "create":
            self.tags.add(args[2] + ":latest")
            self.loaded.add(args[2] + ":latest")
            if self.fail_create:
                raise RuntimeError("failed create")
            if self.cancel_create:
                raise asyncio.CancelledError()
        return ""

    def respond(self, request):
        path = request.url.path
        payload = json.loads(request.content) if request.content else {}
        name = payload.get("model", "")
        if name and ":" not in name:
            name += ":latest"
        if path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": x} for x in self.loaded]}
            )
        if path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": x} for x in self.tags]}
            )
        if path == "/api/generate":
            self.loaded.discard(name)
        if path == "/api/delete":
            if name not in self.tags:
                return httpx.Response(404, json={})
            self.tags.remove(name)
        return httpx.Response(200, json={})

    async def test_fuses_scaled_weights_preserves_base_and_native_configuration(self):
        candidate = self.candidate("first")
        handle = await self.runtime.stage(candidate)
        text = (candidate / "Modelfile").read_text()
        self.assertEqual(
            text.split("\n", 1)[1], self.runtime.modelfile.split("\n", 1)[1]
        )
        self.assertTrue(
            text.startswith(f'FROM "{(candidate / "merged.gguf").resolve()}"')
        )
        self.assertNotIn("ADAPTER", text)
        tensors = {
            t.name: t.data
            for t in gguf.GGUFReader(str(candidate / "merged.gguf")).tensors
        }
        np.testing.assert_allclose(tensors["blk.0.attn_q.weight"], 0.08, atol=1e-4)
        np.testing.assert_array_equal(tensors["blk.0.attn_q_norm.weight"], 1)
        self.assertEqual(self.base.read_bytes(), self.before)
        await self.runtime.restore(handle)
        await self.runtime.discard(handle)
        await self.runtime.discard("unrelated-model")
        self.assertTrue((candidate / "merged.gguf").exists())

    async def test_superseded_and_rejected_cleanup_preserves_small_adapters(self):
        first, second = self.candidate("first"), self.candidate("second")
        first_handle = await self.runtime.stage(first)
        await self.runtime.restore(first_handle)
        second_handle = await self.runtime.stage(second)
        await self.runtime.restore(second_handle)
        await self.runtime.discard(first_handle)
        self.assertFalse((first / "merged.gguf").exists())
        self.assertTrue((first / "adapter.gguf").exists())
        await self.runtime.restore(None)
        await self.runtime.discard(second_handle)
        self.assertFalse((second / "merged.gguf").exists())
        self.assertEqual(self.tags, set())
        self.assertEqual(self.loaded, set())

    async def test_restart_reuses_or_reconstructs_accepted_derived_model(self):
        candidate = self.candidate("first")
        handle = await self.runtime.stage(candidate)
        runtime = await self.new_runtime()
        calls = lambda: sum(command[1] == "-m" for command in self.commands)
        self.assertEqual(await runtime.stage(candidate), handle)
        self.assertEqual(calls(), 1)
        (candidate / "merged.gguf").unlink()
        self.assertEqual(await runtime.stage(candidate), handle)
        self.assertEqual(calls(), 2)
        self.assertEqual(self.base.read_bytes(), self.before)

    async def test_failure_and_cancellation_cleanup_even_after_tag_creation(self):
        for failure in ("fail_fusion", "fail_create", "cancel_create"):
            with self.subTest(failure=failure):
                candidate = self.candidate(failure)
                setattr(self, failure, True)
                error = (
                    asyncio.CancelledError
                    if failure == "cancel_create"
                    else RuntimeError
                )
                with self.assertRaises(error):
                    await self.runtime.stage(candidate)
                setattr(self, failure, False)
                self.assertFalse((candidate / "merged.gguf").exists())
                self.assertFalse((candidate / "merged.gguf.partial").exists())
                self.assertTrue((candidate / "adapter.gguf").exists())
                self.assertEqual(self.tags, set())
                self.assertEqual(self.loaded, set())
        self.assertEqual(self.base.read_bytes(), self.before)

    async def test_qwen2_keeps_native_adapter_path(self):
        candidate = self.candidate("legacy")
        self.runtime.architecture = "qwen2"
        await self.runtime.stage(candidate)
        self.assertIn("ADAPTER", (candidate / "Modelfile").read_text())
        self.assertFalse((candidate / "merged.gguf").exists())

    async def test_demo_cleanup_removes_derived_file_and_retains_adapter(self):
        from scripts.wrapper_demo import cleanup_tags

        candidate = self.candidate("demo")
        handle = await self.runtime.stage(candidate)
        client = httpx.Client(
            base_url="http://fake", transport=httpx.MockTransport(self.respond)
        )
        with patch("scripts.wrapper_demo.httpx.Client", return_value=client):
            removed = cleanup_tags(
                SimpleNamespace(service="ollama", upstream="http://fake"), self.state
            )
        self.assertEqual(removed, [handle + ":latest"])
        self.assertFalse((candidate / "merged.gguf").exists())
        self.assertTrue((candidate / "adapter.gguf").exists())
        self.assertEqual(self.base.read_bytes(), self.before)

    async def test_cleanup_rejects_symlink_or_manifest_path_outside_owned_directory(
        self,
    ):
        candidate = self.candidate("guard")
        handle = await self.runtime.stage(candidate)
        merged = candidate / "merged.gguf"
        merged.unlink()
        merged.symlink_to(self.base)
        self.assertFalse(remove_fused_file(self.state, handle))
        (self.state / "ollama-derived.json").write_text(
            json.dumps({handle: str(self.base)})
        )
        self.assertFalse(remove_fused_file(self.state, handle))
        self.assertEqual(self.base.read_bytes(), self.before)

    async def test_fusion_refuses_partial_alias_without_touching_original(self):
        candidate = self.candidate("partial")
        target = candidate / "merged.gguf"
        partial = target.with_suffix(".gguf.partial")
        for link in ("symlink", "hardlink"):
            if link == "symlink":
                partial.symlink_to(self.base)
            else:
                partial.hardlink_to(self.base)
            with self.assertRaises(FileExistsError):
                fuse_gguf(self.base, candidate / "adapter.gguf", target)
            self.assertEqual(self.base.read_bytes(), self.before)
            partial.unlink()
