"""Model-free native LM Studio lifecycle and numerical GGUF merge tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from adaptible.wrap.lmstudio import LMStudio, fuse_gguf


class FusionTest(unittest.TestCase):
    def test_quantized_base_merge_preserves_metadata_and_unmodified_bytes(self):
        self.check_merge("qwen2")

    def test_qwen3_merge_preserves_architecture_and_frozen_norms(self):
        self.check_merge("qwen3")

    def check_merge(self, architecture):
        import gguf
        import numpy as np

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            base, adapter, merged = (
                root / name for name in ("base.gguf", "adapter.gguf", "merged.gguf")
            )
            values = np.arange(128, dtype=np.float32).reshape(4, 32) / 128
            quant = gguf.quantize(values, gguf.GGMLQuantizationType.Q8_0)
            writer = gguf.GGUFWriter(str(base), architecture)
            writer.add_string("tokenizer.chat_template", "{{ messages }}")
            writer.add_array("tokenizer.ggml.tokens", ["a", "b"])
            writer.add_uint32(f"{architecture}.block_count", 1)
            if architecture == "qwen3":
                writer.add_tensor("blk.0.attn_q_norm.weight", np.ones(32, np.float32))
                writer.add_tensor("blk.0.attn_k_norm.weight", np.ones(32, np.float32))
            writer.add_tensor(
                "blk.0.attn_q.weight", quant, raw_dtype=gguf.GGMLQuantizationType.Q8_0
            )
            writer.add_tensor(
                "untouched.weight", quant, raw_dtype=gguf.GGMLQuantizationType.Q8_0
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            a, b = np.full((2, 32), 0.1, np.float32), np.full((4, 2), 0.2, np.float32)
            writer = gguf.GGUFWriter(str(adapter), architecture)
            writer.add_float32("adapter.lora.alpha", 4)
            writer.add_tensor("blk.0.attn_q.weight.lora_a", a)
            writer.add_tensor("blk.0.attn_q.weight.lora_b", b)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            before = base.read_bytes()
            fuse_gguf(base, adapter, merged)
            self.assertEqual(base.read_bytes(), before)
            result = gguf.GGUFReader(str(merged))
            self.assertEqual(
                result.fields["tokenizer.chat_template"].contents(), "{{ messages }}"
            )
            self.assertEqual(
                result.fields["tokenizer.ggml.tokens"].contents(), ["a", "b"]
            )
            self.assertEqual(result.fields[f"{architecture}.block_count"].contents(), 1)
            self.assertEqual(
                result.fields["general.architecture"].contents(), architecture
            )
            tensors = {t.name: t for t in result.tensors}
            if architecture == "qwen3":
                for name in ("q", "k"):
                    np.testing.assert_array_equal(
                        tensors[f"blk.0.attn_{name}_norm.weight"].data,
                        np.ones(32, np.float32),
                    )
            np.testing.assert_array_equal(tensors["untouched.weight"].data, quant)
            self.assertEqual(
                tensors["untouched.weight"].tensor_type, gguf.GGMLQuantizationType.Q8_0
            )
            expected = (
                gguf.dequantize(quant, gguf.GGMLQuantizationType.Q8_0) + 2 * b @ a
            ).astype(np.float16)
            np.testing.assert_array_equal(tensors["blk.0.attn_q.weight"].data, expected)
            with self.assertRaises(ValueError):
                fuse_gguf(base, adapter, base)


class LMStudioTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "original.gguf"
        self.base.write_bytes(b"fixture")
        self.runtime = LMStudio(self.base, self.root / "state", executable="/fake/lms")
        self.keys, self.instances, self.calls = set(), {}, []
        self.short_keys = False
        self.fail_load = None
        self.import_root = self.root / "lmstudio-models"
        self.command = patch(
            "adaptible.wrap.lmstudio.run_command", side_effect=self.run_command
        )
        self.command.start()
        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.request)
        )
        # Models load through LM Studio's SDK API (the only one that accepts a
        # KV cache type); serving stays on REST. The fake keeps the load's
        # contract: one resident wrapper model, and a failed load raises.
        self.runtime._load_instance = self.load_instance

    async def load_instance(self, key):
        self.calls.append(("load", {"model": key}))
        if key == self.fail_load:
            import lmstudio

            raise lmstudio.LMStudioServerError("runtime load failed")
        self.assertIn(key, self.keys)
        self.assertFalse(self.instances, "Only one wrapper model should be resident")
        instance = key + "-instance"
        self.instances[instance] = key
        return instance

    async def asyncTearDown(self):
        await self.runtime.close()
        self.command.stop()
        self.temp.cleanup()

    async def run_command(self, *args):
        self.assertEqual(args[:2], ("/fake/lms", "import"))
        self.assertIn("--symbolic-link", args)
        source = Path(args[2])
        key = args[args.index("--user-repo") + 1]
        link = self.import_root / key / source.name
        link.parent.mkdir(parents=True)
        link.symlink_to(source)
        self.keys.add(key.split("/", 1)[-1] if self.short_keys else key)
        return f"Symbolic link created at {link}\n"

    def request(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((path, body))
        if path == "/api/v1/models":
            return httpx.Response(
                200, json={"models": [{"key": key} for key in self.keys]}
            )
        if path == "/api/v1/models/unload":
            del self.instances[body["instance_id"]]
            return httpx.Response(200, json={"instance_id": body["instance_id"]})
        if path == "/v1/chat/completions":
            self.assertIn(body["model"], self.instances)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "native LM Studio answer"}}]},
            )
        raise AssertionError(path)

    async def candidate(self):
        directory = self.root / "state/adapters/candidate"
        directory.mkdir(parents=True)
        (directory / "merged.gguf").write_bytes(b"fused fixture")
        return directory, await self.runtime.stage(directory)

    async def test_discover_imports_original_directly_without_state_symlink(self):
        with (
            patch.object(Path, "symlink_to", side_effect=AssertionError("ExFAT state")),
            patch.object(
                self.runtime, "_import", new_callable=AsyncMock, return_value="base-key"
            ) as imported,
        ):
            self.assertEqual(await self.runtime.discover(), self.base.resolve())
        imported.assert_awaited_once_with(self.base.resolve(), "base")
        self.assertFalse((self.runtime.directory / "base.gguf").exists())

    async def test_legacy_wrong_base_link_is_rejected(self):
        other = self.root / "other.gguf"
        other.write_bytes(b"other base")
        (self.runtime.directory / "base.gguf").symlink_to(other)
        with self.assertRaisesRegex(ValueError, "another model"):
            await self.runtime.discover()

    async def test_public_preparation_switches_loaded_model_without_changing_active(
        self,
    ):
        await self.runtime.discover()
        _, candidate = await self.candidate()
        await self.runtime.restore(candidate)
        body = dict(model="public-alias", messages=[], stream=True)
        prepared = await self.runtime.prepare_payload(body, frozen=True)
        self.assertEqual(prepared["model"], self.runtime.base_key + "-instance")
        self.assertEqual(self.runtime.active, candidate)
        self.assertEqual(self.runtime.loaded, self.runtime.base_key)
        prepared = await self.runtime.prepare_payload(body)
        self.assertEqual(prepared["model"], candidate + "-instance")
        self.assertEqual(self.runtime.active, candidate)
        self.assertEqual(body, dict(model="public-alias", messages=[], stream=True))

    async def test_stage_frozen_judge_restore_and_cleanup(self):
        await self.runtime.discover()
        await self.runtime.restore(None)
        base_key = self.runtime.loaded
        directory, candidate = await self.candidate()
        self.assertEqual(self.runtime.active, None)
        self.assertEqual(
            await self.runtime.complete([], handle=candidate), "native LM Studio answer"
        )
        await self.runtime.complete([], frozen=True)
        self.assertEqual(self.runtime.loaded, candidate)
        self.assertEqual(
            self.runtime.payload({}, handle=candidate)["model"], candidate + "-instance"
        )
        await self.runtime.restore(None)
        await self.runtime.discard(candidate)
        self.assertEqual(self.runtime.loaded, base_key)
        self.assertEqual(len(self.runtime.imports), 1)
        self.assertFalse((directory / "merged.gguf").exists())
        await self.runtime.cleanup_imports()
        self.assertFalse(self.instances)
        self.assertFalse(self.runtime.imports)
        self.assertFalse(list(self.import_root.rglob("*.gguf")))
        self.assertTrue(self.base.exists())

    async def test_active_candidate_survives_frozen_generation_and_resume(self):
        await self.runtime.discover()
        directory, candidate = await self.candidate()
        self.runtime.active = candidate
        await self.runtime.complete([], frozen=True)
        self.assertEqual(self.runtime.loaded, candidate)
        await self.runtime.release_for_training()
        self.assertFalse(self.instances)
        await self.runtime.restore(candidate)
        await self.runtime.discard(candidate)
        self.assertIn(candidate, self.runtime.imports)
        await self.runtime.close()
        self.assertFalse(self.instances)
        self.assertTrue(self.runtime.imports_path.exists())
        # Discover is idempotent and never imports/downloads the base twice.
        with patch(
            "adaptible.wrap.lmstudio.run_command", new_callable=AsyncMock
        ) as command:
            self.runtime.client = httpx.AsyncClient(
                transport=httpx.MockTransport(self.request)
            )
            await self.runtime.discover()
            await self.runtime.stage(directory)
            command.assert_not_awaited()

    async def test_remote_service_rejected(self):
        with self.assertRaisesRegex(ValueError, "local service"):
            LMStudio(
                self.base,
                self.root / "remote",
                url="http://other-machine:1234",
                executable="lms",
            )

    async def test_native_api_short_keys_without_publisher(self):
        self.short_keys = True
        await self.runtime.discover()
        await self.runtime.restore(None)
        self.assertTrue(self.runtime.base_key.startswith(self.runtime.prefix))
        directory, candidate = await self.candidate()
        await self.runtime.restore(None)
        await self.runtime.discard(candidate)
        self.assertFalse((directory / "merged.gguf").exists())
        self.assertEqual(len(self.runtime.imports), 1)

    async def test_failed_fusion_removes_partial_artifact(self):
        directory = self.root / "state/adapters/interrupted"
        directory.mkdir(parents=True)
        partial = directory / "merged.gguf.partial"

        async def interrupted(*args):
            partial.write_bytes(b"partial output")
            raise RuntimeError("fusion interrupted")

        with patch("adaptible.wrap.lmstudio.run_command", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                await self.runtime.stage(directory)
        self.assertFalse(partial.exists())
        self.assertEqual(self.base.read_bytes(), b"fixture")

    async def test_load_failure_can_restore_base(self):
        await self.runtime.discover()
        await self.runtime.restore(None)
        self.fail_load = "bad-model"
        import lmstudio

        with self.assertRaises(lmstudio.LMStudioError):
            await self.runtime._ensure("bad-model")
        self.assertIsNone(self.runtime.loaded)
        self.assertFalse(self.instances)
        await self.runtime.restore(None)
        self.assertEqual(self.runtime.loaded, self.runtime.base_key)

    async def test_stage_load_failure_discards_unreturned_candidate(self):
        await self.runtime.discover()
        await self.runtime.restore(None)
        self.fail_load = f"adaptible/{self.runtime.prefix}-candidate"
        import lmstudio

        with self.assertRaises(lmstudio.LMStudioError):
            await self.candidate()
        self.assertFalse(self.instances)
        self.assertEqual(len(self.runtime.imports), 1)  # Original base import.
        self.assertFalse((self.root / "state/adapters/candidate/merged.gguf").exists())
        self.assertFalse(list(self.import_root.rglob("merged.gguf")))
        await self.runtime.restore(None)
        self.assertEqual(self.runtime.loaded, self.runtime.base_key)

    async def test_stage_import_failure_removes_only_derived_weights(self):
        directory = self.root / "state/adapters/import-failure"
        directory.mkdir(parents=True)
        (directory / "merged.gguf").write_bytes(b"large derived model")
        (directory / "adapter.gguf").write_bytes(b"small learned adapter")
        with patch.object(
            self.runtime, "_import", side_effect=RuntimeError("import failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "import failed"):
                await self.runtime.stage(directory)
        self.assertFalse((directory / "merged.gguf").exists())
        self.assertEqual(
            (directory / "adapter.gguf").read_bytes(), b"small learned adapter"
        )
        self.assertEqual(self.base.read_bytes(), b"fixture")


if __name__ == "__main__":
    unittest.main()


class NeutralSamplingTest(unittest.IsolatedAsyncioTestCase):
    """LM Studio applied its own sampler to every request.

    It is llama.cpp underneath and ships the same house defaults, so an
    unpinned request is served under repeat_penalty 1.1. A repetition penalty
    biases the argmax, so it changes greedy output: LM Studio scored `geo_001`
    4/4 where the original and two other runtimes scored 3/4, and the wrapper
    then skipped the whole repair as "current answers already pass the
    re-asks". The original decodes with `make_sampler(temp=...)`, which applies
    no penalty and no truncation.
    """

    async def test_the_payload_pins_the_samplers_the_original_leaves_alone(self):
        from adaptible.wrap.runtime import NEUTRAL_SAMPLING

        runtime = LMStudio.__new__(LMStudio)
        runtime.base_key = "base"
        runtime.active = None
        runtime.loaded = "base"
        runtime.instance = "instance"
        body = runtime.payload({"messages": []})
        for key, value in NEUTRAL_SAMPLING.items():
            self.assertEqual(body[key], value, key)
        self.assertEqual(body["repeat_penalty"], 1.0)

    async def test_an_explicit_request_still_wins(self):
        runtime = LMStudio.__new__(LMStudio)
        runtime.base_key = "base"
        runtime.active = None
        runtime.loaded = "base"
        runtime.instance = "instance"
        body = runtime.payload({"messages": [], "top_k": 40})
        self.assertEqual(body["top_k"], 40)

    def test_the_prompt_cache_is_off_so_a_seeded_draw_reproduces(self):
        """Defect 11 on a fourth runtime, measured the same way.

        The same seed and prompt returned two different samples through LM
        Studio while every other runtime reproduced. It is llama.cpp
        underneath, so a reused KV prefix is evaluated in a different batch
        shape and a temperature draw lands elsewhere.
        """
        runtime = LMStudio.__new__(LMStudio)
        runtime.base_key = "base"
        runtime.active = None
        runtime.loaded = "base"
        runtime.instance = "instance"
        self.assertIs(runtime.payload({"messages": []})["cache_prompt"], False)
        self.assertIs(
            runtime.payload({"messages": [], "cache_prompt": True})["cache_prompt"],
            True,
        )


class LMStudioPrecisionTest(unittest.IsolatedAsyncioTestCase):
    """LM Studio must load with an f32 KV cache and the other two settings.

    Its REST load rejects every KV cache key, so the wrapper loads through the
    SDK API. Measured on the real checkpoint: with an f16 KV cache LM Studio
    departs from the original at character 191 of one of eight generations,
    byte-identical to llama.cpp run with an f16 KV cache; with these settings
    that generation is identical to the original.
    """

    async def test_every_load_asks_for_full_precision(self):
        from unittest.mock import MagicMock

        from adaptible.wrap.lmstudio import LM_STUDIO_FULL_PRECISION

        loads = []

        class FakeClient:
            def __init__(self, host):
                self.host = host
                self.llm = MagicMock()

                async def load_new_instance(key, config):
                    loads.append((host, key, config))
                    return MagicMock(identifier=key + "-instance")

                self.llm.load_new_instance = load_new_instance

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        runtime = LMStudio.__new__(LMStudio)
        runtime.url = "http://127.0.0.1:1234"
        runtime.context_size = 8192
        with patch("lmstudio.AsyncClient", FakeClient):
            instance = await runtime._load_instance("some-model")

        self.assertEqual(instance, "some-model-instance")
        self.assertEqual(len(loads), 1)
        host, key, config = loads[0]
        self.assertEqual(host, "127.0.0.1:1234")
        self.assertEqual(config["context_length"], 8192)
        for name, value in LM_STUDIO_FULL_PRECISION.items():
            self.assertEqual(config[name], value, name)
        self.assertEqual(config["llama_k_cache_quantization_type"], "f32")
        self.assertEqual(config["llama_v_cache_quantization_type"], "f32")
        self.assertIs(config["flash_attention"], False)
        self.assertEqual(config["eval_batch_size"], 8)
