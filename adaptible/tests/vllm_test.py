"""Documented vLLM HTTP/lifecycle contracts; no vLLM install or weights needed."""

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from adaptible._src.wrap.vllm import VLLM


class VLLMTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = self.root / "qwen2-existing"
        self.model.mkdir()
        self.runtime = VLLM(self.model, self.root / "state", executable="vllm")
        await self.runtime.client.aclose()
        self.requests = []
        self.fail_load = False
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.respond)
        )
        self.addAsyncCleanup(self.runtime.close)
        self.launch = mock.patch.object(self.runtime, "launch", new=mock.AsyncMock())
        self.launch.start()
        self.addCleanup(self.launch.stop)

    def respond(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        if request.url.path == "/v1/load_lora_adapter" and self.fail_load:
            return httpx.Response(400, json={"error": "LoRA is not enabled"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "A model reply."}}]},
            )
        return httpx.Response(200, text="Success")

    def candidate(self, name):
        path = self.root / "state" / "adapters" / name
        adapter = path / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 8})
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"contract fixture")
        return path

    async def test_stage_routes_candidate_only_after_explicit_acceptance(self):
        directory = self.candidate("candidate")
        handle = await self.runtime.stage(directory)
        self.assertEqual(
            self.requests[-1],
            (
                "/v1/load_lora_adapter",
                {"lora_name": handle, "lora_path": str(directory / "adapter")},
            ),
        )
        self.assertEqual(self.runtime.payload({})["model"], self.model.name)
        self.assertEqual(self.runtime.payload({}, handle=handle)["model"], handle)
        await self.runtime.restore(handle)
        self.assertEqual(self.runtime.payload({})["model"], handle)
        self.assertEqual(
            self.runtime.payload({}, frozen=True)["model"], self.model.name
        )
        await self.runtime.discard(handle)
        self.assertEqual(len(self.requests), 1, "Never unload an active adapter")

    async def test_training_restart_reloads_previous_on_rejection(self):
        previous = await self.runtime.stage(self.candidate("accepted"))
        await self.runtime.restore(previous)
        await self.runtime.release_for_training()
        self.assertEqual(self.runtime.loaded, set())
        candidate = await self.runtime.stage(self.candidate("candidate"))
        await self.runtime.restore(previous)
        await self.runtime.discard(candidate)
        self.assertEqual(self.runtime.active, previous)
        self.assertEqual(self.runtime.loaded, {previous})
        self.assertEqual(
            self.requests[-1],
            ("/v1/unload_lora_adapter", {"lora_name": candidate}),
        )
        self.assertEqual(
            sum(
                path == "/v1/load_lora_adapter" and body["lora_name"] == previous
                for path, body in self.requests
            ),
            2,
        )

    async def test_unavailable_dynamic_lora_fails_without_activating(self):
        self.fail_load = True
        with self.assertRaisesRegex(RuntimeError, "LoRA and runtime loading"):
            await self.runtime.stage(self.candidate("bad"))
        self.assertIsNone(self.runtime.active)
        self.assertFalse(self.runtime.loaded)
        self.assertFalse(self.runtime.adapters)
        self.assertEqual(self.requests[-1][0], "/v1/unload_lora_adapter")

    async def test_cancelled_stage_stops_child_without_new_admin_request(self):
        with (
            mock.patch.object(
                self.runtime,
                "_load",
                new=mock.AsyncMock(side_effect=asyncio.CancelledError),
            ),
            mock.patch.object(self.runtime, "stop", new=mock.AsyncMock()) as stop,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self.runtime.stage(self.candidate("cancelled"))
            stop.assert_awaited_once()
        self.assertFalse(self.requests)

    async def test_no_foreign_adapter_control_or_unloaded_routing(self):
        await self.runtime.discard("someone-elses-adapter")
        self.assertFalse(self.requests)
        with self.assertRaisesRegex(ValueError, "not owned"):
            await self.runtime.restore("someone-elses-adapter")
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            self.runtime.payload({}, handle="someone-elses-adapter")

    async def test_complete_preserves_generation_options_and_schema(self):
        handle = await self.runtime.stage(self.candidate("accepted"))
        await self.runtime.restore(handle)
        schema = {"type": "json_schema", "json_schema": {"name": "answer"}}
        text = await self.runtime.complete(
            [{"role": "user", "content": "Question?"}],
            frozen=True,
            temperature=0.7,
            seed=34,
            response_format=schema,
        )
        self.assertEqual(text, "A model reply.")
        body = self.requests[-1][1]
        self.assertEqual(body["model"], self.model.name)
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["seed"], 34)
        self.assertEqual(body["response_format"], schema)
        self.assertFalse(body["stream"])

    async def test_qwen3_internal_review_disables_thinking_without_changing_public_payload(
        self,
    ):
        self.runtime.architecture = "qwen3"
        await self.runtime.complete([{"role": "user", "content": "Question?"}])
        body = self.requests[-1][1]
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        public = self.runtime.payload({"reasoning_effort": "high"})
        self.assertEqual(public["reasoning_effort"], "high")
        self.assertNotIn("chat_template_kwargs", public)

    async def test_incomplete_or_wrong_rank_adapter_fails_before_http(self):
        directory = self.candidate("wrong-rank")
        (directory / "adapter" / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 16})
        )
        with self.assertRaisesRegex(ValueError, "rank-8"):
            await self.runtime.stage(directory)
        self.assertFalse(self.requests)

    async def test_launch_is_local_offline_and_stops_owned_group(self):
        self.launch.stop()
        process = mock.Mock(pid=123456, returncode=None)
        process.wait = mock.AsyncMock(return_value=0)

        def health(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": self.model.name}]})
            return httpx.Response(200)

        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(health))
        with (
            mock.patch(
                "adaptible._src.wrap.vllm.asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=process),
            ) as spawn,
            mock.patch("adaptible._src.wrap.vllm.os.killpg") as kill,
        ):
            await self.runtime.launch()
            args = spawn.call_args.args
            env = spawn.call_args.kwargs["env"]
            self.assertEqual(args[:3], ("vllm", "serve", str(self.model)))
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            self.assertIn("--enable-lora", args)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"], "True")
            self.assertEqual(env["VLLM_PLUGINS"], "")
            self.assertEqual(
                env["VLLM_CPU_KVCACHE_SPACE"],
                os.environ.get("VLLM_CPU_KVCACHE_SPACE", "1"),
            )
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            await self.runtime.release_for_training()
            self.assertEqual(
                kill.call_args_list,
                [mock.call(123456, signal.SIGTERM), mock.call(123456, signal.SIGKILL)],
            )
            self.assertIsNone(self.runtime.process)
            self.assertIsNone(self.runtime.log)

    async def test_startup_failure_cleans_child_and_reports_log(self):
        self.launch.stop()
        process = mock.Mock(pid=123456, returncode=2)
        with (
            mock.patch(
                "adaptible._src.wrap.vllm.asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=process),
            ),
            mock.patch("adaptible._src.wrap.vllm.os.killpg") as kill,
        ):
            with self.assertRaisesRegex(RuntimeError, "vllm-server.log"):
                await self.runtime.launch()
            kill.assert_any_call(123456, signal.SIGTERM)
        self.assertIsNone(self.runtime.process)
        self.assertIsNone(self.runtime.log)

    async def test_restart_rebuilds_stable_private_adapter_name(self):
        directory = self.candidate("accepted")
        handle = await self.runtime.stage(directory)
        other = VLLM(self.model, self.runtime.directory, executable="vllm")
        await other.client.aclose()
        other.client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))
        self.addAsyncCleanup(other.close)
        with mock.patch.object(other, "launch", new=mock.AsyncMock()):
            reloaded = await other.stage(directory)
            await other.restore(reloaded)
        self.assertEqual(reloaded, handle)
        self.assertEqual(other.active, handle)


@unittest.skipUnless(
    os.environ.get("ADAPTIBLE_VLLM_EXECUTABLE"),
    "Set ADAPTIBLE_VLLM_EXECUTABLE to opt into real vLLM with a tiny random model",
)
class VLLMLiveAdapterTest(unittest.IsolatedAsyncioTestCase):
    """Actual inference/training/load/rollback/restart without a model download.

    This proves numerical adapter application, not factual repair quality.
    CPU installations can bound KV RAM with VLLM_CPU_KVCACHE_SPACE=1.
    """

    async def test_local_training_changes_logits_and_base_survives_restart(self):
        import torch
        from tokenizers import pre_tokenizers
        from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM

        from adaptible._src.wrap.repair import Trainer

        with tempfile.TemporaryDirectory(prefix="adaptible-vllm-live-") as temporary:
            root = Path(temporary).resolve()
            model_path = root / "tiny-qwen2"
            vocab = {
                token: i
                for i, token in enumerate(
                    ["[UNK]", "[EOS]", *sorted(pre_tokenizers.ByteLevel.alphabet())]
                )
            }
            tokenizer = Qwen2Tokenizer(
                vocab=vocab,
                merges=[],
                unk_token="[UNK]",
                pad_token="[EOS]",
                eos_token="[EOS]",
            )
            tokenizer.chat_template = (
                "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
                "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            )
            torch.manual_seed(42)
            model = Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=len(tokenizer),
                    # vLLM's real CPU attention kernels require head_dim >=32.
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    max_position_embeddings=256,
                    eos_token_id=1,
                    pad_token_id=1,
                )
            )
            model.save_pretrained(model_path, safe_serialization=True)
            tokenizer.save_pretrained(model_path)
            del model
            runtime = VLLM(
                model_path,
                root / "state",
                executable=os.environ["ADAPTIBLE_VLLM_EXECUTABLE"],
                context_size=256,
                max_tokens=8,
            )
            messages = [{"role": "user", "content": "Question"}]

            async def probabilities(handle=None, frozen=False):
                response = await runtime.client.post(
                    runtime.url + "/v1/chat/completions",
                    json=runtime.payload(
                        dict(
                            messages=messages,
                            max_tokens=1,
                            temperature=0,
                            seed=0,
                            logprobs=True,
                            top_logprobs=20,
                        ),
                        handle=handle,
                        frozen=frozen,
                    ),
                )
                self.assertEqual(response.status_code, 200, response.text)
                return response.json()["choices"][0]["logprobs"]["content"][0]

            candidate = root / "state" / "adapters" / "candidate"
            try:
                await runtime.restore(None)
                baseline = await probabilities(frozen=True)
                await runtime.release_for_training()
                stats = await Trainer().train(model_path, messages, "Answer", candidate)
                self.assertGreater(stats["steps"], 0)
                handle = await runtime.stage(candidate)
                adapted = await probabilities(handle=handle)
                self.assertNotEqual(baseline, adapted, "Real LoRA must change logits")
                self.assertEqual(baseline, await probabilities(frozen=True))
                await runtime.restore(None)
                await runtime.discard(handle)
                self.assertEqual(baseline, await probabilities())
                handle = await runtime.stage(candidate)
                await runtime.restore(handle)
                await runtime.release_for_training()
                await runtime.restore(handle)
                self.assertEqual(adapted, await probabilities())
            except BaseException:
                logs = Path(tempfile.mkdtemp(prefix="adaptible-vllm-failure-"))
                for path in (
                    root / "state" / "vllm-server.log",
                    candidate / "train.log",
                ):
                    if path.exists():
                        shutil.copyfile(path, logs / path.name)
                print(f"vLLM live-test logs: {logs}", file=sys.stderr)
                raise
            finally:
                await runtime.close()


if __name__ == "__main__":
    unittest.main()


class ReasoningParserSelectionTest(unittest.TestCase):
    """A reasoning model needs its thought separated from its answer.

    Gating the parser on `architecture == "qwen3"` left a qwen2-architecture
    distilled reasoner with its whole thought in `content`. A 7,599-character
    unterminated ramble that ended on the wrong answer then scored as correct,
    because the key term appeared somewhere inside it. The experiment counts
    exactly that as a miss: its `closed(r)` requires a closed think block.
    """

    def runtime(self, template, architecture):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        blob = directory / "model"
        blob.mkdir()
        (blob / "config.json").write_text(
            json.dumps({"architectures": ["Qwen2ForCausalLM"]})
        )
        if template is not None:
            (blob / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": template})
            )
        with mock.patch("shutil.which", return_value="/bin/vllm"):
            runtime = VLLM(str(blob), directory / "state", executable="/bin/vllm")
        runtime.architecture = architecture
        return runtime

    def test_a_thinking_template_selects_the_deepseek_parser(self):
        runtime = self.runtime("...<think>\n...</think>...", "qwen2")
        self.assertTrue(runtime.prefills_thinking())
        self.assertIn("deepseek_r1", runtime.command())

    def test_a_plain_template_selects_no_parser(self):
        runtime = self.runtime("{{ prompt }}", "qwen2")
        self.assertFalse(runtime.prefills_thinking())
        self.assertNotIn("--reasoning-parser", runtime.command())

    def test_a_missing_tokenizer_config_is_not_an_error(self):
        runtime = self.runtime(None, "qwen2")
        self.assertFalse(runtime.prefills_thinking())

    def test_qwen3_keeps_its_own_parser(self):
        runtime = self.runtime("...<think>...</think>...", "qwen3")
        command = runtime.command()
        self.assertIn("qwen3", command)
        self.assertNotIn("deepseek_r1", command)


class CheckpointPrecisionTest(unittest.TestCase):
    """vLLM must serve the weights it was given, not the platform's preference.

    `--dtype auto` resolves a float32 checkpoint down to bfloat16 on ARM and
    reports it in one line: "Downcasting torch.float32 to torch.bfloat16". That
    made vLLM the only runtime in a five-way comparison not reading the f32
    weights every other runtime read, and a greedy decode against the identical
    checkpoint first disagreed with the original about 35 tokens in -- which is
    what a 7-bit mantissa against 23 predicts.
    """

    def runtime(self, config):
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "checkpoint"
            blob.mkdir()
            if config is not None:
                (blob / "config.json").write_text(json.dumps(config))
            runtime = VLLM.__new__(VLLM)
            runtime.blob = blob
            return runtime.checkpoint_dtype()

    def test_the_checkpoints_own_precision_is_pinned(self):
        self.assertEqual(self.runtime({"torch_dtype": "float32"}), "float32")
        self.assertEqual(self.runtime({"torch_dtype": "bfloat16"}), "bfloat16")

    def test_auto_only_when_the_checkpoint_declares_nothing(self):
        self.assertEqual(self.runtime({}), "auto")
        self.assertEqual(self.runtime({"torch_dtype": None}), "auto")
        self.assertEqual(self.runtime(None), "auto")

    def test_the_launch_pins_dtype_and_disables_prefix_caching(self):
        with tempfile.TemporaryDirectory() as directory:
            blob = Path(directory) / "checkpoint"
            blob.mkdir()
            (blob / "config.json").write_text(json.dumps({"torch_dtype": "float32"}))
            runtime = VLLM.__new__(VLLM)
            runtime.blob = blob
            runtime.executable = "vllm"
            runtime.name = "m"
            runtime.port = 1
            runtime.context_size = 8192
            runtime.architecture = "qwen2"
            args = runtime.command()
            self.assertIn("--dtype", args)
            self.assertEqual(args[args.index("--dtype") + 1], "float32")
            self.assertIn("--no-enable-prefix-caching", args)
