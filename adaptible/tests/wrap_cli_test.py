"""No-model contracts for the four-runtime CLI and honest live-demo reports."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from adaptible._src.wrap.__main__ import parser as wrapper_parser, serve
from adaptible._src.wrap.model_source import fingerprint_base
from adaptible._src.wrap.store import Store
from scripts import wrapper_demo as demo


class WrapperDemoContractTest(unittest.TestCase):
    def test_all_four_user_commands_default_to_verified_thinking_budgets(self):
        for service in ("ollama", "llama-cpp", "lm-studio", "vllm"):
            with self.subTest(service=service):
                args = wrapper_parser().parse_args([service, "existing"])
                self.assertEqual(args.max_tokens, 2048)
                self.assertEqual(args.context_size, 8192)
                explicit = wrapper_parser().parse_args(
                    [
                        service,
                        "existing",
                        "--max-tokens",
                        "512",
                        "--context-size",
                        "4096",
                    ]
                )
                self.assertEqual(explicit.max_tokens, 512)
                self.assertEqual(explicit.context_size, 4096)

    def test_all_four_demo_commands_reach_the_wrapper_parser(self):
        cases = (
            ("ollama", "existing:0.5b", ["--upstream", "http://localhost:11435"]),
            (
                "llama-cpp",
                "/models/existing.gguf",
                ["--llama-server", "/bin/llama-server"],
            ),
            (
                "lm-studio",
                "/models/existing.gguf",
                ["--lms", "/bin/lms", "--upstream", "http://localhost:1235"],
            ),
            ("vllm", "/models/existing-hf", ["--vllm-server", "/bin/vllm"]),
        )
        with tempfile.TemporaryDirectory() as temp:
            for service, model, extra in cases:
                with self.subTest(service=service):
                    args = demo.parser().parse_args([service, model, *extra])
                    process = demo.WrapperProcess(args, Path(temp))
                    parsed = wrapper_parser().parse_args(process.command[4:])
                    self.assertEqual((parsed.service, parsed.model), (service, model))
                    self.assertTrue(
                        parsed.no_web_search, "Fixed-reference demos stay offline"
                    )
                    for flag, value in zip(extra[::2], extra[1::2]):
                        self.assertEqual(
                            getattr(parsed, flag[2:].replace("-", "_")), value
                        )
                    self.assertEqual(process.env["HF_HUB_OFFLINE"], "1")
                    self.assertEqual(process.env["TRANSFORMERS_OFFLINE"], "1")
                    self.assertEqual(
                        Path(process.env["HF_HOME"]), Path(temp) / "hf-cache"
                    )

    def test_upstream_default_is_resolved_by_the_selected_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            for service in ("ollama", "llama-cpp", "lm-studio", "vllm"):
                args = demo.parser().parse_args([service, "existing"])
                command = demo.WrapperProcess(args, Path(temp)).command
                self.assertNotIn("--upstream", command)

    def test_web_search_is_enabled_by_default_for_all_four_services(self):
        for service in ("ollama", "llama-cpp", "lm-studio", "vllm"):
            self.assertFalse(
                wrapper_parser().parse_args([service, "existing"]).no_web_search
            )
            self.assertTrue(
                wrapper_parser()
                .parse_args([service, "existing", "--no-web-search"])
                .no_web_search
            )

    def test_live_web_demo_cannot_use_a_prepared_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            for service in ("ollama", "llama-cpp", "lm-studio", "vllm"):
                args = demo.parser().parse_args([service, "existing", "--web-search"])
                command = demo.WrapperProcess(args, Path(temp)).command
                self.assertNotIn("--documents", command)
                self.assertNotIn("--no-web-search", command)

    def test_server_bounds_graceful_shutdown_before_runtime_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            blob = root / "existing.gguf"
            args = wrapper_parser().parse_args(
                ["llama-cpp", str(blob), "--state-dir", str(root / "state")]
            )
            runtime = mock.Mock()
            runtime.serving_template_digest = None
            runtime.name = "existing"
            runtime.discover = mock.AsyncMock(return_value=blob)
            runtime.detect_reasoning = mock.AsyncMock(return_value=False)
            runtime.close = mock.AsyncMock()
            server = mock.Mock(serve=mock.AsyncMock())
            with (
                mock.patch(
                    "adaptible._src.wrap.__main__.LlamaCpp", return_value=runtime
                ),
                mock.patch(
                    "adaptible._src.wrap.__main__.read_architecture",
                    return_value="qwen2",
                ),
                mock.patch(
                    "adaptible._src.wrap.__main__.repairable",
                    return_value=(True, ""),
                ),
                mock.patch(
                    "adaptible._src.wrap.__main__.fingerprint_base",
                    return_value="base-digest",
                ),
                mock.patch("uvicorn.Config") as config,
                mock.patch("uvicorn.Server", return_value=server) as server_factory,
                mock.patch("builtins.print"),
            ):
                asyncio.run(serve(args))
            self.assertEqual(config.call_args.kwargs["timeout_graceful_shutdown"], 5)
            server_factory.assert_called_once_with(config.return_value)
            server.serve.assert_awaited_once()
            runtime.close.assert_awaited_once()

    def run_report_fixture(self, directory, *, baseline, review):
        requests = []

        def respond(request):
            requests.append((request.method, request.url.path))
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "existing"}]})
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(
                    200,
                    headers={"X-Interaction-Idx": "1"},
                    json={
                        "model": "existing",
                        "choices": [{"message": {"content": baseline}}],
                    },
                )
            if request.url.path == "/feedback":
                return httpx.Response(200, json={"status": "queued"})
            if request.url.path == "/sync":
                return httpx.Response(200, json={"reviews": [review]})
            raise AssertionError(f"Unexpected request {request.url}")

        args = demo.parser().parse_args(
            ["llama-cpp", "/existing/model.gguf", "--output-dir", str(directory)]
        )
        client = httpx.Client(
            base_url="http://wrapper", transport=httpx.MockTransport(respond)
        )
        with (
            mock.patch.object(demo.WrapperProcess, "start") as start,
            mock.patch.object(demo.WrapperProcess, "stop") as stop,
            mock.patch.object(demo.httpx, "Client", return_value=client),
        ):
            report = demo.run_demo(args)
        start.assert_called_once()
        stop.assert_called_once()
        written = json.loads((Path(report["directory"]) / "report.json").read_text())
        self.assertEqual(report, written)
        return report, requests

    def test_already_correct_is_inconclusive_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as temp:
            report, requests = self.run_report_fixture(
                temp, baseline="Casablanca.", review={}
            )
        self.assertEqual(report["integration_status"], "inconclusive")
        self.assertNotIn(("POST", "/feedback"), requests)

    def test_rejected_repair_is_a_failure_even_if_training_ran(self):
        with tempfile.TemporaryDirectory() as temp:
            report, requests = self.run_report_fixture(
                temp,
                baseline="Rabat.",
                review={"interaction_idx": 1, "status": "rejected", "steps": 4},
            )
        self.assertEqual(report["integration_status"], "failed")
        self.assertIn("No adapter accepted", report["error"])
        self.assertIn(("POST", "/feedback"), requests)

    def test_worker_failure_remains_visible_in_the_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report, _ = self.run_report_fixture(
                temp,
                baseline="Rabat.",
                review={
                    "interaction_idx": 1,
                    "status": "failed",
                    "reason": "worker exited",
                },
            )
        self.assertEqual(report["integration_status"], "failed")
        self.assertIn("worker exited", report["error"])

    def test_training_artifact_check_handles_both_checkpoint_formats(self):
        for kind in ("gguf", "hf"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / ("base.gguf" if kind == "gguf" else "base")
                if kind == "gguf":
                    source.write_bytes(b"test-base-weights")
                else:
                    source.mkdir()
                    (source / "config.json").write_text('{"model_type": "qwen2"}')
                    (source / "model.safetensors").write_bytes(b"test-base-weights")
                candidate = root / "state" / "adapters" / "candidate"
                (candidate / "adapter").mkdir(parents=True)
                for name in ("adapter_config.json", "adapter_model.safetensors"):
                    (candidate / "adapter" / name).write_bytes(b"test-adapter")
                (candidate / "stats.json").write_text('{"steps": 1, "loss": 0.1}')
                link = candidate / ("model.gguf" if kind == "gguf" else "model")
                link.symlink_to(source, target_is_directory=kind == "hf")
                if kind == "gguf":
                    (candidate / "adapter.gguf").write_bytes(b"test-export")
                state = root / "state"
                store = Store(state, "service:model:" + fingerprint_base(source))
                store.set("accepted", {"directory": str(candidate)})
                store.close()
                report = {}
                demo.verify_training(state, report)
                self.assertEqual(report["base_file"], str(source.resolve()))
                self.assertEqual(report["training"]["steps"], 1)
                # New workers use a small provenance record on filesystems
                # without symlinks; older symlink-based reports remain valid.
                link.unlink()
                metadata = dict(
                    schema_version=1,
                    path=str(source.resolve()),
                    kind=kind,
                    sha256=fingerprint_base(source),
                )
                record = candidate / "source.json"
                record.write_text(json.dumps(metadata))
                (candidate / "job.json").write_text(json.dumps(dict(blob=str(source))))
                with mock.patch.object(
                    Path, "symlink_to", side_effect=AssertionError("No symlinks")
                ):
                    demo.verify_training(state, report)
                self.assertEqual(report["base_file"], str(source.resolve()))
                record.write_text(json.dumps({**metadata, "sha256": "wrong"}))
                with self.assertRaisesRegex(
                    AssertionError, "Base weights changed since training"
                ):
                    demo.verify_training(state, {})
                record.write_text(json.dumps({**metadata, "path": str(candidate)}))
                with self.assertRaisesRegex(AssertionError, "copied a base checkpoint"):
                    demo.verify_training(state, {})
                record.write_text(json.dumps(metadata))
                changed = source if kind == "gguf" else source / "model.safetensors"
                changed.write_bytes(b"changed-base-weights")
                with self.assertRaisesRegex(AssertionError, "Base weights changed"):
                    demo.verify_training(state, {})


if __name__ == "__main__":
    unittest.main()
