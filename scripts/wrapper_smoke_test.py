"""CLI smoke orchestration contracts and explicitly opted-in live runtimes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from scripts import wrapper_smoke


class WrapperSmokeContractTest(unittest.TestCase):
    def test_only_real_sources_and_nonfailed_review_allow_smoke_success(self):
        real_client = httpx.Client
        cases = [
            ("skipped", True, "passed"),
            ("skipped", False, "failed"),
            ("failed", True, "failed"),
        ]
        for status, sources, expected in cases:
            with (
                self.subTest(status=status, sources=sources),
                tempfile.TemporaryDirectory() as temporary,
            ):
                row = dict(
                    id=1,
                    note="",
                    status=status,
                    reason="fixture outcome",
                    references=dict(
                        kind="web",
                        sources=(
                            [dict(url="https://example.org/fact", text="Evidence")]
                            if sources
                            else []
                        ),
                    ),
                )

                def respond(request):
                    path = request.url.path
                    if path == "/status":
                        return httpx.Response(200, json=dict(status="up"))
                    if path == "/v1/models":
                        return httpx.Response(200, json=dict(data=[dict(id="fixture")]))
                    if path == "/history":
                        return httpx.Response(200, json=dict(history=[row]))
                    if path == "/sync":
                        return httpx.Response(200, json=dict(reviews=[row]))
                    self.assertEqual(path, "/feedback")
                    self.assertEqual(
                        json.loads(request.content),
                        dict(interaction_idx=1, thumbs="down"),
                    )
                    return httpx.Response(200, json={})

                args = wrapper_smoke.parser().parse_args(
                    ["vllm", "/existing/model", "--output-dir", temporary]
                )
                with (
                    mock.patch.object(wrapper_smoke, "WrapperProcess") as process,
                    mock.patch.object(
                        wrapper_smoke, "chat", return_value=("answer", 1)
                    ),
                    mock.patch.object(
                        wrapper_smoke, "stream_chat", return_value="answer"
                    ),
                    mock.patch.object(
                        wrapper_smoke.httpx,
                        "Client",
                        side_effect=lambda **kwargs: real_client(
                            transport=httpx.MockTransport(respond), **kwargs
                        ),
                    ),
                ):
                    process.return_value.url = "http://test"
                    process.return_value.command = [
                        "python",
                        "-m",
                        "adaptible",
                        "wrap",
                        "vllm",
                    ]
                    report = wrapper_smoke.run_smoke(args)
                    self.assertEqual(report["smoke_status"], expected)
                    self.assertTrue(args.web_search)
                    self.assertEqual(
                        process.return_value.start.call_count,
                        2 if expected == "passed" else 1,
                    )
                    self.assertTrue(
                        (Path(report["directory"]) / "report.json").is_file()
                    )


class WrapperSmokeLiveTest(unittest.TestCase):
    """Opt-in commands use installed models; no runtime or checkpoint is mocked."""

    def run_backend(self, service, variable, flag=None, executable=None):
        model = os.environ.get(variable)
        if not model:
            self.skipTest(
                f"Set {variable} to run the actual CLI and automatic web lookup"
            )
        argv = [service]
        if model == "tiny" and service == "vllm":
            argv += ["--tiny-vllm"]
        else:
            argv += [model]
        if flag and os.environ.get(executable):
            argv += [flag, os.environ[executable]]
        args = wrapper_smoke.parser().parse_args(argv)
        report = wrapper_smoke.run_smoke(args)
        self.assertEqual(report["smoke_status"], "passed", json.dumps(report, indent=2))

    def test_ollama(self):
        self.run_backend("ollama", "ADAPTIBLE_SMOKE_OLLAMA_MODEL")

    def test_llama_cpp(self):
        self.run_backend(
            "llama-cpp",
            "ADAPTIBLE_SMOKE_LLAMA_GGUF",
            "--llama-server",
            "ADAPTIBLE_TEST_LLAMA_SERVER",
        )

    def test_lm_studio(self):
        self.run_backend(
            "lm-studio", "ADAPTIBLE_SMOKE_LM_STUDIO_GGUF", "--lms", "ADAPTIBLE_TEST_LMS"
        )

    def test_vllm(self):
        self.run_backend(
            "vllm",
            "ADAPTIBLE_SMOKE_VLLM_MODEL",
            "--vllm-server",
            "ADAPTIBLE_VLLM_EXECUTABLE",
        )


if __name__ == "__main__":
    unittest.main()
