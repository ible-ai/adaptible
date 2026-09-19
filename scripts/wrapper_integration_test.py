"""Opt-in real-runtime tests. Never downloads a model; skipped in ordinary CI.

ADAPTIBLE_TEST_OLLAMA_MODEL=qwen2.5:0.5b python -m unittest scripts.wrapper_integration_test -v
ADAPTIBLE_TEST_LLAMA_GGUF=/path/model.gguf python -m unittest scripts.wrapper_integration_test -v
ADAPTIBLE_TEST_LM_STUDIO_GGUF=/path/model.gguf python -m unittest scripts.wrapper_integration_test -v
ADAPTIBLE_TEST_VLLM_MODEL=/path/hf-checkpoint python -m unittest scripts.wrapper_integration_test -v

vLLM needs a working local vLLM runtime on supported hardware. Tests never
download either a checkpoint or a runtime. Optional binary overrides are
ADAPTIBLE_TEST_LLAMA_SERVER, ADAPTIBLE_TEST_LMS, ADAPTIBLE_TEST_VLLM_SERVER.
Each test owns its wrapper process and fresh state. Base-already-correct and
rejected repairs fail the integration assertion; neither is a successful demo.
"""

import os
import unittest

from scripts.wrapper_demo import parser, run_demo


class LiveWrapperTest(unittest.TestCase):
    def check_runtime(self, service, model):
        argv = [service, model]
        if os.environ.get("ADAPTIBLE_TEST_WEB_SEARCH") == "1":
            argv.append("--web-search")
        service_options = {
            "ollama": (("ADAPTIBLE_TEST_OLLAMA_URL", "--upstream"),),
            "llama-cpp": (("ADAPTIBLE_TEST_LLAMA_SERVER", "--llama-server"),),
            "lm-studio": (
                ("ADAPTIBLE_TEST_LM_STUDIO_URL", "--upstream"),
                ("ADAPTIBLE_TEST_LMS", "--lms"),
            ),
            "vllm": (("ADAPTIBLE_TEST_VLLM_SERVER", "--vllm-server"),),
        }
        for env, flag in service_options[service] + (
            ("ADAPTIBLE_TEST_OUTPUT_DIR", "--output-dir"),
            ("ADAPTIBLE_TEST_TIMEOUT", "--timeout"),
            ("ADAPTIBLE_TEST_QUESTION", "--question"),
            ("ADAPTIBLE_TEST_REFERENCE", "--reference"),
            ("ADAPTIBLE_TEST_EXPECTED", "--expected"),
            ("ADAPTIBLE_TEST_CONTEXT_SIZE", "--context-size"),
        ):
            if os.environ.get(env):
                argv += [flag, os.environ[env]]
        report = run_demo(parser().parse_args(argv))
        # Already-correct baselines are not passes: training must be exercised.
        self.assertEqual(report["integration_status"], "passed", str(report))

    @unittest.skipUnless(
        os.environ.get("ADAPTIBLE_TEST_OLLAMA_MODEL"),
        "Set ADAPTIBLE_TEST_OLLAMA_MODEL to opt in",
    )
    def test_ollama_feedback_train_stream_and_restart(self):
        self.check_runtime("ollama", os.environ["ADAPTIBLE_TEST_OLLAMA_MODEL"])

    @unittest.skipUnless(
        os.environ.get("ADAPTIBLE_TEST_LLAMA_GGUF"),
        "Set ADAPTIBLE_TEST_LLAMA_GGUF to opt in",
    )
    def test_llama_cpp_feedback_train_stream_and_restart(self):
        self.check_runtime("llama-cpp", os.environ["ADAPTIBLE_TEST_LLAMA_GGUF"])

    @unittest.skipUnless(
        os.environ.get("ADAPTIBLE_TEST_LM_STUDIO_GGUF"),
        "Set ADAPTIBLE_TEST_LM_STUDIO_GGUF to opt in",
    )
    def test_lm_studio_feedback_train_stream_and_restart(self):
        self.check_runtime("lm-studio", os.environ["ADAPTIBLE_TEST_LM_STUDIO_GGUF"])

    @unittest.skipUnless(
        os.environ.get("ADAPTIBLE_TEST_VLLM_MODEL"),
        "Set ADAPTIBLE_TEST_VLLM_MODEL to opt in on supported vLLM hardware",
    )
    def test_vllm_feedback_train_stream_and_restart(self):
        self.check_runtime("vllm", os.environ["ADAPTIBLE_TEST_VLLM_MODEL"])
