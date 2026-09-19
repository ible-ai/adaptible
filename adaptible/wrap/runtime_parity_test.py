"""The four runtimes must do the same thing at every step of the repair loop.

Generation and adapter application are the only genuinely runtime-specific
parts. Everything the wrapper puts around them -- what it asks the model for,
how it reads the reply, what it records as complete, what it trains, and what
it judges -- has to be identical whichever runtime is in use, or the same
experiment measures different things on different infrastructure.

Every divergence these tests pin was a real defect found by running the
experiment instead: a reasoning parser selected only for one architecture
(so one runtime returned its whole thought as the answer), a native route
that never asked for the thought, a probe issued before the server it probes
existed. Each one cost hours and produced numbers that had to be withdrawn.

Each test drives every runtime through the same scenario with a mocked
transport and asserts the observable behaviour matches.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptible.wrap.lmstudio import LMStudio
from adaptible.wrap.runtime import LlamaCpp, Ollama
from adaptible.wrap.thinking import completion_details
from adaptible.wrap.vllm import VLLM

MESSAGES = [dict(role="user", content="What is the capital of Australia?")]
REASONING = "Okay, the note says Canberra."
ANSWER = "The capital of Australia is Canberra."


def _checkpoint(directory):
    """A minimal local HF checkpoint, enough for the runtimes that want one."""
    blob = Path(directory) / "model"
    blob.mkdir(parents=True, exist_ok=True)
    (blob / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"]})
    )
    (blob / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "x<think>\ny</think>z"})
    )
    (blob / "tokenizer.json").write_text("{}")
    return blob


class RuntimeParityTest(unittest.TestCase):
    """Same scenario, every runtime, identical observable behaviour."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.blob = _checkpoint(self.root)

    def runtimes(self):
        """Every runtime, constructed the way the wrapper constructs them."""
        made = {}
        with mock.patch("adaptible.wrap.runtime.shutil.which", return_value="x"):
            made["ollama"] = Ollama(
                "deepseek-r1", self.root / "ollama", url="http://127.0.0.1:11434"
            )
        made["llama-cpp"] = LlamaCpp(
            str(self.blob / "config.json"), self.root / "llamacpp", executable="x"
        )
        with mock.patch("adaptible.wrap.lmstudio.shutil.which", return_value="x"):
            made["lm-studio"] = LMStudio(
                str(self.blob / "config.json"),
                self.root / "lmstudio",
                url="http://127.0.0.1:1234",
                executable="x",
            )
        with mock.patch("shutil.which", return_value="x"):
            made["vllm"] = VLLM(str(self.blob), self.root / "vllm", executable="x")
        for runtime in made.values():
            runtime.architecture = "qwen2"
            runtime.always_reasons = True
        # Serving state each runtime would hold once its model is up. The
        # parity questions are about what the wrapper sends and reads, not
        # about loading.
        made["lm-studio"].base_key = "base"
        made["lm-studio"].loaded = "base"
        made["lm-studio"].instance = "base-instance"
        made["vllm"].loaded = {"adapter-handle"}
        made["vllm"].adapters = {"adapter-handle": self.blob}
        self.addCleanup(
            lambda: [
                r.client.aclose.__self__ and None for r in made.values()
            ]  # clients are closed by the event loop in async tests
        )
        return made

    # --- what the wrapper asks the model for -------------------------------

    def test_every_runtime_sends_the_same_generation_controls(self):
        """Temperature, seed and budget must reach every provider alike.

        A runtime that silently drops the seed makes its candidates
        unreproducible, and one that drops the budget scores a different
        number of tokens than the others.
        """
        seen = {}
        for name, runtime in self.runtimes().items():
            body = runtime.payload(
                dict(
                    messages=MESSAGES,
                    stream=False,
                    temperature=0.7,
                    seed=1000,
                    max_tokens=1024,
                ),
                frozen=True,
            )
            body = runtime.normalize_payload(body)
            seen[name] = {
                key: body.get(key)
                for key in ("messages", "temperature", "seed", "max_tokens", "stream")
            }
        reference = seen["llama-cpp"]
        for name, body in seen.items():
            self.assertEqual(body, reference, f"{name} sends different controls")

    def test_no_runtime_injects_a_system_prompt(self):
        """The experiment sends one user turn and nothing else."""
        for name, runtime in self.runtimes().items():
            body = runtime.normalize_payload(
                runtime.payload(dict(messages=MESSAGES), frozen=True)
            )
            self.assertEqual(body["messages"], MESSAGES, f"{name} altered the messages")

    # --- how the wrapper reads the reply -----------------------------------

    def test_a_completed_thought_is_read_identically(self):
        """Each provider frames reasoning differently; the wrapper must not.

        vLLM without a reasoning parser returned the entire thought as the
        answer, so a rambling non-answer scored as correct on that runtime
        and as a miss everywhere else.
        """
        shapes = {
            "openai_split": dict(content=ANSWER, reasoning_content=REASONING),
            "openai_alias": dict(content=ANSWER, reasoning=REASONING),
            "native_ollama": dict(content=ANSWER, thinking=REASONING),
            "inline_tags": dict(content=f"<think>\n{REASONING}\n</think>\n\n{ANSWER}"),
        }
        read = {
            name: completion_details(message, "stop")
            for name, message in shapes.items()
        }
        for name, details in read.items():
            self.assertEqual(details["content"], ANSWER, f"{name} answer differs")
            self.assertIn(REASONING, details["reasoning"], f"{name} reasoning lost")
            self.assertTrue(details["complete"], f"{name} not complete")
            self.assertTrue(details["framing_valid"], f"{name} framing invalid")

    def test_an_unfinished_thought_is_rejected_identically(self):
        """A thought that never closes is not an answer, on any runtime."""
        shapes = [
            dict(content="", reasoning_content=REASONING),
            dict(content=f"<think>\n{REASONING}"),
        ]
        for message in shapes:
            details = completion_details(message, "length")
            self.assertFalse(details["complete"], f"{message} counted as complete")

    # --- what the wrapper trains -------------------------------------------

    def test_every_runtime_declares_the_same_training_options(self):
        """Training is one shared worker; no runtime may steer it differently."""
        options = {
            name: runtime.training_options()
            for name, runtime in self.runtimes().items()
        }
        for name, value in options.items():
            self.assertIsInstance(value, dict, f"{name} returned {value!r}")
            self.assertNotIn(
                "learning_rate", value, f"{name} overrides the flagship rate"
            )

    # --- what the wrapper serves -------------------------------------------

    def test_frozen_requests_never_carry_an_adapter(self):
        """A frozen call is the untrained model, whichever runtime serves it.

        Reference reading, candidate sampling and control questions all rely
        on this; a runtime that leaked the adapter into a frozen call would
        sample corrections from the model it is trying to correct.
        """
        for name, runtime in self.runtimes().items():
            runtime.active = "adapter-handle"
            # LM Studio swaps models rather than holding both, so each call is
            # made in the state that runtime would actually be in.
            if name == "lm-studio":
                runtime.loaded, runtime.instance = "base", "base-instance"
            frozen = runtime.payload(dict(messages=MESSAGES), frozen=True)
            if name == "lm-studio":
                runtime.loaded, runtime.instance = (
                    "adapter-handle",
                    "adapter-instance",
                )
            elif name == "vllm":
                runtime.loaded = {"adapter-handle"}
            else:
                runtime.loaded = "adapter-handle"
            adapted = runtime.payload(dict(messages=MESSAGES), frozen=False)
            self.assertNotEqual(
                json.dumps(frozen, sort_keys=True, default=str),
                json.dumps(adapted, sort_keys=True, default=str),
                f"{name} serves the same payload frozen and adapted",
            )

    def test_a_reasoning_model_gets_a_parser_whatever_its_architecture(self):
        """Reasoning support must follow behaviour, not the architecture name.

        Four defects came from gating on "qwen3": the vLLM reasoning parser,
        Ollama's native think control, the thinking-mode decision and the
        startup probe were each keyed to it and silently wrong for a qwen2
        distilled reasoner, which returned its whole thought as the answer.
        """
        with mock.patch("shutil.which", return_value="x"):
            runtime = VLLM(str(self.blob), self.root / "vllm-parser", executable="x")
        runtime.architecture = "qwen2"
        self.assertTrue(
            runtime.prefills_thinking(),
            "a template that opens a thought was not recognised",
        )
        self.assertIn(
            "--reasoning-parser",
            runtime.command(),
            "a qwen2 reasoning model was served without a reasoning parser",
        )

    def test_a_nonreasoning_model_gets_no_parser(self):
        """The converse: a plain model must not be given a reasoning parser."""
        plain = Path(self.root) / "plain"
        plain.mkdir()
        (plain / "config.json").write_text(
            json.dumps({"architectures": ["Qwen2ForCausalLM"]})
        )
        (plain / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ prompt }}"})
        )
        with mock.patch("shutil.which", return_value="x"):
            runtime = VLLM(str(plain), self.root / "vllm-plain", executable="x")
        runtime.architecture = "qwen2"
        self.assertFalse(runtime.prefills_thinking())
        self.assertNotIn("--reasoning-parser", runtime.command())


class ScoringParityTest(unittest.TestCase):
    """The judge and the keep rule are pure and must not vary by runtime."""

    def test_the_judge_is_runtime_independent(self):
        import sys

        sys.path.insert(0, "scripts")
        from wrapper_cycles import ok

        from adaptible.eval import generate_default_dataset

        item = {i.id: i for i in generate_default_dataset()}["geo_001"]
        cases = [
            ("The capital of Australia is Canberra.", True),
            ("<think>\nhmm\n</think>\nCanberra.", True),
            ("<think>\nCanberra maybe", False),
            ("The capital of Australia is Sydney.", False),
        ]
        for text, expected in cases:
            self.assertEqual(ok(item, text), expected, repr(text[:40]))
