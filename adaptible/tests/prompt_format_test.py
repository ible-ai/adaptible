"""Native Go-rendered fixtures guard Ollama/HF training prompt alignment."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from adaptible._src.wrap.prompt_format import (
    OLLAMA_QWEN3_FORMAT,
    OLLAMA_QWEN3_TEMPLATE_SHA256,
    render_ollama_qwen3,
)
from adaptible._src.wrap.runtime import Ollama


class PromptFormatTest(unittest.TestCase):
    def test_plain_text_prefixes_match_actual_go_template_output(self):
        cases = json.loads(
            (Path(__file__).parent / "fixtures/ollama_qwen3_prompts.json").read_text()
        )["cases"]
        for case in cases:
            with self.subTest(case=case["name"]):
                before = json.dumps(case["messages"])
                self.assertEqual(
                    render_ollama_qwen3(
                        case["messages"], system=case["system"], thinking=case["think"]
                    ),
                    case["prompt"],
                )
                self.assertEqual(json.dumps(case["messages"]), before)

    def test_worker_uses_native_prefix_for_primary_and_additional_examples(self):
        from adaptible._src.wrap.training_examples import encode_examples
        from transformers import Qwen2Tokenizer
        from tokenizers.pre_tokenizers import ByteLevel

        tokenizer = Qwen2Tokenizer(
            vocab={
                v: i
                for i, v in enumerate(["<|endoftext|>", *sorted(ByteLevel.alphabet())])
            },
            merges=[],
        )
        messages = [dict(role="user", content="First?")]
        other = [dict(role="user", content="Second?")]
        options = dict(
            prompt_format=OLLAMA_QWEN3_FORMAT,
            template_sha256=OLLAMA_QWEN3_TEMPLATE_SHA256,
            system="Be precise.",
        )
        job = dict(
            messages=messages,
            target="Answer",
            examples=[dict(messages=other, target="Answer")],
            training_options=options,
        )
        with patch.object(
            tokenizer,
            "apply_chat_template",
            side_effect=AssertionError("Must not use mismatched HF Jinja"),
        ):
            inputs = encode_examples(tokenizer, job, "qwen3")
        for index, conversation in enumerate((messages, other)):
            prefix = tokenizer.encode(
                render_ollama_qwen3(conversation, system=options["system"]),
                add_special_tokens=False,
            )
            self.assertEqual(inputs["input_ids"][index, : len(prefix)].tolist(), prefix)
            self.assertEqual(
                inputs["labels"][index, : len(prefix)].tolist(), [-100] * len(prefix)
            )
            self.assertNotEqual(inputs["labels"][index, len(prefix)].item(), -100)

    def test_unsupported_history_fails_instead_of_guessing_a_native_prompt(self):
        for messages in (
            [dict(role="tool", content="result"), dict(role="user", content="Next")],
            [dict(role="user", content=[dict(type="text", text="Hi")])],
        ):
            with (
                self.subTest(messages=messages),
                self.assertRaisesRegex(ValueError, "plain text"),
            ):
                render_ollama_qwen3(messages)


class OllamaPromptDiscoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_template_system_fingerprint_and_unsupported_template_guard(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("shutil.which", return_value="ollama"),
        ):
            root = Path(directory)
            base = root / "model.gguf"
            base.write_bytes(b"fixture")
            runtime = Ollama("fixture", root / "state")
            await runtime.client.aclose()
            info = dict(
                modelfile=f'FROM "{base}"',
                template="custom template",
                system="Be precise.",
            )
            # The wrapper pins its sampler on a tag it creates for itself, so
            # discovery is create-then-show even with no context size asked
            # for. A transport that answers every path with the same document
            # reports no `status`, and the wrapper correctly refuses it.
            pinned = {}

            def respond(request):
                body = json.loads(request.content) if request.content else {}
                if request.url.path == "/api/create":
                    pinned.update(body["parameters"])
                    return httpx.Response(200, json={"status": "success"})
                if request.url.path == "/api/delete":
                    return httpx.Response(200, json={})
                if request.url.path == "/api/ps":
                    return httpx.Response(200, json={"models": []})
                return httpx.Response(
                    200,
                    json={
                        **info,
                        "parameters": "\n".join(
                            f"{key} {value}" for key, value in pinned.items()
                        ),
                    },
                )

            runtime.client = httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            )
            try:
                await runtime.discover()
                initial = runtime.serving_template_digest
                runtime.architecture = "qwen3"
                with self.assertRaisesRegex(ValueError, "template is not supported"):
                    runtime.training_options()
                known_digest = hashlib.sha256(info["template"].encode()).hexdigest()
                with patch(
                    "adaptible._src.wrap.runtime.OLLAMA_QWEN3_TEMPLATE_SHA256",
                    known_digest,
                ):
                    options = runtime.training_options()
                self.assertEqual(options["system"], "Be precise.")
                self.assertEqual(options["prompt_format"], OLLAMA_QWEN3_FORMAT)
                self.assertEqual(options["template_sha256"], known_digest)
                runtime.modelfile += '\nMESSAGE user "Hidden preface"'
                with self.assertRaisesRegex(ValueError, "MESSAGE history"):
                    runtime.training_options()
                info["system"] = "Different system"
                await runtime.discover()
                self.assertNotEqual(initial, runtime.serving_template_digest)
                initial = runtime.serving_template_digest
                info["template"] = "other template"
                await runtime.discover()
                self.assertNotEqual(initial, runtime.serving_template_digest)
                initial = runtime.serving_template_digest
                info["modelfile"] += '\nMESSAGE user "Hidden preface"'
                await runtime.discover()
                self.assertNotEqual(initial, runtime.serving_template_digest)
                self.assertEqual(
                    runtime.payload({"reasoning_effort": "high"})["reasoning_effort"],
                    "high",
                )
            finally:
                await runtime.close()
