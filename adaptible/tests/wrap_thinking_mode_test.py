"""Endpoint-specific repair modes must match actual native request controls."""

import copy
import unittest

from adaptible._src.wrap.thinking import generation_mode


class ThinkingModeTest(unittest.TestCase):
    def mode(self, body, **options):
        before = copy.deepcopy(body)
        result = generation_mode(body, "qwen3", **options)
        self.assertEqual(body, before, "Mode inspection must preserve serving payload")
        return result

    def test_ollama_openai_ignored_controls_cannot_select_training_mode(self):
        for body in (
            {"think": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ):
            with self.subTest(body=body):
                self.assertIn("error", self.mode(body, native=True))

    def test_ollama_openai_supported_flat_and_nested_controls(self):
        for key, value in (
            ("reasoning_effort", "none"),
            ("reasoning", {"effort": "none"}),
        ):
            with self.subTest(key=key):
                self.assertFalse(self.mode({key: value}, native=True)["thinking"])
        self.assertTrue(
            self.mode({"reasoning": {"effort": "medium"}}, native=True)["thinking"]
        )

    def test_native_chat_and_generate_require_native_think(self):
        for path in ("/api/chat", "/api/generate"):
            with self.subTest(path=path):
                self.assertFalse(
                    self.mode({"think": False}, native=True, path=path)["thinking"]
                )
                self.assertTrue(
                    self.mode({"think": True}, native=True, path=path)["thinking"]
                )
                self.assertIn(
                    "error",
                    self.mode({"reasoning_effort": "none"}, native=True, path=path),
                )
                self.assertIn(
                    "error", self.mode({"think": True}, native=False, path=path)
                )

    def test_native_control_not_effective_on_other_openai_services(self):
        self.assertIn("error", self.mode({"think": False}))
        self.assertIn("error", self.mode({"reasoning": {"effort": "none"}}))

    def test_vllm_and_llama_template_and_effort_controls(self):
        for provider in ("VLLM", "LlamaCpp"):
            for body in (
                {"reasoning_effort": "none"},
                {"chat_template_kwargs": {"enable_thinking": False}},
            ):
                with self.subTest(provider=provider, body=body):
                    self.assertFalse(self.mode(body, provider=provider)["thinking"])

    def test_lmstudio_uses_real_effort_control(self):
        self.assertIn(
            "error",
            self.mode(
                {"chat_template_kwargs": {"enable_thinking": False}},
                provider="LMStudio",
            ),
        )
        for effort, enabled in (("none", False), ("medium", True)):
            result = self.mode(
                {
                    "reasoning_effort": effort,
                    "chat_template_kwargs": {"enable_thinking": enabled},
                },
                provider="LMStudio",
            )
            self.assertEqual(result["thinking"], enabled)

    def test_conflicting_controls_decline_repair_without_guessing_precedence(self):
        self.assertIn(
            "error",
            self.mode(
                {
                    "reasoning_effort": "none",
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            ),
        )
        self.assertIn(
            "error",
            self.mode(
                {"reasoning_effort": "none", "reasoning": {"effort": "high"}},
                native=True,
            ),
        )

    def test_custom_formatting_cannot_silently_train_standard_template(self):
        for body in (
            {"raw": True},
            {"context": [1]},
            {"template": "custom"},
            {"chat_template": "custom"},
            {"continue_final_message": True},
            {"add_generation_prompt": False},
            {"documents": [{"text": "x"}]},
            {"tools": [{"type": "function"}]},
            {"tool_choice": "required"},
            {"chat_template_kwargs": {"custom_prefix": "changed"}},
        ):
            with self.subTest(body=body):
                self.assertIn("error", self.mode(body))

    def test_malformed_controls_decline_repair(self):
        for body in (
            {"chat_template_kwargs": []},
            {"chat_template_kwargs": {"enable_thinking": 0}},
            {"reasoning_effort": []},
            {"reasoning": None},
            {"reasoning": {"effort": "medium", "other": True}},
        ):
            with self.subTest(body=body):
                self.assertIn("error", self.mode(body, native=True))
        self.assertIn("error", self.mode({"think": 0}, native=True, path="/api/chat"))

    def test_default_and_other_architecture(self):
        self.assertTrue(self.mode({})["thinking"])
        self.assertEqual(generation_mode({"think": False}, "qwen2"), {})


if __name__ == "__main__":
    unittest.main()
