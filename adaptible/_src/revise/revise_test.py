"""Unit tests for self-reflective model revision utilities."""

import unittest
from unittest.mock import MagicMock

import mlx.core as mx

from .._classes import InteractionHistory, TrainingExample
from .revise import (
    REWRITE_INSTRUCTIONS,
    REWRITE_INSTRUCTIONS_FEWSHOT,
    THINK_CLOSE,
    InvalidRevisionError,
    _collate_fn,
    _isolate_turn_to_rewritten_turn_index,
    _make_revision_prompt,
    _pad,
    _parse_rewritten_response,
    _serialize_interactions_to_string,
    make_collated_training_example,
    make_revision_prompt,
    revision_prompt_preset,
    strip_think_tags,
    validate_revision_response,
)


class IsolateTurnIndexTest(unittest.TestCase):
    """Tests for _isolate_turn_to_rewritten_turn_index."""

    def test_single_turn_index(self):
        """Parse a response with a single turn index."""
        response = "[[0]] Here is my revised response. [[/0]]"
        self.assertEqual(_isolate_turn_to_rewritten_turn_index(response), 0)

    def test_turn_index_with_surrounding_text(self):
        """Parse a response with text before and after markers."""
        response = "I think turn 2 needs work. [[2]] Better answer here. [[/2]] That's my revision."
        self.assertEqual(_isolate_turn_to_rewritten_turn_index(response), 2)

    def test_multiple_same_indices(self):
        """Handle duplicate indices (start and end markers)."""
        response = "[[5]] The improved response text [[/5]]"
        self.assertEqual(_isolate_turn_to_rewritten_turn_index(response), 5)

    def test_multi_digit_index(self):
        """Handle multi-digit turn indices."""
        response = "[[12]] A much better answer for turn 12. [[/12]]"
        self.assertEqual(_isolate_turn_to_rewritten_turn_index(response), 12)

    def test_multiple_different_indices_returns_minimum(self):
        """When multiple indices appear, return the minimum."""
        response = "[[3]] First revision [[/3]] and [[1]] Second revision [[/1]]"
        self.assertEqual(_isolate_turn_to_rewritten_turn_index(response), 1)

    def test_no_indices_raises_value_error(self):
        """Raise ValueError when no turn indices are found."""
        response = "This response has no turn markers at all."
        with self.assertRaises(ValueError) as ctx:
            _isolate_turn_to_rewritten_turn_index(response)
        self.assertIn("Failed to parse a turn ID", str(ctx.exception))

    def test_empty_string_raises_value_error(self):
        """Raise ValueError for empty input."""
        with self.assertRaises(ValueError):
            _isolate_turn_to_rewritten_turn_index("")

    def test_malformed_markers_no_numbers(self):
        """Markers without numbers should raise ValueError."""
        response = "[[]] Some text [[/]]"
        with self.assertRaises(ValueError):
            _isolate_turn_to_rewritten_turn_index(response)


class ParseRewrittenResponseTest(unittest.TestCase):
    """Tests for _parse_rewritten_response."""

    def test_basic_extraction(self):
        """Extract text between matching markers."""
        response = "[[0]] This is the revised text. [[/0]]"
        result = _parse_rewritten_response(response, 0)
        self.assertEqual(result, "This is the revised text.")

    def test_extraction_with_surrounding_text(self):
        """Extract only the content between markers, ignoring surroundings."""
        response = "Preamble text [[1]] The actual revision [[/1]] and trailing text"
        result = _parse_rewritten_response(response, 1)
        self.assertEqual(result, "The actual revision")

    def test_multi_digit_index(self):
        """Handle multi-digit indices correctly."""
        response = "[[15]] Revision for turn fifteen [[/15]]"
        result = _parse_rewritten_response(response, 15)
        self.assertEqual(result, "Revision for turn fifteen")

    def test_multiline_content(self):
        """Extract multiline content between markers."""
        response = "[[0]] Line one.\nLine two.\nLine three. [[/0]]"
        result = _parse_rewritten_response(response, 0)
        self.assertEqual(result, "Line one.\nLine two.\nLine three.")

    def test_whitespace_stripping(self):
        """Whitespace should be stripped from extracted content."""
        response = "[[0]]   Padded content   [[/0]]"
        result = _parse_rewritten_response(response, 0)
        self.assertEqual(result, "Padded content")

    def test_multiple_markers_uses_last_start(self):
        """When multiple start markers exist, use the last one."""
        response = "[[0]] First attempt [[0]] Second attempt [[/0]]"
        result = _parse_rewritten_response(response, 0)
        self.assertEqual(result, "Second attempt")

    def test_missing_end_marker_returns_to_end(self):
        """Missing end marker returns content to end of string."""
        response = "[[0]] Content without closing marker"
        result = _parse_rewritten_response(response, 0)
        self.assertEqual(result, "Content without closing marker")

    def test_missing_start_marker_returns_empty_or_partial(self):
        """Missing start marker results in None slice behavior."""
        response = "Content with only [[/0]] end marker"
        result = _parse_rewritten_response(response, 0)
        # With no start marker, sor_index is None, so result is response[None:start_of_end]
        self.assertEqual(result, "Content with only")

    def test_wrong_index_returns_none_slice(self):
        """Requesting wrong index returns None-sliced content."""
        response = "[[0]] Content for index 0 [[/0]]"
        result = _parse_rewritten_response(response, 5)
        # Neither marker matches, so both indices are None
        self.assertEqual(result, response.strip())


class MakeRevisionPromptTest(unittest.TestCase):
    """Tests for _make_revision_prompt (internal)."""

    def test_basic_prompt_structure(self):
        """Verify prompt contains dialog markers and content."""
        dialog = "User: Hello\nAssistant: Hi there"
        result = _make_revision_prompt(dialog)
        self.assertIn("<PAST_DIALOG>", result)
        self.assertIn("</PAST_DIALOG>", result)
        self.assertIn(dialog, result)

    def test_custom_instructions(self):
        """Custom instructions should replace default."""
        dialog = "Some dialog"
        custom = "You are a helpful reviewer."
        result = _make_revision_prompt(dialog, instructions=custom)
        self.assertIn(custom, result)
        self.assertIn(dialog, result)

    def test_default_instructions_present(self):
        """Default instructions mention key concepts."""
        dialog = "Dialog content"
        result = _make_revision_prompt(dialog)
        self.assertIn("professional editor", result)
        self.assertIn("[[X]]", result)
        self.assertIn("[[/X]]", result)


class SerializeInteractionsTest(unittest.TestCase):
    """Tests for _serialize_interactions_to_string."""

    def setUp(self):
        """Create mock tokenizer."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda conversation, tokenize, continue_final_message: f"<user>{conversation[0]['content']}</user><assistant>{conversation[1]['content']}</assistant>"
        )

    def test_single_interaction_enumerated(self):
        """Serialize single interaction with enumeration."""
        interactions = [
            InteractionHistory(idx=0, user_input="Hello", llm_response="Hi")
        ]
        result, turns = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
        )
        self.assertIn("[[0]]", result)
        self.assertIn("<user>Hello</user>", result)
        self.assertEqual(len(turns), 1)

    def test_multiple_interactions_enumerated(self):
        """Serialize multiple interactions with sequential indices."""
        interactions = [
            InteractionHistory(idx=0, user_input="First", llm_response="Response 1"),
            InteractionHistory(idx=1, user_input="Second", llm_response="Response 2"),
        ]
        result, turns = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
        )
        self.assertIn("[[0]]", result)
        self.assertIn("[[1]]", result)
        self.assertEqual(len(turns), 2)

    def test_not_enumerated(self):
        """Serialize without enumeration markers."""
        interactions = [
            InteractionHistory(idx=0, user_input="Hello", llm_response="Hi")
        ]
        result, turns = _serialize_interactions_to_string(
            interactions,
            should_enumerate=False,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
        )
        self.assertNotIn("[[0]]", result)
        self.assertIn("<user>Hello</user>", result)

    def test_returns_individual_turns(self):
        """Verify turns list contains individual formatted turns."""
        interactions = [
            InteractionHistory(idx=0, user_input="Q1", llm_response="A1"),
            InteractionHistory(idx=1, user_input="Q2", llm_response="A2"),
        ]
        _, turns = _serialize_interactions_to_string(
            interactions,
            should_enumerate=False,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
        )
        self.assertEqual(len(turns), 2)
        self.assertIn("Q1", turns[0])
        self.assertIn("Q2", turns[1])


class SerializeInteractionsPlainTest(unittest.TestCase):
    """dialog_style="plain" renders User:/Assistant: lines without the tokenizer."""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda conversation, tokenize, continue_final_message: "<|im_start|>user\n"
            f"{conversation[0]['content']}<|im_end|><|im_start|>assistant\n"
            f"{conversation[1]['content']}<|im_end|>"
        )

    def test_plain_has_user_assistant_lines_and_no_template_tokens(self):
        interactions = [
            InteractionHistory(idx=0, user_input="What is 2+2?", llm_response="5"),
            InteractionHistory(idx=1, user_input="Sure?", llm_response="Yes."),
        ]
        result, turns = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
            dialog_style="plain",
        )
        self.assertEqual(
            result,
            "[[0]] User: What is 2+2?\nAssistant: 5\n[[1]] User: Sure?\nAssistant: Yes.",
        )
        self.assertEqual(turns, ["User: What is 2+2?\nAssistant: 5", "User: Sure?\nAssistant: Yes."])
        self.assertNotIn("<|im_start|>", result)
        self.mock_tokenizer.apply_chat_template.assert_not_called()

    def test_plain_not_enumerated(self):
        interactions = [InteractionHistory(idx=0, user_input="Hi", llm_response="Hello")]
        result, _ = _serialize_interactions_to_string(
            interactions,
            should_enumerate=False,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
            dialog_style="plain",
        )
        self.assertEqual(result, "User: Hi\nAssistant: Hello")

    def test_plain_strips_think_tags(self):
        interactions = [
            InteractionHistory(
                idx=0, user_input="Hi", llm_response="<think>hmm</think> Hello"
            )
        ]
        result, _ = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
            dialog_style="plain",
        )
        self.assertEqual(result, "[[0]] User: Hi\nAssistant: Hello")

    def test_chat_is_default_and_uses_template(self):
        interactions = [InteractionHistory(idx=0, user_input="Hi", llm_response="Hello")]
        result, _ = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
        )
        self.assertIn("<|im_start|>user", result)
        self.assertNotIn("User: Hi", result)

    def test_unknown_style_rejected(self):
        with self.assertRaises(ValueError):
            _serialize_interactions_to_string(
                [InteractionHistory(idx=0, user_input="Hi", llm_response="Hello")],
                should_enumerate=True,
                tokenizer=self.mock_tokenizer,
                continue_final_message=False,
                dialog_style="markdown",
            )


class FewShotPromptTest(unittest.TestCase):
    """REWRITE_INSTRUCTIONS_FEWSHOT and revision_prompt_preset."""

    def test_fewshot_instructions_contain_two_marked_examples(self):
        text = REWRITE_INSTRUCTIONS_FEWSHOT
        self.assertIn("Example 1", text)
        self.assertIn("Example 2", text)
        self.assertEqual(text.count("[[0]] User:"), 2)
        self.assertEqual(text.count("Assistant:"), 2)
        # Each example output is a complete [[0]] ... [[/0]] rewrite.
        self.assertEqual(text.count("[[/0]]"), 2)
        self.assertEqual(text.count("Output:\n[[0]] "), 2)
        self.assertIn("Output ONLY the rewrite", text)
        self.assertIn("Do not explain", text)
        self.assertIn("Do not restate", text)
        self.assertLess(len(text), len(REWRITE_INSTRUCTIONS) * 3)

    def test_fewshot_prompt_over_plain_dialog(self):
        instructions, style = revision_prompt_preset("fewshot")
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = AssertionError("must not be called")
        interactions = [
            InteractionHistory(idx=0, user_input="Who wrote Hamlet?", llm_response="Dickens.")
        ]
        prompt = make_revision_prompt(
            interactions, tokenizer, instructions=instructions, dialog_style=style
        )
        self.assertTrue(prompt.startswith(REWRITE_INSTRUCTIONS_FEWSHOT))
        self.assertIn("Example 1", prompt)
        self.assertIn("Example 2", prompt)
        self.assertIn("[[/0]]", prompt)
        self.assertIn(
            "<PAST_DIALOG>\n[[0]] User: Who wrote Hamlet?\nAssistant: Dickens.\n</PAST_DIALOG>",
            prompt,
        )
        tokenizer.apply_chat_template.assert_not_called()

    def test_preset_lookup(self):
        self.assertEqual(revision_prompt_preset("default"), (REWRITE_INSTRUCTIONS, "chat"))
        self.assertEqual(
            revision_prompt_preset("fewshot"), (REWRITE_INSTRUCTIONS_FEWSHOT, "plain")
        )
        with self.assertRaises(ValueError):
            revision_prompt_preset("zero_shot")

    def test_default_prompt_unchanged(self):
        self.assertTrue(REWRITE_INSTRUCTIONS.startswith("You are a professional editor"))
        self.assertNotIn("Example 1", REWRITE_INSTRUCTIONS)


class PadTest(unittest.TestCase):
    """Tests for _pad function."""

    def test_pad_short_array(self):
        """Pad array shorter than target length."""
        arr = mx.array([1, 2, 3])
        result = _pad(arr, max_len=5, padding_token=0)
        expected = mx.array([1, 2, 3, 0, 0])
        self.assertTrue(mx.array_equal(result, expected))

    def test_pad_exact_length(self):
        """Array already at target length needs no padding."""
        arr = mx.array([1, 2, 3])
        result = _pad(arr, max_len=3, padding_token=0)
        expected = mx.array([1, 2, 3])
        self.assertTrue(mx.array_equal(result, expected))

    def test_custom_padding_token(self):
        """Use custom padding token value."""
        arr = mx.array([1, 2])
        result = _pad(arr, max_len=4, padding_token=-1)
        expected = mx.array([1, 2, -1, -1])
        self.assertTrue(mx.array_equal(result, expected))

    def test_empty_array(self):
        """Pad empty array to target length."""
        arr = mx.array([], dtype=mx.int32)
        result = _pad(arr, max_len=3, padding_token=0)
        expected = mx.array([0, 0, 0])
        self.assertTrue(mx.array_equal(result, expected))


class CollateFnTest(unittest.TestCase):
    """Tests for _collate_fn function."""

    def test_single_example(self):
        """Collate single training example."""
        example = TrainingExample(
            input=mx.array([1, 2, 3]),
            label=mx.array([2, 3, 4]),
            mask=mx.array([0, 1, 1]),
        )
        result = _collate_fn([example])
        self.assertEqual(result.input.shape, (1, 3))
        self.assertEqual(result.label.shape, (1, 3))
        self.assertEqual(result.mask.shape, (1, 3))

    def test_multiple_examples_same_length(self):
        """Collate multiple examples of same length."""
        examples = [
            TrainingExample(
                input=mx.array([1, 2]),
                label=mx.array([2, 3]),
                mask=mx.array([1, 1]),
            ),
            TrainingExample(
                input=mx.array([4, 5]),
                label=mx.array([5, 6]),
                mask=mx.array([1, 1]),
            ),
        ]
        result = _collate_fn(examples)
        self.assertEqual(result.input.shape, (2, 2))
        self.assertEqual(result.label.shape, (2, 2))

    def test_multiple_examples_different_lengths(self):
        """Collate examples with padding to max length."""
        examples = [
            TrainingExample(
                input=mx.array([1, 2]),
                label=mx.array([2, 3]),
                mask=mx.array([1, 1]),
            ),
            TrainingExample(
                input=mx.array([4, 5, 6, 7]),
                label=mx.array([5, 6, 7, 8]),
                mask=mx.array([1, 1, 1, 1]),
            ),
        ]
        result = _collate_fn(examples)
        self.assertEqual(result.input.shape, (2, 4))
        # First example should be padded
        self.assertTrue(mx.array_equal(result.input[0], mx.array([1, 2, 0, 0])))

    def test_custom_padding_token(self):
        """Use custom padding token in collation."""
        examples = [
            TrainingExample(
                input=mx.array([1]),
                label=mx.array([2]),
                mask=mx.array([1]),
            ),
            TrainingExample(
                input=mx.array([3, 4, 5]),
                label=mx.array([4, 5, 6]),
                mask=mx.array([1, 1, 1]),
            ),
        ]
        result = _collate_fn(examples, padding_token=-100)
        self.assertTrue(mx.array_equal(result.input[0], mx.array([1, -100, -100])))


class MakeRevisionPromptPublicTest(unittest.TestCase):
    """Tests for public make_revision_prompt function."""

    def setUp(self):
        """Create mock tokenizer."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda conversation, tokenize, continue_final_message: f"[USER]{conversation[0]['content']}[/USER][ASSISTANT]{conversation[1]['content']}[/ASSISTANT]"
        )

    def test_creates_prompt_with_interactions(self):
        """Generate revision prompt from interactions."""
        interactions = [
            InteractionHistory(idx=0, user_input="What is 2+2?", llm_response="5"),
        ]
        result = make_revision_prompt(interactions, self.mock_tokenizer)
        self.assertIn("<PAST_DIALOG>", result)
        self.assertIn("</PAST_DIALOG>", result)
        self.assertIn("[[0]]", result)
        self.assertIn("What is 2+2?", result)

    def test_multiple_interactions(self):
        """Generate prompt with multiple interactions enumerated."""
        interactions = [
            InteractionHistory(idx=0, user_input="Q1", llm_response="A1"),
            InteractionHistory(idx=1, user_input="Q2", llm_response="A2"),
        ]
        result = make_revision_prompt(interactions, self.mock_tokenizer)
        self.assertIn("[[0]]", result)
        self.assertIn("[[1]]", result)

    def test_custom_instructions(self):
        """Use custom instructions in prompt."""
        interactions = [
            InteractionHistory(idx=0, user_input="Hi", llm_response="Hello"),
        ]
        custom = "Be concise in your feedback."
        result = make_revision_prompt(
            interactions, self.mock_tokenizer, instructions=custom
        )
        self.assertIn(custom, result)


class MakeCollatedTrainingExampleTest(unittest.TestCase):
    """Tests for make_collated_training_example function."""

    def setUp(self):
        """Create mock tokenizer with realistic behavior."""
        self.mock_tokenizer = MagicMock()

        # Mock apply_chat_template
        # Mirrors a real chat template: with ``add_generation_prompt`` (or
        # ``continue_final_message``) the rendered string ends at the open
        # assistant tag; otherwise the assistant turn is closed.
        self.mock_tokenizer.apply_chat_template.side_effect = lambda conversation, tokenize=False, continue_final_message=False, add_generation_prompt=False: (
            f"<user>{conversation[0]['content']}</user><assistant>"
            if continue_final_message or add_generation_prompt or len(conversation) < 2
            else f"<user>{conversation[0]['content']}</user><assistant>{conversation[1]['content']}</assistant>"
        )

        # Mock encode to return predictable token IDs
        self.mock_tokenizer.encode.side_effect = lambda text, add_special_tokens: [
            ord(c) for c in text[:10]
        ]

        # Mock special tokens
        self.mock_tokenizer.special_tokens_map = {
            "eos_token": "</s>",
            "bos_token": "<s>",
        }

    def test_creates_training_example(self):
        """Generate training example from revision response."""
        interactions = [
            InteractionHistory(idx=0, user_input="What is 2+2?", llm_response="5"),
        ]
        response = "[[0]] The answer is 4. [[/0]]"

        result = make_collated_training_example(
            response, interactions, self.mock_tokenizer
        )

        self.assertIsInstance(result, TrainingExample)
        self.assertIsInstance(result.input, mx.array)
        self.assertIsInstance(result.label, mx.array)
        self.assertIsInstance(result.mask, mx.array)

    def test_mask_has_correct_structure(self):
        """Mask should be 0 for context, 1 for revision."""
        interactions = [
            InteractionHistory(idx=0, user_input="Hi", llm_response="Hello"),
        ]
        response = "[[0]] Better greeting [[/0]]"

        result = make_collated_training_example(
            response, interactions, self.mock_tokenizer
        )

        # Mask should have both 0s and 1s
        flat_mask = result.mask.flatten()
        mask_values = set(int(flat_mask[i]) for i in range(flat_mask.size))
        self.assertIn(0, mask_values)
        self.assertIn(1, mask_values)

    def test_input_label_offset_by_one(self):
        """Label should be input shifted by one position."""
        interactions = [
            InteractionHistory(idx=0, user_input="Test", llm_response="Response"),
        ]
        response = "[[0]] Better [[/0]]"

        result = make_collated_training_example(
            response, interactions, self.mock_tokenizer
        )

        # input[:-1] and label[1:] alignment is implicit in the slicing
        self.assertEqual(result.input.shape, result.label.shape)

    def test_handles_eos_token_list(self):
        """Handle tokenizers that return eos_token as list."""
        self.mock_tokenizer.special_tokens_map = {
            "eos_token": ["</s>", "<|endoftext|>"],
            "bos_token": ["<s>"],
        }

        interactions = [
            InteractionHistory(idx=0, user_input="Q", llm_response="A"),
        ]
        response = "[[0]] Better A [[/0]]"

        # Should not raise
        result = make_collated_training_example(
            response, interactions, self.mock_tokenizer
        )
        self.assertIsInstance(result, TrainingExample)

    def test_revises_correct_turn_in_multi_turn(self):
        """Correctly identify and revise specific turn in multi-turn dialog."""
        interactions = [
            InteractionHistory(idx=0, user_input="Q1", llm_response="A1"),
            InteractionHistory(idx=1, user_input="Q2", llm_response="A2"),
        ]
        response = "[[1]] Improved A2 [[/1]]"

        result = make_collated_training_example(
            response, interactions, self.mock_tokenizer
        )

        self.assertIsInstance(result, TrainingExample)


class _CharTokenizer:
    """One token per character so label regions can be decoded back to text."""

    special_tokens_map = {"eos_token": "<eos>"}

    def __init__(self, generation_suffix: str):
        self.generation_suffix = generation_suffix

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]

    def decode(self, tokens):
        return "".join(chr(t - 1) for t in tokens if t > 0)

    def apply_chat_template(
        self, conversation, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in conversation)
        if add_generation_prompt:
            text += self.generation_suffix
        return text


class CloseThinkTest(unittest.TestCase):
    """make_collated_training_example closes an open <think> block in the prefix."""

    THINK_PREFIX = "<assistant><think>\n"  # DeepSeek-R1-Distill shape
    PLAIN_PREFIX = "<assistant>"

    def _decode(self, tokenizer, example):
        labels = example.label.tolist()[0]
        mask = example.mask.tolist()[0]
        inputs = example.input.tolist()[0]
        masked = tokenizer.decode(t for t, m in zip(labels, mask) if m)
        # The full sequence is input[0] + label (label is input shifted by one).
        full = tokenizer.decode([inputs[0]] + labels)
        return masked, full

    def _example(self, suffix, **kwargs):
        tokenizer = _CharTokenizer(suffix)
        interactions = [InteractionHistory(idx=0, user_input="Q?", llm_response="wrong")]
        example = make_collated_training_example(
            "[[0]] Right answer. [[/0]]", interactions, tokenizer, **kwargs
        )
        return tokenizer, example

    def test_open_think_is_closed_and_masked(self):
        tokenizer, example = self._example(self.THINK_PREFIX)
        masked, full = self._decode(tokenizer, example)
        self.assertEqual(masked, f"{THINK_CLOSE}Right answer.<eos>")
        self.assertTrue(masked.startswith("</think>"))
        self.assertEqual(
            full, f"<user>Q?</user>{self.THINK_PREFIX}</think>\n\nRight answer.<eos>"
        )
        # The unmasked region is exactly the chat-template prefix.
        prefix_len = len(f"<user>Q?</user>{self.THINK_PREFIX}")
        mask = example.mask.tolist()[0]
        self.assertEqual(mask[: prefix_len - 1], [0] * (prefix_len - 1))
        self.assertEqual(mask[prefix_len - 1], 1)

    def test_no_think_tag_inserts_nothing(self):
        tokenizer, example = self._example(self.PLAIN_PREFIX)
        masked, full = self._decode(tokenizer, example)
        self.assertEqual(masked, "Right answer.<eos>")
        self.assertNotIn("</think>", full)
        self.assertEqual(full, "<user>Q?</user><assistant>Right answer.<eos>")

    def test_close_think_false_reproduces_old_sequence(self):
        tokenizer, example = self._example(self.THINK_PREFIX, close_think=False)
        masked, full = self._decode(tokenizer, example)
        self.assertEqual(masked, "Right answer.<eos>")
        self.assertEqual(
            full, f"<user>Q?</user>{self.THINK_PREFIX}Right answer.<eos>"
        )

    def test_think_without_trailing_newline_is_still_closed(self):
        tokenizer, example = self._example("<assistant><think>")
        masked, _ = self._decode(tokenizer, example)
        self.assertEqual(masked, f"{THINK_CLOSE}Right answer.<eos>")

    def test_closed_think_in_prefix_is_left_alone(self):
        tokenizer, example = self._example("<assistant><think>\n</think>\n")
        masked, _ = self._decode(tokenizer, example)
        self.assertEqual(masked, "Right answer.<eos>")


class MultiTurnPrefixTest(unittest.TestCase):
    """make_collated_training_example's prefix carries the turns before the revised one."""

    GEN = "<assistant><think>\n"

    INTERACTIONS = [
        InteractionHistory(
            idx=0, user_input="Q0", llm_response="<think>\nreasoning0\n</think>\n\nA0"
        ),
        InteractionHistory(idx=1, user_input="Q1", llm_response="reasoning1</think>A1"),
        InteractionHistory(idx=2, user_input="Q2", llm_response="wrong"),
    ]

    def _run(self, response):
        tokenizer = _CharTokenizer(self.GEN)
        example = make_collated_training_example(response, self.INTERACTIONS, tokenizer)
        inputs = example.input.tolist()[0]
        labels = example.label.tolist()[0]
        mask = example.mask.tolist()[0]
        full = tokenizer.decode([inputs[0]] + labels)
        masked = tokenizer.decode(t for t, m in zip(labels, mask) if m)
        return full, masked, mask

    def test_revising_last_turn_includes_prior_turns(self):
        full, masked, mask = self._run("[[2]] Right answer. [[/2]]")
        prefix = (
            "<user>Q0</user><assistant>A0</assistant>"
            "<user>Q1</user><assistant>A1</assistant>"
            f"<user>Q2</user>{self.GEN}"
        )
        target = f"{THINK_CLOSE}Right answer.<eos>"
        self.assertEqual(full, prefix + target)
        self.assertEqual(masked, target)
        self.assertNotIn("reasoning", full)
        # Mask is 0 over the whole prefix and 1 over the whole target
        # (mask[1:] alignment: position i predicts sequence[i + 1]).
        n = len(prefix)
        self.assertEqual(mask[: n - 1], [0] * (n - 1))
        self.assertEqual(mask[n - 1 :], [1] * len(target))

    def test_revising_first_turn_includes_no_prior_turns(self):
        full, masked, _ = self._run("[[0]] Right answer. [[/0]]")
        self.assertEqual(
            full, f"<user>Q0</user>{self.GEN}{THINK_CLOSE}Right answer.<eos>"
        )
        self.assertNotIn("Q1", full)
        self.assertNotIn("Q2", full)
        self.assertEqual(masked, f"{THINK_CLOSE}Right answer.<eos>")

    def test_single_turn_dialog_unchanged(self):
        tokenizer = _CharTokenizer(self.GEN)
        example = make_collated_training_example(
            "[[0]] Right answer. [[/0]]", self.INTERACTIONS[:1], tokenizer
        )
        seq = tokenizer.encode(
            f"<user>Q0</user>{self.GEN}{THINK_CLOSE}Right answer.<eos>"
        )
        self.assertEqual(example.input.tolist()[0], seq[:-1])
        self.assertEqual(example.label.tolist()[0], seq[1:])


class EdgeCaseTest(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    def test_isolate_index_with_nested_brackets(self):
        """Handle nested or malformed bracket patterns."""
        # This tests robustness of regex parsing
        response = "[[0]] Text with [brackets] inside [[/0]]"
        idx = _isolate_turn_to_rewritten_turn_index(response)
        self.assertEqual(idx, 0)

    def test_parse_response_with_special_characters(self):
        """Parse responses containing special regex characters."""
        response = "[[0]] Response with $pecial ch@rs & symbols! [[/0]]"
        result = _parse_rewritten_response(response, 0)
        self.assertEqual(result, "Response with $pecial ch@rs & symbols!")

    def test_empty_interactions_list(self):
        """Handle empty interactions gracefully."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = ""

        result, turns = _serialize_interactions_to_string(
            [],
            should_enumerate=True,
            tokenizer=mock_tokenizer,
            continue_final_message=False,
        )
        self.assertEqual(result, "")
        self.assertEqual(turns, [])


class StripThinkTagsTest(unittest.TestCase):
    """Tests for strip_think_tags function."""

    def test_strip_simple_think_tag(self):
        """Remove a simple think tag."""
        text = "<think>internal thought</think> Actual response"
        result = strip_think_tags(text)
        self.assertEqual(result, "Actual response")

    def test_strip_multiline_think_tag(self):
        """Remove multiline think tag content."""
        text = "<think>Line 1\nLine 2\nLine 3</think> Response"
        result = strip_think_tags(text)
        self.assertEqual(result, "Response")

    def test_no_think_tags(self):
        """Text without think tags is unchanged."""
        text = "Just a normal response"
        result = strip_think_tags(text)
        self.assertEqual(result, "Just a normal response")

    def test_multiple_think_tags(self):
        """Remove multiple think tags."""
        text = "<think>first</think> Middle <think>second</think> End"
        result = strip_think_tags(text)
        self.assertEqual(result, "Middle End")

    def test_empty_think_tag(self):
        """Handle empty think tags."""
        text = "<think></think> Response"
        result = strip_think_tags(text)
        self.assertEqual(result, "Response")

    def test_think_tag_only(self):
        """Handle text that is only a think tag."""
        text = "<think>only thinking</think>"
        result = strip_think_tags(text)
        self.assertEqual(result, "")


class ValidateRevisionResponseTest(unittest.TestCase):
    """Tests for validate_revision_response function."""

    def test_valid_response_passes(self):
        """A properly formatted response should pass validation."""
        response = "I'll revise turn 0.\n\n[[0]] This is a much better response that is helpful. [[/0]]"
        # Should not raise
        validate_revision_response(response, num_interactions=2)

    def test_missing_markers_raises(self):
        """Response without markers should raise InvalidRevisionError."""
        response = "I think the response could be improved."
        with self.assertRaises(InvalidRevisionError) as ctx:
            validate_revision_response(response, num_interactions=2)
        self.assertIn("markers", str(ctx.exception).lower())

    def test_index_out_of_bounds_raises(self):
        """Turn index beyond interaction count should raise."""
        response = "[[5]] Revised response [[/5]]"
        with self.assertRaises(InvalidRevisionError) as ctx:
            validate_revision_response(response, num_interactions=2)
        self.assertIn("out of bounds", str(ctx.exception).lower())

    def test_missing_closing_marker_raises(self):
        """Response without closing marker should raise."""
        response = "[[0]] This response has no closing marker"
        with self.assertRaises(InvalidRevisionError) as ctx:
            validate_revision_response(response, num_interactions=2)
        self.assertIn("closing marker", str(ctx.exception).lower())

    def test_content_too_short_raises(self):
        """Very short content should raise."""
        response = "[[0]] Hi [[/0]]"
        with self.assertRaises(InvalidRevisionError) as ctx:
            validate_revision_response(
                response, num_interactions=2, min_content_length=10
            )
        self.assertIn("too short", str(ctx.exception).lower())

    def test_garbage_pattern_raises(self):
        """Repetitive garbage content should raise."""
        response = "[[0]] [[1]][[2]][[3]][[4]][[5]] [[/0]]"
        with self.assertRaises(InvalidRevisionError) as ctx:
            validate_revision_response(response, num_interactions=2)
        self.assertIn("garbage", str(ctx.exception).lower())

    def test_real_garbage_output_raises(self):
        """The actual garbage output from the model should be caught."""
        garbage = "[[1]]</PAST_DIALOG>[[2]]</PAST_DIALOG>[[3]]</PAST_DIALOG>[[4]]</PAST_DIALOG>"
        with self.assertRaises(InvalidRevisionError):
            validate_revision_response(garbage, num_interactions=2)


class SerializeInteractionsWithThinkTagsTest(unittest.TestCase):
    """Tests for think tag stripping in serialization."""

    def setUp(self):
        """Create mock tokenizer."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template.side_effect = (
            lambda conversation, tokenize, continue_final_message: f"<user>{conversation[0]['content']}</user><assistant>{conversation[1]['content']}</assistant>"
        )

    def test_strips_think_tags_by_default(self):
        """Think tags should be stripped by default."""
        interactions = [
            InteractionHistory(
                idx=0,
                user_input="Hello",
                llm_response="<think>internal</think> Visible response",
            ),
        ]
        result, _ = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
        )
        self.assertNotIn("<think>", result)
        self.assertIn("Visible response", result)

    def test_preserves_think_tags_when_disabled(self):
        """Think tags should be preserved when strip_thinking=False."""
        interactions = [
            InteractionHistory(
                idx=0,
                user_input="Hello",
                llm_response="<think>internal</think> Visible response",
            ),
        ]
        result, _ = _serialize_interactions_to_string(
            interactions,
            should_enumerate=True,
            tokenizer=self.mock_tokenizer,
            continue_final_message=False,
            strip_thinking=False,
        )
        self.assertIn("<think>", result)


if __name__ == "__main__":
    unittest.main()
