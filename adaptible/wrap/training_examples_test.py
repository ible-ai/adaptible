"""Variable-length correction targets keep independent shifted loss masks."""

import unittest

from adaptible.wrap.training_examples import encode_examples


class TokenFixture:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {
            "short": [10, 11],
            "long": [12, 13, 14, 15],
            "opened": [16, 17],
            "empty": [],
            "overflow": [10] * 4096,
        }[messages[0]["content"]]

    def decode(self, ids):
        return "prompt <think>" if ids == [16, 17] else "prompt"

    def encode(self, text, **kwargs):
        if text.startswith("untokenizable"):
            return []
        self.assert_eos = text.endswith(self.eos_token)
        return [ord(character) for character in text.removesuffix(self.eos_token)] + [2]


def example(prompt="short", target="A"):
    return dict(messages=[dict(role="user", content=prompt)], target=target)


class TrainingExamplesTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenFixture()

    def test_unequal_prompts_mask_each_prefix_and_right_padding_after_shift(self):
        job = {**example(), "examples": [example("long", "BC")]}
        batch = encode_examples(self.tokenizer, job, "qwen2")
        self.assertEqual(
            batch["input_ids"].tolist(),
            [[10, 11, 65, 2, 0, 0, 0], [12, 13, 14, 15, 66, 67, 2]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        )
        self.assertEqual(
            batch["labels"].tolist(),
            [
                [-100, -100, 65, 2, -100, -100, -100],
                [-100, -100, -100, -100, 66, 67, 2],
            ],
        )
        # HF causal models shift labels left internally. The first target is
        # therefore predicted by each row's own final prompt token.
        transitions = []
        for tokens, labels in zip(batch["input_ids"], batch["labels"]):
            transitions.append(
                [
                    (int(token), int(label))
                    for token, label in zip(tokens[:-1], labels[1:])
                    if label != -100
                ]
            )
        self.assertEqual(
            transitions, [[(11, 65), (65, 2)], [(15, 66), (66, 67), (67, 2)]]
        )

    def test_single_example_matches_original_unpadded_encoding(self):
        batch = encode_examples(self.tokenizer, example(), "qwen2")
        self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 65, 2]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 65, 2]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1]])
        self.assertNotIn("enable_thinking", self.tokenizer.calls[0])

    def test_reasoning_prefix_does_not_duplicate_opening_and_uses_restored_eos(self):
        batch = encode_examples(
            self.tokenizer, example("opened", "<think>\n\nA"), "qwen3"
        )
        self.assertEqual(batch["input_ids"].tolist(), [[16, 17, 65, 2]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 65, 2]])
        self.assertIs(self.tokenizer.calls[0]["enable_thinking"], False)
        self.assertTrue(self.tokenizer.assert_eos)

    def test_empty_or_overflowing_example_rejects_entire_batch(self):
        invalid = (
            example("empty"),
            example("overflow"),
            example(target=""),
            example(target="   "),
            example(target="untokenizable"),
            dict(messages=[], target="A"),
            dict(messages=example()["messages"], target=None),
        )
        for bad in invalid:
            for position in ("original", "additional"):
                with self.subTest(bad=bad, position=position):
                    job = (
                        bad
                        if position == "original"
                        else {**example(), "examples": [bad]}
                    )
                    with self.assertRaises(ValueError):
                        encode_examples(self.tokenizer, job, "qwen3")

    def test_additional_examples_are_bounded_and_cannot_replace_original(self):
        for extras in (None, {}, [example()] * 3, [None]):
            with self.subTest(extras=extras):
                with self.assertRaises(ValueError):
                    encode_examples(
                        self.tokenizer, {**example(), "examples": extras}, "qwen2"
                    )
        with self.assertRaises(ValueError):
            encode_examples(self.tokenizer, {"examples": [example()]}, "qwen2")
        batch = encode_examples(
            self.tokenizer, {**example(), "examples": [example(), example()]}, "qwen2"
        )
        self.assertEqual(batch["input_ids"].shape[0], 3)

    def test_absent_pad_token_uses_eos_but_never_supervises_padding(self):
        self.tokenizer.pad_token_id = None
        batch = encode_examples(
            self.tokenizer, {**example(), "examples": [example("long")]}, "qwen2"
        )
        self.assertEqual(batch["input_ids"][0].tolist()[-2:], [2, 2])
        self.assertEqual(batch["labels"][0].tolist()[-2:], [-100, -100])
        self.assertEqual(batch["attention_mask"][0].tolist()[-2:], [0, 0])


if __name__ == "__main__":
    unittest.main()
