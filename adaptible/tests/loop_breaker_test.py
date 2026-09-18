"""The wrapper stops a generation where the original's loop breakers would."""

import asyncio
import importlib.util
import random
import unittest

from adaptible._src.wrap import loop_breaker
from adaptible.tests import tiny_models as tiny

HAS_MLX = importlib.util.find_spec("mlx") is not None


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


class TokenLoopTest(unittest.TestCase):
    @unittest.skipUnless(HAS_MLX, "compares with the original, which imports MLX")
    def test_agrees_with_the_original(self):
        from adaptible._src._llm import (
            _LOOP_DETECTION_MAX_REPETITIONS,
            _LOOP_DETECTION_SEQUENCE_LENGTH,
            _detect_token_loop,
        )

        self.assertEqual(loop_breaker.SEQUENCE_LENGTH, _LOOP_DETECTION_SEQUENCE_LENGTH)
        self.assertEqual(loop_breaker.MAX_REPETITIONS, _LOOP_DETECTION_MAX_REPETITIONS)
        rng = random.Random(0)
        for _ in range(2000):
            unit = [rng.randrange(3) for _ in range(rng.randrange(1, 10))]
            tokens = [rng.randrange(3) for _ in range(rng.randrange(8))]
            tokens += unit * rng.randrange(4)
            self.assertEqual(
                loop_breaker.token_loop(tokens), _detect_token_loop(tokens, 8, 3)
            )


class TruncateTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny.tiny_tokenizer()

    def run_on(self, text, **options):
        tokens = encode(self.tokenizer, text)
        options.setdefault("stopped", True)
        options.setdefault("max_tokens", 4096)
        return loop_breaker.truncate(tokens, self.tokenizer, **options)

    def test_a_line_seen_twice_stops_on_its_third_occurrence(self):
        text, cut = self.run_on("Paris.\nRome.\nParis.\nRome.\nParis.\nOslo.\n")
        self.assertTrue(cut)
        self.assertEqual(text, "Paris.\nRome.\nParis.\nRome.\nParis.\n")

    def test_sampling_applies_only_the_token_rule(self):
        text, cut = self.run_on("Paris.\nRome.\nParis.\nRome.\nParis.\n", lines=False)
        self.assertFalse(cut)
        self.assertEqual(text, "Paris.\nRome.\nParis.\nRome.\nParis.\n")

    def test_ordinary_text_is_returned_whole(self):
        text = "The capital is Canberra.\nIt was chosen in 1908.\n"
        self.assertEqual(self.run_on(text), (text, False))

    def test_the_cap_ends_the_text_without_a_cut(self):
        tokens = encode(self.tokenizer, "one two three four five six")
        text, cut = loop_breaker.truncate(
            tokens, self.tokenizer, stopped=False, max_tokens=3
        )
        self.assertFalse(cut)
        self.assertEqual(text, self.tokenizer.decode(tokens[:3]))

    @unittest.skipUnless(HAS_MLX, "compares with mlx_lm's detokenizer")
    def test_pieces_match_mlx_lm(self):
        from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

        theirs = BPEStreamingDetokenizer(self.tokenizer)
        ours = loop_breaker._Detokenizer(self.tokenizer)
        for token in encode(self.tokenizer, "Hi , it's Casablanca! é ü\n\n  x  y"):
            theirs.add_token(token)
            ours.add(token)
            self.assertEqual(ours.segment(), theirs.last_segment)


class LoopingRuntime:
    """Returns one fixed generation, framed as a reasoning parser frames it."""

    max_tokens = 4096

    def __init__(self, reasoning, content):
        self.reasoning, self.content = reasoning, content

    async def complete(self, messages, *, details, **options):
        details.update(
            reasoning=self.reasoning, content=self.content, finish_reason="stop"
        )
        return self.content


class CompleteAsOriginalTest(unittest.TestCase):
    def complete(self, reasoning, content):
        details = {}
        response = asyncio.run(
            loop_breaker.complete_as_original(
                LoopingRuntime(reasoning, content),
                tiny.tiny_tokenizer(),
                [dict(role="user", content="q")],
                lines=True,
                details=details,
            )
        )
        return response, details

    def test_a_looping_thought_stays_unclosed(self):
        loop = "\nHmm.\nHmm.\nHmm.\nHmm.\n"
        response, details = self.complete(loop, "\n\nParis.")
        self.assertEqual(response, "")
        self.assertFalse(details["complete"])
        self.assertTrue(details["loop_cut"])
        self.assertEqual(details["reasoning"], "Hmm.\nHmm.\nHmm.\n")

    def test_a_finished_generation_is_untouched(self):
        response, details = self.complete("\nIt is Paris.\n", "\n\nParis.")
        self.assertEqual(response, "\n\nParis.")
        self.assertNotIn("loop_cut", details)


if __name__ == "__main__":
    unittest.main()
