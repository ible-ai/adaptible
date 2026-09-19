"""A positive classifier cannot make missing rank evidence relevant."""

import unittest

from adaptible.wrap.source_relevance import relevant_source_sentences


class SourceRelevanceTest(unittest.TestCase):
    def test_capital_and_big_do_not_establish_largest(self):
        sentences = [
            "Ash is the capital of Eloria.",
            "Big city: Birch.",
            "Cedar is the largest city in Eloria.",
        ]
        self.assertEqual(
            relevant_source_sentences("What is the largest city in Eloria?", sentences),
            [sentences[2]],
        )

    def test_mixed_source_keeps_relevant_span_without_rewriting_or_reordering(self):
        sentences = [
            "  CEDAR is certainly the largest city in the country.  ",
            "Ash is its capital,",
            "Fir is the biggest city in a different country.",
        ]
        original = list(sentences)
        self.assertEqual(
            relevant_source_sentences("Which city is largest?", sentences),
            [sentences[0], sentences[2]],
        )
        self.assertEqual(sentences, original)
        # Scope and answer identity still need separate grounding checks.

    def test_opposite_or_different_measure_does_not_satisfy_requested_rank(self):
        for question, wrong, right in (
            (
                "Which tower is tallest?",
                "Oak is the largest tower.",
                "Elm is the tallest tower.",
            ),
            (
                "Which route is longest?",
                "Oak is the shortest route.",
                "Elm is the longest route.",
            ),
            (
                "Which vessel is oldest?",
                "Oak is the newest vessel.",
                "Elm is the oldest vessel.",
            ),
            (
                "Which lake is deepest?",
                "Oak is the widest lake.",
                "Elm is the deepest lake.",
            ),
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    relevant_source_sentences(question, [wrong, right]), [right]
                )

    def test_negated_and_hedged_rankings_are_not_positive_evidence(self):
        wrong = [
            "Oak is not the largest island.",
            "Oak isn't the largest island.",
            "Oak is no longer the largest island.",
            "Neither Oak nor Elm is largest.",
            "It is not true that Oak is the largest island.",
            "It is false that Oak is the largest island.",
            "Oak is one of the largest islands.",
            "Oak is among the largest islands.",
            "Oak is one of the country's largest islands.",
            "Oak is among the ten largest islands.",
        ]
        self.assertEqual(
            relevant_source_sentences("Which island is largest?", wrong), []
        )

    def test_ordinal_modifier_does_not_promote_runner_up_to_largest(self):
        first = "Oak is the largest island."
        second = "Elm is the second-largest island."
        numbered = "Fir is the 2nd largest island."
        lower = "Larch is the eleventh largest island."
        self.assertEqual(
            relevant_source_sentences(
                "Which island is largest?", [first, second, numbered, lower]
            ),
            [first],
        )
        self.assertEqual(
            relevant_source_sentences(
                "Which is the second largest island?", [first, second, numbered]
            ),
            [second, numbered],
        )

    def test_standalone_ordinals_match_words_or_numbers(self):
        first = "Ada was the first chair of the society."
        numbered = "Ada was the 1st chair of the society."
        second = "Bea was the second chair of the society."
        self.assertEqual(
            relevant_source_sentences(
                "Who was the first chair?", [first, numbered, second]
            ),
            [first, numbered],
        )

    def test_most_and_least_qualified_relations_are_distinct(self):
        source = [
            "Oak is the most populous district.",
            "Elm is the least populous district.",
            "Fir is the most visited district.",
        ]
        self.assertEqual(
            relevant_source_sentences("Which district is most populous?", source),
            source[:1],
        )

    def test_literal_word_boundaries_and_unicode(self):
        source = [
            "The artist's surname is Tallestman.",
            "The ＴＡＬＬＥＳＴ tower is Elm.",
        ]
        self.assertEqual(
            relevant_source_sentences("Which tower is tallest?", source), source[1:]
        )

    def test_negative_rank_question_conservatively_abstains(self):
        self.assertEqual(
            relevant_source_sentences(
                "Which island is not the largest?", ["Oak is the largest island."]
            ),
            [],
        )

    def test_unknown_relation_is_unchanged_even_with_negation(self):
        sentences = ["Oak is not the capital.", "Elm is the capital."]
        self.assertEqual(
            relevant_source_sentences("What is the capital?", sentences), sentences
        )


if __name__ == "__main__":
    unittest.main()
