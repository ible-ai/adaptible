"""Source-derived evidence spans for the model's extractive answer selection."""

import re


def source_sentences(text):
    """Keep verbatim nonempty spans; the model selects an index, never a copy.

    Sentence boundaries are intentionally simple. A semicolon or abbreviation
    may stay inside a span; exact source provenance matters more than grammar.
    """
    return [
        span.strip()
        for span in re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"“])|[\r\n]+", text)
        if span.strip()
    ]
