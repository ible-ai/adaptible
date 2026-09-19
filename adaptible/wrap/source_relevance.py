"""Necessary lexical rank constraints for extractive source selection.

This is deliberately not an entailment judge. Matching a rank word does not
establish the answer's identity, scope, date, or truth. Unknown relations pass
through; recognized ranks can conservatively discard useful paraphrases.
"""

import re
import unicodedata

_SUPERLATIVES = {
    "largest",
    "biggest",
    "smallest",
    "tallest",
    "highest",
    "lowest",
    "longest",
    "shortest",
    "oldest",
    "youngest",
    "earliest",
    "latest",
    "fastest",
    "slowest",
    "richest",
    "poorest",
    "deepest",
    "shallowest",
    "widest",
    "narrowest",
    "heaviest",
    "lightest",
    "nearest",
    "farthest",
    "furthest",
    "newest",
    "cheapest",
    "busiest",
}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
    "hundredth": 100,
    "thousandth": 1000,
}
_NEGATION = re.compile(
    r"\b(?:not|never|neither|nor|false|untrue|incorrect)\b|\bno\s+longer\b|\b\w+n't\b"
)
_QUALIFIED_RANK = re.compile(r"\b(?:(?:one|some|any)\s+of|among)\b(?:\W+\w+){0,5}\W*$")


def _text(value):
    value = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    return re.sub(r"[‐‑‒–—−-]", " ", value)


def _ordinal(word):
    if word in _ORDINALS:
        return _ORDINALS[word]
    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", word)
    return int(match[1]) if match else None


def _ranks(text):
    """Return (rank phrase, ordinal modifier, start offset) triples."""
    tokens = list(re.finditer(r"\b\w+\b", text))
    ranks = []
    consumed = set()
    for index, token in enumerate(tokens):
        word = token[0]
        phrase = None
        if word in _SUPERLATIVES:
            # Biggest and largest express the same size superlative. Height,
            # population, age, and other dimensions are not interchangeable.
            phrase = "largest" if word == "biggest" else word
        elif word in {"most", "least"} and index + 1 < len(tokens):
            phrase = word + " " + tokens[index + 1][0]
            consumed.add(index + 1)
        if phrase is not None:
            modifier = _ordinal(tokens[index - 1][0]) if index else None
            if modifier is not None:
                consumed.add(index - 1)
            ranks.append((phrase, modifier, token.start()))
    for index, token in enumerate(tokens):
        number = _ordinal(token[0])
        if number is not None and index not in consumed:
            ranks.append(("ordinal", number, token.start()))
    return ranks


def relevant_source_sentences(question, sentences):
    """Keep literal spans in order when explicit requested ranks are present.

    Negated ranked claims and 'one of the largest' do not establish the requested
    positive rank. Qualified rankings must match: second-largest is not largest.
    No sentence is rewritten, scored by vote, or supplied an expected answer.
    """
    question_text = _text(question)
    requested = {(phrase, number) for phrase, number, _ in _ranks(question_text)}
    if not requested:
        return list(sentences)
    # Negative ranking questions require a different logical operation. Do not
    # treat positive mentions as answers to them or guess their complement.
    if _NEGATION.search(question_text):
        return []
    eligible = []
    for sentence in sentences:
        text = _text(sentence)
        if _NEGATION.search(text):
            continue
        offered = set()
        for phrase, number, offset in _ranks(text):
            if _QUALIFIED_RANK.search(text[:offset]):
                continue
            offered.add((phrase, number))
        if requested <= offered:
            eligible.append(sentence)
    return eligible
