"""Necessary guards for classifying two otherwise unparsed factual questions.

Explicit names, compatible known output categories, and directional/qualifier
constraints only shortlist a pair. Unknown categories require the same semantic
classifier; no candidate alone establishes equivalence. Missing or ambiguous
names and extra clauses conservatively abstain.
"""

import re
import unicodedata

from .fact_scope import question_core

_HEADS = {
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "how",
    "name",
    "identify",
    "tell",
    "please",
    "on",
    "at",
}
_AMBIGUOUS = set(
    [
        "it",
        "its",
        "they",
        "them",
        "their",
        "he",
        "him",
        "his",
        "she",
        "her",
        "this",
        "that",
        "these",
        "those",
        "another",
        "other",
        "same",
        "and",
        "or",
        "also",
        "then",
        "except",
        "excluding",
        "including",
        "unless",
        "if",
        "because",
        "although",
        "whether",
    ]
)
_UNITS = set(
    [
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "hour",
        "hours",
        "minute",
        "minutes",
        "second",
        "seconds",
        "meter",
        "meters",
        "kilometre",
        "kilometres",
        "kilometer",
        "kilometers",
        "mile",
        "miles",
        "gram",
        "grams",
        "kilogram",
        "kilograms",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "percent",
        "percentage",
        "annually",
        "monthly",
        "daily",
        "hourly",
    ]
)
_TIME = set(
    [
        "now",
        "today",
        "yesterday",
        "tomorrow",
        "currently",
        "previously",
        "former",
        "formerly",
        "historical",
        "ancient",
        "future",
        "past",
        "present",
        "before",
        "after",
        "during",
        "since",
        "until",
    ]
)
_CATEGORIES = {
    "person": {"person", "individual"},
    "organization": {
        "company",
        "organization",
        "organisation",
        "institution",
        "publisher",
        "manufacturer",
    },
    "location": {"venue", "location", "address"},
    "duration": {"duration"},
    "material": {"material", "substance"},
    "amount": {"price", "cost", "fee"},
    "time": {"date"},
    "city": {"city"},
    "country": {"country"},
}


_WH = frozenset({"who", "whom", "what", "which", "where", "when", "how"})
_PREPOSITIONS = frozenset({"in", "at", "on", "by", "from", "to", "for", "with", "of"})
_AUX = frozenset(
    {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "will"}
)
_COPULAS = frozenset({"is", "are", "was", "were"})
_ARTICLES = frozenset({"the", "a", "an"})
_PAST_AUX = frozenset({"was", "were", "did", "had"})
_MODALS = frozenset({"may", "might", "could", "would", "will", "should", "must"})
_CLAUSE_MARKERS = frozenset({"but", "while", "however", "yet"})
_IRREGULAR_PASSIVES = frozenset(
    {"built", "made", "known", "run", "written", "given", "taken", "seen"}
)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)?", text.casefold())


def explicit_names(question: str) -> frozenset[str]:
    """Extract complete capitalized name spans without inferring name aliases."""
    text = unicodedata.normalize("NFKC", question_core(question)).replace("’", "'")
    found = []
    current = []
    end = None
    for token in re.finditer(r"[^\W\d_]\w*(?:'s)?", text):
        word = token[0]
        if word[0].isupper() and not (token.start() == 0 and word.casefold() in _HEADS):
            if current and text[end : token.start()] not in (" ", "-", "'"):
                found.append("".join(current))
                current = []
            if current:
                current.append(text[end : token.start()])
            current.append(word.removesuffix("'s"))
            end = token.end()
        else:
            if current:
                found.append("".join(current))
                current = []
            end = None
    if current:
        found.append("".join(current))
    return frozenset(name.casefold() for name in found)


def _name_text(text: str) -> str:
    """Exclude an initial preposition only when it introduces the WH phrase."""
    return re.sub(
        r"^(In|At|On|By|From|To|For|With|Of)(?=\s+(?:which|what|who|whom)\b)",
        lambda match: match[0].lower(),
        text,
    )


def _contextual_names(
    text: str,
    other: str,
    names: frozenset[str],
    other_names: frozenset[str],
) -> frozenset[str]:
    """Disambiguate an initial article using the counterpart's literal syntax.

    A leading 'The X' can align with lowercase grammatical 'the X' in the other
    question. Quoted titles and state-name prefixes receive no alias treatment.
    """
    leading_article = re.match(r"^(The|A|An)\s+", text)
    if leading_article is None:
        return names
    article = leading_article[1].lower()
    marked = re.search(r"(?<!\w)" + article + r"\s+([^?!.;,]+)", other)
    if marked is None:
        return names
    result = set(names)
    for full_name in names:
        if not full_name.startswith(article + " "):
            continue
        remainder = full_name[len(article) + 1 :]
        if remainder not in other_names or not marked[1].casefold().startswith(
            remainder
        ):
            continue
        # The counterpart must contain the complete name, not a shared prefix.
        tail = marked[1][len(remainder) :]
        if not tail or not tail[0].isalnum():
            result.remove(full_name)
            result.add(remainder)
    return frozenset(result)


def _paired_names(left: str, right: str) -> tuple[frozenset[str], frozenset[str]]:
    left_names = explicit_names(_name_text(left))
    right_names = explicit_names(_name_text(right))
    return (
        _contextual_names(left, right, left_names, right_names),
        _contextual_names(right, left, right_names, left_names),
    )


def _category(wh: str, tail: list[str]) -> str | None:
    if wh in ("who", "whom"):
        return "person"
    if wh == "where":
        return "location"
    if wh == "when":
        return "time"
    if wh == "how":
        modifier = tail[0] if tail else ""
        return {"long": "duration", "much": "amount", "many": "count"}.get(modifier)
    while tail and tail[0] in _AUX | _ARTICLES:
        tail = tail[1:]
    noun_phrase = []
    for token in tail:
        if token in _AUX | _PREPOSITIONS:
            break
        noun_phrase.append(token)
    head = noun_phrase[-1] if noun_phrase else None
    return next((kind for kind, nouns in _CATEGORIES.items() if head in nouns), None)


def _passive_patient(tail: list[str]) -> bool:
    """Recognize a passive predicate before its 'by' complement.

    Intervening adverbs do not change the requested argument. Do not search
    through a noun phrase for a later participle ('the author of ... written by').
    """
    if "by" not in tail:
        return False
    predicate = tail[1 : tail.index("by")]
    while predicate and predicate[-1].endswith("ly"):
        predicate = predicate[:-1]
    if not predicate or set(predicate) & (_PREPOSITIONS | _WH | _ARTICLES):
        return False
    return predicate[-1].endswith("ed") or predicate[-1] in _IRREGULAR_PASSIVES


def _direction(wh: str, prefix: list[str], tail: list[str]) -> str | None:
    if prefix[-1:] == ["by"]:
        return "agent"
    if wh not in ("who", "whom"):
        return None
    if wh == "whom" or tail[:1] in (["do"], ["does"], ["did"]):
        return "patient"
    if tail and tail[0] in _COPULAS:
        return "patient" if _passive_patient(tail) else None
    return "agent" if tail else None


def _wh_frame(words: list[str]) -> dict | None:
    positions = [index for index, word in enumerate(words) if word in _WH]
    if len(positions) != 1:
        return None
    index = positions[0]
    wh = words[index]
    prefix = words[:index]
    leading_preposition = len(prefix) == 1 and prefix[0] in _PREPOSITIONS
    # An ordinary passive clause can put its requested agent at the end.
    passive_clause = prefix[-1:] == ["by"] and bool(set(prefix) & _AUX)
    if index and not (leading_preposition or passive_clause):
        return None
    tail = words[index + 1 :]
    return dict(
        wh_index=index,
        category=_category(wh, tail),
        direction=_direction(wh, prefix, tail),
    )


def _dummy_it(text: str) -> bool:
    return bool(
        re.match(
            r"^how (?:much|long) (?:does|did|will) it (?:cost|take) to \w+",
            text.casefold(),
        )
    )


def _ambiguous_clause(text: str, words: list[str]) -> bool:
    ambiguous = set(words) & (_AMBIGUOUS | _CLAUSE_MARKERS)
    if ambiguous == {"it"} and _dummy_it(text):
        ambiguous = set()
    return bool(ambiguous or re.search(r"[;,:\n]|[?!.].+\w", text))


def _qualifiers(words: list[str]) -> set[str]:
    return (set(words) & (_UNITS | _TIME | _MODALS)) | {
        word for word in words if any(char.isdigit() for char in word)
    }


def candidate_diagnostics(left: str, right: str) -> dict:
    """Explain whether a pair may reach the question-only semantic classifier.

    Returns an eligibility flag and reason, with extracted names and frames when
    available. Eligibility is a necessary precondition, never a routing verdict.
    """
    left, right = question_core(left), question_core(right)
    left_words, right_words = _words(left), _words(right)
    result = dict(eligible=False, reason=None)
    if _ambiguous_clause(left, left_words) or _ambiguous_clause(right, right_words):
        return dict(result, reason="ambiguous_reference_or_extra_clause")
    left_names, right_names = _paired_names(left, right)
    result.update(names_a=sorted(left_names), names_b=sorted(right_names))
    if not left_names or left_names != right_names:
        return dict(result, reason="different_or_missing_explicit_names")
    left_frame, right_frame = _wh_frame(left_words), _wh_frame(right_words)
    result.update(frame_a=left_frame, frame_b=right_frame)
    if left_frame is None or right_frame is None:
        return dict(result, reason="unsupported_question_structure")
    known_categories = bool(left_frame["category"] and right_frame["category"])
    if known_categories and left_frame["category"] != right_frame["category"]:
        return dict(result, reason="incompatible_known_categories")
    if _qualifiers(left_words) != _qualifiers(right_words):
        return dict(result, reason="qualifier_mismatch")
    if bool(set(left_words) & _PAST_AUX) != bool(set(right_words) & _PAST_AUX):
        return dict(result, reason="tense_mismatch")
    known_directions = bool(left_frame["direction"] and right_frame["direction"])
    if known_directions and left_frame["direction"] != right_frame["direction"]:
        return dict(result, reason="direction_mismatch")
    return dict(
        result,
        eligible=True,
        reason="candidate_requires_classifier",
        category_uncertain=not known_categories,
    )


def candidate_pair(left: str, right: str) -> bool:
    """Shortlist question-only candidates; a completed classifier must decide."""
    return candidate_diagnostics(left, right)["eligible"]
