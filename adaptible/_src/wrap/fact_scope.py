"""A small, fully consuming grammar for factual question scopes.

This recognizes a few question forms, not arbitrary English or factual truth.
Productive adjective spellings can collide with real names; they are considered
only in an explicit adjectival position, never between two nominal subjects.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class FactScope:
    subject: str
    relation: str
    qualifiers: tuple[str, ...] = ()
    adjectival: bool = False


# These are syntax/modifier guards, not entity or answer dictionaries.
_FORBIDDEN_SUBJECT = set(
    [
        "and",
        "or",
        "not",
        "no",
        "non",
        "ex",
        "never",
        "except",
        "excluding",
        "without",
        "before",
        "after",
        "during",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "as",
        "designated",
        "than",
        "which",
        "what",
        "who",
        "whose",
        "when",
        "where",
        "how",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "please",
        "tell",
        "explain",
        "answer",
        "reply",
        "respond",
        "only",
        "most",
        "largest",
        "biggest",
        "smallest",
        "population",
        "area",
        "metropolitan",
        "urban",
        "rural",
        "current",
        "former",
        "official",
        "unofficial",
        "administrative",
        "legislative",
        "judicial",
        "capital",
        "city",
        "cities",
        "currency",
        "used",
        "uses",
        "use",
        "name",
        "identify",
    ]
)
_QUALIFIER = r"(?:(?P<qualifier>administrative|legislative|judicial|official|constitutional)\s+)?"
_POP = r"(?:largest|biggest|most populous|most populated)"
_REQUEST = r"(?:what is|which is|name|identify)"
_CITY = rf"(?:(?:what|which) city|{_REQUEST} (?:the )?city (?:that|which))"
_CURRENCY = rf"(?:(?:what|which) currency|{_REQUEST} (?:the )?currency (?:that|which))"
_CAPITAL_PREDICATE = r"(?:is designated as|serves as|is)"
_POP_COUNT = (
    r"has the (?:most (?:inhabitants|residents|people)|(?:largest|greatest) population)"
)


def question_core(question):
    """Remove only a trailing, explicit answer-format instruction."""
    return re.sub(
        r"(?i)(?<=[?.!])\s*(?:please\s+)?(?:answer|reply|respond)\s+"
        r"(?:(?:with\s+only|only\s+with|with|only)\s+)?(?:the\s+)?"
        r"(?:(?:city|country|person)\s+)?(?:name|number|answer)(?:\s+only)?"
        r"(?:,?\s+(?:with\s+)?(?:no|without)\s+explanation)?[.!]?\s*$",
        "",
        question,
    ).strip()


def _subject(value):
    value = value.strip()
    if (
        not value
        or len(value) > 160
        or not re.fullmatch(r"[^\W\d_]+(?:[ '’\-][^\W\d_]+)*", value)
    ):
        return None
    words = re.findall(r"[^\W\d_]+", value)
    if len(words) > 8 or set(words) & _FORBIDDEN_SUBJECT:
        return None
    return value


def adjective_pair(base, adjective):
    """Return a bounded productive spelling candidate, without a name list."""
    if not base.isalpha() or not adjective.isalpha():
        return False
    stems = {base}
    if base.endswith(tuple("aeiouy")):
        stems.add(base[:-1])
    return any(
        len(stem) >= 4 and adjective == stem + suffix
        for stem in stems
        for suffix in ("an", "ian", "ean", "ese", "ish")
    )


def parse_question(question):
    """Return a complete recognized scope, or None for unsupported wording."""
    if not isinstance(question, str):
        return None
    text = (
        unicodedata.normalize("NFKC", question_core(question))
        .replace("’", "'")
        .casefold()
        .strip()
    )
    if "\n" in text:
        return None
    text = text.rstrip("?.!").strip()
    # No clause splitting or substring extraction: each regex consumes the
    # entire question, and residual dates/modifiers cannot be swallowed as names.
    population = [
        rf"(?:what is|which is|name|identify) (?:the )?{_POP} city in (?P<subject>.+?)(?: by population)?",
        rf"{_CITY} in (?P<subject>.+?) is (?:the )?{_POP}(?: by population)?",
        rf"{_CITY} is (?:the )?{_POP} in (?P<subject>.+?)(?: by population)?",
        rf"{_CITY} in (?P<subject>.+?) {_POP_COUNT}",
        rf"(?:what is|which is|name|identify) (?P<subject>.+?)'s (?:the )?{_POP} city(?: by population)?",
        rf"(?:what|which) (?P<adjective>.+?) city is (?:the )?{_POP}(?: by population)?",
        rf"(?:what|which) (?P<adjective>.+?) city {_POP_COUNT}",
        rf"{_CITY} {_POP_COUNT} in (?P<subject>.+?)",
    ]
    capital = [
        rf"(?:what is|which is|name|identify) (?:the )?{_QUALIFIER}capital(?: city)? of (?P<subject>.+?)",
        rf"{_CITY} {_CAPITAL_PREDICATE} (?:the )?{_QUALIFIER}capital of (?P<subject>.+?)",
        rf"(?:what is|which is|name|identify) (?P<subject>.+?)'s {_QUALIFIER}capital(?: city)?",
        rf"{_CITY} {_CAPITAL_PREDICATE} (?P<subject>.+?)'s {_QUALIFIER}capital",
        rf"{_CITY} in (?P<subject>.+?) {_CAPITAL_PREDICATE} (?:the )?{_QUALIFIER}capital",
        rf"(?:what|which) (?P<adjective>.+?) city {_CAPITAL_PREDICATE} (?:the )?{_QUALIFIER}capital",
    ]
    currency = [
        r"(?:what is|which is|name|identify) (?:the )?currency (?:of|used in) (?P<subject>.+?)",
        rf"{_CURRENCY} is used in (?P<subject>.+?)",
        r"(?:what|which) currency does (?P<subject>.+?) use",
        r"(?:what is|which is|name|identify) (?P<subject>.+?)'s currency",
    ]
    for patterns in (population, capital):
        adjectival = [pattern for pattern in patterns if "(?P<adjective>" in pattern]
        patterns.extend(
            pattern.replace(
                r"(?:what|which) (?P<adjective>.+?) city",
                rf"{_REQUEST} (?:the )?(?P<adjective>.+?) city (?:that|which)",
            )
            for pattern in adjectival
        )
    for relation, patterns in (
        ("city_population_max", population),
        ("capital", capital),
        ("currency", currency),
    ):
        for pattern in patterns:
            match = re.fullmatch(pattern, text)
            if match is None:
                continue
            groups = match.groupdict()
            subject = _subject(groups.get("subject") or groups.get("adjective"))
            if subject is None:
                continue
            qualifier = groups.get("qualifier")
            return FactScope(
                subject,
                relation,
                (qualifier,) if qualifier else (),
                bool(groups.get("adjective")),
            )
    return None


def equivalent(left, right):
    """Compare parsed scopes; nominal spelling similarities do not authorize aliases."""
    if (
        left is None
        or right is None
        or left.relation != right.relation
        or left.qualifiers != right.qualifiers
    ):
        return False
    if left.adjectival == right.adjectival:
        return left.subject == right.subject
    nominal, adjective = (right, left) if left.adjectival else (left, right)
    base_words, adjective_words = nominal.subject.split(), adjective.subject.split()
    if (
        len(base_words) != len(adjective_words)
        or base_words[:-1] != adjective_words[:-1]
    ):
        return False
    return adjective_pair(base_words[-1], adjective_words[-1])
