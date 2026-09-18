"""Necessary current-capital source constraints, not semantic entailment.

Supports explicit copular, possessive, appositive, continuing-present,
status-as, and labelled capital claims with literal jurisdiction names.
Government seats and formal state-name variants are not assumed equivalent.
Unknown capital-question forms, historical queries, missing requested capital
qualifiers, and dated designation events abstain. Other factual families keep
their existing path. This deliberately sacrifices coverage rather than treating
word presence as entailment. No answer dictionary or quote rewriting is used.
"""

import re
import unicodedata

from .fact_scope import parse_question

_QUALIFIERS = r"(?:administrative|legislative|judicial|official|constitutional|national|political|federal)"

# Modifiers real sources put in front of "capital". Sources rarely write the
# bare qualifier the question uses: they coordinate several ("administrative
# and legislative capital") and mark currency ("the current legislative
# capital"). None of these words weakens the claim -- historical, designation
# and hypothetical wording is rejected before this runs.
_MODIFIER = (
    r"(?:current|present|currently|today's|sole|only|de\s+jure|de\s+facto|new|"
    r"administrative|legislative|judicial|official|constitutional|national|"
    r"political|federal)"
)
_MODIFIER_RUN = r"(?P<mods>(?:" + _MODIFIER + r"(?:\s+and\s+|\s*,\s*|\s+)){0,4})"
_NEGATION = re.compile(
    r"\b(?:not|never|neither|nor|false|untrue|incorrect)\b|\bno\s+longer\b|\b\w+n't\b"
)
_HISTORY = re.compile(
    r"\b(?:former|formerly|previous|first|old)\s+(?:\w+\s+){0,2}(?:capital|capitol|seat of government)\b"
    r"|\bused\s+to\s+(?:be|serve)\b"
    r"|\b(?:was|were)\s+(?:the\s+)?(?:"
    + _QUALIFIERS
    + r"\s+)?(?:capital|capitol|seat of government)\b"
)
_DESIGNATION = re.compile(
    r"\b(?:was|were)\s+(?:\w+\s+){0,3}(?:declared|named|designated|inaugurated)\b|\bbecame\s+(?:the\s+)?(?:capital|seat of government)\b"
)
_HYPOTHETICAL = re.compile(
    r"\b(?:may|might|could|would|will)\s+(?:become|be|serve as)\s+(?:the\s+)?(?:"
    + _QUALIFIERS
    + r"\s+)?(?:capital|seat of government)\b"
    r"|\b(?:proposed|planned|future)\s+(?:\w+\s+){0,2}capital\b"
)


def _text(value):
    return " ".join(
        unicodedata.normalize("NFKC", value).replace("’", "'").casefold().split()
    )


def _patterns(subject, qualifiers, candidate):
    # Jurisdiction names remain literal. State-type prefixes can distinguish
    # different countries and must never be treated as optional aliases.
    owner = re.escape(subject)
    owner_end = r"(?=$|[,.;:]|\s+since\b)"
    # The requested qualifier is checked against the matched run afterwards, so
    # it may appear anywhere in it rather than immediately before "capital".
    capital = _MODIFIER_RUN + r"capital(?:\s+city)?"
    value = re.escape(candidate) if candidate is not None else r"[^.;:!?]{1,160}?"
    # Parenthetical pronunciation/clarification stays in the original quote;
    # matching it here does not remove or reconstruct evidence.
    value += r"(?:\s*\([^()]{0,160}\))?"
    prefix = r"(?:^|[.;,!?]\s*)"
    status_prefix = r"(?:^|[.;,!?]\s*|\b(?:cemented|cementing|confirmed|confirming|maintained|maintaining|recognized|recognizing)\s+)"
    return [
        prefix
        + value
        + r"\s+(?:is|remains|serves as|has been)\s+(?:the\s+)?"
        + capital
        + r"\s+of\s+"
        + owner
        + owner_end,
        prefix + value + r",\s*(?:the\s+)?" + capital + r"\s+of\s+" + owner + r"\s*,",
        status_prefix
        + value
        + r"'s\s+status\s+as\s+(?:the\s+)?"
        + capital
        + r"\s+of\s+"
        + owner
        + owner_end,
        prefix + owner + r"'s\s+" + capital + r"\s+is\s+" + value + r"(?=$|[,.;:])",
        prefix
        + r"(?:the\s+)?"
        + capital
        + r"\s+of\s+"
        + owner
        + r"\s+is\s+"
        + value
        + r"(?=$|[,.;:])",
        prefix
        + owner
        + r"[^\w:]{0,8}:\s*"
        + capital
        + r"\s+"
        + value
        + r"(?=$|[,.;:])",
    ]


def _matches(text, subject, qualifiers, candidate):
    """Whether a pattern matches and carries the qualifier the question asked for."""
    required = qualifiers[0].casefold() if qualifiers else None
    for pattern in _patterns(subject, qualifiers, candidate):
        for match in re.finditer(pattern, text):
            if required is None or required in (match.groupdict().get("mods") or ""):
                return True
    return False


def _negated(text):
    """Whether a negation applies to the clause that carries the capital claim.

    "Despite not being the largest city, X is the capital" negates a different
    property in a subordinate clause; rejecting the whole sentence for it loses
    an ordinary and correct source.
    """
    for clause in re.split(r"[,;:()]", text):
        if re.search(r"\bcapital\b", clause) and not _NEGATION.search(clause):
            return False
    return bool(_NEGATION.search(text))


def capital_source_check(question, sentence, candidate=None):
    """Return applicability, eligibility, and a diagnostic rejection reason.

    Call before extraction with candidate=None, then after extraction with the
    selected literal name. Passing this necessary guard does not prove truth.
    """
    scope = parse_question(question)
    question_text = _text(question)
    if scope is None or scope.relation != "capital":
        if re.search(r"\bcapital\b", question_text):
            return dict(
                applicable=True, eligible=False, reason="unsupported_capital_question"
            )
        return dict(applicable=False, eligible=True, reason="other_question_family")
    text = _text(sentence)
    if _negated(text):
        return dict(applicable=True, eligible=False, reason="negated_claim")
    if _HISTORY.search(text):
        return dict(applicable=True, eligible=False, reason="historical_role")
    if _DESIGNATION.search(text):
        return dict(
            applicable=True,
            eligible=False,
            reason="historical_designation_not_current_evidence",
        )
    if _HYPOTHETICAL.search(text):
        return dict(applicable=True, eligible=False, reason="hypothetical_role")
    if not re.search(r"\bcapital\b", text):
        return dict(applicable=True, eligible=False, reason="missing_capital_relation")
    if candidate is not None:
        if (
            not isinstance(candidate, str)
            or not candidate.strip()
            or len(candidate.split()) > 8
        ):
            return dict(applicable=True, eligible=False, reason="invalid_candidate")
        candidate = _text(candidate)
    eligible = _matches(text, scope.subject, scope.qualifiers, candidate)
    return dict(
        applicable=True,
        eligible=eligible,
        reason=(
            "direct_current_capital_claim"
            if eligible
            else "unsupported_relation_or_argument_role"
        ),
    )
