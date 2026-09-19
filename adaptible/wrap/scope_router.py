"""Conservative question-only routing to the latest cumulative adapter.

Lexical guards reduce the small classifier's opportunity for false positives;
this is not a semantic-equivalence proof. Unrecognized wording can abstain.
"""

import json
import re
import unicodedata
from collections import OrderedDict

import httpx

from .fact_scope import equivalent, parse_question, question_core
from .question_candidate import candidate_pair
from .scope_classifier import classification_messages

# Ordinary English request/function words, never answer or benchmark vocabulary.
_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "how",
        "please",
        "tell",
        "me",
        "us",
        "name",
        "identify",
        "give",
        "state",
        "say",
        "answer",
        "question",
        "city",
        "cities",
        "town",
        "towns",
        "country",
        "countries",
        "place",
        "places",
        "its",
        "it",
        "that",
        "this",
        "these",
        "those",
        "there",
        "and",
        "or",
        "about",
        "called",
        "known",
        "serves",
        "serve",
    ]
)
_RELATIONS = {
    "capital": "capital",
    "capitals": "capital",
    "largest": "maximum_size",
    "biggest": "maximum_size",
    "populous": "population",
    "populated": "population",
    "smallest": "minimum_size",
    "tallest": "maximum_height",
    "highest": "maximum_height",
    "lowest": "minimum_height",
    "longest": "maximum_length",
    "shortest": "minimum_length",
    "oldest": "maximum_age",
    "youngest": "minimum_age",
    "nearest": "minimum_distance",
    "closest": "minimum_distance",
    "farthest": "maximum_distance",
    "furthest": "maximum_distance",
    "president": "president",
    "presidents": "president",
    "population": "population",
    "residents": "population",
    "inhabitants": "population",
    "people": "population",
}
_QUALIFIERS = set(
    [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "current",
        "currently",
        "former",
        "formerly",
        "previous",
        "previously",
        "historical",
        "historic",
        "ancient",
        "modern",
        "official",
        "unofficial",
        "administrative",
        "legislative",
        "judicial",
        "economic",
        "area",
        "land",
        "metropolitan",
        "metro",
        "urban",
        "rural",
        "proper",
        "nominal",
        "real",
        "total",
        "average",
        "mean",
        "median",
        "per",
        "capita",
        "not",
        "never",
        "except",
        "excluding",
        "least",
        "before",
        "after",
        "during",
    ]
)


def _words(text):
    text = re.sub(r"(?<=\w)['’]s\b", "", text)
    return re.findall(
        r"\w+(?:['’]\w+)?", unicodedata.normalize("NFKC", text).casefold()
    )


def _question(text):
    return question_core(text)


def _key(text):
    # Do not erase punctuation such as a minus sign, apostrophe, or rank dash.
    return " ".join(
        unicodedata.normalize("NFKC", _question(text)).casefold().split()
    ).rstrip("?.")


def _signature(text):
    words = _words(_question(text))
    values = set(words)
    relations = {_RELATIONS[w] for w in values if w in _RELATIONS}
    if "maximum_size" in relations:
        relations.discard("population")
    if "population" in relations and values & {"most", "greatest"}:
        relations.discard("population")
        relations.add("maximum_size")
    # Preserve the requested answer category; "largest country" must not
    # become the same scope as "largest city" after function-word filtering.
    kinds = {
        "city": "city",
        "cities": "city",
        "town": "town",
        "towns": "town",
        "country": "country",
        "countries": "country",
    }
    if "capital" not in relations:
        relations |= {"object:" + kinds[w] for w in values if w in kinds}
    qualifiers = values & _QUALIFIERS
    qualifiers |= set(re.findall(r"(?<!\w)[+−-]\d+(?:\.\d+)?", _question(text)))
    qualifiers |= {w for w in values if any(c.isdigit() for c in w)}
    if any(w.endswith(("n't", "n’t")) for w in values):
        qualifiers.add("not")
    topics = (
        values
        - _STOP
        - set(_RELATIONS)
        - _QUALIFIERS
        - {
            "most",
            "greatest",
            "size",
            "number",
            "many",
            "more",
        }
    )
    return topics, relations, qualifiers


def _grammar_match(left, right):
    return equivalent(parse_question(left), parse_question(right))


def _shortlist(left, right):
    parsed_left, parsed_right = parse_question(left), parse_question(right)
    if parsed_left is not None or parsed_right is not None:
        # Once a directed relation is recognized, unordered fallback tokens
        # cannot authorize an unparsed inverse or extra clause.
        return equivalent(parsed_left, parsed_right)
    a, ar, aq = _signature(left)
    b, br, bq = _signature(right)
    # Discovery can allow predicate synonyms, but never inferred name aliases,
    # incompatible categories, explicit qualifiers, or reversed argument roles.
    return ar == br and aq == bq and candidate_pair(left, right)


def _plain_question(messages):
    if not isinstance(messages, list) or not messages or len(messages) > 32:
        return None
    for entry in messages:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"role", "content"}
            or entry.get("role") not in {"system", "user", "assistant"}
            or not isinstance(entry.get("content"), str)
        ):
            return None
    if sum(len(entry["content"]) for entry in messages) > 32000:
        return None
    message = messages[-1]
    if not isinstance(message, dict) or set(message) - {"role", "content"}:
        return None
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return None
    text = message["content"].strip()
    if not text or len(text) > 2000 or any(x in text for x in ("\n", "<", ">", "```")):
        return None
    return text


def _mode(mode):
    if not isinstance(mode, dict) or mode.get("error"):
        return "invalid"
    thinking = mode.get("thinking", False)
    if type(thinking) is not bool:
        return "invalid"
    if "model_reasons_unconditionally" in (mode.get("source") or []):
        # This model decides per turn whether to reason and the request carries
        # no control to ask either way, so thinking describes one answer rather
        # than the request. Partitioning scopes on it files the repair and the
        # identical re-ask under different modes, and serves the frozen base to
        # both -- the correction trains, is accepted, and is then never used.
        return "model_decides"
    return thinking


class ScopeRouter:
    def __init__(self, runtime, *, cache_size=256):
        self.runtime = runtime
        self.cache_size = max(0, min(cache_size, 1024))
        self.cache = OrderedDict()

    async def choose(self, messages, repairs, mode):
        question = _plain_question(messages)
        if question is None or _mode(mode) == "invalid":
            return dict(scope=None, reason="unsupported_context")
        # Explicit allowlist: neither classifier nor cache sees reference notes,
        # expected names, evidence, or adapter paths. Exact context is cached
        # for prefix matching but is never sent to the classifier.
        grouped = {}
        for repair in repairs:
            if (
                not isinstance(repair, dict)
                or type(repair.get("interaction_idx")) is not int
            ):
                continue
            original = repair.get("question")
            prompts = repair.get("prompts", [])
            if not isinstance(prompts, list):
                prompts = []
            original_messages = repair.get(
                "messages", [{"role": "user", "content": original}]
            )
            if _plain_question(original_messages) is None:
                continue
            if not isinstance(original, str) or not original.strip():
                original = original_messages[-1]["content"]
            prefix = original_messages[:-1]
            variants = []
            for prompt in prompts:
                if isinstance(prompt, str):
                    variants.append(prompt)
                elif _plain_question(prompt) is not None and prompt[:-1] in (
                    [],
                    prefix,
                ):
                    variants.append(prompt[-1]["content"])
            questions = list(
                dict.fromkeys(
                    value
                    for value in [original, *variants]
                    if isinstance(value, str) and value.strip() and len(value) <= 2000
                )
            )
            original_scope = parse_question(original)
            if original_scope is not None:
                questions = [
                    value
                    for value in questions
                    if equivalent(original_scope, parse_question(value))
                ]
            if questions:
                repair_mode = _mode(repair.get("generation_mode", {}))
                identity = json.dumps(
                    [_key(original), repair_mode, prefix],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                previous = grouped.get(identity)
                idx = repair["interaction_idx"]
                if previous is not None:
                    idx = max(idx, previous[0])
                    questions = list(dict.fromkeys([*previous[1], *questions]))
                # A repeated repair extends the same learned scope. Different
                # original questions remain separate even if their vocabulary
                # overlaps; no model judgment is used to merge identities.
                grouped[identity] = (idx, questions, repair_mode, prefix)
        scopes = sorted(grouped.values(), key=lambda scope: scope[0])
        key = json.dumps(
            [messages, scopes, _mode(mode)], ensure_ascii=False, sort_keys=True
        )
        if key in self.cache:
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        result = await self._choose(question, scopes, _mode(mode), messages[:-1])
        if self.cache_size:
            self.cache[key] = dict(result)
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result

    async def _choose(self, question, scopes, mode, prefix):
        contexts = [s for s in scopes if s[3] == prefix]
        if scopes and not contexts:
            return dict(scope=None, reason="unsupported_context")
        eligible = [s for s in contexts if s[2] == mode]
        for idx, prompts, _, _ in eligible:
            if any(_key(question) == _key(p) for p in prompts):
                return dict(scope=idx, reason="exact")
        grammar_candidates = [
            idx
            for idx, prompts, _, _ in eligible
            if _grammar_match(question, prompts[0])
        ]
        if len(grammar_candidates) == 1:
            return dict(
                scope=grammar_candidates[0],
                reason="scope_match",
                matcher="question_grammar_v1",
            )
        if grammar_candidates:
            return dict(scope=None, reason="router_abstained")
        candidates = [
            (idx, prompts)
            for idx, prompts, _, _ in eligible
            if _shortlist(question, prompts[0])
        ]
        if not candidates:
            mismatch = any(
                m != mode
                and any(
                    _key(question) == _key(p) or _shortlist(question, p)
                    for p in prompts
                )
                for _, prompts, m, _ in contexts
            )
            return dict(scope=None, reason="mode_mismatch" if mismatch else "no_scope")
        # Ambiguous or excessively broad scope sets abstain without model calls.
        if len(candidates) != 1:
            return dict(scope=None, reason="router_abstained")
        idx, prompts = candidates[0]
        schema = dict(
            type="object",
            properties={"same_fact": {"type": "boolean"}},
            required=["same_fact"],
            additionalProperties=False,
        )
        classifier = dict(
            input=dict(question_a=prompts[0], question_b=question),
            scope_questions=prompts,
            raw_output=None,
            generation={},
            verdict="transport_error",
        )
        try:
            result = await self.runtime.complete(
                classification_messages(prompts[0], question),
                frozen=True,
                temperature=0,
                seed=7300,
                details=classifier["generation"],
                response_format=dict(
                    type="json_schema",
                    json_schema=dict(name="same_fact", schema=schema),
                ),
            )
            classifier["raw_output"] = result
            classifier["verdict"] = "malformed"
            parsed = json.loads(result)
        except (
            ValueError,
            TypeError,
            RuntimeError,
            TimeoutError,
            httpx.HTTPError,
        ) as exc:
            classifier["error_type"] = type(exc).__name__
            return dict(scope=None, reason="router_abstained", classifier=classifier)
        generation = classifier["generation"]
        if (
            generation.get("complete") is not True
            or generation.get("framing_valid") is not True
            or generation.get("finish_reason") not in ("stop", "eos", "end_turn")
        ):
            # A truncated response can happen to contain complete JSON. Parsing
            # that prefix must not authorize a scope the generation did not finish.
            classifier["verdict"] = "incomplete_generation"
            return dict(scope=None, reason="router_abstained", classifier=classifier)
        if (
            isinstance(parsed, dict)
            and set(parsed) == {"same_fact"}
            and type(parsed["same_fact"]) is bool
        ):
            classifier["verdict"] = (
                "same_fact" if parsed["same_fact"] else "different_fact"
            )
        if (
            isinstance(parsed, dict)
            and set(parsed) == {"same_fact"}
            and parsed["same_fact"] is True
        ):
            return dict(scope=idx, reason="scope_match", classifier=classifier)
        return dict(scope=None, reason="router_abstained", classifier=classifier)
