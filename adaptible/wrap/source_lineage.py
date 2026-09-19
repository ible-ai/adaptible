"""Conservative attribution and duplicate-evidence guards for search snippets.

Different URLs are not proof of independent evidence. These guards detect
known nested-result contamination and merge obvious shared textual evidence;
they do not certify publisher independence or discover undetectable copying.
"""

import re
import unicodedata
from urllib.parse import urlsplit


def source_site(url):
    # Conservatively collapse subdomains and multi-label public suffixes.
    host = (urlsplit(url).hostname or "").casefold().rstrip(".")
    return ".".join(host.split(".")[-2:])


def _text(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def _contains(large, small):
    return bool(small) and f" {small} " in f" {large} "


def attribution_conflicts(sources):
    """Return indices of containers incorporating a different result's card.

    Require both a substantial nested title and substantial nested body. Plain
    citations, shared short headings, and a common fact alone are insufficient.
    Inspect untruncated bounded provider fields before selecting the first five.
    """
    normalized = [(_text(s["title"]), _text(s["text"])) for s in sources]
    rejected = {}
    for i, outer in enumerate(sources):
        title, body = normalized[i]
        for j, inner in enumerate(sources):
            other_title, other_body = normalized[j]
            if (
                i != j
                and source_site(outer["url"]) != source_site(inner["url"])
                and len(other_title) >= 16
                and title != other_title
                # DDGS joins titles without intervening whitespace.
                and other_title in title
                and len(other_body) >= 80
                and _contains(body, other_body)
            ):
                rejected[i] = {
                    "reason": "nested_search_result",
                    "nested_url": inner["url"],
                }
                break
    return rejected


def evidence_lineages(sources):
    """Group agreeing sources that cannot supply separate corroboration votes.

    Same website or substantial duplicate/contained source bodies connect
    lineages transitively when that shared passage contains both selected
    claims. Short identical fact clauses alone do not imply shared lineage.
    Empty quotes never vote; this heuristic does not prove independence.
    Returns source-index groups for transparent persisted diagnostics.
    """
    valid = [
        i
        for i, s in enumerate(sources)
        if source_site(s.get("url", "")) and _text(s.get("quote", ""))
    ]
    parents = {i: i for i in valid}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for pos, i in enumerate(valid):
        first = sources[i]
        quote = _text(first["quote"])
        body = _text(first.get("text", ""))
        for j in valid[pos + 1 :]:
            second = sources[j]
            other_quote = _text(second["quote"])
            other_body = _text(second.get("text", ""))
            shared = body if len(body) <= len(other_body) else other_body
            container = other_body if shared == body else body
            duplicated_passage = (
                len(shared) >= 120
                and len(shared.split()) >= 20
                and _contains(container, shared)
                and _contains(shared, quote)
                and _contains(shared, other_quote)
            )
            if (
                source_site(first["url"]) == source_site(second["url"])
                or duplicated_passage
            ):
                parents[root(j)] = root(i)
    groups = {}
    for i in valid:
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())
