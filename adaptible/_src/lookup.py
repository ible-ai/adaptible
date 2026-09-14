"""A sealed document store: the node's stand-in for looking something up.

The autonomous node searches the web. For a reproducible run the node instead
reads from a fixed set of short passages, so what it can find is known in
advance and the run says only what the model does with what it found.
"""

import re
import unicodedata
from typing import Mapping

_STOP = frozenset(
    "a an the of in on at to is are was were be been what which who whom whose where when "
    "how why does do did name city star our this that it its for by with as from and or".split()
)


def _terms(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return {w for w in re.findall(r"[a-z0-9]+", text) if w not in _STOP and len(w) > 1}


# What the demo node can look up: the five facts the cycle runs repair, the
# two control facts, and five distractors. Passages, not answer templates.
DEFAULT_PASSAGES = {
 "Morocco": "Morocco is a kingdom in North Africa. Its capital is Rabat, on the Atlantic coast; Casablanca is the largest city and the economic centre.",
 "Turkey": "Turkey straddles Europe and Asia. Ankara, in central Anatolia, has been the capital since 1923; Istanbul is the largest city.",
 "Australia": "Australia is a federation of six states and two territories. The federal capital is Canberra, a planned city; Sydney and Melbourne are larger.",
 "Philippines": "The Philippines is an archipelago in Southeast Asia. Manila is the capital, and Quezon City in the same metropolitan area is the most populous city.",
 "Nearest star": "The star nearest to Earth is the Sun, about 150 million kilometres away. The next nearest, Proxima Centauri, is about 4.2 light-years away.",
 "Vietnam": "Vietnam is in Southeast Asia. Hanoi, in the north, is the capital; Ho Chi Minh City in the south is the largest city.",
 "Gold": "Gold is a chemical element with symbol Au, from the Latin aurum, and atomic number 79.",
 "Canada": "Canada is the second-largest country by area. Ottawa is the capital; Toronto is the largest city.",
 "Brazil": "Brazil is the largest country in South America. Brasilia, inaugurated in 1960, is the capital; Sao Paulo is the largest city.",
 "Moon": "The Moon is Earth's only natural satellite, about 384,000 kilometres away, and the fifth-largest moon in the Solar System.",
 "Mount Everest": "Mount Everest, on the border of Nepal and China, is the highest mountain above sea level at 8,849 metres.",
 "Switzerland": "Switzerland is a federal republic in central Europe. Bern is the federal city; Zurich is the largest city.",
}


class DocStore:
    """Keyword-overlap retrieval over a handful of passages.

    Args:
        passages: ``{title: passage}``. Retrieval scores each passage by how many
            of the query's content words appear in its title or text.
        min_overlap: Fewest shared content words for a hit; below it ``search``
            returns ``None`` (the node found nothing and must leave the answer alone).
    """

    def __init__(self, passages: Mapping[str, str], min_overlap: int = 1) -> None:
        self._docs = {t: (p, _terms(t) | _terms(p)) for t, p in passages.items()}
        self._min_overlap = min_overlap

    def search(self, query: str) -> str | None:
        """Returns the best-matching passage for ``query``, or ``None`` if nothing overlaps."""
        q = _terms(query)
        if not q:
            return None
        best, best_n = None, 0
        for passage, terms in self._docs.values():
            n = len(q & terms)
            if n > best_n:
                best, best_n = passage, n
        return best if best_n >= self._min_overlap else None

    def __len__(self) -> int:
        return len(self._docs)
