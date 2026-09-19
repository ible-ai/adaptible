"""No-network regressions for search-card attribution and quorum lineage."""

import unittest
from unittest import mock

from adaptible.wrap.references import (
    ReferenceSearchError,
    SEARCH_BACKENDS,
    _search_sync,
    _sources,
)
from adaptible.wrap.source_lineage import (
    attribution_conflicts,
    evidence_lineages,
)


def source(site, quote, *, title="A page about the question", text=None):
    return dict(
        url=f"https://{site}/fact",
        title=title,
        text=quote if text is None else text,
        quote=quote,
    )


class BackendSelectionTest(unittest.TestCase):
    def test_only_explicit_supported_backends_are_requested(self):
        from ddgs.engines import ENGINES

        with mock.patch("ddgs.DDGS") as factory:
            search = factory.return_value.__enter__.return_value.text
            search.return_value = []
            self.assertEqual(_search_sync("A question", 5, 20), [])
            requested = search.call_args.kwargs["backend"].split(",")
            self.assertEqual(requested, list(SEARCH_BACKENDS))
            self.assertTrue(set(requested) <= ENGINES["text"].keys())
            self.assertFalse(set(requested) & {"auto", "all", "yahoo"})

    def test_missing_supported_engines_never_fall_back_to_auto(self):
        with (
            mock.patch("ddgs.engines.ENGINES", {"text": {"yahoo": object()}}),
            mock.patch("ddgs.DDGS") as factory,
        ):
            with self.assertRaisesRegex(ReferenceSearchError, "No supported"):
                _search_sync("A question", 5, 20)
            factory.assert_not_called()

    def test_partial_install_requests_only_available_allowed_engine(self):
        with (
            mock.patch(
                "ddgs.engines.ENGINES",
                {"text": {"google": object(), "yahoo": object()}},
            ),
            mock.patch("ddgs.DDGS") as factory,
        ):
            search = factory.return_value.__enter__.return_value.text
            search.return_value = []
            _search_sync("A question", 2, 20)
            self.assertEqual(search.call_args.kwargs["backend"], "google")


class AttributionTest(unittest.TestCase):
    def test_nested_provider_cards_cannot_vote_as_false_first_url(self):
        # Synthetic form of the DDGS 9.16 Yahoo parser's reproduced output.
        # Do not require future dependency versions to retain their parser bug.
        first_body = (
            "This first page has background about the region and contains no answer."
        )
        second_body = "The official capital is Larch. This is the constitutional capital and the place where the national assembly meets."
        malformed = [
            dict(
                href="https://alpha.example/fact",
                title="First background pageSecond official capital page",
                body=first_body + second_body,
            ),
            dict(
                href="https://beta.example/fact",
                title="Second official capital page",
                body=second_body,
            ),
        ]
        clean = _sources(malformed, 5)
        self.assertEqual([s["url"] for s in clean], ["https://beta.example/fact"])
        clean[0]["quote"] = "The official capital is Larch."
        self.assertEqual(len(evidence_lineages(clean)), 1)

    def test_contamination_is_detected_before_truncation_and_result_limit(self):
        child = source(
            "beta.example",
            "The official capital is Larch. "
            + "A substantial descriptive clause. " * 5,
            title="Second official capital source",
        )
        outer = source(
            "alpha.example",
            "",
            title="First page" * 30 + child["title"],
            text="Background. " * 200 + child["text"],
        )
        clean = _sources([outer, child], 1)
        self.assertEqual(clean[0]["url"], child["url"])

    def test_shared_short_heading_and_body_alone_do_not_prove_nested_card(self):
        a = source("alpha.example", "The capital is Larch.", title="Capital")
        b = source("beta.example", "The capital is Larch.", title="Capital")
        self.assertEqual(attribution_conflicts([a, b]), {})

    def test_common_long_body_without_nested_title_is_preserved_for_lineage(self):
        text = "The capital is Larch. " * 10
        rows = [
            source("alpha.example", text, title="First genuine article"),
            source("beta.example", text, title="Second genuine article"),
        ]
        self.assertEqual(attribution_conflicts(rows), {})
        self.assertEqual(len(evidence_lineages(rows)), 1)


class LineageTest(unittest.TestCase):
    def test_independent_wording_on_two_sites_can_corroborate(self):
        rows = [
            source("alpha.example", "The capital is Larch."),
            source(
                "beta.example",
                "Larch hosts the nation's parliament as its official capital.",
            ),
        ]
        self.assertEqual(evidence_lineages(rows), [[0], [1]])

    def test_short_identical_fact_with_distinct_context_counts_each_site(self):
        rows = [
            source(
                "alpha.example",
                "The capital is Larch.",
                text="The capital is Larch. Its parliament meets in a building by the river.",
            ),
            source(
                "beta.example",
                "The capital is Larch!",
                text="A history of the nation and its development. The capital is Larch!",
            ),
        ]
        self.assertEqual(evidence_lineages(rows), [[0], [1]])

    def test_short_identical_bodies_do_not_imply_shared_origin(self):
        rows = [
            source("alpha.example", "The capital is Larch."),
            source("beta.example", "The capital is Larch."),
        ]
        self.assertEqual(evidence_lineages(rows), [[0], [1]])

    def test_substantial_contained_body_is_one_vote(self):
        body = (
            "The capital is Larch. "
            + "Its national assembly meets beside the river in a historic building where representatives discuss the laws and policies of the country."
        )
        rows = [
            source("alpha.example", "The capital is Larch.", text=body),
            source(
                "beta.example",
                "The capital is Larch.",
                text="Background. " + body + " More detail.",
            ),
        ]
        self.assertEqual(evidence_lineages(rows), [[0, 1]])

    def test_containment_requires_complete_words(self):
        rows = [source("alpha.example", "York"), source("beta.example", "Yorkshire")]
        self.assertEqual(len(evidence_lineages(rows)), 2)

    def test_same_site_and_shared_quote_connections_are_transitive(self):
        body = (
            "Second statement about Larch. "
            + "The city contains the national assembly and a historic palace beside the river where officials meet to discuss national policies every week."
        )
        rows = [
            source("en.alpha.example", "First statement about Larch."),
            source("fr.alpha.example", "Second statement about Larch.", text=body),
            source("beta.example", "Second statement about Larch.", text=body),
        ]
        self.assertEqual(evidence_lineages(rows), [[0, 1, 2]])

    def test_empty_missing_or_unattributed_quotes_never_vote(self):
        self.assertEqual(
            evidence_lineages([source("alpha.example", ""), {}, dict(quote="Larch")]),
            [],
        )

    def test_duplicate_substantial_evidence_with_different_selected_quotes_merges(self):
        one = "Larch is the constitutional capital and contains the national assembly, royal palace and supreme court."
        two = "The largest commercial port and seat of the national executive is the neighboring city of Willow."
        rows = [
            source("alpha.example", one, text=one + " " + two),
            source("beta.example", two, text=one + " " + two),
        ]
        self.assertEqual(evidence_lineages(rows), [[0, 1]])
