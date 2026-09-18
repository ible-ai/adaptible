"""Automatic reference discovery through the shared wrapper HTTP API.

The search and serving boundaries are deterministic: these tests need neither a
network connection nor a checkpoint, but exercise scheduling, grounded extraction,
training, adapter acceptance, and durable evidence together.
"""

import asyncio
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from adaptible._src.wrap.app import create_app
from adaptible._src.wrap.repair import Controller
from adaptible._src.wrap.references import ReferenceSearchError
from adaptible._src.wrap.store import Store
from adaptible.tests.wrap_test import FakeRuntime, FakeTrainer

QUESTION = "What is the capital of Australia?"
SOURCES = [
    dict(
        title="Australia's capital",
        url="https://nation.example/australia",
        text="Canberra is the capital of Australia.",
    ),
    dict(
        title="Visit the capital",
        url="https://travel.example/capital",
        text="The capital of Australia is Canberra.",
    ),
]


class SearchFixture:
    def __init__(self, sources=None):
        self.sources = copy.deepcopy(SOURCES if sources is None else sources)
        self.queries = []
        self.error = None
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def search(self, query):
        self.queries.append(query)
        self.started.set()
        try:
            if self.block:
                await self.release.wait()
            if self.error:
                raise self.error
            return copy.deepcopy(self.sources)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class AutomaticReferenceIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.search = SearchFixture()
        await self.open_wrapper()

    async def open_wrapper(self, **options):
        self.runtime = FakeRuntime()
        self.trainer = FakeTrainer()
        self.trainer.runtime = self.runtime
        self.store = Store(Path(self.temp.name), "automatic-reference-test")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            reference_search=self.search,
            idle_seconds=60,
            **options,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def close_wrapper(self):
        await self.client.aclose()
        await self.controller.close()

    async def asyncTearDown(self):
        if not self.controller.closing:
            await self.close_wrapper()
        self.temp.cleanup()

    async def flag(self, **feedback):
        response = await self.client.post("/interact", json={"prompt": QUESTION})
        self.assertEqual(response.status_code, 200)
        idx = response.json()["interaction_idx"]
        response = await self.client.post(
            "/feedback", json=dict(interaction_idx=idx, thumbs="down", **feedback)
        )
        self.assertEqual(response.status_code, 200)
        return idx

    async def review(self):
        response = await self.client.post("/trigger_review")
        self.assertEqual(response.status_code, 200)
        response = await asyncio.wait_for(self.client.get("/sync"), 3)
        self.assertEqual(response.status_code, 200)
        rows = (await self.client.get("/history")).json()["history"]
        return rows[-1]

    async def test_downvote_alone_discovers_trains_and_preserves_evidence_on_restart(
        self,
    ):
        idx = await self.flag()
        row = await self.review()
        self.assertEqual(self.search.queries, [QUESTION])
        self.assertEqual(row["interaction_idx"], idx)
        self.assertEqual(row["note"], "")
        self.assertEqual(row["status"], "kept")
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertEqual(self.trainer.calls[0][2], "Canberra.")
        evidence = row["references"]
        self.assertEqual(evidence["kind"], "web")
        self.assertEqual(evidence["query"], QUESTION)
        self.assertEqual(evidence["answer"], "Canberra")
        self.assertEqual(len(evidence["source_lineages"]), 2)
        self.assertTrue(
            all(s.get("grounding_diagnostics") for s in evidence["sources"])
        )
        self.assertEqual(
            {source["url"] for source in evidence["sources"]},
            {source["url"] for source in SOURCES},
        )
        accepted = self.store.get("accepted")
        self.assertIsNotNone(accepted)
        response = await self.client.post(
            "/interact", json={"prompt": QUESTION, "use_history": False}
        )
        self.assertEqual(response.json()["response"], "Canberra")
        await self.close_wrapper()
        await self.open_wrapper()
        self.assertEqual(self.runtime.active, accepted["directory"])
        rows = (await self.client.get("/history")).json()["history"]
        self.assertEqual(rows[0]["references"], evidence)
        self.assertEqual(rows[0]["status"], "kept")
        self.assertEqual(self.search.queries, [QUESTION])
        self.assertEqual(self.trainer.calls, [])

    async def test_no_results_does_not_train(self):
        self.search.sources = []
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertTrue(row["reason"])
        self.assertEqual(row["references"]["sources"], [])
        self.assertEqual(self.trainer.calls, [])
        self.assertIsNone(self.runtime.active)

    async def test_copied_passage_on_two_sites_does_not_supply_two_votes(self):
        body = (
            "Canberra is the capital of Australia. "
            "This long shared passage also describes regional planning, municipal "
            "services, public gardens, neighboring villages, and the local transport network."
        )
        self.search.sources = [dict(s, text=body) for s in SOURCES]
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(len(row["references"]["source_lineages"]), 1)
        self.assertEqual(self.trainer.calls, [])

    async def test_shared_short_fact_with_distinct_source_context_can_corroborate(self):
        self.search.sources[0][
            "text"
        ] = "Canberra is the capital of Australia. A guide to lakes and trails follows."
        self.search.sources[1][
            "text"
        ] = "Canberra is the capital of Australia. This report covers public administration."
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "kept")
        self.assertEqual(len(row["references"]["source_lineages"]), 2)

    async def test_provider_error_is_visible_without_training_or_disabling_chat(self):
        self.search.error = ReferenceSearchError("search service unavailable")
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertIn("search service unavailable", str(row["references"]))
        self.assertEqual(self.trainer.calls, [])
        response = await self.client.post("/interact", json={"prompt": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.client.get("/status")).json()["status"], "up")

    async def test_conflicting_grounded_sources_do_not_train(self):
        self.search.sources[1]["text"] = "Sydney is the capital of Australia."
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertTrue(row["reason"])
        self.assertEqual(len(row["references"]["sources"]), 2)
        self.assertEqual(self.trainer.calls, [])

    async def test_two_pages_on_same_host_do_not_count_as_corroboration(self):
        self.search.sources[1]["url"] = "https://nation.example/another-page"
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(self.trainer.calls, [])

    async def test_subdomains_of_one_website_do_not_count_as_corroboration(self):
        self.search.sources[0]["url"] = "https://www.nation.example/capital"
        self.search.sources[1]["url"] = "https://news.nation.example/capital"
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(self.trainer.calls, [])

    async def test_model_cannot_invent_its_supporting_source_sentence(self):
        original = self.runtime.complete

        async def invented_quote(messages, **kwargs):
            if messages[-1]["content"].startswith("Reference sentences:"):
                return json.dumps(
                    dict(
                        name="Canberra",
                        sentence_index=999,
                    )
                )
            return await original(messages, **kwargs)

        self.runtime.complete = invented_quote
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(self.trainer.calls, [])

    async def test_unrelated_pages_do_not_train_even_with_two_hosts(self):
        self.search.sources[0]["text"] = "Australia is in the southern hemisphere."
        self.search.sources[1]["text"] = "Australia contains six states."
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(self.trainer.calls, [])

    async def test_status_is_responsive_and_shutdown_cancels_search(self):
        self.search.block = True
        idx = await self.flag()
        await self.client.post("/trigger_review")
        await asyncio.wait_for(self.search.started.wait(), 1)
        response = await asyncio.wait_for(self.client.get("/status"), 0.5)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["reviewing"])
        response = await asyncio.wait_for(self.client.get("/history"), 0.5)
        self.assertEqual(response.json()["history"][0]["status"], "reviewing")
        await asyncio.wait_for(self.close_wrapper(), 1)
        self.assertTrue(self.search.cancelled.is_set())
        self.assertTrue(self.runtime.client.is_closed)
        self.assertEqual(self.trainer.calls, [])
        # A restart can retry interrupted retrieval instead of losing the flag.
        store = Store(Path(self.temp.name), "automatic-reference-test")
        try:
            self.assertEqual([row["id"] for row in store.pending()], [idx])
        finally:
            store.close()

    async def test_explicit_feedback_note_takes_precedence_over_web(self):
        self.search.error = AssertionError("Explicit correction should not search")
        await self.flag(note="Canberra is the capital of Australia.")
        row = await self.review()
        self.assertEqual(row["status"], "kept")
        self.assertEqual(self.search.queries, [])
        self.assertEqual(row["references"]["kind"], "note")

    async def test_matching_optional_document_takes_precedence_over_web(self):
        await self.close_wrapper()
        await self.open_wrapper(
            documents={"Australia": "Canberra is the capital of Australia."}
        )
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "kept")
        self.assertEqual(self.search.queries, [])
        self.assertEqual(row["references"]["kind"], "documents")

    async def test_offline_mode_skips_without_documents_and_never_searches(self):
        await self.close_wrapper()
        await self.open_wrapper(web_search=False)
        await self.flag()
        row = await self.review()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(self.search.queries, [])
        self.assertEqual(self.trainer.calls, [])

    async def test_new_feedback_clears_old_reference_evidence(self):
        idx = await self.flag()
        row = await self.review()
        self.assertTrue(row["references"]["sources"])
        response = await self.client.post(
            "/feedback", json=dict(interaction_idx=idx, thumbs="up")
        )
        self.assertEqual(response.status_code, 200)
        row = (await self.client.get("/history")).json()["history"][0]
        self.assertEqual(row["references"], {})
        self.assertEqual(row["status"], "new")


class KnownAnswerJudgmentTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = FakeRuntime()
        self.controller = Controller(self.runtime, None)

    async def asyncTearDown(self):
        await self.runtime.close()

    async def test_invalid_copy_gets_exact_quote_retry_without_forcing_the_answer(
        self,
    ):
        text = "Canberra is the capital of Australia, according to the reference."
        schemas = []

        async def copy_answer(*args, **kwargs):
            schemas.append(
                copy.deepcopy(kwargs["response_format"]["json_schema"]["schema"])
            )
            quote = (
                "Canberra is the capital of Australia." if len(schemas) == 1 else text
            )
            return json.dumps(dict(quote=quote, name="Canberra"))

        self.runtime.complete = AsyncMock(side_effect=copy_answer)
        self.assertTrue(await self.controller.judge(QUESTION, text, "Canberra"))
        self.assertEqual(len(schemas), 2)
        self.assertNotIn("enum", schemas[0]["properties"]["quote"])
        for schema in schemas:
            self.assertNotIn("enum", schema["properties"]["name"])
        self.assertEqual(set(schemas[1]["properties"]["quote"]["enum"]), {text, ""})

    async def test_multiline_math_accepts_valid_short_quote_without_full_copy(self):
        quote = r"\[ 12 \times 12 = 144 \]"
        text = "Multiply the two numbers:\n\n" + quote + "\n\nThe product is 144."
        self.runtime.complete = AsyncMock(
            return_value=json.dumps(dict(quote=quote, name="144"))
        )
        self.assertTrue(
            await self.controller.judge("What is 12 times 12?", text, "144")
        )
        self.runtime.complete.assert_awaited_once()
        properties = self.runtime.complete.call_args.kwargs["response_format"][
            "json_schema"
        ]["schema"]["properties"]
        self.assertNotIn("enum", properties["quote"])

    async def test_invalid_structured_judgments_fail_closed(self):
        text = "Canberra is the capital of Australia, according to the reference."
        outputs = [
            "not JSON",
            json.dumps(dict(quote=text, name="Australia")),
            json.dumps(
                dict(quote="Canberra is the capital of Australia.", name="Canberra")
            ),
            json.dumps(dict(quote=text, name="")),
            json.dumps(dict(quote="", name="Canberra")),
        ]
        for output in outputs:
            with self.subTest(output=output):
                self.runtime.complete = AsyncMock(return_value=output)
                self.assertFalse(
                    await self.controller.judge(QUESTION, text, "Canberra")
                )

    async def test_negative_or_uncertain_model_decision_is_not_overridden_by_mention(
        self,
    ):
        for text in (
            "Sydney, not Canberra, is the capital of Australia.",
            "Canberra might be Australia's capital, but I am uncertain.",
            "Canberra is a city; Sydney is Australia's capital.",
        ):
            with self.subTest(text=text):
                self.runtime.complete = AsyncMock(
                    return_value=json.dumps(dict(quote="", name=""))
                )
                self.assertFalse(
                    await self.controller.judge(QUESTION, text, "Canberra")
                )
                self.assertGreater(self.runtime.complete.await_count, 0)

    async def test_valid_wrong_answer_is_not_resampled_until_it_agrees(self):
        text = "Sydney, not Canberra, is the capital of Australia."
        self.runtime.complete = AsyncMock(
            side_effect=[
                json.dumps(dict(quote=text, name="Sydney")),
                json.dumps(dict(quote=text, name="Canberra")),
            ]
        )
        self.assertFalse(await self.controller.judge(QUESTION, text, "Canberra"))
        self.runtime.complete.assert_awaited_once()

    async def test_explicit_abstention_is_not_resampled_until_it_agrees(self):
        text = "Canberra might be Australia's capital, but I am uncertain."
        self.runtime.complete = AsyncMock(
            side_effect=[
                json.dumps(dict(quote="", name="")),
                json.dumps(dict(quote=text, name="Canberra")),
            ]
        )
        self.assertFalse(await self.controller.judge(QUESTION, text, "Canberra"))
        self.runtime.complete.assert_awaited_once()


class ReferenceHistoryMigrationTest(unittest.TestCase):
    def test_existing_history_migrates_without_losing_feedback_or_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            database = sqlite3.connect(path / "history.sqlite3")
            database.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE interactions (
                    id INTEGER PRIMARY KEY, messages TEXT NOT NULL,
                    response TEXT NOT NULL, created REAL NOT NULL,
                    flagged INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'new',
                    reason TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT ''
                );
                """)
            accepted = {"directory": str(path / "old-adapter")}
            database.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                [
                    ("identity", json.dumps("old-model")),
                    ("accepted", json.dumps(accepted)),
                ],
            )
            messages = [dict(role="user", content=QUESTION)]
            database.execute(
                "INSERT INTO interactions VALUES (1, ?, 'Sydney', 1, 1, 'pending', '', '')",
                (json.dumps(messages),),
            )
            database.commit()
            database.close()
            store = Store(path, "old-model")
            try:
                row = store.pending()[0]
                self.assertEqual(row["messages"], messages)
                self.assertEqual(row["response"], "Sydney")
                self.assertEqual(row["references"], {})
                self.assertEqual(store.get("accepted"), accepted)
                store.references(1, {"kind": "web", "sources": SOURCES})
            finally:
                store.close()
            store = Store(path, "old-model")
            try:
                self.assertEqual(store.rows()[0]["references"]["sources"], SOURCES)
                self.assertEqual(store.get("accepted"), accepted)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
