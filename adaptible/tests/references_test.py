"""Model-free evidence lookup, source hygiene, and worker lifecycle tests."""

import asyncio
import json
import sys
import unittest
from unittest import mock

from adaptible._src.wrap.references import (
    ReferenceSearchError,
    SEARCH_BACKENDS,
    WebReferences,
    _public_url,
    _sources,
    _search_sync,
)


class ReferenceSourcesTest(unittest.TestCase):
    def test_public_sources_are_clean_bounded_and_diverse(self):
        rows = [
            {
                "title": "<b>City</b>",
                "href": "https://www.example.com/city#section",
                "body": "<p>Casablanca is Morocco's largest city.</p><script>evil()</script>",
            },
            {
                "title": "Duplicate",
                "href": "http://example.com/city/",
                "body": "Duplicate content should not be retained.",
            },
            {
                "title": "Second",
                "href": "https://example.com/other",
                "body": "Another useful passage from the same domain.",
            },
            {
                "title": "Third",
                "href": "https://example.com/third",
                "body": "Too many passages from the same domain.",
            },
            {
                "title": "T" * 1000,
                "href": "https://second.example.org/",
                "body": "a" * 10_000,
            },
            *[
                {
                    "title": str(i),
                    "href": f"https://source{i}.org/",
                    "body": "A useful public factual search snippet.",
                }
                for i in range(10)
            ],
        ]
        result = _sources(rows, 5)
        self.assertEqual(len(result), 5)
        self.assertEqual(
            result[0],
            {
                "title": "City",
                "url": "https://www.example.com/city",
                "text": "Casablanca is Morocco's largest city.",
            },
        )
        self.assertEqual(result[1]["title"], "Second")
        self.assertEqual(len(result[2]["title"]), 200)
        self.assertEqual(len(result[2]["text"]), 1800)

    def test_private_and_malformed_links_are_rejected(self):
        for url in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "http://localhost/x",
            "http://localhost./",
            "http://service.local/x",
            "http://service.internal/x",
            "http://192.168.1.5/",
            "http://10.0.0.1/",
            "http://127.1/",
            "http://2130706433/",
            "http://0177.0.0.1/",
            "http://0x7f000001/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://169.254.169.254/latest/",
            "https://user:secret@example.com/",
            "https://example.com:bad/",
            "https://example.com\\@localhost/",
            "https://example.com/\nfoo",
            "https://bad%2eexample.com/",
            "not a url",
            None,
        ]:
            with self.subTest(url=url):
                self.assertIsNone(_public_url(url))
        self.assertEqual(_public_url("https://8.8.8.8/"), "https://8.8.8.8/")
        self.assertEqual(
            _public_url("HTTPS://EXAMPLE.COM:443/path#anchor"),
            "https://example.com/path",
        )

    def test_invalid_and_empty_snippets_are_ignored(self):
        self.assertEqual(
            _sources(
                [
                    None,
                    {},
                    {"href": "https://example.com", "body": ""},
                    {"href": "https://example.com", "body": ["not text"]},
                ],
                5,
            ),
            [],
        )
        with self.assertRaises(ValueError):
            _sources({"unexpected": "object"}, 5)


class WebReferencesTest(unittest.IsolatedAsyncioTestCase):
    async def _with_worker(self, script, callback):
        original = asyncio.create_subprocess_exec
        created = []
        requests = []

        async def spawn(*args, **kwargs):
            process = await original(sys.executable, "-c", script, **kwargs)
            created.append(process)
            communicate = process.communicate

            async def capture(data):
                requests.append(json.loads(data))
                return await communicate(data)

            process.communicate = capture
            return process

        with mock.patch(
            "adaptible._src.wrap.references.asyncio.create_subprocess_exec",
            side_effect=spawn,
        ):
            await callback(created, requests)
        return created, requests

    async def test_search_returns_results_and_only_sends_bounded_question(self):
        async def check(created, requests):
            result = await WebReferences().search(" Which  city? " + "q" * 1000)
            self.assertEqual(
                result[0]["text"], "Casablanca is the largest city in Morocco."
            )
            self.assertEqual(set(requests[0]), {"question", "max_results", "timeout"})
            self.assertEqual(len(requests[0]["question"]), 600)
            self.assertTrue(requests[0]["question"].startswith("Which city?"))
            self.assertIsNotNone(created[0].returncode)

        await self._with_worker(
            "import json,sys;json.load(sys.stdin);print(json.dumps({'results':[{'url':'https://example.org/city','title':'City','text':'Casablanca is the largest city in Morocco.'}]}))",
            check,
        )

    async def test_no_results_is_not_provider_failure(self):
        async def check(created, requests):
            self.assertEqual(await WebReferences().search("a factual question"), [])

        await self._with_worker("print('{\"results\": []}')", check)
        with mock.patch(
            "adaptible._src.wrap.references.asyncio.create_subprocess_exec"
        ) as spawn:
            self.assertEqual(await WebReferences().search("  "), [])
            spawn.assert_not_called()

    async def test_provider_errors_and_bad_protocol_are_reported(self):
        for script in [
            'print(\'{"error": "Provider unavailable"}\')',
            "print('not JSON')",
            "print('{\"results\": {}}')",
            "print('x'*66000)",
            "raise SystemExit(3)",
        ]:

            async def check(created, requests):
                with self.assertRaises(ReferenceSearchError):
                    await WebReferences().search("a factual question")
                self.assertIsNotNone(created[0].returncode)

            await self._with_worker(script, check)

    async def test_timeout_kills_and_reaps_worker(self):
        async def check(created, requests):
            with self.assertRaisesRegex(ReferenceSearchError, "timed out"):
                await WebReferences(timeout=0.1).search("a factual question")
            self.assertIsNotNone(created[0].returncode)
            self.assertNotEqual(created[0].returncode, 0)

        await self._with_worker("import time;time.sleep(60)", check)

    async def test_cancellation_kills_and_reaps_worker(self):
        async def check(created, requests):
            task = asyncio.create_task(WebReferences().search("a factual question"))
            while not requests:
                await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNotNone(created[0].returncode)
            self.assertNotEqual(created[0].returncode, 0)

        await self._with_worker("import time;time.sleep(60)", check)

    async def test_cancellation_during_spawn_still_reaps_worker(self):
        original = asyncio.create_subprocess_exec
        ready, release = asyncio.Event(), asyncio.Event()
        created = []

        async def spawn(*args, **kwargs):
            process = await original(
                sys.executable, "-c", "import time;time.sleep(60)", **kwargs
            )
            created.append(process)
            ready.set()
            await release.wait()
            return process

        with mock.patch(
            "adaptible._src.wrap.references.asyncio.create_subprocess_exec",
            side_effect=spawn,
        ):
            task = asyncio.create_task(WebReferences().search("a factual question"))
            await ready.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsNotNone(created[0].returncode)

    def test_provider_no_result_exception_is_distinct_from_provider_error(self):
        from ddgs.exceptions import DDGSException

        with mock.patch("ddgs.DDGS") as provider:
            search = provider.return_value.__enter__.return_value.text
            search.side_effect = DDGSException("No results found.")
            self.assertEqual(_search_sync("city question", 5, 20), [])
            search.assert_called_once_with(
                "city question", max_results=15, backend=",".join(SEARCH_BACKENDS)
            )
            search.side_effect = DDGSException("provider connection failed")
            with self.assertRaises(DDGSException):
                _search_sync("city question", 5, 20)

    def test_invalid_limits(self):
        for kwargs in [
            {"timeout": 0},
            {"timeout": float("inf")},
            {"max_results": 0},
            {"max_results": 6},
        ]:
            with self.assertRaises(ValueError):
                WebReferences(**kwargs)


if __name__ == "__main__":
    unittest.main()
