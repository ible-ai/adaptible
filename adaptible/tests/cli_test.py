"""Model-free tests for the terminal client in adaptible.cli."""

import io
import unittest

from fastapi.testclient import TestClient

import adaptible
from adaptible.tests.api_test import StubModel


class CliClientTest(unittest.TestCase):
    def setUp(self):
        self.model = StubModel()
        self.api = adaptible.Adaptible(model=self.model)
        self.out = io.StringIO()
        self.cli = adaptible.cli.Client(http=TestClient(self.api.app), out=self.out)

    def test_ask_streams_and_remembers_idx(self):
        self.assertTrue(self.cli.run("Hello"))
        self.assertEqual(self.out.getvalue(), "Streamed: Hello\n")
        self.assertEqual(self.cli.last_idx, 0)
        self.cli.run("Again")
        self.assertEqual(self.cli.last_idx, 1)
        self.assertEqual(self.api.interaction_history[1].llm_response, "Streamed: Again")

    def test_down_flags_last_answer(self):
        self.cli.run("Hello")
        self.cli.run("/down")
        self.assertTrue(self.api.interaction_history[0].flagged)
        self.assertIn("[down] answer 0 flagged", self.out.getvalue())
        self.cli.run("/up")
        self.assertFalse(self.api.interaction_history[0].flagged)

    def test_rate_before_ask(self):
        self.cli.run("/down")
        self.assertIn("nothing to rate", self.out.getvalue())

    def test_review_trains_and_syncs(self):
        self.cli.run("Hello")
        self.cli.run("/down")
        self.cli.run("/review")
        self.assertEqual(len(self.model.self_correct_calls), 1)
        self.assertRegex(self.out.getvalue(), r"\[review\] started on 1 answer\(s\).*\n\[review\] done in \d+ s")
        self.out.truncate(0)
        self.out.seek(0)
        self.cli.run("/review")
        self.assertIn("nothing to review", self.out.getvalue())

    def test_new_resets_conversation(self):
        self.cli.run("Hello")
        self.cli.run("/new")
        self.assertEqual(self.model.resets, 1)
        self.assertIsNone(self.cli.last_idx)

    def test_quit_and_unknown(self):
        self.assertFalse(self.cli.run("/quit"))
        self.assertTrue(self.cli.run("/bogus"))
        self.assertIn("unknown command /bogus", self.out.getvalue())


class CliColorTest(unittest.TestCase):
    """Colour is ANSI only on a TTY; StringIO output stays plain."""

    def setUp(self):
        self.api = adaptible.Adaptible(model=StubModel())

    def test_no_escapes_when_not_a_tty(self):
        out = io.StringIO()
        cli = adaptible.cli.Client(http=TestClient(self.api.app), out=out)
        for line in ("Hello", "/down", "/up", "/review", "/new", "/bogus"):
            cli.run(line)
        self.assertFalse(cli.color)
        self.assertNotIn("\033[", out.getvalue())

    def test_escapes_when_color_on(self):
        out = io.StringIO()
        cli = adaptible.cli.Client(http=TestClient(self.api.app), out=out, color=True)
        cli.run("Hello")
        cli.run("/down")
        cli.run("/new")
        text = out.getvalue()
        self.assertIn("\033[31m[down] answer 0 flagged for review\033[0m", text)
        self.assertIn("\033[36m[new chat]\033[0m", text)
        # The stub streams no </think>, so the whole reply is dim reasoning.
        self.assertRegex(text, r"^\033\[2mStreamed: (\033\[0m\033\[2m)?Hello\033\[0m\n")

    def test_think_tag_split_across_chunks(self):
        class SplitModel(StubModel):
            async def stream_response(self, prompt, use_history=True):
                for chunk in ("Thinking</thi", "nk>\nAnswer."):
                    yield chunk

        api = adaptible.Adaptible(model=SplitModel())
        out = io.StringIO()
        cli = adaptible.cli.Client(http=TestClient(api.app), out=out, color=True)
        cli.run("Q")
        text = out.getvalue()
        self.assertRegex(text, r"\033\[2m[^\033]*</think>\033\[0m\nAnswer\.\n$")  # tag dim, answer plain
        self.assertEqual(text.replace("\033[2m", "").replace("\033[0m", ""), "Thinking</think>\nAnswer.\n")
        plain = io.StringIO()
        adaptible.cli.Client(http=TestClient(adaptible.Adaptible(model=SplitModel()).app), out=plain).run("Q")
        self.assertEqual(plain.getvalue(), "Thinking</think>\nAnswer.\n")


class CliMainTest(unittest.TestCase):
    def test_no_server(self):
        self.assertEqual(adaptible.cli.main(["--url", "http://127.0.0.1:1"]), 1)


if __name__ == "__main__":
    unittest.main()
