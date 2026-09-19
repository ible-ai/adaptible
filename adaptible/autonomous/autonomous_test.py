"""Model-free tests for the autonomous node's claim filtering and training policy."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptible.autonomous import node as node_module
from adaptible.autonomous.node import (
    AutonomousNode,
    Claim,
    LearningEvent,
    NodeState,
    _claim_is_plausible,
)
from adaptible.db import Database


def _claim(question: str, answer: str) -> Claim:
    return Claim(question=question, answer=answer, source="src", url="https://x.test/")


class ClaimIsPlausibleTest(unittest.TestCase):
    SNIPPET = (
        "Apple Inc. reported revenue of $94.9 billion for Q4 2024, up 6% year over "
        "year, the company said on Thursday."
    )

    def test_accepts_grounded_question(self):
        claim = _claim(
            "What was Apple's Q4 2024 revenue?",
            "Apple reported revenue of $94.9 billion for Q4 2024.",
        )
        self.assertTrue(_claim_is_plausible(claim, self.SNIPPET))

    def test_rejects_non_question(self):
        claim = _claim(
            "NBCNews.com provides the latest top news stories.",
            "NBCNews.com offers breaking headlines and video updates.",
        )
        self.assertFalse(_claim_is_plausible(claim, "NBCNews.com provides the latest top news stories and video."))

    def test_rejects_short_question(self):
        claim = _claim("Revenue?", "Apple reported revenue of $94.9 billion.")
        self.assertFalse(_claim_is_plausible(claim, self.SNIPPET))

    def test_rejects_boilerplate_about_the_page(self):
        claim = _claim(
            "What is the main focus of the content?",
            "The main focus is to provide top news stories from The Associated Press.",
        )
        snippet = "Top News: US & International Top News Stories Today. The Associated Press."
        self.assertFalse(_claim_is_plausible(claim, snippet))

    def test_rejects_site_name_in_answer(self):
        claim = _claim(
            "Where can you read the latest headlines today?",
            "You can read them on apnews.com.",
        )
        self.assertFalse(_claim_is_plausible(claim, "Read headlines on apnews.com today."))

    def test_rejects_ungrounded_answer(self):
        claim = _claim(
            "Who won the 2025 Super Bowl championship?",
            "The Philadelphia Eagles won the championship.",
        )
        # Snippet shares no content word with the answer.
        self.assertFalse(_claim_is_plausible(claim, self.SNIPPET))

    def test_grounding_ignores_stopwords_and_short_tokens(self):
        claim = _claim(
            "What did they say about it in the end?",
            "That they were there with them.",
        )
        # Every token is a stopword or too short -> no content tokens -> reject.
        self.assertFalse(_claim_is_plausible(claim, "That they were there with them."))

    def test_rejects_empty_answer(self):
        claim = _claim("What was Apple's Q4 2024 revenue?", "   ")
        self.assertFalse(_claim_is_plausible(claim, self.SNIPPET))


class _NoModelNode(AutonomousNode):
    """AutonomousNode whose model calls are scripted, never loading MLX weights."""

    def __init__(self, responses, **kwargs):
        self._scripted = list(responses)
        self.trained: list[tuple[str, str, str]] = []
        super().__init__(**kwargs)

    def _ask(self, prompt, max_tokens=None):
        return self._scripted.pop(0) if self._scripted else ""

    def _train_on_correction(self, question, before, correct):
        self.trained.append((question, before, correct))
        return True

    @property
    def model(self):
        tok = mock.MagicMock()
        tok.encode.side_effect = lambda text: list(text)
        return mock.MagicMock(_tokenizer=tok)


class ExtractClaimsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.results = [
            {
                "title": "Apple Reports Q4 2024 Earnings",
                "snippet": ClaimIsPlausibleTest.SNIPPET,
                "url": "https://example.test/apple",
            }
        ]

    def _node(self, responses):
        return _NoModelNode(
            responses,
            search_fn=lambda q: self.results,
            seed_topics=["t"],
            state_path=self.tmp / "state.json",
            log_dir=self.tmp / "logs",
            db=Database(self.tmp / "db.sqlite"),
        )

    def test_parses_q_and_a(self):
        node = self._node(
            ["Q: What was Apple's Q4 2024 revenue?\nA: Apple reported revenue of $94.9 billion."]
        )
        claims = node._extract_claims(self.results, "t")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].question, "What was Apple's Q4 2024 revenue?")

    def test_drops_response_without_q_and_a_lines(self):
        node = self._node(["Apple made $94.9 billion in revenue in Q4 2024."])
        self.assertEqual(node._extract_claims(self.results, "t"), [])

    def test_drops_answer_only(self):
        node = self._node(["A: Apple reported revenue of $94.9 billion."])
        self.assertEqual(node._extract_claims(self.results, "t"), [])

    def test_drops_no_claim(self):
        node = self._node(["NO CLAIM"])
        self.assertEqual(node._extract_claims(self.results, "t"), [])

    def test_drops_implausible_claim_and_logs_it(self):
        node = self._node(
            ["Q: What is the main focus of the content?\nA: Apple revenue coverage."]
        )
        self.assertEqual(node._extract_claims(self.results, "t"), [])
        logs = list((self.tmp / "logs").glob("*.txt"))
        self.assertEqual(len(logs), 1)
        self.assertIn("DROPPED", logs[0].read_text())


class TrainingPolicyTest(unittest.TestCase):
    """Only conflicting beliefs are trained on unless train_on_new_knowledge."""

    EXTRACT = "Q: What was Apple's Q4 2024 revenue?\nA: Apple reported revenue of $94.9 billion for Q4 2024."

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.results = [
            {
                "title": "Apple Reports Q4 2024 Earnings",
                "snippet": ClaimIsPlausibleTest.SNIPPET,
                "url": "https://example.test/apple",
            }
        ]

    def _node(self, responses, **kwargs):
        return _NoModelNode(
            responses,
            search_fn=lambda q: self.results,
            seed_topics=["t"],
            state_path=self.tmp / "state.json",
            log_dir=self.tmp / "logs",
            db=Database(self.tmp / "db.sqlite"),
            **kwargs,
        )

    def test_knowledge_gap_recorded_but_not_trained_by_default(self):
        node = self._node(
            [
                self.EXTRACT,
                "RESPONSE: I don't know.\nCONFIDENCE: LOW",  # belief -> gap
            ]
        )
        result = node.explore_once("t")
        self.assertEqual(result.claims_found, 1)
        self.assertEqual(node.trained, [])
        self.assertEqual(result.updates_made, 0)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_type, "new")
        self.assertFalse(result.events[0].trained)
        self.assertIsNone(result.events[0].after_training_answer)
        # Persisted to state.json at the configured path.
        saved = json.loads((self.tmp / "state.json").read_text())
        self.assertEqual(len(saved["learning_history"]), 1)

    def test_knowledge_gap_trained_when_flag_on(self):
        node = self._node(
            [
                self.EXTRACT,
                "RESPONSE: I don't know.\nCONFIDENCE: LOW",
                "RESPONSE: $94.9 billion.\nCONFIDENCE: HIGH",  # post-training belief
            ],
            train_on_new_knowledge=True,
        )
        result = node.explore_once("t")
        self.assertEqual(len(node.trained), 1)
        self.assertEqual(result.updates_made, 1)
        self.assertEqual(result.events[0].event_type, "new")
        self.assertTrue(result.events[0].trained)

    def test_conflicting_belief_is_trained(self):
        node = self._node(
            [
                self.EXTRACT,
                "RESPONSE: Apple made $50 billion.\nCONFIDENCE: HIGH",  # belief
                "NEEDS CORRECTION: YES",  # fact-checker
                "RESPONSE: $94.9 billion.\nCONFIDENCE: HIGH",  # post-training
            ]
        )
        result = node.explore_once("t")
        self.assertEqual(len(node.trained), 1)
        self.assertEqual(node.trained[0][1], "Apple made $50 billion.")
        self.assertEqual(result.events[0].event_type, "correction")
        self.assertTrue(result.events[0].trained)
        self.assertEqual(result.updates_made, 1)

    def test_consistent_belief_is_not_trained_or_recorded(self):
        node = self._node(
            [
                self.EXTRACT,
                "RESPONSE: Apple made $94.9 billion.\nCONFIDENCE: HIGH",
                "NEEDS CORRECTION: NO",
            ]
        )
        result = node.explore_once("t")
        self.assertEqual(node.trained, [])
        self.assertEqual(result.events, [])
        self.assertEqual(result.updates_made, 0)


class DefaultPathsTest(unittest.TestCase):
    def test_defaults_follow_outputs_dir(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"ADAPTIBLE_OUTPUTS_DIR": tmp}
        ):
            node = _NoModelNode(
                [], search_fn=lambda q: [], seed_topics=["t"], db=Database(Path(tmp) / "x.db")
            )
            self.assertEqual(node.state_path, Path(tmp) / "autonomous" / "state.json")
            self.assertEqual(node.log_dir, Path(tmp) / "autonomous" / "logs")
            self.assertEqual(node._model_path, Path(tmp) / "autonomous" / "checkpoint")
            self.assertFalse(node.train_on_new_knowledge)


class NodeStateLegacyLoadTest(unittest.TestCase):
    def test_loads_legacy_old_new_answer_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "learning_history": [
                            {
                                "timestamp": "2025-12-08T11:50:16",
                                "question": "How many leaders?",
                                "old_answer": None,
                                "new_answer": "15",
                                "source": "src",
                                "source_url": "https://x",
                                "event_type": "new",
                            }
                        ],
                        "topics_explored": ["t"],
                        "total_updates": 1,
                        "total_searches": 1,
                        "started_at": "2025-12-08",
                    }
                )
            )
            state = NodeState.load(path)
        self.assertEqual(len(state.learning_history), 1)
        event = state.learning_history[0]
        self.assertIsInstance(event, LearningEvent)
        self.assertEqual(event.after_training_answer, "15")
        self.assertIsNone(event.before_training_answer)

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = NodeState(started_at="now")
            state.learning_history.append(
                LearningEvent(
                    timestamp="t", question="q?", before_training_answer="a",
                    after_training_answer=None, source="s", source_url=None,
                    event_type="new", verified_answer="v",
                    confidence_before_training=0.0, confidence_after_training=0.0,
                    trained=False,
                )
            )
            state.save(path)
            loaded = NodeState.load(path)
        self.assertEqual(loaded.learning_history, state.learning_history)

    def test_shipped_state_file_loads(self):
        shipped = Path(__file__).resolve().parents[2] / "outputs" / "autonomous" / "state.json"
        if not shipped.exists():
            self.skipTest("no shipped state.json")
        state = NodeState.load(shipped)
        self.assertGreater(len(state.learning_history), 0)


if __name__ == "__main__":
    unittest.main()
