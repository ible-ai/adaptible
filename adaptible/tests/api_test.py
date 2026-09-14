"""Tests for the FastAPI endpoints in _api.py.

These tests use a stub model to test API logic in isolation.
"""

from typing import List

import unittest
from fastapi.testclient import TestClient

import adaptible


class StubModel:
    """Minimal stub that satisfies the model interface for API testing."""

    def __init__(self, train_error: Exception | None = None):
        self.ok = True
        self._call_count = 0
        # Every call to self_correct_and_train, as (interaction_history, indices).
        self.self_correct_calls: List[
            tuple[List[adaptible.InteractionHistory], List[int] | None]
        ] = []
        self._train_error = train_error

    def generate_response(self, prompt: str, use_history: bool = True) -> str:
        self._call_count += 1
        self.use_history_calls = getattr(self, "use_history_calls", []) + [use_history]
        return f"Response to: {prompt}"

    def self_correct_and_train(
        self,
        interaction_history: List[adaptible.InteractionHistory],
        indices_to_review: List[int] | None = None,
        verbose: bool = False,
    ) -> bool:
        del verbose
        self.self_correct_calls.append((list(interaction_history), indices_to_review))
        if self._train_error is not None:
            raise self._train_error
        return True

    def reset_conversation(self) -> None:
        self.resets = getattr(self, "resets", 0) + 1

    async def stream_response(self, prompt: str, use_history: bool = True):
        for chunk in ("Streamed: ", prompt):
            yield chunk


class InteractEndpointTest(unittest.TestCase):
    """Tests for the /interact endpoint."""

    def setUp(self):
        self.stub_model = StubModel()
        self.api = adaptible.Adaptible(model=self.stub_model)
        self.client = TestClient(self.api.app)

    def test_interact_returns_response(self):
        """POST /interact should return model response."""
        response = self.client.post("/interact", json={"prompt": "Hello"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "Response to: Hello")
        self.assertEqual(data["interaction_idx"], 0)

    def test_interact_use_history_flag_is_passed(self):
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/interact", json={"prompt": "Q2", "use_history": False})
        self.assertEqual(self.stub_model.use_history_calls, [True, False])

    def test_interact_empty_prompt_returns_400(self):
        """POST /interact with empty prompt should return 400."""
        response = self.client.post("/interact", json={"prompt": ""})

        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_interact_stores_history(self):
        """POST /interact should store interaction in history."""
        self.client.post("/interact", json={"prompt": "First question"})
        self.client.post("/interact", json={"prompt": "Second question"})

        self.assertEqual(len(self.api.interaction_history), 2)
        self.assertEqual(self.api.interaction_history[0].user_input, "First question")
        self.assertEqual(
            self.api.interaction_history[0].llm_response, "Response to: First question"
        )
        self.assertEqual(self.api.interaction_history[1].user_input, "Second question")

    def test_interact_increments_index(self):
        """Each interaction should have incrementing indices."""
        response1 = self.client.post("/interact", json={"prompt": "Q1"})
        response2 = self.client.post("/interact", json={"prompt": "Q2"})
        response3 = self.client.post("/interact", json={"prompt": "Q3"})

        self.assertEqual(response1.json()["interaction_idx"], 0)
        self.assertEqual(response2.json()["interaction_idx"], 1)
        self.assertEqual(response3.json()["interaction_idx"], 2)

    def test_interact_adds_to_unreviewed(self):
        """Interactions should be added to unreviewed list."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/interact", json={"prompt": "Q2"})

        self.assertEqual(len(self.api.unreviewed_interaction_history_indices), 2)
        self.assertIn(0, self.api.unreviewed_interaction_history_indices)
        self.assertIn(1, self.api.unreviewed_interaction_history_indices)

    def test_interact_calls_model(self):
        """POST /interact should call the model's generate_response."""
        self.client.post("/interact", json={"prompt": "Test"})

        self.assertEqual(self.stub_model._call_count, 1)


class TriggerReviewEndpointTest(unittest.TestCase):
    """Tests for the /trigger_review endpoint."""

    def setUp(self):
        self.stub_model = StubModel()
        self.api = adaptible.Adaptible(model=self.stub_model)
        self.client = TestClient(self.api.app)

    def test_trigger_review_no_unreviewed(self):
        """POST /trigger_review with no unreviewed should return message."""
        response = self.client.post("/trigger_review")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["unreviewed_count"], 0)
        self.assertIn("No unreviewed", data["message"])

    def test_trigger_review_with_unreviewed(self):
        """POST /trigger_review with unreviewed should initiate review."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/interact", json={"prompt": "Q2"})

        response = self.client.post("/trigger_review")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["unreviewed_count"], 2)
        self.assertIn("initiated", data["message"].lower())

    def test_trigger_review_runs_training_on_sync(self):
        """POST /trigger_review + GET /sync should run self_correct_and_train once."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/interact", json={"prompt": "Q2"})

        self.client.post("/trigger_review")
        self.assertEqual(len(self.api.outstanding_tasks), 1)
        response = self.client.get("/sync")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.stub_model.self_correct_calls), 1)
        history, indices = self.stub_model.self_correct_calls[0]
        self.assertIsNone(indices)
        self.assertEqual([h.user_input for h in history], ["Q1", "Q2"])
        self.assertEqual(
            [h.llm_response for h in history],
            ["Response to: Q1", "Response to: Q2"],
        )
        # /sync drains the queue.
        self.assertEqual(len(self.api.outstanding_tasks), 0)

    def test_trigger_review_clears_unreviewed_indices(self):
        """Dispatched interactions must not be reviewed again by a second trigger."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/interact", json={"prompt": "Q2"})

        self.client.post("/trigger_review")
        self.assertEqual(self.api.unreviewed_interaction_history_indices, [])

        # Nothing new: the second trigger must be a no-op.
        response = self.client.post("/trigger_review")
        self.assertEqual(response.json()["unreviewed_count"], 0)
        self.client.get("/sync")
        self.assertEqual(len(self.stub_model.self_correct_calls), 1)

        # A new interaction is reviewed on its own, without the earlier ones.
        self.client.post("/interact", json={"prompt": "Q3"})
        self.assertEqual(self.api.unreviewed_interaction_history_indices, [2])
        response = self.client.post("/trigger_review")
        self.assertEqual(response.json()["unreviewed_count"], 1)
        self.client.get("/sync")
        self.assertEqual(len(self.stub_model.self_correct_calls), 2)
        history, _ = self.stub_model.self_correct_calls[1]
        self.assertEqual([h.user_input for h in history], ["Q3"])

    def test_training_exception_does_not_crash_sync(self):
        """A failing training task is logged and /sync still returns 200."""
        failing_model = StubModel(train_error=RuntimeError("backprop exploded"))
        api = adaptible.Adaptible(model=failing_model)
        client = TestClient(api.app)
        client.post("/interact", json={"prompt": "Q1"})
        client.post("/trigger_review")

        with self.assertLogs("asyncio", level="ERROR") as logs:
            response = client.get("/sync")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tasks_count"], 1)
        self.assertEqual(len(failing_model.self_correct_calls), 1)
        self.assertTrue(any("backprop exploded" in line for line in logs.output))
        self.assertEqual(len(api.outstanding_tasks), 0)


class FeedbackEndpointTest(unittest.TestCase):
    """Tests for /feedback and the stream path's history record."""

    def setUp(self):
        self.stub_model = StubModel()
        self.api = adaptible.Adaptible(model=self.stub_model)
        self.client = TestClient(self.api.app)

    def test_thumbs_down_flags_interaction(self):
        self.client.post("/interact", json={"prompt": "Q1"})
        response = self.client.post("/feedback", json={"interaction_idx": 0, "thumbs": "down"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"interaction_idx": 0, "flagged": True})
        self.assertTrue(self.api.interaction_history[0].flagged)

    def test_thumbs_up_clears_flag(self):
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/feedback", json={"interaction_idx": 0, "thumbs": "down"})
        response = self.client.post("/feedback", json={"interaction_idx": 0, "thumbs": "up"})
        self.assertFalse(response.json()["flagged"])
        self.assertFalse(self.api.interaction_history[0].flagged)

    def test_flag_after_review_requeues_interaction(self):
        """A thumbs-down on an already-reviewed answer puts it back in the review queue."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/trigger_review")
        self.client.get("/sync")
        self.assertEqual(self.api.unreviewed_interaction_history_indices, [])
        self.client.post("/feedback", json={"interaction_idx": 0, "thumbs": "down"})
        self.assertEqual(self.api.unreviewed_interaction_history_indices, [0])
        self.client.post("/trigger_review")
        self.client.get("/sync")
        history, _ = self.stub_model.self_correct_calls[1]
        self.assertTrue(history[0].flagged)

    def test_new_chat_resets_conversation_and_keeps_history(self):
        self.client.post("/interact", json={"prompt": "Q1"})
        response = self.client.post("/new_chat")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub_model.resets, 1)
        self.assertEqual(len(self.api.interaction_history), 1)

    def test_unknown_index_is_404(self):
        response = self.client.post("/feedback", json={"interaction_idx": 3, "thumbs": "down"})
        self.assertEqual(response.status_code, 404)

    def test_bad_thumbs_is_400(self):
        self.client.post("/interact", json={"prompt": "Q1"})
        response = self.client.post("/feedback", json={"interaction_idx": 0, "thumbs": "sideways"})
        self.assertEqual(response.status_code, 400)

    def test_stream_interact_records_history_with_index_header(self):
        response = self.client.post("/stream_interact", json={"prompt": "Q1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Interaction-Idx"], "0")
        self.assertEqual(response.text, "Streamed: Q1")
        self.assertEqual(len(self.api.interaction_history), 1)
        self.assertEqual(self.api.interaction_history[0].llm_response, "Streamed: Q1")
        self.assertEqual(self.api.unreviewed_interaction_history_indices, [0])
        # The streamed answer can be rated like any other.
        response = self.client.post("/feedback", json={"interaction_idx": 0, "thumbs": "down"})
        self.assertTrue(response.json()["flagged"])


class HistoryEndpointTest(unittest.TestCase):
    """Tests for the /history endpoint."""

    def setUp(self):
        self.stub_model = StubModel()
        self.api = adaptible.Adaptible(model=self.stub_model)
        self.client = TestClient(self.api.app)

    def test_history_empty(self):
        """GET /history with no interactions should return empty list."""
        response = self.client.get("/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["history"], [])

    def test_history_returns_interactions(self):
        """GET /history should return all interactions."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/interact", json={"prompt": "Q2"})

        response = self.client.get("/history")

        self.assertEqual(response.status_code, 200)
        history = response.json()["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["user_input"], "Q1")
        self.assertEqual(history[0]["llm_response"], "Response to: Q1")
        self.assertEqual(history[1]["user_input"], "Q2")


class StatusEndpointTest(unittest.TestCase):
    """Tests for the /status endpoint."""

    def setUp(self):
        self.stub_model = StubModel()
        self.api = adaptible.Adaptible(model=self.stub_model)
        self.client = TestClient(self.api.app)

    def test_status_returns_up(self):
        """GET /status should return status up."""
        response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "up")


class SyncEndpointTest(unittest.TestCase):
    """Tests for the /sync endpoint."""

    def setUp(self):
        self.stub_model = StubModel()
        self.api = adaptible.Adaptible(model=self.stub_model)
        self.client = TestClient(self.api.app)

    def test_sync_when_model_ok(self):
        """GET /sync should return immediately when model is ok."""
        response = self.client.get("/sync")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("tasks_count", data)
        self.assertIn("elapsed_time", data)

    def test_sync_reports_task_count(self):
        """GET /sync should report number of outstanding tasks."""
        self.client.post("/interact", json={"prompt": "Q1"})
        self.client.post("/trigger_review")

        response = self.client.get("/sync")

        self.assertEqual(response.json()["tasks_count"], 1)


if __name__ == "__main__":
    unittest.main()
