"""Tests for the FastAPI endpoints in _api.py.

These tests use a stub model to test API logic in isolation.
"""

from typing import List

import unittest
from fastapi.testclient import TestClient

import adaptible


class StubModel:
    """Minimal stub that satisfies the model interface for API testing."""

    def __init__(self):
        self.ok = True
        self._call_count = 0

    def generate_response(self, prompt: str) -> str:
        self._call_count += 1
        return f"Response to: {prompt}"

    def self_correct_and_train(
        self,
        interaction_history: List[adaptible.InteractionHistory],
        indices_to_review: List[int] | None = None,
        verbose: bool = False,
    ) -> bool:
        del interaction_history, indices_to_review, verbose
        return True

    def stream_response(self, prompt: str):
        del prompt


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

    def test_trigger_review_adds_to_outstanding_tasks(self):
        """POST /trigger_review should add task to outstanding_tasks."""
        self.client.post("/interact", json={"prompt": "Q1"})

        self.client.post("/trigger_review")

        self.assertEqual(len(self.api.outstanding_tasks), 1)


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
