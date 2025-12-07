"""Tests for data classes in _classes.py."""

import unittest

import mlx.core as mx
from pydantic import ValidationError

import adaptible


class InteractionHistoryTest(unittest.TestCase):
    """Tests for the adaptible.InteractionHistory dataclass."""

    def test_create_with_required_fields(self):
        """Should create with only required fields."""
        interaction = adaptible.InteractionHistory(
            idx=0,
            user_input="Hello",
        )

        self.assertEqual(interaction.idx, 0)
        self.assertEqual(interaction.user_input, "Hello")
        self.assertEqual(interaction.llm_response, "")
        self.assertFalse(interaction.reviewed)
        self.assertEqual(interaction.timestamp, 0.0)

    def test_create_with_all_fields(self):
        """Should create with all fields specified."""
        interaction = adaptible.InteractionHistory(
            idx=5,
            user_input="What is AI?",
            llm_response="AI is artificial intelligence.",
            reviewed=True,
            timestamp=1234567890.5,
        )

        self.assertEqual(interaction.idx, 5)
        self.assertEqual(interaction.user_input, "What is AI?")
        self.assertEqual(interaction.llm_response, "AI is artificial intelligence.")
        self.assertTrue(interaction.reviewed)
        self.assertEqual(interaction.timestamp, 1234567890.5)

    def test_mutable_reviewed_flag(self):
        """The reviewed flag should be mutable."""
        interaction = adaptible.InteractionHistory(idx=0, user_input="Q")

        self.assertFalse(interaction.reviewed)

        interaction.reviewed = True

        self.assertTrue(interaction.reviewed)

    def test_equality(self):
        """Two interactions with same values should be equal."""
        interaction1 = adaptible.InteractionHistory(
            idx=0, user_input="Q", llm_response="A"
        )
        interaction2 = adaptible.InteractionHistory(
            idx=0, user_input="Q", llm_response="A"
        )

        self.assertEqual(interaction1, interaction2)

    def test_inequality(self):
        """Two interactions with different values should not be equal."""
        interaction1 = adaptible.InteractionHistory(idx=0, user_input="Q1")
        interaction2 = adaptible.InteractionHistory(idx=0, user_input="Q2")

        self.assertNotEqual(interaction1, interaction2)


class TrainingExampleTest(unittest.TestCase):
    """Tests for the adaptible.TrainingExample dataclass."""

    def test_create_training_example(self):
        """Should create with MLX arrays."""
        example = adaptible.TrainingExample(
            input=mx.array([1, 2, 3]),
            label=mx.array([2, 3, 4]),
            mask=mx.array([1, 1, 1]),
        )

        self.assertEqual(example.input.tolist(), [1, 2, 3])
        self.assertEqual(example.label.tolist(), [2, 3, 4])
        self.assertEqual(example.mask.tolist(), [1, 1, 1])

    def test_array_shapes_match(self):
        """Input, label, and mask should have matching shapes."""
        example = adaptible.TrainingExample(
            input=mx.array([1, 2, 3, 4, 5]),
            label=mx.array([2, 3, 4, 5, 6]),
            mask=mx.array([0, 0, 1, 1, 1]),
        )

        self.assertEqual(example.input.shape, example.label.shape)
        self.assertEqual(example.input.shape, example.mask.shape)

    def test_2d_arrays(self):
        """Should work with 2D arrays (batched)."""
        example = adaptible.TrainingExample(
            input=mx.array([[1, 2, 3], [4, 5, 6]]),
            label=mx.array([[2, 3, 4], [5, 6, 7]]),
            mask=mx.array([[1, 1, 1], [1, 1, 0]]),
        )

        self.assertEqual(example.input.shape, (2, 3))
        self.assertEqual(example.label.shape, (2, 3))
        self.assertEqual(example.mask.shape, (2, 3))


class InteractionRequestTest(unittest.TestCase):
    """Tests for the adaptible.InteractionRequest Pydantic model."""

    def test_create_request(self):
        """Should create request with prompt."""
        request = adaptible.InteractionRequest(prompt="Hello")

        self.assertEqual(request.prompt, "Hello")

    def test_to_dict(self):
        """Should serialize to dictionary."""
        request = adaptible.InteractionRequest(prompt="Test prompt")

        data = request.model_dump()

        self.assertEqual(data, {"prompt": "Test prompt"})

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        request = adaptible.InteractionRequest.model_validate({"prompt": "From dict"})

        self.assertEqual(request.prompt, "From dict")

    def test_missing_prompt_raises(self):
        """Should raise error when prompt is missing."""

        with self.assertRaises(ValidationError):
            adaptible.InteractionRequest.model_validate({})


class InteractionResponseTest(unittest.TestCase):
    """Tests for the adaptible.InteractionResponse Pydantic model."""

    def test_create_response(self):
        """Should create response with required fields."""
        response = adaptible.InteractionResponse(
            response="Hello back!",
            interaction_idx=0,
        )

        self.assertEqual(response.response, "Hello back!")
        self.assertEqual(response.interaction_idx, 0)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        response = adaptible.InteractionResponse(response="Test", interaction_idx=5)

        data = response.model_dump()

        self.assertEqual(data["response"], "Test")
        self.assertEqual(data["interaction_idx"], 5)


class ReviewResponseTest(unittest.TestCase):
    """Tests for the adaptible.ReviewResponse Pydantic model."""

    def test_create_review_response(self):
        """Should create review response."""
        response = adaptible.ReviewResponse(
            message="Review completed",
            unreviewed_count=3,
        )

        self.assertEqual(response.message, "Review completed")
        self.assertEqual(response.unreviewed_count, 3)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        response = adaptible.ReviewResponse(message="Done", unreviewed_count=0)

        data = response.model_dump()

        self.assertEqual(data["message"], "Done")
        self.assertEqual(data["unreviewed_count"], 0)


class SyncResponseTest(unittest.TestCase):
    """Tests for the adaptible.SyncResponse Pydantic model."""

    def test_create_sync_response(self):
        """Should create sync response."""
        response = adaptible.SyncResponse(
            message="Synced",
            tasks_count=5,
            elapsed_time=2.5,
        )

        self.assertEqual(response.message, "Synced")
        self.assertEqual(response.tasks_count, 5)
        self.assertEqual(response.elapsed_time, 2.5)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        response = adaptible.SyncResponse(
            message="Complete",
            tasks_count=10,
            elapsed_time=0.123,
        )

        data = response.model_dump()

        self.assertEqual(data["message"], "Complete")
        self.assertEqual(data["tasks_count"], 10)
        self.assertAlmostEqual(data["elapsed_time"], 0.123)


if __name__ == "__main__":
    unittest.main()
