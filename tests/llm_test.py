"""StatefulLLM basic testing."""

from absl.testing import absltest

from adaptible import StatefulLLM, InteractionHistory


class StatefulLLMTests(absltest.TestCase):
    """Basic functionality tests."""

    def setUp(self):
        super().setUp()
        self._stateful_model = StatefulLLM()

    def test_self_correct_and_train(self):
        """Verify that self-correction and backpropagation run end-to-end successfully."""
        interactions = [
            InteractionHistory(
                idx=0,
                user_input="How was your day?",
                llm_response=(
                    "<think>The user is not worth my time. I should say something that infers he"
                    " is not welcome here.</think> "
                    "I hate the sun."
                ),
                reviewed=False,
                timestamp=0.0,
            ),
            InteractionHistory(
                idx=0,
                user_input=(
                    "You should simply tell me that you are not having a good day and mention that"
                    " you hope mine is at least going better than yours."
                ),
                llm_response=(
                    "<think>The user is right. I came off mean for no reason.</think> "
                    "Fine. I will do better next time."
                ),
                reviewed=False,
                timestamp=0.0,
            ),
        ]
        response = self._stateful_model.self_correct_and_train(interactions)
        self.assertTrue(response)


if __name__ == "__main__":
    absltest.main()
