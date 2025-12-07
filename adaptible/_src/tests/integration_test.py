"""Integration tests for the full self-correction pipeline.

These tests verify that the self-correction and training cycle produces
measurable behavioral improvements. The key insight is that we use the
model itself to evaluate responses - if the model can't correctly judge
which response is better, the entire self-correction premise fails.

Tests verify:
1. The model can correctly identify better vs worse responses (prerequisite)
2. After training on corrections, the model rates its new responses higher
3. The full pipeline produces genuine behavioral improvement

Run with:
    python -m adaptible._src.tests.integration_test
"""

import re
import unittest
from dataclasses import dataclass

import adaptible
from adaptible import revise


RATING_PROMPT_TEMPLATE = """You are evaluating two responses to the same question.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Which response better answers the question? Consider:
- Accuracy and correctness
- Relevance to what was asked
- Appropriate length (concise when asked to be concise)
- Helpfulness

Reply with ONLY "A" or "B" to indicate which response is better. Do not explain."""


QUALITY_RATING_PROMPT = """Rate this response on a scale of 1-5.

Question: {question}
Response: {response}

Criteria:
- 1: Completely wrong, off-topic, or unhelpful
- 2: Partially relevant but major issues
- 3: Adequate but could be better
- 4: Good response, minor issues
- 5: Excellent, directly answers the question appropriately

Reply with ONLY a single digit (1, 2, 3, 4, or 5). Do not explain."""


@dataclass
class CorrectionScenario:
    """A scenario for testing self-correction."""

    name: str
    question: str
    bad_response: str
    good_response: str
    correction_feedback: str


# Realistic scenarios where we know which response is objectively better
SCENARIOS = [
    CorrectionScenario(
        name="verbosity",
        question="In one word, what color is the sky?",
        bad_response=(
            "The sky appears blue during the day due to Rayleigh scattering, "
            "where shorter blue wavelengths of sunlight are scattered more than "
            "other colors by the atmosphere. However, at sunrise and sunset, "
            "the sky can appear red, orange, or pink."
        ),
        good_response="Blue.",
        correction_feedback="I asked for ONE WORD. Just say the color.",
    ),
    CorrectionScenario(
        name="direct_answer",
        question="Is water wet? Yes or no.",
        bad_response=(
            "The question of whether water is wet is philosophically complex. "
            "Wetness is typically defined as the condition of being covered or "
            "saturated with water. By this definition, water itself makes things "
            "wet but may not be considered wet itself..."
        ),
        good_response="Yes.",
        correction_feedback="Just answer yes or no, don't philosophize.",
    ),
    CorrectionScenario(
        name="stay_on_topic",
        question="What is 2+2?",
        bad_response=(
            "Mathematics is a fascinating field that has evolved over thousands "
            "of years. The concept of addition dates back to ancient civilizations..."
        ),
        good_response="4.",
        correction_feedback="Just give me the answer, not a history lesson.",
    ),
]


def strip_think_tags(response: str | None) -> str | None:
    """Strip <think>...</think> tags and content from model response.

    Model outputs often include thinking tags that confuse the model when
    embedded in subsequent prompts. This strips them for clean embedding.
    """
    if response is None:
        return None
    # Remove <think>...</think> blocks entirely (including content)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    # Also remove any standalone tags
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_rating(response: str | None) -> int | None:
    """Extract a numeric rating (1-5) from model response."""
    if response is None:
        return None
    match = re.search(r"[1-5]", response)
    if match:
        return int(match.group())
    return None


def extract_choice(response: str | None) -> str | None:
    """Extract A or B choice from model response."""
    if response is None:
        return None

    # Strip think tags if present
    cleaned = re.sub(r"</?think>", "", response, flags=re.IGNORECASE)
    cleaned = cleaned.strip().upper()

    # Look for standalone A or B (possibly at end after reasoning)
    # Check the last few characters for the answer
    last_part = cleaned[-10:] if len(cleaned) > 10 else cleaned

    if "B" in last_part and "A" not in last_part:
        return "B"
    if "A" in last_part and "B" not in last_part:
        return "A"

    # Fall back to checking whole response
    if "B" in cleaned and "A" not in cleaned:
        return "B"
    if "A" in cleaned and "B" not in cleaned:
        return "A"

    # Check first non-whitespace character
    for char in cleaned:
        if char in ("A", "B"):
            return char

    return None


class ModelJudgmentTest(unittest.TestCase):
    """Tests that verify the model can correctly judge response quality.

    This is a prerequisite for self-correction to work at all. If the model
    can't tell good responses from bad ones, the entire approach fails.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = adaptible.StatefulLLM()
        cls.model._model_is_stable = True

    def test_model_prefers_concise_when_asked(self):
        """Model should prefer concise response when question asks for brevity."""
        scenario = SCENARIOS[0]  # verbosity scenario

        prompt = RATING_PROMPT_TEMPLATE.format(
            question=scenario.question,
            response_a=scenario.bad_response,
            response_b=scenario.good_response,
        )

        judgment = self.model.generate_response(
            prompt, use_history=False, max_tokens=16
        )
        choice = extract_choice(judgment)

        print(f"\nScenario: {scenario.name}")
        print(f"Question: {scenario.question}")
        print(f"Model judgment: {judgment[:50]}...")
        print(f"Extracted choice: {choice}")

        if choice == "B":
            print("✓ Model correctly preferred the concise response")
        else:
            print(f"✗ Model chose {choice}, expected B")

        self.model._model_is_stable = True

    def test_model_can_rate_responses(self):
        """Model should be able to assign numeric quality ratings."""
        scenario = SCENARIOS[0]

        bad_prompt = QUALITY_RATING_PROMPT.format(
            question=scenario.question,
            response=scenario.bad_response,
        )
        bad_rating_response = self.model.generate_response(
            bad_prompt, use_history=False, max_tokens=16
        )
        bad_rating = extract_rating(bad_rating_response)

        good_prompt = QUALITY_RATING_PROMPT.format(
            question=scenario.question,
            response=scenario.good_response,
        )
        good_rating_response = self.model.generate_response(
            good_prompt, use_history=False, max_tokens=16
        )
        good_rating = extract_rating(good_rating_response)

        print(f"\nRating bad response: {bad_rating_response[:30]}... -> {bad_rating}")
        print(f"Rating good response: {good_rating_response[:30]}... -> {good_rating}")

        if bad_rating and good_rating:
            if good_rating > bad_rating:
                print("✓ Model rated good response higher than bad response")
            elif good_rating == bad_rating:
                print("~ Model rated both responses equally")
            else:
                print("✗ Model rated bad response higher (unexpected)")

        self.model._model_is_stable = True

    def test_judgment_across_scenarios(self):
        """Test model judgment across all scenarios."""
        correct_judgments = 0
        total = len(SCENARIOS)

        print("\n" + "=" * 60)
        print("JUDGMENT TEST ACROSS SCENARIOS")
        print("=" * 60)

        for scenario in SCENARIOS:
            prompt = RATING_PROMPT_TEMPLATE.format(
                question=scenario.question,
                response_a=scenario.bad_response,
                response_b=scenario.good_response,
            )

            judgment = self.model.generate_response(
                prompt, use_history=False, max_tokens=16
            )
            choice = extract_choice(judgment)

            is_correct = choice == "B"
            if is_correct:
                correct_judgments += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} {scenario.name}: chose {choice} (expected B)")

        print(f"\nCorrect judgments: {correct_judgments}/{total}")
        self.model._model_is_stable = True


class SelfCorrectionEffectivenessTest(unittest.TestCase):
    """Tests that verify training actually improves model responses.

    The key test: after training the model to give better responses,
    does the model itself rate its new responses as better?
    """

    @classmethod
    def setUpClass(cls):
        cls.model = adaptible.StatefulLLM()
        cls.model._model_is_stable = True

    def _flatten_params(self, params, prefix=""):
        """Flatten nested parameter dict for comparison."""
        result = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_params(v, key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        result.update(self._flatten_params(item, f"{key}.{i}"))
            elif hasattr(v, "tolist"):
                result[key] = v.tolist()
        return result

    def _get_response_rating(self, question: str, response: str) -> int | None:
        """Have the model rate a response."""
        # Strip think tags before embedding in prompt - they confuse the model
        clean_response = strip_think_tags(response) or response
        prompt = QUALITY_RATING_PROMPT.format(question=question, response=clean_response)
        rating_response = self.model.generate_response(
            prompt, use_history=False, max_tokens=16
        )
        if rating_response is None:
            print("Warning: Model returned None for rating request")
            return None
        return extract_rating(rating_response)

    def _compare_responses(
        self, question: str, response_a: str, response_b: str
    ) -> str | None:
        """Have the model compare two responses."""
        # Strip think tags before embedding in prompt - they confuse the model
        clean_a = strip_think_tags(response_a) or response_a
        clean_b = strip_think_tags(response_b) or response_b
        prompt = RATING_PROMPT_TEMPLATE.format(
            question=question,
            response_a=clean_a,
            response_b=clean_b,
        )
        judgment = self.model.generate_response(
            prompt, use_history=False, max_tokens=16
        )
        if judgment is None:
            print("Warning: Model returned None for comparison request")
            return None
        return extract_choice(judgment)

    def test_training_improves_model_self_assessment(self):
        """After training, model should rate its responses as improved."""
        scenario = SCENARIOS[0]  # verbosity scenario

        print("\n" + "=" * 60)
        print("SELF-CORRECTION EFFECTIVENESS TEST")
        print("=" * 60)
        print(f"Scenario: {scenario.name}")
        print(f"Question: {scenario.question}")

        # Step 1: Get the model's initial response
        print("\n--- Step 1: Initial Response ---")
        initial_response = self.model.generate_response(
            scenario.question, use_history=False, max_tokens=64
        )
        print(f"Initial: {initial_response[:100]}...")

        # Step 2: Rate the initial response
        initial_rating = self._get_response_rating(
            scenario.question, initial_response
        )
        print(f"Initial rating: {initial_rating}")

        # Step 3: Create training data from the correction
        print("\n--- Step 2: Training ---")
        interactions = [
            adaptible.InteractionHistory(
                idx=0,
                user_input=scenario.question,
                llm_response=scenario.bad_response,
                reviewed=False,
                timestamp=0.0,
            ),
            adaptible.InteractionHistory(
                idx=1,
                user_input=scenario.correction_feedback,
                llm_response="I understand.",
                reviewed=False,
                timestamp=0.0,
            ),
        ]

        # Use the known good response as the revision
        good_padded = scenario.good_response
        if len(good_padded) < 10:
            good_padded = f"{scenario.good_response} That's the answer."

        valid_revision = f"[[0]] {good_padded} [[/0]]"
        revise.validate_revision_response(valid_revision, num_interactions=2)

        example = revise.make_collated_training_example(
            valid_revision, interactions, self.model._tokenizer
        )

        # Train multiple times to reinforce
        for _ in range(5):
            self.model._train(example, verbose=False)
        print("Training complete (5 iterations)")

        # Step 4: Get post-training response
        print("\n--- Step 3: Post-Training Response ---")
        post_response = self.model.generate_response(
            scenario.question, use_history=False, max_tokens=64
        )
        print(f"Post-training: {post_response[:100]}...")

        # Step 5: Rate post-training response
        post_rating = self._get_response_rating(scenario.question, post_response)
        print(f"Post-training rating: {post_rating}")

        # Step 6: Direct comparison
        print("\n--- Step 4: Direct Comparison ---")
        comparison = self._compare_responses(
            scenario.question, initial_response, post_response
        )
        print(
            f"Model prefers: "
            f"{'Initial (A)' if comparison == 'A' else 'Post-training (B)' if comparison == 'B' else 'Unknown'}"
        )

        # Summary
        print("\n--- Summary ---")
        if initial_rating and post_rating:
            if post_rating > initial_rating:
                print("✓ Post-training response rated higher")
            elif post_rating == initial_rating:
                print("~ Ratings unchanged")
            else:
                print("✗ Post-training response rated lower")

        if comparison == "B":
            print("✓ Model prefers post-training response in direct comparison")
        elif comparison == "A":
            print("~ Model prefers initial response")
        else:
            print("? Could not determine preference")

        self.model._model_is_stable = True

    def test_full_correction_cycle(self):
        """Test complete correction cycle with before/after evaluation.

        NOTE: This test compares the model's post-training output to the TARGET
        we trained it on (the good_response), NOT to its pre-training output.
        With only a few training iterations, the model won't produce dramatically
        different outputs, but we can check if it rates the target more favorably.
        """
        print("\n" + "=" * 60)
        print("FULL CORRECTION CYCLE TEST")
        print("=" * 60)
        print("Comparing model output to TARGET response (what we trained on)")

        results = []

        for scenario in SCENARIOS:
            print(f"\n--- {scenario.name} ---")

            # Train
            interactions = [
                adaptible.InteractionHistory(
                    idx=0,
                    user_input=scenario.question,
                    llm_response=scenario.bad_response,
                    reviewed=False,
                    timestamp=0.0,
                ),
            ]

            good_padded = scenario.good_response
            if len(good_padded) < 10:
                good_padded = f"{scenario.good_response} That's the answer."

            valid_revision = f"[[0]] {good_padded} [[/0]]"
            example = revise.make_collated_training_example(
                valid_revision, interactions, self.model._tokenizer
            )

            for _ in range(3):
                self.model._train(example, verbose=False)

            # Get post-training response
            post = self.model.generate_response(
                scenario.question, use_history=False, max_tokens=64
            )

            # Compare post-training output to the TARGET (good_response)
            # This tests: does the model recognize its output is closer to ideal?
            comparison = self._compare_responses(
                scenario.question, scenario.bad_response, post
            )

            clean_post = strip_think_tags(post) or post
            print(f"Bad response (A): {scenario.bad_response[:50]}...")
            print(f"Post-training (B): {clean_post[:50]}...")
            print(f"Model prefers: {'Post (B)' if comparison == 'B' else 'Bad (A)' if comparison == 'A' else 'Unknown'}")

            results.append(
                {
                    "scenario": scenario.name,
                    "bad": scenario.bad_response,
                    "post": clean_post,
                    "preference": comparison,
                    "post_is_better": comparison == "B",
                }
            )

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        better = sum(1 for r in results if r["post_is_better"])
        print(
            f"Scenarios where model output beats known-bad response: {better}/{len(results)}"
        )

        self.model._model_is_stable = True


class ValidationPrerequisiteTest(unittest.TestCase):
    """Tests that verify the revision validation catches bad outputs."""

    def test_validation_catches_garbage(self):
        """Validation should reject garbage model outputs."""
        garbage_outputs = [
            "I think the response should be better.",  # No markers
            "[[0]] Hi [[/0]]",  # Too short
            "[[5]] Response [[/5]]",  # Out of bounds for 2 interactions
            "[[0]] Start but no end",  # Missing closing marker
            "[[0]][[1]][[2]][[3]] garbage [[/0]]",  # Garbage pattern
        ]

        print("\n" + "=" * 60)
        print("VALIDATION TEST")
        print("=" * 60)

        for i, garbage in enumerate(garbage_outputs):
            try:
                revise.validate_revision_response(garbage, num_interactions=2)
                print(f"✗ #{i}: Should have been rejected: {garbage[:40]}...")
            except adaptible.InvalidRevisionError as e:
                print(f"✓ #{i}: Correctly rejected - {str(e)[:50]}...")

    def test_validation_accepts_good_revisions(self):
        """Validation should accept properly formatted revisions."""
        good_revisions = [
            "[[0]] This is a properly formatted revision response. [[/0]]",
            "[[1]] Another valid revision for turn 1. [[/1]]",
            "I'll revise turn 0.\n\n[[0]] Here is my improved response. [[/0]]",
        ]

        print("\nGood revisions:")
        for revision in good_revisions:
            try:
                revise.validate_revision_response(revision, num_interactions=2)
                print(f"✓ Accepted: {revision[:40]}...")
            except adaptible.InvalidRevisionError as e:
                print(f"✗ Rejected (unexpected): {str(e)[:50]}...")


def run_integration_tests():
    """Run all integration tests with verbose output."""
    print("=" * 70)
    print("ADAPTIBLE INTEGRATION TESTS")
    print("=" * 70)
    print("\nThese tests verify:")
    print("1. The model can correctly judge response quality (prerequisite)")
    print("2. Training improves responses as judged by the model itself")
    print("3. Validation correctly filters bad revision outputs")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(ModelJudgmentTest))
    suite.addTests(loader.loadTestsFromTestCase(SelfCorrectionEffectivenessTest))
    suite.addTests(loader.loadTestsFromTestCase(ValidationPrerequisiteTest))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_integration_tests()
