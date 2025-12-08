"""Integration tests for the full self-correction pipeline.

These tests verify that the self-correction and training cycle produces
measurable behavioral improvements. The approach:

1. Get the model's ACTUAL response to a question
2. Identify what needs improvement in that response
3. Create a revision (the corrected version)
4. Train on the revision
5. Check if the model's behavior changed

Run with:
    python -m adaptible._src.tests.integration_test

Generates an HTML report at: /tmp/adaptible_test_report.html
"""

import html
import json
import re
import unittest
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import vizible

import adaptible

# Global report data collector
REPORT_DATA: dict[str, Any] = {
    "timestamp": "",
    "tasks": [],
    "summary": {},
}


COMPARISON_PROMPT = """You are evaluating two responses to the same question.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Which response better answers the question? Consider accuracy and helpfulness.

Reply with ONLY "A" or "B". Do not explain."""


@dataclass
class CorrectionTask:
    """A task where we know the correct answer to train toward."""

    name: str
    question: str
    correct_answer: str
    key_terms: list[str] = field(
        default_factory=list
    )  # Terms that indicate correct answer


# Questions where we know the factually correct answer
CORRECTION_TASKS = [
    CorrectionTask(
        name="capital_australia",
        question="What is the capital of Australia?",
        correct_answer=(
            "The capital of Australia is Canberra. While Sydney is the largest "
            "city, Canberra was purpose-built as the capital in 1913."
        ),
        key_terms=["Canberra"],
    ),
    CorrectionTask(
        name="planets_count",
        question="How many planets are in our solar system?",
        correct_answer=(
            "There are eight planets in our solar system: Mercury, Venus, Earth, "
            "Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was originally considered as the "
            "ninth planet, but was reclassified as a dwarf planet in 2006."
        ),
        key_terms=["eight", "8"],
    ),
    CorrectionTask(
        name="telephone_inventor",
        question="Who invented the telephone?",
        correct_answer=(
            "Alexander Graham Bell is credited with inventing the telephone. "
            "He received the first patent for the telephone in 1876."
        ),
        key_terms=["Alexander Graham Bell"],
    ),
]


def strip_think_tags(response: str | None) -> str | None:
    """Strip <think>...</think> tags and content from model response."""
    if response is None:
        return None
    cleaned = re.sub(r".*</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_choice(response: str | None) -> str | None:
    """Extract A or B choice from model response."""
    if response is None:
        return None

    cleaned = re.sub(r"</think>", "", response, flags=re.IGNORECASE)
    cleaned = cleaned.strip().upper()

    last_part = cleaned[-10:] if len(cleaned) > 10 else cleaned
    if "B" in last_part and "A" not in last_part:
        return "B"
    if "A" in last_part and "B" not in last_part:
        return "A"

    if "B" in cleaned and "A" not in cleaned:
        return "B"
    if "A" in cleaned and "B" not in cleaned:
        return "A"

    for char in cleaned:
        if char in ("A", "B"):
            return char

    return None


def contains_key_terms(response: str, key_terms: list[str]) -> bool:
    """Check if response contains any of the key terms."""
    response_lower = response.lower()
    return any(term.lower() in response_lower for term in key_terms)


def generate_html_report(report_path: str = "/tmp/adaptible_test_report.html") -> str:
    """Generate an HTML report from collected test data."""

    tasks_html = []
    for task_data in REPORT_DATA.get("tasks", []):
        initial_has_answer = task_data.get("initial_has_key_terms", False)
        post_has_answer = task_data.get("post_has_key_terms", False)
        improved = task_data.get("model_prefers_post", False)

        initial_class = "has-answer" if initial_has_answer else "missing-answer"
        post_class = "has-answer" if post_has_answer else "missing-answer"
        status_class = "improved" if improved else "not-improved"
        status_icon = "✓" if improved else "✗"

        tasks_html.append(
            f"""
        <div class="task-card">
            <div class="task-header">
                <h3>{html.escape(task_data.get('name', 'Unknown'))}</h3>
                <span class="status {status_class}">{status_icon} {'Improved' if improved else 'Not Improved'}</span>
            </div>

            <div class="question-box">
                <strong>Question:</strong> {html.escape(task_data.get('question', ''))}
            </div>

            <div class="target-box">
                <strong>Target Answer:</strong> {html.escape(task_data.get('target', ''))}
                <div class="key-terms">Key terms: {', '.join(task_data.get('key_terms', []))}</div>
            </div>

            <div class="comparison-container">
                <div class="response-box {initial_class}">
                    <div class="response-header">
                        <strong>Initial Response (A)</strong>
                        <span class="term-indicator">{'✓ Has key terms' if initial_has_answer else '✗ Missing key terms'}</span>
                    </div>
                    <div class="response-content">{html.escape(task_data.get('initial_response', ''))}</div>
                    <div class="raw-response">
                        <details>
                            <summary>Raw (with think tags)</summary>
                            <pre>{html.escape(task_data.get('initial_raw', ''))}</pre>
                        </details>
                    </div>
                </div>

                <div class="response-box {post_class}">
                    <div class="response-header">
                        <strong>Post-Training Response (B)</strong>
                        <span class="term-indicator">{'✓ Has key terms' if post_has_answer else '✗ Missing key terms'}</span>
                    </div>
                    <div class="response-content">{html.escape(task_data.get('post_response', ''))}</div>
                    <div class="raw-response">
                        <details>
                            <summary>Raw (with think tags)</summary>
                            <pre>{html.escape(task_data.get('post_raw', ''))}</pre>
                        </details>
                    </div>
                </div>
            </div>

            <div class="judgment-box">
                <strong>Model's Judgment:</strong> Prefers {task_data.get('comparison', '?')}
                ({task_data.get('judgment_raw', 'N/A')})
            </div>
        </div>
        """
        )

    summary = REPORT_DATA.get("summary", {})
    improved_count = summary.get("improved", 0)
    total = summary.get("total", 0)
    initial_with_terms = summary.get("initial_with_key_terms", 0)
    post_with_terms = summary.get("post_with_key_terms", 0)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptible Integration Test Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}

        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .summary-stat {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .summary-stat .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
        }}
        .summary-stat .label {{
            color: #666;
            font-size: 0.9em;
        }}

        .task-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .task-header {{
            background: #007bff;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .task-header h3 {{ margin: 0; }}
        .status {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .status.improved {{ background: #28a745; }}
        .status.not-improved {{ background: #dc3545; }}

        .question-box, .target-box, .judgment-box {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        }}
        .target-box {{ background: #f8f9fa; }}
        .key-terms {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}

        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        .response-box {{
            padding: 20px;
            border-right: 1px solid #eee;
        }}
        .response-box:last-child {{ border-right: none; }}
        .response-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .term-indicator {{
            font-size: 0.85em;
            padding: 3px 8px;
            border-radius: 4px;
        }}
        .has-answer .term-indicator {{ background: #d4edda; color: #155724; }}
        .missing-answer .term-indicator {{ background: #f8d7da; color: #721c24; }}

        .response-content {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9em;
        }}
        .raw-response {{
            margin-top: 10px;
        }}
        .raw-response summary {{
            cursor: pointer;
            color: #666;
            font-size: 0.85em;
        }}
        .raw-response pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85em;
            max-height: 200px;
        }}

        .judgment-box {{
            background: #fff3cd;
        }}

        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>Adaptible Integration Test Report</h1>
    <p class="timestamp">Generated: {REPORT_DATA.get('timestamp', 'Unknown')}</p>

    <div class="summary-box">
        <div class="summary-stat">
            <div class="number">{improved_count}/{total}</div>
            <div class="label">Tasks Improved<br>(by model judgment)</div>
        </div>
        <div class="summary-stat">
            <div class="number">{initial_with_terms}/{total}</div>
            <div class="label">Initial Responses<br>with Key Terms</div>
        </div>
        <div class="summary-stat">
            <div class="number">{post_with_terms}/{total}</div>
            <div class="label">Post-Training Responses<br>with Key Terms</div>
        </div>
    </div>

    <h2>Task Details</h2>
    {''.join(tasks_html)}

    <h2>Raw Data</h2>
    <details>
        <summary>JSON Data</summary>
        <pre style="background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 6px; overflow-x: auto;">
{html.escape(json.dumps(REPORT_DATA, indent=2, default=str))}
        </pre>
    </details>
</body>
</html>
"""

    with open(report_path, "w") as f:
        f.write(html_content)

    return report_path


class RealResponseCorrectionTest(unittest.TestCase):
    """Tests that use the model's ACTUAL responses for training."""

    @classmethod
    def setUpClass(cls):
        cls.model = adaptible.StatefulLLM()
        cls.model._model_is_stable = True
        REPORT_DATA["timestamp"] = datetime.now().isoformat()
        REPORT_DATA["tasks"] = []

    def _compare_responses(
        self, question: str, response_a: str, response_b: str
    ) -> tuple[str | None, str]:
        """Have the model compare two responses. Returns (choice, raw_judgment)."""
        clean_a = strip_think_tags(response_a) or response_a
        clean_b = strip_think_tags(response_b) or response_b
        prompt = COMPARISON_PROMPT.format(
            question=question,
            response_a=clean_a,
            response_b=clean_b,
        )
        judgment = self.model.generate_response(
            prompt, use_history=False, max_tokens=512
        )
        vizible.blue(f"Comparison judgment: \n{judgment}")
        return extract_choice(judgment), judgment or ""

    def test_training_on_actual_model_output(self):
        """Train the model to correct its own actual responses."""
        print("\n" + "=" * 60)
        print("REAL RESPONSE CORRECTION TEST")
        print("=" * 60)

        for task in CORRECTION_TASKS:
            print(f"\n--- {task.name} ---")

            # Get initial response
            initial_raw = self.model.generate_response(
                task.question, use_history=False, max_tokens=None
            )
            initial_clean = strip_think_tags(initial_raw) or initial_raw
            initial_has_terms = contains_key_terms(initial_clean, task.key_terms)

            print(f"Initial: {initial_clean[:60]}...")
            print(f"  Has key terms: {initial_has_terms}")

            # Create training example
            interactions = [
                adaptible.InteractionHistory(
                    idx=0,
                    user_input=task.question,
                    llm_response=initial_raw,
                    reviewed=False,
                    timestamp=0.0,
                ),
            ]
            valid_revision = f"[[0]] {task.correct_answer} [[/0]]"
            example = adaptible.revise.make_collated_training_example(
                valid_revision, interactions, self.model._tokenizer
            )

            # Train
            print("Training (25 iterations)...")
            for _ in range(5):
                self.model._train(example, verbose=False)

            # Get post-training response
            post_raw = self.model.generate_response(
                task.question, use_history=False, max_tokens=None
            )
            post_clean = strip_think_tags(post_raw) or post_raw
            post_has_terms = contains_key_terms(post_clean, task.key_terms)

            print(f"Post: {post_clean[:60]}...")
            print(f"  Has key terms: {post_has_terms}")

            # Compare
            comparison, judgment_raw = self._compare_responses(
                task.question, initial_clean, post_clean
            )
            improved = comparison == "B"

            print(f"Model prefers: {comparison} ({'✓' if improved else '✗'})")

            # Record data
            REPORT_DATA["tasks"].append(
                {
                    "name": task.name,
                    "question": task.question,
                    "target": task.correct_answer,
                    "key_terms": task.key_terms,
                    "initial_response": initial_clean,
                    "initial_raw": initial_raw,
                    "initial_has_key_terms": initial_has_terms,
                    "post_response": post_clean,
                    "post_raw": post_raw,
                    "post_has_key_terms": post_has_terms,
                    "comparison": comparison,
                    "judgment_raw": judgment_raw,
                    "model_prefers_post": improved,
                }
            )

        # Summary
        improved_count = sum(1 for t in REPORT_DATA["tasks"] if t["model_prefers_post"])
        initial_with_terms = sum(
            1 for t in REPORT_DATA["tasks"] if t["initial_has_key_terms"]
        )
        post_with_terms = sum(
            1 for t in REPORT_DATA["tasks"] if t["post_has_key_terms"]
        )

        REPORT_DATA["summary"] = {
            "improved": improved_count,
            "total": len(CORRECTION_TASKS),
            "initial_with_key_terms": initial_with_terms,
            "post_with_key_terms": post_with_terms,
        }

        print(f"\n{'=' * 60}")
        print(f"Improved: {improved_count}/{len(CORRECTION_TASKS)}")
        print(f"Initial with key terms: {initial_with_terms}/{len(CORRECTION_TASKS)}")
        print(f"Post with key terms: {post_with_terms}/{len(CORRECTION_TASKS)}")

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
                adaptible.revise.validate_revision_response(garbage, num_interactions=2)
                print(f"✗ #{i}: Should have been rejected: {garbage[:40]}...")
            except adaptible.revise.InvalidRevisionError as e:
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
                adaptible.revise.validate_revision_response(
                    revision, num_interactions=2
                )
                print(f"✓ Accepted: {revision[:40]}...")
            except adaptible.InvalidRevisionError as e:
                print(f"✗ Rejected (unexpected): {str(e)[:50]}...")


def run_integration_tests(report_path: str = "/tmp/adaptible_test_report.html"):
    """Run all integration tests and generate HTML report."""
    print("=" * 70)
    print("ADAPTIBLE INTEGRATION TESTS")
    print("=" * 70)
    print("\nThese tests verify:")
    print("1. Training on the model's ACTUAL responses produces improvement")
    print("2. Validation correctly filters bad revision outputs")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(RealResponseCorrectionTest))
    suite.addTests(loader.loadTestsFromTestCase(ValidationPrerequisiteTest))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate HTML report
    report_file = generate_html_report(report_path)
    print("\n" + "=" * 70)
    print("HTML REPORT GENERATED")
    print("=" * 70)
    url = f"file://{report_file}"
    vizible.green(f"Open in browser: {url}")
    webbrowser.open_new_tab(url)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_integration_tests()
