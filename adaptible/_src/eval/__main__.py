"""Runner script for evaluation experiments.

Usage:
    python -m adaptible._src.eval [options]

Options:
    --name NAME           Experiment name (default: "default")
    --train-ratio RATIO   Fraction to use for training (default: 0.8)
    --iterations N        Training iterations per example (default: 25)
    --shuffle             Shuffle the dataset
    --seed SEED           Random seed for shuffling (default: 42)
    --training-source S   "ground_truth" (fine-tune on the label) or
                          "self_generated" (train on the model's own revision)
    --revision-prompt P   Revision prompt preset for self_generated:
                          "default" or "fewshot"
    --[no]close_think     Close the chat template's open <think> block before
                          the training target (default: on)
    --subset N            Only use first N items (for quick tests)
    --category CAT        Filter to specific category
    --output PATH         Output path for HTML report
    --save-dataset PATH   Save the dataset to JSON file
    --load-dataset PATH   Load dataset from JSON file instead of default
"""

import webbrowser

import vizible
from absl import app, flags

from . import (
    TRAINING_SOURCES,
    EvaluationConfig,
    EvaluationHarness,
    generate_default_dataset,
    generate_html_report,
    load_dataset,
    save_dataset,
)
from ..revise import REVISION_PROMPTS

_NAME = flags.DEFINE_string("name", "default", "Experiment name")
_TRAIN_RATIO = flags.DEFINE_float("train_ratio", 0.8, "Train/holdout split ratio")
_ITERATIONS = flags.DEFINE_integer("iterations", 25, "Training iterations per example")
_SHUFFLE = flags.DEFINE_boolean("shuffle", False, "Shuffle the dataset")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_TRAINING_SOURCE = flags.DEFINE_enum(
    "training_source",
    "ground_truth",
    list(TRAINING_SOURCES),
    "What the model is trained on: the dataset label (ground_truth) or its own "
    "revision of its baseline answer (self_generated).",
)
_REVISION_PROMPT = flags.DEFINE_string(
    "revision_prompt",
    "default",
    f"Revision prompt preset used with --training_source self_generated; one of "
    f"{', '.join(REVISION_PROMPTS)}. 'fewshot' is a short imperative prompt with "
    "two worked examples over a plain User:/Assistant: dialog (no chat-template "
    "tokens).",
)
_CLOSE_THINK = flags.DEFINE_boolean(
    "close_think",
    True,
    "If the chat template's generation prompt ends with an open <think> tag, "
    "train on '</think>\\n\\n{revision}' so the target is a well-formed sequence. "
    "--noclose_think reproduces the old target (revision inside the open think "
    "block) for comparison.",
)
_SUBSET = flags.DEFINE_integer("subset", None, "Only use first N items")
_CATEGORY = flags.DEFINE_string("category", None, "Filter to specific category")
_OUTPUT = flags.DEFINE_string(
    "output", "/tmp/adaptible_eval_report.html", "Output HTML path"
)
_SAVE_DATASET = flags.DEFINE_string("save_dataset", None, "Save dataset to JSON file")
_LOAD_DATASET = flags.DEFINE_string("load_dataset", None, "Load dataset from JSON file")
_NO_BROWSER = flags.DEFINE_boolean("no_browser", False, "Don't open browser")


def main(_):
    # Load or generate dataset
    if _LOAD_DATASET.value:
        print(f"Loading dataset from {_LOAD_DATASET.value}...")
        dataset = load_dataset(_LOAD_DATASET.value)
    else:
        print("Generating default dataset...")
        dataset = generate_default_dataset()

    print(f"Dataset: {dataset.name} ({len(dataset)} items)")
    print(f"Categories: {', '.join(sorted(dataset.categories))}")

    # Save dataset if requested
    if _SAVE_DATASET.value:
        save_dataset(dataset, _SAVE_DATASET.value)
        print(f"Saved dataset to {_SAVE_DATASET.value}")

    # Apply filters
    if _CATEGORY.value:
        dataset = dataset.by_category(_CATEGORY.value)
        print(f"Filtered to category '{_CATEGORY.value}': {len(dataset)} items")

    if _SUBSET.value:
        dataset = dataset.subset(list(range(min(_SUBSET.value, len(dataset)))))
        print(f"Using subset: {len(dataset)} items")

    if len(dataset) == 0:
        print("Error: No items in dataset after filtering")
        return 1

    # Create config
    config = EvaluationConfig(
        name=_NAME.value,
        training_iterations=_ITERATIONS.value,
        shuffle=_SHUFFLE.value,
        seed=_SEED.value,
        train_ratio=_TRAIN_RATIO.value,
        training_source=_TRAINING_SOURCE.value,
        revision_prompt=_REVISION_PROMPT.value,
        close_think=_CLOSE_THINK.value,
    )

    # Run evaluation
    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)
    print()

    harness = EvaluationHarness()
    result = harness.run(dataset, config, verbose=True)

    # Generate report
    report_path = generate_html_report(result, _OUTPUT.value)
    print()
    print("=" * 70)
    print("REPORT GENERATED")
    print("=" * 70)
    vizible.green(f"Report: file://{report_path}")

    if not _NO_BROWSER.value:
        webbrowser.open_new_tab(f"file://{report_path}")

    return 0


if __name__ == "__main__":
    app.run(main)
