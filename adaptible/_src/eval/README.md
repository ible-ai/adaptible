# Evaluation Framework

Systematic evaluation of Adaptible's self-correction effectiveness using trivia datasets.

## Overview

The evaluation framework measures how well the model learns from corrections by:

1. Getting baseline responses to all questions
2. Training on a subset (default 80%)
3. Re-evaluating all questions post-training
4. Comparing train vs holdout performance

## Quick Start

### Command Line

```bash
# Run full evaluation (100+ questions)
python -m adaptible.eval

# Quick test with 20 questions
python -m adaptible.eval --subset 20 --shuffle

# Test specific category
python -m adaptible.eval --category geography --iterations 50

# Save results to custom location
python -m adaptible.eval --output ~/my_report.html
```

### Programmatic Usage

```python
import adaptible.eval as eval

# Load built-in dataset (100+ trivia questions)
dataset = eval.generate_default_dataset()

# Configure experiment
config = eval.EvaluationConfig(
    name="my_experiment",
    training_iterations=25,
    train_ratio=0.8,
    shuffle=True,
)

# Run evaluation
harness = eval.EvaluationHarness()
result = harness.run(dataset, config, verbose=True)

# Generate HTML report
eval.generate_html_report(result, "/tmp/report.html")

# Access metrics
print(f"Baseline accuracy: {result.baseline_accuracy:.1%}")
print(f"Train improvement: {result.train_improvement_rate:.1%}")
print(f"Holdout accuracy: {result.holdout_accuracy:.1%}")
```

## Dataset

The built-in dataset includes 100+ trivia questions across 6 categories:

- **Geography** (20 questions) - Capitals, landmarks, physical geography
- **Science** (25 questions) - Physics, chemistry, biology, astronomy
- **History** (20 questions) - Inventions, inventors, major events
- **Math** (15 questions) - Basic calculations, constants, geometry
- **Language** (10 questions) - Linguistics, writing systems
- **Miscellaneous** (15 questions) - Animals, geography records, culture

Each item includes:

- Question text
- Correct answer
- Key terms for automated grading
- Difficulty level (easy/medium/hard)

### Custom Datasets

```python
from adaptible.eval import TriviaDataset, TriviaItem, save_dataset

# Create custom dataset
dataset = TriviaDataset(
    name="my_dataset",
    version="1.0.0",
    items=[
        TriviaItem(
            id="q1",
            category="science",
            question="What is the speed of light?",
            correct_answer="299,792 km/s",
            key_terms=["299,792", "300,000"],
            difficulty="medium",
        ),
        # ... more items
    ],
)

# Save for reuse
save_dataset(dataset, "my_dataset.json")

# Load later
dataset = eval.load_dataset("my_dataset.json")
```

## Metrics

The evaluation computes several metrics:

| Metric                     | Description                                         |
| -------------------------- | --------------------------------------------------- |
| **Baseline Accuracy**      | % of items with key terms before training           |
| **Train Post-Accuracy**    | % of trained items correct after training           |
| **Train Improvement Rate** | % of incorrect items that became correct            |
| **Train Retention Rate**   | % of correct items that stayed correct              |
| **Holdout Accuracy**       | % of untrained items correct (generalization check) |

## HTML Report

The generated HTML report includes:

- Summary metrics with visual indicators
- Per-item comparison (before/after responses)
- Filtering by category, training status, and outcome
- Color-coded results (improved/retained/degraded)

## CLI Options

| Flag             | Default                           | Description                             |
| ---------------- | --------------------------------- | --------------------------------------- |
| `--name`         | `"default"`                       | Experiment name                         |
| `--train-ratio`  | `0.8`                             | Fraction for training (rest is holdout) |
| `--iterations`   | `25`                              | Training iterations per example         |
| `--shuffle`      | `False`                           | Randomize question order                |
| `--seed`         | `42`                              | Random seed for shuffling               |
| `--subset`       | `None`                            | Use only first N questions              |
| `--category`     | `None`                            | Filter to specific category             |
| `--output`       | `/tmp/adaptible_eval_report.html` | Report path                             |
| `--save-dataset` | `None`                            | Save dataset to JSON                    |
| `--load-dataset` | `None`                            | Load custom dataset from JSON           |
| `--no-browser`   | `False`                           | Don't auto-open report in browser       |

## Files

```text
eval/
├── __init__.py           # Public API
├── __main__.py           # CLI entry point
├── README.md             # This file
└── _src/
    ├── dataset.py        # TriviaDataset, TriviaItem, built-in questions
    ├── harness.py        # EvaluationHarness, metrics computation
    └── report.py         # HTML report generation
```

## Relationship to Other Modules

- **`adaptible.revise`** - Used internally to create training examples from corrections
- **`adaptible.autonomous`** - Provides online learning (eval is offline/controlled)
- **`adaptible.StatefulLLM`** - The model being evaluated

## Example Workflow

```bash
# 1. Quick test to verify setup
python -m adaptible.eval --subset 5

# 2. Test specific category with more training
python -m adaptible.eval --category science --iterations 50

# 3. Full evaluation with custom settings
python -m adaptible.eval \
  --name "full_eval_v1" \
  --shuffle \
  --iterations 100 \
  --output ~/results/eval_report.html

# 4. Save dataset for reproducibility
python -m adaptible.eval --save-dataset ~/datasets/trivia_v1.json

# 5. Rerun with same dataset
python -m adaptible.eval --load-dataset ~/datasets/trivia_v1.json
```
