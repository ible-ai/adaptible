# Evaluation Framework

Offline measurement of how much a `StatefulLLM` learns from a training pass over a trivia dataset.

## Overview

`EvaluationHarness.run`:

1. Records a baseline response to every question
2. Trains on a subset (default 80%), one item at a time
3. Re-infers every question post-training
4. Compares trained vs holdout items

Correctness is judged by `contains_key_terms`: substring matching of any key term against the response, with both sides NFKC-normalized and casefolded (since 1.0.0a3, so `H₂O` matches `H2O`). It is a weak judge. A response is graded correct if a key term appears anywhere in it, and wrong if a correct answer is phrased without the key term.

## Training source

What the model is trained on is controlled by `training_source` (`EvaluationConfig.training_source`, `MetaLearningConfig.training_source`, `--training_source` on the CLI). Added in 1.0.0a3.

| Value | Training target | What it measures |
|---|---|---|
| `ground_truth` (default) | `[[0]] {item.correct_answer} [[/0]]`, the dataset label | How well the model absorbs a correction it is handed. This is supervised LoRA fine-tuning. |
| `self_generated` | The model's own revision of its baseline answer, produced with `revise.make_revision_prompt` and checked by `validate_revision_response` | Self-correction, the path the server's `/trigger_review` uses |

Every published result from this repository used `ground_truth`. With `self_generated`, items whose revision fails validation are skipped (not trained) and counted in `EvaluationResult.revision_invalid_count`.

### Revision prompt

With `self_generated`, the prompt that asks the model for its revision is chosen by `revision_prompt` (`EvaluationConfig.revision_prompt`, `MetaLearningConfig.revision_prompt`, `--revision_prompt` on the CLI). It is ignored for `ground_truth`. Presets come from `adaptible.revise.revision_prompt_preset`:

| Value | Instructions | Dialog rendering |
|---|---|---|
| `default` | `revise.REWRITE_INSTRUCTIONS`, the server's prompt | Through `tokenizer.apply_chat_template`, so the past dialog carries the model's special tokens |
| `fewshot` | `revise.REWRITE_INSTRUCTIONS_FEWSHOT`: short, imperative, with two worked `[[0]] ... [[/0]]` examples | Plain `User: ...` / `Assistant: ...` lines, no tokenizer involvement |

`fewshot` exists because the first `self_generated` run with `default` got a valid revision from the 1.5B model for only 1 of 84 items: most responses carried no `[[0]]` marker at all, and several started by echoing the chat-template tokens that `default` puts in the dialog. Whether `fewshot` does better has not been measured yet.

### Training target and `close_think`

Whatever the revision prompt looked like, the training target is always built by `revise.make_collated_training_example` from the tokenizer's real chat template (`add_generation_prompt=True`), so training matches inference. For DeepSeek-R1-Distill that generation prompt ends with an open `<think>\n`, and the original code put the revision straight after it, training the model on an answer inside an unclosed think block. One such example was enough to collapse responses from ~722 to ~16 tokens.

`close_think` (`EvaluationConfig.close_think`, `MetaLearningConfig.close_think`, `--close_think` / `--noclose_think`; default on) fixes this: when the prefix ends with `<think>`, the target becomes `<think>\n</think>\n\n{revision}{eos}`, an empty reasoning block followed by the answer, with the loss mask covering `</think>\n\n` plus the revision. `--noclose_think` reproduces the old target for comparison. Templates without a think tag are unaffected either way. Both `training_source` values go through this path.

## Quick Start

### Command Line

```bash
# Full evaluation (105 questions), ground-truth SFT
python -m adaptible.eval

# Quick test with 20 questions
python -m adaptible.eval --subset 20 --shuffle

# Measure self-correction rather than SFT
python -m adaptible.eval --training_source self_generated --shuffle

# Same, with the few-shot revision prompt
python -m adaptible.eval --training_source self_generated --revision_prompt fewshot --shuffle

# Specific category, more iterations, custom report path, no browser
python -m adaptible.eval --category geography --iterations 50 \
    --output ~/eval.html --no_browser
```

Flags are `absl.flags`: use underscores (`--train_ratio`, `--no_browser`), not dashes.

### Programmatic Usage

```python
import adaptible.eval as eval

dataset = eval.generate_default_dataset()

config = eval.EvaluationConfig(
    name="my_experiment",
    training_iterations=25,
    train_ratio=0.8,
    shuffle=True,
    training_source="ground_truth",   # or "self_generated"
    revision_prompt="default",        # or "fewshot"; only used by self_generated
    close_think=True,                 # False reproduces the pre-fix training target
)

harness = eval.EvaluationHarness()    # note: loads <outputs>/autonomous/checkpoint if present
result = harness.run(dataset, config, verbose=True)

eval.generate_html_report(result, "/tmp/report.html")

print(f"Baseline accuracy: {result.baseline_accuracy:.1%}")
print(f"Train improvement: {result.train_improvement_rate:.1%}")
print(f"Holdout accuracy: {result.holdout_accuracy:.1%}")
```

`EvaluationHarness()` builds a `StatefulLLM()` with the default `model_path`, which loads `<outputs>/autonomous/checkpoint` if that directory exists. Pass `EvaluationHarness(model=adaptible.StatefulLLM(model_path=None))` for a clean baseline.

## Dataset

The built-in dataset has 105 trivia questions across 6 categories:

- **Geography** (20) - Capitals, landmarks, physical geography
- **Science** (25) - Physics, chemistry, biology, astronomy
- **History** (20) - Inventions, inventors, major events
- **Math** (15) - Basic calculations, constants, geometry
- **Language** (10) - Linguistics, writing systems
- **Miscellaneous** (15) - Animals, records, culture

Each item has a question, correct answer, key terms for grading, and a difficulty (easy/medium/hard).

### Custom Datasets

```python
from adaptible.eval import TriviaDataset, TriviaItem, save_dataset, load_dataset

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
    ],
)
save_dataset(dataset, "my_dataset.json")
dataset = load_dataset("my_dataset.json")
```

## Metrics (single evaluation)

| Metric                     | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| **Baseline Accuracy**      | % of all items with key terms before training                 |
| **Train Post-Accuracy**    | % of trained items correct after training                     |
| **Train Improvement Rate** | % of trained items that were wrong at baseline and are right after |
| **Train Retention Rate**   | % of trained items that were right at baseline and stayed right |
| **Holdout Accuracy**       | % of untrained items correct after training                   |
| **Revision Invalid Count** | `self_generated` only: items skipped because the revision failed validation |

All results are also written to `<outputs>/adaptible.db` (see `adaptible/_src/db.py`), where responses are stored raw and re-judged at query time.

## Meta-learning experiments (`meta.py`)

`MetaLearningExperiment.run(dataset, MetaLearningConfig)` runs the harness once per seed (each seed with a fresh model, `model_path=None`) and records a `Checkpoint` every `checkpoint_interval` training items. Driven by `scripts/run_meta_experiment.py`.

### Cumulative vs window metrics

Each `Checkpoint` carries two sets of counts:

- **Cumulative** (`improved`, `retained`, `regressed`, `stuck`, `improvement_rate`, `forgetting_rate`, `post_accuracy`): every item trained so far, re-inferred at this checkpoint. Later checkpoints are supersets of earlier ones.
- **Window** (`window_improved`, `window_retained`, `window_regressed`, `window_stuck`, `window_size`, `window_ids`): only the items trained since the previous checkpoint. Windows are disjoint.

  - `window_improvement_rate = window_improved / (window_improved + window_stuck)`
  - `window_forgetting_rate = window_regressed / (window_retained + window_regressed)`

### Meta-learning score

```
meta_learning_score = (late_improvement − early_improvement) + (early_forgetting − late_forgetting)
```

where each rate is the **window** rate averaged over the first third (`early`) and last third (`late`) of checkpoints. Cumulative rates are not used because "late" would contain "early" and the difference shrinks toward zero regardless of what the model does.

**`None` rule.** `SeedTrajectory.meta_learning_score` is `None`, with the reason in `meta_learning_score_reason`, when there are fewer than 3 checkpoints or when any checkpoint in the early or late third has `window_size < MIN_WINDOW_ITEMS` (5). Loading the shipped `outputs/meta/meta_experiment.json` (written before window counts existed) yields `None` for every seed for this reason.

### Noise control: `--repeats`

`MetaLearningConfig.repeats = N` runs each seed N times with an identical shuffle. Differences between repeats of one seed are generation and training noise, not seed effects. `MetaLearningResult` then reports:

- `within_seed_variance`: mean over seeds of the variance of the score across that seed's repeats
- `across_seed_variance`: variance across seeds of each seed's mean score over its repeats
- `signal_to_noise`: `across_seed_variance / within_seed_variance`

`within_seed_variance` is `None` unless some seed has 2+ scored repeats; `across_seed_variance` is `None` with fewer than 2 scored seeds; `signal_to_noise` is `None` if either is undefined or the within-seed variance is zero. A per-seed score is only meaningful when `across_seed_variance` is well above `within_seed_variance`.

### Holdout

`SeedTrajectory.holdout_accuracy` is the final holdout accuracy (stored since 1.0.0a3; earlier runs discarded it). With `holdout_every_checkpoint=True` / `--holdout_every_checkpoint`, each `Checkpoint` also gets `holdout_correct`, `holdout_total`, `holdout_accuracy`, at the cost of one holdout-sized inference pass per checkpoint. `final_trained_accuracy` is accuracy on trained items only and is not a generalization number.

### Running

```bash
python scripts/run_meta_experiment.py --seeds 42,123,456
python scripts/run_meta_experiment.py --seeds 42 --repeats 3                    # noise floor
python scripts/run_meta_experiment.py --training_source self_generated          # self-correction
python scripts/run_meta_experiment.py --training_source self_generated --revision_prompt fewshot
python scripts/run_meta_experiment.py --holdout_every_checkpoint --subset 40
```

Results are saved to `outputs/meta/<name>.json` (or `--output`) plus an HTML summary alongside.

```python
import adaptible.eval as eval

config = eval.MetaLearningConfig(
    name="meta", seeds=[42, 123], checkpoint_interval=10,
    training_iterations=25, train_ratio=0.8,
    training_source="ground_truth", repeats=1, holdout_every_checkpoint=False,
)
result = eval.MetaLearningExperiment().run(eval.generate_default_dataset(), config)
for seed, traj in result.trajectories.items():
    print(seed, traj.meta_learning_score, traj.meta_learning_score_reason)
print(result.within_seed_variance, result.across_seed_variance, result.signal_to_noise)
result.save("outputs/meta/meta.json")
result = eval.MetaLearningResult.load("outputs/meta/meta.json")
```

## CLI Options (`python -m adaptible.eval`)

| Flag                | Default                           | Description                                              |
| ------------------- | --------------------------------- | -------------------------------------------------------- |
| `--name`            | `"default"`                       | Experiment name                                          |
| `--train_ratio`     | `0.8`                             | Fraction for training (rest is holdout)                  |
| `--iterations`      | `25`                              | Training iterations per example                          |
| `--shuffle`         | `False`                           | Randomize question order                                 |
| `--seed`            | `42`                              | Random seed for shuffling                                |
| `--training_source` | `ground_truth`                    | `ground_truth` or `self_generated` (see above)           |
| `--revision_prompt` | `default`                         | `default` or `fewshot`; revision prompt preset for `self_generated` |
| `--close_think`     | `True`                            | Close an open `<think>` block before the training target; `--noclose_think` for the old target |
| `--subset`          | `None`                            | Use only first N questions                               |
| `--category`        | `None`                            | Filter to specific category                              |
| `--output`          | `/tmp/adaptible_eval_report.html` | Report path                                              |
| `--save_dataset`    | `None`                            | Save dataset to JSON                                     |
| `--load_dataset`    | `None`                            | Load custom dataset from JSON                            |
| `--no_browser`      | `False`                           | Don't auto-open report in browser                        |

`scripts/run_meta_experiment.py` adds `--seeds`, `--checkpoint_interval`, `--repeats`, `--holdout_every_checkpoint`, and defaults `--output` to `outputs/meta/<name>.json`.

## Files

```text
adaptible/_src/eval/
├── __init__.py           # Public API (re-exported as adaptible.eval)
├── __main__.py           # CLI entry point (absl flags)
├── README.md             # This file
├── dataset.py            # TriviaDataset, TriviaItem, built-in questions
├── harness.py            # EvaluationHarness, contains_key_terms, training-source logic
├── meta.py               # Checkpoint, SeedTrajectory, MetaLearningExperiment
└── report.py             # HTML report generation

adaptible/eval/           # Alias package so `python -m adaptible.eval` works
adaptible/tests/eval_test.py   # Model-free tests
scripts/run_meta_experiment.py
```

## Relationship to Other Modules

- **`adaptible.revise`** - Builds the collated training example (and, for `self_generated`, the revision prompt)
- **`adaptible.autonomous`** - Online learning from web claims; eval is offline and controlled
- **`adaptible.StatefulLLM`** - The model being evaluated
