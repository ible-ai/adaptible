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

### Revision quality (judged before training)

A valid revision is not necessarily a good one. For `self_generated` runs the harness also judges the revision itself, before training on it, so you can tell whether the loop has anything worth learning from separately from whether training absorbed it. Per item (`ItemResult`):

| Field | Meaning |
|---|---|
| `revision_answer` | The text between the `[[X]]` markers (`harness.extract_revision`), i.e. exactly what the loss mask covered |
| `revision_has_key_terms` | `contains_key_terms(revision_answer, key_terms)`: the revision's own verdict |
| `revision_changed_text` | `revision_answer != initial_response`; `False` means the model restated its baseline verbatim |
| `revision_changed_verdict` | `revision_has_key_terms != initial_has_key_terms` |

All four are `None` for `ground_truth` runs, holdout items, and invalid revisions. Over the trained items, `EvaluationResult` exposes `revision_attempted_count`, `revision_valid_count`, `revision_correct_count` (valid and has a key term), `revision_fixed_count` (baseline wrong, revision right), `revision_broke_count` (baseline right, revision wrong), `revision_unchanged_text_count`, and `revision_summary()` / `revision_summary_text()`. The harness prints the summary line at the end of a `self_generated` run and the report puts it in the header:

```text
Revisions: 84 attempted, 3 valid, of which 1 correct; fixed 1 wrong answers, broke 0 right ones (1 restated the baseline verbatim).
```

The `responses` table in `adaptible.db` has only `baseline` and `post_training` phases, so the revision is not stored there; it lives in the `EvaluationResult` (and the report's JSON dump) only.

### Revision prompt

With `self_generated`, the prompt that asks the model for its revision is chosen by `revision_prompt` (`EvaluationConfig.revision_prompt`, `MetaLearningConfig.revision_prompt`, `--revision_prompt` on the CLI). It is ignored for `ground_truth`. Presets come from `adaptible.revise.revision_prompt_preset`:

| Value | Instructions | Dialog rendering |
|---|---|---|
| `default` | `revise.REWRITE_INSTRUCTIONS`, the server's prompt | Through `tokenizer.apply_chat_template`, so the past dialog carries the model's special tokens |
| `fewshot` | `revise.REWRITE_INSTRUCTIONS_FEWSHOT`: short, imperative, with two worked `[[0]] ... [[/0]]` examples | Plain `User: ...` / `Assistant: ...` lines, no tokenizer involvement |

`fewshot` exists because the first `self_generated` run with `default` got a valid revision from the 1.5B model for only 1 of 84 items: most responses carried no `[[0]]` marker at all, and several started by echoing the chat-template tokens that `default` puts in the dialog. Whether `fewshot` does better has not been measured yet.

### Training target and `think_mode`

Whatever the revision prompt looked like, the training target is always built by `revise.make_revision_training_example` from the tokenizer's real chat template (`add_generation_prompt=True`), so training matches inference. For DeepSeek-R1-Distill that generation prompt ends with an open `<think>\n`, so every response has the form `{reasoning}</think>\n\n{answer}`. `think_mode` (`EvaluationConfig.think_mode`, `MetaLearningConfig.think_mode`, `--think_mode`; one of `revise.THINK_MODES`) says what the target does with that open block:

| `think_mode` | Sequence after the prefix `...<think>\n` | In the loss | Notes |
|---|---|---|---|
| `none` | `{revision}{eos}` | all of it | The original target: an answer inside an unclosed think block. One example collapsed responses from ~722 to ~16 tokens. Kept for comparison. |
| `empty` | `</think>\n\n{revision}{eos}` | all of it | Well-formed, but teaches "skip reasoning, answer in two words" as a global style. After 84 items (`gt_close`): trained 51/84 -> 73/84, holdout 12/21 -> 6/21, mean response 766 -> 6 tokens, every response opening with `</think>`. |
| `baseline` | `{baseline_think}\n</think>\n\n{revision}{eos}` | `{revision}{eos}` only | `baseline_think` is the model's own reasoning from its baseline response (`revise.split_think`). Teaches "given this reasoning, the answer is X" without touching the reasoning itself. Falls back to `empty` when the baseline had no reasoning. Probed 2026-09-09: the correction does not take. The target is "after reasoning that concludes Montreal, output Ottawa"; the teacher-forced loss reaches 0.57 and at inference the model re-derives Montreal. |
| `rationale` (default) | `{rationale}\n</think>\n\n{revision}{eos}` | all of it | The rationale is reasoning that concludes the revision: for `self_generated` the revision generation's own think block (`split_think(revision_raw)`); for `ground_truth` one extra generation with `harness.make_rationale_prompt` (`"{question}\n\nThe correct answer is: {answer}\nReason it through step by step, then state the answer."`), think block taken. Falls back to `empty` when that is empty, counted in `EvaluationResult.rationale_missing_count` / `SeedTrajectory.rationale_missing_count`; `ItemResult.rationale_text` and the report card carry the rationale used. |

Templates without a think tag are unaffected by `think_mode`. Both `training_source` values go through this path. `close_think` survives as a deprecated alias (`--noclose_think` is `--think_mode none`; after construction `config.close_think` is `think_mode != "none"`).

In every mode the example carries a `stop_mask` (`TrainingExample.stop_mask`) over exactly `{revision}{eos}`, and the loss target below is compared against the loss over those answer tokens, not the whole target. `_llm._loss_fn` returns `(train_loss, stop_loss)` from one forward pass (`mlx.core.value_and_grad` differentiates the first element of a tuple and passes the rest through); `TrainingStats.final_loss` is the answer loss and `TrainingStats.final_train_loss` the whole-target loss. Under `rationale` the two differ, and the whole-target loss is reported (`ItemResult.train_final_train_loss`) but never stops training.

### Rehearsal (`rehearsal_k`)

### Loss-targeted training

Every training call (`StatefulLLM.train_on_example`) runs single optimizer steps and stops as soon as a step's loss falls below `loss_target` (`EvaluationConfig.loss_target`, `MetaLearningConfig.loss_target`, `--loss_target`; default `0.6`), or after `training_iterations` steps (`--iterations`; default `12`, now a cap). A per-step probe on the real model with one correction (target "Ottawa"+eos, `think_mode=baseline`) started at loss 6.05; at a mean target loss of about 0.6 the greedy answer flipped to the correction while the reasoning stayed intact and an unrelated item was unaffected, and once the loss was driven below about 0.1 the model emitted the bare answer with no reasoning and answered "Ottawa" to unrelated questions. Every earlier run trained a fixed 5-25 iterations, which drives the loss to ~0; pass `--loss_target 0` (or `loss_target=None`) to reproduce that.

Per item, `ItemResult` records `train_steps`, `train_initial_loss`, `train_final_loss` (answer loss), `train_final_train_loss` (whole-target loss), `train_hit_cap`, and (with rehearsal) `train_rehearsal_final_loss`; the `training_events.training_iterations` column stores the steps the correction actually took. The verbose log prints `Trained (3 steps, loss 6.05 → 0.58 (train 0.80, rehearsal 0.41), 4.1s)` per item and the summary (and report header) carries `Training: mean 3.2 steps/item (cap 12, answer-loss target 0.60), mean final answer loss 0.55 (train loss 0.80, rehearsal 0.41); N items hit the cap; R rationales missing`; the `train loss` part appears only when some item reported a whole-target loss and the `rehearsal` part only when some item trained with rehearsal. Only the correction's answer loss is ever compared against the target. The server's `/trigger_review` path (`self_correct_and_train`) uses `StatefulLLM(loss_target=0.6, max_train_steps=12)`.

The `empty` run above also drifted unrelated facts ("The skin" for the largest planet). `rehearsal_k` (`EvaluationConfig.rehearsal_k`, `MetaLearningConfig.rehearsal_k`, `--rehearsal_k`; default 0) folds `k` self-distillation examples into every step of a correction: other *trained-split* items whose baseline answer was judged correct, with the model's own full baseline output (`{think}</think>\n\n{answer}{eos}`, whole target in the loss) as the target. The item is one `StatefulLLM.train_on_examples(correction, rehearsal, loss_target=..., max_steps=training_iterations, rehearsal_weight=...)` call. Each step computes the correction's gradient and every rehearsal example's gradient as separate single-sequence passes (never a padded batch; a `(1+k, L)` batch of rehearsal targets that ran to the generation cap exhausted a 16 GB machine), combines them as `grad(correction) + rehearsal_weight * mean(grad(rehearsal))` (`_llm.combine_grads`), and applies one optimizer update; training stops when the *correction* loss is below `loss_target` or at the step cap. `rehearsal_weight` (`EvaluationConfig.rehearsal_weight`, `MetaLearningConfig.rehearsal_weight`, `--rehearsal_weight`; default 1.0) scales the rehearsal term; `0` keeps the passes and the loss reporting but lets only the correction move the weights. With `rehearsal_k=0`, or an empty pool, the item is a plain `train_on_example` call.

This replaces the earlier scheme that trained each rehearsal example in its own `train_on_example` call after the correction. Under a loss target that did nothing: a rehearsal target's loss is the model's own output and already sits below 0.6, so every rehearsal call stopped after one tiny step. A 32-correction sequential run with k=4 rehearsal that way still collapsed (responses 718 → 21 tokens, 36/40 answers opening with an immediately closed think block, holdout 1/8 → 0/8). Whether the joint objective changes that has not been measured yet.

One training event is recorded per item. Rehearsal items are sampled with `seed + item index`, never include the item being corrected, and never include holdout items. `rehearsal_max_tokens` (`--rehearsal_max_tokens`; default 768) keeps items whose raw baseline response is longer than that out of the pool; if fewer than `k` items remain, the ones that do are used. `ItemResult.rehearsal_item_ids` records which items were used.

### LoRA capacity

`StatefulLLM` converts the last 24 layers to rank-32 LoRA with scale 10.0 by default, which is a lot of capacity for a target a few tokens long. Both CLIs expose `--lora_rank` (default 32), `--lora_layers` (default 24), and `--lora_scale` (default 10.0); `harness.lora_model_kwargs(rank, layers, scale)` turns them into `StatefulLLM(num_lora_layers=..., lora_parameters={"rank": ..., "dropout": 0.0, "scale": ...})` through the existing `model_kwargs` path. The values are recorded in the experiment's `config_json` (`model_kwargs` and a derived `lora` entry), on `EvaluationResult.model_kwargs` / `MetaLearningResult.model_kwargs`, and shown as `LoRA: rank 32, layers 24, scale 10` in the verbose header, the summary, and both reports. Runs recorded before these flags existed report the defaults, which is what they trained with.

### Collapse signals

`EvaluationResult.mean_baseline_tokens`, `mean_post_tokens`, and `post_empty_think_count` (post responses that open with `</think>`) are printed at the end of every run as

```
Response length: baseline 711 tok -> post 10 tok; empty-think responses after training: 105/105
Holdout accuracy: 28.6% (6/21, baseline 12/21)
```

so a run that "learned" its training items by giving up reasoning is obvious from the console.

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

# Specific category, higher step cap, custom report path, no browser
python -m adaptible.eval --category geography --iterations 20 \
    --output ~/eval.html --no_browser

# Fixed-count training (the pre-loss-target behaviour): 25 steps per item
python -m adaptible.eval --loss_target 0 --iterations 25
```

Flags are `absl.flags`: use underscores (`--train_ratio`, `--no_browser`), not dashes.

### Programmatic Usage

```python
import adaptible.eval as eval

dataset = eval.generate_default_dataset()

config = eval.EvaluationConfig(
    name="my_experiment",
    training_iterations=12,           # step cap per training call
    loss_target=0.6,                  # stop once a step's loss is below this; None = fixed count
    train_ratio=0.8,
    shuffle=True,
    training_source="ground_truth",   # or "self_generated"
    revision_prompt="default",        # or "fewshot"; only used by self_generated
    think_mode="rationale",           # or "baseline" / "empty" / "none"; see above
    rehearsal_k=0,                    # >0 folds self-distillation examples into each correction step
    rehearsal_max_tokens=768,         # rehearsal pool skips baselines longer than this
    rehearsal_weight=1.0,             # multiplier on the mean rehearsal gradient
)

from adaptible._src.eval.harness import lora_model_kwargs  # not re-exported from adaptible.eval yet

model_kwargs = {"learning_rate": 5e-5, **lora_model_kwargs(rank=8, layers=8, scale=10.0)}
harness = eval.EvaluationHarness(model_kwargs=model_kwargs)  # note: loads <outputs>/autonomous/checkpoint if present
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
| **Revision summary**       | `self_generated` only: attempted / valid / correct / fixed / broke counts of the revisions themselves, judged before training (see above) |

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
    training_iterations=12, loss_target=0.6, train_ratio=0.8,
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
| `--iterations`      | `12`                              | Step cap per training call (exact count if `--loss_target` is off) |
| `--loss_target`     | `0.6`                             | Stop a training call once a step's loss is below this; `0` or negative disables |
| `--shuffle`         | `False`                           | Randomize question order                                 |
| `--seed`            | `42`                              | Random seed for shuffling                                |
| `--training_source` | `ground_truth`                    | `ground_truth` or `self_generated` (see above)           |
| `--revision_prompt` | `default`                         | `default` or `fewshot`; revision prompt preset for `self_generated` |
| `--think_mode`      | `rationale`                       | `rationale`, `baseline`, `empty`, or `none` (see above)  |
| `--close_think`     | `None`                            | Deprecated; `--noclose_think` is `--think_mode none`      |
| `--rehearsal_k`     | `0`                               | Self-distillation examples folded into each correction step |
| `--rehearsal_max_tokens` | `768`                        | Baseline token cap for rehearsal-pool items              |
| `--rehearsal_weight` | `1.0`                            | Multiplier on the mean rehearsal gradient                |
| `--learning_rate`   | `None`                            | `StatefulLLM(learning_rate=...)`; model default if unset |
| `--lora_rank`       | `32`                              | LoRA rank                                                |
| `--lora_layers`     | `24`                              | Trailing layers converted to LoRA                        |
| `--lora_scale`      | `10.0`                            | LoRA scale                                               |
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
