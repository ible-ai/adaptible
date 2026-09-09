# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Adaptible wraps a small MLX LLM in a server that stores its interactions, asks the model to
critique and rewrite its own past responses during idle time, and LoRA-fine-tunes on those
rewrites. The evaluation and meta-learning modules are meant to measure whether that loop
produces net learning across seeds.

Where things stand (September 2026): the serving loop and training code work; every published
number is from ground-truth supervised fine-tuning, not self-correction; the self-correction
path is implemented but has never been measured; the 3-seed meta-learning result is
inconclusive. The README "Status" section and `outputs/meta/analysis_report.md` carry the
details. Do not write documentation or commit messages that claim more than that.

Apple Silicon only — `mlx` is a hard dependency of every code path that touches the model.
Requires Python 3.13+. Version `1.0.0a3`.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e .   # editable install; use .venv/bin/python below

# Fast, model-free tests (what CI runs, on macos-latest)
python -m unittest adaptible.tests.classes_test adaptible.tests.api_test \
    adaptible.tests.local_test adaptible.tests.paths_test adaptible.tests.eval_test \
    adaptible.tests.autonomous_test adaptible._src.revise.revise_test -v
python -m unittest adaptible.tests.api_test.InteractEndpointTest.test_x -v   # one test

# Slow: download + train a real model, mutate <outputs>/autonomous/checkpoint
python -m unittest adaptible.tests.llm_test adaptible.tests.integration_test -v

# Run things
python -m adaptible.local                                  # FastAPI server + web UI at /static/
python -m adaptible.eval --subset 20 --shuffle --no_browser # offline eval, HTML report
python -m adaptible.eval --training_source self_generated   # measure self-correction (run once: 1/84 valid revisions)
python -m adaptible.autonomous --cycles 3 --no_browser      # online learning against live web search
python scripts/run_meta_experiment.py --seeds 42,123 --subset 10
python scripts/run_meta_experiment.py --seeds 42 --repeats 3 --holdout_every_checkpoint
python scripts/explore_db.py           # REPL over <outputs>/adaptible.db (--demo, --summary N)
```

`python -m adaptible.eval` and `scripts/run_meta_experiment.py` train on ground truth by default;
pass `--training_source self_generated` to measure actual self-correction.

Test cost is very uneven. `classes_test`, `api_test`, `local_test`, `paths_test`, `eval_test`,
`autonomous_test`, and `revise_test` are model-free (`api_test` uses a `StubModel`), as are three
classes in `llm_test` (`ConversationHistoryTest`, `TokenLoopDetectionTest`, `ValidationTest`).
The rest of `llm_test` and all of `integration_test` load and *train* a real model — minutes per
run. Set `ADAPTIBLE_OUTPUTS_DIR` to a scratch directory when running them, as CI does.

CLI flags are `absl.flags`, not argparse (underscores: `--no_browser`, `--train_ratio`), in
`adaptible/_src/eval/__main__.py`, `adaptible/_src/autonomous/__main__.py`, and
`scripts/run_meta_experiment.py`.

Formatting is black (see `.vscode/settings.json`); there is no configured linter.

## Architecture

### The learning loop

`StatefulLLM` (`adaptible/_src/_llm.py`) owns the model, tokenizer, optimizer, and all training.
On construction it freezes the base weights and converts the last `num_lora_layers` linear layers
to LoRA — **only LoRA parameters ever receive gradients**.

`adaptible/_src/revise/revise.py` is the glue that turns a conversation into a training example:

1. `make_revision_prompt` serializes past turns and asks the model to rewrite one, labelling it
   `[[X]] ... [[/X]]`.
2. `validate_revision_response` rejects malformed or degenerate rewrites by raising
   `InvalidRevisionError` — callers are expected to catch this and skip the training step.
3. `make_collated_training_example` re-tokenizes `chat_template(user_turn) + rewritten_answer`
   and builds a **loss mask that is zero over the prompt and one over the revision**. This mask
   is why the model is nudged toward the revision without being retrained on its own prompt.
   The input/label/mask alignment (`seq[:-1]`, `seq[1:]`, `mask[1:]`) is easy to break; the
   corresponding tests in `revise_test.py` are the guard.

`_loss_fn` uses `reduction="none"` deliberately — a mean reduction broadcasts the same scalar
across every masked position and silently defeats the mask.

Generation has two independent loop breakers: repeated identical *lines*, and
`_detect_token_loop` over repeated *token* sequences. Small distilled models loop often enough
that removing these makes evaluation runs hang.

### Two training sources

`eval/harness.py::_build_training_example` builds the revision either from the label
(`training_source="ground_truth"`: `[[0]] {item.correct_answer} [[/0]]`) or from the model's own
rewrite of its baseline answer (`"self_generated"`, through `make_revision_prompt` +
`validate_revision_response`). The server's `/trigger_review` uses `StatefulLLM.self_correct_and_train`,
which is the self-generated path. Only the ground-truth path has ever produced numbers.

### Paths and the checkpoint gotcha

`adaptible/_src/_paths.py` resolves `<outputs>` as `$ADAPTIBLE_OUTPUTS_DIR` if set, else
`<cwd>/outputs`; never from `__file__`. Everything persisted lives there: `adaptible.db`,
`autonomous/checkpoint`, `autonomous/state.json`, `autonomous/logs/`.

`StatefulLLM.__init__` defaults `model_path=MODEL_PATH` (`<outputs>/autonomous/checkpoint`) and
**loads from it if the directory exists**. So an eval or server started after any autonomous run
silently continues from the autonomous node's trained weights. `MetaLearningExperiment` avoids
this with a `model_factory` defaulting to `StatefulLLM(model_path=None)`; `EvaluationHarness()`
and `Adaptible()` do not. Pass `model_path=None` (or a `model=`) whenever a clean baseline matters.

### Serving

`Adaptible` (`_api.py`) builds the FastAPI app around any object satisfying `ModelProtocol`, so
tests can inject a stub. It keeps interaction history in memory and tracks unreviewed indices.
`/trigger_review` clears those indices and schedules `self_correct_and_train` with
`asyncio.create_task(asyncio.to_thread(...))`, appending the task to `outstanding_tasks`; `/sync`
awaits those tasks (logging failures rather than raising), then polls `model.ok` (which is `False`
during backprop) until it stabilizes. `MutableHostedLLM` (`local/_server.py`) is a
`uvicorn.Server` subclass adding awaitable `up()`/`down()`.

### Experiments and persistence

`db.py` is a plain-sqlite layer shared by both experiment types (`ExperimentType.EVAL` and
`AUTONOMOUS`) writing to `<outputs>/adaptible.db`. Responses are stored raw and correctness is
**judged at query time** by a pluggable `Judge` (`default_judge` matches key terms), so grading can
be revised without re-running inference. `examples` are unique on `(canonical_id, valid_at)`, where
`valid_at` is NULL for timeless trivia and a date for time-sensitive web-scraped claims.
`compute_metrics`, `get_regressions`, `get_improvements`, and `get_stuck` are the analysis entry
points used by the notebook.

`eval/harness.py` runs baseline → train on `train_ratio` of items → re-inference on everything.
`contains_key_terms` is the judge: substring match, NFKC-normalized and casefolded.

`eval/meta.py` runs that per seed with periodic checkpoints. Each `Checkpoint` has cumulative
counts (every item trained so far) and **window** counts (only items trained since the previous
checkpoint). `meta_learning_score = (late − early window improvement rate) + (early − late window
forgetting rate)` over the first and last third of checkpoints; it is `None` (see
`meta_learning_score_reason`) with fewer than 3 checkpoints or any window under
`MIN_WINDOW_ITEMS = 5`. `repeats=N` reruns a seed with the same shuffle so
`MetaLearningResult.within_seed_variance` / `across_seed_variance` / `signal_to_noise` give a noise
floor. `holdout_every_checkpoint` probes the holdout set per checkpoint; final holdout accuracy is
stored on the `SeedTrajectory`. `final_trained_accuracy` is trained-items-only.

`autonomous/node.py` is the online counterpart: search → filter claims (`_claim_is_plausible`) →
ask the model what it believes → fact-check → train only on contradicted beliefs by default
(`train_on_new_knowledge=False`) → re-verify, persisting `NodeState` to
`<outputs>/autonomous/state.json`. `adaptible/_src/autonomous/README.md` has the policy table.

### Layout convention

Everything public is re-exported from `adaptible/__init__.py`; implementation lives under
`adaptible/_src/`. `adaptible/{eval,local,autonomous,revise}/` are thin alias packages
(`from .._src.X import *` plus a `__main__.py`) so `python -m adaptible.eval` etc. work. Import
from the package root or those aliases rather than reaching into `_src` directly. Tests are in
`adaptible/tests/` (plus `_src/revise/revise_test.py`). Each of `_src/{eval,local,autonomous}/`
has its own README.

`adaptible/_src/dev/` holds debugging scripts, not part of the test suite and excluded from the
wheel.

## Known rough edges

- The judge is substring matching against key terms. Items can be graded wrong for phrasing a
  correct answer differently, or right for mentioning a term in passing.
- `make_revision_prompt` takes a `tokenizer` argument it does not use.
- The eval and autonomous entry points open a browser and write reports to `/tmp` by default;
  pass `--no_browser` / `--output` in non-interactive runs.
