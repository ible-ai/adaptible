# Adaptible

Stateful LLM serving instances that self-reflect and learn from their mistakes during idle time.

## Vision

Self-improvement is itself a learnable trait. Different model instances undergoing online learning arrive at different end states—some become stronger self-learners than others. Adaptible explores this phenomenon by creating multiple instances that self-update, releasing them into production, and pruning the weaker learners while propagating successful ones.

The core hypothesis: by diversifying training bets autonomously and greedily sampling from winners, we create an evolutionary bottleneck that selects for models adept at self-improvement. The goal isn't just a model that learns—it's discovering which models *learn to learn*.

## Status (September 2026)

Plainly, where the project stands:

- **The serving loop and the training code work.** `python -m adaptible.local` serves a 1.5B MLX model, records interactions, and `/trigger_review` runs the critique-rewrite-LoRA cycle in the background (fixed in 1.0.0a3; before that the coroutine was never awaited and no training happened).
- **Every number this repo has ever published is ground-truth supervised fine-tuning, not self-correction.** The eval harness and the meta-learning experiment train on `[[0]] {correct_answer} [[/0]]`, the dataset label, and measure how much of it the model absorbs. That is ordinary LoRA SFT on a known answer.
- **Self-correction is implemented but unmeasured.** `StatefulLLM.self_correct_and_train` (used by the server) and the `--training_source self_generated` path (added in 1.0.0a3) have never been run through an evaluation.
- **The meta-learning result is inconclusive.** Three seeds, 80 training items, and no control arm cannot separate seed effects from run-to-run noise. See [What has been measured](#what-has-been-measured) and `outputs/meta/analysis_report.md`.
- **The autonomous node's only run trained on 12 web-scraped claims, several of them boilerplate, and its belief-conflict path never fired.** Filters and a stricter default training policy were added since; see `adaptible/_src/autonomous/README.md`.

## What it does

Adaptible wraps an LLM in a server that:

1. Serves responses to user prompts
2. Stores interaction history (user and assistant turns; assistant turns were dropped before 1.0.0a3)
3. During idle periods, asks the model to critique and revise its past responses
4. Fine-tunes the model on those revisions using LoRA

## Requirements

- Python 3.13+
- Apple Silicon Mac (uses MLX for inference and training)

## Installation

```bash
pip install adaptible
```

Or from source:

```bash
git clone https://github.com/your-org/adaptible.git
cd adaptible
python -m venv .venv && .venv/bin/pip install -e .
```

Version `1.0.0a3`. The wheel is about 200 KB; `torch`, `lm-eval`, and `optax` are no longer dependencies, and `ddgs` replaces `duckduckgo-search`.

## Quick Start

### Run the server

```bash
python -m adaptible.local
```

This starts a FastAPI server at `http://127.0.0.1:8000`. The web UI is available at `/static/`.

### Programmatic usage

```python
import asyncio
import adaptible

async def main():
    server = adaptible.MutableHostedLLM(host="127.0.0.1", port=8000)
    await server.up()
    await asyncio.sleep(3600)   # server runs until you stop it
    await server.down()

asyncio.run(main())
```

### Direct model usage

```python
import adaptible

model = adaptible.StatefulLLM()
print(model.generate_response("What is the capital of France?"))
```

**Gotcha:** `StatefulLLM()` defaults `model_path` to `<outputs>/autonomous/checkpoint` and loads it if the directory exists. Any run after an autonomous run silently starts from those trained weights. Pass `model_path=None` for a fresh model.

## API Endpoints

| Endpoint           | Method | Description                                                              |
| ------------------ | ------ | ------------------------------------------------------------------------ |
| `/interact`        | POST   | Send a prompt, get a response                                            |
| `/stream_interact` | POST   | Stream the response                                                      |
| `/trigger_review`  | POST   | Start the self-correction cycle as an `asyncio` task (runs since 1.0.0a3) |
| `/sync`            | GET    | Await outstanding training tasks, then wait for `model.ok`               |
| `/history`         | GET    | Get all interactions                                                     |
| `/status`          | GET    | Health check                                                             |

```bash
curl -X POST http://127.0.0.1:8000/interact \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'

curl -X POST http://127.0.0.1:8000/trigger_review   # start self-correction
curl http://127.0.0.1:8000/sync                      # block until training is done
```

Details and a streaming client are in `adaptible/_src/local/README.md`.

## How the self-correction works

1. The model receives a prompt containing its past interactions
2. It's asked to identify a response that could be improved and rewrite it, labelled `[[i]] ... [[/i]]`
3. The rewrite is validated (format, length, not degenerate); invalid rewrites are skipped
4. The model is fine-tuned on the rewrite using LoRA
5. The loss mask is zero over the prompt and one over the rewrite, so only the rewrite is learned

This is the path the server uses. It is the path no evaluation has measured yet.

## Evaluation

`adaptible.eval` runs baseline inference on a trivia set, trains on a fraction of it, and re-infers everything. Since 1.0.0a3 it has a `--training_source` flag:

- `ground_truth` (default): train on the dataset label. Measures how well the model absorbs a supplied correction. **All published numbers use this.**
- `self_generated`: train on the model's own revision of its baseline answer. Measures self-correction. **Never run yet.**

```bash
python -m adaptible.eval --subset 20 --shuffle --no_browser --output /tmp/eval.html
python -m adaptible.eval --training_source self_generated --shuffle
```

The judge (`contains_key_terms`) is substring matching, NFKC-normalized and casefolded since 1.0.0a3. It is weak: a response can be graded correct for mentioning a key term in passing, or wrong for phrasing a correct answer differently.

Flags, metrics, dataset format, and the programmatic API are in `adaptible/_src/eval/README.md`.

## Meta-Learning Experiments

`scripts/run_meta_experiment.py` runs the eval per seed with periodic checkpoints and computes, per seed, `meta_learning_score = (late − early window improvement rate) + (early − late window forgetting rate)` over the first and last third of checkpoints. Since 1.0.0a3 the rates are *window* (marginal) rates over the items trained since the previous checkpoint, the score is `None` when any window has fewer than 5 items, `--repeats N` runs a seed N times with an identical shuffle and reports `within_seed_variance`, `across_seed_variance`, and `signal_to_noise`, and `--holdout_every_checkpoint` probes the holdout set at each checkpoint.

```bash
python scripts/run_meta_experiment.py --seeds 42,123,456 --output outputs/meta/run.json
```

See `adaptible/_src/eval/README.md` for definitions.

## What has been measured

All measurements used `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B`, LoRA rank 32 on the last 24 layers, 25 training iterations per item, and the ground-truth training source.

**Single eval runs** (`tests/artifacts/*.html`, README numbers before 1.0.0a3): baseline accuracy on the 105-item trivia set is roughly half. After direct SFT on the training split, more trained items are graded correct than were graded wrong afterwards. Holdout accuracy did not move, which is expected: the items are independent facts. These are results of fine-tuning on the answer key, and say nothing about the model's ability to correct itself.

**Meta-learning run** (2025-12-18, `outputs/meta/meta_experiment.json`, ~36 hours): seeds 42, 123, 456; 80 training items each; 8 cumulative checkpoints.

| Seed | Accuracy on the 80 *trained* items after training | Published score (cumulative rates) | Score under current code |
| ---- | ------------------------------------------------- | ---------------------------------- | ------------------------ |
| 42   | 58.8%                                             | +0.017                             | `None`                   |
| 123  | 57.5%                                             | +0.043                             | `None`                   |
| 456  | 53.8%                                             | −0.052                             | `None`                   |

What those numbers mean:

- 53–59% of the facts the model was directly fine-tuned on are graded correct afterwards. Holdout accuracy was computed and discarded, so generalization is unknown for this run.
- The published scores were built from cumulative checkpoints (the "late" item set contained the "early" set) and the early windows had 0/5, 1/4, and 1/7 improvements. A spread of −0.052..+0.043 (variance 0.0016) is inside binomial noise at n≈5.
- No seed was run twice, so none of the spread can be attributed to the seed rather than to sampling and training noise.
- Loading the shipped JSON with the current code gives `meta_learning_score=None` for every seed, reason `window(s) below 5 trained items`.

The full analysis with a revision note is `outputs/meta/analysis_report.md`.

**Autonomous node** (2025-12-08, `outputs/autonomous/state.json`): 12 training events, all with an empty prior belief; several claims were page boilerplate. No learning was measured. Details in `adaptible/_src/autonomous/README.md`.

## What would settle it

1. Measure self-correction instead of SFT:

   ```bash
   python -m adaptible.eval --training_source self_generated --shuffle --no_browser --output outputs/eval_self.html
   python scripts/run_meta_experiment.py --training_source self_generated --seeds 42,123,456
   ```

2. Get a noise floor: run one seed three times with the same shuffle and compare `within_seed_variance` with `across_seed_variance`:

   ```bash
   python scripts/run_meta_experiment.py --seeds 42 --repeats 3
   ```

3. Track generalization at every checkpoint, not only at the end:

   ```bash
   python scripts/run_meta_experiment.py --seeds 42,123,456 --holdout_every_checkpoint
   ```

Until (1) has been run there is no evidence about self-correction in this repository, and until (2) has been run there is no evidence that seeds differ.

## Experiment Database

Experiments are persisted to SQLite at `<outputs>/adaptible.db`, where `<outputs>` is `$ADAPTIBLE_OUTPUTS_DIR` if set, else `<cwd>/outputs` (`adaptible/_src/_paths.py`). Responses are stored raw and judged at query time, so grading can change without re-running inference.

```text
examples        canonical_id, question, ground_truth_answer, key_terms, category,
                difficulty, source_type (static_trivia | web_scrape), valid_at, created_at
experiments     name, experiment_type (eval | autonomous), config_json, model_checkpoint,
                started_at, completed_at
responses       example_id, experiment_id, response_text, response_raw,
                phase (baseline | post_training), token_count, max_tokens, truncated
training_events example_id, experiment_id, training_iterations, training_time_seconds
```

```bash
python scripts/explore_db.py            # REPL with helpers
python scripts/explore_db.py --demo     # create demo data
python scripts/explore_db.py --summary 1
```

`notebooks/explore_experiments.ipynb` uses `Database.compute_metrics`, `get_regressions`, `get_improvements`, `get_stuck`, and `export_experiment_summary`.

## Autonomous Learning

```bash
python -m adaptible.autonomous --cycles 3 --no_browser
```

Search → extract claims → ask the model what it believes → fact-check → train on contradicted beliefs → re-verify. State goes to `<outputs>/autonomous/state.json`, logs to `<outputs>/autonomous/logs/`, weights to `<outputs>/autonomous/checkpoint`. `NodeState.load` on the shipped `state.json` crashed before 1.0.0a3. Claim filtering, the training policy (`train_on_new_knowledge=False` by default), and the CLI flags are documented in `adaptible/_src/autonomous/README.md`.

## `StatefulLLM` parameters

| Parameter                        | Default                                       | Description                                       |
| -------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| `model_name`                     | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B` | HuggingFace model path                            |
| `model_path`                     | `<outputs>/autonomous/checkpoint`             | Loaded if it exists; `None` for a fresh model     |
| `learning_rate`                  | `5e-5`                                        | Training learning rate                            |
| `max_tokens`                     | `2048`                                        | Max tokens per response                           |
| `epochs`                         | `5`                                           | Training epochs per revision                      |
| `num_lora_layers`                | `24`                                          | Number of LoRA layers                             |
| `lora_parameters`                | `{"rank": 32, "dropout": 0.0, "scale": 10.0}` | LoRA config                                       |
| `loop_detection_sequence_length` | `8`                                           | Token sequence length for loop check              |
| `loop_detection_max_repetitions` | `3`                                           | Repetitions before stopping generation            |

## Limitations

- No measurement of self-correction exists yet; see [Status](#status-september-2026).
- The judge is substring matching against key terms.
- Apple Silicon only (MLX dependency).

## Project Structure

```text
.
├── adaptible/
│   ├── __init__.py              # Public API (re-exports from _src)
│   ├── eval/ local/ autonomous/ revise/   # Thin alias packages over _src/*; make
│   │                            #   `python -m adaptible.eval` etc. work
│   ├── _src/
│   │   ├── _api.py              # FastAPI routes (Adaptible, ModelProtocol)
│   │   ├── _classes.py          # Data models
│   │   ├── _llm.py              # StatefulLLM: generation, LoRA training, loop detection
│   │   ├── _paths.py            # $ADAPTIBLE_OUTPUTS_DIR / <cwd>/outputs resolution
│   │   ├── db.py                # SQLite layer and default_judge
│   │   ├── autonomous/          # AutonomousNode + CLI (README.md inside)
│   │   ├── eval/                # dataset.py, harness.py, meta.py, report.py + CLI (README.md inside)
│   │   ├── local/               # _server.py (MutableHostedLLM) + CLI (README.md inside)
│   │   ├── revise/              # revise.py, revise_test.py
│   │   ├── static/              # Web UI
│   │   └── dev/                 # Debug scripts, not in the test suite
│   └── tests/                   # Unit and integration tests
├── examples/                    # server_demo.py, online_learning_demo.py
├── notebooks/explore_experiments.ipynb
├── outputs/                     # Default $ADAPTIBLE_OUTPUTS_DIR; meta/ and autonomous/state.json are tracked
├── scripts/                     # run_meta_experiment.py, explore_db.py
└── pyproject.toml
```

## Running Tests

Fast, model-free (these run in GitHub Actions on macOS):

```bash
python -m unittest adaptible.tests.classes_test adaptible.tests.api_test \
    adaptible.tests.local_test adaptible.tests.paths_test adaptible.tests.eval_test \
    adaptible.tests.autonomous_test adaptible._src.revise.revise_test -v
```

Slow, downloads and trains a real model, mutates `<outputs>/autonomous/checkpoint`:

```bash
python -m unittest adaptible.tests.llm_test adaptible.tests.integration_test -v
```

Three classes in `llm_test` (`ConversationHistoryTest`, `TokenLoopDetectionTest`, `ValidationTest`) are model-free; the rest are not. Set `ADAPTIBLE_OUTPUTS_DIR` to a scratch directory to keep test state out of the repo.

## License

Contact for information.
