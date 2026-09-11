# Adaptible

A small language model that runs on your Mac, keeps what you say to it, and
retrains itself on its own corrections while idle.

Adaptible wraps an MLX model (`DeepSeek-R1-Distill-Qwen-1.5B` by default) in a
server that records every interaction, periodically asks the model to critique
and rewrite its past answers, and LoRA-fine-tunes the model on the rewrites.
Around that core it provides an evaluation harness to measure whether such
updates actually help, and a self-repair loop that keeps an update only when
the model's own generated answers get better.

Whether this works, and how far, is a research question. The experiments and
their data are in [`results/`](results/README.md).

## What you can do with it

| | Command |
|---|---|
| Serve the model with a web UI, record interactions, train on idle | `python -m adaptible.local` |
| Measure baseline accuracy, train on a split, re-measure | `python -m adaptible.eval --subset 20 --shuffle --no_browser` |
| Run the self-repair loop: generate, judge, keep or restore, repeat | `PYTHONPATH=. python scripts/cycles_mlx.py` |
| Same loop on a CUDA GPU (Colab notebook, resumes across sessions) | `scripts/colab/adaptible_cycles.ipynb` |
| Learn from live web search against the model's own beliefs | `python -m adaptible.autonomous --cycles 3 --no_browser` |
| Repeat the eval per seed with checkpoints and a noise floor | `python scripts/run_meta_experiment.py --seeds 42 --repeats 3` |

## Install

Python 3.13+ and an Apple Silicon Mac (MLX). The Colab script is the one
exception and needs only PyTorch.

```bash
pip install adaptible
# or from source
git clone https://github.com/ible-ai/adaptible && cd adaptible
python -m venv .venv && .venv/bin/pip install -e .
```

## Quick start

```bash
python -m adaptible.local
```

Starts a FastAPI server on `http://127.0.0.1:8000` with a chat UI at
`/static/`. Talk to it, then trigger a self-correction pass:

```bash
curl -X POST http://127.0.0.1:8000/interact -H "Content-Type: application/json" \
     -d '{"prompt": "What is the capital of Australia?"}'
curl -X POST http://127.0.0.1:8000/trigger_review   # critique, rewrite, train in the background
curl http://127.0.0.1:8000/sync                      # block until training is done
```

From Python:

```python
import adaptible

model = adaptible.StatefulLLM(model_path=None)   # fresh weights; see the gotcha below
print(model.generate_response("What is the capital of Australia?"))
```

Gotcha: `StatefulLLM()` loads `<outputs>/autonomous/checkpoint` if it exists,
so anything run after an autonomous session starts from those trained weights.
Pass `model_path=None` when you want the base model.

## How the pieces fit

- **`StatefulLLM`** (`adaptible/_src/_llm.py`) owns the model, tokenizer, and
  optimiser. Base weights are frozen; only LoRA adapters on the last N layers
  train. Generation has two loop breakers (repeated lines, repeated token
  sequences) because a 1.5B reasoner loops often enough to hang a run
  otherwise.
- **Revision** (`adaptible/_src/revise/`) turns a conversation into a training
  example: the model rewrites one of its turns, the rewrite is validated, and
  the loss mask covers only the rewrite, never the prompt.
- **Server** (`adaptible/_src/local/`) exposes `/interact`, `/stream_interact`,
  `/trigger_review`, `/sync`, `/history`, `/status`. Endpoints and a streaming
  client are in its README.
- **Evaluation** (`adaptible/_src/eval/`) runs baseline, trains, re-infers, and
  writes an HTML report. Every response is stored raw in SQLite
  (`<outputs>/adaptible.db`) and judged at query time, so grading can change
  without re-running inference. Flags and metrics are in its README.
- **Self-repair loop** (`scripts/cycles_mlx.py`, `scripts/colab/cycles_torch.py`)
  samples training targets from the model's own reasoning, trains a few steps,
  generates the answers again, and keeps the update only if they improved,
  restoring the weights otherwise. `scripts/cycles_results.py` turns a run log
  into the tables under `results/`.
- **Autonomous node** (`adaptible/_src/autonomous/`) searches the web, extracts
  claims, compares them with what the model believes, and trains on
  contradictions. Its README has the policy table.

`<outputs>` is `$ADAPTIBLE_OUTPUTS_DIR` if set, else `./outputs`. Everything
persisted lives there: the database, checkpoints, state, logs.

## Tests

Model-free, run in CI on macOS:

```bash
python -m unittest adaptible.tests.classes_test adaptible.tests.api_test \
    adaptible.tests.local_test adaptible.tests.paths_test adaptible.tests.eval_test \
    adaptible.tests.autonomous_test adaptible._src.revise.revise_test -v
```

`adaptible.tests.llm_test` and `adaptible.tests.integration_test` download and
train a real model; set `ADAPTIBLE_OUTPUTS_DIR` to a scratch directory first.

## Layout

```text
adaptible/                package; public API re-exported from adaptible/__init__.py
  _src/_llm.py            StatefulLLM: generation, LoRA training, loop detection
  _src/revise/            conversation -> training example, loss masks
  _src/eval/              dataset, harness, meta-learning, reports
  _src/local/             FastAPI server and web UI
  _src/autonomous/        web-search learning node
scripts/cycles_mlx.py     self-repair loop (MLX)
scripts/cycles_results.py run log -> results tables and plot
scripts/colab/            PyTorch port of the loop, Colab notebook
results/                  experiment report and data
```

## Limitations

- The judge is substring matching against key terms.
- Greedy decode on a small model flips on near-tie prompts under tiny weight
  changes; single-run differences of a couple of items are noise.
- Apple Silicon only, except the Colab script.

## License

Contact for information.
