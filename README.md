# Adaptible

Can a small language model running on a laptop repair its own factual errors,
using only its own reasoning as the training signal, without falling apart?

Adaptible is the harness for asking that question: an MLX server that stores
its interactions and LoRA-trains on its own rewrites during idle time, an
evaluation harness with a results database, and a self-repair loop that
generates, judges, keeps or restores, and repeats.

## Why a small local model

The model is `DeepSeek-R1-Distill-Qwen-1.5B`, bf16, on a 16 GB M3 laptop at
27 tokens per second. It is underpowered for this on purpose. A frontier model
could absorb a correction and generalise it; that would tell us little. A 1.5B
distilled reasoner gets basic capitals wrong, loops in its own chain of thought,
and has parameters so entangled that one update moves dozens of unrelated facts.
If anything resembling self-teaching shows up here, it shows up under the worst
conditions, and the failure modes are visible instead of hidden behind capacity.

The bar is therefore not "did it become smart". It is: over repeated cycles of
self-repair, does accuracy on held-out phrasings go up, does the model stay
coherent, and where does it break.

## The result

Five facts the model gets wrong at baseline (capitals of Morocco, Turkey,
Australia, the Philippines; the nearest star), each tested on its original
question plus three hand-written paraphrases: 20 prompts. Each cycle, every item
still under 4/4 gets up to two candidate training targets sampled from the
model's own reasoning (temperature 0.7, with a one-line reference note giving
the answer), is trained at most 4 LoRA steps toward an answer-token loss of
0.15, and the update is kept only if that item's 4-prompt score rises, otherwise
the weights are restored from a checksummed copy. Greedy decode, substring
judge, no rehearsal, no cross-item guard.

![score per cycle](results/self-repair-cycles-2026-09-11-mlx/score.svg)

| Cycle | 0 | 5 | 10 | 16 | 21 | 26 | 31 |
|---|---|---|---|---|---|---|---|
| Correct prompts, of 20 | 8 | 11 | 12 | **16** | 10 | **19** | 7 |
| Generations with no answer | 3 | 2 | 4 | 0 | 10 | 1 | 13 |

Baseline before any training: 4 of 20.

What it shows:

1. **Lift.** From 4 to 16 over 16 cycles with the loop count falling to zero,
   then to 19 at cycle 26. Four of the five facts reached 4/4 and held for
   multiple cycles. The filter does the work: 53 of 209 candidates were kept,
   and most rejected candidates made their own item worse.
2. **Drift, with a mechanism.** From about cycle 17 the model's samples converge
   on the phrasing of the prompt they are sampled from ("The correct answer
   is..."), the answer-token loss on its own samples reaches zero, and the
   kept updates become tie-break noise. Scores then swing between 7 and 19
   with loops rising.
3. **No nosedive.** Through 31 cycles the model never collapsed to the bare-word
   or empty-think regimes that whole-target fine-tuning produced in earlier
   experiments. It stays a reasoning model that answers most prompts.
4. **One fact never took.** Turkey trains to zero loss under its own sample's
   reasoning and still says Istanbul when it thinks freely.

Caveats, stated plainly: five items, one model, greedy decode with a noise
floor of about two points, a substring judge, and the candidates were sampled
with the correct answer in the prompt, so the facts came from outside; the
reasoning that carried them into the weights is the model's own.

Everything behind the plot is in
[`results/self-repair-cycles-2026-09-11-mlx/`](results/self-repair-cycles-2026-09-11-mlx/):
a per-cycle table with per-item marks, a per-candidate table with loss, marks,
and keep/restore, and the exact configuration. A second run of the same loop
in PyTorch on a Colab T4 is in progress and will be added alongside.

## Reproduce

Mac, MLX (about 25 minutes per cycle):

```bash
python -m venv .venv && .venv/bin/pip install -e .
CYCLES=40 PYTHONPATH=. .venv/bin/python scripts/cycles_mlx.py | tee cycles.log
.venv/bin/python scripts/cycles_results.py --log cycles.log --out results/my-run
```

Colab, PyTorch (free T4, resumes across sessions): open
[`scripts/colab/adaptible_cycles.ipynb`](scripts/colab/adaptible_cycles.ipynb)
in Colab and Run all. Details in [`scripts/colab/README.md`](scripts/colab/README.md).

## What else is here

- **Server.** `python -m adaptible.local` serves the model with a web UI at
  `/static/`, records interactions, and `/trigger_review` runs the
  critique-rewrite-LoRA cycle in the background. Endpoints and a streaming
  client: `adaptible/_src/local/README.md`.
- **Evaluation harness.** `python -m adaptible.eval` runs baseline, trains on a
  split, re-infers everything, and writes an HTML report. Responses are stored
  raw in SQLite and judged at query time. Flags and metrics:
  `adaptible/_src/eval/README.md`.
- **Meta-learning experiment.** `scripts/run_meta_experiment.py` runs the eval
  per seed with checkpoints and a noise-floor option (`--repeats`).
- **Autonomous node.** `python -m adaptible.autonomous` searches the web,
  extracts claims, checks them against the model's beliefs, and trains on
  contradictions. Policy table: `adaptible/_src/autonomous/README.md`.

Earlier measurements with these tools, for the record: fine-tuning directly on
dataset labels (rank-32 LoRA, 25 steps per item) taught 53 to 59 percent of
trained facts with no held-out movement; the server's own revision prompt was
followed by the 1.5B model in 1 of 84 cases; and a three-seed meta-learning run
was inside binomial noise. Those runs motivated the loop above and are not
evidence for it.

## Requirements and install

Python 3.13+, Apple Silicon (MLX) for everything except the Colab script.

```bash
pip install adaptible            # or, from source:
python -m venv .venv && .venv/bin/pip install -e .
```

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
adaptible/            package; public API re-exported from adaptible/__init__.py
  _src/_llm.py        StatefulLLM: generation, LoRA training, loop detection
  _src/revise/        prompt -> training example, loss masks (revise_test.py guards alignment)
  _src/eval/          dataset, harness, meta-learning, reports
  _src/local/         FastAPI server
  _src/autonomous/    web-search learning node
scripts/cycles_mlx.py           the self-repair loop (MLX)
scripts/cycles_results.py       log -> results/ tables and plot
scripts/colab/                  PyTorch port of the loop + notebook
results/                        flagship results, parsable
```

## Limitations

- Substring judge against key terms; a correct answer phrased differently is
  graded wrong, a term mentioned in passing is graded right.
- Greedy decode on a small model flips on near-tie prompts under weight changes
  far below what training makes; treat single-cycle differences under about
  two points as noise.
- Apple Silicon only for the MLX paths.

## License

Contact for information.
