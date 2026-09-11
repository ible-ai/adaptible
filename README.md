# Adaptible

A language model that remembers what you told it, notices when it was wrong,
and retrains itself, on hardware you own.

## What it is

Most deployed language models are frozen. They answer, forget the exchange,
and answer the same way tomorrow. Adaptible is an experiment in the opposite
arrangement: a small reasoning model (`DeepSeek-R1-Distill-Qwen-1.5B`) that
keeps its conversations, revisits them when idle, and changes its own weights
based on what it finds. It runs on a laptop, on Apple Silicon through MLX, and
its self-repair loop also runs on a CUDA GPU through PyTorch.

The model is deliberately small. A frontier model corrected once would simply
absorb the correction; that would show nothing about the mechanism. A 1.5B
distilled reasoner gets basic facts wrong, loops in its own chain of thought,
and has weights so entangled that one update moves unrelated facts. Every
failure mode of self-training is visible at this scale, which is what makes
lift, when it appears, informative.

## How it works

**Serve and remember.** The server answers prompts and stores every turn, user
and assistant, in memory and in a SQLite database. Nothing is thrown away, so a
mistake made at noon is still available to be examined at midnight.

**Reflect.** When asked (or on a schedule), the model is shown its past
exchanges and asked to pick a response it could improve and rewrite it. The
rewrite is checked for shape and sanity: it must address the right turn, be a
plausible length, and not be degenerate. Rewrites that fail are discarded
rather than trained on.

**Train on the rewrite only.** Each accepted rewrite becomes one training
example: the original prompt followed by the new answer, with a loss mask that
is zero over the prompt and one over the rewrite. The model is nudged toward
the answer it now prefers, never toward its own prompt. Only LoRA adapters on
the last few layers receive gradients; the base weights stay frozen, so an
update is small, cheap, and reversible.

**Keep it only if it helped.** This is the part that makes the loop work on a
model this small. Small models produce training targets that are wrong, or
right for the wrong reasons, more often than not, and a bad update can wreck
facts the model already had. So the loop generates candidate targets from the
model's own reasoning, trains a few steps, regenerates the answers to the
original question and to paraphrases of it, and keeps the new weights only if
the answers actually improved. Otherwise the weights are restored from a copy
and the next candidate is tried. Every update that survives has been checked
against what the model says, not against a loss curve.

**Stay coherent.** Reasoning models this size loop: they repeat a sentence,
or circle inside their chain of thought and never answer. Generation carries
two loop breakers, and the judge counts a generation that never reaches an
answer as wrong, so the loop cannot be satisfied by a model that has stopped
answering.

## What we found

Running the self-repair loop on five facts the model gets wrong, each tested
on its original phrasing and three paraphrases, accuracy on the twenty prompts
rose from 4 to 16 over sixteen cycles and reached 19 by cycle 26, with the
model staying coherent throughout. After that the samples it draws from itself
converge on a single phrasing, the updates shrink to nothing, and the score
drifts. One of the five facts never took. The full write-up, with the data
behind every number, is in [`results/`](results/README.md).

## Demo

The browser UI at `/static/` shows the loop end to end. The interaction looks
like this:

1. Ask the model a question it gets wrong. `What is the capital of Australia?`
   The 1.5B model reasons for a paragraph and answers Sydney.
2. Correct it in the conversation. `No. The capital is Canberra; Sydney is the
   largest city.`
3. Trigger a review. The model rereads the exchange, rewrites its earlier
   answer, and trains on the rewrite in the background. `/sync` returns when
   training has finished.
4. Ask again, in a new conversation, in different words. `Which city is
   Australia's capital?` and see whether the answer, and the reasoning that
   leads to it, changed.

Whether step 4 comes out right depends on whether the model produced a usable
rewrite in step 3, which at this size it often does not; the self-repair loop
exists to make that step reliable. TODO: screen recording of this sequence.

## Run it

Python 3.13+. Apple Silicon for the server and evaluation harness; any CUDA
GPU for the self-repair loop through `scripts/colab/`.

```bash
git clone https://github.com/ible-ai/adaptible && cd adaptible
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m adaptible.local                      # server + UI at http://127.0.0.1:8000/static/
PYTHONPATH=. .venv/bin/python scripts/cycles_mlx.py      # the self-repair loop
.venv/bin/python -m adaptible.eval --subset 20 --shuffle --no_browser   # baseline / train / re-measure
```

The Colab notebook `scripts/colab/adaptible_cycles.ipynb` runs the loop on a
free T4 and resumes across sessions. Module READMEs under `adaptible/_src/`
document endpoints, flags, and metrics. Model-free tests:

```bash
python -m unittest adaptible.tests.classes_test adaptible.tests.api_test \
    adaptible.tests.local_test adaptible.tests.paths_test adaptible.tests.eval_test \
    adaptible.tests.autonomous_test adaptible._src.revise.revise_test
```

## Layout

```text
adaptible/_src/_llm.py        StatefulLLM: generation, LoRA training, loop breakers
adaptible/_src/revise/        conversation -> training example, loss masks
adaptible/_src/local/         FastAPI server and web UI
adaptible/_src/eval/          dataset, evaluation harness, meta-learning, reports
adaptible/_src/autonomous/    learning from web search against the model's beliefs
scripts/cycles_mlx.py         self-repair loop (MLX)
scripts/colab/                the same loop in PyTorch, Colab notebook
scripts/cycles_results.py     run log -> tables and plot
results/                      write-up and data
```

## License

Contact for information.
