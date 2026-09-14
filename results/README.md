# Self-repair in a 1.5B reasoning model: lift, then drift

## Abstract

We ask whether a small reasoning model can correct its own factual errors
using its own generated reasoning as the training signal, and whether the
process is stable over repeated application. Using
`DeepSeek-R1-Distill-Qwen-1.5B` with rank-8 LoRA adapters, we run a loop that
samples candidate training targets from the model, trains briefly, regenerates
the answers, and keeps an update only if the answers improved. On five facts
the model gets wrong, tested on twenty prompts (each fact in four phrasings),
accuracy rises from 4/20 to 16/20 over sixteen cycles and reaches 19/20 at
cycle 26 while the model remains coherent. Beyond that point the model's
samples converge on a single template, per-step losses fall to zero, and
accuracy fluctuates between 7 and 19 without collapsing. The run ends at
cycles 37 to 39 with 18, 19, and 18 of 20 and no loops, and the fact that had
resisted for 37 cycles is learned on all four phrasings. Four replicates in
PyTorch on a GPU reproduce the lift (three of four reach 16 or more within ten
cycles, two reach 19); after the lift one freezes, two drift, and one degrades
into looping and does not recover. We conclude that generate-judge-restore
selection makes self-training viable at this scale, and that the limiting
factor is the diversity of the model's own samples rather than the capacity of
the update.

## 1. Setup

**Model.** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B`, bf16, on a 16 GB
Apple M3 laptop (27 tokens/s). LoRA rank 8, scale 10, on every linear layer in
the last 8 of 28 transformer blocks; 2.6M trainable parameters. AdamW, learning
rate 2e-5, optimiser state reset whenever weights are restored.

**Items.** Five questions the base model answers wrongly under greedy decode:
the capitals of Morocco, Turkey, Australia, and the Philippines, and the
nearest star to Earth. Each has three hand-written paraphrases, giving twenty
prompts. Paraphrases test whether a fact was learned rather than a string.

**Judge.** A generation is correct if its think block closed and the answer
that follows contains the key term (NFKC-normalised substring match). For the
star item, an answer naming Proxima or Alpha Centauri is not credited. A
generation that never closes its think block counts as wrong and is recorded
as a loop.

## 2. Method

Each cycle:

1. Score all twenty prompts with greedy decode.
2. For every item under 4/4, draw up to six samples at temperature 0.7 from
   the question followed by a one-line reference note stating the answer. A
   sample is a candidate if its think block closed and the first answer
   sentence contains the key term. At most two candidates per item are used.
3. For each candidate, build a training example from the sample itself: its
   think block, the closing tag, and its first two answer sentences. Train at
   most four steps, stopping early once the mean loss over the answer tokens
   is below 0.15. Loss is taken over the whole target; the stop rule watches
   the answer tokens only.
4. Regenerate the item's four prompts. Keep the weights if the number correct
   increased; otherwise restore the LoRA parameters from a copy taken before
   training and verify a checksum.

There is no rehearsal on other items, no cross-item acceptance test, and no
learning-rate schedule. The loop is greedy and local by design: the question
is whether local improvements accumulate or cancel.

## 3. Results

Data: [`self-repair-cycles-2026-09-11-mlx/`](self-repair-cycles-2026-09-11-mlx/):
`cycles.csv` (per cycle: score, loops, per-item marks), `candidates.csv`
(every candidate: steps, final answer loss, marks after training, kept or
restored), `score.svg`, and the configuration.

![score per cycle](self-repair-cycles-2026-09-11-mlx/score.svg)

| Cycle | 0 | 5 | 10 | 16 | 21 | 26 | 31 | 35 | 38 | 39 |
|---|---|---|---|---|---|---|---|---|---|---|
| Correct prompts (of 20) | 8 | 11 | 12 | 16 | 10 | 19 | 7 | 7 | 19 | 18 |
| Loops (no answer) | 3 | 2 | 4 | 0 | 10 | 1 | 13 | 13 | 0 | 0 |

Baseline before training: 4/20, loops 3.

**Lift (cycles 0 to 16).** Accuracy rises from 4 to 16 and loops fall from 3
to 0. Four of the five items reach 4/4 and hold across later cycles while
other items are trained. Of 269 candidates over 40 cycles, 73 are kept. Most
rejected candidates lower the score on their own item; the restore step
prevents these from accumulating.

**Drift (cycle 17 onward).** The candidates' answer sentences converge on the
reference note's phrasing ("The correct answer is X"). Once the model
reproduces its own samples exactly, the answer-token loss reaches 0.00 to 0.10
after a single step and the kept updates are tie-breaks on near-tie prompts.
Scores fluctuate between 7 and 19, loops rise to as many as 13 of 20 (cycles
31 and 35), and items that were 4/4 for many cycles lose and regain prompts
with no update between cycles.

**Recovery (cycles 37 to 39).** The last three cycles score 18, 19, and 18 with
no loops, the best sustained stretch of the run. Nothing in the loop changed;
the same selection rule that let the score drift down let it climb back.

**Coherence.** At no cycle does the model collapse to the failure modes seen
with unfiltered fine-tuning (single-word answers, empty reasoning, or a fixed
answer to every question). Through the drift phase it remains a reasoning
model that answers most prompts. One replicate (seed 4000, below) does
degrade: after cycle 10 most of its prompts loop and the score stays under
10.

**The slow item.** Turkey is not fixed on any prompt until cycle 20 and not on
all four until cycle 37, after 68 candidates of which 8 were kept. Its early
candidates train to near-zero answer loss under their own reasoning while free
generation still concludes Istanbul: the correction held only conditioned on
the sampled reasoning, which the model did not reproduce when it thought
unprompted. It holds 4/4 at cycles 37 and 38 and 3/4 at cycle 39.

**Replicates.** The same loop, ported to PyTorch and run on one GPU with four
sampling seeds (data: `self-repair-cycles-2026-09-12-gpu-seed*/`). Baselines
differ slightly from the Mac's because the frameworks round differently.

| Cycle | 0 | 5 | 10 | 16 | 21 | 26 | 31 | 35 | 39 | Peak |
|---|---|---|---|---|---|---|---|---|---|---|
| Seed 2000 | 7 | 13 | 14 | 10 | 16 | 14 | 16 | 15 | 15 | 19 at cycle 29 |
| Seed 3000 | 8 | 15 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 at cycle 9 |
| Seed 4000 | 5 | 13 | 7 | 7 | 4 | 6 | 8 | 7 | 7 | 13 at cycle 5 |
| Seed 5000 | 6 | 16 | 16 | 16 | 16 | 13 | 19 | 16 | 10 | 19 at cycle 30 |
| Mac | 8 | 11 | 12 | 16 | 10 | 19 | 7 | 7 | 18 | 19 at cycle 26 |

Three of the four seeds reach 16 or more within ten cycles and two reach 19,
matching the Mac. What happens after the lift differs by seed. Seed 3000
freezes: four items at 4/4 and Turkey at 0/4 for thirty cycles, no loops, every
later candidate restored. Seeds 2000 and 5000 drift as the Mac did. Seed 4000
is the failure case: it reaches 13 at cycle 5, then falls to between 4 and 9
with 8 to 16 of 20 prompts looping, and never recovers. Turkey is learned in
two of the five runs (seeds 2000 and 5000, and the Mac at cycle 37).

## 4. Discussion

The accept-or-restore step, not the training target, is what makes the loop
work. The same targets applied without the check, in sequence, undid earlier
corrections; with it, the minority of updates that land accumulate and the
rest cost nothing. Small updates (at most four steps) plus selection outperform
larger updates with rehearsal, which overwrote neighbouring facts.

The drift phase identifies the limit. Candidates are drawn from the model
itself, so as the model learns them the distribution narrows toward whatever
phrasing the sampling prompt induces. When the model can reproduce its own
samples with zero loss, there is nothing left to learn from them, and
selection begins to act on decode noise. Sustaining lift therefore requires
sample diversity that does not collapse: varying the sampling prompt,
rejecting candidates that echo it, or drawing candidates from a different
source than the model being trained.

The facts in this experiment entered through the reference note; the model did
not discover them. What the experiment shows is that the model's own reasoning
is a workable carrier for a correction into its weights, and that a judge on
generated answers is sufficient to keep the process from destroying what it
already knows.

## 5. Limitations

Five items and one model. Greedy decode on a 1.5B model flips on near-tie
prompts under weight changes far smaller than a training step, giving a noise
floor of about two points per cycle. The judge is substring matching. The
sampling prompt contains the answer. Five runs in total, with one that
degrades, is too few to put a rate on the failure case.

## 6. The loop as a running node

The cycle loop above hands the model two things a node on its own would not
have: a reference note that carries the answer, and a judge that decides
whether to keep an update. This run removes both. The model is served over
HTTP (`python -m adaptible.local`), a client asks the five questions, and
thumbs down the wrong answers. On review the node looks each flagged question
up in a sealed store of twelve short passages (the five facts, the two
controls, and five distractors; keyword overlap picks the passage), reads the
passage and names the answer it gives, writes three rephrasings of the
question, samples candidate answers with the passage attached (from the
adapter the process started with), and trains at most four steps on one. The
update is kept when the number of prompts, question plus rephrasings, whose
first sentence names that answer rises, and three generic control prompts
still answer; otherwise the weights are restored. The model does every read
itself: the passage, its own candidates, and its own replies. The key-term
judge and the hand-written paraphrases never enter the loop; they only score
the session afterwards, together with two control facts the node is never
asked to repair.

Sixteen sessions, 71 candidates, 11 kept. Data in
[`live-repair-2026-09-13-mlx/`](live-repair-2026-09-13-mlx/): `sessions.csv`
(one row per session), `repairs.csv` (one row per candidate with the node's
own marks before and after), `score.svg`.

| session | flagged | kept | correct of 20 | controls of 2 |
|---|---|---|---|---|
| 0 | 5 | 3 | 4 | 1 |
| 1 | 4 | 1 | 4 | 2 |
| 2 | 3 | 1 | 11 | 2 |
| 3 to 9 | 2 | 0 | 11 | 2 |
| 10 | 2 | 1 | 7 | 1 |
| 11 | 3 | 0 | 7 | 1 |
| 12 | 3 | 1 | 6 | 2 |
| 13 | 2 | 1 | 9 | 1 |
| 14 | 3 | 2 | 7 | 2 |
| 15 | 5 | 1 | 10 | 2 |

Three facts are repaired by session 2 and hold for eight sessions: Morocco
and Australia on all four phrasings, the Philippines on three. Loading the
saved adapter in a fresh process reproduces the final session's marks
exactly (10 of 20, Morocco and Australia 4 of 4, both control facts right),
so the patches are in the weights, not the process. Turkey never moves: the
node reads the passage right, writes Ankara, and four steps never change the
greedy answer on any phrasing. The nearest star reaches two of four on the
node's own prompts once, at session 10, and that update is where the churn
starts: it passes the three generic controls and knocks Morocco from three
phrasings to one, the Philippines from three to two, and makes the Vietnam
control loop. From then on each kept repair moves one fact up and a neighbour
down. The node's control prompts are not the neighbours, so it cannot see
this; the outside score can. The cycle loop shows the same interference
(section 3) but outruns it over forty rounds with the exact answer as the
hint; here, with a passage to read and the node's own rephrasings as the
yardstick, sixteen sessions end in a trade.

## Reproduce

```bash
CYCLES=40 PYTHONPATH=. .venv/bin/python scripts/cycles_mlx.py | tee cycles.log
.venv/bin/python scripts/cycles_results.py --log cycles.log --out results/my-run
```

Or open [`scripts/colab/adaptible_cycles.ipynb`](../scripts/colab/adaptible_cycles.ipynb)
on a Colab GPU and Run all.

The running node:

```bash
PYTHONPATH=. .venv/bin/python scripts/live_repair.py --sessions 16 --out outputs/runs/live
.venv/bin/python scripts/live_results.py --run outputs/runs/live --out results/my-live-run
```
