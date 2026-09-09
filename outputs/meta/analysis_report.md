# Meta-Learning Experiment Analysis Report

> ## Revision note (2026-09-09)
>
> **The conclusion of this report is withdrawn.** The experiment below was
> re-examined while fixing the analysis code in 1.0.0a3, and three problems
> mean it cannot support "the hypothesis is supported":
>
> 1. **Training targets were the ground truth, not the model's own
>    corrections.** `harness.py` built every training example as
>    `[[0]] {item.correct_answer} [[/0]]`. The experiment measured supervised
>    LoRA fine-tuning on the answer key. It did not exercise self-correction
>    (`StatefulLLM.self_correct_and_train`) at all. The `--training_source
>    self_generated` flag that would do so was added in 1.0.0a3 and has not
>    yet been run.
> 2. **The score was computed from nested checkpoints on ~5 items per window,
>    with no control arm.** Checkpoints were cumulative, so the "late" item set
>    contained the "early" set and the comparison was not between independent
>    samples. Recomputed as marginal (per-window) rates, the early windows
>    contained 0/5, 1/4, and 1/7 improvements. A score spread of
>    −0.052..+0.043 (variance 0.0016) is inside binomial noise at that n. No
>    seed was run twice, so run-to-run noise (sampling during generation,
>    training nondeterminism) was never separated from seed effects. The
>    current code returns `meta_learning_score=None` for every seed in the
>    shipped `meta_experiment.json`, with reason "window(s) below 5 trained
>    items"; `--repeats N` now exists to produce the missing noise floor.
> 3. **"Final accuracy" was accuracy on the trained items only.** Holdout
>    accuracy was computed during the run and discarded before saving, so
>    nothing in this report speaks to generalization. The result file now
>    stores holdout accuracy and `--holdout_every_checkpoint` probes it at each
>    checkpoint.
>
> The per-item observations below (stuck items, the Unicode judge bug, seed-
> dependent items, oscillation) are retained as observations. The causal
> narratives that were built on the score ("Paradox of Seed 456", "early
> struggle predicts late success", the orthogonal-subspace hypothesis) are
> marked as untested speculation or removed. Tables are unchanged.

**Date**: 2025-12-18
**Experiment**: 3-seed meta-learning run (seeds 42, 123, 456)
**Duration**: ~36 hours
**Dataset**: 100 trivia questions, 80% train / 20% holdout
**Training source**: ground truth (`[[0]] {correct_answer} [[/0]]`), 25 iterations per item
**Model**: `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B`, LoRA rank 32 on 24 layers

---

## Executive Summary

This experiment was intended to test the hypothesis that **self-improvement is itself a learnable trait**: that different model instances undergoing online learning would diverge in their learning trajectories, with some becoming stronger self-learners than others.

**The result is inconclusive.** Three model instances starting from identical weights produced these numbers:

| Seed | Meta-Learning Score (as published, cumulative rates) | Accuracy on trained items | Score under current code |
|------|---------------------|----------------|----------------|
| 123  | +0.043 | 57.5% | `None` |
| 42   | +0.017 | 58.8% | `None` |
| 456  | −0.052 | 53.8% | `None` |

The score variance of 0.0016 across seeds is consistent with sampling noise at the window sizes used (see the revision note). Whether identical starting conditions lead to different learning trajectories is not answered by this run. The "Interpretation" column of the original table ("learning accelerated", "steady", "degraded") has been removed because the data do not distinguish those cases.

The accuracy column is accuracy on the 80 items each seed was fine-tuned on. It says that after direct SFT on 80 facts, 53–59% of those facts are graded correct by a substring judge. Holdout accuracy was not recorded.

---

## What the Meta-Learning Score Measures

The formula:
```
meta_learning_score = (late_improvement_rate - early_improvement_rate)
                    + (early_forgetting_rate - late_forgetting_rate)
```

This measures **acceleration**, not absolute performance. A positive score means improvements accelerate while forgetting decelerates.

As published, the rates were cumulative (re-evaluated over every item trained so far at each checkpoint), so "late" was a superset of "early". The current code (`adaptible/_src/eval/meta.py`) uses *window* rates over the items trained since the previous checkpoint, averaged over the first and last third of checkpoints, and refuses to produce a score when any window has fewer than 5 items.

Seed 42 had the highest trained-item accuracy (58.8%) and a lower published score than seed 123. In the original report this was presented as a tension between "current performance" and "improvement trajectory". At this sample size the difference between +0.017 and +0.043 carries no information, so there is no tension to resolve.

---

## Seed 456's trajectory (observation only)

Seed 456's checkpoints, cumulative rates as published:

| Step | Improvement Rate | Forgetting Rate | Net Learning |
|------|------------------|-----------------|--------------|
| 20   | **36.4%** | **0%** | +4 |
| 50   | 8.7% | 18.5% | -3 |
| 80   | 17.9% | 12.2% | +2 |

At step 20, seed 456 had the highest single-checkpoint improvement rate in the run (4 improvements out of 11 improvable items, zero regressions), and later checkpoints were worse.

The original report offered a story for this: early success on easy items created brittleness, the model overfit to a "correction style", and continued LoRA updates then interfered. It also contrasted seed 123's poor step 20 (net −2, 27% forgetting) and concluded that "early struggle may predict late success". **None of this was tested and it should be read as speculation.** Four events at step 20 versus a handful later is exactly the kind of fluctuation a `--repeats` control arm would be expected to show within a single seed. Nothing here should be used as a selection signal.

---

## Three Failure Modes (per-item observations)

Analysis of the 80 trained items showed three patterns. These are observations about individual items and do not depend on the score.

### 1. Universally Stuck (17 items)
Items that no seed learned despite training:

**Judging Artifact (1 item)**: `sci_013` (H₂O) - The model outputs `H₂O` with Unicode subscript (₂, ord=8322), but the key term uses ASCII `2` (ord=50). The model was correct; the judge was wrong. Fixed in 1.0.0a3: `contains_key_terms` now NFKC-normalizes and casefolds both sides, so this item would grade correct on re-judging.

**Prompt Parsing Failure (1 item)**: `geo_003` (Capital of France) - The model consistently answers about **Brazil** when asked about **France**. This persists through training.

**Strong Wrong Priors (15 items)**: Items where the model gives a confident wrong answer that 25 iterations of LoRA on the label did not change:
- Capitals: Answers largest city instead of capital (Montreal for Canada, Lagos for Nigeria, Marrakech for Morocco)
- Inventors: Attributes to adjacent famous names (James Thomson for dynamite instead of Nobel)
- Misconceptions: Liver instead of skin for largest organ

Some of these may be judge failures of the same kind as `sci_013` (a correct answer phrased without the key term). The judge is substring matching and has not been audited item by item.

### 2. Seed-Dependent Learning (8 items)
Items where some seeds learned and others didn't, with the same training target:

| Item | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| sci_011 (Red Planet) | Stuck | Improved | Improved |
| math_006 (Fibonacci) | Improved | Stuck | Stuck |
| geo_013 (Philippines capital) | Stuck | Improved | Stuck |

The original report said this "validates the core hypothesis". It does not: without a same-seed repeat, item outcomes that differ across seeds cannot be distinguished from item outcomes that would differ across two runs of the same seed. Generation is sampled and training order differs per seed; either is enough to flip an item near the judge's boundary.

### 3. Forgetting (8 items)
Items that regressed (right→wrong) in at least one seed. Within this run, once a seed had a regression it tended to keep it at later checkpoints. Whether "forgetting" and "improvement" are governed by different mechanisms, as the original report suggested, is not something this data can show.

---

## Oscillating Items

Some items flip between states across checkpoints:

| Item | Trajectory (I=improved, S=stuck, R=retained, X=regressed) |
|------|-----------------------------------------------------------|
| math_014 | S → S → I → S → I → S → I → S (6 transitions) |
| hist_004 | I → S → I → S → S → S → I (4 transitions) |
| sci_018 | I → I → S → I → S → S → S (3 transitions) |

`math_014` (area of a circle = πr²) oscillates because the model sometimes outputs the formula in a form that matches the key terms and sometimes does not. This is partly a judge artifact and partly sampling variance on an item the model is unsure about. Oscillation of this kind is the noise that the window-size rule and `--repeats` are meant to expose; it is not itself a signal about learning.

---

## Information-Theoretic Interpretation (untested speculation, retained for reference)

The original report proposed that seeds with low forgetting might have LoRA updates that become more orthogonal over time, and seeds with high forgetting might have correlated updates that repeatedly perturb the same subspaces, and suggested measuring cosine similarity between successive LoRA updates.

This was never measured, and given the revision note there is no established "low-forgetting seed" to compare against. It is kept only as a possible analysis if a future run shows a seed effect above the noise floor.

---

## Implications for the Adaptible Vision

### 1. The selection problem is not yet shown to be real
Whether different seeds produce meaningfully different learners is the open question. The published variance (~0.1 range in score) is not distinguishable from noise at n≈5 per window.

### 2. Selection criteria need a noise floor before refinement
Any selection criterion (score, oscillation rate, stuck count, forgetting half-life) must first be shown to be stable across repeats of the same seed. `--repeats N` provides `within_seed_variance`, `across_seed_variance`, and `signal_to_noise` for this.

### 3. Curriculum effects are untested
Shuffle order differs per seed. The original report proposed front-loading hard items, interleaving categories, and adversarial curricula. These remain ideas; the run did not vary curriculum independently of seed.

### 4. Some failures are not fixed by 25 iterations of LoRA on the label
`geo_003` (Brazil for France) and the strong-prior capitals did not move under direct SFT. This is a real observation about the base model and the training budget, subject to the judge caveat above.

### 5. Judge quality matters
`sci_013` was a false negative from Unicode handling (now fixed). Others may remain. Before attributing failures to the model, check the judge.

---

## What This Experiment Cannot Tell Us

### 1. Self-correction
The training target was the dataset label. Nothing here measures the model's ability to identify and fix its own errors. Run `scripts/run_meta_experiment.py --training_source self_generated` for that.

### 2. Generalization
Holdout accuracy was discarded. Run with `--holdout_every_checkpoint` to record it at each checkpoint, or at minimum read `holdout_accuracy` from the saved result (stored since 1.0.0a3).

### 3. Seed effects versus noise
No seed was repeated. Run `--seeds 42 --repeats 3` and compare `within_seed_variance` with `across_seed_variance` before interpreting any per-seed number.

### 4. Long-term trajectories
80 training steps is not enough to see whether trajectories plateau or collapse as LoRA updates accumulate.

### 5. Causality
Even with a seed effect established, this design cannot say whether it comes from shuffle order, sampling during generation, or training nondeterminism.

---

## Recommendations for Future Experiments

### Short-term (next experiment)
1. **Measure self-correction**: `python scripts/run_meta_experiment.py --training_source self_generated --seeds 42,123,456`
2. **Establish the noise floor**: `python scripts/run_meta_experiment.py --seeds 42 --repeats 3`
3. **Track holdout at every checkpoint**: add `--holdout_every_checkpoint`
4. **Use a larger `--checkpoint_interval` or more items** so every window has well over 5 items; the current code returns `None` below that

### Medium-term
5. **Controlled curriculum experiments** - Same items, different orders, with repeats
6. **Extended runs** - 160+ steps to see whether trajectories continue, plateau, or collapse
7. **Audit the judge** - Re-judge the "stuck" items by hand and replace substring matching where it fails

### Long-term
8. **Multi-generation selection** - Propagate the best seed, re-diversify, repeat; only meaningful once (2) shows a seed effect
9. **Analyze LoRA weight evolution** - If a seed effect exists, measure orthogonality of updates across checkpoints
10. **Larger models** - Does any variance change with scale?

---

## Conclusion

This run does not support or refute the hypothesis that self-improvement is a learnable trait that varies across instances. It fine-tuned three copies of a 1.5B model on the answer key for 80 trivia items and observed that 53–59% of those items were graded correct afterwards, with per-seed differences that fall within the noise of the measurement.

What the run did produce:
- A judge bug (`sci_013`), since fixed
- A list of items the base model does not learn from 25 iterations of LoRA on the label
- Evidence that the original scoring method (cumulative, nested windows, no repeats) could not have detected a seed effect even if one existed

The framework runs end to end and the analysis code has been corrected. The next steps are the three commands under "Short-term" above; until they have been run there is no evidence about self-correction or about seed-dependent learning in this repository.
