# Meta-Learning Experiment Analysis Report

**Date**: 2025-12-18
**Experiment**: 3-seed meta-learning run (seeds 42, 123, 456)
**Duration**: ~36 hours
**Dataset**: 100 trivia questions, 80% train / 20% holdout

---

## Executive Summary

This experiment tested the hypothesis that **self-improvement is itself a learnable trait**—that different model instances undergoing online learning would diverge in their learning trajectories, with some becoming stronger self-learners than others.

**The hypothesis is supported.** Three model instances starting from identical weights diverged significantly:

| Seed | Meta-Learning Score | Final Accuracy | Interpretation |
|------|---------------------|----------------|----------------|
| 123  | **+0.043** | 57.5% | Learning accelerated over time |
| 42   | +0.017 | 58.8% | Steady, consistent learner |
| 456  | **-0.052** | 53.8% | Learning degraded over time |

The score variance of 0.0016 across seeds demonstrates that identical starting conditions lead to meaningfully different learning trajectories.

---

## What the Meta-Learning Score Measures

The formula:
```
meta_learning_score = (late_improvement_rate - early_improvement_rate)
                    + (early_forgetting_rate - late_forgetting_rate)
```

This measures **acceleration**, not absolute performance. A positive score means the model is getting better at getting better—improvements accelerate while forgetting decelerates.

Key insight: **Seed 42 had the highest final accuracy (58.8%) but a lower meta-learning score than seed 123.** This creates a genuine tension in selection criteria:

- For single-generation selection: pick seed 42 (best current performance)
- For multi-generation selection: pick seed 123 (fastest improvement trajectory)

The optimal choice depends on time horizon and whether you believe trajectories will continue or plateau.

---

## The Paradox of Seed 456

Seed 456's trajectory is the most instructive failure:

| Step | Improvement Rate | Forgetting Rate | Net Learning |
|------|------------------|-----------------|--------------|
| 20   | **36.4%** | **0%** | +4 |
| 50   | 8.7% | 18.5% | -3 |
| 80   | 17.9% | 12.2% | +2 |

At step 20, seed 456 had the **best single checkpoint across all seeds in the entire experiment**—36% improvement with zero forgetting. Then it collapsed.

**Hypothesis**: Early success created brittleness. The model may have overfit to a particular "correction style" that worked on easy items but generalized poorly. When it encountered harder items requiring prior-overwriting, it couldn't adapt—and the continued LoRA updates started creating interference.

Contrast with seed 123, which had a terrible step 20 (net learning -2, 27% forgetting). That early struggle may have forced development of more robust learning patterns.

**Implication**: Early struggle may predict late success. Easy early wins might be a negative signal for meta-learning potential.

---

## Three Failure Modes

Analysis of the 80 trained items revealed three distinct failure patterns:

### 1. Universally Stuck (17 items)
Items that no seed could learn despite training:

**Judging Artifact (1 item)**: `sci_013` (H₂O) - The model outputs `H₂O` with Unicode subscript (₂, ord=8322), but the key term uses ASCII `2` (ord=50). The model is actually correct; the judge is wrong.

**Prompt Parsing Failure (1 item)**: `geo_003` (Capital of France) - The model consistently answers about **Brazil** when asked about **France**. This persists through training. Not a knowledge problem—a fundamental failure to parse the prompt correctly in the base model.

**Strong Wrong Priors (15 items)**: Items where the model has confident wrong answers that LoRA cannot overcome:
- Capitals: Answers largest city instead of capital (Montreal for Canada, Lagos for Nigeria, Marrakech for Morocco)
- Inventors: Attributes to adjacent famous names (James Thomson for dynamite instead of Nobel)
- Misconceptions: Liver instead of skin for largest organ

### 2. Seed-Dependent Learning (8 items)
Items where some seeds learned and others didn't—same item, same training, different outcomes:

| Item | Seed 42 | Seed 123 | Seed 456 |
|------|---------|----------|----------|
| sci_011 (Red Planet) | Stuck | Improved | Improved |
| math_006 (Fibonacci) | Improved | Stuck | Stuck |
| geo_013 (Philippines capital) | Stuck | Improved | Stuck |

This validates the core hypothesis: the learning path creates or closes opportunities. Whether an item is learnable depends on the model's trajectory, not just the item's difficulty.

### 3. Forgetting (8 items)
Items that regressed (right→wrong) in at least one seed. Forgetting rates were "sticky"—once a seed developed a forgetting problem, it tended to persist. This suggests forgetting and improvement are governed by different mechanisms.

---

## Oscillating Items: The Instability Signal

Some items oscillate between states across checkpoints:

| Item | Trajectory (I=improved, S=stuck, R=retained, X=regressed) |
|------|-----------------------------------------------------------|
| math_014 | S → S → I → S → I → S → I → S (6 transitions) |
| hist_004 | I → S → I → S → S → S → I (4 transitions) |
| sci_018 | I → I → S → I → S → S → S (3 transitions) |

`math_014` (area of a circle = πr²) oscillates because the model sometimes outputs the formula in a format that matches key terms and sometimes doesn't. The model is on the boundary of capability—small sampling variations produce different outputs.

**Implication**: High oscillation may indicate items where additional training could tip the balance, but also where the model is fundamentally unstable.

---

## Information-Theoretic Interpretation

Each training step compresses correction information into LoRA weights. The model must:

1. Parse the correction and extract the key fact
2. Identify which weights are producing the wrong answer
3. Update those weights to produce the right answer
4. Do this without disrupting unrelated knowledge

Step 4 is where forgetting comes from. LoRA updates are low-rank perturbations—they can't surgically target specific facts. They necessarily affect nearby representations.

**Hypothesis**: Seeds that develop low forgetting rates may have LoRA updates that become more **orthogonal** over time—each update occupying a different subspace, minimizing interference. Seeds with high forgetting may have **correlated** updates that repeatedly perturb the same subspaces.

This could be tested by analyzing the LoRA weight matrices across checkpoints—measuring cosine similarity between successive updates.

---

## Implications for the Adaptible Vision

### 1. The Selection Problem is Real
Different seeds do produce meaningfully different learners. The variance is large enough (~0.1 in meta-learning score) to make selection worthwhile.

### 2. Selection Criteria Need Refinement
Current meta-learning score captures acceleration but not stability. Seed 42 (steady performer) vs seed 123 (accelerating performer) represent different tradeoffs. Consider:
- **Oscillation rate**: How stable is the model's knowledge?
- **Stuck count**: How many items remain unlearnable? (Represents closed learning pathways)
- **Forgetting half-life**: How long until a learned fact is forgotten?

### 3. Curriculum Matters More Than Expected
Shuffle order determines which items each seed sees first. Seed 456's early success on easy items may have prevented development of robust strategies. Consider:
- Front-loading hard items to force early development of prior-overwriting capacity
- Interleaving categories to prevent category-specific overfitting
- Adversarial curriculum: items where the obvious answer is wrong

### 4. Some Failures Are Not Fixable by LoRA
The prompt parsing failure (`geo_003` answering about Brazil when asked about France) and strong priors (Montreal for Canada's capital) suggest LoRA fine-tuning may be insufficient for certain corrections. The base model's priors are too strong.

### 5. Judge Quality Matters
At least one "universally stuck" item (`sci_013`) is actually a false negative due to Unicode handling. Before attributing failures to the model, ensure the evaluation is correct.

---

## What This Experiment Cannot Tell Us

### 1. Generalization
The current setup doesn't track holdout performance over time. We don't know if improving on trained items transfers to unseen items. True meta-learning would show holdout improvement without explicit training.

### 2. Long-term Trajectories
80 training steps showed divergence, but not whether that divergence is permanent, converges to a ceiling, or eventually collapses. The LoRA updates are accumulating—at some point they may create catastrophic interference.

### 3. Causality
We observe that seeds diverge, but not *why*. Is it the shuffle order (curriculum)? Random sampling during generation? Gradient noise during training? Disentangling these would require controlled experiments.

---

## Recommendations for Future Experiments

### Short-term (next experiment)
1. **Fix the Unicode judge issue** - Normalize Unicode before key term matching
2. **Track holdout accuracy at every checkpoint** - Measure generalization, not just memorization
3. **Run more seeds** - 3 seeds shows the effect; 10+ would give statistical confidence

### Medium-term
4. **Analyze LoRA weight evolution** - Measure orthogonality of updates across checkpoints
5. **Controlled curriculum experiments** - Same items, different orders, measure outcome
6. **Extended runs** - 160+ steps to see if trajectories continue, plateau, or collapse

### Long-term
7. **Multi-generation selection** - Propagate best seed, re-diversify, repeat
8. **Different training signals** - Self-generated corrections vs ground truth corrections
9. **Larger models** - Does meta-learning variance increase or decrease with scale?

---

## Conclusion

The core hypothesis—that self-improvement is a learnable trait that varies across instances—is supported by this experiment. Three identical starting points produced meaningfully different learning trajectories.

However, the experiment also revealed that many failures are not about meta-learning at all:
- Judge artifacts (fixable)
- Prompt parsing bugs (base model issue)
- Priors too strong for LoRA to overcome (architectural limitation)

The true meta-learning signal is in the **seed-dependent items**—the 8 items where some seeds learned and others didn't. These represent the cases where trajectory matters. Understanding what distinguishes learnable-for-seed-123 from learnable-for-seed-42 is the key to understanding what makes a good self-learner.

The 36-hour investment in this experiment produced actionable insights. The framework is working. The next step is refining the selection criteria and extending the time horizon to see if early trajectory differences compound into large outcome differences.
