"""Runner script for evaluation experiments.

Usage:
    python -m adaptible.eval [options]

Options:
    --name NAME           Experiment name (default: "default")
    --train-ratio RATIO   Fraction to use for training (default: 0.8)
    --iterations N        Step cap per training call (default: 12)
    --loss_target L       Stop a training call once a step's answer-token
                          loss is below L (default: 0.6; 0 or negative
                          disables, so exactly --iterations steps run)
    --shuffle             Shuffle the dataset
    --seed SEED           Random seed for shuffling (default: 42)
    --training-source S   "ground_truth" (fine-tune on the label) or
                          "self_generated" (train on the model's own revision)
    --revision-prompt P   Revision prompt preset for self_generated:
                          "default" or "fewshot"
    --think_mode M        How the training target treats the chat template's
                          open <think> block: "rationale" (the default; train
                          on "{rationale}\n</think>\n\n{answer}" where the
                          rationale concludes the answer, stop on the answer
                          loss), "baseline" (model's own reasoning in the
                          unmasked prefix, answer only in the loss), "empty"
                          (train on "</think>\n\n{answer}"), or "none" (old
                          malformed target, for comparison)
    --[no]close_think     Deprecated alias: --noclose_think == --think_mode none
    --rehearsal_k K       Fold K self-distillation examples from correct
                          trained-split items into every correction step
                          (default: 0)
    --rehearsal_max_tokens N
                          Skip rehearsal items whose baseline is longer than N
                          tokens (default: 768)
    --rehearsal_weight W  Multiplier on the mean rehearsal gradient in the
                          joint step (default: 1.0)
    --rehearsal_margin M  Rehearsal hinge: a rehearsal example's gradient is
                          applied on a step only when its loss is more than M
                          above its loss at the call's first step
                          (default: 0.05)
    --rationale_max_tokens N
                          Cap the rationale in the training target at N
                          tokens, cut at a sentence boundary (default: 512)
    --[no]train_correct_items
                          Train train-split items whose baseline answer is
                          already correct (default: False; they are skipped,
                          re-inferred, and reported as interference)
    --verify_steps N      After a correction reaches the loss target or cap,
                          generate and judge the answer; while wrong and
                          under the cap, train N more steps and check again
                          (default: 0 = off)
    --learning_rate LR    Optimizer learning rate for StatefulLLM (default: model's)
    --lora_rank R         LoRA rank (default: 32)
    --lora_layers N       Number of trailing layers converted to LoRA (default: 24)
    --lora_scale S        LoRA scale (default: 10.0)
    --subset N            Only use first N items (for quick tests)
    --category CAT        Filter to specific category
    --output PATH         Output path for HTML report
    --save-dataset PATH   Save the dataset to JSON file
    --load-dataset PATH   Load dataset from JSON file instead of default
"""

import webbrowser

import vizible
from absl import app, flags

from . import (
    TRAINING_SOURCES,
    EvaluationConfig,
    EvaluationHarness,
    generate_default_dataset,
    generate_html_report,
    load_dataset,
    save_dataset,
)
from .harness import (
    DEFAULT_LORA_LAYERS,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_SCALE,
    lora_model_kwargs,
    lora_settings_text,
)
from ..revise import DEFAULT_RATIONALE_MAX_TOKENS, REVISION_PROMPTS, THINK_MODES

_NAME = flags.DEFINE_string("name", "default", "Experiment name")
_TRAIN_RATIO = flags.DEFINE_float("train_ratio", 0.8, "Train/holdout split ratio")
_ITERATIONS = flags.DEFINE_integer(
    "iterations", 12, "Step cap per training call (exact count if --loss_target <= 0)"
)
_LOSS_TARGET = flags.DEFINE_float(
    "loss_target",
    0.6,
    "Stop each training call as soon as a step's loss is below this; "
    "--iterations is then a cap. The greedy answer flips to the correction at "
    "~0.6 with the reasoning intact; driving the loss to ~0 collapses the "
    "reasoning. 0 or a negative value disables the target and trains exactly "
    "--iterations steps.",
)
_SHUFFLE = flags.DEFINE_boolean("shuffle", False, "Shuffle the dataset")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed")
_TRAINING_SOURCE = flags.DEFINE_enum(
    "training_source",
    "ground_truth",
    list(TRAINING_SOURCES),
    "What the model is trained on: the dataset label (ground_truth) or its own "
    "revision of its baseline answer (self_generated).",
)
_REVISION_PROMPT = flags.DEFINE_string(
    "revision_prompt",
    "default",
    f"Revision prompt preset used with --training_source self_generated; one of "
    f"{', '.join(REVISION_PROMPTS)}. 'fewshot' is a short imperative prompt with "
    "two worked examples over a plain User:/Assistant: dialog (no chat-template "
    "tokens).",
)
_THINK_MODE = flags.DEFINE_enum(
    "think_mode",
    "rationale",
    list(THINK_MODES),
    "How the training target treats the chat template's open <think> tag. "
    "'rationale': train on '{rationale}\\n</think>\\n\\n{revision}' where the "
    "rationale is reasoning that concludes the revision (the revision's own "
    "think block, or one generated from the label for ground_truth); the "
    "loss target applies to the answer tokens only. 'baseline': the model's "
    "own reasoning from its baseline response goes in the unmasked prefix and "
    "only the corrected answer is trained on. 'empty': train on "
    "'</think>\\n\\n{revision}' (teaches the model to stop reasoning). "
    "'none': the old malformed target (revision inside the open think block).",
)
_CLOSE_THINK = flags.DEFINE_boolean(
    "close_think",
    None,
    "Deprecated; use --think_mode. --noclose_think is --think_mode none.",
)
_REHEARSAL_K = flags.DEFINE_integer(
    "rehearsal_k",
    0,
    "Fold K rehearsal examples into every correction step: other trained-split "
    "items the model already answered correctly, with their own baseline output "
    "as the target (self-distillation). Each step applies grad(correction) + "
    "rehearsal_weight * mean(grad(rehearsal)), one single-sequence pass per "
    "example, and stops on the correction loss alone. 0 disables.",
)
_REHEARSAL_MAX_TOKENS = flags.DEFINE_integer(
    "rehearsal_max_tokens",
    768,
    "Exclude items whose baseline response is longer than this many tokens from "
    "the rehearsal pool.",
)
_REHEARSAL_WEIGHT = flags.DEFINE_float(
    "rehearsal_weight",
    1.0,
    "Multiplier on the mean rehearsal gradient in the joint step (>= 0).",
)
_REHEARSAL_MARGIN = flags.DEFINE_float(
    "rehearsal_margin",
    0.05,
    "Rehearsal hinge: a rehearsal example's gradient is applied on a step only "
    "when its loss is more than this above its loss at the call's first step "
    "(>= 0). Rehearsal anchors the model; it is never minimised on its own.",
)
_RATIONALE_MAX_TOKENS = flags.DEFINE_integer(
    "rationale_max_tokens",
    DEFAULT_RATIONALE_MAX_TOKENS,
    "Cap the rationale placed in the training target at this many tokens, cut "
    "at the last sentence boundary. Under --think_mode rationale an output "
    "with no </think> is taken whole as the rationale (the model reasons "
    "inside the open think block); an item with no rationale at all is skipped.",
)
_TRAIN_CORRECT_ITEMS = flags.DEFINE_boolean(
    "train_correct_items",
    False,
    "Train train-split items whose baseline answer is already judged correct. "
    "Off by default: such items are skipped (not trained, not holdout), still "
    "re-inferred after training, and the ones that regressed are reported as "
    "interference from the other items' training. For self_generated this "
    "skip is an oracle; it measures the ceiling.",
)
_VERIFY_STEPS = flags.DEFINE_integer(
    "verify_steps",
    0,
    "Verify-after-target: once a correction's training call returns (loss "
    "target reached or cap), generate the item's answer and judge it; while it "
    "is wrong and the step cap has not been reached, train this many more "
    "steps (no loss target) and check again. 0 disables.",
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", None, "StatefulLLM learning rate (default: the model's)."
)
_LORA_RANK = flags.DEFINE_integer(
    "lora_rank", DEFAULT_LORA_RANK, "LoRA rank of every converted linear layer."
)
_LORA_LAYERS = flags.DEFINE_integer(
    "lora_layers",
    DEFAULT_LORA_LAYERS,
    "Number of trailing transformer layers whose linears are converted to LoRA.",
)
_LORA_SCALE = flags.DEFINE_float(
    "lora_scale", DEFAULT_LORA_SCALE, "LoRA scale (alpha / rank)."
)
_SUBSET = flags.DEFINE_integer("subset", None, "Only use first N items")
_CATEGORY = flags.DEFINE_string("category", None, "Filter to specific category")
_OUTPUT = flags.DEFINE_string(
    "output", "/tmp/adaptible_eval_report.html", "Output HTML path"
)
_SAVE_DATASET = flags.DEFINE_string("save_dataset", None, "Save dataset to JSON file")
_LOAD_DATASET = flags.DEFINE_string("load_dataset", None, "Load dataset from JSON file")
_NO_BROWSER = flags.DEFINE_boolean("no_browser", False, "Don't open browser")


def main(_):
    # Load or generate dataset
    if _LOAD_DATASET.value:
        print(f"Loading dataset from {_LOAD_DATASET.value}...")
        dataset = load_dataset(_LOAD_DATASET.value)
    else:
        print("Generating default dataset...")
        dataset = generate_default_dataset()

    print(f"Dataset: {dataset.name} ({len(dataset)} items)")
    print(f"Categories: {', '.join(sorted(dataset.categories))}")

    # Save dataset if requested
    if _SAVE_DATASET.value:
        save_dataset(dataset, _SAVE_DATASET.value)
        print(f"Saved dataset to {_SAVE_DATASET.value}")

    # Apply filters
    if _CATEGORY.value:
        dataset = dataset.by_category(_CATEGORY.value)
        print(f"Filtered to category '{_CATEGORY.value}': {len(dataset)} items")

    if _SUBSET.value:
        dataset = dataset.subset(list(range(min(_SUBSET.value, len(dataset)))))
        print(f"Using subset: {len(dataset)} items")

    if len(dataset) == 0:
        print("Error: No items in dataset after filtering")
        return 1

    # Create config
    config = EvaluationConfig(
        name=_NAME.value,
        training_iterations=_ITERATIONS.value,
        loss_target=_LOSS_TARGET.value,
        shuffle=_SHUFFLE.value,
        seed=_SEED.value,
        train_ratio=_TRAIN_RATIO.value,
        training_source=_TRAINING_SOURCE.value,
        revision_prompt=_REVISION_PROMPT.value,
        think_mode=_THINK_MODE.value,
        close_think=_CLOSE_THINK.value,
        rehearsal_k=_REHEARSAL_K.value,
        rehearsal_max_tokens=_REHEARSAL_MAX_TOKENS.value,
        rehearsal_weight=_REHEARSAL_WEIGHT.value,
        rehearsal_margin=_REHEARSAL_MARGIN.value,
        rationale_max_tokens=_RATIONALE_MAX_TOKENS.value,
        train_correct_items=_TRAIN_CORRECT_ITEMS.value,
        verify_steps=_VERIFY_STEPS.value,
    )
    model_kwargs = lora_model_kwargs(
        rank=_LORA_RANK.value, layers=_LORA_LAYERS.value, scale=_LORA_SCALE.value
    )
    if _LEARNING_RATE.value is not None:
        model_kwargs["learning_rate"] = _LEARNING_RATE.value

    # Run evaluation
    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)
    print(lora_settings_text(model_kwargs))
    print()

    harness = EvaluationHarness(model_kwargs=model_kwargs)
    result = harness.run(dataset, config, verbose=True)

    # Generate report
    report_path = generate_html_report(result, _OUTPUT.value)
    print()
    print("=" * 70)
    print("REPORT GENERATED")
    print("=" * 70)
    vizible.green(f"Report: file://{report_path}")

    if not _NO_BROWSER.value:
        webbrowser.open_new_tab(f"file://{report_path}")

    return 0


if __name__ == "__main__":
    app.run(main)
