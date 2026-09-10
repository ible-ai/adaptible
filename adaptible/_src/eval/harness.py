"""Evaluation harness for running experiments."""

import dataclasses
import json
import random
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from .._classes import InteractionHistory, TrainingExample
from .. import _llm
from .._llm import StatefulLLM, TrainingStats
from ..db import (
    Database,
    Example,
    Experiment,
    ExperimentType,
    Phase,
    Response,
    SourceType,
    TrainingEvent,
)
from ..revise import (
    DEFAULT_RATIONALE_MAX_TOKENS,
    THINK_MODES,
    InvalidRevisionError,
    collate_training_examples,
    make_revision_prompt,
    make_revision_training_example,
    make_training_example,
    rationale_from_output,
    resolve_think_mode,
    revision_prompt_preset,
    strip_think_tags,
    template_opens_think,
    validate_rationale_max_tokens,
    validate_revision_response,
)
from .dataset import TriviaDataset, TriviaItem

# What the training target is built from.
#   ground_truth:   the dataset label, wrapped as "[[0]] <answer> [[/0]]". This is
#                   supervised fine-tuning on the label; it measures whether the
#                   model can absorb a correction, not whether it can produce one.
#   self_generated: the model's own revision of its baseline answer, obtained via
#                   the same make_revision_prompt -> validate -> train pipeline the
#                   server uses. This is the self-correction loop the README describes.
TRAINING_SOURCES = ("ground_truth", "self_generated")


def validate_training_source(training_source: str) -> None:
    """Raise ValueError unless ``training_source`` is a known value."""
    if training_source not in TRAINING_SOURCES:
        raise ValueError(
            f"training_source must be one of {TRAINING_SOURCES}, "
            f"got {training_source!r}"
        )


# ``StatefulLLM``'s LoRA defaults, exposed so the CLIs and reports share them.
DEFAULT_LORA_RANK: int = int(_llm._LORA_PARAMETERS["rank"])
DEFAULT_LORA_LAYERS: int = int(_llm._NUM_LORA_LAYERS)
DEFAULT_LORA_SCALE: float = float(_llm._LORA_PARAMETERS["scale"])


def lora_model_kwargs(
    rank: int = DEFAULT_LORA_RANK,
    layers: int = DEFAULT_LORA_LAYERS,
    scale: float = DEFAULT_LORA_SCALE,
) -> dict[str, Any]:
    """``StatefulLLM`` keyword arguments for a LoRA configuration.

    The CLIs thread ``--lora_rank`` / ``--lora_layers`` / ``--lora_scale``
    through ``model_kwargs`` with this, so the values are recorded in the
    experiment's config_json alongside everything else the model was built
    with.
    """
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"lora rank must be a positive int, got {rank!r}")
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError(f"lora layers must be a positive int, got {layers!r}")
    if scale <= 0:
        raise ValueError(f"lora scale must be positive, got {scale!r}")
    return {
        "num_lora_layers": layers,
        "lora_parameters": {"rank": rank, "dropout": 0.0, "scale": float(scale)},
    }


def lora_settings(model_kwargs: dict[str, Any] | None) -> tuple[int, int, float]:
    """``(rank, layers, scale)`` a model built with ``model_kwargs`` uses.

    Missing keys fall back to ``StatefulLLM``'s defaults, so runs recorded
    before the LoRA flags existed report the capacity they actually trained
    with.
    """
    model_kwargs = model_kwargs or {}
    params = model_kwargs.get("lora_parameters") or {}
    return (
        int(params.get("rank", DEFAULT_LORA_RANK)),
        int(model_kwargs.get("num_lora_layers", DEFAULT_LORA_LAYERS)),
        float(params.get("scale", DEFAULT_LORA_SCALE)),
    )


def lora_settings_text(model_kwargs: dict[str, Any] | None) -> str:
    """``LoRA: rank 32, layers 24, scale 10.0`` for logs and reports."""
    rank, layers, scale = lora_settings(model_kwargs)
    return f"LoRA: rank {rank}, layers {layers}, scale {scale:g}"


@dataclasses.dataclass
class EvaluationConfig:
    """Configuration for an evaluation run.

    Attributes:
        training_source: Where the training target comes from; see
            ``TRAINING_SOURCES``. Every result derived from a run must carry this
            value so "ground_truth" numbers are never mistaken for self-correction.
        revision_prompt: Which revision prompt preset ``self_generated`` uses;
            see ``revise.revision_prompt_preset``. Ignored for ``ground_truth``.
        think_mode: How the training target treats the chat template's open
            ``<think>`` block; one of ``revise.THINK_MODES`` ("none", "empty",
            "baseline", "rationale"). "rationale" (the default) trains on
            ``{rationale}\n</think>\n\n{answer}{eos}`` where the rationale is
            reasoning that concludes the answer: the revision generation's
            own think block for ``self_generated``, or, for ``ground_truth``,
            one generated with ``make_rationale_prompt`` from the label
            (``revise.rationale_from_output``: the think block, or the whole
            output when the model never closed the tag). Items with no
            rationale at all are **not trained** (the "empty" fallback they
            used to get collapses the model) and are counted in
            ``EvaluationResult.rationale_missing_count``. "baseline" keeps the
            model's own baseline reasoning in the unmasked prefix and trains
            only on the corrected answer; on DeepSeek-R1-Distill the correction
            then does not take at inference (the model re-derives its old
            answer).
        close_think: Deprecated alias for ``think_mode``. ``False`` forces
            ``think_mode="none"`` (the pre-1.0.0a4 target, for comparison);
            ``None``/``True`` defer to ``think_mode``. After construction it is
            always ``think_mode != "none"``.
        rehearsal_k: When > 0, every training step combines the correction with
            ``k`` rehearsal examples: other *trained-split* items whose baseline
            answer was judged correct, with the model's own full baseline output
            as the target (self-distillation). Sampled with
            ``seed + item index``; the item being corrected is never in its own
            rehearsal set. Holdout items are never used. The item is trained
            with one ``StatefulLLM.train_on_examples`` call whose every step
            applies ``grad(correction) + rehearsal_weight * mean(grad(rehearsal))``,
            each gradient from its own single-sequence pass, so memory is
            bounded by one sequence (see ``_train_one_item``).
        rehearsal_weight: Multiplier on the mean rehearsal gradient in that
            joint step. ``0`` keeps the rehearsal passes (and their loss
            reporting) but lets only the correction move the weights.
        rehearsal_margin: The rehearsal hinge (``_llm.active_rehearsal``): a
            rehearsal example's gradient is applied on a step only when its
            loss is more than this above its loss at the call's first step.
            Rehearsal anchors the model; it is never minimised on its own.
            With the gradient always on, a 40-item screen drove the rehearsal
            loss from ~0.38 to 0.09 and the model memorised its dozen pool
            outputs. ``ItemResult.train_rehearsal_active_steps`` and the
            summary line's ``rehearsal active in P% of steps`` say how often
            the hinge fired.
        rationale_max_tokens: Token cap on the rationale placed in the
            target; longer ones are cut at the last sentence boundary
            (``revise.truncate_at_sentence``), recorded per item as
            ``rationale_tokens`` / ``rationale_truncated``.
        rehearsal_max_tokens: Items whose raw baseline response is longer than
            this many tokens are excluded from the rehearsal pool. Rehearsal
            targets are the model's own full baseline output, which can run to
            the generation cap; this keeps every training sequence bounded.
        training_iterations: Cap on optimizer steps per training call. With a
            ``loss_target`` this is a ceiling, not a count.
        loss_target: Stop each training call as soon as a step's *answer*
            loss falls below this: the loss over the answer tokens plus eos
            (``TrainingExample.stop_mask``), whatever else the training loss
            covers. A per-step probe on the real model showed the greedy
            answer flips to the correction at a mean target loss of ~0.6 with
            the reasoning intact, while driving the loss to ~0 (what a fixed
            5-25 iterations does) collapses the reasoning and bleeds the answer
            into unrelated questions. ``None`` (or any value <= 0) disables the
            target and trains exactly ``training_iterations`` steps. Only the
            correction's answer loss is compared against the target; the
            training loss over the whole target
            (``ItemResult.train_final_train_loss``) and the rehearsal loss
            (``ItemResult.train_rehearsal_final_loss``) are reported but never
            stop training.
        train_correct_items: Whether train-split items whose baseline answer
            is already judged correct are trained. ``False`` (the default)
            skips them: they get ``ItemResult.skipped_correct=True``, stay
            ``was_trained=False``, are excluded from both the train metrics
            and the holdout set, and are still re-inferred after training so
            the ones that regressed count as *interference* from the other
            items' training (``EvaluationResult.skipped_correct_regressed_count``).
            A 40-item screen spent 60% of its gradient steps on 20 items with
            nothing to correct, and every regression was one of them. With
            ``training_source="self_generated"`` this skip is an oracle (a live
            system cannot know which of its answers are wrong); it exists to
            measure the ceiling. ``True`` restores training every train-split
            item.
        verify_steps: When > 0, after a correction's training call returns
            (target reached or cap) the harness *generates* the item's answer
            and judges it with ``contains_key_terms``. If it is still wrong
            and the step cap has not been reached, it trains ``verify_steps``
            more steps on the same examples with no loss target, then checks
            again, until the answer is right or the cap is reached. Per item
            ``ItemResult.verify_attempts`` counts the checks and
            ``ItemResult.verified`` says whether the last one passed (``None``
            when off); ``train_steps`` is the total. Motivation: some items
            reach an answer loss of 0.05 after one step yet the free-running
            model still gives its old answer, because the teacher-forced
            answer is easy given the rationale while the model's own reasoning
            never reaches it. For those the loss target is the wrong stop
            signal; generating and checking is the only reliable one. The
            verification generations are intermediate and are not recorded in
            the database.
    """

    name: str = "default"
    training_iterations: int = 12  # Step cap per train_on_example call
    loss_target: float | None = 0.6
    epochs_per_call: int = 5  # Matches StatefulLLM._epochs
    shuffle: bool = False
    seed: int = 42
    train_ratio: float = 0.8  # Fraction to use for training
    max_tokens: int | None = None  # Use model default if None
    training_source: str = "ground_truth"
    revision_prompt: str = "default"
    think_mode: str = "rationale"
    close_think: bool | None = None
    rehearsal_k: int = 0
    rehearsal_max_tokens: int = 768
    rehearsal_weight: float = 1.0
    rehearsal_margin: float = 0.05
    rationale_max_tokens: int = DEFAULT_RATIONALE_MAX_TOKENS
    train_correct_items: bool = False
    verify_steps: int = 0

    def __post_init__(self) -> None:
        validate_training_source(self.training_source)
        revision_prompt_preset(self.revision_prompt)  # raises ValueError if unknown
        self.think_mode = resolve_think_mode(self.think_mode, self.close_think)
        self.close_think = self.think_mode != "none"
        validate_rehearsal_k(self.rehearsal_k)
        validate_rehearsal_max_tokens(self.rehearsal_max_tokens)
        self.rehearsal_weight = validate_rehearsal_weight(self.rehearsal_weight)
        self.rehearsal_margin = validate_rehearsal_margin(self.rehearsal_margin)
        validate_rationale_max_tokens(self.rationale_max_tokens)
        self.loss_target = normalize_loss_target(self.loss_target)
        self.train_correct_items = bool(self.train_correct_items)
        validate_verify_steps(self.verify_steps)


def normalize_loss_target(loss_target: float | None) -> float | None:
    """Map a loss target to ``None`` when disabled (``None`` or <= 0)."""
    if loss_target is None or loss_target <= 0:
        return None
    return float(loss_target)


def validate_verify_steps(verify_steps: int) -> None:
    """Raise ValueError unless ``verify_steps`` is a non-negative int."""
    if (
        isinstance(verify_steps, bool)
        or not isinstance(verify_steps, int)
        or verify_steps < 0
    ):
        raise ValueError(
            f"verify_steps must be a non-negative int, got {verify_steps!r}"
        )


def validate_rehearsal_weight(rehearsal_weight: float) -> float:
    """Return ``rehearsal_weight`` as a float; raise ValueError if negative."""
    if isinstance(rehearsal_weight, bool) or rehearsal_weight is None:
        raise ValueError(f"rehearsal_weight must be a number, got {rehearsal_weight!r}")
    if rehearsal_weight < 0:
        raise ValueError(f"rehearsal_weight must be >= 0, got {rehearsal_weight!r}")
    return float(rehearsal_weight)


def validate_rehearsal_margin(rehearsal_margin: float) -> float:
    """Return ``rehearsal_margin`` as a float; raise ValueError if negative."""
    if isinstance(rehearsal_margin, bool) or rehearsal_margin is None:
        raise ValueError(f"rehearsal_margin must be a number, got {rehearsal_margin!r}")
    if rehearsal_margin < 0:
        raise ValueError(f"rehearsal_margin must be >= 0, got {rehearsal_margin!r}")
    return float(rehearsal_margin)


def validate_rehearsal_k(rehearsal_k: int) -> None:
    """Raise ValueError unless ``rehearsal_k`` is a non-negative int."""
    if not isinstance(rehearsal_k, int) or rehearsal_k < 0:
        raise ValueError(f"rehearsal_k must be a non-negative int, got {rehearsal_k!r}")


def validate_rehearsal_max_tokens(rehearsal_max_tokens: int) -> None:
    """Raise ValueError unless ``rehearsal_max_tokens`` is a positive int."""
    if not isinstance(rehearsal_max_tokens, int) or rehearsal_max_tokens <= 0:
        raise ValueError(
            f"rehearsal_max_tokens must be a positive int, got {rehearsal_max_tokens!r}"
        )


@dataclasses.dataclass
class ItemResult:
    """Result for a single trivia item.

    Attributes:
        revision_text: The model's own revision output when the training source is
            ``self_generated`` (raw, including markers); None otherwise.
        revision_invalid: True if the item was scheduled for training but the
            self-generated revision failed validation, so no training happened.
        revision_answer: The content between the ``[[X]]`` markers of a valid
            self-generated revision, i.e. exactly what the model was trained on.
            None for ``ground_truth`` runs, invalid revisions, and holdout items.
        revision_has_key_terms: ``contains_key_terms`` applied to
            ``revision_answer``: was the revision itself judged correct, before
            any training happened? None whenever ``revision_answer`` is None.
        revision_changed_text: ``revision_answer != initial_response``. False
            means the model restated its baseline answer verbatim.
        revision_changed_verdict: ``revision_has_key_terms !=
            initial_has_key_terms``: the revision flipped the judge's verdict in
            either direction.
        initial_token_count: Tokens in the raw baseline response.
        post_token_count: Tokens in the raw post-training response; None until
            the post-training pass runs.
        rehearsal_item_ids: Ids of the items whose baseline outputs were batched
            with this item's correction (``EvaluationConfig.rehearsal_k``).
        rationale_text: With ``think_mode="rationale"``, the reasoning that
            was placed before ``</think>`` in the training target (the
            revision's own think block, or the generated one for
            ``ground_truth``); None otherwise or when none was available.
        rationale_missing: The item was scheduled for training with
            ``think_mode="rationale"`` on a think template but no rationale
            could be had (the model's output held no reasoning at all), so
            it was **skipped**: not trained, and excluded from the train
            metrics like an invalid revision.
        rationale_tokens: Tokens in the rationale placed in the target (None
            without one).
        rationale_truncated: The rationale was cut at a sentence boundary to
            fit ``EvaluationConfig.rationale_max_tokens``.
        train_steps: Optimizer steps the correction took (0 if not trained).
        train_initial_loss: Answer loss at the correction's first step.
        train_final_loss: Answer loss at the correction's last step (the loss
            the stop rule watched; over the answer tokens plus eos).
        train_final_train_loss: Training loss over the whole target at the
            correction's last step; equals ``train_final_loss`` unless the
            target carries more than the answer (``think_mode="rationale"``).
        train_hit_cap: The correction ran out of steps without reaching the
            loss target.
        train_rehearsal_final_loss: Mean rehearsal loss at the correction's last
            step; None when the item was trained without rehearsal.
        train_rehearsal_initial_loss: Mean rehearsal loss at the first step,
            the hinge's anchor; None without rehearsal.
        train_rehearsal_active_steps: ``(step, rehearsal example)`` pairs
            whose gradient the hinge let through, out of
            ``train_steps * len(rehearsal_item_ids)``.
        skipped_correct: The item was in the train split but its baseline
            was already judged correct and ``EvaluationConfig.train_correct_items``
            is False, so it was **not trained**. Neither a train item nor a
            holdout item; it is still re-inferred after training, and a
            regression here is interference from the other items' training.
        verify_attempts: With ``EvaluationConfig.verify_steps`` > 0, the
            number of generate-and-judge checks made for this item after its
            training (0 when off or not trained).
        verified: Whether the last verification check found the answer
            correct; ``None`` when verification is off or the item was not
            trained. ``False`` means the item is still wrong at the step cap.
    """

    item_id: str
    question: str
    correct_answer: str
    key_terms: list[str]
    initial_response: str
    initial_response_raw: str
    initial_has_key_terms: bool
    post_response: str | None = None
    post_response_raw: str | None = None
    post_has_key_terms: bool | None = None
    was_trained: bool = False
    training_time_seconds: float = 0.0
    skipped_correct: bool = False
    verify_attempts: int = 0
    verified: bool | None = None
    revision_text: str | None = None
    revision_invalid: bool = False
    revision_answer: str | None = None
    revision_has_key_terms: bool | None = None
    revision_changed_text: bool | None = None
    revision_changed_verdict: bool | None = None
    initial_token_count: int = 0
    post_token_count: int | None = None
    rehearsal_item_ids: list[str] = dataclasses.field(default_factory=list)
    rationale_text: str | None = None
    rationale_missing: bool = False
    rationale_tokens: int | None = None
    rationale_truncated: bool = False
    train_steps: int = 0
    train_initial_loss: float | None = None
    train_final_loss: float | None = None
    train_final_train_loss: float | None = None
    train_hit_cap: bool = False
    train_rehearsal_final_loss: float | None = None
    train_rehearsal_initial_loss: float | None = None
    train_rehearsal_active_steps: int = 0

    @property
    def train_rehearsal_pairs(self) -> int:
        """``train_steps * k``: what ``train_rehearsal_active_steps`` is out of."""
        return self.train_steps * len(self.rehearsal_item_ids)

    @property
    def post_empty_think(self) -> bool:
        """The post-training response opened with ``</think>``: no reasoning at all."""
        return (self.post_response_raw or "").lstrip().startswith("</think>")

    @property
    def revision_fixed(self) -> bool:
        """Baseline was wrong and the revision is right."""
        return bool(self.revision_has_key_terms) and not self.initial_has_key_terms

    @property
    def revision_broke(self) -> bool:
        """Baseline was right and the revision is wrong."""
        return self.revision_has_key_terms is False and self.initial_has_key_terms

    @property
    def regressed(self) -> bool:
        """Baseline was right and the post-training answer is wrong."""
        return self.initial_has_key_terms and self.post_has_key_terms is False


@dataclasses.dataclass
class EvaluationResult:
    """Complete results from an evaluation run."""

    config: EvaluationConfig
    dataset_name: str
    timestamp: str
    items: list[ItemResult] = dataclasses.field(default_factory=list)
    total_time_seconds: float = 0.0
    # ``StatefulLLM`` keyword arguments the harness was given (LoRA capacity,
    # learning rate); empty when a pre-built model was passed in.
    model_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Computed metrics
    @property
    def train_items(self) -> list[ItemResult]:
        return [item for item in self.items if item.was_trained]

    @property
    def holdout_items(self) -> list[ItemResult]:
        """Items never scheduled for training (skipped items excluded).

        An item scheduled for training but skipped (invalid revision, no
        rationale, or baseline already correct under
        ``train_correct_items=False``) is neither trained nor holdout.
        """
        return [
            item
            for item in self.items
            if not item.was_trained
            and not item.revision_invalid
            and not item.rationale_missing
            and not item.skipped_correct
        ]

    # Items in the train split that were not trained because their baseline
    # was already correct. Their post-training verdict is the interference
    # measure: nothing was done to them, so a regression came from training
    # other items.
    @property
    def skipped_correct_items(self) -> list[ItemResult]:
        return [item for item in self.items if item.skipped_correct]

    @property
    def skipped_correct_count(self) -> int:
        return len(self.skipped_correct_items)

    @property
    def skipped_correct_regressed_count(self) -> int:
        """Skipped-correct items judged wrong after the other items' training."""
        return sum(1 for item in self.skipped_correct_items if item.regressed)

    def interference_summary_text(self) -> str:
        """``Skipped (baseline correct): N; of which M regressed ...`` for logs and reports."""
        return interference_summary_text(
            self.skipped_correct_count, self.skipped_correct_regressed_count
        )

    # Verify-after-target (``EvaluationConfig.verify_steps``).
    @property
    def verified_count(self) -> int:
        """Trained items whose last verification check found the answer right."""
        return sum(1 for item in self.train_items if item.verified is True)

    @property
    def verify_still_wrong_count(self) -> int:
        """Trained items still wrong at the step cap after verification."""
        return sum(1 for item in self.train_items if item.verified is False)

    @property
    def mean_verify_attempts(self) -> float:
        """Mean verification checks per trained item (0 when off)."""
        if not self.train_items:
            return 0.0
        return sum(item.verify_attempts for item in self.train_items) / len(
            self.train_items
        )

    def verification_summary_text(self) -> str:
        """``Verification: K/N items verified ...`` for logs and reports."""
        return verification_summary_text(
            [item.verified for item in self.train_items],
            [item.verify_attempts for item in self.train_items],
        )

    @property
    def revision_invalid_items(self) -> list[ItemResult]:
        """Items scheduled for training whose self-generated revision was rejected."""
        return [item for item in self.items if item.revision_invalid]

    @property
    def revision_invalid_count(self) -> int:
        return len(self.revision_invalid_items)

    # Revision quality (self_generated only). These judge the model's revision
    # *before* training on it, so they say whether the self-correction loop has
    # anything useful to learn from, independent of whether training absorbed it.
    @property
    def revision_valid_items(self) -> list[ItemResult]:
        """Trained items whose self-generated revision passed validation."""
        return [item for item in self.train_items if item.revision_answer is not None]

    @property
    def revision_attempted_count(self) -> int:
        """Items a revision was requested for (valid + invalid)."""
        return len(self.revision_valid_items) + self.revision_invalid_count

    @property
    def revision_valid_count(self) -> int:
        return len(self.revision_valid_items)

    @property
    def revision_correct_count(self) -> int:
        """Valid revisions that contain a key term."""
        return sum(
            1 for item in self.revision_valid_items if item.revision_has_key_terms
        )

    @property
    def revision_fixed_count(self) -> int:
        """Valid revisions that turned a wrong baseline into a right answer."""
        return sum(1 for item in self.revision_valid_items if item.revision_fixed)

    @property
    def revision_broke_count(self) -> int:
        """Valid revisions that turned a right baseline into a wrong answer."""
        return sum(1 for item in self.revision_valid_items if item.revision_broke)

    @property
    def revision_unchanged_text_count(self) -> int:
        """Valid revisions that restated the baseline answer verbatim."""
        return sum(
            1
            for item in self.revision_valid_items
            if item.revision_changed_text is False
        )

    def revision_summary(self) -> dict[str, int]:
        """Counts describing revision quality before training (self_generated)."""
        return {
            "attempted": self.revision_attempted_count,
            "valid": self.revision_valid_count,
            "invalid": self.revision_invalid_count,
            "correct": self.revision_correct_count,
            "fixed": self.revision_fixed_count,
            "broke": self.revision_broke_count,
            "unchanged_text": self.revision_unchanged_text_count,
        }

    def revision_summary_text(self) -> str:
        """One-line version of ``revision_summary`` for logs and reports."""
        s = self.revision_summary()
        return (
            f"Revisions: {s['attempted']} attempted, {s['valid']} valid, of which "
            f"{s['correct']} correct; fixed {s['fixed']} wrong answers, broke "
            f"{s['broke']} right ones ({s['unchanged_text']} restated the baseline "
            "verbatim)."
        )

    @property
    def baseline_accuracy(self) -> float:
        """Fraction of items with key terms in initial response."""
        if not self.items:
            return 0.0
        return sum(1 for item in self.items if item.initial_has_key_terms) / len(
            self.items
        )

    @property
    def train_improvement_rate(self) -> float:
        """Fraction of trained items that improved (gained key terms)."""
        trained = [
            item
            for item in self.items
            if item.was_trained and item.post_has_key_terms is not None
        ]
        if not trained:
            return 0.0
        improved = sum(
            1
            for item in trained
            if item.post_has_key_terms and not item.initial_has_key_terms
        )
        improvable = sum(1 for item in trained if not item.initial_has_key_terms)
        return improved / improvable if improvable > 0 else 1.0

    @property
    def train_retention_rate(self) -> float:
        """Fraction of trained items that retained correctness."""
        trained = [
            item
            for item in self.items
            if item.was_trained and item.post_has_key_terms is not None
        ]
        if not trained:
            return 0.0
        retained = sum(
            1
            for item in trained
            if item.post_has_key_terms and item.initial_has_key_terms
        )
        was_correct = sum(1 for item in trained if item.initial_has_key_terms)
        return retained / was_correct if was_correct > 0 else 1.0

    @property
    def train_post_accuracy(self) -> float:
        """Fraction of trained items with key terms in post response."""
        trained = [
            item
            for item in self.items
            if item.was_trained and item.post_has_key_terms is not None
        ]
        if not trained:
            return 0.0
        return sum(1 for item in trained if item.post_has_key_terms) / len(trained)

    @property
    def _judged_holdout_items(self) -> list[ItemResult]:
        return [
            item for item in self.holdout_items if item.post_has_key_terms is not None
        ]

    @property
    def holdout_total(self) -> int:
        return len(self._judged_holdout_items)

    @property
    def holdout_correct(self) -> int:
        """Holdout items judged correct after training."""
        return sum(1 for item in self._judged_holdout_items if item.post_has_key_terms)

    @property
    def holdout_baseline_correct(self) -> int:
        """Holdout items judged correct before training."""
        return sum(
            1 for item in self._judged_holdout_items if item.initial_has_key_terms
        )

    @property
    def holdout_accuracy(self) -> float:
        """Fraction of holdout items with key terms (post-training baseline check)."""
        if not self.holdout_total:
            return 0.0
        return self.holdout_correct / self.holdout_total

    # Response-length / reasoning-collapse signals. The "empty" think target
    # taught DeepSeek-R1-Distill to skip reasoning globally (766 -> 6 tokens);
    # these make that visible in the summary instead of only in the HTML.
    @property
    def mean_baseline_tokens(self) -> float:
        """Mean raw token count of the baseline responses over all items."""
        if not self.items:
            return 0.0
        return sum(item.initial_token_count for item in self.items) / len(self.items)

    @property
    def mean_post_tokens(self) -> float:
        """Mean raw token count of the post-training responses over judged items."""
        counted = [
            item.post_token_count
            for item in self.items
            if item.post_token_count is not None
        ]
        if not counted:
            return 0.0
        return sum(counted) / len(counted)

    @property
    def post_empty_think_count(self) -> int:
        """Post-training responses that start with ``</think>`` (no reasoning)."""
        return sum(
            1
            for item in self.items
            if item.post_response_raw is not None and item.post_empty_think
        )

    @property
    def post_count(self) -> int:
        """Items with a post-training response."""
        return sum(1 for item in self.items if item.post_response_raw is not None)

    def collapse_summary_text(self) -> str:
        """One-line response-length / empty-think summary for logs and reports."""
        return (
            f"Response length: baseline {self.mean_baseline_tokens:.0f} tok → post "
            f"{self.mean_post_tokens:.0f} tok; empty-think responses after training: "
            f"{self.post_empty_think_count}/{self.post_count}"
        )

    def holdout_summary_text(self) -> str:
        return (
            f"Holdout accuracy: {self.holdout_accuracy:.1%} "
            f"({self.holdout_correct}/{self.holdout_total}, baseline "
            f"{self.holdout_baseline_correct}/{self.holdout_total})"
        )

    # Loss-targeted training: how many steps each correction needed.
    @property
    def mean_train_steps(self) -> float:
        """Mean optimizer steps per trained item's correction."""
        if not self.train_items:
            return 0.0
        return sum(item.train_steps for item in self.train_items) / len(
            self.train_items
        )

    @property
    def mean_train_final_loss(self) -> float | None:
        """Mean final answer loss over trained items that reported one."""
        losses = [
            item.train_final_loss
            for item in self.train_items
            if item.train_final_loss is not None
        ]
        if not losses:
            return None
        return sum(losses) / len(losses)

    @property
    def mean_train_final_train_loss(self) -> float | None:
        """Mean final training (whole-target) loss over trained items."""
        losses = [
            item.train_final_train_loss
            for item in self.train_items
            if item.train_final_train_loss is not None
        ]
        if not losses:
            return None
        return sum(losses) / len(losses)

    @property
    def rationale_missing_items(self) -> list[ItemResult]:
        """Items scheduled for training but skipped for want of a rationale."""
        return [item for item in self.items if item.rationale_missing]

    @property
    def rationale_missing_count(self) -> int:
        return len(self.rationale_missing_items)

    @property
    def rationale_truncated_count(self) -> int:
        """Trained items whose rationale was cut to ``rationale_max_tokens``."""
        return sum(1 for item in self.train_items if item.rationale_truncated)

    @property
    def train_rehearsal_active_steps(self) -> int:
        """``(step, rehearsal example)`` pairs whose gradient was applied, over all trained items."""
        return sum(item.train_rehearsal_active_steps for item in self.train_items)

    @property
    def train_rehearsal_pairs(self) -> int:
        """All ``(step, rehearsal example)`` pairs over trained items."""
        return sum(item.train_rehearsal_pairs for item in self.train_items)

    @property
    def mean_train_rehearsal_final_loss(self) -> float | None:
        """Mean final rehearsal loss over trained items that had rehearsal."""
        losses = [
            item.train_rehearsal_final_loss
            for item in self.train_items
            if item.train_rehearsal_final_loss is not None
        ]
        if not losses:
            return None
        return sum(losses) / len(losses)

    @property
    def train_cap_hit_count(self) -> int:
        """Trained items whose correction ran to the step cap."""
        return sum(1 for item in self.train_items if item.train_hit_cap)

    def training_summary_text(self) -> str:
        """One-line steps / final-loss / cap summary for logs and reports."""
        return training_summary_text(
            [item.train_steps for item in self.train_items],
            [item.train_final_loss for item in self.train_items],
            self.train_cap_hit_count,
            self.config.training_iterations,
            self.config.loss_target,
            [item.train_rehearsal_final_loss for item in self.train_items],
            [item.train_final_train_loss for item in self.train_items],
            self.rationale_missing_count,
            self.train_rehearsal_active_steps,
            self.train_rehearsal_pairs,
        )

    def lora_settings_text(self) -> str:
        """``LoRA: rank 32, layers 24, scale 10.0`` for the model this run used."""
        return lora_settings_text(self.model_kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "training_iterations": self.config.training_iterations,
                "loss_target": self.config.loss_target,
                "epochs_per_call": self.config.epochs_per_call,
                "shuffle": self.config.shuffle,
                "seed": self.config.seed,
                "train_ratio": self.config.train_ratio,
                "training_source": self.config.training_source,
                "revision_prompt": self.config.revision_prompt,
                "think_mode": self.config.think_mode,
                "close_think": self.config.close_think,
                "rehearsal_k": self.config.rehearsal_k,
                "rehearsal_max_tokens": self.config.rehearsal_max_tokens,
                "rehearsal_weight": self.config.rehearsal_weight,
                "rehearsal_margin": self.config.rehearsal_margin,
                "rationale_max_tokens": self.config.rationale_max_tokens,
                "train_correct_items": self.config.train_correct_items,
                "verify_steps": self.config.verify_steps,
            },
            "model_kwargs": self.model_kwargs,
            "lora": dict(
                zip(("rank", "layers", "scale"), lora_settings(self.model_kwargs))
            ),
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "total_time_seconds": self.total_time_seconds,
            "metrics": {
                "baseline_accuracy": self.baseline_accuracy,
                "train_improvement_rate": self.train_improvement_rate,
                "train_retention_rate": self.train_retention_rate,
                "train_post_accuracy": self.train_post_accuracy,
                "holdout_accuracy": self.holdout_accuracy,
                "holdout_correct": self.holdout_correct,
                "holdout_baseline_correct": self.holdout_baseline_correct,
                "train_count": len(self.train_items),
                "holdout_count": len(self.holdout_items),
                "mean_baseline_tokens": self.mean_baseline_tokens,
                "mean_post_tokens": self.mean_post_tokens,
                "post_empty_think_count": self.post_empty_think_count,
                "revision_invalid_count": self.revision_invalid_count,
                "revision_summary": self.revision_summary(),
                "mean_train_steps": self.mean_train_steps,
                "mean_train_final_loss": self.mean_train_final_loss,
                "mean_train_final_train_loss": self.mean_train_final_train_loss,
                "mean_train_rehearsal_final_loss": self.mean_train_rehearsal_final_loss,
                "train_cap_hit_count": self.train_cap_hit_count,
                "rationale_missing_count": self.rationale_missing_count,
                "rationale_truncated_count": self.rationale_truncated_count,
                "train_rehearsal_active_steps": self.train_rehearsal_active_steps,
                "train_rehearsal_pairs": self.train_rehearsal_pairs,
                "skipped_correct_count": self.skipped_correct_count,
                "skipped_correct_regressed_count": self.skipped_correct_regressed_count,
                "verified_count": self.verified_count,
                "verify_still_wrong_count": self.verify_still_wrong_count,
                "mean_verify_attempts": self.mean_verify_attempts,
            },
            "items": [
                {
                    "item_id": item.item_id,
                    "question": item.question,
                    "correct_answer": item.correct_answer,
                    "key_terms": item.key_terms,
                    "initial_response": item.initial_response,
                    "initial_response_raw": item.initial_response_raw,
                    "initial_has_key_terms": item.initial_has_key_terms,
                    "post_response": item.post_response,
                    "post_response_raw": item.post_response_raw,
                    "post_has_key_terms": item.post_has_key_terms,
                    "was_trained": item.was_trained,
                    "training_time_seconds": item.training_time_seconds,
                    "revision_text": item.revision_text,
                    "revision_invalid": item.revision_invalid,
                    "revision_answer": item.revision_answer,
                    "revision_has_key_terms": item.revision_has_key_terms,
                    "revision_changed_text": item.revision_changed_text,
                    "revision_changed_verdict": item.revision_changed_verdict,
                    "initial_token_count": item.initial_token_count,
                    "post_token_count": item.post_token_count,
                    "rehearsal_item_ids": item.rehearsal_item_ids,
                    "rationale_text": item.rationale_text,
                    "rationale_missing": item.rationale_missing,
                    "rationale_tokens": item.rationale_tokens,
                    "rationale_truncated": item.rationale_truncated,
                    "train_steps": item.train_steps,
                    "train_initial_loss": item.train_initial_loss,
                    "train_final_loss": item.train_final_loss,
                    "train_final_train_loss": item.train_final_train_loss,
                    "train_hit_cap": item.train_hit_cap,
                    "train_rehearsal_final_loss": item.train_rehearsal_final_loss,
                    "train_rehearsal_initial_loss": item.train_rehearsal_initial_loss,
                    "train_rehearsal_active_steps": item.train_rehearsal_active_steps,
                    "skipped_correct": item.skipped_correct,
                    "verify_attempts": item.verify_attempts,
                    "verified": item.verified,
                }
                for item in self.items
            ],
        }


def interference_summary_text(skipped: int, regressed: int) -> str:
    """Format the one-line interference summary.

    ``Skipped (baseline correct): 20; of which 4 regressed after other items'
    training``. Shared by the harness and the meta-learning experiment.
    """
    return (
        f"Skipped (baseline correct): {skipped}; of which {regressed} regressed "
        "after other items' training"
    )


def verification_summary_text(
    verified: list[bool | None], attempts: list[int]
) -> str:
    """Format the one-line verify-after-target summary.

    ``Verification: 9/12 items verified correct after training (mean 1.8
    checks/item); 3 still wrong at the cap``. ``verified`` and ``attempts``
    are per trained item; items with ``None`` (verification off) are not
    counted as verified or still wrong.
    """
    n = len(verified)
    ok = sum(1 for v in verified if v is True)
    wrong = sum(1 for v in verified if v is False)
    mean = sum(attempts) / n if n else 0.0
    return (
        f"Verification: {ok}/{n} items verified correct after training "
        f"(mean {mean:.1f} checks/item); {wrong} still wrong at the cap"
    )


def training_summary_text(
    steps: list[int],
    final_losses: list[float | None],
    cap_hits: int,
    cap: int,
    loss_target: float | None,
    rehearsal_final_losses: list[float | None] | None = None,
    train_final_losses: list[float | None] | None = None,
    rationale_missing: int = 0,
    rehearsal_active_steps: int = 0,
    rehearsal_pairs: int = 0,
) -> str:
    """Format the one-line training summary.

    ``Training: mean 3.2 steps/item (cap 12, answer-loss target 0.60), mean
    final answer loss 0.55 (train loss 0.80, rehearsal 0.41); 0 items hit the
    cap; 0 rationales missing; rehearsal active in 33% of steps``. Shared by
    the harness and the meta-learning experiment so both print the same
    line. The parenthesised train loss (whole-target) and rehearsal loss
    each appear only when some item reported one; the ``rehearsal active``
    part (``rehearsal_active_steps / rehearsal_pairs``, pairs being
    ``(step, rehearsal example)``) only when ``rehearsal_pairs > 0``.
    """
    if not steps:
        return "Training: no items trained"
    mean_steps = sum(steps) / len(steps)
    losses = [l for l in final_losses if l is not None]
    loss_text = f"{sum(losses) / len(losses):.2f}" if losses else "n/a"
    extras = []
    train = [l for l in (train_final_losses or []) if l is not None]
    if train:
        extras.append(f"train loss {sum(train) / len(train):.2f}")
    rehearsal = [l for l in (rehearsal_final_losses or []) if l is not None]
    if rehearsal:
        extras.append(f"rehearsal {sum(rehearsal) / len(rehearsal):.2f}")
    if extras:
        loss_text += f" ({', '.join(extras)})"
    target_text = f"{loss_target:.2f}" if loss_target is not None else "off"
    text = (
        f"Training: mean {mean_steps:.1f} steps/item (cap {cap}, answer-loss "
        f"target {target_text}), mean final answer loss {loss_text}; "
        f"{cap_hits} items hit the cap; {rationale_missing} rationales missing"
    )
    if rehearsal_pairs > 0:
        text += (
            f"; rehearsal active in "
            f"{rehearsal_active_steps / rehearsal_pairs:.0%} of steps"
        )
    return text


def _normalize_for_match(text: str) -> str:
    """NFKC-normalize and casefold text so judge comparisons ignore presentation.

    NFKC folds compatibility characters (subscripts like the "₂" in "H₂O",
    full-width forms, ligatures) onto their base characters; casefold is a
    stronger, locale-independent lower().
    """
    return unicodedata.normalize("NFKC", text).casefold()


def contains_key_terms(response: str, key_terms: list[str]) -> bool:
    """Check if response contains any of the key terms.

    Both sides are NFKC-normalized and casefolded, so "H₂O" matches "H2O" and
    "straße" matches "STRASSE". Pure function; no I/O.
    """
    response_norm = _normalize_for_match(response or "")
    return any(_normalize_for_match(term) in response_norm for term in key_terms)


def extract_revision(response: str) -> str:
    """Return the rewritten answer between the ``[[X]]`` and ``[[/X]]`` markers.

    Mirrors the parsing ``revise.make_collated_training_example`` does when it
    builds the training target: the turn index is the smallest ``[[N]]`` in the
    response and the content is whatever sits between the last ``[[N]]`` and the
    last ``[[/N]]``, stripped. Pure function; assumes ``response`` has already
    passed ``validate_revision_response``.

    Raises:
        ValueError: If no ``[[N]]`` marker is present.
    """
    indices = re.findall(r"\[\[([0-9]+)\]\]", response)
    if not indices:
        raise ValueError(f"No [[X]] marker in revision: {response[:100]!r}")
    idx = min(map(int, indices))
    start = None
    end = None
    for match in re.finditer(rf"\[\[{idx}\]\]", response):
        start = match.end()
    for match in re.finditer(rf"\[\[/{idx}\]\]", response):
        end = match.start()
    return response[start:end].strip()


# --------------------------------------------------------------------------
# Shared phase helpers (used by EvaluationHarness and MetaLearningExperiment)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class InferenceRecord:
    """One model response to an item, already judged and persisted."""

    raw: str
    clean: str
    correct: bool
    token_count: int
    truncated: bool


def _infer_and_record(
    model: Any,
    db: Database,
    item: TriviaItem,
    example_id: int,
    experiment_id: int,
    phase: Phase,
    max_tokens: int | None,
    effective_max_tokens: int,
) -> InferenceRecord:
    """Generate a response for ``item``, judge it, and persist it.

    Args:
        model: Anything with ``generate_response`` and ``_tokenizer``.
        db: Database the response row is written to.
        item: The trivia item to answer.
        example_id: DB id of the example row for ``item``.
        experiment_id: DB id of the running experiment.
        phase: Which phase the response belongs to (BASELINE / POST_TRAINING).
        max_tokens: Explicit cap passed to the model (None = model default).
        effective_max_tokens: The cap actually in force, used for truncation.

    Returns:
        The judged response.
    """
    raw = model.generate_response(
        item.question, use_history=False, max_tokens=max_tokens
    )
    raw = raw or ""
    clean = strip_think_tags(raw)
    correct = contains_key_terms(clean, item.key_terms)
    token_count = len(model._tokenizer.encode(raw))
    truncated = token_count >= effective_max_tokens - 1

    db.insert_response(
        Response(
            id=None,
            example_id=example_id,
            experiment_id=experiment_id,
            response_text=clean,
            response_raw=raw,
            confidence=None,
            phase=phase,
            created_at=None,
            token_count=token_count,
            max_tokens=effective_max_tokens,
            truncated=truncated,
        )
    )
    return InferenceRecord(
        raw=raw,
        clean=clean,
        correct=correct,
        token_count=token_count,
        truncated=truncated,
    )


def _judge_only(
    model: Any, item: TriviaItem, max_tokens: int | None
) -> tuple[str, bool]:
    """Generate and judge a response without persisting it (checkpoint probes)."""
    raw = model.generate_response(
        item.question, use_history=False, max_tokens=max_tokens
    )
    clean = strip_think_tags(raw or "")
    return clean, contains_key_terms(clean, item.key_terms)


RATIONALE_PROMPT = (
    "{question}\n\nThe correct answer is: {correct_answer}\n"
    "Reason it through step by step, then state the answer."
)
"""Prompt that asks the model to reason its way to a known answer.

Used by ``think_mode="rationale"`` with ``training_source="ground_truth"``:
the think block of the response becomes the reasoning in the training
target, so the model is trained on reasoning that concludes the label rather
than on its own reasoning that concluded something else.
"""


def make_rationale_prompt(question: str, correct_answer: str) -> str:
    """``RATIONALE_PROMPT`` filled in for one item."""
    return RATIONALE_PROMPT.format(question=question, correct_answer=correct_answer)


@dataclasses.dataclass
class BuiltExample:
    """What ``_build_training_example`` produced for one item.

    Attributes:
        example: The 1-D correction example; batch it with
            ``collate_training_examples``.
        revision_text: The model's raw revision output for "self_generated"
            (think block and markers included); None for "ground_truth".
        rationale_text: The reasoning placed in the target under
            ``think_mode="rationale"``; None otherwise.
        rationale_tokens: Tokens in that rationale; None without one.
        rationale_truncated: The rationale was cut to ``rationale_max_tokens``.
    """

    example: TrainingExample
    revision_text: str | None = None
    rationale_text: str | None = None
    rationale_tokens: int | None = None
    rationale_truncated: bool = False


class RationaleMissingError(Exception):
    """``think_mode="rationale"`` on a think template, but the model's output held no reasoning.

    Raised by ``_build_training_example`` instead of falling back to the
    "empty" target; callers skip the item.
    """


def _build_training_example(
    model: Any,
    item: TriviaItem,
    baseline_response: str,
    training_source: str,
    revision_prompt: str = "default",
    think_mode: str = "rationale",
    rationale_max_tokens: int = DEFAULT_RATIONALE_MAX_TOKENS,
) -> BuiltExample:
    """Build the (unbatched) correction training example for one item.

    With ``think_mode="rationale"`` the reasoning in the target comes from
    the revision generation's own output (``self_generated``; the raw
    revision text is kept for that) or, for ``ground_truth``, from one extra
    generation with ``make_rationale_prompt``. Either way
    ``revise.rationale_from_output`` picks the reasoning out: the think
    block, or the *whole* output when the model never emitted ``</think>``
    (it generates inside the open think block, so such an output is
    reasoning that ran to the cap), capped at ``rationale_max_tokens`` at a
    sentence boundary. If nothing is left the item is not trained:
    ``RationaleMissingError`` is raised (the old fallback to the "empty"
    target is the target proven to collapse the model). Templates without an
    open think tag skip all of this (no extra generation).

    Args:
        model: Anything with ``generate_response`` and ``_tokenizer``.
        item: The trivia item being trained on.
        baseline_response: The model's *raw* baseline answer to ``item.question``
            (think block included, so ``think_mode="baseline"`` can reuse it).
        training_source: "ground_truth" or "self_generated".
        revision_prompt: Preset name passed to ``revision_prompt_preset``; only
            used for "self_generated".
        think_mode: Passed to ``make_revision_training_example``.
        rationale_max_tokens: Token cap on the rationale.

    Returns:
        A ``BuiltExample``.

    Raises:
        InvalidRevisionError: If a self-generated revision fails validation.
            Callers must catch this and skip the training step.
        RationaleMissingError: ``think_mode="rationale"`` on a think template
            with no rationale to be had. Callers must catch this and skip
            the training step.
    """
    validate_training_source(training_source)
    instructions, dialog_style = revision_prompt_preset(revision_prompt)
    interactions = [
        InteractionHistory(
            idx=0,
            user_input=item.question,
            llm_response=baseline_response,
            reviewed=False,
            timestamp=0.0,
        ),
    ]
    tokenizer = model._tokenizer
    if training_source == "ground_truth":
        revision = f"[[0]] {item.correct_answer} [[/0]]"
        revision_text: str | None = None
    else:
        prompt = make_revision_prompt(
            interactions,
            tokenizer,
            instructions=instructions,
            dialog_style=dialog_style,
        )
        revision = model.generate_response(prompt, use_history=False) or ""
        validate_revision_response(revision, num_interactions=len(interactions))
        revision_text = revision

    rationale: str | None = None
    rationale_tokens: int | None = None
    rationale_truncated = False
    messages = [{"role": "user", "content": item.question}]
    if think_mode == "rationale" and template_opens_think(tokenizer, messages):
        if training_source == "self_generated":
            output = revision
        else:
            output = model.generate_response(
                make_rationale_prompt(item.question, item.correct_answer),
                use_history=False,
            )
        rationale, rationale_tokens, rationale_truncated = rationale_from_output(
            output, tokenizer, rationale_max_tokens
        )
        if not rationale:
            raise RationaleMissingError(
                f"no rationale for {item.id}: the model's output held no reasoning"
            )
    example = make_revision_training_example(
        revision, interactions, tokenizer, think_mode=think_mode, rationale=rationale
    )
    return BuiltExample(
        example=example,
        revision_text=revision_text,
        rationale_text=rationale,
        rationale_tokens=rationale_tokens,
        rationale_truncated=rationale_truncated,
    )


def make_rehearsal_example(
    item: TriviaItem, baseline_raw: str, tokenizer: Any
) -> TrainingExample:
    """Self-distillation example: the model's own full baseline output as target.

    Prefix is the chat template for ``item.question``; the target is
    ``{baseline_raw}{eos}`` with the whole target in the loss. For a think
    template ``baseline_raw`` already has the ``{think}</think>\n\n{answer}``
    shape the model produced, so the sequence matches inference exactly.
    """
    eos = tokenizer.special_tokens_map.get("eos_token", "")
    if isinstance(eos, list):
        eos = eos[0]
    messages = [{"role": "user", "content": item.question}]
    return make_training_example(messages, baseline_raw + eos, tokenizer)


def sample_rehearsal_ids(
    pool_ids: list[str], exclude_id: str, k: int, rng: random.Random
) -> list[str]:
    """Pick up to ``k`` ids from ``pool_ids`` (order-stable), never ``exclude_id``."""
    candidates = [i for i in pool_ids if i != exclude_id]
    if k <= 0 or not candidates:
        return []
    return rng.sample(candidates, min(k, len(candidates)))


def _record_training_event(
    db: Database,
    example_id: int,
    experiment_id: int,
    training_iterations: int,
    training_time_seconds: float,
) -> int:
    """Persist a training event row and return its id."""
    return db.insert_training_event(
        TrainingEvent(
            id=None,
            example_id=example_id,
            experiment_id=experiment_id,
            training_iterations=training_iterations,
            training_time_seconds=training_time_seconds,
            created_at=None,
        )
    )


@dataclasses.dataclass
class TrainingOutcome:
    """What happened when one item was scheduled for training."""

    trained: bool
    revision_text: str | None = None
    revision_invalid: bool = False
    revision_error: str | None = None
    training_time_seconds: float = 0.0
    # Parsed content of a valid self-generated revision and its judge verdict,
    # recorded before training so revision quality can be measured on its own.
    revision_answer: str | None = None
    revision_has_key_terms: bool | None = None
    rehearsal_item_ids: list[str] = dataclasses.field(default_factory=list)
    # Reasoning placed in the target under think_mode="rationale"; or, when
    # none could be had, rationale_missing=True and the item was skipped.
    rationale_text: str | None = None
    rationale_missing: bool = False
    rationale_tokens: int | None = None
    rationale_truncated: bool = False
    # Stats for the item's single training call; the losses are the
    # correction's answer loss (what the stop rule watched).
    train_steps: int = 0
    train_initial_loss: float | None = None
    train_final_loss: float | None = None
    train_hit_cap: bool = False
    # Whole-target training loss at the last step.
    train_final_train_loss: float | None = None
    # Mean rehearsal loss at the last step (None without rehearsal), at the
    # first step (the hinge anchor), and the (step, example) pairs whose
    # rehearsal gradient the hinge let through.
    train_rehearsal_final_loss: float | None = None
    train_rehearsal_initial_loss: float | None = None
    train_rehearsal_active_steps: int = 0
    # The full stats of every training call made for the item: the first
    # (loss-targeted) call, then one per verify-after-target round.
    training_stats: list[TrainingStats] = dataclasses.field(default_factory=list)
    # Verify-after-target: generate-and-judge checks made, and whether the
    # last one passed (None when verification is off).
    verify_attempts: int = 0
    verified: bool | None = None

    @property
    def train_rehearsal_pairs(self) -> int:
        """``steps * k``: what ``train_rehearsal_active_steps`` is out of."""
        return self.train_steps * len(self.rehearsal_item_ids)

    def verification_text(self) -> str:
        """``verified ✓ (2 checks)`` / ``not verified at cap (3 checks)``; empty when off."""
        if self.verified is None:
            return ""
        checks = f"{self.verify_attempts} check{'s' if self.verify_attempts != 1 else ''}"
        if self.verified:
            return f"verified ✓ ({checks})"
        return f"not verified at cap ({checks})"

    def training_text(self) -> str:
        """``3 steps, loss 6.05 → 0.58 (train 0.80, rehearsal 0.38→0.36 (active 3/9))`` for the verbose per-item line."""
        if self.train_initial_loss is None or self.train_final_loss is None:
            text = f"{self.train_steps} steps"
            if self.verified is not None:
                text += f", {self.verification_text()}"
            return text
        text = (
            f"{self.train_steps} steps, loss {self.train_initial_loss:.2f} → "
            f"{self.train_final_loss:.2f}"
        )
        extras = []
        if (
            self.train_final_train_loss is not None
            and self.train_final_train_loss != self.train_final_loss
        ):
            extras.append(f"train {self.train_final_train_loss:.2f}")
        if self.train_rehearsal_final_loss is not None:
            initial = (
                f"{self.train_rehearsal_initial_loss:.2f}→"
                if self.train_rehearsal_initial_loss is not None
                else ""
            )
            extras.append(
                f"rehearsal {initial}{self.train_rehearsal_final_loss:.2f} "
                f"(active {self.train_rehearsal_active_steps}/"
                f"{self.train_rehearsal_pairs})"
            )
        if extras:
            text += f" ({', '.join(extras)})"
        if self.verified is not None:
            text += f", {self.verification_text()}"
        return text


def _merge_training_stats(calls: list[TrainingStats]) -> TrainingStats:
    """Fold the stats of consecutive training calls on one example into one.

    Steps, per-step losses, and rehearsal active pairs are summed or
    concatenated; the initial losses come from the first call and the final
    ones from the last. ``stopped_early`` is the last call's, so
    ``hit_cap`` is only meaningful for a single call: with verification the
    caller decides the cap verdict from the verification outcome.
    """
    if len(calls) == 1:
        return calls[0]
    first, last = calls[0], calls[-1]
    return TrainingStats(
        steps=sum(c.steps for c in calls),
        initial_loss=first.initial_loss,
        final_loss=last.final_loss,
        stopped_early=last.stopped_early,
        losses=[l for c in calls for l in c.losses],
        rehearsal_final_loss=last.rehearsal_final_loss,
        rehearsal_count=first.rehearsal_count,
        final_train_loss=getattr(last, "final_train_loss", float("nan")),
        train_losses=[l for c in calls for l in getattr(c, "train_losses", [])],
        rehearsal_initial_loss=getattr(first, "rehearsal_initial_loss", None),
        rehearsal_active_steps=sum(
            getattr(c, "rehearsal_active_steps", 0) for c in calls
        ),
    )


def _verify_answer(model: Any, item: TriviaItem, max_tokens: int | None) -> bool:
    """Generate the item's answer and judge it; nothing is persisted."""
    raw = model.generate_response(
        item.question, use_history=False, max_tokens=max_tokens
    )
    return contains_key_terms(strip_think_tags(raw or ""), item.key_terms)


def _train_one_item(
    model: Any,
    db: Database,
    item: TriviaItem,
    baseline_response: str,
    example_id: int,
    experiment_id: int,
    training_iterations: int,
    training_source: str,
    revision_prompt: str = "default",
    think_mode: str = "rationale",
    rehearsal: list[tuple[TriviaItem, str]] | None = None,
    loss_target: float | None = None,
    rehearsal_weight: float = 1.0,
    rehearsal_margin: float = 0.05,
    rationale_max_tokens: int = DEFAULT_RATIONALE_MAX_TOKENS,
    verify_steps: int = 0,
    max_tokens: int | None = None,
) -> TrainingOutcome:
    """Build the target, train, and record the event for one item.

    ``baseline_response`` is the raw baseline output (think block included).
    ``rehearsal`` is a list of ``(item, baseline_raw)`` pairs turned into
    self-distillation examples. With rehearsal the item is one
    ``model.train_on_examples(correction, rehearsal, ...)`` call: every step
    combines the correction's gradient with ``rehearsal_weight`` times the
    mean rehearsal gradient, each from its own single-row pass, so peak
    memory is bounded by one sequence (padding a batch of rehearsal targets
    that can run to the generation cap exhausted a 16 GB machine). Without
    rehearsal it is one ``model.train_on_example(correction, ...)`` call.
    Either way the call gets ``loss_target`` and ``max_steps =
    training_iterations`` and stops as soon as the correction's *answer*
    loss is below the target; the whole-target training loss and the
    rehearsal loss are only reported. One training event is recorded for the
    item with ``training_iterations`` set to the steps actually taken.

    ``rehearsal_margin`` is the rehearsal hinge (``_llm.active_rehearsal``)
    and ``rationale_max_tokens`` the cap on the rationale in the target.

    With ``verify_steps > 0`` the loss-targeted call is followed by a
    verify-after-target loop: generate the item's answer (``max_tokens`` as
    at inference) and judge it; while it is wrong and the total steps are
    under ``training_iterations``, train ``verify_steps`` more steps (clipped
    to the remaining budget) on the same examples with ``loss_target=None``
    and check again. The loop stops at the first passing check. The outcome's
    ``train_steps`` is the total over every call, ``verify_attempts`` the
    number of checks, ``verified`` the last verdict, and ``train_hit_cap`` is
    then "still wrong at the cap". The checks are not persisted anywhere.

    On ``InvalidRevisionError`` the item is not trained and the outcome carries
    ``revision_invalid=True`` plus the error text; on ``RationaleMissingError``
    it is not trained and carries ``rationale_missing=True``.
    """
    try:
        built = _build_training_example(
            model,
            item,
            baseline_response,
            training_source,
            revision_prompt,
            think_mode,
            rationale_max_tokens,
        )
    except InvalidRevisionError as e:
        return TrainingOutcome(
            trained=False, revision_invalid=True, revision_error=str(e)
        )
    except RationaleMissingError:
        return TrainingOutcome(trained=False, rationale_missing=True)
    correction = built.example
    revision_text = built.revision_text
    tokenizer = model._tokenizer
    rehearsal = rehearsal or []
    correction_row = collate_training_examples([correction], tokenizer)
    rehearsal_rows = [
        collate_training_examples(
            [make_rehearsal_example(r_item, r_raw, tokenizer)], tokenizer
        )
        for r_item, r_raw in rehearsal
    ]

    revision_answer: str | None = None
    revision_has_key_terms: bool | None = None
    if revision_text is not None:
        revision_answer = extract_revision(revision_text)
        revision_has_key_terms = contains_key_terms(revision_answer, item.key_terms)

    def train_call(target: float | None, steps: int) -> TrainingStats:
        if rehearsal_rows:
            return model.train_on_examples(
                correction_row,
                rehearsal_rows,
                loss_target=target,
                max_steps=steps,
                rehearsal_weight=rehearsal_weight,
                rehearsal_margin=rehearsal_margin,
            )
        return model.train_on_example(
            correction_row,
            iterations=steps,
            loss_target=target,
            max_steps=steps,
        )

    train_start = time.time()
    calls = [train_call(loss_target, training_iterations)]
    total_steps = calls[0].steps
    verify_attempts = 0
    verified: bool | None = None
    if verify_steps > 0:
        # Verify-after-target: the loss target says the teacher-forced answer
        # is cheap, not that the free-running model produces it. Generate and
        # check; while wrong and under the cap, train a few more steps.
        while True:
            verify_attempts += 1
            verified = _verify_answer(model, item, max_tokens)
            if verified or total_steps >= training_iterations:
                break
            extra = min(verify_steps, training_iterations - total_steps)
            calls.append(train_call(None, extra))
            total_steps += calls[-1].steps
    elapsed = time.time() - train_start
    stats = _merge_training_stats(calls)
    hit_cap = stats.hit_cap if verified is None else not verified
    _record_training_event(db, example_id, experiment_id, stats.steps, elapsed)
    return TrainingOutcome(
        trained=True,
        revision_text=revision_text,
        training_time_seconds=elapsed,
        revision_answer=revision_answer,
        revision_has_key_terms=revision_has_key_terms,
        rehearsal_item_ids=[r_item.id for r_item, _ in rehearsal],
        rationale_text=built.rationale_text,
        rationale_tokens=built.rationale_tokens,
        rationale_truncated=built.rationale_truncated,
        train_steps=stats.steps,
        train_initial_loss=stats.initial_loss,
        train_final_loss=stats.final_loss,
        train_hit_cap=hit_cap,
        train_final_train_loss=_final_train_loss(stats),
        train_rehearsal_final_loss=stats.rehearsal_final_loss,
        train_rehearsal_initial_loss=getattr(stats, "rehearsal_initial_loss", None),
        train_rehearsal_active_steps=getattr(stats, "rehearsal_active_steps", 0),
        training_stats=calls,
        verify_attempts=verify_attempts,
        verified=verified,
    )


def _final_train_loss(stats: TrainingStats) -> float | None:
    """``stats.final_train_loss`` as ``None`` when NaN or absent (stub stats)."""
    value = getattr(stats, "final_train_loss", None)
    if value is None or value != value:
        return None
    return float(value)


def _insert_dataset_examples(db: Database, dataset: TriviaDataset) -> dict[str, int]:
    """Insert every item as an Example row; return item_id -> example_id."""
    example_ids: dict[str, int] = {}
    for item in dataset:
        example_ids[item.id] = db.insert_example(
            Example(
                id=None,
                canonical_id=item.id,  # Use the trivia item ID as canonical
                question=item.question,
                ground_truth_answer=item.correct_answer,
                key_terms=item.key_terms if item.key_terms else None,
                category=item.category,
                difficulty=item.difficulty,
                source_type=SourceType.STATIC_TRIVIA,
                source_url=None,
                source_title=None,
                valid_at=None,  # Static trivia is timeless
                created_at=None,
            )
        )
    return example_ids


class EvaluationHarness:
    """Runs evaluation experiments on a dataset."""

    def __init__(
        self,
        model: StatefulLLM | None = None,
        db: Database | None = None,
        db_path: Path | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize harness with optional pre-loaded model and database.

        Args:
            model: Pre-loaded StatefulLLM. If None, will be loaded on first use.
            db: Pre-initialized Database. If None, will be created using db_path.
            db_path: Path to SQLite database. If None, uses default location.
            model_kwargs: Keyword arguments for the lazily constructed
                ``StatefulLLM`` (e.g. ``{"learning_rate": 1e-5}``). Ignored when
                ``model`` is given; recorded in the experiment's config_json.
        """
        self._model = model
        self._model_loaded = model is not None
        self._model_kwargs = dict(model_kwargs or {})
        if db is not None:
            self._db = db
        elif db_path is not None:
            self._db = Database(db_path)
        else:
            self._db = Database()  # Use default path

    @property
    def model(self) -> StatefulLLM:
        """Lazy-load model on first access."""
        if self._model is None:
            print("Loading model...")
            self._model = StatefulLLM(**self._model_kwargs)
            self._model._model_is_stable = True
            self._model_loaded = True
        return self._model

    def run(
        self,
        dataset: TriviaDataset,
        config: EvaluationConfig | None = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """Run a full evaluation on the dataset.

        Process:
        1. Get baseline responses for all items
        2. Split into train/holdout sets
        3. Train on training items sequentially
        4. Get post-training responses for all items

        All results are persisted to the database in addition to being
        returned as an EvaluationResult.
        """
        if config is None:
            config = EvaluationConfig()

        start_time = time.time()
        result = EvaluationResult(
            config=config,
            dataset_name=dataset.name,
            timestamp=datetime.now().isoformat(),
            model_kwargs=dict(self._model_kwargs),
        )

        # Create experiment record in database
        experiment = Experiment(
            id=None,
            name=config.name,
            experiment_type=ExperimentType.EVAL,
            config_json=json.dumps(
                {
                    "name": config.name,
                    "training_iterations": config.training_iterations,
                    "loss_target": config.loss_target,
                    "epochs_per_call": config.epochs_per_call,
                    "shuffle": config.shuffle,
                    "seed": config.seed,
                    "train_ratio": config.train_ratio,
                    "max_tokens": config.max_tokens,
                    "training_source": config.training_source,
                    "revision_prompt": config.revision_prompt,
                    "think_mode": config.think_mode,
                    "close_think": config.close_think,
                    "rehearsal_k": config.rehearsal_k,
                    "rehearsal_max_tokens": config.rehearsal_max_tokens,
                    "rehearsal_weight": config.rehearsal_weight,
                    "rehearsal_margin": config.rehearsal_margin,
                    "rationale_max_tokens": config.rationale_max_tokens,
                    "train_correct_items": config.train_correct_items,
                    "verify_steps": config.verify_steps,
                    "model_kwargs": self._model_kwargs,
                    "lora": dict(
                        zip(("rank", "layers", "scale"), lora_settings(self._model_kwargs))
                    ),
                    "dataset_name": dataset.name,
                    "dataset_version": dataset.version,
                }
            ),
            model_checkpoint=None,
            started_at=datetime.now(),
            completed_at=None,
        )
        experiment_id = self._db.insert_experiment(experiment)

        # Prepare item ordering
        indices = list(range(len(dataset)))
        if config.shuffle:
            random.seed(config.seed)
            random.shuffle(indices)

        # Split into train/holdout
        train_count = int(len(indices) * config.train_ratio)
        train_indices = set(indices[:train_count])

        if verbose:
            print(f"Dataset: {dataset.name} ({len(dataset)} items)")
            print(f"Train: {train_count}, Holdout: {len(dataset) - train_count}")
            print(f"Config: {config.name}")
            print(f"Training source: {config.training_source}")
            if config.training_source == "self_generated":
                print(f"Revision prompt: {config.revision_prompt}")
            print(
                f"Think mode: {config.think_mode} "
                f"(rationale max tokens: {config.rationale_max_tokens})"
            )
            print(
                f"Rehearsal k: {config.rehearsal_k} "
                f"(max tokens: {config.rehearsal_max_tokens}, "
                f"weight: {config.rehearsal_weight:g}, "
                f"margin: {config.rehearsal_margin:g})"
            )
            print(
                f"Loss target: {config.loss_target} "
                f"(step cap: {config.training_iterations})"
            )
            print(
                f"Train correct items: {config.train_correct_items}; "
                f"verify steps: {config.verify_steps}"
            )
            print(lora_settings_text(self._model_kwargs))
            if self._model_kwargs:
                print(f"Model kwargs: {self._model_kwargs}")
            print(f"Experiment ID: {experiment_id}")
            print()

        example_ids = _insert_dataset_examples(self._db, dataset)

        # Phase 1: Baseline inference
        if verbose:
            print("=" * 60)
            print("PHASE 1: Baseline Inference")
            print("=" * 60)

        # Get effective max tokens (model is lazy-loaded here)
        effective_max_tokens = config.max_tokens or self.model._max_tokens

        item_results: dict[str, ItemResult] = {}
        for i, idx in enumerate(indices):
            item = dataset[idx]
            if verbose:
                print(f"  [{i+1}/{len(dataset)}] {item.id}: {item.question[:40]}...")

            rec = _infer_and_record(
                self.model,
                self._db,
                item,
                example_ids[item.id],
                experiment_id,
                Phase.BASELINE,
                config.max_tokens,
                effective_max_tokens,
            )
            item_results[item.id] = ItemResult(
                item_id=item.id,
                question=item.question,
                correct_answer=item.correct_answer,
                key_terms=item.key_terms,
                initial_response=rec.clean,
                initial_response_raw=rec.raw,
                initial_has_key_terms=rec.correct,
                was_trained=(idx in train_indices),
                initial_token_count=rec.token_count,
            )

            if verbose:
                status = "✓" if rec.correct else "✗"
                print(f"       {status} Key terms: {rec.correct}")

        # Phase 2: Training on train set
        if verbose:
            print()
            print("=" * 60)
            print(f"PHASE 2: Training (source={config.training_source})")
            print("=" * 60)

        train_items = [(dataset[idx], idx) for idx in indices if idx in train_indices]
        # Rehearsal pool: trained-split items whose baseline was judged correct
        # and is short enough to train on. Holdout items never enter it, so
        # holdout stays untouched by training.
        rehearsal_pool = [
            item.id
            for item, _ in train_items
            if item_results[item.id].initial_has_key_terms
            and item_results[item.id].initial_token_count <= config.rehearsal_max_tokens
        ]
        items_by_id = {item.id: item for item, _ in train_items}
        for i, (item, idx) in enumerate(train_items):
            item_result = item_results[item.id]
            if verbose:
                print(f"  [{i+1}/{len(train_items)}] Training on {item.id}...")

            if not config.train_correct_items and item_result.initial_has_key_terms:
                # Nothing to correct: skip it, and let its post-training
                # verdict measure interference from the other items.
                item_result.skipped_correct = True
                item_result.was_trained = False
                if verbose:
                    print("       Skipped: baseline correct (not trained)")
                continue

            rehearsal_ids = sample_rehearsal_ids(
                rehearsal_pool,
                item.id,
                config.rehearsal_k,
                random.Random(config.seed + i),
            )
            rehearsal = [
                (items_by_id[rid], item_results[rid].initial_response_raw)
                for rid in rehearsal_ids
            ]
            outcome = _train_one_item(
                self.model,
                self._db,
                item,
                item_result.initial_response_raw,
                example_ids[item.id],
                experiment_id,
                config.training_iterations,
                config.training_source,
                config.revision_prompt,
                config.think_mode,
                rehearsal,
                loss_target=config.loss_target,
                rehearsal_weight=config.rehearsal_weight,
                rehearsal_margin=config.rehearsal_margin,
                rationale_max_tokens=config.rationale_max_tokens,
                verify_steps=config.verify_steps,
                max_tokens=config.max_tokens,
            )
            item_result.revision_text = outcome.revision_text
            item_result.revision_invalid = outcome.revision_invalid
            item_result.training_time_seconds = outcome.training_time_seconds
            item_result.verify_attempts = outcome.verify_attempts
            item_result.verified = outcome.verified
            item_result.rehearsal_item_ids = outcome.rehearsal_item_ids
            item_result.rationale_text = outcome.rationale_text
            item_result.rationale_missing = outcome.rationale_missing
            item_result.rationale_tokens = outcome.rationale_tokens
            item_result.rationale_truncated = outcome.rationale_truncated
            item_result.train_steps = outcome.train_steps
            item_result.train_initial_loss = outcome.train_initial_loss
            item_result.train_final_loss = outcome.train_final_loss
            item_result.train_final_train_loss = outcome.train_final_train_loss
            item_result.train_hit_cap = outcome.train_hit_cap
            item_result.train_rehearsal_final_loss = outcome.train_rehearsal_final_loss
            item_result.train_rehearsal_initial_loss = (
                outcome.train_rehearsal_initial_loss
            )
            item_result.train_rehearsal_active_steps = (
                outcome.train_rehearsal_active_steps
            )
            # An item whose revision was rejected, or that had no rationale,
            # was never trained on; keep it out of the train metrics but flag
            # it so it is counted.
            item_result.was_trained = outcome.trained
            if outcome.revision_answer is not None:
                item_result.revision_answer = outcome.revision_answer
                item_result.revision_has_key_terms = outcome.revision_has_key_terms
                item_result.revision_changed_text = (
                    outcome.revision_answer != item_result.initial_response
                )
                item_result.revision_changed_verdict = (
                    outcome.revision_has_key_terms != item_result.initial_has_key_terms
                )

            if verbose:
                if outcome.revision_invalid:
                    print(
                        f"       Skipped: invalid revision ({outcome.revision_error})"
                    )
                elif outcome.rationale_missing:
                    print("       Skipped: no rationale")
                else:
                    extra = (
                        f", rehearsal {outcome.rehearsal_item_ids}"
                        if outcome.rehearsal_item_ids
                        else ""
                    )
                    print(
                        f"       Trained ({outcome.training_text()}, "
                        f"{outcome.training_time_seconds:.1f}s{extra})"
                    )
                    if outcome.rationale_truncated:
                        print(
                            f"       Rationale truncated to {outcome.rationale_tokens} "
                            f"tokens (cap {config.rationale_max_tokens})"
                        )
                    if outcome.revision_answer is not None:
                        was = "✓" if item_result.initial_has_key_terms else "✗"
                        now = "✓" if outcome.revision_has_key_terms else "✗"
                        same = "" if item_result.revision_changed_text else " (verbatim)"
                        print(f"       Revision judged {was} → {now}{same}")

        # Phase 3: Post-training inference
        if verbose:
            print()
            print("=" * 60)
            print("PHASE 3: Post-Training Inference")
            print("=" * 60)

        for i, idx in enumerate(indices):
            item = dataset[idx]
            item_result = item_results[item.id]
            if verbose:
                print(f"  [{i+1}/{len(dataset)}] {item.id}: {item.question[:40]}...")

            rec = _infer_and_record(
                self.model,
                self._db,
                item,
                example_ids[item.id],
                experiment_id,
                Phase.POST_TRAINING,
                config.max_tokens,
                effective_max_tokens,
            )
            item_result.post_response = rec.clean
            item_result.post_response_raw = rec.raw
            item_result.post_has_key_terms = rec.correct
            item_result.post_token_count = rec.token_count

            if verbose:
                was = "✓" if item_result.initial_has_key_terms else "✗"
                now = "✓" if rec.correct else "✗"
                if item_result.was_trained:
                    trained = "(trained)"
                elif item_result.skipped_correct:
                    trained = "(skipped: baseline correct)"
                else:
                    trained = "(holdout)"
                print(f"       {was} → {now} {trained}")

        # Compile results
        result.items = list(item_results.values())
        result.total_time_seconds = time.time() - start_time

        # Mark experiment as completed
        self._db.complete_experiment(experiment_id)

        self.model._model_is_stable = True

        if verbose:
            print()
            print("=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Training source: {config.training_source}")
            print(f"Total time: {result.total_time_seconds:.1f}s")
            print(f"Baseline accuracy: {result.baseline_accuracy:.1%}")
            n_train = len(result.train_items)
            print(f"Train post-accuracy: {result.train_post_accuracy:.1%} (n={n_train})")
            print(f"Train improvement rate: {result.train_improvement_rate:.1%} (n={n_train})")
            print(f"Train retention rate: {result.train_retention_rate:.1%} (n={n_train})")
            print(result.holdout_summary_text())
            print(result.interference_summary_text())
            if config.verify_steps > 0:
                print(result.verification_summary_text())
            if config.training_source == "self_generated":
                print(f"Revision prompt: {config.revision_prompt}")
                print(f"Invalid revisions (skipped): {result.revision_invalid_count}")
                print(result.revision_summary_text())
            print(
                f"Think mode: {config.think_mode} "
                f"(rationale max tokens: {config.rationale_max_tokens}, "
                f"{result.rationale_missing_count} skipped for no rationale, "
                f"{result.rationale_truncated_count} truncated); "
                f"rehearsal k: {config.rehearsal_k} "
                f"(max tokens: {config.rehearsal_max_tokens}, "
                f"weight: {config.rehearsal_weight:g}, "
                f"margin: {config.rehearsal_margin:g})"
            )
            print(result.lora_settings_text())
            print(result.collapse_summary_text())
            print(result.training_summary_text())
            print()
            print(f"Results saved to database (experiment_id={experiment_id})")

        return result

    @property
    def db(self) -> Database:
        """Access the database for queries."""
        return self._db
