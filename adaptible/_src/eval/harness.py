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
from .._llm import StatefulLLM
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
    THINK_MODES,
    InvalidRevisionError,
    collate_training_examples,
    make_revision_prompt,
    make_revision_training_example,
    make_training_example,
    resolve_think_mode,
    revision_prompt_preset,
    strip_think_tags,
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
            "baseline"). "baseline" keeps the model's own reasoning in the
            unmasked prefix and trains only on the corrected answer.
        close_think: Deprecated alias for ``think_mode``. ``False`` forces
            ``think_mode="none"`` (the pre-1.0.0a4 target, for comparison);
            ``None``/``True`` defer to ``think_mode``. After construction it is
            always ``think_mode != "none"``.
        rehearsal_k: When > 0, every training step batches the correction with
            ``k`` rehearsal examples: other *trained-split* items whose baseline
            answer was judged correct, with the model's own full baseline output
            as the target (self-distillation). Sampled with
            ``seed + item index``; the item being corrected is never in its own
            rehearsal set. Holdout items are never used. The correction and each
            rehearsal example are trained as separate single-row calls (see
            ``_train_one_item``), so memory is bounded by one sequence.
        rehearsal_max_tokens: Items whose raw baseline response is longer than
            this many tokens are excluded from the rehearsal pool. Rehearsal
            targets are the model's own full baseline output, which can run to
            the generation cap; this keeps every training sequence bounded.
    """

    name: str = "default"
    training_iterations: int = 25  # Total iterations per example (epochs * calls)
    epochs_per_call: int = 5  # Matches StatefulLLM._epochs
    shuffle: bool = False
    seed: int = 42
    train_ratio: float = 0.8  # Fraction to use for training
    max_tokens: int | None = None  # Use model default if None
    training_source: str = "ground_truth"
    revision_prompt: str = "default"
    think_mode: str = "baseline"
    close_think: bool | None = None
    rehearsal_k: int = 0
    rehearsal_max_tokens: int = 768

    def __post_init__(self) -> None:
        validate_training_source(self.training_source)
        revision_prompt_preset(self.revision_prompt)  # raises ValueError if unknown
        self.think_mode = resolve_think_mode(self.think_mode, self.close_think)
        self.close_think = self.think_mode != "none"
        validate_rehearsal_k(self.rehearsal_k)
        validate_rehearsal_max_tokens(self.rehearsal_max_tokens)


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
    revision_text: str | None = None
    revision_invalid: bool = False
    revision_answer: str | None = None
    revision_has_key_terms: bool | None = None
    revision_changed_text: bool | None = None
    revision_changed_verdict: bool | None = None
    initial_token_count: int = 0
    post_token_count: int | None = None
    rehearsal_item_ids: list[str] = dataclasses.field(default_factory=list)

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


@dataclasses.dataclass
class EvaluationResult:
    """Complete results from an evaluation run."""

    config: EvaluationConfig
    dataset_name: str
    timestamp: str
    items: list[ItemResult] = dataclasses.field(default_factory=list)
    total_time_seconds: float = 0.0

    # Computed metrics
    @property
    def train_items(self) -> list[ItemResult]:
        return [item for item in self.items if item.was_trained]

    @property
    def holdout_items(self) -> list[ItemResult]:
        """Items never scheduled for training (invalid-revision items excluded)."""
        return [
            item
            for item in self.items
            if not item.was_trained and not item.revision_invalid
        ]

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "training_iterations": self.config.training_iterations,
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
            },
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
                }
                for item in self.items
            ],
        }


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


def _build_training_example(
    model: Any,
    item: TriviaItem,
    baseline_response: str,
    training_source: str,
    revision_prompt: str = "default",
    think_mode: str = "baseline",
) -> tuple[TrainingExample, str | None]:
    """Build the (unbatched) correction training example for one item.

    Args:
        model: Anything with ``generate_response`` and ``_tokenizer``.
        item: The trivia item being trained on.
        baseline_response: The model's *raw* baseline answer to ``item.question``
            (think block included, so ``think_mode="baseline"`` can reuse it).
        training_source: "ground_truth" or "self_generated".
        revision_prompt: Preset name passed to ``revision_prompt_preset``; only
            used for "self_generated".
        think_mode: Passed to ``make_revision_training_example``.

    Returns:
        ``(example, revision_text)``. ``revision_text`` is the model's raw
        revision output for "self_generated" and None for "ground_truth".
        ``example`` is 1-D; batch it with ``collate_training_examples``.

    Raises:
        InvalidRevisionError: If a self-generated revision fails validation.
            Callers must catch this and skip the training step.
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
    example = make_revision_training_example(
        revision, interactions, tokenizer, think_mode=think_mode
    )
    return example, revision_text


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
    think_mode: str = "baseline",
    rehearsal: list[tuple[TriviaItem, str]] | None = None,
) -> TrainingOutcome:
    """Build the target, train, and record the event for one item.

    ``baseline_response`` is the raw baseline output (think block included).
    ``rehearsal`` is a list of ``(item, baseline_raw)`` pairs whose self-
    distillation examples are trained after the correction. Every example is
    its own single-row ``train_on_example`` call (correction first, then the
    rehearsal examples in order) so peak memory is bounded by one sequence;
    padding a batch of rehearsal targets that can run to the generation cap
    exhausted a 16 GB machine. One training event is recorded for the item,
    with ``training_time_seconds`` summed over the calls.

    On ``InvalidRevisionError`` the item is not trained and the outcome carries
    ``revision_invalid=True`` plus the error text.
    """
    try:
        correction, revision_text = _build_training_example(
            model, item, baseline_response, training_source, revision_prompt, think_mode
        )
    except InvalidRevisionError as e:
        return TrainingOutcome(
            trained=False, revision_invalid=True, revision_error=str(e)
        )
    tokenizer = model._tokenizer
    rehearsal = rehearsal or []
    examples = [correction] + [
        make_rehearsal_example(r_item, r_raw, tokenizer) for r_item, r_raw in rehearsal
    ]

    revision_answer: str | None = None
    revision_has_key_terms: bool | None = None
    if revision_text is not None:
        revision_answer = extract_revision(revision_text)
        revision_has_key_terms = contains_key_terms(revision_answer, item.key_terms)

    train_start = time.time()
    for example in examples:
        batched = collate_training_examples([example], tokenizer)
        model.train_on_example(batched, iterations=training_iterations)
    elapsed = time.time() - train_start
    _record_training_event(db, example_id, experiment_id, training_iterations, elapsed)
    return TrainingOutcome(
        trained=True,
        revision_text=revision_text,
        training_time_seconds=elapsed,
        revision_answer=revision_answer,
        revision_has_key_terms=revision_has_key_terms,
        rehearsal_item_ids=[r_item.id for r_item, _ in rehearsal],
    )


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
                    "model_kwargs": self._model_kwargs,
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
            print(f"Think mode: {config.think_mode}")
            print(
                f"Rehearsal k: {config.rehearsal_k} "
                f"(max tokens: {config.rehearsal_max_tokens})"
            )
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
            )
            item_result.revision_text = outcome.revision_text
            item_result.revision_invalid = outcome.revision_invalid
            item_result.training_time_seconds = outcome.training_time_seconds
            item_result.rehearsal_item_ids = outcome.rehearsal_item_ids
            # An item whose revision was rejected was never trained on; keep it
            # out of the train metrics but flag it so it is counted.
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
                else:
                    extra = (
                        f", rehearsal {outcome.rehearsal_item_ids}"
                        if outcome.rehearsal_item_ids
                        else ""
                    )
                    print(f"       Trained ({outcome.training_time_seconds:.1f}s{extra})")
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
                trained = "(trained)" if item_result.was_trained else "(holdout)"
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
            if config.training_source == "self_generated":
                print(f"Revision prompt: {config.revision_prompt}")
                print(f"Invalid revisions (skipped): {result.revision_invalid_count}")
                print(result.revision_summary_text())
            print(
                f"Think mode: {config.think_mode}; rehearsal k: {config.rehearsal_k} "
                f"(max tokens: {config.rehearsal_max_tokens})"
            )
            print(result.collapse_summary_text())
            print()
            print(f"Results saved to database (experiment_id={experiment_id})")

        return result

    @property
    def db(self) -> Database:
        """Access the database for queries."""
        return self._db
