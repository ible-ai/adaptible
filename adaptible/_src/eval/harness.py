"""Evaluation harness for running experiments."""

import dataclasses
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .._classes import InteractionHistory
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
from ..revise import make_collated_training_example, strip_think_tags
from .dataset import TriviaDataset


@dataclasses.dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    name: str = "default"
    training_iterations: int = 25  # Total iterations per example (epochs * calls)
    epochs_per_call: int = 5  # Matches StatefulLLM._epochs
    shuffle: bool = False
    seed: int = 42
    train_ratio: float = 0.8  # Fraction to use for training
    max_tokens: int | None = None  # Use model default if None


@dataclasses.dataclass
class ItemResult:
    """Result for a single trivia item."""

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
        return [item for item in self.items if not item.was_trained]

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
    def holdout_accuracy(self) -> float:
        """Fraction of holdout items with key terms (post-training baseline check)."""
        holdout = [
            item
            for item in self.items
            if not item.was_trained and item.post_has_key_terms is not None
        ]
        if not holdout:
            return 0.0
        return sum(1 for item in holdout if item.post_has_key_terms) / len(holdout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "training_iterations": self.config.training_iterations,
                "epochs_per_call": self.config.epochs_per_call,
                "shuffle": self.config.shuffle,
                "seed": self.config.seed,
                "train_ratio": self.config.train_ratio,
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
                "train_count": len(self.train_items),
                "holdout_count": len(self.holdout_items),
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
                }
                for item in self.items
            ],
        }


def contains_key_terms(response: str, key_terms: list[str]) -> bool:
    """Check if response contains any of the key terms."""
    response_lower = response.lower()
    return any(term.lower() in response_lower for term in key_terms)


class EvaluationHarness:
    """Runs evaluation experiments on a dataset."""

    def __init__(
        self,
        model: StatefulLLM | None = None,
        db: Database | None = None,
        db_path: Path | str | None = None,
    ):
        """Initialize harness with optional pre-loaded model and database.

        Args:
            model: Pre-loaded StatefulLLM. If None, will be loaded on first use.
            db: Pre-initialized Database. If None, will be created using db_path.
            db_path: Path to SQLite database. If None, uses default location.
        """
        self._model = model
        self._model_loaded = model is not None
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
            self._model = StatefulLLM()
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
            print(f"Experiment ID: {experiment_id}")
            print()

        # Insert all examples into database and map item_id -> example_id
        example_ids: dict[str, int] = {}
        for item in dataset:
            db_example = Example(
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
            example_ids[item.id] = self._db.insert_example(db_example)

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

            initial_raw = self.model.generate_response(
                item.question, use_history=False, max_tokens=config.max_tokens
            )
            initial_clean = strip_think_tags(initial_raw)
            initial_has_terms = contains_key_terms(initial_clean, item.key_terms)

            # Estimate token count and truncation
            initial_token_count = len(self.model._tokenizer.encode(initial_raw or ""))
            initial_truncated = initial_token_count >= effective_max_tokens - 1

            item_result = ItemResult(
                item_id=item.id,
                question=item.question,
                correct_answer=item.correct_answer,
                key_terms=item.key_terms,
                initial_response=initial_clean,
                initial_response_raw=initial_raw or "",
                initial_has_key_terms=initial_has_terms,
                was_trained=(idx in train_indices),
            )
            item_results[item.id] = item_result

            # Record baseline response in database
            baseline_response = Response(
                id=None,
                example_id=example_ids[item.id],
                experiment_id=experiment_id,
                response_text=initial_clean,
                response_raw=initial_raw or "",
                confidence=None,
                phase=Phase.BASELINE,
                created_at=None,
                token_count=initial_token_count,
                max_tokens=effective_max_tokens,
                truncated=initial_truncated,
            )
            self._db.insert_response(baseline_response)

            if verbose:
                status = "✓" if initial_has_terms else "✗"
                print(f"       {status} Key terms: {initial_has_terms}")

        # Phase 2: Training on train set
        if verbose:
            print()
            print("=" * 60)
            print("PHASE 2: Training")
            print("=" * 60)

        train_items = [(dataset[idx], idx) for idx in indices if idx in train_indices]
        for i, (item, idx) in enumerate(train_items):
            item_result = item_results[item.id]
            if verbose:
                print(f"  [{i+1}/{len(train_items)}] Training on {item.id}...")

            # Create training example
            interactions = [
                InteractionHistory(
                    idx=0,
                    user_input=item.question,
                    llm_response=item_result.initial_response_raw,
                    reviewed=False,
                    timestamp=0.0,
                ),
            ]
            valid_revision = f"[[0]] {item.correct_answer} [[/0]]"
            example = make_collated_training_example(
                valid_revision, interactions, self.model._tokenizer
            )

            # Train using the shared method
            train_start = time.time()
            self.model.train_on_example(example, iterations=config.training_iterations)
            item_result.training_time_seconds = time.time() - train_start

            # Record training event in database
            training_event = TrainingEvent(
                id=None,
                example_id=example_ids[item.id],
                experiment_id=experiment_id,
                training_iterations=config.training_iterations,
                training_time_seconds=item_result.training_time_seconds,
                created_at=None,
            )
            self._db.insert_training_event(training_event)

            if verbose:
                print(f"       Trained ({item_result.training_time_seconds:.1f}s)")

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

            post_raw = self.model.generate_response(
                item.question, use_history=False, max_tokens=config.max_tokens
            )
            post_clean = strip_think_tags(post_raw)
            post_has_terms = contains_key_terms(post_clean, item.key_terms)

            # Estimate token count and truncation
            post_token_count = len(self.model._tokenizer.encode(post_raw or ""))
            post_truncated = post_token_count >= effective_max_tokens - 1

            item_result.post_response = post_clean
            item_result.post_response_raw = post_raw or ""
            item_result.post_has_key_terms = post_has_terms

            # Record post-training response in database
            post_response = Response(
                id=None,
                example_id=example_ids[item.id],
                experiment_id=experiment_id,
                response_text=post_clean,
                response_raw=post_raw or "",
                confidence=None,
                phase=Phase.POST_TRAINING,
                created_at=None,
                token_count=post_token_count,
                max_tokens=effective_max_tokens,
                truncated=post_truncated,
            )
            self._db.insert_response(post_response)

            if verbose:
                was = "✓" if item_result.initial_has_key_terms else "✗"
                now = "✓" if post_has_terms else "✗"
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
            print(f"Total time: {result.total_time_seconds:.1f}s")
            print(f"Baseline accuracy: {result.baseline_accuracy:.1%}")
            print(f"Train post-accuracy: {result.train_post_accuracy:.1%}")
            print(f"Train improvement rate: {result.train_improvement_rate:.1%}")
            print(f"Train retention rate: {result.train_retention_rate:.1%}")
            print(f"Holdout accuracy: {result.holdout_accuracy:.1%}")
            print()
            print(f"Results saved to database (experiment_id={experiment_id})")

        return result

    @property
    def db(self) -> Database:
        """Access the database for queries."""
        return self._db
