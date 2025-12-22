"""Meta-learning experiment infrastructure.

This module provides tools for measuring and comparing meta-learning ability
across multiple model instances trained with different random seeds.

The core hypothesis: self-improvement ability varies across instances, and
some instances become stronger self-learners than others. This module
provides the infrastructure to test this hypothesis.
"""

import dataclasses
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
from .harness import contains_key_terms


@dataclasses.dataclass
class Checkpoint:
    """Metrics at a specific point during training."""

    step: int  # Number of training events completed
    timestamp: str

    # Accuracy metrics
    baseline_correct: int  # Items correct before any training
    baseline_total: int
    post_correct: int  # Items correct after training
    post_total: int

    # Transition counts (for trained items only)
    improved: int  # wrong -> right
    retained: int  # right -> right
    regressed: int  # right -> wrong
    stuck: int  # wrong -> wrong

    # Per-item tracking (item IDs in each category)
    improved_ids: list[str] = dataclasses.field(default_factory=list)
    retained_ids: list[str] = dataclasses.field(default_factory=list)
    regressed_ids: list[str] = dataclasses.field(default_factory=list)
    stuck_ids: list[str] = dataclasses.field(default_factory=list)

    @property
    def baseline_accuracy(self) -> float:
        return (
            self.baseline_correct / self.baseline_total
            if self.baseline_total > 0
            else 0.0
        )

    @property
    def post_accuracy(self) -> float:
        return self.post_correct / self.post_total if self.post_total > 0 else 0.0

    @property
    def improvement_rate(self) -> float:
        """Fraction of improvable items that improved."""
        improvable = self.improved + self.stuck
        return self.improved / improvable if improvable > 0 else 0.0

    @property
    def forgetting_rate(self) -> float:
        """Fraction of forgettable items that regressed."""
        forgettable = self.retained + self.regressed
        return self.regressed / forgettable if forgettable > 0 else 0.0

    @property
    def net_learning(self) -> int:
        """Net items learned (improved - regressed)."""
        return self.improved - self.regressed

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "baseline_correct": self.baseline_correct,
            "baseline_total": self.baseline_total,
            "post_correct": self.post_correct,
            "post_total": self.post_total,
            "improved": self.improved,
            "retained": self.retained,
            "regressed": self.regressed,
            "stuck": self.stuck,
            "improved_ids": self.improved_ids,
            "retained_ids": self.retained_ids,
            "regressed_ids": self.regressed_ids,
            "stuck_ids": self.stuck_ids,
            "baseline_accuracy": self.baseline_accuracy,
            "post_accuracy": self.post_accuracy,
            "improvement_rate": self.improvement_rate,
            "forgetting_rate": self.forgetting_rate,
            "net_learning": self.net_learning,
        }


@dataclasses.dataclass
class SeedTrajectory:
    """Complete training trajectory for one seed."""

    seed: int
    checkpoints: list[Checkpoint] = dataclasses.field(default_factory=list)
    total_time_seconds: float = 0.0
    experiment_id: int | None = None

    @property
    def meta_learning_score(self) -> float | None:
        """Measure how learning efficiency changes over time.

        A model that "learns to learn" should show:
        1. Increasing improvement_rate over time
        2. Decreasing forgetting_rate over time

        Returns:
            Score > 0 indicates meta-learning, < 0 indicates degradation.
            None if insufficient checkpoints to compute (need at least 3).
        """
        if len(self.checkpoints) < 3:
            return None

        # Split into early and late thirds
        third = len(self.checkpoints) // 3
        early = self.checkpoints[:third]
        late = self.checkpoints[-third:]

        early_improvement = sum(c.improvement_rate for c in early) / len(early)
        late_improvement = sum(c.improvement_rate for c in late) / len(late)

        early_forgetting = sum(c.forgetting_rate for c in early) / len(early)
        late_forgetting = sum(c.forgetting_rate for c in late) / len(late)

        # Meta-learning = improvement accelerates, forgetting decelerates
        improvement_delta = late_improvement - early_improvement
        forgetting_delta = (
            early_forgetting - late_forgetting
        )  # Reversed: lower is better

        return improvement_delta + forgetting_delta

    @property
    def has_meta_learning_score(self) -> bool:
        """Whether we have enough checkpoints to compute meta-learning score."""
        return len(self.checkpoints) >= 3

    @property
    def final_accuracy(self) -> float:
        """Accuracy at the final checkpoint."""
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1].post_accuracy

    @property
    def total_net_learning(self) -> int:
        """Total items learned across all checkpoints."""
        if not self.checkpoints:
            return 0
        return self.checkpoints[-1].net_learning

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "experiment_id": self.experiment_id,
            "total_time_seconds": self.total_time_seconds,
            "meta_learning_score": self.meta_learning_score,
            "has_meta_learning_score": self.has_meta_learning_score,
            "final_accuracy": self.final_accuracy,
            "total_net_learning": self.total_net_learning,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }


@dataclasses.dataclass
class MetaLearningConfig:
    """Configuration for a meta-learning experiment."""

    name: str = "meta_experiment"
    seeds: list[int] = dataclasses.field(
        default_factory=lambda: [42, 123, 456, 789, 1011]
    )
    checkpoint_interval: int = 10  # Checkpoint every N training events
    training_iterations: int = 25  # Iterations per training event
    train_ratio: float = 0.8  # Fraction used for training
    max_tokens: int | None = None  # Use model default if None


@dataclasses.dataclass
class MetaLearningResult:
    """Complete results from a meta-learning experiment."""

    config: MetaLearningConfig
    dataset_name: str
    timestamp: str
    trajectories: dict[int, SeedTrajectory] = dataclasses.field(default_factory=dict)
    holdout_results: dict[int, float] = dataclasses.field(default_factory=dict)

    def _ranking_key(self, seed: int) -> tuple[float, float, int]:
        """Return a tuple for ranking seeds.

        Priority: meta_learning_score (if available), final_accuracy, net_learning.
        """
        traj = self.trajectories[seed]
        meta_score = traj.meta_learning_score if traj.meta_learning_score is not None else 0.0
        return (meta_score, traj.final_accuracy, traj.total_net_learning)

    @property
    def best_seed(self) -> int | None:
        """Seed with best performance (meta-learning score, then accuracy, then net learning)."""
        if not self.trajectories:
            return None
        return max(self.trajectories.keys(), key=self._ranking_key)

    @property
    def worst_seed(self) -> int | None:
        """Seed with worst performance (meta-learning score, then accuracy, then net learning)."""
        if not self.trajectories:
            return None
        return min(self.trajectories.keys(), key=self._ranking_key)

    @property
    def has_meta_learning_scores(self) -> bool:
        """Whether any trajectory has enough checkpoints for meta-learning score."""
        return any(t.has_meta_learning_score for t in self.trajectories.values())

    @property
    def score_variance(self) -> float | None:
        """Variance in meta-learning scores across seeds.

        Returns None if insufficient checkpoints to compute scores.
        """
        if len(self.trajectories) < 2:
            return None
        scores = [
            t.meta_learning_score
            for t in self.trajectories.values()
            if t.meta_learning_score is not None
        ]
        if len(scores) < 2:
            return None
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "seeds": self.config.seeds,
                "checkpoint_interval": self.config.checkpoint_interval,
                "training_iterations": self.config.training_iterations,
                "train_ratio": self.config.train_ratio,
            },
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "best_seed": self.best_seed,
            "worst_seed": self.worst_seed,
            "has_meta_learning_scores": self.has_meta_learning_scores,
            "score_variance": self.score_variance,
            "trajectories": {
                seed: traj.to_dict() for seed, traj in self.trajectories.items()
            },
            "holdout_results": self.holdout_results,
        }

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "MetaLearningResult":
        """Load results from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        config = MetaLearningConfig(
            name=data["config"]["name"],
            seeds=data["config"]["seeds"],
            checkpoint_interval=data["config"]["checkpoint_interval"],
            training_iterations=data["config"]["training_iterations"],
            train_ratio=data["config"]["train_ratio"],
        )

        result = cls(
            config=config,
            dataset_name=data["dataset_name"],
            timestamp=data["timestamp"],
            holdout_results=data.get("holdout_results", {}),
        )

        for seed_str, traj_data in data["trajectories"].items():
            seed = int(seed_str)
            trajectory = SeedTrajectory(
                seed=seed,
                total_time_seconds=traj_data["total_time_seconds"],
                experiment_id=traj_data.get("experiment_id"),
            )
            for cp_data in traj_data["checkpoints"]:
                checkpoint = Checkpoint(
                    step=cp_data["step"],
                    timestamp=cp_data["timestamp"],
                    baseline_correct=cp_data["baseline_correct"],
                    baseline_total=cp_data["baseline_total"],
                    post_correct=cp_data["post_correct"],
                    post_total=cp_data["post_total"],
                    improved=cp_data["improved"],
                    retained=cp_data["retained"],
                    regressed=cp_data["regressed"],
                    stuck=cp_data["stuck"],
                    # Per-item IDs (may not exist in older files)
                    improved_ids=cp_data.get("improved_ids", []),
                    retained_ids=cp_data.get("retained_ids", []),
                    regressed_ids=cp_data.get("regressed_ids", []),
                    stuck_ids=cp_data.get("stuck_ids", []),
                )
                trajectory.checkpoints.append(checkpoint)
            result.trajectories[seed] = trajectory

        return result


class MetaLearningExperiment:
    """Runs meta-learning experiments across multiple seeds.

    This class orchestrates training multiple model instances with different
    random seeds on the same dataset, tracking their learning trajectories
    to measure meta-learning ability.
    """

    def __init__(
        self,
        model_factory: Callable[[], StatefulLLM] | None = None,
        db: Database | None = None,
        db_path: Path | str | None = None,
    ):
        """Initialize the experiment.

        Args:
            model_factory: Factory function to create fresh model instances.
                If None, uses default StatefulLLM constructor.
            db: Pre-initialized Database. If None, will be created using db_path.
            db_path: Path to SQLite database. If None, uses default location.
        """
        self._model_factory = model_factory or (lambda: StatefulLLM(model_path=None))
        if db is not None:
            self._db = db
        elif db_path is not None:
            self._db = Database(db_path)
        else:
            self._db = Database()

    def run(
        self,
        dataset: TriviaDataset,
        config: MetaLearningConfig | None = None,
        verbose: bool = True,
    ) -> MetaLearningResult:
        """Run the meta-learning experiment.

        Args:
            dataset: Dataset to train on.
            config: Experiment configuration.
            verbose: Enable progress output.

        Returns:
            MetaLearningResult with trajectories for each seed.
        """
        if config is None:
            config = MetaLearningConfig()

        result = MetaLearningResult(
            config=config,
            dataset_name=dataset.name,
            timestamp=datetime.now().isoformat(),
        )

        for seed_idx, seed in enumerate(config.seeds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Seed {seed_idx + 1}/{len(config.seeds)}: {seed}")
                print("=" * 60)

            trajectory = self._run_single_seed(dataset, config, seed, verbose)
            result.trajectories[seed] = trajectory

        # Summary
        if verbose:
            print("\n" + "=" * 60)
            print("META-LEARNING SUMMARY")
            print("=" * 60)
            for seed, traj in result.trajectories.items():
                print(f"  Seed {seed}:")
                score_str = (
                    f"{traj.meta_learning_score:.4f}"
                    if traj.meta_learning_score is not None
                    else "N/A (need 3+ checkpoints)"
                )
                print(f"    Meta-learning score: {score_str}")
                print(f"    Final accuracy: {traj.final_accuracy:.1%}")
                print(f"    Net learning: {traj.total_net_learning}")
            print()
            if result.best_seed is not None:
                best = result.trajectories[result.best_seed]
                best_score = (
                    f"{best.meta_learning_score:.4f}"
                    if best.meta_learning_score is not None
                    else "N/A"
                )
                print(
                    f"  Best seed: {result.best_seed} "
                    f"(score={best_score}, acc={best.final_accuracy:.1%})"
                )
            if result.worst_seed is not None:
                worst = result.trajectories[result.worst_seed]
                worst_score = (
                    f"{worst.meta_learning_score:.4f}"
                    if worst.meta_learning_score is not None
                    else "N/A"
                )
                print(
                    f"  Worst seed: {result.worst_seed} "
                    f"(score={worst_score}, acc={worst.final_accuracy:.1%})"
                )
            if result.score_variance is not None:
                print(f"  Score variance: {result.score_variance:.6f}")
            else:
                print("  Score variance: N/A (insufficient checkpoints)")

        return result

    def _run_single_seed(
        self,
        dataset: TriviaDataset,
        config: MetaLearningConfig,
        seed: int,
        verbose: bool,
    ) -> SeedTrajectory:
        """Run training for a single seed."""
        start_time = time.time()
        trajectory = SeedTrajectory(seed=seed)

        # Create fresh model
        if verbose:
            print("  Loading fresh model...")
        model = self._model_factory()
        model._model_is_stable = True

        # Get effective max tokens
        effective_max_tokens = config.max_tokens or model._max_tokens

        # Create experiment record
        experiment = Experiment(
            id=None,
            name=f"{config.name}_seed{seed}",
            experiment_type=ExperimentType.EVAL,
            config_json=json.dumps(
                {
                    "seed": seed,
                    "training_iterations": config.training_iterations,
                    "checkpoint_interval": config.checkpoint_interval,
                    "train_ratio": config.train_ratio,
                }
            ),
            model_checkpoint=None,
            started_at=datetime.now(),
            completed_at=None,
        )
        experiment_id = self._db.insert_experiment(experiment)
        trajectory.experiment_id = experiment_id

        # Shuffle and split dataset
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_count = int(len(indices) * config.train_ratio)
        train_indices = indices[:train_count]
        holdout_indices = indices[train_count:]

        if verbose:
            print(
                f"  Dataset: {len(dataset)} items ({train_count} train, {len(holdout_indices)} holdout)"
            )

        # Phase 1: Get baseline responses for all items
        if verbose:
            print("  Phase 1: Baseline inference...")

        baseline_responses: dict[str, tuple[str, bool]] = (
            {}
        )  # item_id -> (response, correct)
        example_ids: dict[str, int] = {}

        for i, idx in enumerate(indices):
            item = dataset[idx]

            # Insert example
            db_example = Example(
                id=None,
                canonical_id=item.id,
                question=item.question,
                ground_truth_answer=item.correct_answer,
                key_terms=item.key_terms if item.key_terms else None,
                category=item.category,
                difficulty=item.difficulty,
                source_type=SourceType.STATIC_TRIVIA,
                source_url=None,
                source_title=None,
                valid_at=None,
                created_at=None,
            )
            example_ids[item.id] = self._db.insert_example(db_example)

            # Get baseline response
            raw = model.generate_response(
                item.question, use_history=False, max_tokens=config.max_tokens
            )
            clean = strip_think_tags(raw)
            correct = contains_key_terms(clean, item.key_terms)
            baseline_responses[item.id] = (clean, correct)

            # Record in database
            token_count = len(model._tokenizer.encode(raw or ""))
            response = Response(
                id=None,
                example_id=example_ids[item.id],
                experiment_id=experiment_id,
                response_text=clean,
                response_raw=raw or "",
                confidence=None,
                phase=Phase.BASELINE,
                created_at=None,
                token_count=token_count,
                max_tokens=effective_max_tokens,
                truncated=token_count >= effective_max_tokens - 1,
            )
            self._db.insert_response(response)

        # Phase 2: Train with checkpoints
        if verbose:
            print("  Phase 2: Training with checkpoints...")

        post_responses: dict[str, tuple[str, bool]] = {}  # Updated as we train
        trained_items: set[str] = set()

        for batch_idx in range(0, len(train_indices), config.checkpoint_interval):
            batch = train_indices[batch_idx : batch_idx + config.checkpoint_interval]

            # Train on this batch
            for idx in batch:
                item = dataset[idx]
                baseline_clean, baseline_correct = baseline_responses[item.id]

                # Create training example
                interactions = [
                    InteractionHistory(
                        idx=0,
                        user_input=item.question,
                        llm_response=baseline_clean,
                        reviewed=False,
                        timestamp=0.0,
                    ),
                ]
                valid_revision = f"[[0]] {item.correct_answer} [[/0]]"
                example = make_collated_training_example(
                    valid_revision, interactions, model._tokenizer
                )

                # Train
                train_start = time.time()
                model.train_on_example(example, iterations=config.training_iterations)
                training_time = time.time() - train_start

                # Record training event
                training_event = TrainingEvent(
                    id=None,
                    example_id=example_ids[item.id],
                    experiment_id=experiment_id,
                    training_iterations=config.training_iterations,
                    training_time_seconds=training_time,
                    created_at=None,
                )
                self._db.insert_training_event(training_event)

                trained_items.add(item.id)

            # Checkpoint: Evaluate all trained items
            for item_id in trained_items:
                item = next(
                    dataset[i] for i in range(len(dataset)) if dataset[i].id == item_id
                )
                raw = model.generate_response(
                    item.question, use_history=False, max_tokens=config.max_tokens
                )
                clean = strip_think_tags(raw)
                correct = contains_key_terms(clean, item.key_terms)
                post_responses[item_id] = (clean, correct)

            # Compute checkpoint metrics
            checkpoint = self._compute_checkpoint(
                step=len(trained_items),
                baseline_responses=baseline_responses,
                post_responses=post_responses,
                trained_items=trained_items,
            )
            trajectory.checkpoints.append(checkpoint)

            if verbose:
                print(
                    f"    Checkpoint {len(trajectory.checkpoints)}: "
                    f"step={checkpoint.step}, "
                    f"acc={checkpoint.post_accuracy:.1%}, "
                    f"net={checkpoint.net_learning}"
                )

        # Phase 3: Final evaluation on holdout
        if verbose:
            print("  Phase 3: Holdout evaluation...")

        holdout_correct = 0
        for idx in holdout_indices:
            item = dataset[idx]
            raw = model.generate_response(
                item.question, use_history=False, max_tokens=config.max_tokens
            )
            clean = strip_think_tags(raw)
            if contains_key_terms(clean, item.key_terms):
                holdout_correct += 1

            # Record post-training response
            token_count = len(model._tokenizer.encode(raw or ""))
            response = Response(
                id=None,
                example_id=example_ids[item.id],
                experiment_id=experiment_id,
                response_text=clean,
                response_raw=raw or "",
                confidence=None,
                phase=Phase.POST_TRAINING,
                created_at=None,
                token_count=token_count,
                max_tokens=effective_max_tokens,
                truncated=token_count >= effective_max_tokens - 1,
            )
            self._db.insert_response(response)

        holdout_accuracy = (
            holdout_correct / len(holdout_indices) if holdout_indices else 0.0
        )
        if verbose:
            print(f"    Holdout accuracy: {holdout_accuracy:.1%}")

        # Mark experiment complete
        self._db.complete_experiment(experiment_id)
        model._model_is_stable = True

        trajectory.total_time_seconds = time.time() - start_time
        return trajectory

    def _compute_checkpoint(
        self,
        step: int,
        baseline_responses: dict[str, tuple[str, bool]],
        post_responses: dict[str, tuple[str, bool]],
        trained_items: set[str],
    ) -> Checkpoint:
        """Compute metrics for a checkpoint."""
        improved = 0
        retained = 0
        regressed = 0
        stuck = 0

        # Track item IDs in each category
        improved_ids: list[str] = []
        retained_ids: list[str] = []
        regressed_ids: list[str] = []
        stuck_ids: list[str] = []

        for item_id in trained_items:
            if item_id not in post_responses:
                continue
            _, baseline_correct = baseline_responses[item_id]
            _, post_correct = post_responses[item_id]

            if not baseline_correct and post_correct:
                improved += 1
                improved_ids.append(item_id)
            elif baseline_correct and post_correct:
                retained += 1
                retained_ids.append(item_id)
            elif baseline_correct and not post_correct:
                regressed += 1
                regressed_ids.append(item_id)
            else:
                stuck += 1
                stuck_ids.append(item_id)

        baseline_correct_count = sum(1 for _, c in baseline_responses.values() if c)
        post_correct_count = sum(1 for _, c in post_responses.values() if c)

        return Checkpoint(
            step=step,
            timestamp=datetime.now().isoformat(),
            baseline_correct=baseline_correct_count,
            baseline_total=len(baseline_responses),
            post_correct=post_correct_count,
            post_total=len(post_responses),
            improved=improved,
            retained=retained,
            regressed=regressed,
            stuck=stuck,
            improved_ids=improved_ids,
            retained_ids=retained_ids,
            regressed_ids=regressed_ids,
            stuck_ids=stuck_ids,
        )

    @property
    def db(self) -> Database:
        """Access the database for queries."""
        return self._db
