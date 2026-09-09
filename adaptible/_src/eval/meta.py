"""Meta-learning experiment infrastructure.

This module provides tools for measuring and comparing meta-learning ability
across multiple model instances trained with different random seeds.

The core hypothesis: self-improvement ability varies across instances, and
some instances become stronger self-learners than others. This module
provides the infrastructure to test this hypothesis.

Two things every reader of a result must know:

* ``MetaLearningConfig.training_source`` says what the model was trained on.
  ``"ground_truth"`` is supervised fine-tuning on the dataset label;
  ``"self_generated"`` is the model's own revision (real self-correction).
* Checkpoint transition counts come in two flavours. The cumulative fields
  (``improved`` etc.) cover every item trained so far; the ``window_*`` fields
  cover only the items trained since the previous checkpoint. The
  meta-learning score is built from the window fields.
"""

import dataclasses
import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .._llm import StatefulLLM
from ..db import Database, Experiment, ExperimentType, Phase
from .dataset import TriviaDataset, TriviaItem
from .harness import (
    _infer_and_record,
    _insert_dataset_examples,
    _judge_only,
    _train_one_item,
    validate_training_source,
)
from ..revise import revision_prompt_preset

# Minimum improvable+forgettable items a window needs before its rates feed the
# meta-learning score. Below this a single item swings a rate by 20+ points.
MIN_WINDOW_ITEMS = 5


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _population_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


@dataclasses.dataclass
class Checkpoint:
    """Metrics at a specific point during training.

    The ``improved``/``retained``/``regressed``/``stuck`` counts are *cumulative*:
    they classify every item trained so far by baseline -> current correctness.
    The ``window_*`` counts classify only the items trained since the previous
    checkpoint, so consecutive checkpoints describe disjoint sets of items.

    ``holdout_correct``/``holdout_total`` are populated only when the experiment
    ran with ``holdout_every_checkpoint``; otherwise they stay None.
    """

    step: int  # Number of training events completed
    timestamp: str

    # Accuracy metrics
    baseline_correct: int  # Items correct before any training
    baseline_total: int
    post_correct: int  # Items correct after training
    post_total: int

    # Cumulative transition counts (all trained items so far)
    improved: int  # wrong -> right
    retained: int  # right -> right
    regressed: int  # right -> wrong
    stuck: int  # wrong -> wrong

    # Per-item tracking (item IDs in each category, cumulative)
    improved_ids: list[str] = dataclasses.field(default_factory=list)
    retained_ids: list[str] = dataclasses.field(default_factory=list)
    regressed_ids: list[str] = dataclasses.field(default_factory=list)
    stuck_ids: list[str] = dataclasses.field(default_factory=list)

    # Marginal transition counts (items trained since the previous checkpoint)
    window_improved: int = 0
    window_retained: int = 0
    window_regressed: int = 0
    window_stuck: int = 0
    window_ids: list[str] = dataclasses.field(default_factory=list)

    # Items in this window whose self-generated revision was rejected (untrained)
    revision_invalid_ids: list[str] = dataclasses.field(default_factory=list)

    # Optional per-checkpoint holdout probe
    holdout_correct: int | None = None
    holdout_total: int | None = None

    @property
    def baseline_accuracy(self) -> float:
        return _rate(self.baseline_correct, self.baseline_total)

    @property
    def post_accuracy(self) -> float:
        return _rate(self.post_correct, self.post_total)

    @property
    def improvement_rate(self) -> float:
        """Cumulative: fraction of improvable trained items that improved."""
        return _rate(self.improved, self.improved + self.stuck)

    @property
    def forgetting_rate(self) -> float:
        """Cumulative: fraction of forgettable trained items that regressed."""
        return _rate(self.regressed, self.retained + self.regressed)

    @property
    def net_learning(self) -> int:
        """Net items learned (improved - regressed), cumulative."""
        return self.improved - self.regressed

    @property
    def window_size(self) -> int:
        """Number of trained items in this checkpoint's window."""
        return (
            self.window_improved
            + self.window_retained
            + self.window_regressed
            + self.window_stuck
        )

    @property
    def window_improvement_rate(self) -> float:
        """Marginal: fraction of this window's improvable items that improved."""
        return _rate(self.window_improved, self.window_improved + self.window_stuck)

    @property
    def window_forgetting_rate(self) -> float:
        """Marginal: fraction of this window's forgettable items that regressed."""
        return _rate(
            self.window_regressed, self.window_retained + self.window_regressed
        )

    @property
    def window_net_learning(self) -> int:
        return self.window_improved - self.window_regressed

    @property
    def holdout_accuracy(self) -> float | None:
        if self.holdout_total is None or self.holdout_total == 0:
            return None
        return (self.holdout_correct or 0) / self.holdout_total

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
            "window_improved": self.window_improved,
            "window_retained": self.window_retained,
            "window_regressed": self.window_regressed,
            "window_stuck": self.window_stuck,
            "window_ids": self.window_ids,
            "window_size": self.window_size,
            "revision_invalid_ids": self.revision_invalid_ids,
            "holdout_correct": self.holdout_correct,
            "holdout_total": self.holdout_total,
            "holdout_accuracy": self.holdout_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "post_accuracy": self.post_accuracy,
            "improvement_rate": self.improvement_rate,
            "forgetting_rate": self.forgetting_rate,
            "net_learning": self.net_learning,
            "window_improvement_rate": self.window_improvement_rate,
            "window_forgetting_rate": self.window_forgetting_rate,
            "window_net_learning": self.window_net_learning,
        }

    @classmethod
    def from_dict(cls, cp_data: dict[str, Any]) -> "Checkpoint":
        """Build from a dict; fields added after the first release default."""
        return cls(
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
            window_improved=cp_data.get("window_improved", 0),
            window_retained=cp_data.get("window_retained", 0),
            window_regressed=cp_data.get("window_regressed", 0),
            window_stuck=cp_data.get("window_stuck", 0),
            window_ids=cp_data.get("window_ids", []),
            revision_invalid_ids=cp_data.get("revision_invalid_ids", []),
            holdout_correct=cp_data.get("holdout_correct"),
            holdout_total=cp_data.get("holdout_total"),
        )


@dataclasses.dataclass
class SeedTrajectory:
    """Complete training trajectory for one (seed, repeat).

    Attributes:
        seed: Shuffle seed.
        repeat: Which repeat of this seed (0-based). Repeats share the seed's
            shuffle, so differences between them are generation/training noise.
        holdout_correct/holdout_total: Final holdout evaluation after training.
        revision_invalid_count: Items skipped because the self-generated
            revision failed validation (always 0 for ``ground_truth``).
    """

    seed: int
    checkpoints: list[Checkpoint] = dataclasses.field(default_factory=list)
    total_time_seconds: float = 0.0
    experiment_id: int | None = None
    repeat: int = 0
    holdout_correct: int | None = None
    holdout_total: int | None = None
    revision_invalid_count: int = 0

    @property
    def holdout_accuracy(self) -> float | None:
        """Final holdout accuracy, or None if no holdout was evaluated."""
        if self.holdout_total is None or self.holdout_total == 0:
            return None
        return (self.holdout_correct or 0) / self.holdout_total

    @property
    def window_sizes(self) -> list[int]:
        """Per-checkpoint count of trained items in that checkpoint's window."""
        return [c.window_size for c in self.checkpoints]

    def _score_windows(self) -> tuple[list[Checkpoint], list[Checkpoint]]:
        third = len(self.checkpoints) // 3
        return self.checkpoints[:third], self.checkpoints[-third:]

    @property
    def meta_learning_score_reason(self) -> str | None:
        """Why ``meta_learning_score`` is None, or None if it is computable."""
        if len(self.checkpoints) < 3:
            return f"need 3+ checkpoints, have {len(self.checkpoints)}"
        early, late = self._score_windows()
        small = [c for c in early + late if c.window_size < MIN_WINDOW_ITEMS]
        if small:
            sizes = ", ".join(f"step {c.step}: n={c.window_size}" for c in small)
            return f"window(s) below {MIN_WINDOW_ITEMS} trained items ({sizes})"
        return None

    @property
    def meta_learning_score(self) -> float | None:
        """Measure how learning efficiency changes over time.

        A model that "learns to learn" should show:
        1. Increasing improvement rate over time
        2. Decreasing forgetting rate over time

        The score is ``(late_improvement - early_improvement) +
        (early_forgetting - late_forgetting)`` where the rates are the
        *window* rates (``Checkpoint.window_improvement_rate`` etc.), averaged
        over the first and last third of checkpoints.

        Window rates are used rather than the cumulative ``improvement_rate``
        because cumulative rates are re-evaluated over every item trained so
        far: by the last checkpoint they are dominated by items that were
        trained early, and the "late" set is a superset of the "early" set.
        That nesting shrinks any real early-vs-late difference toward zero
        and makes the score mostly a function of the first few items. Window
        rates cover disjoint sets of items, so early and late are independent
        samples of the model's learning behaviour at that point in training.

        Returns:
            Score > 0 indicates meta-learning, < 0 indicates degradation.
            None if there are fewer than 3 checkpoints or any window used has
            fewer than ``MIN_WINDOW_ITEMS`` trained items; see
            ``meta_learning_score_reason``.
        """
        if self.meta_learning_score_reason is not None:
            return None
        early, late = self._score_windows()

        early_improvement = sum(c.window_improvement_rate for c in early) / len(early)
        late_improvement = sum(c.window_improvement_rate for c in late) / len(late)

        early_forgetting = sum(c.window_forgetting_rate for c in early) / len(early)
        late_forgetting = sum(c.window_forgetting_rate for c in late) / len(late)

        # Meta-learning = improvement accelerates, forgetting decelerates
        improvement_delta = late_improvement - early_improvement
        forgetting_delta = early_forgetting - late_forgetting  # lower is better
        return improvement_delta + forgetting_delta

    @property
    def has_meta_learning_score(self) -> bool:
        """Whether the meta-learning score is computable for this trajectory."""
        return self.meta_learning_score_reason is None

    @property
    def final_trained_accuracy(self) -> float:
        """Accuracy on *trained items only* at the final checkpoint.

        This is not a generalization number; see ``holdout_accuracy`` for that.
        """
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1].post_accuracy

    @property
    def final_accuracy(self) -> float:
        """Deprecated alias for ``final_trained_accuracy``."""
        warnings.warn(
            "SeedTrajectory.final_accuracy is trained-items-only accuracy; "
            "use final_trained_accuracy (or holdout_accuracy).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.final_trained_accuracy

    @property
    def total_net_learning(self) -> int:
        """Net learning (improved - regressed) at the final checkpoint.

        Because checkpoint counts are cumulative this is the net over all
        trained items, not a sum over checkpoints.
        """
        if not self.checkpoints:
            return 0
        return self.checkpoints[-1].net_learning

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "repeat": self.repeat,
            "experiment_id": self.experiment_id,
            "total_time_seconds": self.total_time_seconds,
            "meta_learning_score": self.meta_learning_score,
            "has_meta_learning_score": self.has_meta_learning_score,
            "meta_learning_score_reason": self.meta_learning_score_reason,
            "window_sizes": self.window_sizes,
            "final_trained_accuracy": self.final_trained_accuracy,
            "holdout_correct": self.holdout_correct,
            "holdout_total": self.holdout_total,
            "holdout_accuracy": self.holdout_accuracy,
            "revision_invalid_count": self.revision_invalid_count,
            "total_net_learning": self.total_net_learning,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, traj_data: dict[str, Any], seed: int) -> "SeedTrajectory":
        trajectory = cls(
            seed=seed,
            repeat=traj_data.get("repeat", 0),
            total_time_seconds=traj_data["total_time_seconds"],
            experiment_id=traj_data.get("experiment_id"),
            holdout_correct=traj_data.get("holdout_correct"),
            holdout_total=traj_data.get("holdout_total"),
            revision_invalid_count=traj_data.get("revision_invalid_count", 0),
        )
        for cp_data in traj_data["checkpoints"]:
            trajectory.checkpoints.append(Checkpoint.from_dict(cp_data))
        return trajectory


@dataclasses.dataclass
class MetaLearningConfig:
    """Configuration for a meta-learning experiment.

    Attributes:
        training_source: "ground_truth" (fine-tune on the label) or
            "self_generated" (train on the model's own revision).
        revision_prompt: Revision prompt preset for "self_generated"; see
            ``revise.revision_prompt_preset``.
        close_think: Close the chat template's open ``<think>`` block before
            the training target; see ``EvaluationConfig.close_think``.
        holdout_every_checkpoint: Also probe the holdout set at every
            checkpoint (costs a holdout-sized inference pass per checkpoint).
        repeats: Run each seed this many times with an identical shuffle. Any
            divergence between repeats is generation/training noise, which is
            the control the across-seed comparison needs.
    """

    name: str = "meta_experiment"
    seeds: list[int] = dataclasses.field(
        default_factory=lambda: [42, 123, 456, 789, 1011]
    )
    checkpoint_interval: int = 10  # Checkpoint every N training events
    training_iterations: int = 25  # Iterations per training event
    train_ratio: float = 0.8  # Fraction used for training
    max_tokens: int | None = None  # Use model default if None
    training_source: str = "ground_truth"
    revision_prompt: str = "default"
    close_think: bool = True
    holdout_every_checkpoint: bool = False
    repeats: int = 1

    def __post_init__(self) -> None:
        validate_training_source(self.training_source)
        revision_prompt_preset(self.revision_prompt)  # raises ValueError if unknown
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seeds": self.seeds,
            "checkpoint_interval": self.checkpoint_interval,
            "training_iterations": self.training_iterations,
            "train_ratio": self.train_ratio,
            "max_tokens": self.max_tokens,
            "training_source": self.training_source,
            "revision_prompt": self.revision_prompt,
            "close_think": self.close_think,
            "holdout_every_checkpoint": self.holdout_every_checkpoint,
            "repeats": self.repeats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaLearningConfig":
        return cls(
            name=data["name"],
            seeds=data["seeds"],
            checkpoint_interval=data["checkpoint_interval"],
            training_iterations=data["training_iterations"],
            train_ratio=data["train_ratio"],
            max_tokens=data.get("max_tokens"),
            training_source=data.get("training_source", "ground_truth"),
            revision_prompt=data.get("revision_prompt", "default"),
            # Files written before close_think existed were trained on the
            # unclosed-think target, so absence means False, not the new default.
            close_think=data.get("close_think", False),
            holdout_every_checkpoint=data.get("holdout_every_checkpoint", False),
            repeats=data.get("repeats", 1),
        )


@dataclasses.dataclass
class MetaLearningResult:
    """Complete results from a meta-learning experiment.

    ``all_trajectories`` holds every run keyed by ``(seed, repeat)``.
    ``trajectories`` is the backward-compatible view keyed by seed that returns
    each seed's first repeat.
    """

    config: MetaLearningConfig
    dataset_name: str
    timestamp: str
    all_trajectories: dict[tuple[int, int], SeedTrajectory] = dataclasses.field(
        default_factory=dict
    )
    holdout_results: dict[int, float] = dataclasses.field(default_factory=dict)

    @property
    def trajectories(self) -> dict[int, SeedTrajectory]:
        """First repeat of each seed, keyed by seed."""
        return {
            seed: traj
            for (seed, repeat), traj in self.all_trajectories.items()
            if repeat == 0
        }

    def trajectories_for_seed(self, seed: int) -> list[SeedTrajectory]:
        """All repeats for one seed, ordered by repeat."""
        return [
            traj for (s, _), traj in sorted(self.all_trajectories.items()) if s == seed
        ]

    def _ranking_key(self, seed: int) -> tuple[float, float, int]:
        """Return a tuple for ranking seeds.

        Priority: meta_learning_score (if available), final trained accuracy,
        net_learning.
        """
        traj = self.trajectories[seed]
        meta_score = (
            traj.meta_learning_score if traj.meta_learning_score is not None else 0.0
        )
        return (meta_score, traj.final_trained_accuracy, traj.total_net_learning)

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
        return any(t.has_meta_learning_score for t in self.all_trajectories.values())

    @property
    def score_variance(self) -> float | None:
        """Variance in meta-learning scores across seeds (first repeats).

        Returns None if fewer than two seeds have a score.
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
        return _population_variance(scores)

    def _per_seed_scores(self) -> dict[int, list[float]]:
        """Scored repeats grouped by seed (seeds with no scored repeat omitted)."""
        by_seed: dict[int, list[float]] = {}
        for (seed, _), traj in sorted(self.all_trajectories.items()):
            score = traj.meta_learning_score
            if score is not None:
                by_seed.setdefault(seed, []).append(score)
        return by_seed

    @property
    def within_seed_variance(self) -> float | None:
        """Mean over seeds of the variance of the score across repeats.

        This is the noise floor: repeats share a shuffle, so any spread is
        generation/training nondeterminism. None unless at least one seed has
        two or more scored repeats.
        """
        variances = [
            _population_variance(scores)
            for scores in self._per_seed_scores().values()
            if len(scores) >= 2
        ]
        if not variances:
            return None
        return sum(variances) / len(variances)

    @property
    def across_seed_variance(self) -> float | None:
        """Variance of the per-seed mean score. None unless 2+ seeds scored."""
        means = [
            sum(scores) / len(scores) for scores in self._per_seed_scores().values()
        ]
        if len(means) < 2:
            return None
        return _population_variance(means)

    @property
    def signal_to_noise(self) -> float | None:
        """``across_seed_variance / within_seed_variance``.

        Values near or below 1 mean seed-to-seed differences are no larger
        than run-to-run noise. None if either variance is undefined or the
        within-seed variance is zero.
        """
        across = self.across_seed_variance
        within = self.within_seed_variance
        if across is None or within is None or within == 0:
            return None
        return across / within

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "best_seed": self.best_seed,
            "worst_seed": self.worst_seed,
            "has_meta_learning_scores": self.has_meta_learning_scores,
            "score_variance": self.score_variance,
            "within_seed_variance": self.within_seed_variance,
            "across_seed_variance": self.across_seed_variance,
            "signal_to_noise": self.signal_to_noise,
            "trajectories": {
                seed: traj.to_dict() for seed, traj in self.trajectories.items()
            },
            "all_trajectories": {
                f"{seed}/{repeat}": traj.to_dict()
                for (seed, repeat), traj in sorted(self.all_trajectories.items())
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
        """Load results from JSON file.

        Files written before repeats/windows/holdout fields existed load with
        those fields defaulted.
        """
        path = Path(path)
        data = json.loads(path.read_text())

        result = cls(
            config=MetaLearningConfig.from_dict(data["config"]),
            dataset_name=data["dataset_name"],
            timestamp=data["timestamp"],
            holdout_results={
                int(k): v for k, v in data.get("holdout_results", {}).items()
            },
        )

        if "all_trajectories" in data:
            for key, traj_data in data["all_trajectories"].items():
                seed_str, repeat_str = key.split("/")
                trajectory = SeedTrajectory.from_dict(traj_data, int(seed_str))
                trajectory.repeat = int(repeat_str)
                result.all_trajectories[(int(seed_str), int(repeat_str))] = trajectory
        else:
            for seed_str, traj_data in data["trajectories"].items():
                seed = int(seed_str)
                result.all_trajectories[(seed, 0)] = SeedTrajectory.from_dict(
                    traj_data, seed
                )

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
            MetaLearningResult with trajectories for each (seed, repeat).
        """
        if config is None:
            config = MetaLearningConfig()

        result = MetaLearningResult(
            config=config,
            dataset_name=dataset.name,
            timestamp=datetime.now().isoformat(),
        )

        for seed_idx, seed in enumerate(config.seeds):
            for repeat in range(config.repeats):
                if verbose:
                    print(f"\n{'='*60}")
                    label = f"Seed {seed_idx + 1}/{len(config.seeds)}: {seed}"
                    if config.repeats > 1:
                        label += f" (repeat {repeat + 1}/{config.repeats})"
                    print(label)
                    print("=" * 60)

                trajectory = self._run_single_seed(
                    dataset, config, seed, repeat, verbose
                )
                result.all_trajectories[(seed, repeat)] = trajectory
                if trajectory.holdout_accuracy is not None and repeat == 0:
                    result.holdout_results[seed] = trajectory.holdout_accuracy

        if verbose:
            self._print_summary(result)

        return result

    @staticmethod
    def _print_summary(result: MetaLearningResult) -> None:
        config = result.config
        print("\n" + "=" * 60)
        print("META-LEARNING SUMMARY")
        print("=" * 60)
        print(f"  Training source: {config.training_source}")
        if config.training_source == "self_generated":
            print(f"  Revision prompt: {config.revision_prompt}")
        print(f"  Close think: {config.close_think}")
        for (seed, repeat), traj in sorted(result.all_trajectories.items()):
            label = f"Seed {seed}" + (f" repeat {repeat}" if config.repeats > 1 else "")
            print(f"  {label}:")
            score_str = (
                f"{traj.meta_learning_score:.4f}"
                if traj.meta_learning_score is not None
                else f"N/A ({traj.meta_learning_score_reason})"
            )
            print(f"    Meta-learning score: {score_str}")
            print(f"    Window sizes: {traj.window_sizes}")
            print(f"    Final trained-item accuracy: {traj.final_trained_accuracy:.1%}")
            holdout = traj.holdout_accuracy
            print(
                "    Holdout accuracy: "
                + (f"{holdout:.1%}" if holdout is not None else "N/A")
            )
            print(f"    Net learning: {traj.total_net_learning}")
            if config.training_source == "self_generated":
                print(f"    Invalid revisions: {traj.revision_invalid_count}")
        print()
        for label, seed in (("Best", result.best_seed), ("Worst", result.worst_seed)):
            if seed is None:
                continue
            traj = result.trajectories[seed]
            score = (
                f"{traj.meta_learning_score:.4f}"
                if traj.meta_learning_score is not None
                else "N/A"
            )
            print(
                f"  {label} seed: {seed} "
                f"(score={score}, trained-acc={traj.final_trained_accuracy:.1%})"
            )
        if result.score_variance is not None:
            print(f"  Score variance (across seeds): {result.score_variance:.6f}")
        else:
            print("  Score variance: N/A (insufficient scored seeds)")
        within = result.within_seed_variance
        across = result.across_seed_variance
        snr = result.signal_to_noise
        print(
            "  Within-seed variance (repeats): "
            + (f"{within:.6f}" if within is not None else "N/A (repeats=1)")
        )
        print(
            "  Across-seed variance: "
            + (f"{across:.6f}" if across is not None else "N/A")
        )
        print(
            "  Signal-to-noise (across/within): "
            + (f"{snr:.3f}" if snr is not None else "N/A")
        )

    def _run_single_seed(
        self,
        dataset: TriviaDataset,
        config: MetaLearningConfig,
        seed: int,
        repeat: int,
        verbose: bool,
    ) -> SeedTrajectory:
        """Run training for a single (seed, repeat)."""
        start_time = time.time()
        trajectory = SeedTrajectory(seed=seed, repeat=repeat)

        # Create fresh model
        if verbose:
            print("  Loading fresh model...")
        model = self._model_factory()
        model._model_is_stable = True

        # Get effective max tokens
        effective_max_tokens = config.max_tokens or model._max_tokens

        # Create experiment record
        name = f"{config.name}_seed{seed}"
        if config.repeats > 1:
            name += f"_r{repeat}"
        experiment = Experiment(
            id=None,
            name=name,
            experiment_type=ExperimentType.EVAL,
            config_json=json.dumps(
                {
                    "seed": seed,
                    "repeat": repeat,
                    "training_iterations": config.training_iterations,
                    "checkpoint_interval": config.checkpoint_interval,
                    "train_ratio": config.train_ratio,
                    "max_tokens": config.max_tokens,
                    "training_source": config.training_source,
                    "revision_prompt": config.revision_prompt,
                    "close_think": config.close_think,
                    "holdout_every_checkpoint": config.holdout_every_checkpoint,
                    "dataset_name": dataset.name,
                    "dataset_version": dataset.version,
                }
            ),
            model_checkpoint=None,
            started_at=datetime.now(),
            completed_at=None,
        )
        experiment_id = self._db.insert_experiment(experiment)
        trajectory.experiment_id = experiment_id

        # Shuffle and split dataset. Repeats reseed identically on purpose.
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_count = int(len(indices) * config.train_ratio)
        train_indices = indices[:train_count]
        holdout_indices = indices[train_count:]
        items_by_id: dict[str, TriviaItem] = {item.id: item for item in dataset}

        if verbose:
            print(
                f"  Dataset: {len(dataset)} items ({train_count} train, {len(holdout_indices)} holdout)"
            )

        example_ids = _insert_dataset_examples(self._db, dataset)

        # Phase 1: Get baseline responses for all items
        if verbose:
            print("  Phase 1: Baseline inference...")

        baseline_responses: dict[str, tuple[str, bool]] = (
            {}
        )  # item_id -> (response, correct)
        for idx in indices:
            item = dataset[idx]
            rec = _infer_and_record(
                model,
                self._db,
                item,
                example_ids[item.id],
                experiment_id,
                Phase.BASELINE,
                config.max_tokens,
                effective_max_tokens,
            )
            baseline_responses[item.id] = (rec.clean, rec.correct)

        # Phase 2: Train with checkpoints
        if verbose:
            print(
                f"  Phase 2: Training with checkpoints (source={config.training_source})..."
            )

        post_responses: dict[str, tuple[str, bool]] = {}  # Updated as we train
        trained_items: list[str] = []  # in training order

        for batch_idx in range(0, len(train_indices), config.checkpoint_interval):
            batch = train_indices[batch_idx : batch_idx + config.checkpoint_interval]
            window_ids: list[str] = []
            revision_invalid_ids: list[str] = []

            # Train on this batch
            for idx in batch:
                item = dataset[idx]
                baseline_clean, _ = baseline_responses[item.id]
                outcome = _train_one_item(
                    model,
                    self._db,
                    item,
                    baseline_clean,
                    example_ids[item.id],
                    experiment_id,
                    config.training_iterations,
                    config.training_source,
                    config.revision_prompt,
                    config.close_think,
                )
                if outcome.revision_invalid:
                    revision_invalid_ids.append(item.id)
                    trajectory.revision_invalid_count += 1
                    if verbose:
                        print(f"    Skipped {item.id}: invalid revision")
                    continue
                trained_items.append(item.id)
                window_ids.append(item.id)

            # Checkpoint: Evaluate all trained items
            for item_id in trained_items:
                post_responses[item_id] = _judge_only(
                    model, items_by_id[item_id], config.max_tokens
                )

            holdout_probe: tuple[int, int] | None = None
            if config.holdout_every_checkpoint:
                correct = sum(
                    1
                    for idx in holdout_indices
                    if _judge_only(model, dataset[idx], config.max_tokens)[1]
                )
                holdout_probe = (correct, len(holdout_indices))

            checkpoint = self._compute_checkpoint(
                step=len(trained_items),
                baseline_responses=baseline_responses,
                post_responses=post_responses,
                trained_items=trained_items,
                window_ids=window_ids,
                revision_invalid_ids=revision_invalid_ids,
                holdout_probe=holdout_probe,
            )
            trajectory.checkpoints.append(checkpoint)

            if verbose:
                print(
                    f"    Checkpoint {len(trajectory.checkpoints)}: "
                    f"step={checkpoint.step}, "
                    f"acc={checkpoint.post_accuracy:.1%}, "
                    f"net={checkpoint.net_learning}, "
                    f"window n={checkpoint.window_size} "
                    f"(+{checkpoint.window_improved}/-{checkpoint.window_regressed})"
                )

        # Phase 3: Final evaluation on holdout
        if verbose:
            print("  Phase 3: Holdout evaluation...")

        holdout_correct = 0
        for idx in holdout_indices:
            item = dataset[idx]
            rec = _infer_and_record(
                model,
                self._db,
                item,
                example_ids[item.id],
                experiment_id,
                Phase.POST_TRAINING,
                config.max_tokens,
                effective_max_tokens,
            )
            if rec.correct:
                holdout_correct += 1

        trajectory.holdout_correct = holdout_correct
        trajectory.holdout_total = len(holdout_indices)
        if verbose:
            acc = trajectory.holdout_accuracy
            print(
                f"    Holdout accuracy: {acc:.1%}"
                if acc is not None
                else "    Holdout: none"
            )

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
        trained_items: list[str],
        window_ids: list[str] | None = None,
        revision_invalid_ids: list[str] | None = None,
        holdout_probe: tuple[int, int] | None = None,
    ) -> Checkpoint:
        """Compute metrics for a checkpoint.

        Args:
            step: Training events completed so far.
            baseline_responses: item_id -> (response, correct) before training.
            post_responses: item_id -> (response, correct) as of this checkpoint.
            trained_items: Every item trained so far (cumulative counts).
            window_ids: Items trained since the previous checkpoint (window counts).
            revision_invalid_ids: Items in this window skipped for invalid revisions.
            holdout_probe: ``(correct, total)`` for the holdout set, if probed.
        """
        window = set(window_ids or [])
        counts = {"improved": 0, "retained": 0, "regressed": 0, "stuck": 0}
        ids: dict[str, list[str]] = {k: [] for k in counts}
        window_counts = dict(counts)

        for item_id in trained_items:
            if item_id not in post_responses:
                continue
            _, baseline_correct = baseline_responses[item_id]
            _, post_correct = post_responses[item_id]

            if not baseline_correct and post_correct:
                kind = "improved"
            elif baseline_correct and post_correct:
                kind = "retained"
            elif baseline_correct and not post_correct:
                kind = "regressed"
            else:
                kind = "stuck"
            counts[kind] += 1
            ids[kind].append(item_id)
            if item_id in window:
                window_counts[kind] += 1

        baseline_correct_count = sum(1 for _, c in baseline_responses.values() if c)
        post_correct_count = sum(1 for _, c in post_responses.values() if c)

        return Checkpoint(
            step=step,
            timestamp=datetime.now().isoformat(),
            baseline_correct=baseline_correct_count,
            baseline_total=len(baseline_responses),
            post_correct=post_correct_count,
            post_total=len(post_responses),
            improved=counts["improved"],
            retained=counts["retained"],
            regressed=counts["regressed"],
            stuck=counts["stuck"],
            improved_ids=ids["improved"],
            retained_ids=ids["retained"],
            regressed_ids=ids["regressed"],
            stuck_ids=ids["stuck"],
            window_improved=window_counts["improved"],
            window_retained=window_counts["retained"],
            window_regressed=window_counts["regressed"],
            window_stuck=window_counts["stuck"],
            window_ids=list(window_ids or []),
            revision_invalid_ids=list(revision_invalid_ids or []),
            holdout_correct=holdout_probe[0] if holdout_probe else None,
            holdout_total=holdout_probe[1] if holdout_probe else None,
        )

    @property
    def db(self) -> Database:
        """Access the database for queries."""
        return self._db
