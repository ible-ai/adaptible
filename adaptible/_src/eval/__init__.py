"""Evaluation framework for Adaptible self-correction."""

from .dataset import (
    TriviaDataset,
    TriviaItem,
    generate_default_dataset,
    load_dataset,
    save_dataset,
)
from .harness import EvaluationConfig, EvaluationHarness, EvaluationResult
from .meta import (
    Checkpoint,
    MetaLearningConfig,
    MetaLearningExperiment,
    MetaLearningResult,
    SeedTrajectory,
)
from .report import generate_html_report

__all__ = [
    # Dataset
    "TriviaDataset",
    "TriviaItem",
    "load_dataset",
    "save_dataset",
    "generate_default_dataset",
    # Evaluation
    "EvaluationHarness",
    "EvaluationConfig",
    "EvaluationResult",
    "generate_html_report",
    # Meta-learning
    "Checkpoint",
    "MetaLearningConfig",
    "MetaLearningExperiment",
    "MetaLearningResult",
    "SeedTrajectory",
]
