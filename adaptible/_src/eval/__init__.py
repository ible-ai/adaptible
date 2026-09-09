"""Evaluation framework for Adaptible self-correction."""

from .dataset import (
    TriviaDataset,
    TriviaItem,
    generate_default_dataset,
    load_dataset,
    save_dataset,
)
from .harness import (
    TRAINING_SOURCES,
    EvaluationConfig,
    EvaluationHarness,
    EvaluationResult,
    contains_key_terms,
)
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
    "TRAINING_SOURCES",
    "contains_key_terms",
    "generate_html_report",
    # Meta-learning
    "Checkpoint",
    "MetaLearningConfig",
    "MetaLearningExperiment",
    "MetaLearningResult",
    "SeedTrajectory",
]
