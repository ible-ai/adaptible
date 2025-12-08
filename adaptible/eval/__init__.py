"""Evaluation framework for Adaptible self-correction.

This module provides tools for systematically evaluating the effectiveness
of Adaptible's self-correction mechanism across large datasets.

Usage:
    # Run from command line
    python -m adaptible.eval --subset 20 --shuffle

    # Or use programmatically
    import adaptible.eval as eval

    dataset = eval.generate_default_dataset()
    harness = eval.EvaluationHarness()
    result = harness.run(dataset, eval.EvaluationConfig(name="my_experiment"))
    eval.generate_html_report(result, "/tmp/report.html")

Classes:
    TriviaDataset: Container for evaluation items with filtering methods
    TriviaItem: Single question/answer pair with key terms
    EvaluationHarness: Runs train/holdout evaluation experiments
    EvaluationConfig: Configuration for evaluation runs
    EvaluationResult: Results with computed metrics

Functions:
    generate_default_dataset: Create the built-in 100+ question dataset
    load_dataset: Load a dataset from JSON
    save_dataset: Save a dataset to JSON
    generate_html_report: Create an HTML report from results
"""

from .._src.tests.eval import (
    TriviaDataset,
    TriviaItem,
    load_dataset,
    save_dataset,
    generate_default_dataset,
    EvaluationHarness,
    EvaluationConfig,
    EvaluationResult,
    generate_html_report,
)

__all__ = [
    "TriviaDataset",
    "TriviaItem",
    "load_dataset",
    "save_dataset",
    "generate_default_dataset",
    "EvaluationHarness",
    "EvaluationConfig",
    "EvaluationResult",
    "generate_html_report",
]
