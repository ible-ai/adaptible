"""Entry point for running evaluation as a module.

Usage:
    python -m adaptible.eval [options]

Options:
    --name NAME           Experiment name (default: "default")
    --train-ratio RATIO   Fraction to use for training (default: 0.8)
    --iterations N        Training iterations per example (default: 25)
    --shuffle             Shuffle the dataset
    --seed SEED           Random seed for shuffling (default: 42)
    --subset N            Only use first N items (for quick tests)
    --category CAT        Filter to specific category
    --output PATH         Output path for HTML report
    --save-dataset PATH   Save the dataset to JSON file
    --load-dataset PATH   Load dataset from JSON file instead of default
"""

from .._src.tests.eval.run_eval import main

if __name__ == "__main__":
    exit(main())
