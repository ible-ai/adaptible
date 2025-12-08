"""Runner script for evaluation experiments.

Usage:
    python -m adaptible._src.tests.eval.run_eval [options]

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

import argparse
import webbrowser

import vizible

from .dataset import generate_default_dataset, load_dataset, save_dataset
from .harness import EvaluationConfig, EvaluationHarness
from .report import generate_html_report


def main():
    parser = argparse.ArgumentParser(description="Run Adaptible evaluation experiments")
    parser.add_argument("--name", default="default", help="Experiment name")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train/holdout split ratio"
    )
    parser.add_argument(
        "--iterations", type=int, default=25, help="Training iterations per example"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset", type=int, help="Only use first N items")
    parser.add_argument("--category", help="Filter to specific category")
    parser.add_argument(
        "--output", default="/tmp/adaptible_eval_report.html", help="Output HTML path"
    )
    parser.add_argument("--save-dataset", help="Save dataset to JSON file")
    parser.add_argument("--load-dataset", help="Load dataset from JSON file")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    # Load or generate dataset
    if args.load_dataset:
        print(f"Loading dataset from {args.load_dataset}...")
        dataset = load_dataset(args.load_dataset)
    else:
        print("Generating default dataset...")
        dataset = generate_default_dataset()

    print(f"Dataset: {dataset.name} ({len(dataset)} items)")
    print(f"Categories: {', '.join(sorted(dataset.categories))}")

    # Save dataset if requested
    if args.save_dataset:
        save_dataset(dataset, args.save_dataset)
        print(f"Saved dataset to {args.save_dataset}")

    # Apply filters
    if args.category:
        dataset = dataset.by_category(args.category)
        print(f"Filtered to category '{args.category}': {len(dataset)} items")

    if args.subset:
        dataset = dataset.subset(list(range(min(args.subset, len(dataset)))))
        print(f"Using subset: {len(dataset)} items")

    if len(dataset) == 0:
        print("Error: No items in dataset after filtering")
        return 1

    # Create config
    config = EvaluationConfig(
        name=args.name,
        training_iterations=args.iterations,
        shuffle=args.shuffle,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )

    # Run evaluation
    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)
    print()

    harness = EvaluationHarness()
    result = harness.run(dataset, config, verbose=True)

    # Generate report
    report_path = generate_html_report(result, args.output)
    print()
    print("=" * 70)
    print("REPORT GENERATED")
    print("=" * 70)
    vizible.green(f"Report: file://{report_path}")

    if not args.no_browser:
        webbrowser.open_new_tab(f"file://{report_path}")

    return 0


if __name__ == "__main__":
    exit(main())
