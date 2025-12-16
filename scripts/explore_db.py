"""Interactive database exploration script.

Usage:
    python scripts/explore_db.py          # Use existing DB at outputs/adaptible.db
    python scripts/explore_db.py --demo   # Create demo data first
"""

import argparse
import sys
import datetime
import pathlib

# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from adaptible import (
    Database,
    Example,
    Experiment,
    Response,
    TrainingEvent,
    ExperimentType,
    Phase,
    SourceType,
    default_judge,
)


def create_demo_data(db: Database) -> int:
    """Populate DB with synthetic demo data. Returns experiment ID."""
    print("Creating demo data...")

    # Create experiment
    exp = Experiment(
        id=None,
        name="demo_eval",
        experiment_type=ExperimentType.EVAL,
        config_json='{"training_iterations": 25, "train_ratio": 0.8}',
        model_checkpoint=None,
        started_at=datetime.datetime.now(),
        completed_at=None,
    )
    exp_id = db.insert_experiment(exp)

    # Demo examples with various outcomes
    demo_items = [
        # Improvements (wrong -> right)
        (
            "geo_001",
            "What is the capital of Australia?",
            "Canberra",
            ["Canberra"],
            "Sydney",
            "Canberra",
        ),
        (
            "geo_002",
            "What is the capital of Canada?",
            "Ottawa",
            ["Ottawa"],
            "Toronto",
            "Ottawa",
        ),
        # Regressions (right -> wrong)
        (
            "geo_003",
            "What is the capital of France?",
            "Paris",
            ["Paris"],
            "Paris",
            "Lyon",
        ),
        # Stuck (wrong -> wrong)
        (
            "geo_004",
            "What is the capital of Myanmar?",
            "Naypyidaw",
            ["Naypyidaw"],
            "Yangon",
            "Rangoon",
        ),
        (
            "geo_005",
            "What is the capital of Nigeria?",
            "Abuja",
            ["Abuja"],
            "Lagos",
            "Lagos",
        ),
        # Retained (right -> right)
        ("sci_001", "What is the chemical symbol for gold?", "Au", ["Au"], "Au", "Au"),
        ("sci_002", "What planet is the Red Planet?", "Mars", ["Mars"], "Mars", "Mars"),
        # Holdout items (not trained)
        (
            "sci_003",
            "What is the powerhouse of the cell?",
            "Mitochondria",
            ["mitochondria"],
            "Mitochondria",
            "Mitochondria",
        ),
        (
            "sci_004",
            "What is the largest planet?",
            "Jupiter",
            ["Jupiter"],
            "Saturn",
            "Saturn",
        ),
    ]

    trained_ids = {
        "geo_001",
        "geo_002",
        "geo_003",
        "geo_004",
        "geo_005",
        "sci_001",
        "sci_002",
    }

    for item_id, question, answer, key_terms, baseline, post in demo_items:
        # Insert example
        ex = Example(
            id=None,
            canonical_id=item_id,
            question=question,
            ground_truth_answer=answer,
            key_terms=key_terms,
            category=item_id.split("_")[0],
            difficulty="medium",
            source_type=SourceType.STATIC_TRIVIA,
            source_url=None,
            source_title=None,
            valid_at=None,
            created_at=None,
        )
        ex_id = db.insert_example(ex)

        # Insert baseline response
        db.insert_response(
            Response(
                id=None,
                example_id=ex_id,
                experiment_id=exp_id,
                response_text=baseline,
                response_raw=baseline,
                confidence=None,
                phase=Phase.BASELINE,
                created_at=None,
            )
        )

        # Insert training event if trained
        if item_id in trained_ids:
            db.insert_training_event(
                TrainingEvent(
                    id=None,
                    example_id=ex_id,
                    experiment_id=exp_id,
                    training_iterations=25,
                    training_time_seconds=2.5,
                    created_at=None,
                )
            )

        # Insert post-training response
        db.insert_response(
            Response(
                id=None,
                example_id=ex_id,
                experiment_id=exp_id,
                response_text=post,
                response_raw=post,
                confidence=None,
                phase=Phase.POST_TRAINING,
                created_at=None,
            )
        )

    db.complete_experiment(exp_id)
    print(f"Created experiment {exp_id} with {len(demo_items)} examples")
    return exp_id


def interactive_session(db: Database, exp_id: int | None = None):
    """Drop into an interactive session with helpful variables."""

    print("\n" + "=" * 60)
    print("INTERACTIVE DATABASE EXPLORER")
    print("=" * 60)
    print(
        """
Available variables:
  db          - Database instance
  exp_id      - Latest experiment ID (if any)

Useful methods:
  db.get_experiments()                    - List all experiments
  db.get_examples(category="geography")   - Get examples by category
  db.compute_metrics(exp_id)              - Get metrics for an experiment
  db.get_regressions(exp_id)              - Examples that got worse
  db.get_improvements(exp_id)             - Examples that improved
  db.get_stuck(exp_id)                    - Examples still wrong after training
  db.get_example_history("geo_001")       - History for a specific question
  db.export_experiment_summary(exp_id)    - LLM-friendly summary
  db.export_comparison([1, 2, 3])         - Compare multiple experiments

Example queries:
  >>> metrics = db.compute_metrics(exp_id)
  >>> print(f"Improvement rate: {metrics['improvement_rate']:.1%}")

  >>> for r in db.get_regressions(exp_id):
  ...     print(f"{r['example'].canonical_id}: {r['baseline_text']} -> {r['post_text']}")

  >>> print(db.export_experiment_summary(exp_id))
"""
    )

    # Get latest experiment if not provided
    if exp_id is None:
        experiments = db.get_experiments(limit=1)
        if experiments:
            exp_id = experiments[0].id
            print(f"Using latest experiment: {exp_id} ({experiments[0].name})")
        else:
            print("No experiments found. Run with --demo to create sample data.")

    # Quick preview
    if exp_id:
        metrics = db.compute_metrics(exp_id)
        print(f"\nQuick metrics for experiment {exp_id}:")
        print(f"  Baseline accuracy:  {metrics['baseline_accuracy']:.1%}")
        print(f"  Post accuracy:      {metrics['post_accuracy']:.1%}")
        print(f"  Improvement rate:   {metrics['improvement_rate']:.1%}")
        print(f"  Retention rate:     {metrics['retention_rate']:.1%}")
        print(f"  Regressions:        {metrics['regression_count']}")

    print("\n" + "=" * 60)
    print("Dropping into Python REPL. Type exit() to quit.")
    print("=" * 60 + "\n")

    # Start interactive session
    import code

    code.interact(local={"db": db, "exp_id": exp_id, "default_judge": default_judge})


def main():
    parser = argparse.ArgumentParser(
        description="Explore the adaptible experiment database"
    )
    parser.add_argument("--demo", action="store_true", help="Create demo data")
    parser.add_argument("--db", type=str, default=None, help="Path to database file")
    parser.add_argument(
        "--summary",
        type=int,
        metavar="EXP_ID",
        help="Print summary for experiment and exit",
    )
    args = parser.parse_args()

    db = Database(args.db) if args.db else Database()
    print(f"Database: {db.db_path}")

    if args.demo:
        exp_id = create_demo_data(db)
    else:
        exp_id = None

    if args.summary:
        print(db.export_experiment_summary(args.summary))
        return

    interactive_session(db, exp_id)


if __name__ == "__main__":
    main()
