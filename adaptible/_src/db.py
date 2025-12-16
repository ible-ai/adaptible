"""Unified experiment database for tracking examples, responses, and training events.

This module provides a SQLite-based storage layer that unifies data from both
the eval framework and autonomous learning node. It enables:
- Tracking the same question across multiple experiments
- Querying for regressions, improvements, and stuck examples
- Time-aware ground truth for web-scraped claims
- Configurable correctness judgments at query time
"""

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


# Default database location
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "outputs" / "adaptible.db"


class SourceType(Enum):
    """Origin of an example."""

    STATIC_TRIVIA = "static_trivia"
    WEB_SCRAPE = "web_scrape"


class Phase(Enum):
    """When a response was generated relative to training."""

    BASELINE = "baseline"
    POST_TRAINING = "post_training"


class ExperimentType(Enum):
    """Type of experiment that generated responses."""

    EVAL = "eval"
    AUTONOMOUS = "autonomous"


# Type alias for judge functions
Judge = Callable[[str, str, list[str] | None], bool]


def default_judge(response: str, ground_truth: str, key_terms: list[str] | None) -> bool:
    """Default correctness judge using key term matching.

    Args:
        response: The model's response (cleaned of think tags).
        ground_truth: The correct answer.
        key_terms: Optional list of terms that indicate correctness.

    Returns:
        True if the response is considered correct.
    """
    if key_terms:
        response_lower = response.lower()
        return any(term.lower() in response_lower for term in key_terms)
    # Fallback: check if ground truth appears in response
    return ground_truth.lower() in response.lower()


@dataclass
class Example:
    """A question with ground truth answer."""

    id: int | None
    canonical_id: str
    question: str
    ground_truth_answer: str
    key_terms: list[str] | None
    category: str | None
    difficulty: str | None
    source_type: SourceType
    source_url: str | None
    source_title: str | None
    valid_at: date | None  # None for timeless facts
    created_at: datetime | None


@dataclass
class Response:
    """A model response to an example."""

    id: int | None
    example_id: int
    experiment_id: int
    response_text: str
    response_raw: str
    confidence: float | None
    phase: Phase
    created_at: datetime | None
    # Token metadata for debugging context-length issues
    token_count: int | None = None
    max_tokens: int | None = None  # The limit that was set
    truncated: bool | None = None  # True if response hit max_tokens


@dataclass
class Experiment:
    """An evaluation or autonomous learning run."""

    id: int | None
    name: str
    experiment_type: ExperimentType
    config_json: str
    model_checkpoint: str | None
    started_at: datetime | None
    completed_at: datetime | None


@dataclass
class TrainingEvent:
    """Record of training on a specific example."""

    id: int | None
    example_id: int
    experiment_id: int
    training_iterations: int
    training_time_seconds: float
    created_at: datetime | None


SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id TEXT NOT NULL,
    question TEXT NOT NULL,
    ground_truth_answer TEXT NOT NULL,
    key_terms TEXT,  -- comma-separated
    category TEXT,
    difficulty TEXT,
    source_type TEXT NOT NULL,
    source_url TEXT,
    source_title TEXT,
    valid_at DATE,  -- NULL for timeless facts
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_id, valid_at)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    config_json TEXT NOT NULL,
    model_checkpoint TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    example_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    response_text TEXT NOT NULL,
    response_raw TEXT NOT NULL,
    confidence REAL,
    phase TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER,
    max_tokens INTEGER,
    truncated INTEGER,  -- SQLite doesn't have bool, use 0/1
    FOREIGN KEY (example_id) REFERENCES examples(id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS training_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    example_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    training_iterations INTEGER NOT NULL,
    training_time_seconds REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (example_id) REFERENCES examples(id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_examples_canonical_id ON examples(canonical_id);
CREATE INDEX IF NOT EXISTS idx_examples_valid_at ON examples(valid_at);
CREATE INDEX IF NOT EXISTS idx_examples_source_type ON examples(source_type);
CREATE INDEX IF NOT EXISTS idx_examples_category ON examples(category);
CREATE INDEX IF NOT EXISTS idx_responses_example_id ON responses(example_id);
CREATE INDEX IF NOT EXISTS idx_responses_experiment_id ON responses(experiment_id);
CREATE INDEX IF NOT EXISTS idx_responses_phase ON responses(phase);
CREATE INDEX IF NOT EXISTS idx_training_events_example_id ON training_events(example_id);
CREATE INDEX IF NOT EXISTS idx_training_events_experiment_id ON training_events(experiment_id);
"""


def canonical_id_for_question(question: str) -> str:
    """Generate a stable canonical ID for a question.

    Uses a hash of the normalized question text.
    """
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class Database:
    """SQLite database for experiment tracking."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Run migrations for existing databases
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Run schema migrations for existing databases."""
        # Check if responses table has new columns
        cursor = conn.execute("PRAGMA table_info(responses)")
        columns = {row[1] for row in cursor.fetchall()}

        # Add token metadata columns if missing
        if "token_count" not in columns:
            conn.execute("ALTER TABLE responses ADD COLUMN token_count INTEGER")
        if "max_tokens" not in columns:
            conn.execute("ALTER TABLE responses ADD COLUMN max_tokens INTEGER")
        if "truncated" not in columns:
            conn.execute("ALTER TABLE responses ADD COLUMN truncated INTEGER")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Examples
    # =========================================================================

    def insert_example(self, example: Example) -> int:
        """Insert an example, returning its ID.

        If an example with the same canonical_id and valid_at exists, returns
        the existing ID instead.
        """
        key_terms_str = ",".join(example.key_terms) if example.key_terms else None

        with self._connect() as conn:
            # Try to find existing
            cursor = conn.execute(
                """
                SELECT id FROM examples
                WHERE canonical_id = ? AND (valid_at = ? OR (valid_at IS NULL AND ? IS NULL))
                """,
                (example.canonical_id, example.valid_at, example.valid_at),
            )
            row = cursor.fetchone()
            if row:
                return row["id"]

            # Insert new
            cursor = conn.execute(
                """
                INSERT INTO examples (
                    canonical_id, question, ground_truth_answer, key_terms,
                    category, difficulty, source_type, source_url, source_title, valid_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    example.canonical_id,
                    example.question,
                    example.ground_truth_answer,
                    key_terms_str,
                    example.category,
                    example.difficulty,
                    example.source_type.value,
                    example.source_url,
                    example.source_title,
                    example.valid_at,
                ),
            )
            return cursor.lastrowid  # type: ignore

    def get_example(self, example_id: int) -> Example | None:
        """Get an example by ID."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM examples WHERE id = ?", (example_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_example(row)
            return None

    def get_example_by_canonical(
        self, canonical_id: str, valid_at: date | None = None
    ) -> Example | None:
        """Get an example by canonical ID and validity date."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM examples
                WHERE canonical_id = ? AND (valid_at = ? OR (valid_at IS NULL AND ? IS NULL))
                """,
                (canonical_id, valid_at, valid_at),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_example(row)
            return None

    def get_examples(
        self,
        source_type: SourceType | None = None,
        category: str | None = None,
        valid_at: date | None = None,
        limit: int | None = None,
    ) -> list[Example]:
        """Query examples with optional filters."""
        query = "SELECT * FROM examples WHERE 1=1"
        params: list[Any] = []

        if source_type is not None:
            query += " AND source_type = ?"
            params.append(source_type.value)

        if category is not None:
            query += " AND category = ?"
            params.append(category)

        if valid_at is not None:
            query += " AND valid_at = ?"
            params.append(valid_at)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_example(row) for row in cursor.fetchall()]

    def _row_to_example(self, row: sqlite3.Row) -> Example:
        """Convert a database row to an Example."""
        key_terms = row["key_terms"].split(",") if row["key_terms"] else None
        valid_at = None
        if row["valid_at"]:
            valid_at = date.fromisoformat(row["valid_at"])
        created_at = None
        if row["created_at"]:
            created_at = datetime.fromisoformat(row["created_at"])
        return Example(
            id=row["id"],
            canonical_id=row["canonical_id"],
            question=row["question"],
            ground_truth_answer=row["ground_truth_answer"],
            key_terms=key_terms,
            category=row["category"],
            difficulty=row["difficulty"],
            source_type=SourceType(row["source_type"]),
            source_url=row["source_url"],
            source_title=row["source_title"],
            valid_at=valid_at,
            created_at=created_at,
        )

    # =========================================================================
    # Experiments
    # =========================================================================

    def insert_experiment(self, experiment: Experiment) -> int:
        """Insert an experiment, returning its ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments (
                    name, experiment_type, config_json, model_checkpoint, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment.name,
                    experiment.experiment_type.value,
                    experiment.config_json,
                    experiment.model_checkpoint,
                    experiment.started_at or datetime.now(),
                ),
            )
            return cursor.lastrowid  # type: ignore

    def complete_experiment(self, experiment_id: int) -> None:
        """Mark an experiment as completed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET completed_at = ? WHERE id = ?",
                (datetime.now(), experiment_id),
            )

    def get_experiment(self, experiment_id: int) -> Experiment | None:
        """Get an experiment by ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_experiment(row)
            return None

    def get_experiments(
        self,
        experiment_type: ExperimentType | None = None,
        limit: int | None = None,
    ) -> list[Experiment]:
        """Query experiments with optional filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params: list[Any] = []

        if experiment_type is not None:
            query += " AND experiment_type = ?"
            params.append(experiment_type.value)

        query += " ORDER BY started_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_experiment(row) for row in cursor.fetchall()]

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        """Convert a database row to an Experiment."""
        started_at = None
        if row["started_at"]:
            started_at = datetime.fromisoformat(row["started_at"])
        completed_at = None
        if row["completed_at"]:
            completed_at = datetime.fromisoformat(row["completed_at"])
        return Experiment(
            id=row["id"],
            name=row["name"],
            experiment_type=ExperimentType(row["experiment_type"]),
            config_json=row["config_json"],
            model_checkpoint=row["model_checkpoint"],
            started_at=started_at,
            completed_at=completed_at,
        )

    # =========================================================================
    # Responses
    # =========================================================================

    def insert_response(self, response: Response) -> int:
        """Insert a response, returning its ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO responses (
                    example_id, experiment_id, response_text, response_raw,
                    confidence, phase, token_count, max_tokens, truncated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.example_id,
                    response.experiment_id,
                    response.response_text,
                    response.response_raw,
                    response.confidence,
                    response.phase.value,
                    response.token_count,
                    response.max_tokens,
                    1 if response.truncated else (0 if response.truncated is False else None),
                ),
            )
            return cursor.lastrowid  # type: ignore

    def get_responses_for_example(
        self,
        example_id: int,
        experiment_id: int | None = None,
        phase: Phase | None = None,
    ) -> list[Response]:
        """Get all responses for an example."""
        query = "SELECT * FROM responses WHERE example_id = ?"
        params: list[Any] = [example_id]

        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)

        if phase is not None:
            query += " AND phase = ?"
            params.append(phase.value)

        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_response(row) for row in cursor.fetchall()]

    def get_responses_for_experiment(
        self, experiment_id: int, phase: Phase | None = None
    ) -> list[Response]:
        """Get all responses for an experiment."""
        query = "SELECT * FROM responses WHERE experiment_id = ?"
        params: list[Any] = [experiment_id]

        if phase is not None:
            query += " AND phase = ?"
            params.append(phase.value)

        query += " ORDER BY example_id, created_at"

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_response(row) for row in cursor.fetchall()]

    def _row_to_response(self, row: sqlite3.Row) -> Response:
        """Convert a database row to a Response."""
        created_at = None
        if row["created_at"]:
            created_at = datetime.fromisoformat(row["created_at"])
        # Handle truncated field (SQLite stores as 0/1/NULL)
        truncated = None
        if "truncated" in row.keys() and row["truncated"] is not None:
            truncated = bool(row["truncated"])
        return Response(
            id=row["id"],
            example_id=row["example_id"],
            experiment_id=row["experiment_id"],
            response_text=row["response_text"],
            response_raw=row["response_raw"],
            confidence=row["confidence"],
            phase=Phase(row["phase"]),
            created_at=created_at,
            token_count=row["token_count"] if "token_count" in row.keys() else None,
            max_tokens=row["max_tokens"] if "max_tokens" in row.keys() else None,
            truncated=truncated,
        )

    # =========================================================================
    # Training Events
    # =========================================================================

    def insert_training_event(self, event: TrainingEvent) -> int:
        """Insert a training event, returning its ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO training_events (
                    example_id, experiment_id, training_iterations, training_time_seconds
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    event.example_id,
                    event.experiment_id,
                    event.training_iterations,
                    event.training_time_seconds,
                ),
            )
            return cursor.lastrowid  # type: ignore

    def get_training_events_for_example(self, example_id: int) -> list[TrainingEvent]:
        """Get all training events for an example."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM training_events WHERE example_id = ? ORDER BY created_at DESC",
                (example_id,),
            )
            return [self._row_to_training_event(row) for row in cursor.fetchall()]

    def get_training_events_for_experiment(
        self, experiment_id: int
    ) -> list[TrainingEvent]:
        """Get all training events for an experiment."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM training_events WHERE experiment_id = ? ORDER BY created_at",
                (experiment_id,),
            )
            return [self._row_to_training_event(row) for row in cursor.fetchall()]

    def _row_to_training_event(self, row: sqlite3.Row) -> TrainingEvent:
        """Convert a database row to a TrainingEvent."""
        created_at = None
        if row["created_at"]:
            created_at = datetime.fromisoformat(row["created_at"])
        return TrainingEvent(
            id=row["id"],
            example_id=row["example_id"],
            experiment_id=row["experiment_id"],
            training_iterations=row["training_iterations"],
            training_time_seconds=row["training_time_seconds"],
            created_at=created_at,
        )

    # =========================================================================
    # Analysis Queries
    # =========================================================================

    def get_example_history(
        self, canonical_id: str, judge: Judge | None = None
    ) -> list[dict[str, Any]]:
        """Get the full history of responses for a question across all experiments.

        Returns a list of dicts with example, responses, and computed correctness.
        """
        if judge is None:
            judge = default_judge

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT e.*, r.*, exp.name as experiment_name, exp.experiment_type
                FROM examples e
                JOIN responses r ON e.id = r.example_id
                JOIN experiments exp ON r.experiment_id = exp.id
                WHERE e.canonical_id = ?
                ORDER BY e.valid_at DESC NULLS FIRST, r.created_at DESC
                """,
                (canonical_id,),
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            example = self._row_to_example(row)
            is_correct = judge(
                row["response_text"], example.ground_truth_answer, example.key_terms
            )
            results.append(
                {
                    "example": example,
                    "response_text": row["response_text"],
                    "response_raw": row["response_raw"],
                    "confidence": row["confidence"],
                    "phase": row["phase"],
                    "experiment_name": row["experiment_name"],
                    "experiment_type": row["experiment_type"],
                    "is_correct": is_correct,
                    "created_at": row["created_at"],
                }
            )
        return results

    def get_regressions(
        self, experiment_id: int, judge: Judge | None = None
    ) -> list[dict[str, Any]]:
        """Get examples that regressed (were correct baseline, incorrect post-training).

        Returns list of dicts with example info, baseline response, and post response.
        """
        if judge is None:
            judge = default_judge

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT e.*,
                       b.response_text as baseline_text, b.response_raw as baseline_raw,
                       p.response_text as post_text, p.response_raw as post_raw
                FROM examples e
                JOIN responses b ON e.id = b.example_id AND b.phase = 'baseline'
                JOIN responses p ON e.id = p.example_id AND p.phase = 'post_training'
                WHERE b.experiment_id = ? AND p.experiment_id = ?
                """,
                (experiment_id, experiment_id),
            )
            rows = cursor.fetchall()

        regressions = []
        for row in rows:
            example = self._row_to_example(row)
            baseline_correct = judge(
                row["baseline_text"], example.ground_truth_answer, example.key_terms
            )
            post_correct = judge(
                row["post_text"], example.ground_truth_answer, example.key_terms
            )
            if baseline_correct and not post_correct:
                regressions.append(
                    {
                        "example": example,
                        "baseline_text": row["baseline_text"],
                        "baseline_raw": row["baseline_raw"],
                        "post_text": row["post_text"],
                        "post_raw": row["post_raw"],
                    }
                )
        return regressions

    def get_improvements(
        self, experiment_id: int, judge: Judge | None = None
    ) -> list[dict[str, Any]]:
        """Get examples that improved (were incorrect baseline, correct post-training).

        Returns list of dicts with example info, baseline response, and post response.
        """
        if judge is None:
            judge = default_judge

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT e.*,
                       b.response_text as baseline_text, b.response_raw as baseline_raw,
                       p.response_text as post_text, p.response_raw as post_raw
                FROM examples e
                JOIN responses b ON e.id = b.example_id AND b.phase = 'baseline'
                JOIN responses p ON e.id = p.example_id AND p.phase = 'post_training'
                WHERE b.experiment_id = ? AND p.experiment_id = ?
                """,
                (experiment_id, experiment_id),
            )
            rows = cursor.fetchall()

        improvements = []
        for row in rows:
            example = self._row_to_example(row)
            baseline_correct = judge(
                row["baseline_text"], example.ground_truth_answer, example.key_terms
            )
            post_correct = judge(
                row["post_text"], example.ground_truth_answer, example.key_terms
            )
            if not baseline_correct and post_correct:
                improvements.append(
                    {
                        "example": example,
                        "baseline_text": row["baseline_text"],
                        "baseline_raw": row["baseline_raw"],
                        "post_text": row["post_text"],
                        "post_raw": row["post_raw"],
                    }
                )
        return improvements

    def get_stuck(
        self, experiment_id: int, judge: Judge | None = None
    ) -> list[dict[str, Any]]:
        """Get examples that remained incorrect after training.

        Returns list of dicts with example info, baseline response, and post response.
        """
        if judge is None:
            judge = default_judge

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT e.*,
                       b.response_text as baseline_text, b.response_raw as baseline_raw,
                       p.response_text as post_text, p.response_raw as post_raw
                FROM examples e
                JOIN responses b ON e.id = b.example_id AND b.phase = 'baseline'
                JOIN responses p ON e.id = p.example_id AND p.phase = 'post_training'
                JOIN training_events t ON e.id = t.example_id AND t.experiment_id = ?
                WHERE b.experiment_id = ? AND p.experiment_id = ?
                """,
                (experiment_id, experiment_id, experiment_id),
            )
            rows = cursor.fetchall()

        stuck = []
        for row in rows:
            example = self._row_to_example(row)
            baseline_correct = judge(
                row["baseline_text"], example.ground_truth_answer, example.key_terms
            )
            post_correct = judge(
                row["post_text"], example.ground_truth_answer, example.key_terms
            )
            if not baseline_correct and not post_correct:
                stuck.append(
                    {
                        "example": example,
                        "baseline_text": row["baseline_text"],
                        "baseline_raw": row["baseline_raw"],
                        "post_text": row["post_text"],
                        "post_raw": row["post_raw"],
                    }
                )
        return stuck

    def compute_metrics(
        self, experiment_id: int, judge: Judge | None = None
    ) -> dict[str, Any]:
        """Compute summary metrics for an experiment.

        Returns dict with accuracy, improvement rate, retention rate, etc.
        """
        if judge is None:
            judge = default_judge

        with self._connect() as conn:
            # Get all examples with baseline and post responses
            cursor = conn.execute(
                """
                SELECT e.*,
                       b.response_text as baseline_text,
                       p.response_text as post_text,
                       CASE WHEN t.id IS NOT NULL THEN 1 ELSE 0 END as was_trained
                FROM examples e
                JOIN responses b ON e.id = b.example_id AND b.phase = 'baseline' AND b.experiment_id = ?
                LEFT JOIN responses p ON e.id = p.example_id AND p.phase = 'post_training' AND p.experiment_id = ?
                LEFT JOIN training_events t ON e.id = t.example_id AND t.experiment_id = ?
                """,
                (experiment_id, experiment_id, experiment_id),
            )
            rows = cursor.fetchall()

        total = 0
        baseline_correct = 0
        post_correct = 0
        trained_count = 0
        trained_improved = 0
        trained_regressed = 0
        trained_baseline_correct = 0
        trained_post_correct = 0
        holdout_count = 0
        holdout_correct = 0

        for row in rows:
            example = self._row_to_example(row)
            total += 1

            b_correct = judge(
                row["baseline_text"], example.ground_truth_answer, example.key_terms
            )
            if b_correct:
                baseline_correct += 1

            if row["post_text"]:
                p_correct = judge(
                    row["post_text"], example.ground_truth_answer, example.key_terms
                )
                if p_correct:
                    post_correct += 1

                if row["was_trained"]:
                    trained_count += 1
                    if b_correct:
                        trained_baseline_correct += 1
                    if p_correct:
                        trained_post_correct += 1
                    if not b_correct and p_correct:
                        trained_improved += 1
                    if b_correct and not p_correct:
                        trained_regressed += 1
                else:
                    holdout_count += 1
                    if p_correct:
                        holdout_correct += 1

        return {
            "total_examples": total,
            "baseline_accuracy": baseline_correct / total if total > 0 else 0,
            "post_accuracy": post_correct / total if total > 0 else 0,
            "trained_count": trained_count,
            "trained_baseline_accuracy": (
                trained_baseline_correct / trained_count if trained_count > 0 else 0
            ),
            "trained_post_accuracy": (
                trained_post_correct / trained_count if trained_count > 0 else 0
            ),
            "improvement_rate": (
                trained_improved / (trained_count - trained_baseline_correct)
                if trained_count > trained_baseline_correct
                else 0
            ),
            "retention_rate": (
                (trained_baseline_correct - trained_regressed) / trained_baseline_correct
                if trained_baseline_correct > 0
                else 1.0
            ),
            "regression_count": trained_regressed,
            "holdout_count": holdout_count,
            "holdout_accuracy": holdout_correct / holdout_count if holdout_count > 0 else 0,
        }

    # =========================================================================
    # Export for LLM Analysis
    # =========================================================================

    def export_experiment_summary(
        self, experiment_id: int, judge: Judge | None = None
    ) -> str:
        """Export an experiment summary in LLM-friendly text format.

        Returns a structured text summary suitable for feeding to an LLM
        for analysis.
        """
        if judge is None:
            judge = default_judge

        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return f"Experiment {experiment_id} not found."

        metrics = self.compute_metrics(experiment_id, judge)
        regressions = self.get_regressions(experiment_id, judge)
        improvements = self.get_improvements(experiment_id, judge)
        stuck = self.get_stuck(experiment_id, judge)

        lines = [
            f"# Experiment Summary: {experiment.name}",
            f"Type: {experiment.experiment_type.value}",
            f"Started: {experiment.started_at}",
            f"Completed: {experiment.completed_at}",
            "",
            "## Metrics",
            f"- Total examples: {metrics['total_examples']}",
            f"- Baseline accuracy: {metrics['baseline_accuracy']:.1%}",
            f"- Post-training accuracy: {metrics['post_accuracy']:.1%}",
            f"- Trained examples: {metrics['trained_count']}",
            f"- Improvement rate: {metrics['improvement_rate']:.1%}",
            f"- Retention rate: {metrics['retention_rate']:.1%}",
            f"- Regressions: {metrics['regression_count']}",
            f"- Holdout accuracy: {metrics['holdout_accuracy']:.1%}",
            "",
        ]

        if regressions:
            lines.append(f"## Regressions ({len(regressions)} examples)")
            lines.append("These examples got WORSE after training:")
            lines.append("")
            for r in regressions[:10]:  # Limit to 10 for brevity
                lines.append(f"### {r['example'].canonical_id}")
                lines.append(f"Question: {r['example'].question}")
                lines.append(f"Ground truth: {r['example'].ground_truth_answer}")
                lines.append(f"Baseline (correct): {r['baseline_text'][:200]}...")
                lines.append(f"Post-training (wrong): {r['post_text'][:200]}...")
                lines.append("")

        if stuck:
            lines.append(f"## Stuck Examples ({len(stuck)} examples)")
            lines.append("These examples remained incorrect after training:")
            lines.append("")
            for s in stuck[:10]:
                lines.append(f"### {s['example'].canonical_id}")
                lines.append(f"Question: {s['example'].question}")
                lines.append(f"Ground truth: {s['example'].ground_truth_answer}")
                lines.append(f"Baseline: {s['baseline_text'][:200]}...")
                lines.append(f"Post-training: {s['post_text'][:200]}...")
                lines.append("")

        if improvements:
            lines.append(f"## Improvements ({len(improvements)} examples)")
            lines.append("These examples improved after training:")
            lines.append("")
            for i in improvements[:10]:
                lines.append(f"### {i['example'].canonical_id}")
                lines.append(f"Question: {i['example'].question}")
                lines.append(f"Ground truth: {i['example'].ground_truth_answer}")
                lines.append(f"Baseline (wrong): {i['baseline_text'][:200]}...")
                lines.append(f"Post-training (correct): {i['post_text'][:200]}...")
                lines.append("")

        return "\n".join(lines)

    def export_comparison(
        self,
        experiment_ids: Sequence[int],
        judge: Judge | None = None,
    ) -> str:
        """Export a comparison of multiple experiments.

        Returns a structured text comparison suitable for LLM analysis.
        """
        if judge is None:
            judge = default_judge

        lines = ["# Experiment Comparison", ""]

        experiments = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                metrics = self.compute_metrics(exp_id, judge)
                experiments.append((exp, metrics))

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append(
            "| Experiment | Baseline | Post | Improvement | Retention | Regressions |"
        )
        lines.append(
            "|------------|----------|------|-------------|-----------|-------------|"
        )
        for exp, metrics in experiments:
            lines.append(
                f"| {exp.name} | {metrics['baseline_accuracy']:.1%} | "
                f"{metrics['post_accuracy']:.1%} | {metrics['improvement_rate']:.1%} | "
                f"{metrics['retention_rate']:.1%} | {metrics['regression_count']} |"
            )
        lines.append("")

        return "\n".join(lines)
