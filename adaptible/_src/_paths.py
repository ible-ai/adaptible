"""Filesystem locations shared by every module that persists state.

Everything Adaptible writes (experiment database, model checkpoints, autonomous
node state and logs) lives under one outputs directory so that installed and
editable installs behave the same way. Resolution order:

1. ``$ADAPTIBLE_OUTPUTS_DIR`` if set.
2. ``<cwd>/outputs`` — the repository layout when run from a checkout.

Never derive this from ``__file__``: for an installed wheel that would point
into ``site-packages``.
"""

import os
from pathlib import Path


def outputs_dir() -> Path:
    """Root directory for all persisted state."""
    override = os.environ.get("ADAPTIBLE_OUTPUTS_DIR")
    return Path(override) if override else Path.cwd() / "outputs"


def default_db_path() -> Path:
    """Location of the SQLite experiment database."""
    return outputs_dir() / "adaptible.db"


def default_checkpoint_path() -> Path:
    """Location the autonomous node saves and reloads model weights from."""
    return outputs_dir() / "autonomous" / "checkpoint"


def autonomous_state_path() -> Path:
    """Location of the autonomous node's persisted ``NodeState``."""
    return outputs_dir() / "autonomous" / "state.json"


def autonomous_log_dir() -> Path:
    """Directory the autonomous node writes per-day logs into."""
    return outputs_dir() / "autonomous" / "logs"
