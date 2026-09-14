"""Public alias for :mod:`adaptible._src.cli` (a real package so `python -m adaptible.cli` works)."""

from .._src.cli import Client, main  # noqa: F401

__all__ = ["Client", "main"]
