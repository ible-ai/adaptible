"""Public exports, loaded on demand so runtime wrappers do not require MLX.

At runtime every name resolves lazily through `__getattr__` and `_EXPORTS`. The
`TYPE_CHECKING` block repeats the same imports for IDEs and type checkers only (it never
executes), so go-to-definition and autocomplete work. `init_test.py` keeps the two in sync.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import autonomous, eval, local, revise, wrap
    from .classes import (
        FeedbackRequest,
        FeedbackResponse,
        InteractionHistory,
        InteractionRequest,
        InteractionResponse,
        ReviewResponse,
        SyncResponse,
        TrainingExample,
    )
    from .db import (
        Database,
        Example,
        Experiment,
        ExperimentType,
        Phase,
        Response,
        SourceType,
        TrainingEvent,
        default_judge,
    )
    from .llm import StatefulLLM
    from .local.api import Adaptible, ModelProtocol
    from .local.server import MutableHostedLLM
    from .lookup import DocStore

__all__ = [
    "Adaptible",
    "ModelProtocol",
    "MutableHostedLLM",
    "FeedbackRequest",
    "FeedbackResponse",
    "InteractionHistory",
    "InteractionRequest",
    "InteractionResponse",
    "ReviewResponse",
    "SyncResponse",
    "TrainingExample",
    "StatefulLLM",
    "DocStore",
    "autonomous",
    "eval",
    "local",
    "revise",
    "wrap",
    "Database",
    "Example",
    "Experiment",
    "Response",
    "TrainingEvent",
    "ExperimentType",
    "Phase",
    "SourceType",
    "default_judge",
]

_EXPORTS = {
    "autonomous": (".", "autonomous"),
    "eval": (".", "eval"),
    "local": (".", "local"),
    "revise": (".", "revise"),
    "wrap": (".", "wrap"),
    "Adaptible": (".local.api", "Adaptible"),
    "ModelProtocol": (".local.api", "ModelProtocol"),
    "MutableHostedLLM": (".local.server", "MutableHostedLLM"),
    "FeedbackRequest": (".classes", "FeedbackRequest"),
    "FeedbackResponse": (".classes", "FeedbackResponse"),
    "InteractionHistory": (".classes", "InteractionHistory"),
    "InteractionRequest": (".classes", "InteractionRequest"),
    "InteractionResponse": (".classes", "InteractionResponse"),
    "ReviewResponse": (".classes", "ReviewResponse"),
    "SyncResponse": (".classes", "SyncResponse"),
    "TrainingExample": (".classes", "TrainingExample"),
    "StatefulLLM": (".llm", "StatefulLLM"),
    "DocStore": (".lookup", "DocStore"),
    "Database": (".db", "Database"),
    "Example": (".db", "Example"),
    "Experiment": (".db", "Experiment"),
    "Response": (".db", "Response"),
    "TrainingEvent": (".db", "TrainingEvent"),
    "ExperimentType": (".db", "ExperimentType"),
    "Phase": (".db", "Phase"),
    "SourceType": (".db", "SourceType"),
    "default_judge": (".db", "default_judge"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _EXPORTS[name]
    if module in (".", ".."):
        value = import_module(module + attr, __name__)
    else:
        value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
