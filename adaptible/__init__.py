"""Public exports, loaded on demand so runtime wrappers do not require MLX."""

from importlib import import_module

__all__ = [
    "Adaptible",
    "ModelProtocol",
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
    "cli",
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
    "cli": (".", "cli"),
    "local": (".", "local"),
    "revise": (".", "revise"),
    "wrap": (".", "wrap"),
    "Adaptible": ("._src._api", "Adaptible"),
    "ModelProtocol": ("._src._api", "ModelProtocol"),
    "FeedbackRequest": ("._src._classes", "FeedbackRequest"),
    "FeedbackResponse": ("._src._classes", "FeedbackResponse"),
    "InteractionHistory": ("._src._classes", "InteractionHistory"),
    "InteractionRequest": ("._src._classes", "InteractionRequest"),
    "InteractionResponse": ("._src._classes", "InteractionResponse"),
    "ReviewResponse": ("._src._classes", "ReviewResponse"),
    "SyncResponse": ("._src._classes", "SyncResponse"),
    "TrainingExample": ("._src._classes", "TrainingExample"),
    "StatefulLLM": ("._src._llm", "StatefulLLM"),
    "DocStore": ("._src.lookup", "DocStore"),
    "Database": ("._src.db", "Database"),
    "Example": ("._src.db", "Example"),
    "Experiment": ("._src.db", "Experiment"),
    "Response": ("._src.db", "Response"),
    "TrainingEvent": ("._src.db", "TrainingEvent"),
    "ExperimentType": ("._src.db", "ExperimentType"),
    "Phase": ("._src.db", "Phase"),
    "SourceType": ("._src.db", "SourceType"),
    "default_judge": ("._src.db", "default_judge"),
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
