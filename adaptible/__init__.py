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
