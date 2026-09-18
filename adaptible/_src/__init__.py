"""Public exports, loaded on demand so runtime wrappers do not require MLX."""

from importlib import import_module

__all__ = [
    "Adaptible",
    "ModelProtocol",
    "InteractionHistory",
    "InteractionRequest",
    "InteractionResponse",
    "ReviewResponse",
    "SyncResponse",
    "TrainingExample",
    "StatefulLLM",
    "InvalidRevisionError",
    "clean_model_response",
    "make_collated_training_example",
    "make_revision_prompt",
    "REWRITE_INSTRUCTIONS",
    "strip_examples_tags",
    "strip_think_tags",
    "validate_revision_response",
    "autonomous",
    "eval",
    "local",
    "revise",
]

_EXPORTS = {
    "autonomous": (".", "autonomous"),
    "eval": (".", "eval"),
    "local": (".", "local"),
    "revise": (".", "revise"),
    "Adaptible": ("._api", "Adaptible"),
    "ModelProtocol": ("._api", "ModelProtocol"),
    "InteractionHistory": ("._classes", "InteractionHistory"),
    "InteractionRequest": ("._classes", "InteractionRequest"),
    "InteractionResponse": ("._classes", "InteractionResponse"),
    "ReviewResponse": ("._classes", "ReviewResponse"),
    "SyncResponse": ("._classes", "SyncResponse"),
    "TrainingExample": ("._classes", "TrainingExample"),
    "StatefulLLM": ("._llm", "StatefulLLM"),
    "REWRITE_INSTRUCTIONS": (".revise", "REWRITE_INSTRUCTIONS"),
    "InvalidRevisionError": (".revise", "InvalidRevisionError"),
    "clean_model_response": (".revise", "clean_model_response"),
    "make_collated_training_example": (".revise", "make_collated_training_example"),
    "make_revision_prompt": (".revise", "make_revision_prompt"),
    "strip_examples_tags": (".revise", "strip_examples_tags"),
    "strip_think_tags": (".revise", "strip_think_tags"),
    "validate_revision_response": (".revise", "validate_revision_response"),
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
