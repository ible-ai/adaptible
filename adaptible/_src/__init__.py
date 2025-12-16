"""Core components for stateful LLMs."""

from . import autonomous, eval, local, revise
from ._api import Adaptible, ModelProtocol
from ._classes import (
    InteractionHistory,
    InteractionRequest,
    InteractionResponse,
    ReviewResponse,
    SyncResponse,
    TrainingExample,
)
from ._llm import StatefulLLM
from .revise import (
    REWRITE_INSTRUCTIONS,
    InvalidRevisionError,
    clean_model_response,
    make_collated_training_example,
    make_revision_prompt,
    strip_examples_tags,
    strip_think_tags,
    validate_revision_response,
)

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
