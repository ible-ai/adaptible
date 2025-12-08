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
    make_collated_training_example,
    make_revision_prompt,
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
    "make_collated_training_example",
    "make_revision_prompt",
    "REWRITE_INSTRUCTIONS",
    "strip_think_tags",
    "validate_revision_response",
    "autonomous",
    "eval",
    "local",
    "revise",
]
