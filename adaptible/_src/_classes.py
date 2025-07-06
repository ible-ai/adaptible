"""Common class definitions."""

import dataclasses
import mlx.core as mx
from pydantic import BaseModel


@dataclasses.dataclass
class InteractionHistory:
    """Event turn during user-LLM dialog

    Attributes:
        idx: Index of current turn amongst all global turns.
        user_input: User-provided prompt.
        llm_response: LLM response.
        reviewed: Whether this interaction has been reviewed already.
        timestamp: When the interaction took place, measured in seconds.
    """
    idx: int
    user_input: str
    llm_response: str = ""
    reviewed: bool = False
    timestamp: float = 0.0

@dataclasses.dataclass
class TrainingExample:
    """Pre-tokenized training data

    Attributes:
        user_input: Actual model response.
        label: Target model response.
        mask: Mask of model response that dictates which parts are used in training.
    """
    input: mx.array
    label: mx.array
    mask: mx.array


class InteractionRequest(BaseModel):
    """User prompt to be sent to the LLM.

    Attributes:
        prompt: User-provided input.
    """
    prompt: str


class InteractionResponse(BaseModel):
    """LLM response to user-provided prompt
    
    Attributes:
        response: LLM-generated text response.
        interaction_id: Index of the current response within the context of the current session.
    """
    response: str
    interaction_idx: int


class ReviewResponse(BaseModel):
    """Response to initiating asynchronous review of entire unreviewed interaction history.

    Attributes:
        message: Human-readable output after completion of review.
        unreviewed_count: the number of unreviewed interactions handled by this operation.
    """
    message: str
    unreviewed_count: int


class SyncResponse(BaseModel):
    """Response after server completes all unfinished background tasks.

    Attributes:
        message: Human-readable message.
        tasks_count: Number of tasks waited on and successfully finished.
        elapsed_time: Amount of time background tasks took to finish.
    """
    message: str
    tasks_count: int
    elapsed_time: float
