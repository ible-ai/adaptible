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
        flagged: The user marked the response as wrong (a thumbs-down). Flagged
            interactions go through ``StatefulLLM.repair`` on review: the model
            looks the question up, writes a corrected answer, and trains on it.
        note: What the lookup returned for this interaction, if anything.
    """

    idx: int
    user_input: str
    llm_response: str = ""
    reviewed: bool = False
    timestamp: float = 0.0
    flagged: bool = False
    note: str = ""


@dataclasses.dataclass
class TrainingExample:
    """Pre-tokenized training data

    Attributes:
        input: Token ids fed to the model.
        label: Target token ids (``input`` shifted by one).
        mask: Loss mask: 1 over the positions trained on, 0 elsewhere.
        stop_mask: Optional mask, same shape as ``mask``, marking only the
            *answer* tokens (plus eos) of the target. Training stops on the
            loss over these positions rather than over the whole ``mask``, so
            a target that also carries a rationale is driven until the answer
            lands, not until the reasoning is memorised. ``None`` means the
            stop loss is the training loss.
    """

    input: mx.array
    label: mx.array
    mask: mx.array
    stop_mask: mx.array | None = None


class InteractionRequest(BaseModel):
    """User prompt to be sent to the LLM.

    Attributes:
        prompt: User-provided input.
        use_history: Whether earlier turns of the conversation are included in
            the prompt. ``False`` asks the question as a fresh chat.
    """

    prompt: str
    use_history: bool = True


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


class FeedbackRequest(BaseModel):
    """User feedback on one earlier response.

    Attributes:
        interaction_idx: Index of the response being rated (``InteractionResponse.interaction_idx``).
        thumbs: ``"down"`` flags the response as wrong; ``"up"`` clears the flag.
    """

    interaction_idx: int
    thumbs: str = "down"


class FeedbackResponse(BaseModel):
    """Outcome of recording feedback.

    Attributes:
        interaction_idx: Index that was rated.
        flagged: Whether the interaction is now flagged for repair.
    """

    interaction_idx: int
    flagged: bool
