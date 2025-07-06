"""Utilities for self-reflective model revisions."""

import copy
import re
from typing import Sequence, Tuple

import mlx.core as mx
from transformers.tokenization_utils import PreTrainedTokenizer

from .._classes import InteractionHistory, TrainingExample

_REWRITE_INSTRUCTIONS = (
    "You are a professional editor that coaches people and LLMs on how to improve their "
    "conversational skills."
    "Today, you have been asked to inspect a dialog between a human user and an LLM and revise "
    "one of the LLM's responses."
    "It is important to consider the full context of the conversation, especially if the LLM's "
    "responses were not satisfactory during the dialog."
    "Be sure that your revised response(s) would be a realistic response considering that the "
    "later context of the conversation was not yet known."
    "Your response MUST:"
    '    * Choose an "assistant" response to revise. Label your rewritten response with the '
    "dialog turn that it relates to by starting your response with '''[[<X>]] "
    "(e.g. '''[[0]], '''[[1]], etc.)."
    "    * End your response with [[/<X>]]'''. This separator is imperative so please do not "
    "forget it."
    "    * Since you are revising one of the existing responses, make sure your '''[[<X>]] label "
    "corresponds with a dialog index that already exists within the original dialog."
)


def _make_revision_prompt(
    past_dialog: str, instructions: str = _REWRITE_INSTRUCTIONS
) -> str:
    """Create an LLM prompt for the model to critique and rewrite a previous response.

    Args:
        past_dialog: Relevant conversation history pre-serialized to a string.
        instructions: Prompt pre-amble containing instructions on how to revise the model response.

    Returns: a formatted text prompt.
    """
    return "\n".join(
        (
            instructions,
            "<PAST_DIALOG>",
            past_dialog,
            "</PAST_DIALOG>",
        )
    )


def _isolate_turn_to_rewritten_turn_index(model_response: str) -> int:
    rewritten_indices = re.findall(r"\[\[([0-9]*)\]\]", model_response)
    unique_rewritten_indices = set(rewritten_indices)
    index_to_rewrite = min(map(int, unique_rewritten_indices))

    if not rewritten_indices:
        raise ValueError(f"Failed to parse a turn ID from {model_response}")
    return index_to_rewrite


def _parse_rewritten_response(model_response: str, idx: int) -> str:
    sor_index = None
    eor_index = None
    for group in re.finditer(rf"\[\[{idx}\]\]", model_response):
        sor_index = group.end()
    for group in re.finditer(rf"\[\[/{idx}\]\]", model_response):
        eor_index = group.start()
    return model_response[sor_index:eor_index].strip()


def _serialize_interactions_to_string(
    interactions: Sequence[InteractionHistory],
    should_enumerate: bool,
    tokenizer: PreTrainedTokenizer,
    continue_final_message: bool,
) -> Tuple[str, Sequence[str]]:
    turns = []
    for interaction in interactions:
        messages = [
            {
                "role": "user",
                "content": interaction.user_input,
            },
            {
                "role": "assistant",
                "content": interaction.llm_response,
            },
        ]
        turn = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            continue_final_message=continue_final_message,
        )
        turns.append(turn)
    if should_enumerate:
        turns_as_text = "\n".join(
            [f"[[{idx}]]{turn}" for idx, turn in enumerate(turns)]
        )
    else:
        turns_as_text = "\n".join(turns)
    return turns_as_text, turns


def _pad(arr: mx.array, max_len: int, padding_token: int) -> mx.array:
    return mx.pad(
        arr,
        (0, max_len - len(arr)),
        mode="constant",
        constant_values=padding_token,
    )


def _collate_fn(
    batch_data: Sequence[TrainingExample], padding_token: int = 0
) -> TrainingExample:
    # Pad sequences to the maximum length in the batch
    max_len = max(max(map(len, (d.input, d.label, d.mask))) for d in batch_data)
    padded_inputs = []
    padded_labels = []
    padded_masks = []
    for item in batch_data:
        padded_inputs.append(_pad(item.input, max_len, padding_token))
        padded_labels.append(_pad(item.label, max_len, padding_token))
        padded_masks.append(_pad(item.mask, max_len, padding_token))
    return TrainingExample(
        input=mx.stack(padded_inputs),
        label=mx.stack(padded_labels),
        mask=mx.stack(padded_masks),
    )


def make_revision_prompt(
    interactions: Sequence[InteractionHistory],
    tokenizer: PreTrainedTokenizer,
    instructions: str = _REWRITE_INSTRUCTIONS,
) -> str:
    """Create a prompt for model self-reflective revision based on past interactions.
    
    Args:
        interactions: Past interactions.
        tokenizer: Tokenizer. Not actually used here. TODO - fix.
        instructions: Revision prompt.

    Returns: 
        Formatted prompt text.
    """
    past_dialog, _ = _serialize_interactions_to_string(
        interactions=interactions,
        should_enumerate=True,
        tokenizer=tokenizer,
        continue_final_message=False,
    )
    return _make_revision_prompt(past_dialog, instructions)

def make_collated_training_example(
    response: str,
    interactions: Sequence[InteractionHistory],
    tokenizer: PreTrainedTokenizer,
    padding_token: int = 0,
) -> TrainingExample:
    """Convert past interactions and model revision response into a batched training example.
    
    Args:
        response: Model-generated revision response.
        interactions: Past interactions considered when generating the model response.
        tokenizer: Model-specific tokenizer.
        padding_token: Token to use for padding.

    Returns: a collated training example, ready for model ingestion.
    
    """
    _, turns = _serialize_interactions_to_string(
        interactions=interactions,
        should_enumerate=True,
        tokenizer=tokenizer,
        continue_final_message=False,
    )
    eos_tag = tokenizer.special_tokens_map.get("eos_token", "")
    if isinstance(eos_tag, list):
        eos_tag = eos_tag[0]
    bos_tag = tokenizer.special_tokens_map.get("bos_token", "")
    if isinstance(bos_tag, list):
        bos_tag = bos_tag[0]
    index_to_rewrite = _isolate_turn_to_rewritten_turn_index(response)
    rewritten_response = _parse_rewritten_response(response, index_to_rewrite)
    interaction_to_revise = copy.deepcopy(interactions[index_to_rewrite])
    interaction_to_revise.llm_response = ""
    revised_dialog_turn, _ = _serialize_interactions_to_string(
        interactions=[interaction_to_revise],
        should_enumerate=False,
        tokenizer=tokenizer,
        continue_final_message=True,
    )
    def _tokenize(text: str, dtype: mx.Dtype = mx.int32) -> mx.array:
        return mx.array(tokenizer.encode(text, add_special_tokens=False), dtype=dtype)


    dialog_pre_revision = _tokenize(
        "\n".join(list(turns[:index_to_rewrite]) + [revised_dialog_turn])
    )
    revision = _tokenize(bos_tag + rewritten_response + eos_tag)
    tokenized_rewritten_dialog = mx.concat([dialog_pre_revision, revision])
    mask = mx.concat([mx.zeros_like(dialog_pre_revision), mx.ones_like(revision)])
    training_example = TrainingExample(
        input=tokenized_rewritten_dialog[:-1],
        label=tokenized_rewritten_dialog[1:],
        mask=mask,
    )
    return _collate_fn([training_example], padding_token)
