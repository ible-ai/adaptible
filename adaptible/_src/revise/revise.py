"""Utilities for self-reflective model revisions."""

import re
from typing import Literal, Sequence, Tuple

import mlx.core as mx
from transformers.tokenization_utils import PreTrainedTokenizer

from .._classes import InteractionHistory, TrainingExample

REWRITE_INSTRUCTIONS = (
    "You are a professional editor that coaches people and LLMs on how to improve their "
    "conversational skills. "
    "Today, you have been asked to inspect a dialog between a human user and an LLM and revise "
    "one of the LLM's responses. "
    "It is important to consider the full context of the conversation, especially if the LLM's "
    "responses were not satisfactory during the dialog. "
    "Be sure that your revised response(s) would be a realistic response considering that the "
    "later context of the conversation was not yet known.\n\n"
    "Your response MUST:\n"
    '* Choose an "assistant" response to revise. Label your rewritten response with the '
    "dialog turn that it relates to by starting your response with [[X]] "
    "(e.g. [[0]], [[1]], etc.).\n"
    "* End your response with [[/X]]. This separator is imperative so please do not "
    "forget it.\n"
    "* Since you are revising one of the existing responses, make sure your [[X]] label "
    "corresponds with a dialog index that already exists within the original dialog.\n"
    "* Write ONLY the improved response between the [[X]] and [[/X]] markers."
)


# A shorter, imperative variant with two worked examples. Meant for small models
# that ignore the prose instructions above and re-answer the question instead of
# emitting a marked rewrite. Pair it with dialog_style="plain" so the dialog the
# model sees looks exactly like the examples (no chat-template special tokens).
REWRITE_INSTRUCTIONS_FEWSHOT = (
    "Rewrite one Assistant reply from the dialog so it is more accurate, complete, "
    "and helpful. Output ONLY the rewrite, wrapped in the markers of the turn you "
    "are rewriting: start with [[X]] and end with [[/X]], where X is the turn "
    "number shown in the dialog. Do not explain your changes. Do not restate or "
    "repeat the dialog. Do not write anything before [[X]] or after [[/X]].\n"
    "\n"
    "Example 1\n"
    "Dialog:\n"
    "[[0]] User: What is the capital of Australia?\n"
    "Assistant: The capital of Australia is Sydney.\n"
    "Output:\n"
    "[[0]] The capital of Australia is Canberra. Sydney is the largest city, but "
    "Canberra has been the capital since 1913. [[/0]]\n"
    "\n"
    "Example 2\n"
    "Dialog:\n"
    "[[0]] User: How many legs does a spider have?\n"
    "Assistant: I am not sure, maybe six?\n"
    "Output:\n"
    "[[0]] A spider has eight legs. Insects have six legs, which is one way to "
    "tell spiders and insects apart. [[/0]]\n"
    "\n"
    "Now do the same for the dialog below. Output only the marked rewrite."
)

# Names accepted by revision_prompt_preset (and the --revision_prompt CLI flag).
REVISION_PROMPTS = ("default", "fewshot")

DialogStyle = Literal["chat", "plain"]

# Prepended to the training target when the chat template's generation prompt
# ends with an open "<think>" tag (DeepSeek-R1-Distill renders
# "...<｜Assistant｜><think>\n"). Without it the revision is trained *inside* an
# unclosed think block, a sequence the model never produces at inference; the
# model then stops reasoning after a single training example.
THINK_CLOSE = "</think>\n\n"

# How the training target relates to the chat template's open ``<think>`` block
# (see ``make_revision_training_example``):
#   none:     the revision goes straight after the open tag, inside the think
#             block. The original, malformed target; kept only for comparison.
#   empty:    ``</think>\n\n`` is inserted before the revision, all of it in the
#             loss. Teaches "skip reasoning, answer" as a style: after ~80 items
#             DeepSeek-R1-Distill answered everything with an empty think block
#             and 6-token responses, and holdout accuracy halved.
#   baseline: the model's own reasoning from its original response is placed in
#             the (unmasked) prefix and only the corrected answer is in the loss:
#             "given this reasoning, the answer is X". Falls back to "empty"
#             when the original response carried no reasoning.
THINK_MODES = ("none", "empty", "baseline")


def validate_think_mode(think_mode: str) -> None:
    """Raise ValueError unless ``think_mode`` is one of ``THINK_MODES``."""
    if think_mode not in THINK_MODES:
        raise ValueError(f"think_mode must be one of {THINK_MODES}, got {think_mode!r}")


def resolve_think_mode(think_mode: str, close_think: bool | None) -> str:
    """Fold the deprecated ``close_think`` flag into a ``think_mode``.

    ``close_think=False`` means the old unclosed target, i.e. ``"none"``.
    ``close_think=True`` or ``None`` leaves ``think_mode`` alone, except that
    ``True`` contradicts ``"none"`` and raises.
    """
    validate_think_mode(think_mode)
    if close_think is False:
        return "none"
    if close_think is True and think_mode == "none":
        raise ValueError('close_think=True contradicts think_mode="none"')
    return think_mode


def revision_prompt_preset(name: str) -> tuple[str, DialogStyle]:
    """Map a preset name to ``(instructions, dialog_style)`` for make_revision_prompt.

    Args:
        name: "default" (REWRITE_INSTRUCTIONS rendered through the tokenizer's chat
            template) or "fewshot" (REWRITE_INSTRUCTIONS_FEWSHOT over a plain
            ``User:``/``Assistant:`` dialog).

    Raises:
        ValueError: If ``name`` is not a known preset.
    """
    if name == "default":
        return REWRITE_INSTRUCTIONS, "chat"
    if name == "fewshot":
        return REWRITE_INSTRUCTIONS_FEWSHOT, "plain"
    raise ValueError(
        f"revision_prompt must be one of {REVISION_PROMPTS}, got {name!r}"
    )


def _make_revision_prompt(
    past_dialog: str, instructions: str = REWRITE_INSTRUCTIONS
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
    if not rewritten_indices:
        raise ValueError(f"Failed to parse a turn ID from {model_response}")
    unique_rewritten_indices = set(rewritten_indices)
    index_to_rewrite = min(map(int, unique_rewritten_indices))
    return index_to_rewrite


def _parse_rewritten_response(model_response: str, idx: int) -> str:
    sor_index = None
    eor_index = None
    for group in re.finditer(rf"\[\[{idx}\]\]", model_response):
        sor_index = group.end()
    for group in re.finditer(rf"\[\[/{idx}\]\]", model_response):
        eor_index = group.start()
    return model_response[sor_index:eor_index].strip()


class InvalidRevisionError(ValueError):
    """Raised when a model-generated revision is invalid or low quality."""

    pass


def validate_revision_response(
    model_response: str,
    num_interactions: int,
    min_content_length: int = 10,
) -> None:
    """Validate that a model-generated revision response is usable.

    Args:
        model_response: The raw model output from the revision prompt.
        num_interactions: Number of interactions in the original dialog.
        min_content_length: Minimum length for the revised content.

    Raises:
        InvalidRevisionError: If the response is invalid or unusable.
    """
    # Check for markers
    if not re.search(r"\[\[[0-9]+\]\]", model_response):
        raise InvalidRevisionError(
            "Revision response does not contain valid [[X]] markers. "
            f"Response: {model_response[:200]}..."
        )

    # Extract the turn index
    try:
        idx = _isolate_turn_to_rewritten_turn_index(model_response)
    except ValueError as e:
        raise InvalidRevisionError(str(e)) from e

    # Validate index is within bounds
    if idx < 0 or idx >= num_interactions:
        raise InvalidRevisionError(
            f"Turn index {idx} is out of bounds. "
            f"Valid range: 0 to {num_interactions - 1}."
        )

    # Check for closing marker
    if not re.search(rf"\[\[/{idx}\]\]", model_response):
        raise InvalidRevisionError(
            f"Revision response missing closing marker [[/{idx}]]."
        )

    # Extract and validate content
    content = _parse_rewritten_response(model_response, idx)

    if len(content) < min_content_length:
        raise InvalidRevisionError(
            f"Revised content is too short ({len(content)} chars). "
            f"Content: {content!r}"
        )

    # Check for garbage patterns (repeated tags, etc.)
    garbage_patterns = [
        r"(\]\]\s*){3,}",  # Multiple consecutive ]] patterns
        r"(</?[A-Z_]+>){3,}",  # Multiple consecutive XML-like tags
        r"(\[\[\d+\]\]){3,}",  # Multiple consecutive [[X]] markers
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, content):
            raise InvalidRevisionError(
                f"Revised content appears to be garbage/repetitive. "
                f"Content: {content[:100]}..."
            )


def strip_think_tags(text: str | None) -> str:
    """Remove <think>...</think> tags and their content from text.

    Handles multiple formats:
    - Full tags: <think>content</think>
    - Partial/unclosed: everything before </think>
    - Case insensitive

    Args:
        text: Input text potentially containing think tags.

    Returns:
        Text with think tags and their content removed.
    """
    if text is None:
        return ""
    # First try to remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Also handle cases where only </think> appears (partial tag)
    cleaned = re.sub(r".*</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def split_think(text: str | None) -> Tuple[str, str]:
    """Split a response into its reasoning and its answer.

    Handles ``<think>...</think>answer``, bare ``reasoning</think>answer``
    (DeepSeek-R1-Distill, whose generation prompt already opened the tag), and
    text with no tag at all (think is ``""``). Both parts are stripped.

    Args:
        text: Raw model response.

    Returns:
        ``(think, answer)``.
    """
    if text is None:
        return "", ""
    match = re.search(r"</think>", text, flags=re.IGNORECASE)
    if match is None:
        return "", text.strip()
    think = text[: match.start()]
    answer = text[match.end() :]
    think = re.sub(r"^\s*<think>", "", think, count=1, flags=re.IGNORECASE)
    return think.strip(), answer.strip()


def strip_examples_tags(text: str | None) -> str:
    """Remove content before </EXAMPLES> closing tag.

    Used for few-shot prompts where the model may echo examples.

    Args:
        text: Input text potentially containing examples tags.

    Returns:
        Text after </EXAMPLES> tag, or original if not found.
    """
    if text is None:
        return ""
    if "</EXAMPLES>" in text:
        return text.split("</EXAMPLES>")[-1].strip()
    return text.strip()


def clean_model_response(text: str | None) -> str:
    """Clean model response by removing think tags and examples echoing.

    Convenience function that applies both strip_think_tags and
    strip_examples_tags in the correct order.

    Args:
        text: Raw model response.

    Returns:
        Cleaned response text.
    """
    cleaned = strip_examples_tags(text)
    cleaned = strip_think_tags(cleaned)
    return cleaned


def _serialize_interactions_to_string(
    interactions: Sequence[InteractionHistory],
    should_enumerate: bool,
    tokenizer: PreTrainedTokenizer,
    continue_final_message: bool,
    strip_thinking: bool = True,
    dialog_style: DialogStyle = "chat",
) -> Tuple[str, Sequence[str]]:
    """Render interactions as text, one entry per turn.

    Args:
        dialog_style: "chat" renders each turn through ``tokenizer.apply_chat_template``
            (so the text carries the model's special tokens); "plain" renders
            ``User: ...\nAssistant: ...`` with no tokenizer involvement.
    """
    if dialog_style not in ("chat", "plain"):
        raise ValueError(f"dialog_style must be 'chat' or 'plain', got {dialog_style!r}")
    turns = []
    for interaction in interactions:
        llm_response = interaction.llm_response
        if strip_thinking:
            llm_response = strip_think_tags(llm_response)
        if dialog_style == "plain":
            turns.append(f"User: {interaction.user_input}\nAssistant: {llm_response}")
            continue
        messages = [
            {
                "role": "user",
                "content": interaction.user_input,
            },
            {
                "role": "assistant",
                "content": llm_response,
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
            [
                f"[[{idx}]]{' ' if dialog_style == 'plain' else ''}{turn}"
                for idx, turn in enumerate(turns)
            ]
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


def padding_token_for(tokenizer: PreTrainedTokenizer) -> int:
    """The id used to pad a batch: the tokenizer's pad id, else its eos id, else 0.

    Padded positions are masked out of the loss either way; this only keeps
    the padded *input* positions on a real token instead of id 0.
    """
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def _collate_fn(
    batch_data: Sequence[TrainingExample], padding_token: int = 0
) -> TrainingExample:
    """Right-pad unbatched examples to a common length and stack them.

    Inputs and labels are padded with ``padding_token``; the mask is always
    padded with 0 so padded positions never contribute to the loss.
    """
    max_len = max(max(map(len, (d.input, d.label, d.mask))) for d in batch_data)
    padded_inputs = []
    padded_labels = []
    padded_masks = []
    for item in batch_data:
        padded_inputs.append(_pad(item.input, max_len, padding_token))
        padded_labels.append(_pad(item.label, max_len, padding_token))
        padded_masks.append(_pad(item.mask, max_len, 0))
    return TrainingExample(
        input=mx.stack(padded_inputs),
        label=mx.stack(padded_labels),
        mask=mx.stack(padded_masks),
    )


def collate_training_examples(
    examples: Sequence[TrainingExample], tokenizer: PreTrainedTokenizer
) -> TrainingExample:
    """Batch unbatched examples (from ``make_training_example``) for one training step.

    Row order is preserved. Padding uses ``padding_token_for(tokenizer)``.
    """
    return _collate_fn(examples, padding_token_for(tokenizer))


def make_revision_prompt(
    interactions: Sequence[InteractionHistory],
    tokenizer: PreTrainedTokenizer,
    instructions: str = REWRITE_INSTRUCTIONS,
    dialog_style: DialogStyle = "chat",
) -> str:
    """Create a prompt for model self-reflective revision based on past interactions.

    Args:
        interactions: Past interactions.
        tokenizer: Tokenizer whose chat template renders the dialog when
            ``dialog_style="chat"``; unused for ``"plain"``.
        instructions: Revision prompt pre-amble.
        dialog_style: How the past dialog is rendered; see
            ``_serialize_interactions_to_string``. ``revision_prompt_preset``
            returns a matching ``(instructions, dialog_style)`` pair.

    Returns:
        Formatted prompt text.
    """
    past_dialog, _ = _serialize_interactions_to_string(
        interactions=interactions,
        should_enumerate=True,
        tokenizer=tokenizer,
        continue_final_message=False,
        dialog_style=dialog_style,
    )
    return _make_revision_prompt(past_dialog, instructions)


def make_training_example(
    prompt_messages: Sequence[dict[str, str]],
    target_text: str,
    tokenizer: PreTrainedTokenizer,
    prompt_suffix: str = "",
) -> TrainingExample:
    """Build one unbatched training example: masked prompt, unmasked target.

    The sequence is ``chat_template(prompt_messages, add_generation_prompt=True)
    + prompt_suffix + target_text``. The loss mask is 0 over the template and
    ``prompt_suffix`` and 1 over ``target_text``. ``target_text`` should already
    end with the eos token if the model is meant to learn to stop.

    The input/label/mask alignment is ``seq[:-1]``, ``seq[1:]``, ``mask[1:]``:
    position ``i`` of the input predicts ``seq[i + 1]``, so the mask must be
    aligned with the labels.

    Args:
        prompt_messages: Chat messages rendered through the tokenizer's chat
            template with ``add_generation_prompt=True``.
        target_text: Text the model is trained to produce after the prompt.
        tokenizer: Model-specific tokenizer.
        prompt_suffix: Extra text placed between the template and the target
            that is *not* in the loss (e.g. the model's own reasoning).

    Returns:
        A 1-D ``TrainingExample``; use ``collate_training_examples`` to batch.
    """
    prompt_prefix = tokenizer.apply_chat_template(
        list(prompt_messages), tokenize=False, add_generation_prompt=True
    )

    def _tokenize(text: str, dtype: mx.Dtype = mx.int32) -> mx.array:
        return mx.array(tokenizer.encode(text, add_special_tokens=False), dtype=dtype)

    prompt = _tokenize(prompt_prefix + prompt_suffix)
    target = _tokenize(target_text)
    sequence = mx.concat([prompt, target])
    mask = mx.concat([mx.zeros_like(prompt), mx.ones_like(target)])
    return TrainingExample(
        input=sequence[:-1],
        label=sequence[1:],
        mask=mask[1:],
    )


def _eos_tag(tokenizer: PreTrainedTokenizer) -> str:
    eos_tag = tokenizer.special_tokens_map.get("eos_token", "")
    if isinstance(eos_tag, list):
        eos_tag = eos_tag[0]
    return eos_tag


def make_revision_training_example(
    response: str,
    interactions: Sequence[InteractionHistory],
    tokenizer: PreTrainedTokenizer,
    think_mode: str = "empty",
    close_think: bool | None = None,
) -> TrainingExample:
    """Convert past interactions and a model revision into an unbatched training example.

    The prompt prefix is always the tokenizer's real chat template with
    ``add_generation_prompt=True`` so training matches inference, regardless of
    how the *revision prompt* rendered the dialog (see ``make_revision_prompt``'s
    ``dialog_style``).

    When the generation prompt ends with an open ``<think>`` tag (DeepSeek-R1-
    Distill renders ``...<｜Assistant｜><think>\n``) the target depends on
    ``think_mode``; see ``THINK_MODES``. For ``"baseline"`` the sequence is
    ``{prefix}{baseline_think}\n</think>\n\n{revision}{eos}`` where
    ``baseline_think`` is the reasoning in the revised turn's ``llm_response``
    (``split_think``), and only ``{revision}{eos}`` is in the loss. Templates
    without a think tag are unaffected by ``think_mode``.

    Args:
        response: Model-generated revision response (``[[X]] ... [[/X]]``).
        interactions: Past interactions considered when generating the model
            response; the revised turn's ``llm_response`` should be the raw
            model output (with its think block) for ``"baseline"`` to work.
        tokenizer: Model-specific tokenizer.
        think_mode: One of ``THINK_MODES``.
        close_think: Deprecated alias; ``False`` forces ``think_mode="none"``.

    Returns: a 1-D training example.
    """
    think_mode = resolve_think_mode(think_mode, close_think)
    eos_tag = _eos_tag(tokenizer)

    index_to_rewrite = _isolate_turn_to_rewritten_turn_index(response)
    rewritten_response = _parse_rewritten_response(response, index_to_rewrite)

    # Build the prompt prefix using the same format as generation: every turn
    # before the revised one as user/assistant pairs (think blocks stripped, as
    # the chat template does for prior assistant turns), then the revised
    # turn's user message. For index 0 this is just that user message.
    messages = []
    for prior in interactions[:index_to_rewrite]:
        messages.append({"role": "user", "content": prior.user_input})
        messages.append(
            {"role": "assistant", "content": strip_think_tags(prior.llm_response)}
        )
    interaction_to_revise = interactions[index_to_rewrite]
    messages.append({"role": "user", "content": interaction_to_revise.user_input})

    prompt_prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    open_think = prompt_prefix.rstrip().endswith("<think>")

    prompt_suffix = ""
    target_text = rewritten_response + eos_tag
    if open_think and think_mode == "baseline":
        baseline_think, _ = split_think(interaction_to_revise.llm_response)
        if baseline_think:
            prompt_suffix = f"{baseline_think}\n{THINK_CLOSE}"
        else:
            think_mode = "empty"
    if open_think and think_mode == "empty":
        target_text = THINK_CLOSE + target_text
    return make_training_example(messages, target_text, tokenizer, prompt_suffix)


def make_collated_training_example(
    response: str,
    interactions: Sequence[InteractionHistory],
    tokenizer: PreTrainedTokenizer,
    padding_token: int = 0,
    close_think: bool | None = None,
    think_mode: str = "empty",
) -> TrainingExample:
    """``make_revision_training_example`` batched to shape ``(1, L)``.

    Args:
        response: Model-generated revision response.
        interactions: Past interactions considered when generating the model response.
        tokenizer: Model-specific tokenizer.
        padding_token: Token to use for padding (unused for a single example).
        close_think: Deprecated alias for ``think_mode``: ``False`` reproduces
            the old target with the revision inside the open think block.
        think_mode: See ``THINK_MODES`` and ``make_revision_training_example``.

    Returns: a collated training example, ready for model ingestion.
    """
    example = make_revision_training_example(
        response, interactions, tokenizer, think_mode=think_mode, close_think=close_think
    )
    return _collate_fn([example], padding_token)
