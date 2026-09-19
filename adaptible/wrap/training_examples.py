"""Collate one repair and up to two generated training wordings."""

from .prompt_format import render_training_prompt
from .thinking import history_messages, valid_reasoning_prefix


def encode_examples(tokenizer, job, architecture, *, device="cpu"):
    """Right-pad examples with independent prompt and padding loss masks.

    Additional examples are training inputs selected by the controller. External
    evaluation prompts must never be passed here.
    """
    import torch

    extras = job.get("examples", [])
    if not isinstance(extras, list) or len(extras) > 2:
        raise ValueError("Repair accepts at most two additional training examples.")
    options = job.get("training_options") or {}
    thinking = options.get("thinking", False)
    examples = [
        dict(
            messages=job.get("messages"),
            target=job.get("target"),
            reasoning_prefix=options.get("reasoning_prefix"),
        ),
        *extras,
    ]
    sequences, masked_labels, stopping_labels = [], [], []
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Each training example must contain messages and target.")
        messages, completion = example.get("messages"), example.get("target")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(completion, str)
        ):
            raise ValueError("Repair requires nonempty messages and a text target.")
        if architecture == "qwen3":

            messages = history_messages(messages)
        if options.get("prompt_format"):

            prompt_text = render_training_prompt(messages, options, architecture)
            prefix = tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
        else:
            prefix = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **({"enable_thinking": thinking} if architecture == "qwen3" else {}),
            )
        supervised_start = len(prefix)
        if thinking:

            reasoning_prefix = example.get("reasoning_prefix")
            # The requirement is a completed reasoning prefix, not a particular
            # architecture version: a Qwen2-architecture distilled reasoning
            # model supplies one just as a Qwen3 does.
            if not valid_reasoning_prefix(reasoning_prefix):
                raise ValueError(
                    "Thinking repair requires a completed nonempty reasoning prefix for every training prompt."
                )
            # Learn the grounded corrected rationale and final answer together.
            # The stopping mask separately measures the final answer only.
            if not options.get("prompt_format"):
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            continuation = reasoning_prefix
            if prompt_text.rstrip().endswith("<think>"):
                prefilled = prompt_text.rsplit("<think>", 1)[1]
                continuation = continuation.removeprefix("<think>")
                if not continuation.startswith(prefilled):
                    raise ValueError(
                        "Observed reasoning prefix does not match the template prefill."
                    )
                continuation = continuation[len(prefilled) :]
            # Tokenize the joined context once; a BPE boundary must not drift
            # between the template and the model's generated reasoning.
            prefix = tokenizer.encode(
                prompt_text + continuation, add_special_tokens=False
            )
        # Preserve the original single-example reasoning-template handling.
        if tokenizer.decode(prefix).rstrip().endswith(
            "<think>"
        ) and completion.lstrip().startswith("<think>"):
            completion = completion.lstrip()[len("<think>") :].lstrip("\n")
        target = tokenizer.encode(
            completion + (tokenizer.eos_token or ""), add_special_tokens=False
        )
        if (
            not prefix
            or not completion.strip()
            or not target
            or len(prefix) + len(target) > 4096
        ):
            raise ValueError(
                "Repair requires a nonempty tokenized prompt and target, and at most 4096 total tokens per example."
            )
        sequence = prefix + target
        if thinking:
            joined = tokenizer.encode(
                prompt_text + continuation + completion + (tokenizer.eos_token or ""),
                add_special_tokens=False,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if joined != sequence or joined[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "Tokenizer merges across the rationale or answer boundary; cannot align loss masks safely."
                )
            supervised_start = len(prompt_ids)
        else:
            supervised_start = len(prefix)
        sequences.append(sequence)
        masked_labels.append([-100] * supervised_start + sequence[supervised_start:])
        stopping_labels.append([-100] * len(prefix) + target)
    width = max(map(len, sequences))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    masks = [
        [1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences
    ]
    result = dict(
        input_ids=torch.tensor(
            [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences],
            device=device,
        ),
        labels=torch.tensor(
            [labels + [-100] * (width - len(labels)) for labels in masked_labels],
            device=device,
        ),
        attention_mask=torch.tensor(masks, device=device),
    )
    if thinking:
        result["stop_labels"] = torch.tensor(
            [labels + [-100] * (width - len(labels)) for labels in stopping_labels],
            device=device,
        )
    return result
