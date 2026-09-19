"""Verified training renderers for native templates that differ from HF Jinja."""

from .thinking import history_messages

# Official registry.ollama.ai/library/qwen3:0.6b template layer, fetched as
# metadata only. Arbitrary Go templates must not be guessed from architecture.
OLLAMA_QWEN3_TEMPLATE_SHA256 = (
    "ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4"
)
OLLAMA_QWEN3_FORMAT = "ollama_qwen3_stock_v1"


def render_ollama_qwen3(messages, *, system="", thinking=False):
    """Match the stock Ollama template for plain text, without requiring Go.

    Ollama v0.12.10 collates adjacent roles and system messages before Go
    rendering. Its non-thinking request adds both /no_think and an empty think
    block; HF's template only adds the latter. Tool and reasoning-trace histories
    require additional native handling and are rejected instead of guessed.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError(
            "Ollama Qwen3 training requires a conversation ending with a user message."
        )

    copied = []
    for message in history_messages(messages):
        if (
            message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or any(message.get(key) for key in ("tool_calls", "images"))
        ):
            raise ValueError(
                "Ollama Qwen3 training currently supports plain text histories without tool calls or reasoning traces."
            )
        copied.append(dict(role=message["role"], content=message["content"]))
    if copied[0]["role"] != "system" and system:
        copied.insert(0, dict(role="system", content=system))
    system_text = "\n\n".join(m["content"] for m in copied if m["role"] == "system")
    collated = []
    for message in copied:
        if collated and collated[-1]["role"] == message["role"]:
            collated[-1]["content"] += "\n\n" + message["content"]
        else:
            collated.append(message.copy())
    parts = []
    if system_text:
        parts.append("<|im_start|>system\n\n" + system_text + "<|im_end|>\n")
    last_user = max(i for i, m in enumerate(collated) if m["role"] == "user")
    for index, message in enumerate(collated):
        role, content = message["role"], message["content"]
        if role == "system":
            continue
        if index == last_user:
            content += " /think" if thinking else " /no_think"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if not thinking:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


def render_training_prompt(messages, options, architecture):
    if (
        architecture != "qwen3"
        or options.get("prompt_format") != OLLAMA_QWEN3_FORMAT
        or options.get("template_sha256") != OLLAMA_QWEN3_TEMPLATE_SHA256
    ):
        raise ValueError("Unsupported native training prompt format.")
    return render_ollama_qwen3(
        messages,
        system=options.get("system", ""),
        thinking=options.get("thinking", False),
    )
