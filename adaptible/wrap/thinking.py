"""Explicit generation mode and final-answer/reasoning separation."""


def generation_mode(
    body,
    architecture,
    *,
    native=False,
    path="/v1/chat/completions",
    provider=None,
    always_reasons=False,
):
    """Record only controls the selected endpoint actually understands.

    Unreproducible requests still pass through unchanged; only their repair is
    declined. Native Ollama and OpenAI-compatible routes use different controls.

    ``always_reasons`` marks a model that reasons on every request and offers no
    control to switch it off -- a distilled reasoning model on the Qwen2
    architecture, for instance. Keying thinking support off the architecture
    version instead refuses such a model a reasoning repair even though it
    produces exactly the completed thought the repair needs. It has no controls
    to reproduce, so nothing about the request can make it unreproducible.
    """
    if always_reasons:
        return dict(thinking=True, source=["model_reasons_unconditionally"])
    if architecture != "qwen3":
        return {}

    def unsupported(reason):
        return {"error": reason + "; cannot reproduce training mode."}

    if (
        any(
            body.get(key)
            for key in (
                "raw",
                "context",
                "template",
                "chat_template",
                "tools",
                "documents",
                "continue_final_message",
            )
        )
        or body.get("add_generation_prompt") is False
        or body.get("tool_choice") is not None
    ):
        return unsupported("Repair cannot reproduce custom request formatting")

    template = body.get("chat_template_kwargs")
    if template is None:
        template = {}
    if not isinstance(template, dict) or set(template) - {"enable_thinking"}:
        return unsupported("Unsupported chat template options")
    if "enable_thinking" in template and not isinstance(
        template["enable_thinking"], bool
    ):
        return unsupported("enable_thinking must be boolean for repair")

    selected = []
    if path.startswith("/api/"):
        if (
            not native
            or template
            or any(key in body for key in ("reasoning", "reasoning_effort"))
        ):
            return unsupported("Native Ollama repair uses only the think control")
        if "think" in body:
            value = body["think"]
            if isinstance(value, bool):
                selected.append(("think", value))
            elif value in ("low", "medium", "high"):
                selected.append(("think", True))
            else:
                return unsupported("Unsupported thinking option")
    else:
        if "think" in body or (native and template):
            return unsupported(
                "This OpenAI endpoint does not use the supplied native thinking control"
            )
        efforts = []
        if "reasoning_effort" in body:
            efforts.append(("reasoning_effort", body["reasoning_effort"]))
        if "reasoning" in body:
            value = body["reasoning"]
            if not native or not isinstance(value, dict) or set(value) != {"effort"}:
                return unsupported("Unsupported nested reasoning control")
            efforts.append(("reasoning.effort", value["effort"]))
        for key, effort in efforts:
            if effort not in ("none", "low", "medium", "high"):
                return unsupported("Unsupported reasoning effort")
            selected.append((key, effort != "none"))
        if "enable_thinking" in template:
            # LM Studio's documented installed control is reasoning_effort.
            # Its OpenAI handler may ignore standalone template kwargs.
            if provider == "LMStudio" and not efforts:
                return unsupported(
                    "LM Studio repair requires an explicit reasoning_effort"
                )
            selected.append(
                ("chat_template_kwargs.enable_thinking", template["enable_thinking"])
            )

    if len({value for _, value in selected}) > 1:
        return unsupported(
            "Conflicting thinking controls; repair requires one consistent mode"
        )
    return dict(
        thinking=selected[0][1] if selected else True,
        source=[key for key, _ in selected] or ["qwen3_template_default"],
    )


def framing_valid(content):
    if not isinstance(content, str):
        return False
    opens, closes = content.count("<think>"), content.count("</think>")
    if opens == closes == 0:
        return True
    if opens == 0 and closes == 1:
        return True  # Native template may prefill the opening delimiter.
    return (
        opens == closes == 1
        and content.index("<think>") < content.index("</think>")
        and not content.split("<think>", 1)[0].strip()
    )


def valid_reasoning_prefix(prefix):
    return (
        isinstance(prefix, str)
        and prefix.startswith("<think>")
        and prefix.count("<think>") == prefix.count("</think>") == 1
        and not prefix.split("</think>", 1)[1].strip()
        and bool(prefix.split("</think>", 1)[0].removeprefix("<think>").strip())
    )


def split_content(content):
    """Return final answer, reasoning, and exact observed reasoning prefix."""
    content = content or ""
    if not isinstance(content, str):
        return "", "", ""
    if not framing_valid(content):
        return "", content.removeprefix("<think>").strip(), ""
    if "</think>" in content:
        head, final = content.split("</think>", 1)
        reasoning = head.split("<think>", 1)[-1]
        whitespace = final[: len(final) - len(final.lstrip())]
        prefix = head if "<think>" in head else "<think>\n" + head
        return final.lstrip(), reasoning.strip(), prefix + "</think>" + whitespace
    if "<think>" in content:
        return "", content.split("<think>", 1)[-1].strip(), ""
    return content, "", ""


def completion_details(message, finish_reason=None):
    raw = message.get("content") or message.get("response") or ""
    valid = framing_valid(raw)
    content, inline_reasoning, prefix = split_content(raw)
    reasoning = next(
        (
            message[key]
            for key in ("reasoning_content", "reasoning", "thinking")
            if isinstance(message.get(key), str) and message[key].strip()
        ),
        inline_reasoning,
    )
    # Structured native parsers remove the delimiters. Qwen3's assistant
    # template supplies this canonical boundary when serializing that response.
    if reasoning and not prefix and content:
        prefix = "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n"
    if reasoning and prefix and not valid_reasoning_prefix(prefix):
        valid = False
    return dict(
        content=content,
        reasoning=reasoning,
        reasoning_prefix=prefix,
        finish_reason=finish_reason,
        framing_valid=valid,
        complete=valid
        and finish_reason in ("stop", "eos", "end_turn")
        and bool(content.strip()),
    )


def thinking_complete(details):
    return bool(
        details.get("complete")
        and details.get("reasoning", "").strip()
        and valid_reasoning_prefix(details.get("reasoning_prefix"))
    )


def history_messages(messages):
    """Past assistant thoughts are omitted by stock Qwen3 chat templates.

    Preserve caller dictionaries; normalize split and inline histories to the
    same final-answer content before training a following user turn.
    """
    copied = []
    for message in messages:
        value = dict(message)
        if value.get("role") == "assistant" and isinstance(value.get("content"), str):
            if not framing_valid(value["content"]):
                raise ValueError(
                    "Incomplete or malformed assistant reasoning history cannot be reproduced for training."
                )
            content, _, _ = split_content(value["content"])
            value["content"] = content
            for key in ("reasoning", "reasoning_content", "thinking"):
                value.pop(key, None)
        copied.append(value)
    return copied
