"""The original's two generation loop breakers, replayed over a finished generation.

`StatefulLLM.generate_response` stops decoding when a streamed line has already
appeared twice, or when the last 8 tokens repeat 3 times in a row
(`_detect_token_loop`); the experiment's `sample` applies the token rule alone.
A runtime has no such rule, so without this a looping thought runs on to the
token cap, can close, and is judged on an answer the original never produced.

Greedy continuation never changes what came before it, so cutting a finished
generation where the original would have stopped gives the original's text.
Both rules see text in the pieces mlx_lm's streaming detokenizer emits, so that
detokenizer is reproduced here: MLX is not a dependency of the wrapper.
"""

# adaptible._src._llm._LOOP_DETECTION_SEQUENCE_LENGTH / _MAX_REPETITIONS.
SEQUENCE_LENGTH = 8
MAX_REPETITIONS = 3


def token_loop(tokens, sequence_length=SEQUENCE_LENGTH, repetitions=MAX_REPETITIONS):
    """`_llm._detect_token_loop`: the last run of tokens repeated back to back."""
    if len(tokens) < sequence_length * repetitions:
        return False
    recent = tokens[-sequence_length:]
    return all(
        tokens[-(i + 1) * sequence_length : -i * sequence_length] == recent
        for i in range(1, repetitions)
    )


class _Detokenizer:
    """mlx_lm.tokenizer_utils.BPEStreamingDetokenizer, which the original streams
    through for this tokenizer family. Pieces are emitted per token, except that
    incomplete UTF-8 and a lone space wait for the next token."""

    _space_matches = (".", "?", "!", ",", "n't", "'m", "'s", "'ve", "'re")

    def __init__(self, tokenizer):
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces
        vocab = tokenizer.get_vocab()
        self.tokenmap = [None] * len(vocab)
        for value, index in vocab.items():
            if index < len(self.tokenmap):
                self.tokenmap[index] = value
        self.byte_decoder = _byte_decoder()
        self.offset, self.unflushed, self.text = 0, "", ""

    def _decode(self, sequence):
        data = bytearray()
        for c in sequence:
            value = self.byte_decoder.get(c, False)
            if value:
                data.append(value)
            else:
                data.extend(bytes(c, "utf-8"))
        return data.decode("utf-8", "replace")

    def _trim(self, text):
        if not text or text[0] != " ":
            return text
        if not self.text:
            return text[1:]
        if self.clean_spaces and text[1:].startswith(self._space_matches):
            return text[1:]
        return text

    def add(self, token):
        value = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self.unflushed += value
        text = self._decode(self.unflushed)
        if not text.endswith("�") and not (
            len(value) == 1 and self.byte_decoder.get(value[0]) == 32
        ):
            self.text += self._trim(text)
            self.unflushed = ""

    def finalize(self):
        text = bytearray(self.byte_decoder[c] for c in self.unflushed).decode(
            "utf-8", "replace"
        )
        self.text += self._trim(text)
        self.unflushed = ""

    def segment(self):
        piece, self.offset = self.text[self.offset :], len(self.text)
        return piece


def _byte_decoder():
    decoder, n = {}, 0
    limits = [0, ord("!"), ord("~") + 1, ord("¡"), ord("¬") + 1, ord("®"), ord("ÿ") + 1]
    for i, (start, stop) in enumerate(zip(limits, limits[1:], strict=False)):
        for b in range(start, stop):
            if i % 2 == 0:
                decoder[chr(2**8 + n)] = b
                n += 1
            else:
                decoder[chr(b)] = b
    return decoder


def truncate(tokens, tokenizer, *, stopped, max_tokens, lines=True):
    """The text the original returns for a generation of these tokens.

    Args:
      tokens: generated token ids, without the end-of-sequence token.
      tokenizer: the checkpoint's Hugging Face tokenizer.
      stopped: True if generation ended at end-of-sequence, False at the cap.
      max_tokens: the cap generation ran under.
      lines: apply the repeated-line rule (`generate_response`), not only the
        token rule (`cycles_mlx.sample`).

    Returns:
      (text, cut): the joined pieces, unstripped, and whether a rule fired.
    """
    detokenizer = _Detokenizer(tokenizer)
    pieces, line, seen, history = [], [], {}, []

    def emit(piece, token):
        # generate_response's order: keep the piece, check lines, then tokens.
        pieces.append(piece)
        line.append(piece)
        if lines and "\n" in piece:
            text = "".join(line)
            line.clear()
            if text:
                if seen.get(text, 0) > 1:
                    return True
                seen[text] = seen.get(text, 0) + 1
        history.append(token)
        return token_loop(history)

    # stream_generate: one piece per token until end-of-sequence or the cap
    # (the cap's own token is flushed into the final piece), then a final piece
    # carrying the last token seen -- the end-of-sequence id when it stopped.
    last = None
    for n, token in enumerate(tokens[:max_tokens]):
        detokenizer.add(token)
        last = token
        if n + 1 == max_tokens:
            break
        if emit(detokenizer.segment(), token):
            return "".join(pieces), True
    detokenizer.finalize()
    if stopped:
        last = tokenizer.eos_token_id
    if emit(detokenizer.segment(), last):
        return "".join(pieces), True
    return "".join(pieces), False


def generated_text(details, content):
    """The generation as decoded, from a runtime's reasoning and answer fields.

    The chat template prefills `<think>\\n`, and a runtime that parses reasoning
    returns that newline at the head of the thought; it was prompt, not output.
    """
    reasoning = details.get("reasoning") or ""
    if not reasoning:
        return content or ""
    return reasoning.removeprefix("\n") + "</think>" + (content or "")


async def complete_as_original(
    runtime, tokenizer, messages, *, lines, details, **options
):
    """`runtime.complete`, cut where the original would have stopped decoding.

    `details` is rewritten to describe the cut text, so a thought that loops is
    unclosed here exactly as it is in the original. Tokens come from encoding
    the returned text, which reproduces the generated ids for ordinary output;
    a runtime that returned its ids directly would remove that assumption.
    """
    from .thinking import completion_details

    content = await runtime.complete(messages, details=details, **options)
    if tokenizer is None:  # a test stand-in without a checkpoint
        return content
    tokens = tokenizer.encode(
        generated_text(details, content), add_special_tokens=False
    )
    text, cut = truncate(
        tokens,
        tokenizer,
        stopped=details.get("finish_reason") != "length",
        max_tokens=options.get("max_tokens") or runtime.max_tokens,
        lines=lines,
    )
    if not cut:
        return content
    reasoning, closed, answer = text.partition("</think>")
    if closed:
        details.update(
            completion_details(
                {"content": answer, "reasoning_content": reasoning},
                finish_reason="stop",
            ),
            loop_cut=True,
        )
        return details["content"]
    details.update(
        content="",
        reasoning=reasoning,
        complete=False,
        finish_reason="loop",
        loop_cut=True,
    )
    return ""
