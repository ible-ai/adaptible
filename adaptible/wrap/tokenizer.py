"""Recover an existing GGUF tokenizer faithfully, including added tokens."""

from pathlib import Path


def restore_gguf_tokens(tokenizer, path):
    """Restore metadata omitted by Transformers' Qwen GGUF converter.

    CONTROL and USER_DEFINED pieces must be recognized atomically even when
    their spelling can also be split by BPE. These are existing vocabulary IDs,
    not new tokens requiring embedding resizing or a tokenizer download.
    """
    import gguf
    from tokenizers import AddedToken

    reader = gguf.GGUFReader(str(path))
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    types = reader.fields["tokenizer.ggml.token_type"].contents()
    if len(tokens) != len(types):
        raise ValueError(
            "GGUF token vocabulary and token types have different lengths."
        )
    original_size = len(tokenizer)
    for token_type, special in ((3, True), (4, False)):
        entries = [
            (index, token)
            for index, (token, kind) in enumerate(zip(tokens, types, strict=False))
            if kind == token_type
        ]
        if any(
            tokenizer.convert_tokens_to_ids(token) != index for index, token in entries
        ):
            raise ValueError(
                "Imported tokenizer vocabulary does not match GGUF token IDs."
            )
        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=False, special=special)
                for _, token in entries
            ],
            special_tokens=special,
        )
    if len(tokenizer) != original_size:
        raise ValueError("Restoring GGUF tokens must not enlarge the model vocabulary.")
    for role, key in (
        ("eos", "eos"),
        ("bos", "bos"),
        ("pad", "padding"),
        ("unk", "unknown"),
    ):
        field = reader.fields.get(f"tokenizer.ggml.{key}_token_id")
        if field is not None:
            index = int(field.contents())
            if not 0 <= index < len(tokens):
                raise ValueError(f"GGUF {role} token ID is outside its vocabulary.")
            setattr(tokenizer, f"{role}_token", tokens[index])
    for option in ("add_bos_token", "add_eos_token"):
        field = reader.fields.get(f"tokenizer.ggml.{option}")
        if field is not None:
            setattr(tokenizer, option, bool(field.contents()))
    return tokenizer


def load_tokenizer(source):
    """Load from an existing checkpoint only; never resolve a second checkpoint."""
    from transformers import AutoTokenizer

    source = Path(source)
    if source.is_dir():
        return AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source.parent, gguf_file=source.name, local_files_only=True
    )
    return restore_gguf_tokens(tokenizer, source)
