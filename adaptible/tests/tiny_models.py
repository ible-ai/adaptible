"""Build tiny randomly-initialised checkpoints for tests. Never downloads.

These are real Transformers models and real GGUF files, small enough to create
in milliseconds. They prove that loading, LoRA attachment, training, export and
fusion are mechanically correct. They cannot show that anything is *learned*:
a two-layer random model has nothing to learn. Use the native runs for that.
"""

from pathlib import Path

# Architectures the wrapper claims to support, plus ones it currently refuses.
# `supported` mirrors gguf_adapter.SUPPORTED_ARCHITECTURES so a test can assert
# the refusal is a deliberate gate rather than an accident of the loader.
ARCHITECTURES = {
    "qwen2": dict(supported=True, gguf="QWEN2", model="Qwen2ForCausalLM"),
    "qwen3": dict(supported=True, gguf="QWEN3", model="Qwen3ForCausalLM"),
    "llama": dict(supported=True, gguf="LLAMA", model="LlamaForCausalLM"),
    "mistral": dict(supported=True, gguf="LLAMA", model="MistralForCausalLM"),
    "gemma3_text": dict(supported=False, gguf=None, model="Gemma3ForCausalLM"),
}

# Projections the worker adapts. Shared by every architecture here.
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Mirrors how a real GGUF types its vocabulary. CONTROL tokens (3) are special
# and stripped when an answer is decoded; the think delimiters are USER_DEFINED
# (4) and must survive decoding, which is exactly what ``restore_gguf_tokens``
# ensures for a real model. Registering them as special here would silently
# delete every rationale.
_CONTROL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
_USER_DEFINED_TOKENS = ("<think>", "</think>")
_SPECIAL_TOKENS = _CONTROL_TOKENS

# Words the end-to-end fixture needs as single tokens. A real tokenizer spells
# a common proper noun in one or two pieces; a BPE trained on three sentences
# would otherwise split them per byte, leaving the toy model a long exact
# sequence to reproduce.
ATOMIC_WORDS = ("Casablanca", "Paris", "Rabat", "Morocco")

# Learned as ordinary merges rather than reserved, so they decode back.
_LEARNED_ATOMS = _USER_DEFINED_TOKENS + ATOMIC_WORDS

_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _config_name(architecture):
    """``gemma3_text`` -> ``Gemma3TextConfig``; ``qwen3`` -> ``Qwen3Config``."""
    return "".join(part.capitalize() for part in architecture.split("_")) + "Config"


# A byte-level BPE built deterministically rather than trained. Training one on
# a three-sentence corpus leaves which words merge into single tokens up to the
# merge budget, which silently changes what the fixtures can express.
_BACKEND = None


def _merge_chain(word):
    """Left-to-right merges that build ``word``, with their intermediates."""
    pieces, merges = list(word), []
    current = pieces[0]
    intermediates = []
    for piece in pieces[1:]:
        merges.append((current, piece))
        current = current + piece
        intermediates.append(current)
    return intermediates, merges


def _backend():
    """Builds the shared tokenizer once per process."""
    global _BACKEND
    if _BACKEND is None:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from tokenizers.pre_tokenizers import ByteLevel

        vocab, merges = {}, []
        for token in _CONTROL_TOKENS + _USER_DEFINED_TOKENS:
            vocab[token] = len(vocab)
        for character in sorted(ByteLevel.alphabet()):
            vocab.setdefault(character, len(vocab))
        for word in ATOMIC_WORDS:
            intermediates, chain = _merge_chain(word)
            merges.extend(chain)
            for piece in intermediates:
                vocab.setdefault(piece, len(vocab))

        backend = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        _BACKEND = backend
    return _BACKEND


def vocabulary():
    """Every token in ID order."""
    import json

    vocab = json.loads(_backend().to_str())["model"]["vocab"]
    return [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def merges():
    """BPE merges as the space-joined pairs a GGUF stores."""
    import json

    pairs = json.loads(_backend().to_str())["model"]["merges"]
    return [" ".join(pair) if isinstance(pair, list) else pair for pair in pairs]


def token_types():
    """GGUF token types, keyed by token rather than by position."""
    types = []
    for token in vocabulary():
        if token in _CONTROL_TOKENS:
            types.append(3)
        elif token in _USER_DEFINED_TOKENS:
            types.append(4)
        else:
            types.append(1)
    return types


def tiny_config(
    architecture,
    vocab_size=None,
    *,
    hidden_size=32,
    num_hidden_layers=2,
    intermediate_size=None,
):
    """A minimal but structurally real config for one architecture.

    ``head_dim`` is left at each architecture's default so that a GGUF written
    from this config reloads with identical projection shapes. Sizes are
    constructor arguments rather than later assignments: several architectures
    derive per-layer tables (``layer_types``) at construction, and mutating the
    layer count afterwards leaves those tables stale.
    """
    import transformers

    config_class = getattr(transformers, _config_name(architecture))
    return config_class(
        vocab_size=vocab_size or len(vocabulary()),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size or hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


def permute_for_gguf(weights, n_head, n_head_kv=None):
    """Apply llama.cpp's Q/K row interleave, as ``convert_hf_to_gguf`` does.

    Llama-family GGUFs store attention Q/K with rotary pairs interleaved.
    Transformers undoes this on load, so a fixture that skips it is not a
    realistic GGUF and will hide adapter-space bugs.
    """
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


def head_dim(config):
    """Per-head width, falling back to the derived value when unset."""
    return getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )


def tiny_tokenizer():
    """A real fast tokenizer over :func:`vocabulary`, with a chat template."""
    import copy

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=copy.deepcopy(_backend()),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )
    # These are in the vocabulary, but the byte-level pre-tokenizer splits on
    # their punctuation, so without this a prompt tokenizes character by
    # character while a GGUF-derived tokenizer reads it atomically -- training
    # and serving would disagree. Control tokens are special (stripped when an
    # answer is decoded); the think delimiters are not, so a rationale survives.
    # Exactly the split ``restore_gguf_tokens`` makes for a real GGUF.
    from tokenizers import AddedToken

    for tokens, special in ((_CONTROL_TOKENS, True), (_USER_DEFINED_TOKENS, False)):
        tokenizer.add_tokens(
            [AddedToken(token, normalized=False, special=special) for token in tokens],
            special_tokens=special,
        )
    tokenizer.chat_template = _CHAT_TEMPLATE
    return tokenizer


def write_hf_checkpoint(root, architecture="qwen3", seed=0, **sizes):
    """Writes a tiny safetensors checkpoint plus tokenizer. Returns the dir."""
    import torch
    import transformers

    root = Path(root)
    config = tiny_config(architecture, **sizes)
    model_class = getattr(transformers, ARCHITECTURES[architecture]["model"])
    torch.manual_seed(seed)
    model_class(config).save_pretrained(root, safe_serialization=True)
    tiny_tokenizer().save_pretrained(root)
    return root


def write_gguf_checkpoint(path, architecture="qwen3", seed=0, source=None, **sizes):
    """Writes a tiny but complete GGUF model, loadable by Transformers.

    Args:
        path: Destination ``.gguf``.
        architecture: Fixture architecture key.
        seed: Seed for a freshly initialised model.
        source: An existing checkpoint directory to convert instead of building
            a random one, so a trained fixture can be served as GGUF.
        **sizes: Forwarded to :func:`tiny_config` when building a new model.
    """
    import gguf
    import torch
    import transformers

    path = Path(path)
    if source is not None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            Path(source), local_files_only=True, dtype=torch.float32
        )
        config = model.config
    else:
        config = tiny_config(architecture, **sizes)
        torch.manual_seed(seed)
        model = getattr(transformers, ARCHITECTURES[architecture]["model"])(config)
    state = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}

    arch = ARCHITECTURES[architecture]["gguf"]
    if arch is None:
        raise ValueError(f"No GGUF architecture mapping for {architecture!r}.")
    writer = gguf.GGUFWriter(str(path), arch.lower())
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_key_length(head_dim(config))
    writer.add_value_length(head_dim(config))
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(config.vocab_size)

    tokens = vocabulary()
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types())
    writer.add_token_merges(merges())
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(False)
    writer.add_chat_template(_CHAT_TEMPLATE)

    name_map = gguf.get_tensor_name_map(
        getattr(gguf.MODEL_ARCH, arch), config.num_hidden_layers
    )
    written = 0
    for key, value in state.items():
        stem = key[: -len(".weight")] if key.endswith(".weight") else key
        mapped = name_map.get_name(stem)
        if mapped is None:
            continue
        if arch == "LLAMA" and ".attn_q" in mapped:
            value = permute_for_gguf(value, config.num_attention_heads)
        elif arch == "LLAMA" and ".attn_k" in mapped:
            value = permute_for_gguf(
                value, config.num_attention_heads, config.num_key_value_heads
            )
        writer.add_tensor(mapped + ".weight", value)
        written += 1
    if not written:
        raise ValueError(f"No tensors mapped for {architecture!r}.")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def training_job(blob, out, *, target="Casablanca", question=None, **extra):
    """A minimal job dict of the shape ``repair.Trainer`` writes for the worker.

    Any keyword overrides or adds a field, so a test can supply
    ``training_options``, ``examples``, ``resume_from`` or ``previous``.
    """
    job = dict(
        blob=str(blob),
        out=str(out),
        messages=[
            {
                "role": "user",
                "content": question or "What is the largest city in Morocco?",
            }
        ],
        target=target,
        previous=None,
        examples=[],
        training_options={},
    )
    job.update(extra)
    return job


# A tiny model that answers confidently and is wrong about exactly one fact --
# the situation the wrapper exists to fix. Randomly initialised weights are not
# enough: with no language structure, a low-rank adapter has nothing to steer,
# and the repair loop cannot demonstrate anything.
WRONG_ANSWER = "Rabat."

FIXTURE_PAIRS = (
    ("What is the largest city in Morocco?", WRONG_ANSWER),
    ("Which city is the largest in Morocco?", WRONG_ANSWER),
    ("Name Morocco's largest city.", WRONG_ANSWER),
    ("What is Morocco's biggest city?", WRONG_ANSWER),
    ("What is the capital of France? Reply with only the name.", "Paris."),
    ("What is 12 times 12? Reply with only the number.", "144."),
    ("How many days are in a week? Reply with only the number.", "7."),
)


def write_trained_checkpoint(
    root,
    architecture="qwen2",
    *,
    pairs=FIXTURE_PAIRS,
    steps=400,
    learning_rate=3e-3,
    hidden_size=64,
    num_hidden_layers=4,
    seed=0,
):
    """Writes a checkpoint briefly fine-tuned to answer ``pairs`` verbatim.

    Full fine-tuning here is deliberate: the fixture is standing in for a
    pretrained base, not demonstrating the wrapper's LoRA training. Takes a
    couple of seconds and downloads nothing.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = write_hf_checkpoint(
        root,
        architecture,
        seed=seed,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, dtype=torch.float32
    )

    encoded = []
    for question, answer in pairs:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded.append(
            (
                tokenizer(prompt, add_special_tokens=False).input_ids,
                tokenizer(answer, add_special_tokens=False).input_ids
                + [tokenizer.eos_token_id],
            )
        )
    width = max(len(prompt) + len(answer) for prompt, answer in encoded)
    ids = torch.full((len(encoded), width), tokenizer.pad_token_id)
    labels = torch.full((len(encoded), width), -100)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, (prompt, answer) in enumerate(encoded):
        whole = prompt + answer
        ids[row, : len(whole)] = torch.tensor(whole)
        attention[row, : len(whole)] = 1
        labels[row, len(prompt) : len(whole)] = torch.tensor(answer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(steps):
        loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.save_pretrained(root, safe_serialization=True)
    return root


# The same fixture with reasoning, for the thinking codepath. Qwen3 serves a
# completed thought and a short final answer; a repair has to keep both.
def _thought(reasoning, answer):
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


THINKING_PAIRS = (
    (
        "What is the largest city in Morocco?",
        _thought("Morocco. Largest city.", WRONG_ANSWER),
    ),
    (
        "Which city is the largest in Morocco?",
        _thought("Morocco. Largest city.", WRONG_ANSWER),
    ),
    ("Name Morocco's largest city.", _thought("Morocco. Largest city.", WRONG_ANSWER)),
    (
        "What is Morocco's biggest city?",
        _thought("Morocco. Largest city.", WRONG_ANSWER),
    ),
    (
        "What is the capital of France? Reply with only the name.",
        _thought("France. Capital city.", "Paris."),
    ),
    (
        "What is 12 times 12? Reply with only the number.",
        _thought("12 times 12.", "144."),
    ),
    (
        "How many days are in a week? Reply with only the number.",
        _thought("Days in a week.", "7."),
    ),
)
