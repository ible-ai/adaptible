"""Exercise the installed CLI, real runtime, and automatic web feedback path.

    python -m scripts.wrapper_smoke ollama qwen2.5:0.5b
    python -m scripts.wrapper_smoke llama-cpp /path/to/model.gguf
    python -m scripts.wrapper_smoke lm-studio /path/to/model.gguf
    python -m scripts.wrapper_smoke vllm --tiny-vllm --vllm-server /path/to/vllm

This is a transport/lifecycle check, not a claim of successful factual learning.
Unlike wrapper_demo, it accepts an explicitly reported skipped/rejected review.
It requires actual retrieved web sources, and fails on runtime/review errors.
No model is downloaded. State and reports remain under ignored outputs/.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import httpx

from scripts.wrapper_demo import (
    WrapperProcess,
    chat,
    cleanup_lm_studio,
    cleanup_tags,
    request,
    require,
    stream_chat,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("service", choices=("ollama", "llama-cpp", "lm-studio", "vllm"))
    p.add_argument("model", nargs="?")
    p.add_argument("--tiny-vllm", action="store_true")
    p.add_argument("--upstream")
    p.add_argument("--llama-server")
    p.add_argument("--lms")
    p.add_argument("--vllm-server")
    p.add_argument("--context-size", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--question", default="What is the capital of Australia?")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wrapper-smoke"))
    return p


def tiny_model(directory, context_size):
    """Create real native Qwen2 weights/tokenizer, deliberately without knowledge."""
    import torch
    from tokenizers import pre_tokenizers
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

    vocab = {
        token: i
        for i, token in enumerate(
            ["[UNK]", "[EOS]", *sorted(pre_tokenizers.ByteLevel.alphabet())]
        )
    }
    tokenizer = Qwen2Tokenizer(
        vocab=vocab, merges=[], unk_token="[UNK]", pad_token="[EOS]", eos_token="[EOS]"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=context_size,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)


def tiny_adapter_check(args, directory, wrapper, client, model):
    """Test actual CLI restoration from seeded state, not semantic acceptance."""
    from adaptible._src.wrap.model_source import fingerprint_base
    from adaptible._src.wrap.repair import Trainer
    from adaptible._src.wrap.store import Store

    messages = [dict(role="user", content="Question")]

    def probabilities():
        response = request(
            client,
            "POST",
            "/v1/chat/completions",
            json=dict(
                model=model,
                messages=messages,
                temperature=0,
                seed=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
            ),
        )
        return response.json()["choices"][0]["logprobs"]["content"][0]

    source = Path(args.model)
    before_digest = fingerprint_base(source)
    baseline = probabilities()
    wrapper.stop()
    state = directory / "state"
    candidate = state / "adapters" / "seeded-cli-lifecycle"
    stats = asyncio.run(Trainer().train(source, messages, "Answer", candidate))
    require(stats["steps"] > 0, "No real adapter training ran")
    with sqlite3.connect(state / "history.sqlite3") as db:
        identity = json.loads(
            db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0]
        )
    store = Store(state, identity)
    try:
        require(
            store.get("accepted") is None,
            "Tiny random model unexpectedly accepted a semantic repair",
        )
        store.set("accepted", dict(directory=str(candidate)))
    finally:
        store.close()
    wrapper.start(args.timeout)
    adapted = probabilities()
    require(adapted != baseline, "CLI did not apply the actual trained adapter")
    wrapper.stop()
    wrapper.start(args.timeout)
    require(probabilities() == adapted, "CLI adapter changed across restart")
    wrapper.stop()
    store = Store(state, identity)
    try:
        store.set("accepted", None)
    finally:
        store.close()
    wrapper.start(args.timeout)
    require(probabilities() == baseline, "CLI rollback did not restore frozen base")
    require(fingerprint_base(source) == before_digest, "Base weights changed")
    return dict(
        status="passed",
        seeded_state=True,
        semantic_acceptance=False,
        training=stats,
        base_sha256=before_digest,
        changed_logits=True,
        restart=True,
        rollback=True,
    )


def run_smoke(args):
    require(args.timeout > 0 and args.question.strip(), "Invalid timeout/question")
    require(
        (args.tiny_vllm and args.service == "vllm" and args.model is None)
        or (not args.tiny_vllm and bool(args.model)),
        "Supply an installed model, or vllm --tiny-vllm without a model argument",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=args.service + "-", dir=args.output_dir)
    ).resolve()
    if args.tiny_vllm:
        args.model = str(directory / "tiny-qwen2")
        tiny_model(Path(args.model), args.context_size)
    args.web_search = True
    wrapper = WrapperProcess(args, directory)
    report = dict(
        service=args.service,
        model=args.model,
        directory=str(directory),
        smoke_status="failed",
        scope="Actual CLI, HTTP chat/stream/history, automatic references, and restart; not repair efficacy",
        random_model=args.tiny_vllm,
        command=wrapper.command,
    )
    started = time.monotonic()
    try:
        wrapper.start(args.timeout)
        with httpx.Client(
            base_url=wrapper.url, timeout=args.timeout, trust_env=False
        ) as client:
            report["status"] = request(client, "GET", "/status").json()
            models = request(client, "GET", "/v1/models").json()["data"]
            require(len(models) == 1, "Expected one wrapped model")
            model = models[0]["id"]
            report["response"], idx = chat(client, model, args.question)
            report["streams"] = {}
            paths = ["/v1/chat/completions"]
            if args.service == "ollama":
                paths += ["/api/chat", "/api/generate"]
            for path in paths:
                report["streams"][path] = stream_chat(
                    client, model, args.question, path
                )
            request(
                client,
                "POST",
                "/feedback",
                json=dict(interaction_idx=idx, thumbs="down"),
            )
            report["review"] = request(client, "GET", "/sync").json()
            history = request(client, "GET", "/history").json()
            row = next(r for r in history["history"] if r["id"] == idx)
            report["review_outcome"] = {
                key: row[key] for key in ("status", "reason", "references")
            }
            require(
                row["note"] == "" and row["references"].get("kind") == "web",
                "Default automatic lookup was bypassed",
            )
            require(
                bool(row["references"].get("sources")),
                "No real web sources were retrieved",
            )
            require(
                not row["references"].get("error"), "Automatic lookup reported an error"
            )
            require(
                row["status"] in {"kept", "rejected", "unchanged", "skipped"},
                f"Review failed: {row['reason']}",
            )
            wrapper.stop()
            wrapper.start(args.timeout)
            require(
                request(client, "GET", "/history").json() == history,
                "History did not survive restart",
            )
            report["after_restart"], _ = chat(client, model, args.question)
            if args.tiny_vllm:
                report["adapter_lifecycle"] = tiny_adapter_check(
                    args, directory, wrapper, client, model
                )
            report["smoke_status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            wrapper.stop()
            report["removed_test_tags"] = cleanup_tags(args, directory / "state")
            report["removed_test_imports"] = cleanup_lm_studio(
                args, directory / "state"
            )
        except Exception as exc:
            report.update(smoke_status="failed", cleanup_error=str(exc))
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    report = run_smoke(parser().parse_args())
    print(json.dumps(report, indent=2))
    return 0 if report["smoke_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
