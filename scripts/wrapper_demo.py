"""Opt-in live wrapper integration demo, using an already installed GGUF.

Run from an editable checkout with the wrap extra installed:
    python -m scripts.wrapper_demo ollama qwen2.5:0.5b
    python -m scripts.wrapper_demo llama-cpp /path/to/model.gguf
    python -m scripts.wrapper_demo lm-studio /path/to/model.gguf
    python -m scripts.wrapper_demo vllm /path/to/local-hf-checkpoint

No model is pulled. Each invocation owns a fresh state directory and wrapper
process. Logs, adapters, and a JSON report stay under ignored outputs/.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

import httpx

QUESTION = "What is the largest city in Morocco? Answer in one sentence."
REFERENCE = "The largest city in Morocco is Casablanca."
EXPECTED = "Casablanca"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("service", choices=("ollama", "llama-cpp", "lm-studio", "vllm"))
    p.add_argument(
        "model",
        help="Installed Ollama name, local GGUF, or existing HF directory (vLLM)",
    )
    p.add_argument(
        "--upstream", help="Ollama or LM Studio URL; uses the app's default port"
    )
    p.add_argument("--llama-server", help="Path to llama-server, otherwise use PATH")
    p.add_argument("--lms", help="Path to the LM Studio CLI, otherwise use PATH")
    p.add_argument("--vllm-server", help="Path to the vllm CLI, otherwise use PATH")
    # Budgets follow the mode: reasoning does not fit in a short answer's
    # allowance, and a truncated turn cannot even be flagged.
    p.add_argument("--context-size", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument(
        "--web-search",
        action="store_true",
        help="Exercise automatic live web references instead of the fixed offline document fixture",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wrapper-demo"))
    p.add_argument(
        "--timeout", type=float, default=300, help="Seconds per request/startup"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--thinking",
        action="store_true",
        help="Ask, flag and re-ask with reasoning enabled",
    )
    mode.add_argument(
        "--nonthinking",
        action="store_true",
        help="Ask, flag and re-ask with reasoning disabled",
    )
    p.add_argument("--question", default=QUESTION)
    p.add_argument("--reference", default=REFERENCE)
    p.add_argument(
        "--expected", default=EXPECTED, help="Expected first-sentence answer marker"
    )
    return p


THINKING_BUDGET = dict(max_tokens=2048, context_size=8192)
ANSWER_BUDGET = dict(max_tokens=160, context_size=4096)


def apply_budgets(args):
    """Fill unset token budgets from the selected generation mode.

    A reasoning turn does not fit in a short answer's allowance; truncated
    output is recorded as incomplete and cannot be flagged, so a demo would
    fail before it ever reached a review.
    """
    defaults = THINKING_BUDGET if getattr(args, "thinking", False) else ANSWER_BUDGET
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def has_answer(text, expected):
    # An engineering assertion only, not a factual judge of the full response.
    final = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in final:
        return False
    first = re.split(r"(?<=[.!?])\s+|\n", final, maxsplit=1)[0]
    return bool(re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", first, re.I))


class WrapperProcess:
    def __init__(self, args, directory):
        # Resolve here too: the command is built from these, and callers that
        # construct a process directly must not emit an unset budget.
        args = apply_budgets(args)
        self.directory = directory
        self.process = None
        self.log = None
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.command = [
            sys.executable,
            "-m",
            "adaptible",
            "wrap",
            args.service,
            args.model,
            "--port",
            str(port),
            "--state-dir",
            str(directory / "state"),
            "--idle-seconds",
            "1",
            "--max-tokens",
            str(getattr(args, "max_tokens", None) or 160),
            "--context-size",
            str(args.context_size),
        ]
        if not args.web_search:
            self.command += [
                "--documents",
                str(directory / "documents.json"),
                "--no-web-search",
            ]
        for flag, value in (
            ("--upstream", args.upstream),
            ("--llama-server", args.llama_server),
            ("--lms", args.lms),
            ("--vllm-server", args.vllm_server),
        ):
            if value:
                self.command += [flag, value]
        self.env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # An empty cache demonstrates that no other HF checkpoint is needed.
            "HF_HOME": str(directory / "hf-cache"),
        }

    def start(self, timeout):
        self.log = (self.directory / "wrapper.log").open("ab")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        with httpx.Client(base_url=self.url, timeout=1, trust_env=False) as client:
            while time.monotonic() < deadline:
                require(self.process.poll() is None, "Wrapper exited; see wrapper.log")
                try:
                    response = client.get("/status")
                    if (
                        response.status_code == 200
                        and response.json()["status"] == "up"
                    ):
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        raise TimeoutError("Wrapper startup timed out; see wrapper.log")

    def stop(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                # Reap any child left behind after failure, only in our own session.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        finally:
            self.process = None
            if self.log:
                self.log.close()
                self.log = None


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Preserve native compatibility errors in the saved harness report.
        # Do not include request headers or request bodies in diagnostics.
        body = response.text
        excerpt = body[:2000]
        if len(body) > 2000:
            excerpt += " [truncated]"
        raise httpx.HTTPStatusError(
            f"{exc}\nResponse body: {excerpt}",
            request=exc.request,
            response=exc.response,
        ) from exc
    return response


def split_thinking(content):
    """Separate one completed thought block; reject malformed/repeated framing."""
    content = (content or "").strip()
    if "<think>" not in content and "</think>" not in content:
        return content, "", True
    if content.count("</think>") != 1 or content.count("<think>") > 1:
        return "", "", False
    if "<think>" in content and not content.startswith("<think>"):
        return "", "", False
    thought, final = content.split("</think>", 1)
    thought = thought.removeprefix("<think>").strip()
    if "<think>" in thought or "<think>" in final:
        return "", "", False
    return final.strip(), thought, True


def response_trace(content, reasoning="", *, finish_reason=None, usage=None, done=True):
    """Keep generation evidence separate from the external answer oracle."""
    content = content or ""
    reasoning = reasoning or ""
    final, inline, framing_valid = split_thinking(content)
    details = (usage or {}).get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens") or 0
    return dict(
        final=final,
        framing_valid=framing_valid,
        reasoning=reasoning or inline,
        reasoning_chars=len((reasoning or inline).strip()),
        reasoning_tokens=reasoning_tokens,
        thinking_observed=bool((reasoning or inline).strip()),
        finish_reason=finish_reason,
        usage=usage or {},
        complete=bool(
            done
            and finish_reason in ("stop", "eos", "end_turn")
            and framing_valid
            and final
        ),
    )


def adaptation_headers(headers):
    """Capture deployment identity without mixing it with generated text."""
    adapter = headers.get("X-Adaptible-Adapter")
    if adapter is None:
        return None
    scope = headers.get("X-Adaptible-Scope")
    return dict(adapter=adapter, scope=int(scope) if scope is not None else None)


def record_adaptation(trace, row):
    """Cross-check response headers against the durable routing decision."""
    recorded = row.get("response_details", {}).get("adaptation")
    headers = trace.get("adaptation_headers")
    require(isinstance(recorded, dict), "Response lacks durable adaptation metadata")
    require(isinstance(headers, dict), "Response lacks adaptation headers")
    require(
        headers == {key: recorded.get(key) for key in ("adapter", "scope")},
        "Adaptation headers disagree with durable history",
    )
    trace["adaptation"] = recorded


def chat(
    client,
    model,
    question,
    *,
    nonthinking=False,
    thinking=False,
    max_tokens=160,
    trace=None,
    native_ollama=False,
):
    body = dict(
        model=model,
        messages=[dict(role="user", content=question)],
        stream=False,
        temperature=0,
        seed=0,
        max_tokens=max_tokens,
    )
    require(not (thinking and nonthinking), "Choose one thinking mode")
    if thinking:
        body.update(
            reasoning_effort="high", chat_template_kwargs=dict(enable_thinking=True)
        )
    if nonthinking:
        body.update(
            reasoning_effort="none", chat_template_kwargs=dict(enable_thinking=False)
        )
    if native_ollama:
        body.pop("chat_template_kwargs", None)
    response = request(
        client,
        "POST",
        "/v1/chat/completions",
        json=body,
    )
    data = response.json()
    require(data["model"] == model, "Public model name changed")
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    if trace is not None:
        trace.update(
            response_trace(
                content,
                message.get("reasoning_content") or message.get("reasoning") or "",
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage"),
            )
        )
        trace["adaptation_headers"] = adaptation_headers(response.headers)
    return content, int(response.headers["X-Interaction-Idx"])


def stream_chat(
    client,
    model,
    question,
    path,
    *,
    nonthinking=False,
    thinking=False,
    max_tokens=160,
    trace=None,
    native_ollama=False,
):
    require(not (thinking and nonthinking), "Choose one thinking mode")
    native = path.startswith("/api/")
    body = dict(model=model, stream=True)
    if path == "/api/generate":
        body["prompt"] = question
    else:
        body["messages"] = [dict(role="user", content=question)]
    if native:
        body["options"] = dict(temperature=0, seed=0, num_predict=max_tokens)
        if thinking or nonthinking:
            body["think"] = thinking
    else:
        body.update(temperature=0, seed=0, max_tokens=max_tokens)
        if thinking:
            body.update(
                reasoning_effort="high",
                chat_template_kwargs=dict(enable_thinking=True),
            )
        if nonthinking:
            body.update(
                reasoning_effort="none",
                chat_template_kwargs=dict(enable_thinking=False),
            )
        if native_ollama:
            body.pop("chat_template_kwargs", None)
    pieces, thoughts, done = [], [], False
    finish_reason, usage = None, {}
    with client.stream("POST", path, json=body) as response:
        response.raise_for_status()
        idx = int(response.headers["X-Interaction-Idx"])
        routing_headers = adaptation_headers(response.headers)
        for line in response.iter_lines():
            if not line:
                continue
            if line == "data: [DONE]":
                done = True
                continue
            data = json.loads(line if native else line.removeprefix("data: "))
            require(not data.get("error"), str(data.get("error")))
            require(data["model"] == model, "Stream exposed a private model name")
            if native:
                pieces.append(
                    data.get("response", data.get("message", {}).get("content", ""))
                )
                thoughts.append(
                    data.get("thinking")
                    or data.get("message", {}).get("thinking")
                    or ""
                )
                done |= data.get("done", False)
                if data.get("done"):
                    finish_reason = data.get("done_reason")
                    usage = {
                        key: data[key]
                        for key in ("eval_count", "prompt_eval_count")
                        if key in data
                    }
            else:
                choices = data.get("choices") or []
                usage = data.get("usage") or usage
                if choices:
                    delta = choices[0].get("delta", {})
                    pieces.append(delta.get("content") or "")
                    thoughts.append(
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    finish_reason = choices[0].get("finish_reason") or finish_reason
    text = "".join(pieces)
    if trace is not None:
        trace.update(
            response_trace(
                text,
                "".join(thoughts),
                finish_reason=finish_reason,
                usage=usage,
                done=done,
            )
        )
        trace["adaptation_headers"] = routing_headers
    require(done and text.strip(), f"Incomplete or empty stream: {path}")
    rows = request(client, "GET", "/history").json()["history"]
    row = next(r for r in rows if r["interaction_idx"] == idx)
    require(
        row["status"] == "new" and row["response"] == split_thinking(text)[0],
        "Stream final-answer history mismatch",
    )
    if trace is not None and routing_headers is not None:
        record_adaptation(trace, row)
    return text


def cleanup_tags(args, state):
    """Remove only tags derived from this invocation's fresh adapter directories."""
    if args.service != "ollama":
        return []
    prefix = (
        "adaptible-"
        + hashlib.sha256(str(state.resolve()).encode()).hexdigest()[:12]
        + "-"
    )
    candidates = {
        prefix + path.name + ":latest" for path in (state / "adapters").glob("*")
    }
    if not candidates:
        return []
    removed = []
    with httpx.Client(
        base_url=args.upstream or "http://127.0.0.1:11434", timeout=30, trust_env=False
    ) as client:
        tags = request(client, "GET", "/api/tags").json()["models"]
        for model in tags:
            if model["name"] in candidates:
                # Deleting a tag alone can leave its runner resident in Ollama.
                request(
                    client,
                    "POST",
                    "/api/generate",
                    json=dict(
                        model=model["name"], prompt="", stream=False, keep_alive=0
                    ),
                )
                for _ in range(60):
                    loaded = request(client, "GET", "/api/ps").json()["models"]
                    if all(m["name"] != model["name"] for m in loaded):
                        break
                    time.sleep(0.25)
                else:
                    raise RuntimeError("Ollama did not unload the private test model")
                request(client, "DELETE", "/api/delete", json={"model": model["name"]})
                removed.append(model["name"])
        # The demo has completed its restart proof and stopped the wrapper.
        # Its small adapter can recreate derived weights; retain only that.
        from adaptible.wrap.ollama import remove_fused_file

        for name in candidates:
            remove_fused_file(state, name)
    return removed


def verify_training(state, report):
    """Check real worker output and unchanged, reused checkpoint bytes."""
    from adaptible.wrap.model_source import fingerprint_base

    with closing(sqlite3.connect(state / "history.sqlite3")) as db:
        meta = {
            key: json.loads(value)
            for key, value in db.execute("SELECT key, value FROM meta")
        }
    adapter = Path(meta["accepted"]["directory"])
    report["training"] = json.loads((adapter / "stats.json").read_text())
    require(report["training"]["steps"] > 0, "No training steps ran")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        require((adapter / "adapter" / name).stat().st_size > 0, f"Missing PEFT {name}")
    source_record = adapter / "source.json"
    metadata = None
    if source_record.exists():
        metadata = json.loads(source_record.read_text())
        require(metadata.get("schema_version") == 1, "Unknown source metadata version")
        source = Path(metadata["path"]).resolve(strict=True)
        require(
            not source.is_relative_to(adapter.resolve()),
            "Trainer copied a base checkpoint instead of reusing it",
        )
        require(
            metadata["kind"] == ("hf" if source.is_dir() else "gguf"),
            "Source checkpoint format does not match metadata",
        )
        job_path = adapter / "job.json"
        if job_path.exists():
            job = json.loads(job_path.read_text())
            require(
                Path(job["blob"]).resolve() == source, "Training job source changed"
            )
        for name in ("model", "model.gguf"):
            candidate = adapter / name
            require(
                not candidate.exists() or candidate.is_symlink(),
                "Trainer copied a base checkpoint instead of reusing it",
            )
    else:
        # Older audit artifacts used a symlink solely to record source identity.
        gguf = adapter / "model.gguf"
        link = gguf if gguf.is_symlink() else adapter / "model"
        require(
            link.is_symlink(), "Missing original source metadata or legacy source link"
        )
        source = link.resolve(strict=True)
    if source.is_file():
        require(
            (adapter / "adapter.gguf").stat().st_size > 0, "Missing exported adapter"
        )
    report["base_sha256"] = fingerprint_base(source)
    require(
        report["base_sha256"] == meta["identity"].rsplit(":", 1)[-1],
        "Base weights changed",
    )
    if metadata:
        require(
            report["base_sha256"] == metadata["sha256"],
            "Base weights changed since training",
        )
    report["base_file"] = str(source)


def cleanup_lm_studio(args, state):
    """Remove this demo's tracked model links after its restart check."""
    manifest = state / "lmstudio-imports.json"
    if args.service != "lm-studio" or not manifest.exists():
        return []
    from adaptible.wrap.lmstudio import LMStudio

    async def cleanup():
        runtime = LMStudio(
            args.model,
            state,
            url=args.upstream or "http://127.0.0.1:1234",
            executable=args.lms,
        )
        keys = list(runtime.imports)
        try:
            await runtime.cleanup_imports()
        finally:
            await runtime.close()
        return keys

    return asyncio.run(cleanup())


def run_demo(args):
    args = apply_budgets(args)
    require(args.timeout > 0, "--timeout must be positive")
    require(
        all(value.strip() for value in (args.question, args.reference, args.expected)),
        "Question, reference, and expected marker must be nonempty",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=args.service + "-", dir=args.output_dir)
    ).resolve()
    state = directory / "state"
    if not args.web_search:
        (directory / "documents.json").write_text(
            json.dumps({args.question: args.reference})
        )
    wrapper = WrapperProcess(args, directory)
    report = dict(
        service=args.service,
        model=args.model,
        directory=str(directory),
        integration_status="failed",
        question=args.question,
        reference=None if args.web_search else args.reference,
        reference_mode="web" if args.web_search else "documents",
        expected=args.expected,
        quality_scope="First-sentence marker only; inspect full answers for other errors.",
    )
    started = time.monotonic()
    try:
        wrapper.start(args.timeout)
        with httpx.Client(
            base_url=wrapper.url, timeout=args.timeout, trust_env=False
        ) as client:
            models = request(client, "GET", "/v1/models").json()["data"]
            require(len(models) == 1, "Expected exactly one wrapped model")
            model = models[0]["id"]
            report["before"], idx = chat(
                client,
                model,
                args.question,
                thinking=args.thinking,
                nonthinking=args.nonthinking,
                max_tokens=args.max_tokens,
            )
            if has_answer(report["before"], args.expected):
                report.update(
                    integration_status="inconclusive",
                    reason="Base already answers this fixture; no repair was exercised. Choose another question/reference/expected triple.",
                )
                return report
            # Feedback has no answer. The model writes the training target from
            # the fixed fixture or, in web mode, automatically discovered sources.
            request(
                client,
                "POST",
                "/feedback",
                json=dict(interaction_idx=idx, thumbs="down"),
            )
            report["review"] = request(client, "GET", "/sync").json()
            if args.web_search:
                rows = request(client, "GET", "/history").json()["history"]
                row = next(r for r in rows if r["interaction_idx"] == idx)
                report["references"] = row["references"]
                require(
                    row["note"] == "" and row["references"]["kind"] == "web",
                    "Automatic lookup was bypassed",
                )
                require(
                    row["references"].get("sources"), "No actual web sources recorded"
                )
            reviews = report["review"]["reviews"]
            require(
                any(
                    r["interaction_idx"] == idx and r["status"] == "kept"
                    for r in reviews
                ),
                f"No adapter accepted: {reviews}",
            )
            report["after"], _ = chat(
                client,
                model,
                args.question,
                thinking=args.thinking,
                nonthinking=args.nonthinking,
                max_tokens=args.max_tokens,
            )
            require(
                has_answer(report["after"], args.expected),
                "Accepted adapter did not produce the expected answer marker",
            )
            history = request(client, "GET", "/history").json()
            wrapper.stop()
            # Verify real training and immutable base, not just a mocked route.
            verify_training(state, report)
            wrapper.start(args.timeout)
            require(
                request(client, "GET", "/history").json() == history,
                "History did not survive restart",
            )
            report["after_restart"], _ = chat(
                client,
                model,
                args.question,
                thinking=args.thinking,
                nonthinking=args.nonthinking,
                max_tokens=args.max_tokens,
            )
            require(
                has_answer(report["after_restart"], args.expected),
                "Accepted adapter did not survive restart",
            )
            report["streams"] = {}
            paths = ["/v1/chat/completions"]
            if args.service == "ollama":
                paths += ["/api/chat", "/api/generate"]
            for path in paths:
                report["streams"][path] = stream_chat(
                    client, model, args.question, path
                )
                require(
                    has_answer(report["streams"][path], args.expected),
                    f"Stream did not use accepted adapter: {path}",
                )
            report["integration_status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            wrapper.stop()
            report["removed_test_tags"] = cleanup_tags(args, state)
            report["removed_test_imports"] = cleanup_lm_studio(args, state)
        except Exception as exc:
            report.update(integration_status="failed", cleanup_error=str(exc))
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    report = run_demo(parser().parse_args(argv))
    print(json.dumps(report, indent=2))
    return {"passed": 0, "failed": 1, "inconclusive": 2}[report["integration_status"]]


if __name__ == "__main__":
    raise SystemExit(main())
