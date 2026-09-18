"""Measure repeated automatic factual repairs through a real wrapper process.

Run with an existing model, for example:
    python -m scripts.wrapper_recurrent ollama MODEL
    python -m scripts.wrapper_recurrent llama-cpp /path/model.gguf
    python -m scripts.wrapper_recurrent lm-studio /path/model.gguf
    python -m scripts.wrapper_recurrent vllm /path/local-hf-model

The fixed evaluation answers below are assertions only: feedback never supplies
them, a reference file, or a hint. Two independent wordings per item are never
flagged or trained on. Every attempted review and complete before/after answer
matrix remains in the report, including failures. Passing requires two distinct
factual repairs to accumulate, preserve previously correct answers, and survive
restart and streaming. This bounded demonstration does not establish general
recursive self-improvement. No checkpoint is downloaded.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import time
import unicodedata
from contextlib import closing
from pathlib import Path

import httpx

from scripts.wrapper_demo import (
    WrapperProcess,
    chat,
    record_adaptation,
    cleanup_lm_studio,
    cleanup_tags,
    request,
    require,
    stream_chat,
    split_thinking,
    verify_training,
)

# Fixed before native trials; do not silently replace items after seeing results.
CASES = (
    (
        "morocco_largest_city",
        ("Casablanca",),
        "What is the largest city in Morocco?",
        "Which Moroccan city has the most inhabitants?",
        "Name Morocco's largest city by population.",
    ),
    (
        "tanzania_capital",
        ("Dodoma",),
        "What is the capital of Tanzania?",
        "Which city is Tanzania's national capital?",
        "Name the capital city of the United Republic of Tanzania.",
    ),
    (
        "palau_capital",
        ("Ngerulmud",),
        "What is the capital of Palau?",
        "Which settlement serves as Palau's national capital?",
        "Name the capital of the Republic of Palau.",
    ),
    (
        "benin_capital",
        ("Porto-Novo", "Porto Novo"),
        "What is the official capital of Benin?",
        "Which city is Benin's official capital?",
        "Name the constitutional capital of the Republic of Benin.",
    ),
    (
        "sri_lanka_legislative_capital",
        ("Sri Jayawardenepura Kotte", "Sri Jayewardenepura Kotte", "Kotte"),
        "What is the legislative capital of Sri Lanka?",
        "Which city serves as Sri Lanka's legislative capital?",
        "Name the Sri Lankan city designated as its legislative capital.",
    ),
    (
        "canada_tallest_mountain",
        ("Mount Logan", "Mt. Logan", "Mt Logan"),
        "What is the tallest mountain in Canada?",
        "Which Canadian mountain has the highest elevation?",
        "Name Canada's highest mountain above sea level.",
    ),
    (
        "cambodia_currency",
        ("Cambodian riel", "riel", "KHR"),
        "What is the official national currency of Cambodia?",
        "Which currency is issued as Cambodia's national currency?",
        "Name the official currency of the Kingdom of Cambodia.",
    ),
    (
        "kazakhstan_capital",
        ("Astana",),
        "What is the current capital of Kazakhstan?",
        "Which city is Kazakhstan's capital today?",
        "Give the present name of the capital city of Kazakhstan.",
    ),
)
ANSWER_STYLE = " Reply with only the name, without explanation."


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("service", choices=("ollama", "llama-cpp", "lm-studio", "vllm"))
    p.add_argument("model")
    p.add_argument("--upstream")
    p.add_argument("--llama-server")
    p.add_argument("--lms")
    p.add_argument("--vllm-server")
    p.add_argument("--context-size", type=int, default=None)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Output budget per generation; thinking defaults to 2048, otherwise 160",
    )
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument(
        "--progress",
        action="store_true",
        help="Write brief phase updates to stderr; stdout remains the final JSON report",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--thinking",
        action="store_true",
        help="Explicitly enable thinking and require observed reasoning in learned answers",
    )
    mode.add_argument(
        "--nonthinking",
        action="store_true",
        help="Disable thinking explicitly on test chat and stream requests",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Maximum passes through the fixed eight items",
    )
    p.add_argument(
        "--required-repairs",
        type=int,
        default=2,
        help="Distinct factual repairs passing all three fixed wordings",
    )
    p.add_argument(
        "--full-budget",
        action="store_true",
        help="Continue through all fixed review opportunities after the required repairs",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wrapper-recurrent"))
    return p


def canonical(text):
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"[*_`\"'“”‘’]", "", text)
    return " ".join(text.rstrip(".!?").split())


def assess(text, aliases):
    if not isinstance(text, str):
        return dict(passed=False, classification="incomplete", final="")
    final, _, valid = split_thinking(text)
    if not valid:
        return dict(passed=False, classification="incomplete", final=final)
    cleaned = canonical(final)
    if not cleaned:
        return dict(passed=False, classification="incomplete", final=final)
    exact = cleaned in {canonical(alias) for alias in aliases}
    mentioned = any(
        re.search(r"(?<!\w)" + re.escape(canonical(alias)) + r"(?!\w)", cleaned)
        for alias in aliases
    )
    return dict(
        passed=exact,
        classification=(
            "correct"
            if exact
            else "formatting_or_extra_claims" if mentioned else "factual_missing"
        ),
        final=final,
    )


def generation_options(args):
    return dict(
        nonthinking=args.nonthinking,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        native_ollama=args.service == "ollama",
    )


def assess_generation(text, aliases, trace, *, thinking=False):
    result = assess(text, aliases)
    if thinking and not trace.get("complete"):
        result.update(passed=False, classification="incomplete")
    return result


def require_thinking(rows, message):
    require(
        all(
            row.get("generation", {}).get("thinking_observed")
            and row["generation"].get("complete")
            for row in rows
        ),
        message,
    )


def evaluate(
    client,
    model,
    *,
    nonthinking=False,
    thinking=False,
    max_tokens=160,
    scoped=False,
    native_ollama=False,
):
    rows = []
    for case_id, aliases, *questions in CASES:
        for prompt_index, question in enumerate(questions):
            prompt = question + ANSWER_STYLE
            trace = {}
            response, idx = chat(
                client,
                model,
                prompt,
                nonthinking=nonthinking,
                thinking=thinking,
                max_tokens=max_tokens,
                trace=trace,
                native_ollama=native_ollama,
            )
            rows.append(
                dict(
                    case_id=case_id,
                    prompt_index=prompt_index,
                    question=prompt,
                    response=response,
                    interaction_idx=idx,
                    generation=trace,
                    **assess_generation(response, aliases, trace, thinking=thinking),
                )
            )
    if scoped:
        history = request(client, "GET", "/history").json()["history"]
        by_id = {row["interaction_idx"]: row for row in history}
        for row in rows:
            record_adaptation(row["generation"], by_id[row["interaction_idx"]])
    return rows


def require_route(rows, artifact=None, *, scopes=None):
    """Prove scoped deployment uses the current cumulative weights, never an old repair."""
    if artifact is not None:
        require(
            re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]),
            "Invalid accepted adapter fingerprint",
        )
    for row in rows:
        route = row.get("generation", {}).get("adaptation")
        require(isinstance(route, dict), "Missing recorded adaptation route")
        require(
            route.get("policy") == "correction_scoped_v1", "Unknown adaptation policy"
        )
        if artifact is None:
            require(
                route.get("adapter") == "base"
                and route.get("adapter_sha256") is None
                and route.get("scope") is None,
                "An unrelated or baseline request used adapted weights",
            )
        else:
            require(
                route.get("adapter") == Path(artifact["directory"]).name
                and route.get("adapter_sha256") == artifact["sha256"]
                and isinstance(route.get("scope"), int)
                and not isinstance(route.get("scope"), bool),
                "A repaired answer did not use the latest cumulative adapter",
            )
            if scopes is not None:
                require(
                    route["scope"] in scopes,
                    "A repaired answer used an unrelated scope",
                )


def require_deployment(rows, artifact, scopes, repaired):
    """Retain learned facts on the newest adapter and unrelated facts on base."""
    require_route([row for row in rows if row["case_id"] not in scopes])
    for case_id in repaired:
        require_route(
            [row for row in rows if row["case_id"] == case_id],
            artifact,
            scopes=scopes[case_id],
        )


def row_key(row):
    return row["case_id"], row["prompt_index"]


def losses(protected, current):
    by_key = {row_key(row): row for row in current}
    return [key for key in protected if not by_key.get(key, {}).get("passed", False)]


def accepted_artifact(state):
    with closing(sqlite3.connect(state / "history.sqlite3")) as db:
        row = db.execute("SELECT value FROM meta WHERE key='accepted'").fetchone()
    require(row is not None, "Kept review did not persist an accepted adapter")
    directory = Path(json.loads(row[0])["directory"])
    data = (directory / "adapter" / "adapter_model.safetensors").read_bytes()
    job = json.loads((directory / "job.json").read_text())
    return dict(
        directory=str(directory),
        sha256=hashlib.sha256(data).hexdigest(),
        previous=job.get("previous"),
    )


def startup_base_digest(state):
    """The wrapper computes this fingerprint before any review or training."""
    with closing(sqlite3.connect(state / "history.sqlite3")) as db:
        row = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    require(row is not None, "Wrapper did not record its initial model identity")
    digest = json.loads(row[0]).rsplit(":", 1)[-1]
    require(re.fullmatch(r"[a-f0-9]{64}", digest), "Invalid initial model fingerprint")
    return digest


def verify_final_base(state, before):
    """Check original source bytes again after runtime shutdown and cleanup."""
    from adaptible._src.wrap.model_source import fingerprint_base

    jobs = sorted((state / "adapters").glob("*/job.json"))
    if not jobs:
        return dict(
            before=before, after=None, unchanged=None, reason="No training job ran"
        )
    sources = {Path(json.loads(job.read_text())["blob"]).resolve() for job in jobs}
    require(len(sources) == 1, "Training jobs unexpectedly used different base models")
    source = sources.pop()
    after = fingerprint_base(source)
    require(after == before, "Original base weights changed during the recurrent run")
    return dict(source=str(source), before=before, after=after, unchanged=True)


def continues_from(artifact, previous):
    """Allow multiple accepted candidates in one review without losing lineage."""
    expected = (Path(previous["directory"]) / "adapter").resolve()
    predecessor = artifact["previous"]
    visited = set()
    while predecessor:
        path = Path(predecessor).resolve()
        if path == expected:
            return True
        if path in visited:
            return False
        visited.add(path)
        job = path.parent / "job.json"
        if not job.is_file():
            return False
        predecessor = json.loads(job.read_text()).get("previous")
    return False


def progress(args, message):
    if args.progress:
        print(f"[recurrent] {message}", file=sys.stderr, flush=True)


def run_recurrent(args):
    require(args.cycles > 0 and args.timeout > 0, "Cycles and timeout must be positive")
    args.required_repairs = getattr(args, "required_repairs", 2)
    args.full_budget = getattr(args, "full_budget", False)
    require(
        2 <= args.required_repairs <= len(CASES),
        "Require between two and eight distinct repairs",
    )
    args.thinking = getattr(args, "thinking", False)
    args.max_tokens = getattr(args, "max_tokens", None) or (
        2048 if args.thinking else 160
    )
    args.context_size = args.context_size or (8192 if args.thinking else 4096)
    require(
        args.max_tokens > 0 and args.context_size > args.max_tokens,
        "Context must leave room beyond the output budget",
    )
    args.web_search = True
    evaluation_options = {**generation_options(args), "scoped": args.thinking}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=args.service + "-", dir=args.output_dir)
    ).resolve()
    state = directory / "state"
    wrapper = WrapperProcess(args, directory)
    report = dict(
        service=args.service,
        model=args.model,
        directory=str(directory),
        recurrent_status="failed",
        reference_mode="automatic_web",
        nonthinking=args.nonthinking,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        context_size=args.context_size,
        required_distinct_repairs=args.required_repairs,
        full_budget=args.full_budget,
        deployment_policy="correction_scoped_v1" if args.thinking else "legacy_adapter",
        review_opportunities=[],
        dataset=CASES,
        attempts=[],
        accumulated_repairs=[],
        quality_scope="Exact whole-answer entity, two external held-out wordings, retention, restart and streaming.",
    )
    started = time.monotonic()
    try:
        progress(args, f"Starting {args.service}; logs: {directory}")
        wrapper.start(args.timeout)
        report["base_sha256_before"] = startup_base_digest(state)
        with httpx.Client(
            base_url=wrapper.url, timeout=args.timeout, trust_env=False
        ) as client:
            models = request(client, "GET", "/v1/models").json()["data"]
            require(len(models) == 1, "Expected one wrapped model")
            model = models[0]["id"]
            progress(args, "Runtime ready; checking the fixed 24-prompt baseline")
            current = report["baseline"] = evaluate(client, model, **evaluation_options)
            if args.thinking:
                require_route(current)
                require(
                    any(row["generation"]["thinking_observed"] for row in current),
                    "No actual thinking observed in baseline",
                )
            eligible = {
                r["case_id"]
                for r in current
                if r["prompt_index"] == 0 and r["classification"] == "factual_missing"
            }
            report["eligible_baseline_errors"] = sorted(eligible)
            progress(
                args,
                f"Baseline complete: {sum(row['passed'] for row in current)}/24 exact answers; {len(eligible)} eligible original errors",
            )
            protected = {row_key(row) for row in current if row["passed"]}
            repaired, previous_artifact, accepted_scopes = set(), None, {}
            if len(eligible) < args.required_repairs:
                report.update(
                    recurrent_status="inconclusive",
                    reason=f"Fewer than {args.required_repairs} original baseline answers actually miss the expected entity; format-only failures cannot count as factual repairs.",
                )
                return report
            for cycle in range(args.cycles):
                for case_id, aliases, question, *_ in CASES:
                    opportunity = dict(
                        cycle=cycle, case_id=case_id, accepted_repairs=len(repaired)
                    )
                    report["review_opportunities"].append(opportunity)
                    if case_id not in eligible or case_id in repaired:
                        opportunity["status"] = (
                            "already_repaired"
                            if case_id in repaired
                            else "not_eligible"
                        )
                        continue
                    before_trace = {}
                    before, idx = chat(
                        client,
                        model,
                        question + ANSWER_STYLE,
                        **generation_options(args),
                        trace=before_trace,
                    )
                    assessment = assess_generation(
                        before, aliases, before_trace, thinking=args.thinking
                    )
                    if assessment["classification"] != "factual_missing":
                        opportunity.update(
                            status="not_current_factual_error",
                            generation=before_trace,
                            assessment=assessment,
                        )
                        continue
                    attempt = dict(
                        cycle=cycle,
                        case_id=case_id,
                        interaction_idx=idx,
                        before=before,
                        before_assessment=assessment,
                        before_generation=before_trace,
                    )
                    report["attempts"].append(attempt)
                    progress(
                        args,
                        f"Cycle {cycle + 1}: reviewing {case_id} with automatic web evidence",
                    )
                    request(
                        client,
                        "POST",
                        "/feedback",
                        json=dict(interaction_idx=idx, thumbs="down"),
                    )
                    attempt["review"] = request(client, "GET", "/sync").json()
                    history = request(client, "GET", "/history").json()["history"]
                    row = next(r for r in history if r["interaction_idx"] == idx)
                    attempt["status"], attempt["references"] = (
                        row["status"],
                        row["references"],
                    )
                    opportunity.update(status=row["status"], interaction_idx=idx)
                    progress(
                        args,
                        f"Review {case_id}: {row['status']}; checking all 24 prompts for gains and retention",
                    )
                    require(
                        row["note"] == "" and row["references"]["kind"] == "web",
                        "Manual evidence bypassed automatic lookup",
                    )
                    require(
                        row["status"] != "failed", f"Review failed: {row.get('reason')}"
                    )
                    if row["status"] == "kept":
                        require(
                            row["references"].get("sources"),
                            "Kept repair has no web evidence",
                        )
                        verify_training(state, attempt)
                        artifact = attempt["artifact"] = accepted_artifact(state)
                        if previous_artifact:
                            require(
                                artifact["sha256"] != previous_artifact["sha256"],
                                "Accepted update did not change adapter weights",
                            )
                            require(
                                continues_from(artifact, previous_artifact),
                                "Training did not continue from the preceding accepted adapter",
                            )
                        previous_artifact = artifact
                        accepted_scopes.setdefault(case_id, set()).add(idx)
                    current = attempt["after_matrix"] = evaluate(
                        client, model, **evaluation_options
                    )
                    if args.thinking:
                        require_deployment(
                            current, previous_artifact, accepted_scopes, repaired
                        )
                    regressions = attempt["regressions"] = losses(protected, current)
                    require(
                        not regressions,
                        f"Previously correct answers regressed: {regressions}",
                    )
                    target = [r for r in current if r["case_id"] == case_id]
                    if row["status"] == "kept" and all(r["passed"] for r in target):
                        if args.thinking:
                            require_thinking(
                                target,
                                "A repaired answer passed without completed actual reasoning",
                            )
                            require_route(
                                target,
                                previous_artifact,
                                scopes=accepted_scopes[case_id],
                            )
                        repaired.add(case_id)
                        report["accumulated_repairs"].append(case_id)
                    progress(
                        args,
                        f"After {case_id}: {sum(row['passed'] for row in current)}/24 exact answers; {len(repaired)}/{args.required_repairs} accumulated repairs",
                    )
                    protected.update(row_key(r) for r in current if r["passed"])
                    (directory / "report.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    if len(repaired) >= args.required_repairs and not args.full_budget:
                        break
                if len(repaired) >= args.required_repairs and not args.full_budget:
                    break
            report["final_matrix"] = current
            if len(repaired) < args.required_repairs:
                report.update(
                    recurrent_status="inconclusive",
                    reason=f"The fixed bounded run did not accumulate {args.required_repairs} accepted factual repairs passing both unseen wordings; all attempts are retained.",
                )
                return report
            history = request(client, "GET", "/history").json()
            progress(
                args,
                f"{len(repaired)} repairs accumulated; checking restart, retained answers and streams",
            )
            wrapper.stop()
            wrapper.start(args.timeout)
            require(
                request(client, "GET", "/history").json() == history,
                "History changed across restart",
            )
            report["restart_matrix"] = evaluate(client, model, **evaluation_options)
            require(
                not losses(protected, report["restart_matrix"]),
                "Learned or baseline-correct answers were lost on restart",
            )
            if args.thinking:
                require_deployment(
                    current, previous_artifact, accepted_scopes, repaired
                )
                require_deployment(
                    report["restart_matrix"],
                    previous_artifact,
                    accepted_scopes,
                    repaired,
                )
                require_thinking(
                    [r for r in current if r["case_id"] in repaired],
                    "Final repairs lost actual reasoning",
                )
                require_thinking(
                    [r for r in report["restart_matrix"] if r["case_id"] in repaired],
                    "Restart repairs lost actual reasoning",
                )
            report["streams"] = []
            for case_id, aliases, question, *_ in CASES:
                if case_id not in repaired:
                    continue
                paths = ["/v1/chat/completions"]
                if args.service == "ollama":
                    paths += ["/api/chat", "/api/generate"]
                for path in paths:
                    stream_trace = {}
                    response = stream_chat(
                        client,
                        model,
                        question + ANSWER_STYLE,
                        path,
                        **generation_options(args),
                        trace=stream_trace,
                    )
                    result = dict(
                        case_id=case_id,
                        path=path,
                        response=response,
                        generation=stream_trace,
                        **assess_generation(
                            response, aliases, stream_trace, thinking=args.thinking
                        ),
                    )
                    report["streams"].append(result)
                    if args.thinking:
                        require_thinking(
                            [result], "Stream lacks completed actual reasoning"
                        )
                        require_route(
                            [result], previous_artifact, scopes=accepted_scopes[case_id]
                        )
                    require(
                        result["passed"],
                        "Streaming did not preserve the corrected whole answer",
                    )
            report["recurrent_status"] = "passed"
    except Exception as exc:
        report.update(recurrent_status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            wrapper.stop()
            report["removed_test_tags"] = cleanup_tags(args, state)
            report["removed_test_imports"] = cleanup_lm_studio(args, state)
            if report.get("base_sha256_before"):
                report["base_fingerprint"] = verify_final_base(
                    state, report["base_sha256_before"]
                )
        except Exception as exc:
            report.update(recurrent_status="failed", cleanup_error=str(exc))
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        progress(
            args,
            f"Finished: {report['recurrent_status']} in {report['elapsed_seconds']} seconds; report: {directory / 'report.json'}",
        )
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    report = run_recurrent(parser().parse_args(argv))
    print(json.dumps(report, indent=2))
    return {"passed": 0, "failed": 1, "inconclusive": 2}[report["recurrent_status"]]


if __name__ == "__main__":
    raise SystemExit(main())
