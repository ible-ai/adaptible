"""The flagship self-repair cycle loop (scripts/cycles_mlx.py), run through the wrapper.

Same protocol, different infrastructure. Each cycle scores every item on its
original question plus three paraphrases; any item under 4/4 is flagged with the
experiment's reference note, the wrapper's review generates the correction from
that note using the model itself, trains, and keeps the update only if its own
checks pass. The flagship's greedy keep/restore is the wrapper's accept/reject.

Fixed items, paraphrases, reference-note wording and judging come from the
experiment. Nothing here supplies a correction: the note states the answer, as
``cycles_mlx.hinted`` does, and the model writes the correction from it.

    python -m scripts.wrapper_cycles llama-cpp /path/model.gguf \
        --llama-server /path/llama-server --cycles 2 --items geo_004,sci_017
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from adaptible.eval.harness import contains_key_terms
from adaptible.eval import generate_default_dataset

# scripts/cycles_mlx.py PARA, verbatim.
PARA = {
    "geo_010": [
        "Which city is the capital of Morocco?",
        "Morocco's seat of government is in which city?",
        "Name the capital city of the Kingdom of Morocco.",
    ],
    "geo_004": [
        "Which city serves as Turkey's capital?",
        "Where is the seat of the Turkish government located?",
        "Name the capital city of Turkey.",
    ],
    "geo_001": [
        "Which city is Australia's capital?",
        "Where is the seat of Australia's federal government?",
        "Name the capital city of Australia.",
    ],
    "geo_013": [
        "Which city is the capital of the Philippines?",
        "Where is the seat of government of the Philippines?",
        "Name the capital city of the Philippines.",
    ],
    "sci_017": [
        "Which star is closest to our planet?",
        "What is the closest star to Earth?",
        "Name the star nearest to the Earth.",
    ],
}

# cycles_mlx.WRONG: answers that contain a key term but are still wrong.
WRONG = {"sci_017": ["Proxima", "Alpha Centauri"]}


def parser():
    p = argparse.ArgumentParser(
        prog="python -m scripts.wrapper_cycles", description=__doc__
    )
    p.add_argument("service", choices=("ollama", "llama-cpp", "lm-studio", "vllm"))
    p.add_argument("model")
    p.add_argument("--upstream")
    p.add_argument("--llama-server")
    p.add_argument("--lms")
    p.add_argument("--vllm-server")
    p.add_argument(
        "--items", default="geo_004,sci_017", help="Comma-separated dataset ids"
    )
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--context-size", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=1800,
        help="Seconds to wait for the wrapper to serve. A runtime that loads "
        "the model on CPU (vLLM on macOS) takes minutes before it answers.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wrapper-cycles"))
    return p


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# A reasoning model's answer is short. Anything this long without a closed
# think block is the thought itself, which the experiment scores as a loop.
_LOOP_CHARACTERS = 800


def closed(text):
    """cycles_mlx.closed: an unclosed thought is a miss, not an answer.

    The flagship asks ``"</think>" in r`` because its model always reasons and
    its serving always delimits the thought. A wrapper runtime may return a
    plain short answer with no markers, which is complete. But a runtime
    serving a reasoning model without a reasoning parser returns the raw
    thought as the answer, and one such response -- 7,599 characters ending on
    the wrong answer -- scored as correct because the key term appeared
    somewhere inside it. Length is the signal available without markers.
    """
    if "<think>" in text:
        return "</think>" in text
    return "</think>" in text or len(text) < _LOOP_CHARACTERS


def answer_of(text):
    """cycles_mlx.answer_of: judge the answer, never the reasoning."""
    return text.split("</think>")[-1].strip()


def ok(item, text):
    """cycles_mlx.ok, verbatim in behaviour.

    Scoring the whole response instead of ``answer_of`` counts a key term that
    appears only inside the model's thinking, and dropping ``closed`` counts a
    generation that was cut off mid-thought. Both inflate the score.
    """
    if not closed(text):
        return False
    body = answer_of(text)
    return contains_key_terms(body, item.key_terms) and not any(
        w.lower() in body.lower() for w in WRONG.get(item.id, [])
    )


def prompts(item):
    return [item.question] + PARA[item.id]


class Wrapper:
    """The wrapper process, and the three calls this protocol needs."""

    def __init__(self, args, directory):
        self.directory = directory
        command = [
            sys.executable,
            "-m",
            "adaptible",
            "wrap",
            args.service,
            args.model,
            "--state-dir",
            str(directory / "state"),
            "--port",
            str(free_port()),
            "--max-tokens",
            str(args.max_tokens),
            "--context-size",
            str(args.context_size),
            "--no-web-search",
            "--flagship-recipe",
        ]
        for flag, value in (
            ("--upstream", args.upstream),
            ("--llama-server", args.llama_server),
            ("--lms", args.lms),
            ("--vllm-server", args.vllm_server),
        ):
            if value:
                command += [flag, value]
        self.log = (directory / "wrapper.log").open("wb")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
        self.url = None

    def wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"wrapper exited; see {self.directory / 'wrapper.log'}"
                )
            text = (self.directory / "wrapper.log").read_text(errors="replace")
            for line in text.splitlines():
                if "Chat: http" in line:
                    self.url = line.split("Chat: ")[1].split("/v1")[0].strip()
                    # The banner is printed before uvicorn binds, so the first
                    # request would race the listener.
                    while time.monotonic() < deadline:
                        try:
                            httpx.get(self.url + "/status", timeout=5)
                            return self.url
                        except httpx.HTTPError:
                            time.sleep(2)
            time.sleep(2)
        raise RuntimeError("wrapper did not report its address")

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


def main(argv=None):
    args = parser().parse_args(argv)
    dataset = {item.id: item for item in generate_default_dataset()}
    items = [dataset[i] for i in args.items.split(",")]
    run = args.output_dir / f"{args.service}-{int(time.time())}"
    run.mkdir(parents=True, exist_ok=True)

    wrapper = Wrapper(args, run)
    report = dict(
        service=args.service, model=args.model, items=[i.id for i in items], cycles=[]
    )
    try:
        url = wrapper.wait_ready(args.startup_timeout)
        client = httpx.Client(base_url=url, timeout=args.timeout)

        def ask(question):
            r = client.post(
                "/interact", json={"prompt": question, "use_history": False}
            )
            r.raise_for_status()
            return r.json()

        def score(item):
            """Original + 3 paraphrases, each in a fresh conversation."""
            marks, texts = [], []
            for question in prompts(item):
                body = ask(question)
                texts.append(body["response"])
                marks.append(ok(item, body["response"]))
            return marks, texts

        baseline = {}
        for item in items:
            marks, texts = score(item)
            baseline[item.id] = marks
            print(f"[cycles] baseline {item.id}: {sum(marks)}/4", flush=True)
        report["baseline"] = {k: sum(v) for k, v in baseline.items()}

        current = dict(baseline)
        for cycle in range(args.cycles):
            entry = dict(cycle=cycle, reviews=[])
            for item in items:
                if sum(current[item.id]) == len(prompts(item)):
                    continue
                asked = ask(item.question)
                # cycles_mlx.hinted: the note states the answer; the model
                # writes its own correction from it.
                note = f"the correct answer is {item.correct_answer}."
                flagged = client.post(
                    "/feedback",
                    json=dict(
                        interaction_idx=asked["interaction_idx"],
                        thumbs="down",
                        note=note,
                        # cycles_mlx.py keeps an update on the item's own
                        # four-prompt score, so the loop is judged against the
                        # experiment's paraphrases, not generated substitutes.
                        reask_prompts=PARA[item.id],
                        # cycles_mlx.WRONG: answers naming these contain the
                        # key term but are still wrong.
                        wrong_terms=WRONG.get(item.id, []),
                    ),
                )
                review = dict(item=item.id, feedback_status=flagged.status_code)
                if flagged.status_code == 200:
                    client.post("/trigger_review")
                    result = client.get("/sync").json()["reviews"]
                    review["review"] = result[0] if result else None
                    marks, texts = score(item)
                    review["before"], review["after"] = sum(current[item.id]), sum(
                        marks
                    )
                    review["answers"] = texts
                    current[item.id] = marks
                    print(
                        f"[cycles] cycle {cycle} {item.id}: "
                        f"{review['review']['status'] if review['review'] else '?'} "
                        f"{review['before']}/4 -> {review['after']}/4",
                        flush=True,
                    )
                else:
                    review["error"] = flagged.text[:200]
                    print(
                        f"[cycles] cycle {cycle} {item.id}: not flaggable", flush=True
                    )
                entry["reviews"].append(review)
            report["cycles"].append(entry)

        report["final"] = {k: sum(v) for k, v in current.items()}
        report["baseline_total"] = sum(report["baseline"].values())
        report["final_total"] = sum(report["final"].values())
        report["prompts_total"] = 4 * len(items)
    finally:
        wrapper.close()
        (run / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
