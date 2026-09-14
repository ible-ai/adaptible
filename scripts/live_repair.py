"""The live loop, end to end: a served model, a thumbs-down, a self-repair, and a re-ask.

Each session asks the server the five target questions and two controls, thumbs
down every target answer the judge marks wrong, triggers a review, waits for it,
then re-asks everything plus three rephrasings per target. The node's lookup is
a sealed DocStore of short passages (targets, controls, distractors); the
key-term judge and the rephrasings never enter the loop. Log lines are grep-able
(ASK / FLAG / SESSION); history.json holds the per-session table.

    PYTHONPATH=. .venv/bin/python scripts/live_repair.py --sessions 8 --out outputs/runs/live
"""
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path

import httpx

import adaptible
from adaptible._src.lookup import DEFAULT_PASSAGES as PASSAGES
from adaptible._src.eval.harness import contains_key_terms
from adaptible.eval import generate_default_dataset
from adaptible.local import MutableHostedLLM

PARA = {
 "geo_010": ["Which city is the capital of Morocco?", "Morocco's seat of government is in which city?", "Name the capital city of the Kingdom of Morocco."],
 "geo_004": ["Which city serves as Turkey's capital?", "Where is the seat of the Turkish government located?", "Name the capital city of Turkey."],
 "geo_001": ["Which city is Australia's capital?", "Where is the seat of Australia's federal government?", "Name the capital city of Australia."],
 "geo_013": ["Which city is the capital of the Philippines?", "Where is the seat of government of the Philippines?", "Name the capital city of the Philippines."],
 "sci_017": ["Which star is closest to our planet?", "What is the closest star to Earth?", "Name the star nearest to the Earth."],
}
CONTROLS = ["geo_012", "sci_002"]  # Hanoi, Au: asked every session, never flagged
WRONG = {"sci_017": ["Proxima", "Alpha Centauri"]}


def answer_of(r): return r.split("</think>")[-1].strip()
def closed(r): return "</think>" in r
def ok(it, r):
    a = answer_of(r)
    return closed(r) and contains_key_terms(a, it.key_terms) and not any(w.lower() in a.lower() for w in WRONG.get(it.id, []))


async def run(args):
    ds = {i.id: i for i in generate_default_dataset()}
    targets = list(PARA) if not args.items else args.items.split(",")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ADAPTIBLE_OUTPUTS_DIR", str(out))
    store = adaptible.DocStore(PASSAGES)
    model = adaptible.StatefulLLM(model_path=out / "checkpoint", lookup=store.search)
    api = adaptible.Adaptible(model=model)
    server = MutableHostedLLM(host="127.0.0.1", port=args.port, app=api.app)
    await server.up()
    base = f"http://127.0.0.1:{args.port}"
    hist_path = out / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    async with httpx.AsyncClient(base_url=base, timeout=None) as c:
        async def ask(q):
            r = await c.post("/interact", json={"prompt": q, "use_history": False}); d = r.json()
            return d["interaction_idx"], d["response"]
        for s in range(len(history), args.sessions):
            t0 = time.time()
            # 1. ask; 2. thumb down wrong targets
            flagged = []
            first = {}
            for k in targets + CONTROLS:
                idx, resp = await ask(ds[k].question)
                good = ok(ds[k], resp)
                first[k] = good
                print(f"ASK session={s} {k} idx={idx} ok={int(good)} | {answer_of(resp)[:70]!r}", flush=True)
                if not good and k in targets:
                    await c.post("/feedback", json={"interaction_idx": idx, "thumbs": "down"}); flagged.append(k)
                    print(f"FLAG session={s} {k} idx={idx}", flush=True)
            # 3. the node reviews on its own
            if flagged:
                await c.post("/trigger_review"); await c.get("/sync")
            # 4. re-ask: originals + rephrasings + controls (never flagged)
            marks = {}
            for k in targets:
                outs = [(await ask(q))[1] for q in [ds[k].question] + PARA[k]]
                marks[k] = "".join("✓" if ok(ds[k], o) else "✗" for o in outs)
            ctl = {k: "✓" if ok(ds[k], (await ask(ds[k].question))[1]) else "✗" for k in CONTROLS}
            score = sum(v.count("✓") for v in marks.values())
            csc = sum(v == "✓" for v in ctl.values())
            history.append(dict(session=s, flagged=flagged, first={k: int(v) for k, v in first.items()}, marks=marks, controls=ctl, score=score, control_score=csc, seconds=round(time.time() - t0)))
            hist_path.write_text(json.dumps(history, indent=1))
            print(f"SESSION {s} flagged={len(flagged)} score={score}/{4*len(targets)} controls={csc}/{len(CONTROLS)} | " + " ".join(f"{k}:{v}" for k, v in marks.items()) + " | " + " ".join(f"{k}:{v}" for k, v in ctl.items()) + f" | {round(time.time()-t0)}s", flush=True)
    await server.down()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--items", default="", help="comma-separated subset of target ids (default: all five)")
    ap.add_argument("--out", default="outputs/runs/live")
    ap.add_argument("--port", type=int, default=8123)
    asyncio.run(run(ap.parse_args()))
