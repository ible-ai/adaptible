"""Turn a live-loop run (scripts/live_repair.py) into a results folder.

    python scripts/live_results.py --run outputs/runs/live-5x16f --out results/<name>

Writes sessions.csv (one row per session: flagged, kept, outside score, controls,
per-item marks), repairs.csv (one row per candidate: item, target, steps, loss,
node's own marks before/after, kept), and score.svg (outside score per session,
controls dashed). The node's own marks come from the run log's REPAIR lines;
everything else from history.json.
"""
import argparse, csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cycles_results import svg  # noqa: E402

ITEMS = ["geo_010", "geo_004", "geo_001", "geo_013", "sci_017"]
CONTROLS = ["geo_012", "sci_002"]


def repairs_from_log(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(errors="replace"))
    # ASK lines map interaction idx -> item; REPAIR lines carry idx.
    item_of = {int(m.group(2)): m.group(1) for m in re.finditer(r"ASK session=\d+ (\S+) idx=(\d+)", text)}
    session_of = {int(m.group(2)): int(m.group(1)) for m in re.finditer(r"ASK session=(\d+) \S+ idx=(\d+)", text)}
    rows, before = [], {}
    for m in re.finditer(r"REPAIR idx=(\d+) (before=([✓✗]+)|cand=(\d) steps=(\d) loss=([\d.]+) target='(.*?)' \| ([✓✗]+) \| (KEEP|restore)|NOCAND|NONOTE)", text):
        idx = int(m.group(1))
        if m.group(3):
            before[idx] = m.group(3); continue
        if m.group(4) is None:
            rows.append(dict(session=session_of.get(idx), item=item_of.get(idx), candidate="", steps="", loss="", target="", before=before.get(idx, ""), after="", kept=0, note=m.group(2)))
            continue
        rows.append(dict(session=session_of.get(idx), item=item_of.get(idx), candidate=int(m.group(4)), steps=int(m.group(5)), loss=float(m.group(6)),
                         target=m.group(7), before=before.get(idx, ""), after=m.group(8), kept=int(m.group(9) == "KEEP"), note=""))
        if m.group(9) == "KEEP":
            before[idx] = m.group(8)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = json.loads((run / "history.json").read_text())
    repairs = repairs_from_log(run / "run.log")
    kept_by_session = {}
    for r in repairs:
        kept_by_session[r["session"]] = kept_by_session.get(r["session"], 0) + r["kept"]
    sessions = []
    for h in hist:
        row = dict(session=h["session"], flagged=len(h["flagged"]), kept=kept_by_session.get(h["session"], 0), score=h["score"], controls=h["control_score"], seconds=h["seconds"])
        row.update({k: h["marks"][k] for k in ITEMS}); row.update({k: h["controls"][k] for k in CONTROLS})
        sessions.append(row)
    with open(out / "sessions.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(sessions[0])); wr.writeheader(); wr.writerows(sessions)
    with open(out / "repairs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(repairs[0])); wr.writeheader(); wr.writerows(repairs)
    # score.svg: outside score per session; dashed line = control misses (0-2) on the same axis
    svg([dict(cycle=s["session"], score=s["score"], loops=2 - s["controls"]) for s in sessions], out / "score.svg")
    text = (out / "score.svg").read_text().replace("generations with no answer (loops)", "control facts wrong (of 2)").replace(">cycle<", ">session<")
    (out / "score.svg").write_text(text)
    print(f"{len(sessions)} sessions, {len([r for r in repairs if r['candidate'] != ''])} candidates ({sum(r['kept'] for r in repairs)} kept) -> {out}")
    print("session flagged kept score controls"); [print(s["session"], s["flagged"], s["kept"], s["score"], s["controls"]) for s in sessions]


if __name__ == "__main__":
    main()
