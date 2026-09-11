"""Turn a self-repair cycle run into the parsable files under results/.

Input is either the MLX run log written by ``scripts/cycles_mlx.py`` (CYCLE and
CAND lines) or the ``history.json`` written by ``scripts/colab/cycles_torch.py``.
Output, in ``--out``:

  cycles.csv       one row per cycle: score out of 20, no-answer loops, and the
                   per-item marks (original + 3 paraphrases, ✓/✗)
  candidates.csv   one row per training candidate: cycle, item, steps, final
                   answer-token loss, the target's first sentence, the 4-prompt
                   marks after training, and KEEP / restore
  score.svg        score and loops per cycle, no plotting dependency

    python scripts/cycles_results.py --log outputs/runs/mve_cycles.log --out results/<name>
    python scripts/cycles_results.py --history history.json --out results/<name>
"""
import argparse, csv, json, re
from pathlib import Path

ITEMS = ["geo_010", "geo_004", "geo_001", "geo_013", "sci_017"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CYCLE = re.compile(r"CYCLE (\d+) score=(\d+)/(\d+) loops=(\d+) \| (.*?)(?: \| \d+s)?$")
CAND = re.compile(r"CAND cycle=(\d+) (\w+) k=(\d+) steps=(\d+) loss=([\d.]+) target_answer='(.*?)' \| ([✓✗]+) loops=(\d+) \| (KEEP|restore)")


def from_log(path):
    cycles, cands = [], []
    for raw in Path(path).read_text().splitlines():
        line = ANSI.sub("", raw)
        m = CYCLE.search(line)
        if m:
            marks = dict(p.split(":") for p in m.group(5).split())
            cycles.append(dict(cycle=int(m.group(1)), score=int(m.group(2)), loops=int(m.group(4)), **{k: marks.get(k, "") for k in ITEMS}))
        m = CAND.search(line)
        if m:
            cands.append(dict(cycle=int(m.group(1)), item=m.group(2), k=int(m.group(3)), steps=int(m.group(4)), loss=float(m.group(5)),
                              target=m.group(6), marks=m.group(7), loops=int(m.group(8)), kept=m.group(9) == "KEEP"))
    return cycles, cands


def from_history(path):
    h = json.loads(Path(path).read_text())
    cycles = [dict(cycle=c["cycle"], score=c["score"], loops=c["loops"], **{k: c["marks"].get(k, "") for k in ITEMS}) for c in h["cycles"]]
    return cycles, []


def svg(cycles, path, total=20):
    w, h, ml, mb = 720, 300, 50, 40
    n = max(len(cycles), 2)
    x = lambda i: ml + i * (w - ml - 20) / (n - 1)
    y = lambda v: h - mb - v * (h - mb - 20) / total
    pts = " ".join(f"{x(i):.1f},{y(c['score']):.1f}" for i, c in enumerate(cycles))
    lps = " ".join(f"{x(i):.1f},{y(c['loops']):.1f}" for i, c in enumerate(cycles))
    grid = "".join(f'<line x1="{ml}" y1="{y(v):.1f}" x2="{w-20}" y2="{y(v):.1f}" stroke="#ddd"/><text x="{ml-8}" y="{y(v)+4:.1f}" font-size="11" text-anchor="end">{v}</text>' for v in range(0, total + 1, 5))
    xt = "".join(f'<text x="{x(i):.1f}" y="{h-mb+16}" font-size="11" text-anchor="middle">{c["cycle"]}</text>' for i, c in enumerate(cycles) if c["cycle"] % 5 == 0)
    out = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif">
<rect width="{w}" height="{h}" fill="white"/>{grid}{xt}
<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2"/>
<polyline points="{lps}" fill="none" stroke="#d62728" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="{ml}" y="14" font-size="12" fill="#1f77b4">correct prompts (of {total})</text>
<text x="{ml+170}" y="14" font-size="12" fill="#d62728">generations with no answer (loops)</text>
<text x="{(w+ml)/2:.0f}" y="{h-4}" font-size="12" text-anchor="middle">cycle</text>
</svg>'''
    Path(path).write_text(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log"); ap.add_argument("--history"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cycles, cands = from_log(a.log) if a.log else from_history(a.history)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "cycles.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["cycle", "score", "loops"] + ITEMS); wr.writeheader(); wr.writerows(cycles)
    if cands:
        with open(out / "candidates.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(cands[0])); wr.writeheader(); wr.writerows(cands)
    svg(cycles, out / "score.svg")
    kept = sum(c["kept"] for c in cands)
    print(f"{len(cycles)} cycles, {len(cands)} candidates ({kept} kept) -> {out}")
    print("cycle score loops"); [print(c["cycle"], c["score"], c["loops"]) for c in cycles]


if __name__ == "__main__":
    main()
