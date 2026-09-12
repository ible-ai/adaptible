"""After-the-fact probes for a self-repair run: does a corrected fact survive prompts that
break the "capital of X?" template? Loads the base model and saved adapters (adapter_NN.pt
from cycles_torch.py) and asks, per item, a reverse question, a yes/no on the old belief, a
question that uses the fact, one that needs both the capital and the largest city, and one
about the wrong city. Two facts the base model already knows are controls.

    python scripts/colab/probe_torch.py --run /workspace/out/seed3000 --adapters base,23,39
"""
import argparse, importlib.util, json, re, sys, unicodedata
from pathlib import Path

spec = importlib.util.spec_from_file_location("cycles_torch", Path(__file__).with_name("cycles_torch.py"))
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)

# item -> (right, wrong, prompts); each prompt: (kind, text, expected terms (all must appear), forbidden terms)
PROBES = {
    "geo_010": ("Rabat", "Casablanca", [
        ("reverse", "Which country has Rabat as its capital?", ["Morocco"], []),
        ("yesno", "Is Casablanca the capital of Morocco? Answer yes or no, then name the capital.", ["Rabat"], []),
        ("use", "I land in Morocco's capital tomorrow. Which city's weather should I check?", ["Rabat"], []),
        ("both", "Name Morocco's capital and its largest city.", ["Rabat", "Casablanca"], []),
        ("wrong", "Tell me one thing Casablanca is famous for.", [], []),
    ]),
    "geo_004": ("Ankara", "Istanbul", [
        ("reverse", "Which country has Ankara as its capital?", ["Turkey"], []),
        ("yesno", "Is Istanbul the capital of Turkey? Answer yes or no, then name the capital.", ["Ankara"], []),
        ("use", "I land in Turkey's capital tomorrow. Which city's weather should I check?", ["Ankara"], []),
        ("both", "Name Turkey's capital and its largest city.", ["Ankara", "Istanbul"], []),
        ("wrong", "Tell me one thing Istanbul is famous for.", [], []),
    ]),
    "geo_001": ("Canberra", "Sydney", [
        ("reverse", "Which country has Canberra as its capital?", ["Australia"], []),
        ("yesno", "Is Sydney the capital of Australia? Answer yes or no, then name the capital.", ["Canberra"], []),
        ("use", "I land in Australia's capital tomorrow. Which city's weather should I check?", ["Canberra"], []),
        ("both", "Name Australia's capital and its largest city.", ["Canberra", "Sydney"], []),
        ("wrong", "Tell me one thing Sydney is famous for.", [], []),
    ]),
    "geo_013": ("Manila", "Quezon City", [
        ("reverse", "Which country has Manila as its capital?", ["Philippines"], []),
        ("yesno", "Is Quezon City the capital of the Philippines? Answer yes or no, then name the capital.", ["Manila"], []),
        ("use", "I land in the Philippines' capital tomorrow. Which city's weather should I check?", ["Manila"], []),
        ("both", "Name the capital of the Philippines and its most populous city.", ["Manila", "Quezon"], []),
        ("wrong", "Tell me one thing Quezon City is famous for.", [], []),
    ]),
    "sci_017": ("Sun", "Proxima Centauri", [
        ("reverse", "Proxima Centauri is the nearest star to which star?", ["Sun"], []),
        ("yesno", "Is Proxima Centauri the nearest star to Earth? Answer yes or no, then name the nearest star.", ["Sun"], []),
        ("use", "I want to look at the nearest star to Earth. Do I go out at night or in the day?", ["day"], []),
        ("both", "Name the nearest star to Earth and the nearest star after that.", ["Sun", "Proxima"], []),
        ("wrong", "Tell me one thing Proxima Centauri is known for.", [], []),
    ]),
    "ctl_fr": ("Paris", "Marseille", [
        ("reverse", "Which country has Paris as its capital?", ["France"], []),
        ("yesno", "Is Marseille the capital of France? Answer yes or no, then name the capital.", ["Paris"], []),
        ("use", "I land in France's capital tomorrow. Which city's weather should I check?", ["Paris"], []),
        ("both", "Name France's capital and its largest city.", ["Paris"], []),
        ("wrong", "Tell me one thing Marseille is famous for.", [], []),
    ]),
    "ctl_jp": ("Tokyo", "Osaka", [
        ("reverse", "Which country has Tokyo as its capital?", ["Japan"], []),
        ("yesno", "Is Osaka the capital of Japan? Answer yes or no, then name the capital.", ["Tokyo"], []),
        ("use", "I land in Japan's capital tomorrow. Which city's weather should I check?", ["Tokyo"], []),
        ("both", "Name Japan's capital and its largest city.", ["Tokyo"], []),
        ("wrong", "Tell me one thing Osaka is famous for.", [], []),
    ]),
}


def norm(s): return unicodedata.normalize("NFKC", s).casefold()


def judge(out, expected, forbidden):
    if not ct.closed(out): return "loop"
    ans = norm(ct.answer_of(out))
    if any(norm(f) in ans for f in forbidden): return "✗"
    if not expected: return "-"
    return "✓" if all(norm(e) in ans for e in expected) else "✗"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with adapter_NN.pt / adapter.pt")
    ap.add_argument("--adapters", default="base,final", help="comma list: base, final, or cycle numbers")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--items", default=",".join(PROBES))
    ap.add_argument("--out", default=None, help="json file for the raw outputs")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke: args.max_new_tokens = 24
    device = "cuda" if ct.torch.cuda.is_available() else "cpu"
    r = ct.Runner(args.model, 2e-5, args.max_new_tokens, device)
    run = Path(args.run); items = args.items.split(",")
    prompts = [(k, kind, q, exp, forb) for k in items for kind, q, exp, forb in PROBES[k][2]]
    results = {}
    for a in args.adapters.split(","):
        if a == "base": pass
        else:
            path = run / ("adapter.pt" if a == "final" else f"adapter_{int(a):02d}.pt")
            if not path.exists(): print(f"SKIP {a}: {path} missing", flush=True); continue
            r.load_adapter(path)
        outs = r.generate([q for _, _, q, _, _ in prompts])
        rows = []
        for (k, kind, q, exp, forb), o in zip(prompts, outs):
            j = judge(o, exp, forb); ans = ct.answer_of(o).replace("\n", " ")[:110] if ct.closed(o) else "(no answer)"
            rows.append(dict(item=k, kind=kind, judge=j, answer=ans, full=o))
            print(f"PROBE adapter={a} {k} {kind:7s} {j} | {ans}", flush=True)
        for k in items:
            marks = "".join(x["judge"] if x["judge"] in "✓✗" else "·" for x in rows if x["item"] == k)
            print(f"ITEM adapter={a} {k} {marks}", flush=True)
        results[a] = rows
    if args.out: Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
