"""Parse an mve_paraphrase log: per generation, phase, think length, closed?, looped?"""
import re, sys, collections
path = sys.argv[1] if len(sys.argv) > 1 else "outputs/runs/mve_paraphrase.log"
raw = open(path).read()
ansi = re.compile(r"\x1b\[[0-9;]*m")
raw = ansi.sub("", raw)
chunks = raw.split("Generating response for:\n")[1:]
gens = []
for c in chunks:
    q, _, body = c.partition("\n")
    body = re.split(r"\n(?:BASE|AFTER|RESULT|FINAL) ", body)[0]
    looped = "Loop detected!" in body
    body_clean = re.sub(r"⚠️  Loop detected!.*", "", body)
    body_clean = re.sub(r"\s*Generated \d+ tokens before loop detection\.", "", body_clean)
    closed = "</think>" in body_clean
    think = body_clean.split("</think>")[0] if closed else body_clean
    gens.append(dict(q=q.strip(), words=len(think.split()), closed=closed, looped=looped,
                     answer=body_clean.split("</think>")[-1].strip()[:60] if closed else "(no answer)"))
# phases: 5 rehearsal, 20 base, then training blocks, then 20 final
n = len(gens)
phase = ["reh"] * 5 + ["base"] * 20 + ["train"] * (n - 45) + ["final"] * 20
for g, p in zip(gens, phase): g["phase"] = p
print(f"{n} generations")
tab = collections.defaultdict(lambda: [0, 0, 0, 0])
for g in gens:
    t = tab[g["phase"]]; t[0] += 1; t[1] += g["words"]; t[2] += (not g["closed"]); t[3] += g["looped"]
print("phase  n  mean_think_words  unclosed  looped")
for p in ["reh", "base", "train", "final"]:
    t = tab[p]; print(f"{p:6} {t[0]:2}  {t[1]/max(t[0],1):8.0f}   {t[2]:3}   {t[3]:3}")
print()
for g in gens:
    if g["phase"] in ("base", "final", "train"):
        print(f"{g['phase']:5} {g['words']:5}w closed={int(g['closed'])} loop={int(g['looped'])} | {g['q'][:55]:55} | {g['answer']}")
