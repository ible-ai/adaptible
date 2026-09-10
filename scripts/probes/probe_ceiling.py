"""Reachability ceiling: for baseline-wrong items of the 40-item subset (db, exp 16),
sample K trajectories at temperature, plain; items with no correct sample are retried
with a one-line reference note in the prompt (STaR-style rationalization upper bound).
Saves every correct trajectory (token ids) to outputs/runs/ceiling.json for the
branch-point probe."""
import os, json, sqlite3, unicodedata, mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
import adaptible
from adaptible._src._llm import _detect_token_loop
from adaptible.revise import strip_think_tags
from adaptible._src.eval.harness import contains_key_terms
from adaptible.eval import generate_default_dataset

K = int(os.environ.get("K", "8")); TEMP = float(os.environ.get("TEMP", "0.7"))
LIMIT = int(os.environ.get("LIMIT", "20")); MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
ds = {i.id: i for i in generate_default_dataset()}
c = sqlite3.connect("outputs/adaptible.db")
rows = c.execute("select e.canonical_id, r.response_text from responses r join examples e on e.id=r.example_id "
                 "where r.experiment_id=16 and r.phase='baseline'").fetchall()
wrong = [ds[cid] for cid, txt in rows if not contains_key_terms(strip_think_tags(txt or ""), ds[cid].key_terms)][:LIMIT]
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0})
tok = m._tokenizer
sampler = make_sampler(temp=TEMP)

def sample(prompt_text, seed):
    mx.random.seed(seed)
    ids = m.apply_chat_template(m._messages_for_prompt(prompt_text, False))
    toks, text = [], []
    for r in stream_generate(m._model, tok, prompt=ids, max_tokens=MAX_TOKENS, sampler=sampler):
        toks.append(r.token); text.append(r.text)
        if _detect_token_loop(toks, m._loop_detection_sequence_length, m._loop_detection_max_repetitions):
            break
    return toks, "".join(text)

def ok(it, text): return contains_key_terms(strip_think_tags(text), it.key_terms)
def hinted(it): return f"{it.question}\n\n(Reference note: the correct answer is {it.correct_answer}.)"

out = {}
n_plain = n_hint = 0
for it in wrong:
    rec = {"question": it.question, "answer": it.correct_answer, "plain": [], "hinted": []}
    for s in range(K):
        toks, text = sample(it.question, s)
        if ok(it, text): rec["plain"].append({"tokens": toks, "text": text})
    if rec["plain"]:
        n_plain += 1
    else:
        for s in range(K):
            toks, text = sample(hinted(it), 100 + s)
            if ok(it, text): rec["hinted"].append({"tokens": toks, "text": text})
        n_hint += bool(rec["hinted"])
    print(f"\nCEIL {it.id}: plain {len(rec['plain'])}/{K} hinted {len(rec['hinted'])}/{K if not rec['plain'] else 0} | {it.correct_answer}", flush=True)
    out[it.id] = rec
    json.dump(out, open("outputs/runs/ceiling.json", "w"))
print(f"\nCEIL SUMMARY: {len(wrong)} wrong items; reachable plain {n_plain}, hinted-only {n_hint}, unreachable {len(wrong)-n_plain-n_hint}", flush=True)
