"""Iterative pivot training under a checksum-verified clean model.
Per item: deviation search on the clean model (cache: branch_paths2.json; geo_010's
earlier path is reused since it was found first, on the clean model). Then rounds:
  positions = every token of T that is not the current argmax (round 1: the pivot)
  train loss only there (AdamW, LR) until each reaches margin M
  greedy-generate; stop when correct
Report per round: steps, positions lifted, fixed, drops>0.5 over the other 39 items."""
import os, re, json, sqlite3, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from mlx_lm import stream_generate
import adaptible
from adaptible._src._llm import _detect_token_loop, _prepare_training_device
from adaptible.revise import make_training_example, strip_think_tags
from adaptible._src.eval.harness import contains_key_terms
from adaptible.eval import generate_default_dataset

LR = float(os.environ.get("LR", "2e-5")); CAP = int(os.environ.get("CAP", "10")); M = float(os.environ.get("M", "2.0"))
ROUNDS = int(os.environ.get("ROUNDS", "6")); NDEV = int(os.environ.get("NDEV", "12"))
ITEMS = os.environ.get("ITEMS", "geo_010,sci_017,geo_001,geo_004").split(",")
PATHS = "outputs/runs/branch_paths2.json"
ds = {i.id: i for i in generate_default_dataset()}
c = sqlite3.connect("outputs/adaptible.db")
base_raw = {cid: raw for cid, raw in c.execute("select e.canonical_id, r.response_raw from responses r join examples e on e.id=r.example_id where r.experiment_id=16 and r.phase='baseline'")}
items = [ds[i] for i in base_raw]
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=LR)
tok = m._tokenizer; EOS = tok.eos_token_id
def msgs(it): return [{"role": "user", "content": it.question}]
def prompt_ids(it): return list(m.apply_chat_template(m._messages_for_prompt(it.question, False)))
def ok(it, text): return contains_key_terms(strip_think_tags(text or ""), it.key_terms)
def gen(it): return m.generate_response(it.question, use_history=False)
def greedy_from(it, forced):
    toks = []
    for r in stream_generate(m._model, tok, prompt=prompt_ids(it) + forced, max_tokens=max(16, 1200 - len(forced))):
        toks.append(r.token)
        if _detect_token_loop(toks, m._loop_detection_sequence_length, m._loop_detection_max_repetitions): break
    T = forced + toks; return T, tok.decode(T)
def logits_path(it, T):
    seq = mx.array(prompt_ids(it) + T, dtype=mx.int32); P = len(prompt_ids(it))
    return m._model(seq[:-1][None])[0][P - 1:]
def decisions(it, T):
    lg = logits_path(it, T); Ta = mx.array(T)
    tgt = mx.take_along_axis(lg, Ta[:, None], axis=1)[:, 0]
    masked = mx.put_along_axis(lg, Ta[:, None], mx.array(-1e9), axis=1)
    out = ((tgt - masked.max(axis=1)).tolist(), masked.argmax(axis=1).tolist()); mx.clear_cache(); return out
def deviation_search(it):
    G, text = greedy_from(it, [])
    if ok(it, text): return None, "already-correct"
    margin, runner = decisions(it, G)
    for j in sorted(range(1, len(G)), key=lambda j: margin[j])[:NDEV]:
        T, text = greedy_from(it, G[:j] + [runner[j]])
        if ok(it, text): return T, f"deviation at {j}/{len(G)} margin {margin[j]:.2f} {tok.decode([G[j]])!r}->{tok.decode([runner[j]])!r}"
    return None, f"no hit in {NDEV} deviations"
def nll(ex, use_stop=False):
    per = nn.losses.cross_entropy(m._model(ex.input[None])[0], ex.label, reduction="none")
    mask = ex.stop_mask if use_stop else ex.mask
    return float((per * mask).sum()), float(mask.sum())
drift_ex = {it.id: make_training_example(msgs(it), base_raw[it.id] + tok.eos_token, tok) for it in items}
def think_prefix(raw): return raw[: raw.index("</think>") + len("</think>\n\n")] if "</think>" in raw else raw + "\n</think>\n\n"
ans_ex = {it.id: make_training_example(msgs(it), think_prefix(base_raw[it.id]) + it.correct_answer, tok, stop_text=it.correct_answer) for it in items}
def measure():
    out = {}
    for it in items:
        s, n = nll(drift_ex[it.id]); a, _ = nll(ans_ex[it.id], use_stop=True); out[it.id] = (s / n, -a)
    mx.clear_cache(); return out
def checksum(): return float(sum(mx.abs(a).sum().item() for _, a in tree_flatten(m._model.trainable_parameters())))
snapshot = tree_map(lambda x: x + 0, m._model.trainable_parameters()); base_sum = checksum()
def restore():
    m._model.update(tree_map(lambda x: x + 0, snapshot)); mx.eval(m._model.parameters())
    assert abs(checksum() - base_sum) < 1e-3 * max(1.0, base_sum), "restore failed"
def train_positions(it, T, positions):
    seq = mx.array(prompt_ids(it) + T + [EOS], dtype=mx.int32); P = len(prompt_ids(it))
    w = [0.0] * len(seq)
    for j in positions: w[P + j] = 1.0
    w = mx.array(w, dtype=mx.float32)[1:]; inputs, labels = seq[:-1], seq[1:]
    def loss_fn(model, inputs, labels, w):
        per = nn.losses.cross_entropy(model(inputs[None])[0], labels, reduction="none"); return (per * w).sum() / w.sum()
    lg = nn.value_and_grad(m._model, loss_fn); _prepare_training_device(False)
    for step in range(1, CAP + 1):
        loss, grads = lg(m._model, inputs, labels, w)
        m._optimizer.update(m._model, grads); mx.eval(m._model.parameters(), m._optimizer.state)
        mg, _ = decisions(it, T)
        if min(mg[j] for j in positions) >= M: break
    mx.clear_cache(); return step

base = measure(); print(f"\nBASE checksum={base_sum:.3f}", flush=True)
old = json.load(open("outputs/runs/branch_paths.json"))
cache = json.load(open(PATHS)) if os.path.exists(PATHS) else {}
if "geo_010" in old and "geo_010" not in cache: cache["geo_010"] = old["geo_010"]
for cid in ITEMS:
    it = ds[cid]; restore()
    if cid in cache: T, how = cache[cid]["T"], cache[cid]["how"] + " (cached)"
    else:
        T, how = deviation_search(it)
        if T is not None: cache[cid] = {"T": T, "how": how}; json.dump(cache, open(PATHS, "w"))
    if T is None: print(f"\nSKIP {cid}: {how}", flush=True); continue
    mg, _ = decisions(it, T); neg = [j for j, x in enumerate(mg) if x < 0]
    print(f"\nPATH {cid}: {how}; len {len(T)}; negative-margin positions on clean model: {len(neg)} {neg[:6]}", flush=True)
    if len(neg) > 20: print(f"\nSKIP {cid}: path not the clean model's (too many disagreements)", flush=True); continue
    m._optimizer = optim.AdamW(learning_rate=LR); total = 0
    for r in range(1, ROUNDS + 1):
        mg, _ = decisions(it, T); positions = [j for j, x in enumerate(mg) if x < M]
        positions = [j for j in positions if mg[j] < 0] or positions[:1]
        steps = train_positions(it, T, positions); total += steps
        out = gen(it); fixed = ok(it, out)
        cur = measure(); d_ans = {i: cur[i][1] - base[i][1] for i in cur}; d_drift = [cur[i][0] - base[i][0] for i in cur]
        print(f"\nROUND {cid} {r}: lifted {len(positions)} at {positions[:6]} steps={steps} total={total} fixed={fixed} | drift max={max(d_drift):+.4f} | drops>0.5 elsewhere {sum(v < -0.5 for i, v in d_ans.items() if i != cid)}/{len(d_ans)-1} worst {min((round(v,2), i) for i, v in d_ans.items() if i != cid)} | {strip_think_tags(out)[:60]!r}", flush=True)
        if fixed: break
restore()
