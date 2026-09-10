"""Noise floor + continuous drift metric.

For the 40-item screen subset (baseline outputs read from the db, experiment 16),
measure under each condition, restoring LoRA between them:
  drift    = mean per-token NLL of the item's own baseline output (self-distillation loss)
  ans_lp   = log P(correct answer | question + the baseline's think prefix)
plus greedy generation on PROBES (known baseline-correct items).
Conditions: base (no training), floor (1 step at lr 1e-7 on WRONG), recipe (lr 2e-5,
loss target 0.6, cap 30, on WRONG)."""
import os, sqlite3, math, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import adaptible
from adaptible.revise import make_training_example, collate_training_examples, strip_think_tags
from adaptible._src.eval.harness import contains_key_terms, _build_training_example
from adaptible.eval import generate_default_dataset

ds = {i.id: i for i in generate_default_dataset()}
wrong = ds[os.environ.get("WRONG", "sci_017")]
probes = [ds[i] for i in os.environ.get("PROBES", "sci_018,sci_012,sci_011,geo_008,sci_005,geo_011,sci_007,sci_013").split(",")]
c = sqlite3.connect("outputs/adaptible.db")
base_raw = {cid: raw for cid, raw in c.execute(
    "select e.canonical_id, r.response_raw from responses r join examples e on e.id=r.example_id "
    "where r.experiment_id=16 and r.phase='baseline'")}
items = [ds[i] for i in base_raw]
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8,
                          lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=2e-5)
tok = m._tokenizer
eos = tok.eos_token
def msgs(it): return [{"role": "user", "content": it.question}]
def nll(ex, use_stop=False):
    logits = m._model(ex.input[None])[0]
    per = nn.losses.cross_entropy(logits, ex.label, reduction="none")
    mask = ex.stop_mask if use_stop else ex.mask
    return float((per * mask).sum()), float(mask.sum())
drift_ex = {it.id: make_training_example(msgs(it), base_raw[it.id] + eos, tok) for it in items}
def think_prefix(raw):
    return raw[: raw.index("</think>") + len("</think>\n\n")] if "</think>" in raw else raw + "\n</think>\n\n"
ans_ex = {it.id: make_training_example(msgs(it), think_prefix(base_raw[it.id]) + it.correct_answer, tok,
                                       stop_text=it.correct_answer) for it in items}
def measure():
    out = {}
    for it in items:
        s, n = nll(drift_ex[it.id]); a, _ = nll(ans_ex[it.id], use_stop=True)
        out[it.id] = (s / n, -a)
    mx.clear_cache()
    return out
def ok(it, r): return contains_key_terms(strip_think_tags(r or ""), it.key_terms)
def gen(it): return m.generate_response(it.question, use_history=False)

base = measure()
base_gen = {p.id: gen(p) for p in probes}
print(f"\nBASE drift={sum(v[0] for v in base.values())/len(base):.4f} probes_ok={[p.id for p in probes if ok(p, base_gen[p.id])]}", flush=True)
built = _build_training_example(m, wrong, base_raw[wrong.id], "ground_truth", think_mode="rationale")
corr = collate_training_examples([built.example], tok)
snapshot = m._model.trainable_parameters()

for name, lr, target, steps in [("floor", 1e-7, None, 1), ("recipe", 2e-5, 0.6, 30)]:
    m._model.update(snapshot); m._optimizer = optim.AdamW(learning_rate=lr)
    st = m.train_on_example(corr, iterations=steps, loss_target=target, max_steps=steps)
    cur = measure()
    d_drift = [cur[i][0] - base[i][0] for i in cur]
    d_ans = {i: cur[i][1] - base[i][1] for i in cur}
    worse = sorted(d_ans.items(), key=lambda kv: kv[1])[:6]
    flips = [p.id for p in probes if ok(p, base_gen[p.id]) and not ok(p, gen(p))]
    print(f"\nCOND {name}: steps={st.steps} answer_loss={st.final_loss:.2f} | mean drift delta={sum(d_drift)/len(d_drift):+.4f} "
          f"max={max(d_drift):+.4f} | answer logp: {wrong.id} {d_ans[wrong.id]:+.2f}; items with drop>0.5: "
          f"{sum(v < -0.5 for v in d_ans.values())}/{len(d_ans)}; worst {[(i, round(v,2)) for i, v in worse]} | gen flips {flips}", flush=True)
