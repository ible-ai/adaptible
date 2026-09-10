"""Interference probe: train one wrong item with the harness recipe (loss target
0.6 + halving verify rounds), then generate N baseline-correct neighbours.
Conditions are "name:lr:margin:k" (rehearsal hinge margin, rehearsal k drawn from
REHEARSAL pool). LoRA weights and optimizer are restored between conditions."""
import os, random, mlx.optimizers as optim
import adaptible
from adaptible.revise import collate_training_examples, strip_think_tags
from adaptible._src.eval.harness import (
    contains_key_terms, make_rehearsal_example, _build_training_example, VERIFY_LOSS_FLOOR)
from adaptible.eval import generate_default_dataset

ds = {i.id: i for i in generate_default_dataset()}
wrong = ds[os.environ.get("WRONG", "sci_017")]
probes = [ds[i] for i in os.environ.get("PROBES", "sci_018,sci_012,sci_011,geo_008,sci_005,geo_011,sci_007,sci_013").split(",")]
pool = [ds[i] for i in os.environ.get("REHEARSAL", "sci_002,geo_012,sci_015,sci_004,geo_005,sci_010").split(",")]
conds = [c.split(":") for c in os.environ.get("CONDS", "lr2e-5:2e-5:0.05:3,lr1e-5:1e-5:0.05:3,lr5e-6:5e-6:0.05:3,m0:1e-5:0:3").split(",")]
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8,
                          lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=2e-5)
tok = m._tokenizer
def raw(item): return m.generate_response(item.question, use_history=False)
def ok(item, r): return contains_key_terms(strip_think_tags(r or ""), item.key_terms)
def ntok(r): return len(tok.encode(r or ""))
def show(r): return strip_think_tags(r or "")[:60].replace("\n", " ")

base = {it.id: raw(it) for it in [wrong, *probes, *pool]}
pool = [it for it in pool if ok(it, base[it.id])]
probes = [p for p in probes if ok(p, base[p.id])]
print(f"\nBASE {wrong.id} ok={ok(wrong, base[wrong.id])} | pool={[i.id for i in pool]} probes_ok={[p.id for p in probes]}", flush=True)
built = _build_training_example(m, wrong, base[wrong.id], "ground_truth", think_mode="rationale")
corr = collate_training_examples([built.example], tok)
reh_all = {it.id: collate_training_examples([make_rehearsal_example(it, base[it.id], tok)], tok) for it in pool}
snapshot = m._model.trainable_parameters()

for name, lr, margin, k in conds:
    lr, margin, k = float(lr), float(margin), int(k)
    m._model.update(snapshot); m._optimizer = optim.AdamW(learning_rate=lr)
    rng = random.Random(0)
    def reh(): return [reh_all[i.id] for i in rng.sample(pool, min(k, len(pool)))]
    target, total = 0.6, 0
    st = m.train_on_examples(corr, reh(), loss_target=target, max_steps=30, rehearsal_weight=1.0, rehearsal_margin=margin)
    total += st.steps; checks = 0
    while True:
        checks += 1
        r = raw(wrong); fixed = ok(wrong, r)
        if fixed or total >= 30: break
        if target <= VERIFY_LOSS_FLOOR and st.final_loss <= VERIFY_LOSS_FLOOR: break
        target = max(VERIFY_LOSS_FLOOR, min(target / 2, st.final_loss / 2))
        st = m.train_on_examples(corr, reh(), loss_target=target, max_steps=min(4, 30 - total), rehearsal_weight=1.0, rehearsal_margin=margin)
        total += st.steps
    print(f"\nCOND {name}: steps={total} checks={checks} fixed={fixed} answer_loss={st.final_loss:.2f} reh_active={st.rehearsal_active_steps} wrong tok={ntok(r)} | {show(r)!r}", flush=True)
    regressed = []
    for p in probes:
        r = raw(p); o = ok(p, r)
        if not o: regressed.append(p.id)
        print(f"\nPROBE {name} {p.id} ok={o} tok={ntok(r)} | {show(r)!r}", flush=True)
    print(f"\nCOND {name}: regressed {len(regressed)}/{len(probes)} {regressed}", flush=True)
