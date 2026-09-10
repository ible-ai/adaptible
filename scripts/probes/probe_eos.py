"""Leakage probe: does supervising <eos> right after the answer teach the model
to stop after that phrase *anywhere*? Train WRONG (default sci_017: nearest star
= Sun) with the harness recipe, then generate neighbouring items, under two
conditions: eos in the loss (as the harness does) vs eos masked out of both the
train mask and the stop mask. LoRA weights are restored between conditions."""
import os, dataclasses, mlx.core as mx, mlx.optimizers as optim
import adaptible
from adaptible.revise import collate_training_examples, strip_think_tags
from adaptible._src.eval.harness import (
    contains_key_terms, make_rehearsal_example, _build_training_example, VERIFY_LOSS_FLOOR)
from adaptible.eval import generate_default_dataset

ds = {i.id: i for i in generate_default_dataset()}
wrong = ds[os.environ.get("WRONG", "sci_017")]
probes = [ds[i] for i in os.environ.get("PROBES", "sci_018,sci_012,sci_011,geo_008,sci_005").split(",")]
reh_items = [ds[i] for i in os.environ.get("REHEARSAL", "sci_002,geo_012,sci_015").split(",")]
LR = float(os.environ.get("LR", "2e-5"))
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8,
                          lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=LR)
tok = m._tokenizer
def raw(item): return m.generate_response(item.question, use_history=False)
def ok(item, r): return contains_key_terms(strip_think_tags(r or ""), item.key_terms)
def ntok(r): return len(tok.encode(r or ""))
def show(r): return strip_think_tags(r or "")[:70].replace("\n", " ")

base = {it.id: raw(it) for it in [wrong, *probes, *reh_items]}
reh_items = [it for it in reh_items if ok(it, base[it.id])]
print(f"BASE {wrong.id} ok={ok(wrong, base[wrong.id])} | rehearsal={[i.id for i in reh_items]}", flush=True)
for p in probes:
    print(f"BASE {p.id} ok={ok(p, base[p.id])} tok={ntok(base[p.id])} | {show(base[p.id])!r}", flush=True)

built = _build_training_example(m, wrong, base[wrong.id], "ground_truth", think_mode="rationale")
ex = built.example
print(f"RATIONALE {built.rationale_tokens} tok; target tail ids {ex.label[-4:].tolist()} eos={tok.eos_token_id}", flush=True)
reh = [collate_training_examples([make_rehearsal_example(it, base[it.id], tok)], tok) for it in reh_items]
snapshot = m._model.trainable_parameters()

def mask_eos(e):
    z = mx.zeros((1,), dtype=e.mask.dtype)
    mask = mx.concatenate([e.mask[:-1], z])
    stop = None if e.stop_mask is None else mx.concatenate([e.stop_mask[:-1], mx.zeros((1,), dtype=e.stop_mask.dtype)])
    return dataclasses.replace(e, mask=mask, stop_mask=stop)

for cond in os.environ.get("CONDS", "eos,noeos").split(","):
    m._model.update(snapshot); m._optimizer = optim.AdamW(learning_rate=LR)
    e = ex if cond == "eos" else mask_eos(ex)
    corr = collate_training_examples([e], tok)
    target, total, calls = 0.6, 0, []
    st = m.train_on_examples(corr, reh, loss_target=target, max_steps=30, rehearsal_weight=1.0, rehearsal_margin=0.05)
    total += st.steps; calls.append(st)
    checks = 0
    while True:
        checks += 1
        r = raw(wrong); fixed = ok(wrong, r)
        if fixed or total >= 30: break
        last = st.final_loss
        if target <= VERIFY_LOSS_FLOOR and last <= VERIFY_LOSS_FLOOR: break
        target = max(VERIFY_LOSS_FLOOR, min(target / 2, last / 2))
        st = m.train_on_examples(corr, reh, loss_target=target, max_steps=min(4, 30 - total), rehearsal_weight=1.0, rehearsal_margin=0.05)
        total += st.steps; calls.append(st)
    print(f"COND {cond}: steps={total} checks={checks} fixed={fixed} final_answer_loss={st.final_loss:.2f} "
          f"wrong tok={ntok(r)} | {show(r)!r}", flush=True)
    regressed = 0
    for p in probes:
        r = raw(p); o = ok(p, r)
        regressed += (ok(p, base[p.id]) and not o)
        print(f"  {cond} {p.id} ok={o} tok={ntok(r)} | {show(r)!r}", flush=True)
    print(f"COND {cond}: regressed {regressed}/{sum(ok(p, base[p.id]) for p in probes)}", flush=True)
