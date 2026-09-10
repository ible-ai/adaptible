"""Probe: joint objective (correction + rehearsal every step). Does the corrected item keep
its reasoning, and do rehearsal/unrelated items survive?  Env: LORA_RANK/LORA_LAYERS."""
import os, sys, mlx.core as mx, mlx.optimizers as optim
from mlx.utils import tree_map
import adaptible
from adaptible.revise import make_collated_training_example, strip_think_tags, collate_training_examples
from adaptible._src.eval.harness import contains_key_terms, make_rehearsal_example
from adaptible.eval import generate_default_dataset
ds = {i.id: i for i in generate_default_dataset()}
wrong, probe = ds["geo_002"], ds["sci_001"]
reh_ids = ["geo_001", "sci_002", "hist_001", "math_001"]
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=int(os.environ.get("LORA_LAYERS", 8)),
        lora_parameters={"rank": int(os.environ.get("LORA_RANK", 8)), "dropout": 0.0, "scale": 10.0}, learning_rate=float(os.environ.get("LR", "1e-5")))
def raw(item): return m.generate_response(item.question, use_history=False, max_tokens=1024)
def ok(item, r): return contains_key_terms(strip_think_tags(r), item.key_terms)
base = {i: raw(ds[i]) for i in reh_ids}
reh_items = [ds[i] for i in reh_ids if ok(ds[i], base[i])][:3]
print("rehearsal pool (baseline-correct):", [i.id for i in reh_items], flush=True)
wrong_raw = raw(wrong); probe_raw = raw(probe)
print(f"BASE wrong_ok={ok(wrong, wrong_raw)} probe_ok={ok(probe, probe_raw)} len={len(wrong_raw.split())}/{len(probe_raw.split())}", flush=True)
inter = [adaptible.InteractionHistory(idx=0, user_input=wrong.question, llm_response=wrong_raw)]
corr = make_collated_training_example(f"[[0]] {wrong.correct_answer} [[/0]]", inter, m._tokenizer, think_mode="baseline")
reh = [collate_training_examples([make_rehearsal_example(it, base[it.id], m._tokenizer)], m._tokenizer) for it in reh_items]
snap = tree_map(lambda a: mx.array(a), m._model.trainable_parameters())
for w in [float(x) for x in (sys.argv[1:] or ["1.0", "3.0"])]:
    m._model.update(tree_map(lambda a: mx.array(a), snap)); m._optimizer = optim.AdamW(learning_rate=m._optimizer.learning_rate)
    st = m.train_on_examples(corr, reh, loss_target=0.6, max_steps=20, rehearsal_weight=w)
    wr, pr = raw(wrong), raw(probe); rr = [raw(it) for it in reh_items]
    print(f"WEIGHT={w} steps={st.steps} loss {st.initial_loss:.2f}->{st.final_loss:.2f} reh_loss={st.rehearsal_final_loss:.2f} | wrong_fixed={ok(wrong, wr)} len={len(wr.split())} | probe_ok={ok(probe, pr)} len={len(pr.split())} | rehearsal_ok={[ok(it, r) for it, r in zip(reh_items, rr)]} lens={[len(r.split()) for r in rr]} | {strip_think_tags(wr)[:50]!r}", flush=True)
