"""Hand-built test of the rationale target: train on {rationale-for-correct-answer}</think>{answer}
(full-sequence mask) with rehearsal every step. Does the answer flip while reasoning survives?"""
import os, mlx.core as mx, mlx.optimizers as optim
from mlx.utils import tree_map
import adaptible
from adaptible.revise import make_training_example, collate_training_examples, split_think, strip_think_tags
from adaptible._src.eval.harness import contains_key_terms, make_rehearsal_example
from adaptible.eval import generate_default_dataset
ds = {i.id: i for i in generate_default_dataset()}
wrong, probe = ds[os.environ.get("WRONG", "geo_002")], ds[os.environ.get("PROBE", "sci_001")]
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=float(os.environ.get("LR", "1e-5")))
tok = m._tokenizer
def raw(item): return m.generate_response(item.question, use_history=False, max_tokens=1024)
def ok(item, r): return contains_key_terms(strip_think_tags(r), item.key_terms)
reh_items = [ds[i] for i in ("sci_002", "hist_001", "math_001")]
base = {it.id: raw(it) for it in reh_items}
reh_items = [it for it in reh_items if ok(it, base[it.id])]
wrong_raw, probe_raw = raw(wrong), raw(probe)
print(f"BASE wrong_ok={ok(wrong, wrong_raw)} probe_ok={ok(probe, probe_raw)} rehearsal={[i.id for i in reh_items]}", flush=True)
# rationale conditioned on the correct answer
rat_raw = m.generate_response(f"{wrong.question}\n\nThe correct answer is: {wrong.correct_answer}\nReason it through step by step, then state the answer.", use_history=False, max_tokens=1024)
rationale, _ = split_think(rat_raw)
print(f"RATIONALE ({len(rationale.split())} words): {rationale[:160]!r}", flush=True)
target = f"{rationale}\n</think>\n\n{wrong.correct_answer}{tok.eos_token}"
corr = collate_training_examples([make_training_example([{"role":"user","content": wrong.question}], target, tok)], tok)
reh = [collate_training_examples([make_rehearsal_example(it, base[it.id], tok)], tok) for it in reh_items]
for cum in (4, 8, 12, 16):
    st = m.train_on_examples(corr, reh, loss_target=None, max_steps=4, rehearsal_weight=1.0)
    wr, pr = raw(wrong), raw(probe)
    print(f"STEPS={cum} train_loss={st.final_loss:.2f} reh_loss={st.rehearsal_final_loss:.2f} | wrong_fixed={ok(wrong, wr)} len={len(wr.split())} | probe_ok={ok(probe, pr)} len={len(pr.split())} | {strip_think_tags(wr)[:60]!r}", flush=True)
