"""Minimum viable experiment: after training on a wrong trace with the committed
recipe, does the model know the fact (paraphrases) and does it hold after the other
items are trained? Sequential, no weight restore. Output: one RESULT line per item."""
import os, mlx.core as mx, adaptible
from adaptible.revise import collate_training_examples, strip_think_tags
from adaptible._src.eval.harness import contains_key_terms, make_rehearsal_example, _build_training_example, VERIFY_LOSS_FLOOR
from adaptible.eval import generate_default_dataset

PARA = {
 "geo_010": ["Which city is the capital of Morocco?", "Morocco's seat of government is in which city?", "Name the capital city of the Kingdom of Morocco."],
 "geo_004": ["Which city serves as Turkey's capital?", "Where is the seat of the Turkish government located?", "Name the capital city of Turkey."],
 "geo_001": ["Which city is Australia's capital?", "Where is the seat of Australia's federal government?", "Name the capital city of Australia."],
 "geo_013": ["Which city is the capital of the Philippines?", "Where is the seat of government of the Philippines?", "Name the capital city of the Philippines."],
 "sci_017": ["Which star is closest to our planet?", "What is the closest star to Earth?", "Name the star nearest to the Earth."],
}
REH = ["sci_002", "geo_012", "sci_015", "sci_004", "geo_005"]
LR = float(os.environ.get("LR", "2e-5")); CAP = 30; TARGET = 0.6; VSTEPS = 4
ds = {i.id: i for i in generate_default_dataset()}
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=LR)
tok = m._tokenizer
def gen(q): return m.generate_response(q, use_history=False)
def ok(it, r): return contains_key_terms(strip_think_tags(r or ""), it.key_terms)
def marks(it, outs): return "".join("✓" if ok(it, r) else "✗" for r in outs)
def prompts(cid): return [ds[cid].question] + PARA[cid]

reh_base = {r: gen(ds[r].question) for r in REH}
reh_items = [ds[r] for r in REH if ok(ds[r], reh_base[r])][:3]
base = {cid: [gen(q) for q in prompts(cid)] for cid in PARA}
for cid in PARA: print(f"\nBASE {cid} {marks(ds[cid], base[cid])} | {[strip_think_tags(r)[:40] for r in base[cid]]}", flush=True)

after, train_info = {}, {}
for cid in PARA:
    it = ds[cid]
    if ok(it, base[cid][0]):
        train_info[cid] = "skipped (baseline correct)"; after[cid] = base[cid]; continue
    built = _build_training_example(m, it, base[cid][0], "ground_truth", think_mode="rationale")
    corr = collate_training_examples([built.example], tok)
    reh = [collate_training_examples([make_rehearsal_example(r, reh_base[r.id], tok)], tok) for r in reh_items]
    target, total, checks = TARGET, 0, 0
    st = m.train_on_examples(corr, reh, loss_target=target, max_steps=CAP, rehearsal_weight=1.0, rehearsal_margin=0.05); total += st.steps
    while True:
        checks += 1; out = gen(it.question); fixed = ok(it, out)
        if fixed or total >= CAP: break
        if target <= VERIFY_LOSS_FLOOR and st.final_loss <= VERIFY_LOSS_FLOOR: break
        target = max(VERIFY_LOSS_FLOOR, min(target / 2, st.final_loss / 2))
        st = m.train_on_examples(corr, reh, loss_target=target, max_steps=min(VSTEPS, CAP - total), rehearsal_weight=1.0, rehearsal_margin=0.05); total += st.steps
    mx.clear_cache()
    train_info[cid] = f"{total} steps, {checks} checks, verified={fixed}"
    after[cid] = [out] + [gen(q) for q in PARA[cid]]
    print(f"\nAFTER {cid} {marks(it, after[cid])} ({train_info[cid]}) | {[strip_think_tags(r)[:40] for r in after[cid]]}", flush=True)

final = {cid: [gen(q) for q in prompts(cid)] for cid in PARA}
print("\n\nRESULT item | baseline | after own training | after all 5 | training  (marks: original + 3 paraphrases)", flush=True)
for cid in PARA:
    print(f"RESULT {cid} | {marks(ds[cid], base[cid])} | {marks(ds[cid], after[cid])} | {marks(ds[cid], final[cid])} | {train_info[cid]}", flush=True)
    print(f"FINAL {cid} {[strip_think_tags(r)[:50] for r in final[cid]]}", flush=True)
