"""Same as mve_paraphrase.py with ONE change: the think-block target is a short
grounded note (below) instead of the model's own ~600-word hinted rationale.
Question: do paraphrases stop looping and does the fact transfer?

Original docstring: Minimum viable experiment: after training on a wrong trace with the committed
recipe, does the model know the fact (paraphrases) and does it hold after the other
items are trained? Sequential, no weight restore. Output: one RESULT line per item."""
import os, mlx.core as mx, adaptible
from adaptible.revise import collate_training_examples, strip_think_tags, make_revision_training_example
from adaptible import InteractionHistory
from adaptible._src.eval.harness import contains_key_terms, make_rehearsal_example, _build_training_example, VERIFY_LOSS_FLOOR
from adaptible.eval import generate_default_dataset

PARA = {
 "geo_010": ["Which city is the capital of Morocco?", "Morocco's seat of government is in which city?", "Name the capital city of the Kingdom of Morocco."],
 "geo_004": ["Which city serves as Turkey's capital?", "Where is the seat of the Turkish government located?", "Name the capital city of Turkey."],
 "geo_001": ["Which city is Australia's capital?", "Where is the seat of Australia's federal government?", "Name the capital city of Australia."],
 "geo_013": ["Which city is the capital of the Philippines?", "Where is the seat of government of the Philippines?", "Name the capital city of the Philippines."],
 "sci_017": ["Which star is closest to our planet?", "What is the closest star to Earth?", "Name the star nearest to the Earth."],
}
NOTE = {
 "geo_010": "Casablanca is Morocco's largest city and its commercial hub, but it is not the capital. The government and the royal palace are in Rabat, on the Atlantic coast north of Casablanca. The capital of Morocco is Rabat.",
 "geo_004": "Istanbul is Turkey's largest city, but when the republic was founded in 1923 the capital was moved to Ankara, in central Anatolia. The parliament and the ministries are in Ankara. The capital of Turkey is Ankara.",
 "geo_001": "Sydney and Melbourne are Australia's largest cities, and the capital was built as a compromise between them. Parliament sits in Canberra, in the Australian Capital Territory. The capital of Australia is Canberra.",
 "geo_013": "Quezon City was the capital of the Philippines from 1948 to 1976, but the capital was then moved back to Manila, where the presidential palace and the government are. The capital of the Philippines is Manila.",
 "sci_017": "Proxima Centauri is the nearest star outside the solar system, about 4.2 light-years away, but the Sun is a star too and is only about 8 light-minutes away. The nearest star to Earth is the Sun.",
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
    inter = [InteractionHistory(idx=0, user_input=it.question, llm_response=base[cid][0], reviewed=False, timestamp=0.0)]
    ex = make_revision_training_example(f"[[0]] {it.correct_answer} [[/0]]", inter, tok, think_mode="rationale", rationale=NOTE[cid])
    corr = collate_training_examples([ex], tok)
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
