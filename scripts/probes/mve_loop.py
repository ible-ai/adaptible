"""Iterative repair loop, minimum viable: for each of the 5 MVE items, try up to K
candidate training targets (the model's OWN correct samples: think + answer sentence,
from a reference-note prompt at T=0.7; stored ones in outputs/runs/ceiling.json reused).
Train with the committed recipe, generate the original + 3 paraphrases, judge, and KEEP
the weights only if the 4-prompt score rises and the loop count does not; else restore
(checksum-verified) and try the next candidate. Same tables as mve_paraphrase.py."""
import os, re, json, mlx.core as mx, mlx.optimizers as optim, adaptible
from mlx.utils import tree_map, tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from adaptible import InteractionHistory
from adaptible._src._llm import _detect_token_loop
from adaptible.revise import collate_training_examples, strip_think_tags, make_revision_training_example
from adaptible._src.eval.harness import contains_key_terms, make_rehearsal_example, VERIFY_LOSS_FLOOR
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
K = int(os.environ.get("K", "3")); TEMP = 0.7; MAX_SAMPLES = 8
ds = {i.id: i for i in generate_default_dataset()}
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=LR)
tok = m._tokenizer
sampler = make_sampler(temp=TEMP)

def gen(q): return m.generate_response(q, use_history=False) or ""
def closed(r): return "</think>" in r
def answer_of(r): return r.split("</think>")[-1].strip()
def ok(it, r): return closed(r) and contains_key_terms(answer_of(r), it.key_terms)
def marks(it, outs): return "".join("✓" if ok(it, r) else "✗" for r in outs)
def loops(outs): return sum(not closed(r) for r in outs)
def prompts(cid): return [ds[cid].question] + PARA[cid]
def score(cid):
    outs = [gen(q) for q in prompts(cid)]; it = ds[cid]
    return sum(ok(it, r) for r in outs), loops(outs), outs

def checksum(): return float(sum(mx.abs(a).sum().item() for _, a in tree_flatten(m._model.trainable_parameters())))
def snapshot(): return tree_map(lambda x: x + 0, m._model.trainable_parameters()), checksum()
def restore(snap):
    params, s = snap
    m._model.update(tree_map(lambda x: x + 0, params)); mx.eval(m._model.parameters())
    m._optimizer = optim.AdamW(learning_rate=LR)
    assert abs(checksum() - s) < 1e-3 * max(1.0, s), "restore failed"

def hinted(it): return f"{it.question}\n\n(Reference note: the correct answer is {it.correct_answer}.)"
def sample(prompt_text, seed):
    mx.random.seed(seed)
    ids = m.apply_chat_template(m._messages_for_prompt(prompt_text, False))
    toks, text = [], []
    for r in stream_generate(m._model, tok, prompt=ids, max_tokens=1024, sampler=sampler):
        toks.append(r.token); text.append(r.text)
        if _detect_token_loop(toks, m._loop_detection_sequence_length, m._loop_detection_max_repetitions): break
    return "".join(text)
def clean(it, text):
    """Closed think, key term in the first sentence of the answer, answer not a loop."""
    if not closed(text): return False
    first = re.split(r"(?<=[.!?])\s", answer_of(text), 1)[0]
    return contains_key_terms(first, it.key_terms) and len(first) < 300
def candidates(cid):
    it = ds[cid]; out = []
    stored = json.load(open("outputs/runs/ceiling.json")).get(cid, {})
    for s in stored.get("hinted", []) + stored.get("plain", []):
        if clean(it, s["text"]) and len(out) < K: out.append(s["text"])
    seed = 1000
    while len(out) < K and seed < 1000 + MAX_SAMPLES:
        t = sample(hinted(it), seed); seed += 1
        if clean(it, t): out.append(t)
    return out
def example_from(it, text):
    think = text.split("</think>")[0].strip()
    ans = " ".join(re.split(r"(?<=[.!?])\s", answer_of(text))[:2]).replace("\n", " ").strip()
    inter = [InteractionHistory(idx=0, user_input=it.question, llm_response="", reviewed=False, timestamp=0.0)]
    return make_revision_training_example(f"[[0]] {ans} [[/0]]", inter, tok, think_mode="rationale", rationale=think), ans

reh_base = {r: gen(ds[r].question) for r in REH}
reh_items = [ds[r] for r in REH if ok(ds[r], reh_base[r])][:3]
reh = [collate_training_examples([make_rehearsal_example(r, reh_base[r.id], tok)], tok) for r in reh_items]
base = {cid: score(cid) for cid in PARA}
for cid in PARA: print(f"\nBASE {cid} {marks(ds[cid], base[cid][2])} loops={base[cid][1]}", flush=True)

after, info = {}, {}
for cid in PARA:
    it = ds[cid]; best_n, best_l, best_outs = base[cid]; snap = snapshot(); accepted = 0; tried = 0
    if best_n == 4: info[cid] = "skipped"; after[cid] = best_outs; continue
    for k, text in enumerate(candidates(cid)):
        tried += 1
        ex, ans = example_from(it, text); corr = collate_training_examples([ex], tok)
        target, total, checks = TARGET, 0, 0
        st = m.train_on_examples(corr, reh, loss_target=target, max_steps=CAP, rehearsal_weight=1.0, rehearsal_margin=0.05); total += st.steps
        while True:
            checks += 1; out = gen(it.question); fixed = ok(it, out)
            if fixed or total >= CAP: break
            if target <= VERIFY_LOSS_FLOOR and st.final_loss <= VERIFY_LOSS_FLOOR: break
            target = max(VERIFY_LOSS_FLOOR, min(target / 2, st.final_loss / 2))
            st = m.train_on_examples(corr, reh, loss_target=target, max_steps=min(VSTEPS, CAP - total), rehearsal_weight=1.0, rehearsal_margin=0.05); total += st.steps
        mx.clear_cache()
        n, l, outs = score(cid)
        keep = n > best_n and l <= best_l
        print(f"\nCAND {cid} k={k} steps={total} checks={checks} target_answer={ans[:50]!r} | {marks(it, outs)} loops={l} | {'KEEP' if keep else 'restore'}", flush=True)
        if keep:
            best_n, best_l, best_outs = n, l, outs; snap = snapshot(); accepted += 1
            if n == 4: break
        else:
            restore(snap)
    after[cid] = best_outs; info[cid] = f"{tried} tried, {accepted} kept"
    print(f"\nAFTER {cid} {marks(it, after[cid])} loops={best_l} ({info[cid]})", flush=True)

final = {cid: score(cid) for cid in PARA}
print("\n\nRESULT item | baseline | after own loop | after all 5 | loop  (marks: original + 3 paraphrases)", flush=True)
for cid in PARA:
    print(f"RESULT {cid} | {marks(ds[cid], base[cid][2])} | {marks(ds[cid], after[cid])} | {marks(ds[cid], final[cid][2])} loops={final[cid][1]} | {info[cid]}", flush=True)
    print(f"FINAL {cid} {[answer_of(r)[:50] if closed(r) else '(no answer)' for r in final[cid][2]]}", flush=True)
