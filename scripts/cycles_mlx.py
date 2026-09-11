"""Self-repair cycle loop (the flagship experiment; see README.md and results/).

Repeated greedy self-repair cycles. Each cycle: score every item (original + 3 paraphrases);
for any item under 4/4 try up to K fresh candidates (the model's OWN correct samples from a
reference-note prompt at T=0.7), train at most MAX_STEPS steps toward answer-loss 0.15 (no verify rounds: do not sear
the loss in), keep if the item's 4-prompt score rises (greedy), else checksum-restore.
No cross-item guard, no rehearsal. Question: does the 20-prompt score stabilise or fall apart?

Derived from mve_loop.py: for each of the 5 MVE items, try up to K
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
K = int(os.environ.get("K", "2")); TEMP = 0.7; MAX_SAMPLES = 6
CYCLES = int(os.environ.get("CYCLES", "8")); MAX_STEPS = int(os.environ.get("MAX_STEPS", "4"))
ds = {i.id: i for i in generate_default_dataset()}
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0}, learning_rate=LR)
tok = m._tokenizer
sampler = make_sampler(temp=TEMP)

def gen(q): return m.generate_response(q, use_history=False) or ""
def closed(r): return "</think>" in r
def answer_of(r): return r.split("</think>")[-1].strip()
WRONG = {"sci_017": ["Proxima", "Alpha Centauri"]}
def ok(it, r): return closed(r) and contains_key_terms(answer_of(r), it.key_terms) and not any(w.lower() in answer_of(r).lower() for w in WRONG.get(it.id, []))
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
    return contains_key_terms(first, it.key_terms) and len(first) < 300 and not any(w.lower() in first.lower() for w in WRONG.get(it.id, []))
def candidates(cid, cycle):
    it = ds[cid]; out = []
    start = seed = 1000 + 100 * cycle
    while len(out) < K and seed < start + MAX_SAMPLES:
        t = sample(hinted(it), seed); seed += 1
        good = clean(it, t)
        first = re.split(r"(?<=[.!?])\s", answer_of(t), 1)[0][:70] if closed(t) else "(unclosed)"
        print(f"SAMPLE {cid} seed={seed-1} clean={int(good)} len={len(t.split())}w | {first!r}", flush=True)
        if good: out.append(t)
    if not out: print(f"NOCAND {cid} cycle={cycle}", flush=True)
    return out
def example_from(it, text):
    think = text.split("</think>")[0].strip()
    ans = " ".join(re.split(r"(?<=[.!?])\s", answer_of(text))[:2]).replace("\n", " ").strip()
    inter = [InteractionHistory(idx=0, user_input=it.question, llm_response="", reviewed=False, timestamp=0.0)]
    return make_revision_training_example(f"[[0]] {ans} [[/0]]", inter, tok, think_mode="rationale", rationale=think), ans


history = []
for cycle in range(CYCLES):
    sc = {cid: score(cid) for cid in PARA}
    for cid in PARA:
        it = ds[cid]; best_n, best_l, best_outs = sc[cid]
        if best_n == 4: continue
        snap = snapshot()
        for k, text in enumerate(candidates(cid, cycle)):
            ex, ans = example_from(it, text); corr = collate_training_examples([ex], tok)
            st = m.train_on_examples(corr, [], loss_target=VERIFY_LOSS_FLOOR, max_steps=MAX_STEPS, rehearsal_weight=0.0, rehearsal_margin=0.05)
            mx.clear_cache()
            n, l, outs = score(cid)
            keep = n > best_n  # greedy on score; an unclosed think already scores as a miss
            print(f"\nCAND cycle={cycle} {cid} k={k} steps={st.steps} loss={st.final_loss:.2f} target_answer={ans[:45]!r} | {marks(it, outs)} loops={l} | {'KEEP' if keep else 'restore'}", flush=True)
            if keep:
                best_n, best_l, best_outs = n, l, outs; snap = snapshot()
                if n == 4: break
            else:
                restore(snap)
        sc[cid] = (best_n, best_l, best_outs)
    total = sum(v[0] for v in sc.values()); loops_ = sum(v[1] for v in sc.values())
    history.append((total, loops_))
    print(f"\nCYCLE {cycle} score={total}/20 loops={loops_} | " + " ".join(f"{cid}:{marks(ds[cid], sc[cid][2])}" for cid in PARA), flush=True)
print("\nSUMMARY cycle score loops")
for i, (t, l) in enumerate(history): print(f"SUMMARY {i} {t}/20 {l}")
