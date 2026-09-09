"""Dose-response probe: for one wrong item, how many steps at each lr until the
corrected answer is produced, and what does that do to an unrelated correct item?"""
import sys, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import adaptible
from adaptible.revise import make_collated_training_example, strip_think_tags
from adaptible._src._llm import _loss_fn
from adaptible.eval import generate_default_dataset
from adaptible._src.eval.harness import contains_key_terms
from mlx.utils import tree_flatten, tree_map

ds = generate_default_dataset()
wrong = next(i for i in ds if i.id == "geo_002")   # capital of Canada -> model says Montreal
probe = next(i for i in ds if i.id == "sci_001")   # unrelated, to watch for damage
m = adaptible.StatefulLLM(model_path=None)
def ask(item): return strip_think_tags(m.generate_response(item.question, use_history=False, max_tokens=1024))
base_w, base_p = ask(wrong), ask(probe)
print(f"BASE wrong={contains_key_terms(base_w, wrong.key_terms)} {base_w[:60]!r} | probe={contains_key_terms(base_p, probe.key_terms)} {base_p[:60]!r}", flush=True)
inter = [adaptible.InteractionHistory(idx=0, user_input=wrong.question, llm_response=m.generate_response(wrong.question, use_history=False))]
ex = make_collated_training_example(f"[[0]] {wrong.correct_answer} [[/0]]", inter, m._tokenizer, think_mode="baseline")
snapshot = tree_map(lambda a: mx.array(a), m._model.trainable_parameters())
for lr in (float(x) for x in sys.argv[1:] or ["1e-5", "2e-5", "5e-5"]):
    m._model.update(tree_map(lambda a: mx.array(a), snapshot))
    m._optimizer = optim.AdamW(learning_rate=lr)
    lg = nn.value_and_grad(m._model, _loss_fn)
    for step in range(1, 13):
        loss, grads = lg(m._model, ex.input, ex.label, ex.mask); m._optimizer.update(m._model, grads); mx.eval(m._model.parameters(), m._optimizer.state)
        if step in (1, 2, 3, 5, 8, 12):
            w, p = ask(wrong), ask(probe)
            print(f"lr={lr:g} step={step:2d} loss={float(loss):.3f} wrong_fixed={contains_key_terms(w, wrong.key_terms)} probe_ok={contains_key_terms(p, probe.key_terms)} len={len(w.split())}/{len(p.split())} | {w[:50]!r}", flush=True)
