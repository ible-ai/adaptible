"""Why are rationales missing? Show raw output of the rationale prompt for a few items."""
import adaptible
from adaptible.revise import split_think
from adaptible._src.eval.harness import make_rationale_prompt
from adaptible.eval import generate_default_dataset
ds = {i.id: i for i in generate_default_dataset()}
m = adaptible.StatefulLLM(model_path=None, num_lora_layers=8, lora_parameters={"rank": 8, "dropout": 0.0, "scale": 10.0})
for iid in ("geo_003", "sci_005", "hist_003", "math_002", "misc_002"):
    it = ds[iid]; out = m.generate_response(make_rationale_prompt(it.question, it.correct_answer), use_history=False, max_tokens=2048)
    think, ans = split_think(out); n = len(m._tokenizer.encode(out))
    print(f"ITEM {iid} tokens={n} has_close={'</think>' in out} think_words={len(think.split())} | head={out[:100]!r} | tail={out[-100:]!r}", flush=True)
