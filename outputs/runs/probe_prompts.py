"""Probe: how often does each revision prompt preset yield a valid revision?
Reuses stored baseline responses (experiment 4) so no baseline regeneration."""
import sqlite3, json, sys
import adaptible
from adaptible.revise import (make_revision_prompt, revision_prompt_preset,
                              validate_revision_response, InvalidRevisionError, strip_think_tags)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
c = sqlite3.connect("outputs/adaptible.db")
rows = c.execute("""select e.question, r.response_text from responses r join examples e on e.id=r.example_id
                    where r.experiment_id=4 and r.phase='baseline' order by r.id limit ?""", (N,)).fetchall()
model = adaptible.StatefulLLM(model_path=None)
out = {}
for preset in ("default", "fewshot"):
    instr, style = revision_prompt_preset(preset)
    valid = 0; reasons = {}
    for q, base in rows:
        inter = [adaptible.InteractionHistory(idx=0, user_input=q, llm_response=base)]
        prompt = make_revision_prompt(inter, model._tokenizer, instructions=instr, dialog_style=style)
        rev = model.generate_response(prompt, use_history=False, max_tokens=1024) or ""
        rev = strip_think_tags(rev) or rev
        try:
            validate_revision_response(rev, num_interactions=1); valid += 1
        except InvalidRevisionError as e:
            k = str(e)[:40]; reasons[k] = reasons.get(k, 0) + 1
    out[preset] = {"valid": valid, "n": len(rows), "rejections": reasons}
    print(f"PROBE {preset}: {valid}/{len(rows)} valid; rejections={reasons}", flush=True)
json.dump(out, open("outputs/runs/probe_prompts.json", "w"), indent=1)
