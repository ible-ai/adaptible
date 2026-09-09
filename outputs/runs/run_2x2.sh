#!/bin/zsh
# Queued after the control arm: 2x2 on the same 105 items, 5 LoRA iterations each.
#   {close_think, noclose_think} x {ground_truth, self_generated+fewshot}
cd /Users/erichansen/Code/public/adaptible
PY=.venv/bin/python
until grep -q "END meta control" outputs/runs/status.txt 2>/dev/null; do sleep 60; done
run() { # name, extra flags...
  local name=$1; shift
  echo "[$(date)] START 2x2 $name" >> outputs/runs/status.txt
  $PY -m adaptible.eval --name "$name" --shuffle --iterations 5 --no_browser \
      --output "outputs/runs/${name}.html" "$@" > "outputs/runs/${name}.log" 2>&1
  echo "[$(date)] END 2x2 $name rc=$?" >> outputs/runs/status.txt
}
run gt_close      --training_source ground_truth   --close_think
run gt_noclose    --training_source ground_truth   --noclose_think
run self_close    --training_source self_generated --revision_prompt fewshot --close_think
run self_noclose  --training_source self_generated --revision_prompt fewshot --noclose_think
echo "[$(date)] 2x2 DONE" >> outputs/runs/status.txt
