#!/bin/zsh
# Fixed pipeline only: wait for gt_close, then the real loop (few-shot + closed think).
cd /Users/erichansen/Code/public/adaptible
PY=.venv/bin/python
until grep -q "END 2x2 gt_close" outputs/runs/status.txt 2>/dev/null; do sleep 30; done
name=self_close
echo "[$(date)] START $name" >> outputs/runs/status.txt
$PY -m adaptible.eval --name $name --shuffle --iterations 5 --no_browser \
    --training_source self_generated --revision_prompt fewshot --close_think \
    --output outputs/runs/$name.html > outputs/runs/$name.log 2>&1
echo "[$(date)] END $name rc=$?" >> outputs/runs/status.txt
