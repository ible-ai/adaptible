#!/bin/zsh
# Sequential: (1) self-generated eval, (2) same-seed repeat control arm.
cd /Users/erichansen/Code/public/adaptible
PY=.venv/bin/python
echo "[$(date)] START eval self_generated" >> outputs/runs/status.txt
$PY -m adaptible.eval --name self_generated_v1 --training_source self_generated \
    --shuffle --no_browser --output outputs/runs/eval_self_generated.html \
    > outputs/runs/eval_self_generated.log 2>&1
echo "[$(date)] END eval self_generated rc=$?" >> outputs/runs/status.txt
echo "[$(date)] START meta control seed42 x3" >> outputs/runs/status.txt
$PY scripts/run_meta_experiment.py --name control_seed42 --seeds 42 --repeats 3 \
    --holdout_every_checkpoint --no_browser --output outputs/runs/control_seed42.json \
    > outputs/runs/control_seed42.log 2>&1
echo "[$(date)] END meta control rc=$?" >> outputs/runs/status.txt
