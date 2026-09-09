#!/bin/zsh
# Quick screen on 40 items (32 train / 8 holdout): does think_mode=baseline + rehearsal stop the collapse?
cd /Users/erichansen/Code/public/adaptible
name=${1:-screen_rh4}; shift
echo "[$(date)] START $name $@" >> outputs/runs/status.txt
.venv/bin/python -m adaptible.eval --name $name --subset 40 --shuffle --iterations 5 --no_browser \
    --think_mode baseline --rehearsal_k 4 --output outputs/runs/$name.html "$@" > outputs/runs/$name.log 2>&1
echo "[$(date)] END $name rc=$?" >> outputs/runs/status.txt
