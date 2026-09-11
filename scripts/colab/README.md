# Running the self-repair cycle loop on a Colab GPU

`cycles_torch.py` is a self-contained PyTorch/PEFT port of
`scripts/probes/mve_cycles.py` (no MLX). Same loop, same rules, batched
generation, and it checkpoints every cycle so a killed session resumes.

## One-time

1. Open `adaptible_cycles.ipynb` in Colab (File → Upload notebook, or open from
   GitHub: `ible-ai/adaptible`, path `scripts/colab/adaptible_cycles.ipynb`).
2. Runtime → Change runtime type → T4 GPU.
3. Runtime → Run all. Approve the Drive mount.

## Every restart

Runtime → Run all. That is the whole restart. The script finds
`MyDrive/adaptible/cycles_t4/history.json` and `adapter.pt` and continues from
the next cycle. Nothing else to do.

## What lands on Drive (`MyDrive/adaptible/cycles_t4/`)

| File | What |
|---|---|
| `status.json` | `cycle`, `score`, `loops`, `state` (`running` / `cycle_done` / `finished`), timestamp, host. Poll this to see whether the session is alive: a `running` state with a stale timestamp means Colab killed it. |
| `history.json` | Every completed cycle: score, loops, per-item marks, the four answers, seconds. |
| `adapter.pt` | LoRA weights after the last completed cycle. |
| `run.log` | Full log, appended across sessions. `grep -E "^(CYCLE|CAND)"` for the tables. |

## Free-tier limits worth knowing

- Sessions end after roughly 12 hours, sooner if idle or if the GPU pool is
  short. Each cycle is checkpointed, so at most one cycle is lost.
- The T4 has no bf16; the base model runs in fp16 with fp32 LoRA weights.
  Numerics differ from the Mac, so the cycle-0 BASE line is the baseline for
  this run, not the MLX one.
- Model download is about 3.5 GB per fresh session (cached on the VM, not on
  Drive).

## Flags

`--cycles 40 --k 2 --max_steps 4 --target 0.15 --lr 2e-5 --max_samples 6
--temp 0.7 --items geo_010,geo_004,...`. `--smoke` runs every code path with
tiny limits (used with `hf-internal-testing/tiny-random-Qwen2ForCausalLM` to
test the script on CPU).
