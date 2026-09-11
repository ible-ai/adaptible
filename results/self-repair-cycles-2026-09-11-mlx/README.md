# Self-repair cycles, MLX, 2026-09-11

One run of `scripts/cycles_mlx.py` (repo commit `c147acc`), started
2026-09-10 23:20 EDT on a 16 GB M3 MacBook Air, 40 cycles requested.

## Configuration

| | |
|---|---|
| Model | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B`, bf16 |
| LoRA | rank 8, scale 10, every linear layer in the last 8 transformer blocks |
| Optimiser | AdamW, lr 2e-5, fresh optimiser state after every restore |
| Items | geo_010 Morocco, geo_004 Turkey, geo_001 Australia, geo_013 Philippines, sci_017 nearest star |
| Prompts per item | original question + 3 hand-written paraphrases (in the script) |
| Judge | think block closed, key term in the answer part (substring, NFKC casefold); sci_017 not credited if the answer names Proxima/Alpha Centauri |
| Candidates per cycle | up to 2 per item under 4/4, from up to 6 samples at T=0.7 of the question plus "(Reference note: the correct answer is X.)", kept only if the think block closed and the first answer sentence contains the key term |
| Training target | the sample's own think block, `</think>`, its first two answer sentences, EOS; loss over the whole target, stop rule on the answer tokens |
| Steps | at most 4, stop when answer-token loss < 0.15 |
| Accept rule | the item's 4-prompt score strictly rises; otherwise restore LoRA weights from a copy and assert the checksum |
| Decode | greedy, 1024-token cap, token-loop breaker (8-token sequence repeated 3 times) |

## Files

- `cycles.csv`: cycle, score (of 20), loops (generations with no closed think
  block), and the 4-prompt marks per item after that cycle's training.
- `candidates.csv`: every candidate: cycle, item, k, steps taken, final
  answer-token loss, first 45 characters of the target's answer, marks after
  training, loops, kept.
- `score.svg`: score and loops per cycle.

The full generation log (every prompt and output, about 40 MB) is not in the
repository; ask for it.

## Reading it

Baseline (cycle 0 before any keep) was 4 of 20. The kept candidates per item
over the run are in `candidates.csv`; `kept == True` rows are the updates that
compose the final weights. Cycles with no kept candidate whose score still
changed measure the greedy-decode noise floor.
