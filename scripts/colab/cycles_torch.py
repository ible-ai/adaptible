"""PyTorch/PEFT port of scripts/probes/mve_cycles.py for a CUDA box (Colab T4 and up).

Self-contained: no adaptible/MLX import. Same loop, same rules:
  each cycle: score every item (original + 3 paraphrases, greedy);
  any item under 4/4: sample K candidates (the model's OWN correct traces from a
  reference-note prompt at T=0.7), train each at most MAX_STEPS steps toward an
  answer-token loss of 0.15, KEEP if the item's 4-prompt score rises, else restore.
Differences from the MLX probe: generation is batched (4 prompts, 6 samples at once),
fp16 base on GPUs without bf16 (T4), and the run checkpoints the LoRA weights, the
history, and a status file to --out every cycle so a killed Colab session resumes
with `python cycles_torch.py --out <same dir>`.

Log lines (grep-able, same as the MLX probe): BASE / SAMPLE / CAND / CYCLE / SUMMARY.
"""
import argparse, json, math, os, re, socket, sys, time, unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, get_peft_model

# ----------------------------------------------------------------------------- items
ITEMS = {
    "geo_010": dict(q="What is the capital of Morocco?", a="Rabat", terms=["Rabat"],
                    para=["Which city is the capital of Morocco?", "Morocco's seat of government is in which city?",
                          "Name the capital city of the Kingdom of Morocco."]),
    "geo_004": dict(q="What is the capital of Turkey?", a="Ankara", terms=["Ankara"],
                    para=["Which city serves as Turkey's capital?", "Where is the seat of the Turkish government located?",
                          "Name the capital city of Turkey."]),
    "geo_001": dict(q="What is the capital of Australia?", a="Canberra", terms=["Canberra"],
                    para=["Which city is Australia's capital?", "Where is the seat of Australia's federal government?",
                          "Name the capital city of Australia."]),
    "geo_013": dict(q="What is the capital of the Philippines?", a="Manila", terms=["Manila"],
                    para=["Which city is the capital of the Philippines?", "Where is the seat of government of the Philippines?",
                          "Name the capital city of the Philippines."]),
    "sci_017": dict(q="What is the nearest star to Earth?", a="The Sun", terms=["Sun"], wrong=["Proxima", "Alpha Centauri"],
                    para=["Which star is closest to our planet?", "What is the closest star to Earth?",
                          "Name the star nearest to the Earth."]),
}
LOOP_SEQ, LOOP_REPS = 8, 3            # adaptible._src._llm token-loop breaker
LORA_RANK, LORA_SCALE, LORA_LAYERS = 8, 10.0, 8


# ----------------------------------------------------------------------------- helpers
def norm(s): return unicodedata.normalize("NFKC", s or "").casefold()
def has_terms(text, terms): return any(norm(t) in norm(text) for t in terms)
def closed(r): return "</think>" in r
def answer_of(r): return r.split("</think>")[-1].strip()
def ok(it, r):
    return closed(r) and has_terms(answer_of(r), it["terms"]) and not any(norm(w) in norm(answer_of(r)) for w in it.get("wrong", []))
def marks(it, outs): return "".join("✓" if ok(it, r) else "✗" for r in outs)
def loops(outs): return sum(not closed(r) for r in outs)
def prompts(it): return [it["q"]] + it["para"]
def hinted(it): return f"{it['q']}\n\n(Reference note: the correct answer is {it['a']}.)"
def first_sentence(text): return re.split(r"(?<=[.!?])\s", answer_of(text), 1)[0]
def clean(it, text):
    if not closed(text): return False
    f = first_sentence(text)
    return has_terms(f, it["terms"]) and len(f) < 300 and not any(norm(w) in norm(f) for w in it.get("wrong", []))


def token_loop(tokens, seq=LOOP_SEQ, reps=LOOP_REPS):
    if len(tokens) < seq * reps: return False
    recent = tokens[-seq:]
    return all(tokens[-(i + 1) * seq: -i * seq] == recent for i in range(1, reps))


class LoopStop(StoppingCriteria):
    """Per-sequence stop when the last LOOP_SEQ tokens repeated LOOP_REPS times."""
    def __init__(self, prompt_len): self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kw):
        n = LOOP_SEQ * LOOP_REPS
        if input_ids.shape[1] - self.prompt_len < n:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        tail = input_ids[:, -n:].view(input_ids.shape[0], LOOP_REPS, LOOP_SEQ)
        return (tail == tail[:, -1:, :]).all(dim=2).all(dim=1)


# ----------------------------------------------------------------------------- optimizer
class MLXAdamW(torch.optim.Optimizer):
    """AdamW exactly as mlx.optimizers.AdamW applies it by default: decoupled weight decay
    (p *= 1 - lr*wd) then an Adam step WITHOUT bias correction. The MLX flagship run used
    this, and without bias correction the first four steps are 3.2x, 4.3x, 5.0x, 5.4x the
    size of torch.optim.AdamW's at the same learning rate; the loop takes at most four steps
    per candidate from a fresh optimizer, so this is the whole difference."""
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if not st: st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
                m, v = st["m"], st["v"]
                p.mul_(1 - lr * wd)
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)


# ----------------------------------------------------------------------------- model
class Runner:
    def __init__(self, model_name, lr, max_new_tokens, device, sample_max_tokens=None, optimizer="mlx"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        # bf16 only on Ampere or newer (capability >= 8); a T4 (7.5) reports bf16 "supported" but emulates it slowly
        bf16_ok = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        self.dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else torch.float32)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype).to(device)
        n_layers = base.config.num_hidden_layers
        cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_SCALE * LORA_RANK, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         layers_to_transform=list(range(n_layers - LORA_LAYERS, n_layers)), task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)
        for n, p in self.model.named_parameters():       # adapters train in fp32 regardless of base dtype
            if p.requires_grad: p.data = p.data.float()
        self.device, self.lr, self.max_new = device, lr, max_new_tokens
        self.sample_max_new, self.optimizer_kind = sample_max_tokens or max_new_tokens, optimizer
        self.eos = self.tok.eos_token
        self.new_optimizer()
        gpu = torch.cuda.get_device_name() if device == "cuda" else device
        print(f"model={model_name} dtype={self.dtype} device={device} gpu={gpu} optimizer={self.optimizer_kind} score_cap={self.max_new} sample_cap={self.sample_max_new} trainable={sum(p.numel() for p in self.model.parameters() if p.requires_grad)}", flush=True)

    def new_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = (MLXAdamW(params, lr=self.lr, weight_decay=0.01) if self.optimizer_kind == "mlx"
                    else torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01))

    # --- weights
    def trainable(self): return {n: p for n, p in self.model.named_parameters() if p.requires_grad}
    def checksum(self): return float(sum(p.detach().abs().sum().item() for p in self.trainable().values()))
    def snapshot(self): return {n: p.detach().clone() for n, p in self.trainable().items()}, self.checksum()
    def restore(self, snap):
        params, s = snap
        with torch.no_grad():
            for n, p in self.trainable().items(): p.copy_(params[n])
        self.new_optimizer()
        assert abs(self.checksum() - s) < 1e-3 * max(1.0, s), "restore failed"
    def save_adapter(self, path): torch.save({n: p.detach().cpu() for n, p in self.trainable().items()}, path)
    def load_adapter(self, path):
        sd = torch.load(path, map_location=self.device)
        with torch.no_grad():
            for n, p in self.trainable().items(): p.copy_(sd[n].to(p.dtype))
        self.new_optimizer()

    # --- generation
    def prefix(self, q):
        if self.tok.chat_template:
            p = self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        else:  # smoke-test models without a template
            p = f"User: {q}\nAssistant: "
        if not p.endswith("<think>\n"): p += "<think>\n"
        return p

    @torch.no_grad()
    def generate(self, questions, temperature=0.0, seed=None, max_new=None):
        self.model.eval()
        enc = self.tok([self.prefix(q) for q in questions], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        if seed is not None: torch.manual_seed(seed)
        kw = dict(do_sample=temperature > 0, temperature=temperature if temperature > 0 else None, top_p=None, top_k=None)
        out = self.model.generate(**enc, max_new_tokens=max_new or self.max_new, pad_token_id=self.tok.pad_token_id,
                                  stopping_criteria=StoppingCriteriaList([LoopStop(enc.input_ids.shape[1])]), **kw)
        gen = out[:, enc.input_ids.shape[1]:]
        return [self.tok.decode(row, skip_special_tokens=True) for row in gen]

    # --- training
    def example(self, it, think, answer):
        ids_p = self.tok.encode(self.prefix(it["q"]), add_special_tokens=False)
        ids_t = self.tok.encode(think.strip() + "\n</think>\n\n", add_special_tokens=False)
        ids_a = self.tok.encode(answer + self.eos, add_special_tokens=False)
        ids = ids_p + ids_t + ids_a
        mask = [0] * len(ids_p) + [1] * (len(ids_t) + len(ids_a))
        stop = [0] * (len(ids_p) + len(ids_t)) + [1] * len(ids_a)
        t = lambda x: torch.tensor(x, device=self.device)
        return t(ids[:-1])[None], t(ids[1:])[None], t(mask[1:]).float()[None], t(stop[1:]).float()[None]

    def train(self, ex, max_steps, target):
        self.model.train()
        x, y, mask, stop = ex
        stop_losses = []
        for _ in range(max_steps):
            logits = self.model(input_ids=x).logits.float()
            per_tok = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none").view_as(y)
            train_loss = (per_tok * mask).sum() / mask.sum()
            stop_loss = ((per_tok * stop).sum() / stop.sum()).item()
            self.opt.zero_grad(set_to_none=True); train_loss.backward(); self.opt.step()
            stop_losses.append(stop_loss)
            if target is not None and stop_loss < target: break
        return len(stop_losses), stop_losses[-1]


# ----------------------------------------------------------------------------- loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="checkpoint/status dir (Drive path on Colab)")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--cycles", type=int, default=40); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=6); ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--target", type=float, default=0.15)
    ap.add_argument("--max_new_tokens", type=int, default=2048, help="decode cap when scoring (MLX generate_response: 2048)")
    ap.add_argument("--sample_max_tokens", type=int, default=1024, help="decode cap when sampling candidates (MLX loop: 1024)")
    ap.add_argument("--optimizer", choices=["mlx", "torch"], default="mlx", help="mlx = AdamW without bias correction (what the MLX run used); torch = torch.optim.AdamW")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--sample_from", choices=["current", "base"], default="current", help="draw candidates from the model being trained (current) or from a frozen adapter (base: untrained, or --sample_adapter)")
    ap.add_argument("--sample_adapter", default=None, help="with --sample_from base: adapter file to sample from instead of the untrained weights")
    ap.add_argument("--init_adapter", default=None, help="start training from this adapter instead of untrained weights (new run dir; not a resume)")
    ap.add_argument("--items", default=",".join(ITEMS), help="comma-separated item ids")
    ap.add_argument("--seed", type=int, default=1000, help="sampling seed base (seed + 100 * cycle); the MLX run used 1000")
    ap.add_argument("--smoke", action="store_true", help="tiny limits to exercise every code path")
    args = ap.parse_args()
    if args.smoke: args.max_new_tokens, args.sample_max_tokens, args.max_samples, args.k = 24, 24, 2, 1
    items = {k: ITEMS[k] for k in args.items.split(",")}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    r = Runner(args.model, args.lr, args.max_new_tokens, device, sample_max_tokens=args.sample_max_tokens, optimizer=args.optimizer)

    hist_path, adapter_path, status_path = out / "history.json", out / "adapter.pt", out / "status.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else {"cycles": [], "cand": {k: [0, 0] for k in items}}
    start = len(history["cycles"])
    zero_snap = r.snapshot()                        # untrained adapter
    if args.sample_adapter: r.load_adapter(args.sample_adapter)
    init_snap = r.snapshot()                        # frozen sampler weights for --sample_from base (untrained unless --sample_adapter)
    if args.init_adapter: r.load_adapter(args.init_adapter); print(f"INIT from {args.init_adapter} checksum={r.checksum():.3f}", flush=True)
    else: r.restore(zero_snap)
    if start and adapter_path.exists():
        r.load_adapter(adapter_path); print(f"RESUME from cycle {start} checksum={r.checksum():.3f}", flush=True)

    def status(cycle, total, lp, state, note=""):
        status_path.write_text(json.dumps(dict(cycle=cycle, score=total, loops=lp, state=state, host=socket.gethostname(),
                                               device=device, time=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(args)), indent=1))
        (out / "status.txt").write_text(f"adaptible cycles status: cycle={cycle} score={total} loops={lp} state={state} time={time.strftime('%Y-%m-%d %H:%M:%S')} device={device} {note}\n")

    def score(it):
        outs = r.generate(prompts(it))
        return sum(ok(it, o) for o in outs), loops(outs), outs

    for cycle in range(start, args.cycles):
        t0 = time.time(); status(cycle, None, None, "running")
        # One 4-prompt batch per item, the same shape as the post-candidate score:
        # greedy output depends on the padded batch shape (a 20-prompt batch and a
        # 4-prompt batch of the same adapter differed by 2/20), so the keep rule
        # must compare like with like.
        sc = {k: score(it) for k, it in items.items()}
        status(cycle, None, None, "running", f"scored all {sum(v[0] for v in sc.values())} {round(time.time()-t0)}s")
        if cycle == 0:
            for k, it in items.items(): print(f"BASE {k} {marks(it, sc[k][2])} loops={sc[k][1]}", flush=True)
        todo = [k for k in items if sc[k][0] < 4]
        if todo and args.sample_from == "base":      # sample from the untrained adapter, then put the trained one back
            cur = r.snapshot(); r.restore(init_snap)
        pooled = r.generate([hinted(items[k]) for k in todo for _ in range(args.max_samples)], temperature=args.temp, seed=args.seed + 100 * cycle, max_new=r.sample_max_new) if todo else []
        if todo and args.sample_from == "base": r.restore(cur)
        status(cycle, None, None, "running", f"sampled {len(pooled)} for {todo} {round(time.time()-t0)}s")
        for idx, k in enumerate(todo):
            it = items[k]
            best_n, best_l, best_outs = sc[k]
            snap = r.snapshot()
            samples = pooled[idx * args.max_samples:(idx + 1) * args.max_samples]
            cands = []
            for i, s in enumerate(samples):
                good = clean(it, s)
                print(f"SAMPLE {k} cycle={cycle} i={i} clean={int(good)} len={len(s.split())}w | {first_sentence(s)[:70] if closed(s) else '(unclosed)'!r}", flush=True)
                if good and len(cands) < args.k: cands.append(s)
            if not cands and args.smoke:  # exercise train/keep/restore even with a random model
                cands = [f"smoke reasoning about {it['a']}.\n</think>\n\nThe answer is {it['a']}."]
            if not cands: print(f"NOCAND {k} cycle={cycle}", flush=True)
            for j, text in enumerate(cands):
                think = text.split("</think>")[0]
                ans = " ".join(re.split(r"(?<=[.!?])\s", answer_of(text))[:2]).replace("\n", " ").strip()
                steps, loss = r.train(r.example(it, think, ans), args.max_steps, args.target)
                n, l, outs = score(it)
                keep = n > best_n
                history["cand"][k][0] += 1
                print(f"CAND cycle={cycle} {k} k={j} steps={steps} loss={loss:.2f} target_answer={ans[:45]!r} | {marks(it, outs)} loops={l} | {'KEEP' if keep else 'restore'}", flush=True)
                status(cycle, None, None, "running", f"cand {k} k={j} steps={steps} loss={loss:.2f} {marks(it, outs)} {'KEEP' if keep else 'restore'} {round(time.time()-t0)}s")
                if keep:
                    best_n, best_l, best_outs = n, l, outs; snap = r.snapshot(); history["cand"][k][1] += 1
                    if n == 4: break
                else:
                    r.restore(snap)
            sc[k] = (best_n, best_l, best_outs)
        total, lp = sum(v[0] for v in sc.values()), sum(v[1] for v in sc.values())
        history["cycles"].append(dict(cycle=cycle, score=total, loops=lp, marks={k: marks(items[k], sc[k][2]) for k in items},
                                      answers={k: [answer_of(o)[:60] if closed(o) else "(no answer)" for o in sc[k][2]] for k in items},
                                      seconds=round(time.time() - t0)))
        r.save_adapter(adapter_path); r.save_adapter(out / f"adapter_{cycle:02d}.pt"); hist_path.write_text(json.dumps(history, indent=1)); status(cycle, total, lp, "cycle_done")
        print(f"CYCLE {cycle} score={total}/{4*len(items)} loops={lp} | " + " ".join(f"{k}:{marks(items[k], sc[k][2])}" for k in items)
              + f" | {round(time.time()-t0)}s", flush=True)
    status(args.cycles, None, None, "finished")
    print("SUMMARY cycle score loops")
    for c in history["cycles"]: print(f"SUMMARY {c['cycle']} {c['score']} {c['loops']}")


if __name__ == "__main__":
    main()
