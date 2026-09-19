# Wrapper implementation

[Getting started](README.md) · [Integration and testing](INTEGRATION.md)

This page describes the feedback loop, model provenance, and runtime lifecycle.
The external wrappers are implemented in this directory and share a controller,
training worker, and persistent store.

The wrapper command defaults to an 8192-token context and a 2048-token output
budget, matching the thinking verification settings. `--context-size` and
`--max-tokens` override these limits. The output budget includes reasoning;
reaching the limit without a completed final answer is not a successful repair.

## Review and acceptance

Reference discovery is shared by all four runtimes. A nonblank feedback note
takes priority, followed by a matching supplied document. Otherwise the default
is web lookup through DDGS, using only the flagged question as the query. No
API key or additional model is required. `--no-web-search` disables that
fallback; the wrapper does not load demo passages implicitly. Lookup uses
search excerpts, without fetching full source pages. Because the query does
not include the conversation, context-dependent follow-ups may be unanswerable.

The frozen model extracts a grounded answer separately from each web source.
It writes the answer name first and selects a source-derived sentence index.
Python copies that exact sentence as evidence, checks that the answer appears
inside it, and asks the model whether the selected sentence supports the answer.
The same evidence span is retained for drafting; the model cannot invent its
quotation or silently substitute a different part of the source. Ambiguous
bracketed answer names are skipped rather than stripped of a possibly meaningful suffix.
At least two different websites must support the same answer; subdomains are
grouped together. Conflicting extracted answers or insufficient evidence cause
a skipped review. This corroboration does not establish independent authorship
or guarantee truth. Search failures also skip training and retain the reason.
A provider's "no results" response can also reflect a rejected request; it does
not establish that relevant evidence is absent from the web.
The interaction's `references` record preserves the lookup kind and query,
source URLs and excerpt text, grounded quotes and answers, and any lookup
reason or error for inspection through `/history`. Supplied notes or documents
retain their text in `sources[].text`. The overall repair outcome is recorded
separately in the interaction's `status` and `reason`.

For a flagged question, the frozen base model reads the evidence and generates
rephrasings. In nonthinking mode, training uses its extracted short answer as the
correction, without generating additional claims. Two source-validated generated wordings augment
the original training prompt; a third generated wording, when available, is
reserved for evaluation. Missing or unusable references lead to a skipped
review. This path currently handles short factual answers.

For a thinking-mode interaction, the same frozen base separately drafts a
reasoning trace and final answer for each of the original and two training
wordings, conditioned on the actual reference. The expected answer is used by
the judge, not injected separately into the drafting prompt. Each draft must
contain nonempty completed reasoning and a final answer that passes the full
factual acceptance rule. A failed draft skips the review; there is no empty-think
or nonthinking fallback. The rationale is evidence-conditioned, but its individual
claims are not independently verified: the factual judge checks the final answer.
The new path must still prove useful through free generation after training.

The trainer loads the exact local base model offline and trains only LoRA
parameters. Current settings are rank 8 on the last eight transformer blocks
(or all blocks for a smaller model), at most 64 AdamW updates, and a learning
rate of `2e-5`. Nonthinking training masks the prompt and learns the answer.
Thinking training masks the original conversation but learns the complete new
grounded rationale, its closing boundary, and the final answer together. A
separate stopping mask measures only final-answer tokens, so a long easy
rationale cannot dilute that stopping loss. Training stops at final-answer loss
0.15 or the update limit; low loss alone never authorizes acceptance. Both losses
come from the same forward pass. If a thinking candidate is rejected solely for
no re-ask improvement, the wrapper can resume the same weights and optimizer
moments at successive doubled total-update budgets, up to 64 updates across the
whole trajectory. It reuses the same accepted parent, prompts, grounded drafts,
learning rate, and optimizer settings. Continuations run their remaining update
budget even when teacher-forced answer loss is already low. The first candidate
that passes behavioral validation ends the review. A control, retention, or
previously passing current re-ask regression ends it without further training.
Hashed continuation manifests bind the source, accepted parent, data, options,
adapter weights, and optimizer state; mismatches reject continuation. Stats record initial/final training and stopping
losses, stopping scope, actual updates, stopping reason, and elapsed time.
Prompt and padding labels are masked independently for each batch row. Each
prompt plus completion is limited to 4096 tokens; ambiguous tokenizer boundaries
or malformed reasoning prefixes are rejected before training.

Thinking training uses the original MLX experiment's AdamW update without bias
correction and with weight decay `0.01`, implemented in Torch. Standard Torch
AdamW applies bias correction, giving substantially smaller first updates at the
same learning rate; numerical tests compare the first and subsequent updates
against MLX. Nonthinking training retains standard Torch AdamW with zero weight
decay. Stats identify the optimizer, learning rate, and decay.

The wrapper is not an exact reproduction of `scripts/cycles_mlx.py`. That
experiment trained one sampled rationale for at most four updates, trying up to
two candidates from six samples, and kept any increase on four fixed prompts.
The wrapper trains up to three wording-specific rationales together, has a
64-update bound, samples each draft once from the frozen base, and additionally
checks controls and prior repairs. Its Qwen3 model also differs from the original
DeepSeek-R1-Distill-Qwen model. Matching the optimizer and rationale loss does
not establish equivalent learning behavior or reproduce the long-run result.

The serving engine evaluates the candidate using re-asks and whichever of three
known control questions it answered correctly before training (at least one).
A candidate is kept only when its re-ask score increases, no previously passing
re-ask is lost, and those controls are preserved. It also rechecks prompts from earlier
accepted repairs, preserving each previously passing prompt in its recorded
thinking mode. Thinking checks require actual nonempty reasoning, valid framing,
a normal generation finish, and a correct completed final answer; truncation or
empty/malformed reasoning cannot pass. The adapter pointer
and repair ledger are committed together so retention evidence survives a crash
before the interaction outcome is saved. Otherwise the prior accepted adapter is restored.
For factual repairs, the complete answer must equal the extracted answer or be
a whole sentence copied from the evidence. This conservative rule rejects unsupported
continuations and can also reject valid paraphrases. A model-only yes/no judge
was not reliable enough to authorize more permissive acceptance.
The judge extracts the answer freely and checks its supporting quote verbatim.
If copying fails, a retry constrains quote copying without supplying the expected
answer as a choice. A valid different answer is rejected without resampling.
Client generation is serialized with review, so clients do not see trial
adapters. Status remains available while review runs.

These checks are limited: grounding and a few controls do not prove complete
factual correctness or preservation of every unrelated answer. Small models
can fail to extract a correction or improve after training. Rejection is an
ordinary review outcome.

## Correction-scoped serving

The current deployment routes learned factual question scopes to the latest
cumulative adapter and other requests to the frozen base. It does not select
an older adapter per fact or inject a saved answer into generation. Candidate
checks and prior-repair retention still exercise cumulative weights directly;
serving unrelated requests from the base is not proof of global weight stability.

Scope matching compares only accepted questions and validated rephrasings.
Thinking mode and the preceding plaintext conversation must match the recorded
interaction. Exact question matches route directly; unfamiliar wordings first
pass conservative subject, relationship, and qualifier constraints, then a
structured same-fact check using the existing frozen base. That check receives
questions, without expected answers, evidence, or assistant history. Ambiguity,
unsupported context, malformed decisions, and classifier failures fall back to
the base. The classifier remains fallible and the lexical constraints may reject
valid paraphrases. Repeated repairs of the same original question, mode, and
context share a scope, retain validated variants, and report the latest repair ID.

Selection and generation run under the review gate, including the full stream.
The accepted ledger restores scopes after restart. Response headers and recorded
adaptation metadata expose the selected weights and scope; see the
[integration guide](INTEGRATION.md#which-weights-answer-a-request).

## Model provenance and persistence

Implemented sources are dense Qwen2/Qwen2.5, Qwen3 and Llama-family (including
Mistral) text GGUF files, or corresponding unquantized local HF checkpoints
containing safetensors and tokenizer files. Other architectures and quantized HF
checkpoints are rejected before training, but still serve: startup identifies
the model without gating it, and reports whether repair is available.

Llama-family GGUFs interleave the rotary pairs of `attn_q` and `attn_k`, so a
LoRA trained in Transformers space is permuted the same way before export;
adding it unpermuted moves 58% of the adapter's effect into the wrong rows.
`adaptible/wrap/gguf_roundtrip_test.py` checks every supported
architecture numerically: it fuses the exported adapter into the base GGUF,
reloads it, and requires the logits to match `base + PEFT adapter` to within
1% of the adapter's own effect. A new architecture is added to
`SUPPORTED_ARCHITECTURES` only once it passes that check. Layer counts and projection sizes
come from the source model; model size is not hardcoded.

Qwen3's seven adapted projections use the same tensor layouts as Qwen2; its
additional query/key normalization weights remain frozen. Architecture metadata
is preserved during adapter export and GGUF fusion. Numerical unit checks pass.
Native Ollama, llama.cpp, LM Studio, and vLLM have all passed the two-repair
automatic protocol on Qwen3 0.6B with thinking disabled. The latest fresh
thinking-enabled llama.cpp run (`llama-cpp-hkgug7nx`) ended inconclusive after
1,219 seconds: one of two required facts repaired, three trained candidates
(one kept, two rejected), and 12 skipped attempts. Its source weights and
protected baseline answers remained unchanged. An earlier unscoped run had
failed on an unrelated-answer regression. No complete two-repair thinking run
has passed across these runtimes, and the separate four-repair full-budget
protocol remains pending. The newly added optimizer-continuation path requires
fresh native verification; unit tests and saved-job diagnostics do not establish
an end-to-end learning pass.

For Qwen3, the included terminal client now enables thinking. Compatible API
requests retain their settings except for verified runtime translations: stock
Ollama Qwen3 supports binary thinking, so nonzero effort levels map to enabled
thinking and `none` remains disabled. Native boolean controls are preserved.
The original request mode is recorded before this translation. The store records the effective
thinking mode, final answer, reasoning, framing, and finish status. Review uses
that mode for correction drafting, training, re-asks, controls, and retention.
Evidence extraction and final-answer judging remain bounded nonthinking calls.
Completed assistant history is normalized to final-answer content to match stock
Qwen3 templates; malformed histories are refused for training. Endpoint-specific
mode controls, including Ollama's different native/OpenAI schemas, are interpreted
explicitly. Unreproducible or conflicting settings skip repair without changing
the serving request. This records thinking mode, not every sampling parameter.

Truncated or malformed public responses, and thinking responses without a
completed thought and final answer, remain in history but cannot be flagged for
training or added to terminal conversation history. Reasoning and final content
are forwarded separately in compatible API streams; the terminal displays them
with think delimiters. An interrupted reasoning stream is not given a fabricated
closing boundary.

Ollama Qwen3 uses a separately verified renderer for its stock `qwen3:0.6b`
template (SHA-256 `ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4`).
Unlike HF Jinja, this template adds `/no_think` to the last user turn when
thinking is disabled. Its
system-message whitespace and adjacent-role collation are also reproduced.
Actual native token IDs match the training renderer in bounded prefix tests.
Other Qwen3 Ollama templates, including the installed Instruct-2507 variant,
are not assumed equivalent and currently fail training with a diagnostic.
The template and system prompt are pinned alongside checkpoint identity;
accepted state predating this pin requires a fresh state directory.

Qwen3.8 is a newer release, but its dense 27B model uses the hybrid `qwen3_5`
architecture, with gated DeltaNet layers and full attention. It cannot be enabled
by relabeling a Qwen3 adapter. The current implementation rejects it, and its dense
training weights alone exceed this development machine's 16 GB RAM. The smaller
Qwen3.5 models share that hybrid architecture and also need an implementation.
See the official [release history](https://github.com/QwenLM/Qwen3.8/blob/main/README.md)
and [Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).

An adapter belongs to its exact training checkpoint. Saved state records the
runtime, model identity, and source fingerprint: SHA-256 for a GGUF, or a
digest of the relevant HF weights, configuration, tokenizer, and template
contents. Restarting with a different base is rejected. Adapters cannot be
shared across sizes or assumed compatible with another checkpoint in the
same family.

History, pending feedback, and the accepted adapter are stored under
`<outputs>/wrap/<model-key>/` unless `--state-dir` is supplied. A lock prevents
two wrappers from using the same state directory. Training produces adapter
files and source metadata, and reads the original local base directly rather than
downloading another checkpoint or creating source symlinks. LM Studio and Ollama's Qwen3 path also retain
the accepted derived GGUF for restart; it can be recreated from the small adapter.

## Runtime lifecycle

Before training, the wrapper releases its serving weights so serving and dense
training copies do not need to occupy GPU memory together. The worker exits
after training, releasing its model memory. Candidate validation then reloads
the serving model. This reduces peak demand, but a GGUF is still dequantized
for training: enough RAM for quantized inference does not imply enough RAM for
training. External storage, including ExFAT, can hold both model files and state.
Storage does not add RAM.

| Runtime | Adapter handling | Memory and process ownership |
| --- | --- | --- |
| Ollama | Private tags using native adapters for Qwen2, locally fused GGUFs for Qwen3 | Unloads the selected base and active adapter before training; other models are left alone |
| llama.cpp | Starts its private `llama-server` with the adapter | Stops the owned child before training; restarts it for validation and rollback |
| LM Studio | Fuses the adapter into a derived local GGUF, then imports it through `lms` | Unloads only its owned model instances; switches to the original base for frozen judging |
| vLLM | Registers local PEFT adapters through native dynamic adapter endpoints | Stops and restarts its owned server around training; reloads the accepted adapter |

Ollama's original model and files are not replaced. Rejected or superseded
private adapters are explicitly unloaded before their tags are deleted;
deleting a tag alone may leave a runner resident. Requests sent directly to
the upstream service can still load models independently of the wrapper.

The installed Ollama 0.12.10 Qwen3 engine rejects native LoRA loading. Adaptible
therefore fuses Qwen3's cumulative adapter into a local serving GGUF, preserving
the original template, system prompt, and parameters. It never trains or replaces
the original model. Rejected and superseded derived files and private tags are
removed; the accepted file remains available for restart. Ollama imports its own
copy into model storage, so both locations need free space.

LM Studio does not expose direct LoRA loading through the API used here.
Fusion preserves unchanged quantized tensors and metadata, and writes adapted
matrices as F16. The derived file is imported by symbolic link. Free space is
checked before fusion; rejected and superseded merged files are removed,
while the accepted file persists. The original GGUF remains unchanged.

The managed vLLM server binds to loopback with LoRA enabled. Requests select
the active adapter through private model handles; frozen checks select the
base. Dynamic adapter administration is not exposed through the wrapper's
public API. Runtime and trainer both use the local checkpoint offline.

Shutdown cancels review work and closes owned runtime resources. Interrupted
pending feedback is recoverable on restart. The wrapper does not take over
unrelated llama.cpp or vLLM processes, or unload unrelated LM Studio models.
