# Wrapper integration and testing

[Getting started](README.md) · [Implementation details](IMPLEMENTATION.md)

Use this page to connect a client, configure a runtime, or repeat the live
integration tests. All commands assume an editable checkout installed with
`python -m pip install -e '.[wrap]'` and an existing local model.

## Runtime setup

| Runtime | Model input | Required runtime |
| --- | --- | --- |
| Ollama | Installed base-model tag | Running local Ollama service |
| llama.cpp | Local dense Qwen2/Qwen2.5, Qwen3, Llama or Mistral GGUF | `llama-server` on PATH, or `--llama-server /path/to/llama-server` |
| LM Studio | Local dense Qwen2/Qwen2.5, Qwen3, Llama or Mistral GGUF | Running local server and `lms` CLI; override with `--lms /path/to/lms` |
| vLLM | Local unquantized dense Qwen2/Qwen2.5, Qwen3, Llama or Mistral HF checkpoint with safetensors and tokenizer | Working vLLM installation; override with `--vllm-server /path/to/vllm` |

```sh
adaptible wrap ollama qwen3:0.6b
adaptible wrap llama-cpp /path/to/model.gguf
adaptible wrap lm-studio /path/to/model.gguf
adaptible wrap vllm /path/to/local-hf-model
```

Run one command for your selected runtime. Ollama and the wrapper must be able
to read the same local model file. Start from a base model without a preexisting
LoRA; the wrapper resumes its own accepted adapters automatically.

Ollama's upstream defaults to `http://127.0.0.1:11434`; LM Studio's defaults to
`http://127.0.0.1:1234`. Override either with `--upstream`, or set
`ADAPTIBLE_OLLAMA_URL` / `ADAPTIBLE_LM_STUDIO_URL` in the environment; the
command-line flag wins. The URL must be absolute HTTP(S) with no credentials,
query, or fragment. llama.cpp and vLLM are started and managed by the wrapper,
so they reject `--upstream` rather than ignoring it. LM Studio can use the
desktop server or headless `llmster`; `lms` is also discovered under
`~/.lmstudio/bin`. Set `LM_API_TOKEN` if its service requires authentication.
LM Studio and Ollama's Qwen3 path need extra disk space for an automatically
generated serving GGUF. Ollama also imports it into its own model storage.

The wrapper starts private llama.cpp and vLLM servers. Install their runtime
separately for your hardware. vLLM's checkpoint and tokenizer are loaded
offline; the wrapper does not download models or modify an existing vLLM
server. A generic OpenAI-compatible endpoint is insufficient because adapter
management also needs the native runtime.

Common options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--documents passages.json` | Optional JSON object mapping document titles to reference passages | None |
| `--no-web-search` | Disable automatic web reference lookup | Web lookup enabled |
| `--state-dir /path/to/state` | Persistent history and accepted adapter | `<outputs>/wrap/<model-key>/` |
| `--host`, `--port` | Wrapper's listening address | `127.0.0.1`, `8000` |
| `--idle-seconds` | Delay before reviewing pending feedback | `2` |
| `--max-tokens` | Completion budget, reasoning included | `2048` |
| `--context-size` | Managed runtime context size | `8192` |
| `--log-level` | Wrapper and runtime log verbosity (`debug`/`info`/`warning`/`error`) | `info` |

`<outputs>` uses `ADAPTIBLE_OUTPUTS_DIR` when set, otherwise `outputs` under the
working directory. Each running wrapper needs its own state directory.

Model files and training state can live on an external SSD, including ExFAT.
Use `--state-dir /path/on/ssd/state` to keep adapters and derived serving files
there. The trainer reads the original model directly; no symbolic link or second
checkpoint is needed. The SSD adds storage, not training RAM.

### Qwen3 model and template support

For Ollama, Qwen3 learning currently supports the verified stock chat template
shipped with `qwen3:0.6b` and plain text conversations. A Qwen3 architecture alone does not
establish compatibility: `qwen3:4b-instruct` / Instruct-2507 uses a different
template and is not supported for training yet. Custom templates, Modelfile
`MESSAGE` histories and tool histories are also rejected for Qwen3 training
with an explicit review error. Completed assistant reasoning is normalized to
the final answer when reproducing conversation history; incomplete or malformed
reasoning history is rejected. Chat remains available.

Adaptible reproduces the supported Ollama template during training, including
its system messages and the selected thinking or nonthinking prefix. It preserves
the installed native template. A GGUF imported into Ollama with only a raw `{{ .Prompt }}` template
does not become a supported chat model merely because the weights are Qwen3.
The same GGUF's training path in llama.cpp or LM Studio uses its embedded Jinja
template; vLLM uses the local HF tokenizer's template.

The included terminal client now enables Qwen3 thinking automatically. Compatible
API requests retain their own settings, and the wrapper records the effective
thinking mode for review, training, and later retention checks. Use explicit
controls rather than relying on a server default:

| Endpoint / runtime | Thinking enabled | Thinking disabled |
| --- | --- | --- |
| Ollama OpenAI `/v1/chat/completions` | `"reasoning_effort": "medium"` | `"reasoning_effort": "none"` |
| LM Studio OpenAI `/v1/chat/completions` | `"reasoning_effort": "medium"` | `"reasoning_effort": "none"` |
| llama.cpp / vLLM OpenAI | `"chat_template_kwargs": {"enable_thinking": true}` | `"chat_template_kwargs": {"enable_thinking": false}` |
| Ollama native `/api/chat` or `/api/generate` | `"think": true` | `"think": false` |

Do not put Ollama's native `think` field on its OpenAI route, or use standalone
template kwargs as an LM Studio mode control. Conflicting controls or custom
request formatting that training cannot reproduce cause repair to be skipped;
the original serving request is still forwarded. Managed vLLM enables its Qwen3
reasoning parser. Reasoning is preserved in the runtime's response fields and
stream chunks; `/history` also records it separately from the final answer.

Thinking repair requires nonempty reasoning, a completed reasoning boundary,
a normal generation finish, and a checked final answer. A token-limit stop,
malformed reasoning, or empty answer cannot count as a completed repair.
If needed, increase the wrapper's `--max-tokens` while leaving space for input
in `--context-size`; the defaults are 2048 completion tokens and an 8192-token context.

The completed all-four tests below explicitly disabled thinking. They do not
establish transfer into thinking mode. The latest fresh llama.cpp thinking run
ended inconclusive: one of two required facts repaired, with protected baseline
answers and source weights preserved. It took 1,219 seconds and recorded three
trained candidates (one kept, two rejected) and 12 skipped attempts. An earlier
unscoped run failed on an unrelated-answer regression. No complete two-repair
thinking run has passed, and all four longer full-budget runs remain pending.

Stock Qwen3 in Ollama uses on/off thinking. Through Adaptible, OpenAI
`reasoning_effort` values `low`, `medium`, and `high` all enable thinking;
`none` disables it. The same mapping applies to supported nested effort controls
and native `think` levels. This compatibility mapping is limited to the verified
stock template; it preserves the client's recorded mode and leaves custom or
conflicting controls unchanged.

Ollama state also records the serving template and system prompt identity.
Changing these requires a fresh `--state-dir`, because accepted adapters were
trained with the previous prompt format. Older state containing an accepted
adapter but no template identity is likewise refused: its compatibility cannot
be established safely. Existing state is preserved, not silently reset. History
without an accepted adapter or previous template identity can acquire that identity automatically.

## Feedback and review

Point an OpenAI-compatible chat client at `http://127.0.0.1:8000/v1` and use
the model name returned by `GET /v1/models`. The name stays stable as adapters
change. Chat options and streaming are forwarded. Only the selected model is
exposed. For Ollama, `/api/chat` and `/api/generate` also work at the wrapper
address, including streaming. Model management and embeddings are not proxied.

Recorded responses carry an `X-Interaction-Idx` header. Wait for generation to
finish before using that index for
feedback; replace `1` below with the value from your response:

```sh
curl http://127.0.0.1:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"interaction_idx": 1, "thumbs": "down"}'
```

No answer note is required: review automatically searches for evidence and
asks the model to write a correction. Feedback may include a `note` string
containing reference material when you want to supply the evidence yourself.

| Endpoint | Purpose |
| --- | --- |
| `POST /feedback` | Record `interaction_idx`, `thumbs` (`up` or `down`), and optional `note` |
| `POST /trigger_review` | Start pending review immediately |
| `GET /sync` | Wait for review and return outcomes with reasons |
| `GET /history` | Read interactions and review outcomes |
| `GET /status` | Check status, including during training |
| `GET /docs` | Open the interactive API documentation |

Review outcomes are `kept`, `rejected`, `unchanged`, `skipped`, or `failed`.
Chat clients need a separate feedback action if they do not implement this API.
The included terminal client (`python -m adaptible.cli`) supports `/down`,
`/review` to review immediately and wait, and `/new` to start a fresh chat.

### Which weights answer a request

Requests matching a learned correction use the latest cumulative adapter,
including all its accepted updates. Other requests use the frozen original
model. Matching uses accepted questions and their validated rephrasings, with
the same thinking mode and exactly matching preceding conversation messages.
Different context, unsupported request formatting, ambiguous matches, and
unrecognized wording fall back to the original model. New paraphrases can match,
but transfer is conservative rather than guaranteed. The same existing base
model checks shortlisted question pairs; routing does not download another
checkpoint or supply reference answers to the client's generation.

A small grammar matches fully recognized capital, currency, and city-population
questions. An unsupported rewording cannot override one of those recognized
scopes. When both questions fall outside that grammar, a fallback can compare
predicate synonyms after checking identical explicit proper names, compatible
known answer categories, qualifiers, and simple argument direction. Unknown
answer categories also require the classifier’s decision. Missing names, extra
clauses, or unclear references abstain. The frozen-base classifier still makes
the final semantic decision for these candidates; this is limited language coverage, not
a guarantee that arbitrary paraphrases are equivalent.

Both streaming and ordinary responses identify the selected weights:

| Field | Meaning |
| --- | --- |
| `X-Adaptible-Adapter` response header | `base` or the accepted adapter directory name |
| `X-Adaptible-Scope` response header | Matching repair's interaction ID; absent for base requests |
| `/history` → `response_details.adaptation` | `policy`, `adapter`, `adapter_sha256`, `scope`, and routing `reason` |

The current policy is `correction_scoped_v1`. Reasons distinguish exact matches,
classified scope matches, absent scopes, context/mode mismatches, and abstention.
Grammar decisions include `matcher`; classified decisions include optional
`classifier` diagnostics with question-only inputs, raw output, completion
details, and verdict.
This makes serving decisions inspectable; it does not prove that cumulative
adapter weights themselves preserve unrelated knowledge. The earlier failed
thinking run remains a failure, and native verification of scoped thinking
learning is pending.

## Reference options

Automatic web lookup is enabled for all four runtimes. It needs no API key or
additional model. The flagged question is sent to a search provider; the rest
of the conversation is not included in the search query. Adaptible asks the
model to extract a quoted answer from search excerpts and requires agreement from
at least two different websites. Missing or conflicting evidence leads to a
skipped review. Full source pages are not fetched. Questions should stand on
their own: a follow-up such as "When did he move?" may lack enough context.

`GET /history` retains the lookup query, source URLs and excerpts, grounded
quotes, extracted answers, and lookup failures in each interaction's
`references` field. The interaction's `status` and `reason` describe the
overall repair outcome.
Different websites can still repeat the same mistake; corroboration is a check,
not a guarantee of accuracy.

For private material or offline work, add `--no-web-search`. You can supply
reference text in the feedback API's `note` field, or an optional collection:

```json
{
  "Morocco": "Morocco's capital is Rabat. Its largest city is Casablanca."
}
```

Save that as `passages.json` and add `--documents passages.json`. A feedback
note takes priority, followed by a matching document; web lookup is the
fallback when enabled. There is no implicit demo collection. With web search
disabled and no matching supplied reference, review is skipped with a reason.

## Check the actual wrapper

These commands launch the real Adaptible CLI and native runtime, then check
chat, streaming, automatic web lookup, recorded history, and restart.

An earlier version of this check passed with all four native runtimes, including LM Studio's
official headless engine and vLLM's native CPU backend on Apple Silicon.
Run the command for your app:

```sh
python -m scripts.wrapper_smoke ollama qwen2.5:0.5b
python -m scripts.wrapper_smoke llama-cpp /path/to/model.gguf \
  --llama-server /path/to/llama-server
python -m scripts.wrapper_smoke lm-studio /path/to/model.gguf
python -m scripts.wrapper_smoke vllm /path/to/local-hf-model
```

No reference file or feedback note is supplied. A pass requires retrieved
web sources and a completed review without runtime errors; a skipped or
rejected correction is allowed. This verifies the running integration, while
the repair demo below requires an accepted update. Reports and runtime logs
remain under `outputs/wrapper-smoke/`.

For vLLM, `--tiny-vllm` can replace the model path with a locally generated
tiny model. It also trains a real adapter, explicitly seeds accepted state,
and checks its application, restart, and rollback through the CLI. Seeded
state tests adapter restoration, not the controller's acceptance decision:

```sh
python -m scripts.wrapper_smoke vllm --tiny-vllm --vllm-server /path/to/vllm
```

## Repeatable live demo

Run the appropriate demo from the repository root. No model is downloaded:

```sh
python -m scripts.wrapper_demo ollama qwen2.5:0.5b
python -m scripts.wrapper_demo llama-cpp /path/to/model.gguf \
  --llama-server /path/to/llama-server
python -m scripts.wrapper_demo lm-studio /path/to/model.gguf
python -m scripts.wrapper_demo vllm /path/to/local-hf-model
```

Each demo starts with fresh wrapper state and an empty offline HF cache. It
asks a question, submits thumbs-down without an answer note, and requires
the model to produce a correction from an explicit fixture document. These
demos disable web lookup for reproducibility; they test runtime repair and
persistence. To exercise automatic reference discovery instead, add
`--web-search` to any of the four commands:

```sh
python -m scripts.wrapper_demo ollama qwen2.5:0.5b --web-search
```

This mode supplies no document or answer note and records the actual search
sources in the report. It needs internet access, and results may vary with
available evidence. Both modes require real training, an accepted adapter,
the expected answer marker, an unchanged
base checkpoint, and working history, adapter restoration, and streaming after
restart. Ollama's native streaming endpoints are checked too.

The demo cleans up its own processes, Ollama test tags, and LM Studio imports
and merged files. It preserves the source model and existing service. Reports,
responses, logs, and adapter artifacts remain under
`outputs/wrapper-demo/<unique-run>/`, outside version control. Inspect
`report.json`: `passed`, `failed`, and `inconclusive` correspond to exit codes
0, 1, and 2. An already-correct baseline is inconclusive; a rejected repair
fails this acceptance-path fixture.

Runtime executable and upstream options work as above. Use `--output-dir`,
`--timeout` (seconds per request/startup), and `--context-size` as needed.
For a different fixture, supply `--question`, `--reference`, and `--expected`
together. In web mode, supply only `--question` and `--expected`; `--reference`
is unused. The expected value is a first-sentence answer marker used by the
test, not supplied to the model or a judgment of every claim in the response.

Earlier fixed-reference and tiny-model tests exercised runtime integration.
The stronger pretrained, whole-answer results are recorded below.

## Repeated automatic repair test

The recurrent test exercises the actual CLI with live web lookup. Its completed
nonthinking runs required two different factual repairs to accumulate. Recorded
native Qwen3 0.6B results, all with thinking disabled:

| Runtime | Result | Exact answers | Time |
| --- | --- | --- | --- |
| llama.cpp, Metal, Q8_0 GGUF | Passed: Palau and Sri Lanka | 0/24 → 6/24 | 280 s |
| LM Studio, Metal, Q8_0 GGUF | Passed: Morocco and Palau | 0/24 → 6/24 | 281 s |
| vLLM, native Apple CPU, local HF checkpoint | Passed: Morocco and Kazakhstan | 0/24 → 8/24 | 2,163 s |
| Ollama, native Qwen3 engine, Q8_0 GGUF | Passed: Palau and Benin | 0/24 → 6/24 | 537 s |

Each pass includes both unseen wordings per counted fact, preservation of
previously correct answers, unchanged source weights, restart, and streaming.
vLLM also learned two of three Benin wordings; that partial result did not count
toward its two required complete repairs. Both rejected updates and reference
skips remain in its report. These baselines had no exact correct answers in the
fixed set, so demonstrated retention concerns newly learned facts; unrelated
knowledge has only the wrapper's limited control checks.

The Ollama run stored both its original model and wrapper state on an external
ExFAT SSD. Training read the original weights directly; serving used an
automatically fused GGUF. No additional pretrained checkpoint was downloaded.

```sh
python -m scripts.wrapper_recurrent ollama YOUR_INSTALLED_MODEL --nonthinking --progress
python -m scripts.wrapper_recurrent llama-cpp /path/to/model.gguf \
  --llama-server /path/to/llama-server --nonthinking --progress
python -m scripts.wrapper_recurrent lm-studio /path/to/model.gguf --nonthinking --progress
python -m scripts.wrapper_recurrent vllm /path/to/local-hf-model --nonthinking --progress
```

The test fixes eight factual questions before the run and checks each original
plus two additional wordings. Only the original questions receive thumbs-down;
feedback contains no answer, note, or document. Expected answers are test
assertions and never enter the model's prompts. An empty answer or one that
mentions the correct entity with extra claims cannot qualify as a factual
baseline error. Passing requires the entire final answer to match an accepted
entity name, with no unsupported continuation.

After each review, all 24 prompts are checked again. Previously correct answers
must stay correct; each counted repair must pass both additional wordings.
Real training, successive adapter lineage, unchanged source weights, history,
restart, and streaming are also required. The wrapper separately records past
accepted repairs and checks them before keeping future updates.

The default budget is two passes through the fixed questions (`--cycles 2`),
stopping early after two complete repairs. This is an integration and initial
accumulation check, not the original 40-cycle reasoning experiment.
An unsuccessful bounded learning attempt is `inconclusive`, not a pass;
regressions and runtime failures are `failed`. All attempts and full answer
matrices remain in `outputs/wrapper-recurrent/<run>/report.json`. `--progress`
writes brief phase updates to stderr; stdout remains the final JSON report.
For thinking models, `--nonthinking` explicitly disables thinking on the test
chat and streaming requests.

### Longer thinking-enabled test

This separate protocol requires four repairs; the latest inconclusive llama.cpp
run described above was the shorter two-repair protocol. No four-repair
full-budget native run has passed. Use the same fixed cases and
automatic evidence, explicitly require thinking, and continue past the first
successes:

```sh
python -m scripts.wrapper_recurrent ollama YOUR_INSTALLED_MODEL \
  --thinking --full-budget --required-repairs 4 --cycles 2 --timeout 1200 --progress
python -m scripts.wrapper_recurrent llama-cpp /path/to/model.gguf \
  --llama-server /path/to/llama-server \
  --thinking --full-budget --required-repairs 4 --cycles 2 --timeout 1200 --progress
python -m scripts.wrapper_recurrent lm-studio /path/to/model.gguf \
  --thinking --full-budget --required-repairs 4 --cycles 2 --timeout 1200 --progress
python -m scripts.wrapper_recurrent vllm /path/to/local-hf-model \
  --thinking --full-budget --required-repairs 4 --cycles 2 --timeout 1200 --progress
```

Two passes over eight fixed cases give 16 review opportunities, not 16 training
updates: already repaired, ineligible, or ungrounded cases are recorded and
skipped. `--full-budget` prevents early success from ending the run. At least
four distinct accepted factual repairs must pass their original and both
external wordings, with earlier passing answers retained after every review
and after restart. Actual nonempty reasoning and a completed final answer are
required for corrected answers and all streaming checks; reasoning token counts
alone do not suffice. Drafting and training never receive the test's expected
answers or external wordings.

Thinking runs default to a 2048-token output budget and an 8192-token serving
context; use `--max-tokens` and `--context-size` to override. Individual training
examples still have a 4096-token limit. `--output-dir /path/on/ssd/runs` places reports,
state, and derived models on an external drive. CPU thinking can take much longer
than short nonthinking answers. Report actual accepted updates, skipped attempts,
retention, and completion failures; elapsed time is not a success criterion.
This remains a bounded test on one small model, not proof of indefinite learning.

## Automated tests

These checks do not download pretrained models:

```sh
python -m unittest adaptible.tests.wrap_test adaptible.tests.wrap_cli_test \
  adaptible.tests.lmstudio_test adaptible.tests.vllm_test -v
python -m unittest adaptible.tests.model_source_test -v
python -m unittest adaptible.tests.retention_test adaptible.tests.wrap_recurrent_test -v
```

The wrapper tests cover API forwarding, streaming, feedback, failure handling,
and persistence. Source tests also train a tiny local HF model. To exercise
a native vLLM installation with a tiny randomly initialized model:

```sh
ADAPTIBLE_VLLM_EXECUTABLE=/path/to/vllm VLLM_CPU_KVCACHE_SPACE=1 \
  python -m unittest adaptible.tests.vllm_test.VLLMLiveAdapterTest -v
```

That test trains a real adapter and verifies native application, frozen-base
behavior, rollback, and restart. It does not test semantic improvement.

The full repair demos are also opt-in unittest cases:

```sh
ADAPTIBLE_TEST_OLLAMA_MODEL=qwen2.5:0.5b \
  python -m unittest adaptible.tests.wrap_integration_test -v

ADAPTIBLE_TEST_LLAMA_GGUF=/path/to/model.gguf \
ADAPTIBLE_TEST_LLAMA_SERVER=/path/to/llama-server \
  python -m unittest adaptible.tests.wrap_integration_test -v

ADAPTIBLE_TEST_LM_STUDIO_GGUF=/path/to/model.gguf \
  python -m unittest adaptible.tests.wrap_integration_test -v

ADAPTIBLE_TEST_VLLM_MODEL=/path/to/local-hf-model \
ADAPTIBLE_TEST_VLLM_SERVER=/path/to/vllm \
  python -m unittest adaptible.tests.wrap_integration_test -v
```

Cases without a model variable skip. Each enabled case requires a successful
repair; an inconclusive demo is not a pass. Optional overrides are:

| Environment variable | Demo option |
| --- | --- |
| `ADAPTIBLE_TEST_OLLAMA_URL`, `ADAPTIBLE_TEST_LM_STUDIO_URL` | `--upstream` for the respective service |
| `ADAPTIBLE_TEST_LLAMA_SERVER`, `ADAPTIBLE_TEST_LMS`, `ADAPTIBLE_TEST_VLLM_SERVER` | Respective runtime executable |
| `ADAPTIBLE_TEST_OUTPUT_DIR` | `--output-dir` |
| `ADAPTIBLE_TEST_TIMEOUT` | `--timeout` |
| `ADAPTIBLE_TEST_CONTEXT_SIZE` | `--context-size` |
| `ADAPTIBLE_TEST_WEB_SEARCH=1` | `--web-search`; applies to all enabled runtimes |
| `ADAPTIBLE_TEST_QUESTION`, `ADAPTIBLE_TEST_REFERENCE`, `ADAPTIBLE_TEST_EXPECTED` | Custom fixture; reference is unused in web mode |

The opt-in native unit-test entry points also support thinking explicitly. For
example, using an existing local checkpoint and server executable:

```sh
ADAPTIBLE_RECURRENT_THINKING=1 \
ADAPTIBLE_RECURRENT_LLAMA_GGUF=/path/to/model.gguf \
ADAPTIBLE_TEST_LLAMA_SERVER=/path/to/llama-server \
python -m unittest adaptible.tests.wrap_recurrent_test.RecurrentLiveTest.test_llama_cpp -v
```

Set `ADAPTIBLE_RECURRENT_OUTPUT_DIR` to place its reports and state on an external
SSD. Omitting the thinking switch keeps these older opt-in tests in explicit
nonthinking mode. Each native test performs real training and takes substantially
longer than the model-free suite.
