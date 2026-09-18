# Teach your local model from feedback

Adaptible adds a learning loop to **Ollama, llama.cpp, LM Studio, and vLLM**.
Flag a wrong answer, and it finds supporting information on the web, trains a
small correction, and keeps the update if it passes its checks. Your conversation
history and accepted corrections survive restarts.

Learned questions use the latest cumulative adapter; other requests use the
original model. Matching respects thinking mode and conversation context;
unfamiliar paraphrases may use the original model. Training, routing, and
adapter management reuse your existing model, without another base checkpoint.

Implemented architectures: **dense Qwen2 / Qwen2.5, Qwen3, Llama and Mistral
text models**. Other models still serve normally; only the correction step
needs an architecture whose adapter export has been checked numerically
against its PEFT reference, and the wrapper says so on startup.
Each native runtime passed two accumulating Qwen3 0.6B repairs with thinking
disabled. Thinking-enabled learning is implemented, but no native learning run
in that mode has passed yet.
Qwen3.8 uses a different hybrid architecture and is not supported yet.
Ollama Qwen3 learning currently requires its stock `qwen3:0.6b` chat template;
see [model and template limits](INTEGRATION.md#qwen3-model-and-template-support).

## Get started

Use Python 3.13+ and an existing local model. From the repository root, install
wrapper support once:

```sh
python -m pip install -e '.[wrap]'
```

Choose the command for your app:

| App | Start the wrapper |
| --- | --- |
| Ollama | `adaptible wrap ollama qwen3:0.6b` |
| llama.cpp | `adaptible wrap llama-cpp /path/to/model.gguf` |
| LM Studio | `adaptible wrap lm-studio /path/to/model.gguf` |
| vLLM | `adaptible wrap vllm /path/to/local-hf-model` |

Ollama or LM Studio must already be running its local server. For llama.cpp
or vLLM, install the runtime for your hardware; Adaptible starts it for you.
The first three apps use an existing GGUF. vLLM needs a local, unquantized
Hugging Face checkpoint with its tokenizer.

[Runtime setup options](INTEGRATION.md) cover executable paths, server
addresses, and authentication.

## Ask, flag, review

With the wrapper running, open another terminal:

```sh
python -m adaptible.cli
```

1. Ask: `What is the largest city in Morocco? Answer in one sentence.`
2. If the answer is wrong, type `/down` to flag it.
3. Type `/review` to review now and wait for the result. Review also starts
   automatically after two idle seconds.
4. If an update was kept, type `/new` and ask again to try the updated model.

A thumbs-down is enough: Adaptible searches for evidence, has the model write
a grounded correction, and tests the resulting update. You do not need to
write the right answer or prepare a document file. Chat requests wait while a
review is running.

The terminal enables Qwen3 thinking by default and displays its reasoning.
Review preserves the flagged answer's thinking mode. In that mode, training
uses a new reasoning draft based on the evidence and its checked final answer.

## Where corrections come from

Web lookup is on by default and needs no API key. Adaptible looks for agreement
between search excerpts from different websites before training. If the evidence is
missing, weak, or conflicting, it skips the update and records why. Sources and
review outcomes are available through `/history`.

The flagged question is sent to a search provider. Use `--no-web-search` for
offline or private work; then supply your own reference material through an
optional document collection or feedback note. See
[reference options](INTEGRATION.md#reference-options) for details.

## Use your existing client

Point an OpenAI-compatible chat client at `http://127.0.0.1:8000/v1` and use
the model name listed by `/v1/models`. Streaming is supported. Ollama clients
can also use `/api/chat` and `/api/generate`.

**Your client must send feedback to Adaptible to trigger learning.** An app's
existing thumbs-down button is not connected automatically. Use the terminal
client above or integrate the [feedback API](INTEGRATION.md#feedback-and-review).

## Current limits and verification

- Automatic correction currently handles short factual answers.
- The latest thinking-enabled llama.cpp test repaired one of the required two
  facts and preserved its protected baseline answers; the result was inconclusive.
  Other runtimes and the longer four-repair test remain pending. See
  [client settings](INTEGRATION.md#qwen3-model-and-template-support).
- Training needs more RAM than inference. Model size is not hardcoded, but
  larger models need their own capacity check.
- LM Studio and Ollama's Qwen3 path need space for a derived model file;
  Adaptible creates and manages it automatically. Original files remain unchanged.
- Corrections belong to the exact model used to train them. Scope matching and
  factual checks can make mistakes; scoped serving is not a claim that adapted
  weights preserve every unrelated answer.

The [repeated automatic repair test](INTEGRATION.md#repeated-automatic-repair-test)
requires two learned facts, two unseen wordings for each, retention, restart,
and streaming. Completed native results on Qwen3 0.6B:

| App | Two repairs, thinking disabled | Two repairs, thinking enabled | Four repairs, full budget |
| --- | --- | --- | --- |
| llama.cpp | Passed | Inconclusive: 1 of 2 repaired | Pending |
| LM Studio | Passed | Pending | Pending |
| Ollama | Passed | Pending | Pending |
| vLLM | Passed | Pending | Pending |

The longer test requires four fully repaired facts and continues through its
fixed review budget. The completed short tests do not establish stability over
the original multi-hour experiments; fast model-free tests cannot establish it either.

[Run the integration demos](INTEGRATION.md#repeatable-live-demo) or read
[how training and persistence work](IMPLEMENTATION.md).
