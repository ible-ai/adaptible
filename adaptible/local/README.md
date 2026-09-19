# Local Server

FastAPI server hosting a `StatefulLLM` with endpoints for chat and for triggering self-correction.

## Overview

`MutableHostedLLM` (`server.py`) is a `uvicorn.Server` subclass with awaitable `up()`/`down()`. It serves the app built by `adaptible.Adaptible` (`adaptible/local/api.py`), which keeps interaction history in memory and tracks which interactions have not yet been reviewed.

## Quick Start

### Command Line

```bash
python -m adaptible.local          # http://127.0.0.1:8000
python -m adaptible.local.cli            # in another terminal: ask, /down, /up, /review, /new, /quit
```

`python -m adaptible.local` did not work before 1.0.0a3 (`adaptible.local` was a module, not a package); it is now a package. The uvicorn ≥0.36 startup crash was fixed in the same release.

### Programmatic Usage

```python
import asyncio
import adaptible

async def main():
    server = adaptible.local.MutableHostedLLM(host="127.0.0.1", port=8000)
    await server.up()
    print("Server ready at http://127.0.0.1:8000")
    await asyncio.sleep(3600)
    await server.down()

asyncio.run(main())
```

### Custom FastAPI App

```python
import adaptible

app = adaptible.Adaptible().app

@app.get("/custom")
def custom_endpoint():
    return {"message": "Custom endpoint"}

server = adaptible.local.MutableHostedLLM(app=app)
```

`Adaptible(model=...)` accepts anything satisfying `ModelProtocol` (`ok`, `generate_response`, `stream_response`, `self_correct_and_train`); `adaptible/local/api_test.py` injects a stub this way.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/interact` | POST | Send prompt, get complete response; recorded as an unreviewed interaction |
| `/stream_interact` | POST | Send prompt, stream response chunks |
| `/trigger_review` | POST | Hand all unreviewed interactions to `model.self_correct_and_train` in an `asyncio` task (via `asyncio.to_thread`) and return immediately |
| `/sync` | GET | Await every outstanding `/trigger_review` task, log any failure, then poll `model.ok` until training is done |
| `/history` | GET | All interactions |
| `/status` | GET | Health check |

Fixed in 1.0.0a3: `/trigger_review` previously created the training coroutine without awaiting or scheduling it, so no training ever ran; `/sync` only polled `model.ok`. A failed training task is logged by `/sync` and the model keeps its pre-training weights.

### Example Requests

```bash
curl -X POST http://127.0.0.1:8000/interact \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

curl -X POST http://127.0.0.1:8000/trigger_review
curl http://127.0.0.1:8000/sync
curl http://127.0.0.1:8000/history
```

### Streaming Example

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/stream_interact",
    json={"prompt": "Explain quantum computing"},
    stream=True,
)
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk:
        print(chunk, end="", flush=True)
```

## Configuration

```python
server = adaptible.local.MutableHostedLLM(
    host="127.0.0.1",
    port=8000,
    app=None,     # optional custom FastAPI app
)
```

If `app` is not provided, a default `Adaptible()` is created with `StatefulLLM()`. That constructor loads `<outputs>/autonomous/checkpoint` if it exists (`<outputs>` is `$ADAPTIBLE_OUTPUTS_DIR` or `<cwd>/outputs`), so a server started after an autonomous run serves the autonomous node's trained weights. Build the model with `StatefulLLM(model_path=None)` and pass `Adaptible(model=...)` for a fresh base model.

The model's chat history records both user and assistant turns (assistant turns were dropped before 1.0.0a3).

## Terminal client

`python -m adaptible.local.cli [--url URL] [line ...]` (`adaptible/local/cli.py`) is a small REPL over these endpoints: typed text goes to `/stream_interact` and is printed as it streams; `/down` and `/up` post `/feedback` for the last answer; `/review` posts `/trigger_review` then blocks on `/sync` and prints the elapsed time; `/new` posts `/new_chat`; `/quit` exits. Lines passed as arguments run in order without the prompt.

## Files

```text
adaptible/local/
├── __init__.py          # Public exports (MutableHostedLLM)
├── __main__.py          # Server entry point (python -m adaptible.local)
├── README.md            # This file
├── api.py               # Adaptible: routes and interaction history
├── cli.py               # Terminal client (python -m adaptible.local.cli)
├── server.py            # MutableHostedLLM implementation
└── *_test.py            # Model-free tests (api, cli, local)
```

## Limitations

- Interaction history is in-memory only (lost on restart); trained weights persist only if saved to a checkpoint
- Single model instance, no load balancing
- No authentication
- Apple Silicon only (MLX dependency)
- Self-correction through this server has not been measured; see the top-level README
