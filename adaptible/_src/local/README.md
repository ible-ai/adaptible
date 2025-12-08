# Local Server

FastAPI server for hosting a stateful LLM with self-correction capabilities.

## Overview

The `local` module provides `MutableHostedLLM`, a server that wraps Adaptible's `StatefulLLM` with HTTP endpoints for:
- Generating responses to prompts
- Streaming responses
- Triggering self-correction cycles
- Accessing interaction history
- Synchronizing with background training

## Quick Start

### Command Line

```bash
# Start server on default port (8000)
python -m adaptible.local

# Access web UI
open http://127.0.0.1:8000/static/
```

### Programmatic Usage

```python
import asyncio
import adaptible

async def main():
    # Create and start server
    server = adaptible.local.MutableHostedLLM(
        host="127.0.0.1",
        port=8000
    )
    await server.up()
    
    # Server is now running
    print("Server ready at http://127.0.0.1:8000")
    
    # Keep running
    await asyncio.sleep(3600)
    
    # Shutdown
    await server.down()

asyncio.run(main())
```

### Custom FastAPI App

```python
from fastapi import FastAPI
import adaptible

# Create custom app with Adaptible routes
app = adaptible.Adaptible().app

# Add your own routes
@app.get("/custom")
def custom_endpoint():
    return {"message": "Custom endpoint"}

# Host it
server = adaptible.local.MutableHostedLLM(app=app)
```

## API Endpoints

Once running, the server exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/interact` | POST | Send prompt, get complete response |
| `/stream_interact` | POST | Send prompt, stream response chunks |
| `/trigger_review` | POST | Start self-correction on recent interactions |
| `/sync` | GET | Wait for background training to complete |
| `/history` | GET | Retrieve all interaction history |
| `/status` | GET | Health check |
| `/static/` | GET | Web UI (HTML/CSS/JS) |

### Example Requests

```bash
# Send a prompt
curl -X POST http://127.0.0.1:8000/interact \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

# Trigger self-correction
curl -X POST http://127.0.0.1:8000/trigger_review

# Wait for training to finish
curl http://127.0.0.1:8000/sync

# Get interaction history
curl http://127.0.0.1:8000/history
```

### Streaming Example

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/stream_interact",
    json={"prompt": "Explain quantum computing"},
    stream=True
)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk:
        print(chunk, end="", flush=True)
```

## Configuration

```python
server = adaptible.local.MutableHostedLLM(
    host="127.0.0.1",      # Bind address
    port=8000,              # Port number
    app=None,               # Optional custom FastAPI app
)
```

If `app` is not provided, a default `Adaptible()` instance is created with a fresh `StatefulLLM`.

## Web UI

The server includes a web-based chat interface at `/static/`:
- Terminal-style interface
- Real-time streaming responses
- Interaction history display
- Trigger learning button

## Files

```
local/
├── __init__.py          # Public exports (MutableHostedLLM)
├── __main__.py          # CLI entry point
├── README.md            # This file
└── _src/
    └── _server.py       # MutableHostedLLM implementation
```

## Relationship to Other Modules

- **`adaptible.Adaptible`** - Provides the FastAPI app with routes
- **`adaptible.StatefulLLM`** - The underlying model being served
- **`adaptible.revise`** - Used for self-correction logic
- **`adaptible/_src/static/`** - Web UI assets (HTML/CSS/JS)

## Typical Workflow

```python
import asyncio
import adaptible

async def run_server():
    # Start server
    server = adaptible.local.MutableHostedLLM()
    await server.up()
    
    try:
        # Run indefinitely
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await server.down()

asyncio.run(run_server())
```

## Development

The server uses:
- **FastAPI** for HTTP routing
- **Uvicorn** for ASGI serving
- **Asyncio** for concurrent request handling
- **StatefulLLM** for model inference and training

## Limitations

- Model state is in-memory only (lost on restart)
- Single model instance (no load balancing)
- Apple Silicon only (MLX dependency)
- No authentication/authorization built-in
