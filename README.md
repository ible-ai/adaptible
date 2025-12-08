# Adaptible

Stateful LLM serving instances that self-reflect and learn from their mistakes during idle time.

## Vision

Self-improvement is itself a learnable trait. Different model instances undergoing online learning arrive at different end states—some become stronger self-learners than others. Adaptible explores this phenomenon by creating multiple instances that self-update, releasing them into production, and pruning the weaker learners while propagating successful ones.

The core hypothesis: by diversifying training bets autonomously and greedily sampling from winners, we create an evolutionary bottleneck that selects for models adept at self-improvement. The goal isn't just a model that learns—it's discovering which models *learn to learn*.

## What it does

Adaptible wraps an LLM in a server that:
1. Serves responses to user prompts
2. Stores interaction history
3. During idle periods, asks the model to critique and revise its past responses
4. Fine-tunes the model on those revisions using LoRA

This enables online learning: models that improve through use without full retraining.

## Requirements

- Python 3.13+
- Apple Silicon Mac (uses MLX for inference and training)

## Installation

```bash
pip install adaptible
```

Or from source:

```bash
git clone https://github.com/your-org/adaptible.git
cd adaptible
pip install -e .
```

## Quick Start

### Run the server

```bash
python -m adaptible.local
```

This starts a FastAPI server at `http://127.0.0.1:8000`. The web UI is available at `/static/`.

### Programmatic usage

```python
import asyncio
import adaptible

async def main():
    server = adaptible.MutableHostedLLM(host="127.0.0.1", port=8000)
    await server.up()

    # Server runs until you stop it
    await asyncio.sleep(3600)

    await server.down()

asyncio.run(main())
```

### Direct model usage

```python
import adaptible

model = adaptible.StatefulLLM()

# Generate a response
response = model.generate_response("What is the capital of France?")
print(response)
```

## API Endpoints

| Endpoint           | Method | Description                              |
| ------------------ | ------ | ---------------------------------------- |
| `/interact`        | POST   | Send a prompt, get a response            |
| `/stream_interact` | POST   | Stream the response                      |
| `/trigger_review`  | POST   | Start the self-correction cycle          |
| `/sync`            | GET    | Wait for background training to complete |
| `/history`         | GET    | Get all interactions                     |
| `/status`          | GET    | Health check                             |

### Example: Interact with the model

```bash
curl -X POST http://127.0.0.1:8000/interact \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

### Example: Trigger learning

```bash
# Trigger self-correction on recent interactions
curl -X POST http://127.0.0.1:8000/trigger_review

# Wait for training to complete
curl http://127.0.0.1:8000/sync
```

## How the self-correction works

1. The model receives a prompt containing its past interactions
2. It's asked to identify a response that could be improved and rewrite it
3. The rewritten response is validated (proper format, reasonable length, not garbage)
4. The model is fine-tuned on the improved response using LoRA
5. Training only updates the revision; original knowledge is preserved via masking

## Evaluation Framework

Adaptible includes a comprehensive evaluation harness for measuring self-correction effectiveness.

### Command line

```bash
# Run evaluation with default settings
python -m adaptible.eval

# Run with options
python -m adaptible.eval \
  --subset 20 \
  --shuffle \
  --category geography \
  --iterations 25
```

### Programmatic usage

```python
import adaptible.eval as eval

# Load the built-in dataset (100+ trivia questions)
dataset = eval.generate_default_dataset()

# Configure the experiment
config = eval.EvaluationConfig(
    name="my_experiment",
    training_iterations=25,
    train_ratio=0.8,
    shuffle=True,
)

# Run evaluation
harness = eval.EvaluationHarness()
result = harness.run(dataset, config)

# Generate HTML report
eval.generate_html_report(result, "/tmp/report.html")
```

The evaluation:
- Uses 100+ trivia questions across 6 categories (geography, science, history, math, language, miscellaneous)
- Splits data into train/holdout sets to measure generalization
- Measures baseline accuracy, improvement rate, retention rate, and holdout accuracy
- Generates an HTML report with detailed per-item results

## Configuration

`StatefulLLM` accepts these parameters:

| Parameter         | Default                                       | Description                  |
| ----------------- | --------------------------------------------- | ---------------------------- |
| `model_name`      | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B` | HuggingFace model path       |
| `learning_rate`   | `5e-5`                                        | Training learning rate       |
| `max_tokens`      | `128`                                         | Max tokens per response      |
| `epochs`          | `5`                                           | Training epochs per revision |
| `num_lora_layers` | `24`                                          | Number of LoRA layers        |
| `lora_parameters` | `{"rank": 32, "dropout": 0.0, "scale": 10.0}` | LoRA config                  |

## Limitations

- The default 1.5B model often produces malformed revision responses. Larger models work better.
- Training happens in-memory; model improvements are lost on restart.
- Apple Silicon only (MLX dependency).

## Project Structure

```
.
├── adaptible
│   ├── __init__.py                    # Public API
│   ├── eval/                          # Evaluation framework (public)
│   │   ├── __init__.py                # EvaluationHarness, TriviaDataset, etc.
│   │   └── __main__.py                # CLI entry point
│   ├── local.py                       # Local server runner
│   └── _src/                          # Internal implementation
│       ├── _api.py                    # FastAPI routes
│       ├── _classes.py                # Data models
│       ├── _llm.py                    # StatefulLLM class
│       ├── _server.py                 # Server entry point
│       ├── libs/                      # Internal libraries
│       └── tests/                     # Tests and eval implementation
├── examples/
└── pyproject.toml
```

## Running Tests

```bash
python -m unittest discover -s adaptible -p '*_test.py' -v
```

## License

Contact for information.
