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

### Evaluation Framework Usage

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
- Persists all results to SQLite database for analysis
- Generates an HTML report with detailed per-item results

## Experiment Database

All experiments are persisted to a SQLite database (`outputs/adaptible.db`) for structured analysis.

### Schema

```text
examples
├── canonical_id, question, ground_truth_answer
├── key_terms, category, difficulty
├── source_type (static_trivia | web_scrape)
├── valid_at (NULL for timeless facts, DATE for time-sensitive)
└── created_at

experiments
├── name, experiment_type (eval | autonomous)
├── config_json, model_checkpoint
└── started_at, completed_at

responses
├── example_id, experiment_id
├── response_text, response_raw
├── phase (baseline | post_training)
├── token_count, max_tokens, truncated
└── created_at

training_events
├── example_id, experiment_id
├── training_iterations, training_time_seconds
└── created_at
```

### Interactive exploration

```bash
# Interactive Python REPL with database helpers
python scripts/explore_db.py

# Create demo data first
python scripts/explore_db.py --demo

# Print summary for a specific experiment
python scripts/explore_db.py --summary 1
```

### Jupyter notebook

The `notebooks/explore_experiments.ipynb` notebook provides SQL-based exploration:

```python
from adaptible import Database

db = Database()

# Get experiment metrics
metrics = db.compute_metrics(experiment_id=1)
print(f"Improvement rate: {metrics['improvement_rate']:.1%}")

# Find regressions (items that got worse after training)
for r in db.get_regressions(experiment_id=1):
    print(f"{r['example'].canonical_id}: {r['baseline_text']} → {r['post_text']}")

# Export LLM-friendly summary
print(db.export_experiment_summary(experiment_id=1))
```

### Tracked metrics

- **Improvement rate**: Fraction of wrong→right transitions among trained items
- **Retention rate**: Fraction of right→right among trained items
- **Regression count**: Items that went right→wrong (indicates forgetting)
- **Stuck count**: Items that remained wrong→wrong despite training
- **Truncation detection**: Responses that hit the token limit

## Autonomous Learning

The autonomous module enables continuous self-improvement by scraping claims from the web, testing them, and training on corrections.

```bash
# Run the autonomous learning loop
python -m adaptible.autonomous
```

### How it works

1. **Claim generation**: Uses web search to find factual claims with ground truth
2. **Baseline test**: Model attempts to answer without seeing the ground truth
3. **Confidence scoring**: Model rates its confidence in its answer
4. **Belief conflict**: Compares model's answer against ground truth
5. **Training**: When wrong, trains on the correct answer
6. **Verification**: Re-tests to confirm learning

### Configuration

The autonomous node persists state to `outputs/autonomous/state.json` and logs to `outputs/autonomous/logs/`. All training events are also recorded in the experiment database.

```python
from adaptible.autonomous import AutonomousNode

node = AutonomousNode(
    max_tokens=2048,
    training_iterations=25,
    db_path="outputs/adaptible.db",
)
node.run()
```

## Meta-Learning Experiments

The core hypothesis: **self-improvement is itself a learnable trait**. Different model instances undergoing online learning arrive at different end states—some become stronger self-learners than others.

### Running Meta-Learning Experiments

```python
import adaptible.eval as eval

# Configure experiment
config = eval.MetaLearningConfig(
    name="meta_experiment",
    seeds=[42, 123, 456, 789, 1011],  # Run N instances
    checkpoint_interval=10,  # Checkpoint every 10 training events
    training_iterations=25,
    train_ratio=0.8,
)

# Load dataset
dataset = eval.generate_default_dataset()

# Run experiment
experiment = eval.MetaLearningExperiment()
result = experiment.run(dataset, config)

# Analyze results
print(f"Best seed: {result.best_seed}")
print(f"Score variance: {result.score_variance}")

# Save for later analysis
result.save("outputs/meta_experiment.json")
```

### Meta-Learning Score

The meta-learning score measures how learning efficiency changes over time:

```python
meta_learning_score = (late_improvement_rate - early_improvement_rate)
                    + (early_forgetting_rate - late_forgetting_rate)
```

- **Score > 0**: Model is learning to learn better (improvements accelerate, forgetting decelerates)
- **Score < 0**: Model is degrading (improvements slow down, forgetting accelerates)
- **High variance across seeds**: Meta-learning ability is sensitive to initialization

### Tracked Metrics per Checkpoint

- `improvement_rate`: Fraction of wrong→right transitions
- `forgetting_rate`: Fraction of right→wrong transitions
- `net_learning`: improved - regressed
- `post_accuracy`: Current accuracy on trained items

## Observed Results

Evaluation runs on a 1.5B parameter model show net positive self-improvement:

- **Baseline**: The model answers roughly half of trivia questions correctly before any training
- **Improvement**: After self-correction training, the model shows a positive delta on items it was trained on
- **Retention**: The model retains most of its existing correct answers, with some forgetting
- **Generalization**: Improvements do not transfer to untrained items (as expected for independent factual questions)

The improvement rate exceeds the forgetting rate, indicating the self-correction loop produces net learning. The model successfully identifies some of its own errors, generates corrections, and updates its weights to reflect those corrections.

This is early-stage work. The gains are modest but demonstrate that end-to-end self-improvement is achievable with small models and LoRA fine-tuning.

## General Configuration

`StatefulLLM` accepts these parameters:

| Parameter                        | Default                                       | Description                            |
| -------------------------------- | --------------------------------------------- | -------------------------------------- |
| `model_name`                     | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B` | HuggingFace model path                 |
| `learning_rate`                  | `5e-5`                                        | Training learning rate                 |
| `max_tokens`                     | `2048`                                        | Max tokens per response                |
| `epochs`                         | `5`                                           | Training epochs per revision           |
| `num_lora_layers`                | `24`                                          | Number of LoRA layers                  |
| `lora_parameters`                | `{"rank": 32, "dropout": 0.0, "scale": 10.0}` | LoRA config                            |
| `loop_detection_sequence_length` | `8`                                           | Token sequence length for loop check   |
| `loop_detection_max_repetitions` | `3`                                           | Repetitions before stopping generation |

## Limitations

- The default 1.5B model achieves modest but net-positive self-improvement. Gains are incremental rather than dramatic.
- Apple Silicon only (MLX dependency).

## Project Structure

```text
.
├── adaptible/
│   ├── __init__.py                    # Public API
│   ├── local.py                       # Local server runner
│   └── _src/                          # Internal implementation
│       ├── _api.py                    # FastAPI routes
│       ├── _classes.py                # Data models
│       ├── _llm.py                    # StatefulLLM class with loop detection
│       ├── _server.py                 # Server entry point
│       ├── db.py                      # SQLite database layer
│       ├── autonomous/                # Autonomous learning module
│       │   ├── __init__.py
│       │   ├── __main__.py            # CLI entry point
│       │   └── node.py                # AutonomousNode class
│       ├── eval/                      # Evaluation framework
│       │   ├── __init__.py            # EvaluationHarness, TriviaDataset, etc.
│       │   ├── __main__.py            # CLI entry point
│       │   ├── dataset.py             # Trivia dataset
│       │   ├── harness.py             # Evaluation harness
│       │   └── meta.py                # Meta-learning experiments
│       ├── revise/                    # Self-correction logic
│       │   └── revise.py              # Revision prompts and training examples
│       └── tests/                     # Unit tests
├── notebooks/
│   └── explore_experiments.ipynb      # SQL-based experiment analysis
├── scripts/
│   └── explore_db.py                  # Interactive database explorer
├── outputs/
│   ├── adaptible.db                   # Experiment database
│   └── autonomous/                    # Autonomous node state and logs
├── examples/
└── pyproject.toml
```

## Running Tests

```bash
python -m unittest discover -s adaptible -p '*_test.py' -v
```

## License

Contact for information.
