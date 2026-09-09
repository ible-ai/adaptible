# Autonomous Learning Node

An LLM that learns from external information sources in real-time.

## Overview

The autonomous node implements a continuous learning loop:

1. **Search** - Query external sources for current information
2. **Extract** - Parse factual claims from search results
3. **Compare** - Ask the model what it currently believes about each claim
4. **Detect** - Identify knowledge gaps or conflicts
5. **Train** - Update weights via LoRA when corrections are needed
6. **Record** - Log what was learned for analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autonomous Learning Node                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │  Search  │────▶│ Extract  │────▶│ Compare  │               │
│   │  Web     │     │ Claims   │     │ Beliefs  │               │
│   └──────────┘     └──────────┘     └────┬─────┘               │
│                                          │                      │
│                                          ▼                      │
│                                   ┌──────────┐                  │
│                                   │ Conflict?│                  │
│                                   └────┬─────┘                  │
│                                        │                        │
│                          ┌─────────────┴─────────────┐          │
│                          ▼                           ▼          │
│                    ┌──────────┐               ┌──────────┐      │
│                    │  Train   │               │   Skip   │      │
│                    │ (LoRA)   │               │          │      │
│                    └──────────┘               └──────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Example

```python
from adaptible.autonomous import AutonomousNode

# You must provide a search function
def search_web(query: str) -> list[dict]:
    """Return list of {title, snippet, url} dicts."""
    # Implement using Brave Search API, SerpAPI, or similar
    import requests
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": "YOUR_API_KEY"},
        params={"q": query, "count": 5},
    )
    data = response.json()
    return [
        {"title": r["title"], "snippet": r["description"], "url": r["url"]}
        for r in data.get("web", {}).get("results", [])
    ]

# Create the node
node = AutonomousNode(search_fn=search_web, seed_topics=["recent SpaceX launches"])

# Run exploration cycles (one per topic; None lets the node pick a topic)
results = node.run(topics=["recent SpaceX launches", None, None])

# Check what was learned
print(node.stats())
```

### Exploring Specific Topics

```python
# Explore a specific topic
result = node.explore_once("recent SpaceX launches")

print(f"Claims found: {result.claims_found}")
print(f"Updates made: {result.updates_made}")

for event in result.events:
    print(f"[{event.event_type}] trained={event.trained} {event.question}")
    print(f"  Verified: {event.verified_answer}")
```

### Quizzing the Model

```python
# Quiz before and after to measure learning
questions = [
    "Who won the 2025 Super Bowl?",
    "What is the current price of Bitcoin?",
]

# Before learning
pre_answers = node.quiz(questions)

# Run learning cycles
node.run(topics=[None] * 20)

# After learning
post_answers = node.quiz(questions)

# Compare
for q in questions:
    print(f"Q: {q}")
    print(f"  Before: {pre_answers[q]['answer'][:100]}")
    print(f"  After: {post_answers[q]['answer'][:100]}")
```

## Integration with Adaptible

The autonomous node integrates directly with Adaptible's `StatefulLLM`:

- Uses the same model and tokenizer
- Training uses LoRA via `revise.make_collated_training_example()`
- State is persisted to JSON between runs
- Can share a model instance with other Adaptible components

```python
import adaptible
from adaptible.autonomous import AutonomousNode

# Share a model instance
model = adaptible.StatefulLLM()

node = AutonomousNode(
    search_fn=my_search,
    model=model,  # Reuse existing model
    training_iterations=25,
)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `search_fn` | (required) | Function returning search results |
| `seed_topics` | (required) | Topics to pick from when none is given |
| `model` | None | StatefulLLM instance (lazy-loaded if None) |
| `model_path` | `<outputs>/autonomous/checkpoint` | Checkpoint to load/save |
| `state_path` | `<outputs>/autonomous/state.json` | Where to persist state |
| `log_dir` | `<outputs>/autonomous/logs/` | Per-day text logs (`YYYYMMDD.txt`) |
| `db_path` | `<outputs>/adaptible.db` | Experiment database |
| `training_iterations` | 25 | Iterations per correction |
| `train_on_new_knowledge` | False | Also train on knowledge gaps (see below) |

`<outputs>` is `$ADAPTIBLE_OUTPUTS_DIR` if set, else `<cwd>/outputs` (see
`adaptible/_src/_paths.py`). The CLI exposes these as `--output_path`,
`--model_path`, `--node_log_dir` and `--train_on_new_knowledge`.

## What gets trained on

Earlier runs trained on whatever the extractor produced, which put page
boilerplate into the weights ("NBCNews.com provides the latest top news
stories.", "What is the main focus of the content?"). Two filters now sit
between extraction and training.

### Claim plausibility filter

`_claim_is_plausible(claim, snippet)` is a pure, model-free check applied to
every extracted claim. A claim is dropped (and logged as `DROPPED`) if:

- the model's extraction did not contain both a `Q:` and an `A:` line;
- the question is shorter than 15 characters or does not end in `?`;
- the question or answer mentions a site name (`nbcnews.com`, `apnews.org`, ...)
  or page boilerplate ("the content", "this page", "top stories",
  "latest news", "main focus", ...);
- the answer is not grounded in the snippet: at least one content token
  (4+ characters, not a stopword) from the answer must appear in the snippet.

### Training policy

For each surviving claim the node asks the model what it believes and runs the
fact-checker prompt. The outcome is one of:

| Model's prior belief | Fact-checker | Event | Trained by default |
|---|---|---|---|
| Answer with MEDIUM/HIGH confidence, consistent | no correction | (none, logged `CONSISTENT`) | no |
| Answer with MEDIUM/HIGH confidence, contradicted | needs correction | `correction` | **yes** |
| Empty answer or LOW confidence (knowledge gap) | (skipped) | `new` | no |

By default only **corrections** change the weights: the model held a belief and
a source contradicted it. Knowledge-gap `new` events are still recorded in
`state.json` (with `trained: false`) and the database so they can be reviewed,
but they do not train unless `train_on_new_knowledge=True` /
`--train_on_new_knowledge` is passed.

## State Persistence

The node saves its state to `state_path` after each exploration cycle:

- Learning history (last 1000 events, including untrained `new` events)
- Topics explored (last 100)
- Total updates and searches
- Start timestamp

Files written before the `before_training_answer` / `after_training_answer`
field names (which used `old_answer` / `new_answer`) still load.

This allows resuming across restarts:

```python
# First run (state at <outputs>/autonomous/state.json)
node = AutonomousNode(search_fn=search_web, seed_topics=topics)
node.run(topics=[None] * 100)

# Later - continues from saved state
node = AutonomousNode(search_fn=search_web, seed_topics=topics)
print(node.stats())  # Shows accumulated stats
node.run(topics=[None] * 100)  # Continues learning
```

## Logs

Each run appends to `<outputs>/autonomous/logs/YYYYMMDD.txt`: one line per
topic (result and claim counts), per dropped claim, and per learning event
(type, whether it trained, verified answer, before/after answers, URL). Logs and
the checkpoint are git-ignored; `state.json` is tracked as a published result.

## Search Function Requirements

Your search function must:

1. Accept a `str` query parameter
2. Return a `list[dict]` with each dict containing:
   - `title`: Title of the result
   - `snippet` or `description`: Text content to extract claims from
   - `url`: Source URL

Example search providers:
- [Brave Search API](https://brave.com/search/api/)
- [SerpAPI](https://serpapi.com/)
- [Tavily](https://tavily.com/)

## Relationship to `adaptible.eval`

The `adaptible.eval` module provides **offline evaluation** - measuring how well the model learns from a fixed dataset of trivia questions.

The autonomous node provides **online learning** - continuously learning from live external sources.

They complement each other:
- Use `eval` to benchmark self-correction effectiveness
- Use `autonomous` for real-world deployment

## Failure Modes

The node handles several failure modes gracefully:

1. **Search failures** - Logged in result.error, cycle continues
2. **No claims extracted** - Normal for some topics, cycle continues
3. **Model uncertainty** - Low confidence records a "new" event (trained only with `train_on_new_knowledge`)
4. **Junk extractions** - Dropped by `_claim_is_plausible` before the model is consulted
5. **Conflicting sources** - Each claim trained independently

## Files

```
autonomous/
├── __init__.py     # Public exports
├── __main__.py     # CLI entrypoint (absl flags)
├── node.py         # AutonomousNode implementation
└── README.md       # This file

outputs/autonomous/          # runtime state (or $ADAPTIBLE_OUTPUTS_DIR/autonomous/)
├── state.json               # NodeState, tracked in git
├── logs/YYYYMMDD.txt        # git-ignored
└── checkpoint/              # git-ignored
```

Tests: `adaptible/tests/autonomous_test.py` (model-free).
