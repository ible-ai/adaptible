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
node = AutonomousNode(search_fn=search_web)

# Run exploration cycles
results = node.run(cycles=10, delay_seconds=2.0)

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
    print(f"Learned: {event.question}")
    print(f"  Answer: {event.new_answer}")
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
node.run(cycles=20)

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
| `model` | None | StatefulLLM instance (lazy-loaded if None) |
| `state_path` | `autonomous_node_state.json` | Where to persist state |
| `training_iterations` | 25 | Iterations per correction |

## State Persistence

The node saves its state to JSON after each exploration cycle:

- Learning history (last 1000 events)
- Topics explored (last 100)
- Total updates and searches
- Start timestamp

This allows resuming across restarts:

```python
# First run
node = AutonomousNode(search_fn=search_web, state_path="my_node.json")
node.run(cycles=100)

# Later - continues from saved state
node = AutonomousNode(search_fn=search_web, state_path="my_node.json")
print(node.stats())  # Shows accumulated stats
node.run(cycles=100)  # Continues learning
```

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
3. **Model uncertainty** - Low confidence triggers "new knowledge" events
4. **Conflicting sources** - Each claim trained independently

## Files

```
autonomous/
├── __init__.py     # Public exports
├── node.py         # AutonomousNode implementation
└── README.md       # This file
```
