"""Demo script for the Autonomous Learning Node.

This script demonstrates how to use the AutonomousNode with a mock search
function. In production, you would replace the mock search with a real
search API (e.g., Brave Search, SerpAPI, Tavily).

Usage:
    python -m adaptible.autonomous [--cycles N] [--topic TOPIC]

Examples:
    # Run 3 exploration cycles with random topics
    python -m adaptible.autonomous --cycles 3

    # Explore a specific topic
    python -m adaptible.autonomous --topic "recent AI announcements"

    # Run demo without loading the model (just shows how it would work)
    python -m adaptible.autonomous --dry-run
"""

from collections.abc import Mapping, Sequence
from typing import Any

import vizible
import time
from absl import app, flags
from ddgs import DDGS

from .node import AutonomousNode
from .._llm import MODEL_PATH

# Specific factual queries - use "current" or 2025 to ensure we're testing
# information beyond the model's training cutoff
_DEFAULT_TOPICS = (
    # Politics / Leaders (current = dynamic)
    "Who is the current president of the United States?",
    "Who is the current prime minister of the United Kingdom?",
    "Who is the current chancellor of Germany?",
    # Sports (2025 = beyond training cutoff)
    "Who won Super Bowl 2025?",
    "Who won the 2025 NBA Finals?",
    "Who won the 2025 World Series?",
    # Entertainment (2025)
    "What movie won Best Picture at the 2025 Oscars?",
    "What is the highest grossing movie of 2025?",
    # Current events / prices (dynamic)
    "What is the current price of Bitcoin?",
    "What is the current price of Tesla stock?",
    # Tech (latest = dynamic)
    "What is the latest iPhone model?",
    "What is the latest version of iOS?",
    # Recent deaths / status (things that changed recently)
    "Is Jimmy Carter still alive?",
    "Is Pope Francis still the Pope?",
)
_CYCLES = flags.DEFINE_integer("cycles", 1, "Number of exploration cycles (default: 1)")
_TOPICS = flags.DEFINE_multi_string(
    "topics", _DEFAULT_TOPICS, "Specific topic to explore (optional)"
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "outputs/autonomous/state.json", "Path to save node state"
)
_MODEL_PATH = flags.DEFINE_string(
    "model_path", str(MODEL_PATH), "Path to save node state"
)


def main(_):
    vizible.blue("\n--- Starting Autonomous Learning Node ---")
    ddgs_client = DDGS()

    def search(query: str) -> Sequence[Mapping[str, Any]]:
        """Web searcher.

        Args:
            query: The search query.

        Returns:
            List of search results with title, snippet, and url.
        """
        to_return = []
        time.sleep(1)
        news_results = ddgs_client.news(query, max_results=2)
        time.sleep(1)
        text_results = ddgs_client.text(query, max_results=2)
        seen_urls = set()
        for result in [*news_results, *text_results]:
            result = {**result}
            url = result.get("href") or result.get("url")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            to_return.append(
                {
                    "title": result["title"],
                    "url": url,
                    "snippet": result["body"],
                }
            )

        return to_return

    # Create the node with mock search
    node = AutonomousNode(
        search_fn=search,
        state_path=_OUTPUT_PATH.value,
        seed_topics=_TOPICS.value,
        model_path=_MODEL_PATH.value,
    )

    # Show initial stats
    stats = node.stats()
    print(f"Node stats: {stats}")
    print()

    # Run exploration
    topics = _TOPICS.value
    if not topics:
        topics = [None] * _CYCLES.value
    print(f"Exploring topic: {topics}")
    results = []
    for topic in topics:
        results.append(node.explore_once(topic))

    results = node.run(topics=topics, verbose=True)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_claims = sum(r.claims_found for r in results)
    total_updates = sum(r.updates_made for r in results)
    print(f"Cycles run: {len(results)}")
    print(f"Total claims found: {total_claims}")
    print(f"Total updates made: {total_updates}")
    print()
    print(f"Final stats: {node.stats()}")

    return 0


if __name__ == "__main__":
    app.run(main)
