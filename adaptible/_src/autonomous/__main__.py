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

from absl import app, flags
from ddgs import DDGS

from . import AutonomousNode

_DEFAULT_TOPICS = (
    "current world leaders",
    "recent scientific discoveries",
    "today's major news events",
    "current stock market status",
    "recent sports results",
    "recently released movies",
    "current technology announcements",
    "recent space exploration news",
)
_CYCLES = flags.DEFINE_integer("cycles", 1, "Number of exploration cycles (default: 1)")
_TOPICS = flags.DEFINE_multi_string(
    "topics", _DEFAULT_TOPICS, "Specific topic to explore (optional)"
)
_STATE_PATH = flags.DEFINE_string(
    "state_path", "autonomous_node_state.json", "Path to save node state"
)


def main(_):
    ddgs = DDGS()

    def search(query: str) -> Sequence[Mapping[str, Any]]:
        """Web searcher.

        Args:
            query: The search query.

        Returns:
            List of search results with title, snippet, and url.
        """
        to_return = []
        for result in ddgs.text(query, max_results=5):
            to_return.append(
                {
                    "title": result["title"],
                    "url": result["href"],
                    "snippet": result["body"],
                }
            )

        return to_return

    # Create the node with mock search
    node = AutonomousNode(
        search_fn=search,
        state_path=_STATE_PATH.value,
        seed_topics=_TOPICS.value,
    )

    # Show initial stats
    stats = node.stats()
    print(f"Node stats: {stats}")
    print()

    # Run exploration
    if topics := _TOPICS.value:
        print(f"Exploring topic: {topics}")
        results = []
        for topic in topics:
            results.append(node.explore_once(topic))

    else:
        results = node.run(cycles=_CYCLES.value, verbose=True)

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
