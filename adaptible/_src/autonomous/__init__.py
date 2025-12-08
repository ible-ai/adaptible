"""Autonomous Learning Node for Adaptible.

This module provides an autonomous learning node that can browse external
information sources, detect knowledge gaps or conflicts with its current
beliefs, and train itself on corrections.

Example usage:

    from adaptible.autonomous import AutonomousNode

    # Provide a search function that returns list of {title, snippet, url}
    def my_search(query: str) -> list[dict]:
        # Implement with Brave Search API, SerpAPI, etc.
        ...

    node = AutonomousNode(search_fn=my_search)

    # Run exploration cycles
    results = node.run(cycles=10, delay_seconds=2.0)

    # Or explore a specific topic
    result = node.explore_once("recent AI announcements")

    # Quiz the model
    answers = node.quiz(["Who is the CEO of OpenAI?", "What is Bitcoin's price?"])

Classes:
    AutonomousNode: The main autonomous learning node
    LearningEvent: Record of a single learning event
    NodeState: Persistent state of the node
    ExplorationResult: Result from one exploration cycle
"""

from .node import AutonomousNode, Claim, ExplorationResult, LearningEvent, NodeState

__all__ = [
    "AutonomousNode",
    "Claim",
    "ExplorationResult",
    "LearningEvent",
    "NodeState",
]
