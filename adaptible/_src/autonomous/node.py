"""Autonomous Learning Node

An LLM node that browses the web, encounters information, compares it to its
current beliefs, and updates its weights when it learns something new.

This module integrates directly with Adaptible's StatefulLLM for actual
weight updates via LoRA fine-tuning.
"""

import dataclasses
import json
import random
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List

from .._classes import InteractionHistory, TrainingExample
from .._llm import MAX_TOKENS, StatefulLLM
from ..revise import make_collated_training_example


@dataclasses.dataclass
class LearningEvent:
    """Record of a learning event."""

    timestamp: str
    question: str
    old_answer: str | None
    new_answer: str
    source: str
    source_url: str | None
    event_type: str  # "new", "correction", "reinforcement"


@dataclasses.dataclass
class Claim:
    """A factual claim extracted from search results."""

    question: str
    answer: str
    source: str
    url: str


@dataclasses.dataclass
class NodeState:
    """Persistent state of the autonomous node."""

    learning_history: List[LearningEvent] = dataclasses.field(default_factory=list)
    topics_explored: List[str] = dataclasses.field(default_factory=list)
    total_updates: int = 0
    total_searches: int = 0
    started_at: str = ""

    def save(self, path: Path):
        """Save state to JSON using dataclasses.dataclasses.asdict()."""
        # Trim to keep state file manageable
        trimmed_state = NodeState(
            learning_history=self.learning_history[-1000:],  # Keep last 1000
            topics_explored=self.topics_explored[-100:],  # Keep last 100
            total_updates=self.total_updates,
            total_searches=self.total_searches,
            started_at=self.started_at,
        )
        path.write_text(json.dumps(dataclasses.asdict(trimmed_state), indent=2))

    @classmethod
    def load(cls, path: Path) -> "NodeState":
        """Load state from JSON."""
        if not path.exists():
            return cls(started_at=datetime.now().isoformat())
        data = json.loads(path.read_text())
        state = cls(
            topics_explored=data.get("topics_explored", []),
            total_updates=data.get("total_updates", 0),
            total_searches=data.get("total_searches", 0),
            started_at=data.get("started_at", datetime.now().isoformat()),
        )
        for e in data.get("learning_history", []):
            state.learning_history.append(LearningEvent(**e))
        return state


@dataclasses.dataclass
class ExplorationResult:
    """Result from a single exploration cycle."""

    timestamp: str
    topic: str
    claims_found: int = 0
    beliefs_checked: int = 0
    updates_made: int = 0
    events: List[LearningEvent] = dataclasses.field(default_factory=list)
    error: str | None = None


class AutonomousNode:
    """An LLM node that learns autonomously from external information.

    The core loop:
    1. Pick a topic to explore
    2. Search for current information (via provided search function)
    3. Extract factual claims from search results
    4. For each claim, ask the model: "What do you believe about this?"
    5. If there's a conflict or gap, train the model on the correction
    6. Record what was learned
    7. Repeat

    This integrates with Adaptible's StatefulLLM for actual weight updates.
    """

    def __init__(
        self,
        search_fn: Callable[[str], Sequence[Mapping[str, Any]]],
        seed_topics: Sequence[str],
        model: StatefulLLM | None = None,
        state_path: Path | str = Path("autonomous_node_state.json"),
        training_iterations: int = 25,
        max_tokens: int = MAX_TOKENS,
    ):
        """Initialize the autonomous learning node.

        Args:
            search_fn: Function that takes a query string and returns a list of
                search results as dicts with 'title', 'snippet'/'description', and 'url'.
            seed_topics: Topics to explore.
            model: Optional pre-loaded StatefulLLM. If None, will be created on first use.
            state_path: Where to persist node state between runs.
            training_iterations: Number of training iterations per correction.
            max_tokens: Max tokens for generation. Must be high enough for <think> tags.
        """
        self.search = search_fn
        self._model = model
        self.state_path = Path(state_path)
        self.state = NodeState.load(self.state_path)
        self.training_iterations = training_iterations
        self._epochs_per_call = 5  # Matches StatefulLLM._epochs
        self.seed_topics = seed_topics
        self._max_tokens = max_tokens

    @property
    def model(self) -> StatefulLLM:
        """Lazy-load model on first access."""
        if self._model is None:
            print("Loading model...")
            self._model = StatefulLLM()
            self._model._model_is_stable = True
        return self._model

    def _ask(self, prompt: str, max_tokens: int | None = None) -> str:
        """Query the model.

        Args:
            prompt: The prompt to send to the model.
            max_tokens: Override max_tokens. If None, uses self._max_tokens.
        """
        if max_tokens is None:
            max_tokens = self._max_tokens
        response = self.model.generate_response(
            prompt, use_history=False, max_tokens=max_tokens
        )
        return response or ""

    def _train_on_correction(
        self, question: str, old_answer: str, correct_answer: str
    ) -> bool:
        """Train the model on a correction using Adaptible's training pipeline.

        Args:
            question: The question that was asked.
            old_answer: The model's original (incorrect) response.
            correct_answer: The correct answer to train on.

        Returns:
            True if training succeeded.
        """
        # Create an interaction history entry with the original response
        interactions = [
            InteractionHistory(
                idx=0,
                user_input=question,
                llm_response=old_answer,
                reviewed=False,
                timestamp=0.0,
            ),
        ]

        # Format the correction as a revision (matching Adaptible's format)
        valid_revision = f"[[0]] {correct_answer} [[/0]]"

        # Create the training example
        example: TrainingExample = make_collated_training_example(
            valid_revision, interactions, self.model._tokenizer
        )

        # Train for the configured number of iterations
        calls = self.training_iterations // self._epochs_per_call
        for _ in range(calls):
            self.model._train(example, verbose=False)

        return True

    def _extract_claims(
        self, search_results: Sequence[Mapping[str, Any]], topic: str
    ) -> Sequence[Claim]:
        """Extract factual claims from search results as Q&A pairs."""
        claims: list[Claim] = []

        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("description", ""))
            url = result.get("url", result.get("link", ""))

            if not snippet:
                continue

            # Ask the model to extract factual claims
            extraction_prompt = f"""From this search result about "{topic}", extract 1-2 specific factual claims as question-answer pairs.

Title: {title}
Content: {snippet}

Format each as:
Q: [specific factual question]
A: [factual answer from the content]

Only extract claims that are clearly stated in the content. Be specific and concise."""

            response = self._ask(extraction_prompt)

            # Parse Q&A pairs from response
            lines = response.strip().split("\n")
            current_q = None
            for line in lines:
                line = line.strip()
                if line.startswith("Q:"):
                    current_q = line[2:].strip()
                elif line.startswith("A:") and current_q:
                    answer = line[2:].strip()
                    if current_q and answer and len(answer) > 5:
                        claims.append(
                            Claim(
                                question=current_q,
                                answer=answer,
                                source=title,
                                url=url,
                            )
                        )
                    current_q = None

        return claims

    def _check_belief(self, question: str) -> tuple[str, float]:
        """Ask the model what it currently believes about a question.

        Returns:
            Tuple of (answer, confidence) where confidence is 0-1.
        """
        prompt = f"""Answer this question based only on what you know. If you don't know or are uncertain, say "I don't know."

Question: {question}

Give a brief, direct answer."""

        response = self._ask(prompt)

        # Estimate confidence based on response
        response_lower = response.lower()
        if any(
            x in response_lower
            for x in ["i don't know", "uncertain", "not sure", "cannot"]
        ):
            confidence = 0.2
        elif any(x in response_lower for x in ["i think", "probably", "might be"]):
            confidence = 0.5
        else:
            confidence = 0.8

        return response, confidence

    def _beliefs_conflict(self, belief: str, new_info: str, question: str) -> bool:
        """Check if current belief conflicts with new information."""
        if "i don't know" in belief.lower():
            return True  # Knowledge gap counts as needing update

        prompt = f"""Do these two answers to the same question conflict or contradict each other?

Question: {question}

Answer 1: {belief}
Answer 2: {new_info}

Reply with only "YES" if they conflict/contradict, or "NO" if they are compatible or say the same thing."""

        response = self._ask(prompt)
        return "YES" in response.upper()

    def _pick_topic(self) -> str:
        """Pick a topic to explore."""
        # 30% chance to follow up on a recent topic
        if self.state.topics_explored and random.random() < 0.3:
            return random.choice(self.state.topics_explored[-10:])
        return random.choice(self.seed_topics)

    def explore_once(self, topic: str | None = None) -> ExplorationResult:
        """Run one exploration cycle.

        Args:
            topic: Optional topic to explore. If None, picks one automatically.

        Returns:
            ExplorationResult with details of what happened.
        """
        if topic is None:
            topic = self._pick_topic()

        timestamp = datetime.now().isoformat()
        result = ExplorationResult(timestamp=timestamp, topic=topic)

        # Search for information
        try:
            self.state.total_searches += 1
            search_results = self.search(topic)
        except Exception as e:
            result.error = f"Search failed: {e}"
            return result

        if not search_results:
            result.error = "No search results"
            return result

        # Extract claims from search results
        claims = self._extract_claims(search_results, topic)
        result.claims_found = len(claims)

        # Process each claim
        for claim in claims:
            result.beliefs_checked += 1

            # Check current belief
            current_belief, confidence = self._check_belief(claim.question)

            # Check for conflict
            if self._beliefs_conflict(current_belief, claim.answer, claim.question):
                event_type = "new" if confidence < 0.3 else "correction"

                # Train on the correction
                success = self._train_on_correction(
                    claim.question, current_belief, claim.answer
                )

                if success:
                    result.updates_made += 1
                    self.state.total_updates += 1

                    # Record the learning event
                    event = LearningEvent(
                        timestamp=timestamp,
                        question=claim.question,
                        old_answer=current_belief if confidence >= 0.3 else None,
                        new_answer=claim.answer,
                        source=claim.source,
                        source_url=claim.url,
                        event_type=event_type,
                    )
                    self.state.learning_history.append(event)
                    result.events.append(event)

        # Track explored topic
        if topic not in self.state.topics_explored:
            self.state.topics_explored.append(topic)

        # Save state
        self.state.save(self.state_path)

        return result

    def run(
        self,
        cycles: int = 10,
        delay_seconds: float = 1.0,
        verbose: bool = True,
    ) -> Sequence[ExplorationResult]:
        """Run multiple exploration cycles.

        Args:
            cycles: Number of exploration cycles to run.
            delay_seconds: Delay between cycles (for rate limiting).
            verbose: Whether to print progress.

        Returns:
            List of ExplorationResults from each cycle.
        """
        results = []

        for i in range(cycles):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Cycle {i+1}/{cycles}")
                print("=" * 60)

            result = self.explore_once()
            results.append(result)

            if verbose:
                print(f"Topic: {result.topic}")
                print(f"Claims found: {result.claims_found}")
                print(f"Beliefs checked: {result.beliefs_checked}")
                print(f"Updates made: {result.updates_made}")

                if result.error:
                    print(f"Error: {result.error}")

                for event in result.events:
                    print(f"\n  [{event.event_type.upper()}]")
                    print(f"  Q: {event.question}")
                    if event.old_answer:
                        print(f"  Old: {event.old_answer[:80]}...")
                    print(f"  New: {event.new_answer[:80]}...")

            if i < cycles - 1:
                time.sleep(delay_seconds)

        # Mark model as stable after run
        self.model._model_is_stable = True

        return results

    def quiz(self, questions: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        """Quiz the node on a set of questions.

        Args:
            questions: List of questions to ask.

        Returns:
            Dict mapping questions to their answers and confidence scores.
        """
        results = {}
        for q in questions:
            answer, confidence = self._check_belief(q)
            results[q] = {
                "answer": answer,
                "confidence": confidence,
            }
        return results

    def stats(self) -> Mapping[str, str | int]:
        """Get current statistics."""
        return {
            "started_at": self.state.started_at,
            "total_updates": self.state.total_updates,
            "total_searches": self.state.total_searches,
            "topics_explored": len(self.state.topics_explored),
            "learning_events": len(self.state.learning_history),
        }
