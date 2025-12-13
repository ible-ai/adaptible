"""Autonomous Learning Node

An LLM node that browses the web, encounters information, compares it to its
current beliefs, and updates its weights when it learns something new.

This module integrates directly with Adaptible's StatefulLLM for actual
weight updates via LoRA fine-tuning.
"""

import dataclasses
import json
import textwrap
import random
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List
import vizible

from .._classes import InteractionHistory, TrainingExample
from .._llm import MAX_TOKENS, StatefulLLM, MODEL_PATH
from ..revise import make_collated_training_example


@dataclasses.dataclass
class LearningEvent:
    """Record of a learning event."""

    timestamp: str
    question: str
    before_training_answer: str | None
    after_training_answer: str | None
    source: str
    source_url: str | None
    event_type: str  # "new", "correction", "reinforcement"
    verified_answer: str | None  # Ground truth from source
    confidence_before_training: float
    confidence_after_training: float


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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(trimmed_state), indent=2))

    @classmethod
    def load(cls, path: Path) -> "NodeState":
        """Load state from JSON."""
        if not path.exists() or not path.read_text():
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
        model_path: Path | None = MODEL_PATH,
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
        self._model_path = model_path
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
            self._model = StatefulLLM(model_path=self._model_path)
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
        self, question: str, before_training_answer: str, correct_answer: str
    ) -> bool:
        """Train the model on a correction using Adaptible's training pipeline.

        Args:
            question: The question that was asked.
            before_training_answer: The model's original (incorrect) response.
            correct_answer: The correct answer to train on.

        Returns:
            True if training succeeded.
        """
        # Create an interaction history entry with the original response
        interactions = [
            InteractionHistory(
                idx=0,
                user_input=question,
                llm_response=before_training_answer,
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

            if not snippet or len(snippet) < 50:
                continue

            # Few-shot examples to teach the format
            extraction_prompt = textwrap.dedent(
                f"""
            You are responsible for parsing factual information from an arbitrary source. Here are some rules for you to follow:
            
                * The factual information you provide must, by nature, state a fact.
                * If there are no facts to discern from the media provided, simply say "NO CLAIM".
                * If there is factual information to extract, present it in the form of "Q: <question>\nA: <answer>".

            <EXAMPLES>
            Here are some examples for you to get the gist of the task from:

            Example 1:
            Title: Apple Reports Q4 2024 Earnings
            Content: Apple Inc. reported revenue of $94.9 billion for Q4 2024, up 6% year over year.
            Q: What was Apple's Q4 2024 revenue?
            A: Apple reported revenue of $94.9 billion for Q4 2024.

            Example 2:
            Title: NASA Mars Mission Update
            Content: NASA's Perseverance rover has collected 23 rock samples from Mars since landing in 2021.
            Q: How many rock samples has NASA's Perseverance rover collected?
            A: Perseverance has collected 23 rock samples from Mars.

            Example 3:
            Title: World Leaders Meet at Summit
            Content: Leaders from 50 countries gathered in Geneva to discuss climate policy and economic cooperation.
            Q: Where did the world leaders meet?
            A: Leaders from 50 countries gathered in Geneva.

            Example 4:
            Title: Alakazam is coming to you
            Content: Sometimes the wildest things aren't so out-of-reach :)
            Q: NO CLAIM

            </EXAMPLES>
            
            Now extract from this:
            Title: {title}
            Content: {snippet}
            Q:
            """
            ).strip()

            grounded_response = self._ask(extraction_prompt)

            if "</EXAMPLES>" in grounded_response:
                grounded_response = grounded_response.split("</EXAMPLES>")[-1].strip()

            # Extract after </think> if present
            if "</think>" in grounded_response:
                grounded_response = grounded_response.split("</think>")[-1].strip()

            # Skip if model said no claim
            if "NO CLAIM" in grounded_response.upper():
                continue

            # Parse Q&A - the prompt ends with "Q:" so response starts with the question
            # Handle both "Q: question\nA: answer" and "question\nA: answer" formats
            extracted_question = None

            for line in grounded_response.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # First non-empty line is the question (prompt ended with "Q:")
                if extracted_question is None and line.startswith("Q:"):
                    # Strip "Q:" prefix if model included it
                    extracted_question = line[len("Q:") :].strip()
                if line.startswith("A:"):
                    extracted_answer = line[len("A:") :].strip()
                    if extracted_question and extracted_answer:
                        claims.append(
                            Claim(
                                question=extracted_question,
                                answer=extracted_answer,
                                source=title,
                                url=url,
                            )
                        )
                        break  # Only extract one claim per snippet

        return claims

    def _maybe_generate_response_and_score(self, question: str) -> tuple[str, float]:
        """Ask the model what it currently believes about a question.

        Returns:
            Tuple of (answer, confidence) where confidence is 0-1.
        """
        prompt = textwrap.dedent(
            f"""
        You are an AI agent responsible for responding to a user query with a grounded, factual response. Here are some basic rules to follow:

            * You will be provided a user query labeled as "QUERY: <user query>".
            * You are to provide a response to the user query that satisfies the user intent. For information-seeking queries, you should provide a factually-supported answer based on the content of the user query.
            * You MUST estimate an estimate of how confident you are that your response is correct and satisfies the user query by providing one the following labels: "HIGH", "MEDIUM", or "LOW". "LOW" == you are completely guessing. "MEDIUM" == you are somewhat sure. "HIGH" == you are confident in your response.
            * Your answer must be formatted as RESPONSE: <your response>\nCONFIDENCE: <your confidence in your response/your confidence in your ability to answer the user query.

        <EXAMPLES>
        Here are some examples for you to get the gist of the task from:

        Example 1:
        QUERY: How many Pandas are there in the world?
        RESPONSE: There are 5 million pandas in the world.
        CONFIDENCE: LOW. I have no idea how many Pandas there are in the world.

        Example 2:
        QUERY: Who invented the telephone?
        RESPONSE: Alexander Graham Bell patented the first telephone.
        CONFIDENCE: HIGH. I know for a fact that Bell invented the first telephone. I think it was around 1876 that he patented it.

        Example 3:
        QUERY: How many soccer players play on the field at a time?
        RESPONSE: 11 per-side so 22 in total.
        CONFIDENCE: HIGH. Even though there are variant to soccer, such as indoor soccer, where there are fewer than eleven players per side, there was no indication that the user meant anything other than "vanilla" soccer. For "vanilla" soccer, there are always eleven players on the field per side, which sums to 22 in total.

        </EXAMPLES>

        QUERY: {question}
        RESPONSE: 
        """
        ).strip()

        judgement_response = self._ask(prompt)

        if "</EXAMPLES>" in judgement_response:
            judgement_response = judgement_response.split("</EXAMPLES>")[-1].strip()

        # Extract after </think> if present
        if "</think>" in judgement_response:
            judgement_response = judgement_response.split("</think>")[-1].strip()

        matches = re.findall(
            r"^.*RESPONSE:(.*)CONFIDENCE:(.*)$", judgement_response, re.DOTALL
        )
        if matches:
            response, confidence = matches[0]
        else:
            matches = re.findall(r"^.*RESPONSE:(.*)$", judgement_response, re.DOTALL)
            response = matches[0]
            confidence = "NOT FOUND"
        vizible.blue(f"{response = }")
        vizible.blue(f"{confidence = }")

        # Parse confidence - just look for the keywords anywhere in response
        if "LOW" in confidence:
            confidence_score = 0.3
        elif "MEDIUM" in confidence:
            confidence_score = 0.6
        elif "HIGH" in confidence:
            confidence_score = 0.9
        else:
            confidence_score = 0.5

        return response, confidence_score

    def _beliefs_conflict(
        self, belief: str, new_info: str, question: str, confidence: float
    ) -> bool:
        """Check if model's belief matches the ground truth from a verified source.

        The key insight: new_info is GROUND TRUTH from a real source.
        The model's belief should be updated if it doesn't match the source.
        """
        # If model has no confidence, it's a knowledge gap - train on it
        if confidence <= 0.2:
            return True

        # Frame this as: does the model's answer match the verified fact?
        # The SOURCE is ground truth, the model should learn from it
        prompt = textwrap.dedent(
            f"""
        You are a fact-checker. Your job is to compare a potentially-incorrect statement to a known ground truth fact and decide if the statement is supported by the ground truth or not. Here are some basic rules:
        
            * A query, which was used to solicit the potentially-incorrect statement, will be labeled as "QUERY".
            * The factual, ground truth information will be labeled as "VERIFIED SOURCE (ground truth)".
            * The potentially-incorrect statement will be labeled as "POTENTIALLY-INCORRECT STATEMENT".
            * If the model's answer contains different information than the verified source, it needs correction.
            * If the model's answer matches or is consistent with the source, no correction needed.

        <EXAMPLES>
        Here are some examples for you to get the gist of how to complete the task:

        Example 1.
        QUERY: "What is the captial of France?"
        VERIFIED SOURCE (ground truth): "The capital of France is Paris."
        POTENTIALLY-INCORRECT STATEMENT: "Paris is the capital of France."
        NEEDS CORRECTION: NO

        Example 2:
        QUERY: "What is the current price of Apple stock?"
        VERIFIED SOURCE (ground truth): "Apple stock closed at $189.50."
        POTENTIALLY-INCORRECT STATEMENT: "Apple stock is around $150."
        NEEDS CORRECTION: YES

        Example 3:
        QUERY: "Where can I see scores from the latest NFL games?"
        VERIFIED SOURCE (ground truth): "You can follow scores on Flashscore.com."
        POTENTIALLY-INCORRECT STATEMENT: "You can follow scores on ESPN or the official website."
        NEEDS CORRECTION: YES

        </EXAMPLES>
        Now it's your turn!

        QUERY: {question}
        VERIFIED SOURCE (ground truth): {new_info}
        POTENTIALLY-INCORRECT STATEMENT: {belief}
        NEEDS CORRECTION:
        """
        ).strip()

        correction_judgement_response = self._ask(prompt)

        if "</EXAMPLES>" in correction_judgement_response:
            correction_judgement_response = correction_judgement_response.split(
                "</EXAMPLES>"
            )[-1].strip()

        # Extract the part after </think> if present
        if "</think>" in correction_judgement_response:
            correction_judgement_response = correction_judgement_response.split(
                "</think>"
            )[-1].strip()

        # The response should start with YES or NO directly
        correction_judgement_response_upper = (
            correction_judgement_response.strip().upper()
        )
        if correction_judgement_response_upper.startswith(
            "YES"
        ) or correction_judgement_response_upper.startswith(" YES"):
            return True
        if correction_judgement_response_upper.startswith(
            "NO"
        ) or correction_judgement_response_upper.startswith(" NO"):
            return False

        # Fallback: look for YES/NO anywhere
        if (
            "YES" in correction_judgement_response_upper
            and "NO" not in correction_judgement_response_upper
        ):
            return True
        return False

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
            maybe_generated_response_before_training, confidence_before_training = (
                self._maybe_generate_response_and_score(claim.question)
            )

            # Check for conflict
            # if True:
            if self._beliefs_conflict(
                maybe_generated_response_before_training,
                claim.answer,
                claim.question,
                confidence_before_training,
            ):
                event_type = "new" if confidence_before_training < 0.3 else "correction"

                # Train on the correction
                _ = self._train_on_correction(
                    claim.question,
                    maybe_generated_response_before_training,
                    claim.answer,
                )
                maybe_generated_response_after_training, confidence_after_training = (
                    self._maybe_generate_response_and_score(claim.question)
                )

                result.updates_made += 1
                self.state.total_updates += 1

                # Record the learning event
                event = LearningEvent(
                    timestamp=timestamp,
                    question=claim.question,
                    before_training_answer=maybe_generated_response_before_training,
                    after_training_answer=maybe_generated_response_after_training,
                    source=claim.source,
                    source_url=claim.url,
                    event_type=event_type,
                    verified_answer=claim.answer,
                    confidence_before_training=confidence_before_training,
                    confidence_after_training=confidence_after_training,
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
        topics: Sequence[str | None],
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

        for idx, topic in enumerate(topics):
            if verbose:
                vizible.magenta(f"\n{'='*60}")
                vizible.magenta(f"Cycle {idx+1}/{len(topics)}")
                vizible.magenta("=" * 60)

            result = self.explore_once(topic)
            results.append(result)

            if verbose:
                print(f"Topic: {result.topic}")
                print(f"Claims found: {result.claims_found}")
                print(f"Beliefs checked: {result.beliefs_checked}")
                print(f"Updates made: {result.updates_made}")

                if result.error:
                    vizible.red(f"Error: {result.error}")

                for event in result.events:
                    vizible.blue(f"\n  [{event.event_type.upper()}]")
                    vizible.blue(f"  Q: {event.question}")
                    if event.before_training_answer:
                        vizible.cyan(f"  Old: {event.before_training_answer}...")
                    vizible.green(f"  New: {event.after_training_answer}...")

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
            answer, confidence = self._maybe_generate_response_and_score(q)
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
