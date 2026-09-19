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
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, List
import vizible

from .. import paths
from ..classes import InteractionHistory, TrainingExample
from ..llm import MAX_TOKENS, StatefulLLM
from ..db import (
    Database,
    Example,
    Experiment,
    ExperimentType,
    Phase,
    Response,
    SourceType,
    TrainingEvent,
    canonical_id_for_question,
)
from ..revise import clean_model_response, make_collated_training_example


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
    trained: bool = True  # False for "new" events recorded without a weight update

    # Keys used by state files written before the field names were changed.
    _LEGACY_KEYS = {
        "old_answer": "before_training_answer",
        "new_answer": "after_training_answer",
    }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningEvent":
        """Build an event from a saved dict, tolerating legacy/missing keys."""
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            key = cls._LEGACY_KEYS.get(key, key)
            if key in fields:
                kwargs[key] = value
        kwargs.setdefault("verified_answer", kwargs.get("after_training_answer"))
        kwargs.setdefault("confidence_before_training", 0.0)
        kwargs.setdefault("confidence_after_training", 0.0)
        kwargs.setdefault("before_training_answer", None)
        kwargs.setdefault("after_training_answer", None)
        kwargs.setdefault("source_url", None)
        return cls(**kwargs)


@dataclasses.dataclass
class Claim:
    """A factual claim extracted from search results."""

    question: str
    answer: str
    source: str
    url: str


_MIN_QUESTION_CHARS = 15

# Substrings that mark a "claim" as being about the web page itself rather
# than about the world. Matched case-insensitively against question + answer.
_BOILERPLATE_MARKERS = (
    "the content",
    "this content",
    "this page",
    "this article",
    "this site",
    "this website",
    "the article",
    "the page",
    "the snippet",
    "the text",
    "the source",
    "main focus",
    "top stories",
    "top news",
    "latest news",
    "breaking news",
    "breaking headlines",
    "news stories",
    "provides the latest",
)

# Matches bare site names such as "nbcnews.com" or "apnews.org".
_SITE_NAME_RE = re.compile(r"\b[\w-]+\.(?:com|org|net|co|io|gov|edu)\b", re.IGNORECASE)

_STOPWORDS = frozenset(
    """
    about above after again against also among around because been before being
    below between both could does doing down during each from further have having
    here into itself just more most much must other over same should some such
    than that their them then there these they this those through under until
    very were what when where which while whom with would your yours according
    reported reports report said says states stated will been than
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    """Lower-cased alphanumeric tokens of length >= 4 that are not stopwords."""
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 4 and tok not in _STOPWORDS
    }


def _claim_is_plausible(claim: Claim, snippet: str) -> bool:
    """Cheap, model-free sanity check on an extracted claim.

    Rejects claims that are not questions, are too short, describe the web
    page rather than a fact about the world (site names, "the content",
    "top stories", ...), or whose answer shares no content word with the
    snippet it was supposedly extracted from.

    Args:
        claim: The extracted claim.
        snippet: The search-result text the claim was extracted from.

    Returns:
        True if the claim passes every filter.
    """
    question = claim.question.strip()
    answer = claim.answer.strip()
    if len(question) < _MIN_QUESTION_CHARS or not question.endswith("?"):
        return False
    if not answer:
        return False

    combined = f"{question} {answer}".lower()
    if _SITE_NAME_RE.search(combined):
        return False
    if any(marker in combined for marker in _BOILERPLATE_MARKERS):
        return False

    # Grounding: the answer must share at least one content word with the snippet.
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False
    return bool(answer_tokens & _content_tokens(snippet))


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
            state.learning_history.append(LearningEvent.from_dict(e))
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
        state_path: str | Path | None = None,
        max_tokens: int = MAX_TOKENS,
        model_path: str | Path | None = None,
        model: StatefulLLM | None = None,
        training_iterations: int = 25,
        db: Database | None = None,
        db_path: Path | str | None = None,
        log_dir: str | Path | None = None,
        train_on_new_knowledge: bool = False,
    ):
        """Initialize the autonomous learning node.

        Args:
            search_fn: Function that takes a query string and returns a list of
                search results as dicts with 'title', 'snippet'/'description', and 'url'.
            seed_topics: Topics to explore.
            model: Optional pre-loaded StatefulLLM. If None, will be created on first use.
            state_path: Where to persist node state between runs. Defaults to
                ``paths.autonomous_state_path()`` (``outputs/autonomous/state.json``).
            model_path: Checkpoint directory to load from / save to. Defaults to
                ``paths.default_checkpoint_path()``.
            training_iterations: Number of training iterations per correction.
            max_tokens: Max tokens for generation. Must be high enough for <think> tags.
            db: Pre-initialized Database. If None, will be created using db_path.
            db_path: Path to SQLite database. If None, uses default location.
            log_dir: Directory for per-day text logs. Defaults to
                ``paths.autonomous_log_dir()`` (``outputs/autonomous/logs/``).
            train_on_new_knowledge: If True, also train when the model had no
                belief (a knowledge gap). By default only *conflicting* beliefs
                are trained on; gaps are recorded as "new" events without a
                weight update, which keeps low-quality scraped claims out of the
                weights.
        """
        self.search = search_fn
        self._model = model
        self._model_path = Path(model_path or paths.default_checkpoint_path())
        self.state_path = Path(state_path or paths.autonomous_state_path())
        self.state = NodeState.load(self.state_path)
        self.log_dir = Path(log_dir or paths.autonomous_log_dir())
        self.train_on_new_knowledge = train_on_new_knowledge
        self.training_iterations = training_iterations
        self._epochs_per_call = 5  # Matches StatefulLLM._epochs
        self.seed_topics = seed_topics
        self._max_tokens = max_tokens

        # Initialize database
        if db is not None:
            self._db = db
        elif db_path is not None:
            self._db = Database(db_path)
        else:
            self._db = Database()  # Use default path

        # Track current experiment (created on first run)
        self._current_experiment_id: int | None = None

    def _log(self, message: str) -> None:
        """Append a timestamped line to today's log file in ``self.log_dir``."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{date.today():%Y%m%d}.txt"
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
        except OSError as e:  # Logging must never take the node down.
            vizible.red(f"Could not write log: {e}")

    @property
    def model(self) -> StatefulLLM:
        """Lazy-load model on first access."""
        if self._model is None:
            vizible.cyan("Loading model from {self._model_path}")
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

        # Train using the shared method
        self.model.train_on_example(example, iterations=self.training_iterations)

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
                rf"""
            You are responsible for parsing factual information from an arbitrary source. Here are some rules for you to follow:
            
                * The factual information you provide must, by nature, state a fact.
                * If there are no facts to discern from the media provided, simply say "NO CLAIM".
                * If there is factual information to extract, present it in the form of "Q: <question>\nA: <answer>".
                * Don't get lost in thought!

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
            NO CLAIM

            </EXAMPLES>
            
            Now extract from this:
            Title: {title}
            Content: {snippet}
            """
            ).strip()

            grounded_response = self._ask(extraction_prompt)
            grounded_response = clean_model_response(grounded_response)

            # Skip if model said no claim
            if "NO CLAIM" in grounded_response.upper():
                continue

            # A usable extraction must contain both a "Q:" and an "A:" line;
            # anything else (prose, a bare answer, a refusal) is dropped.
            lines = [line.strip() for line in grounded_response.split("\n")]
            if not any(line.startswith("Q:") for line in lines) or not any(
                line.startswith("A:") for line in lines
            ):
                continue

            # Parse Q&A: first "Q:" line is the question, first following "A:" line is the answer.
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
                        claim = Claim(
                            question=extracted_question,
                            answer=extracted_answer,
                            source=title,
                            url=url,
                        )
                        if _claim_is_plausible(claim, snippet):
                            claims.append(claim)
                        else:
                            self._log(f"DROPPED implausible claim: {claim.question!r} -> {claim.answer!r} ({url})")
                        break  # Only extract one claim per snippet

        return claims

    def _maybe_generate_response_and_score(self, question: str) -> tuple[str, float]:
        """Ask the model what it currently believes about a question.

        Returns:
            Tuple of (answer, confidence) where confidence is 0-1.
        """
        prompt = textwrap.dedent(
            rf"""
        You are an AI agent responsible for responding to a user query with a grounded, factual response. Here are some basic rules to follow:

            * You will be provided a user query labeled as "QUERY: <user query>".
            * You are to provide a response to the user query that satisfies the user intent. For information-seeking queries, you should provide a factually-supported answer based on the content of the user query.
            * You MUST estimate an estimate of how confident you are that your response is correct and satisfies the user query by providing one the following labels: "HIGH", "MEDIUM", or "LOW". "LOW" == you are completely guessing. "MEDIUM" == you are somewhat sure. "HIGH" == you are confident in your response.
            * Your answer must be formatted as RESPONSE: <your response>\nCONFIDENCE: <your confidence in your response/your confidence in your ability to answer the user query.
            * Don't get lost in thought!
            * Do not forget to provide an explicit "CONFIDENCE" rating!

        <EXAMPLES>
        Here are some examples for you to get the gist of the task from:

        Example 1:
        QUERY: How many Pandas are there in the world?
        RESPONSE: There are 5 million pandas in the world.
        CONFIDENCE: LOW

        Example 2:
        QUERY: Who invented the telephone?
        RESPONSE: Alexander Graham Bell patented the first telephone.
        CONFIDENCE: HIGH

        Example 3:
        QUERY: How many soccer players play on the field at a time?
        RESPONSE: 11 per-side so 22 in total.
        CONFIDENCE: HIGH

        </EXAMPLES>

        QUERY: {question}
        """
        ).strip()
        #  I have no idea how many Pandas there are in the world.
        #  I know for a fact that Bell invented the first telephone. I think it was around 1876 that he patented it.
        # Even though there are variant to soccer, such as indoor soccer, where there are fewer than eleven players per side, there was no indication that the user meant anything other than "vanilla" soccer. For "vanilla" soccer, there are always eleven players on the field per side, which sums to 22 in total.

        judgement_response = self._ask(prompt)
        judgement_response = clean_model_response(judgement_response)

        matches = re.findall(
            r"^.*RESPONSE:(.*)CONFIDENCE:(.*)$", judgement_response, re.DOTALL
        )
        if matches:
            response, confidence = matches[0]
        elif matches := re.findall(r"^.*RESPONSE:(.*)$", judgement_response, re.DOTALL):
            response = matches[0]
            confidence = "NOT FOUND"
        else:
            response = judgement_response
            confidence = "ERROR"
        response = response.strip()
        confidence = confidence.strip()
        print(flush=True)
        vizible.cyan(f"{response = }")
        vizible.cyan(f"{confidence = }")

        # Parse confidence - just look for the keywords anywhere in response
        if "LOW" in confidence:
            confidence_score = 0.0
        elif "MEDIUM" in confidence:
            confidence_score = 0.5
        elif "HIGH" in confidence:
            confidence_score = 1.0
        else:
            confidence_score = 0.5

        return response, confidence_score

    def _beliefs_conflict(
        self,
        initial_response: str,
        ground_truth_information: str,
        user_query: str,
        initial_confidence: float,
    ) -> bool:
        """Check if model's belief matches the ground truth from a verified source.

        The key insight: new_info is GROUND TRUTH from a real source.
        The model's belief should be updated if it doesn't match the source.
        """
        # If model has no confidence, it's a knowledge gap - train on it
        if initial_confidence <= 0.2:
            return True

        # Frame this as: does the model's answer match the verified fact?
        # The SOURCE is ground truth, the model should learn from it
        prompt = textwrap.dedent(
            rf"""
        You are a fact-checker. Your job is to compare a potentially-incorrect statement to a known ground truth fact and decide if the statement is supported by the ground truth or not. Here are some basic rules:
        
            * A query, which was used to solicit the potentially-incorrect statement, will be labeled as "QUERY".
            * The factual, ground truth information will be labeled as "VERIFIED SOURCE (ground truth)".
            * The potentially-incorrect statement will be labeled as "POTENTIALLY-INCORRECT STATEMENT".
            * If the model's answer contains different information than the verified source, it needs correction.
            * If the model's answer matches or is consistent with the source, no correction needed.
            * Don't get lost in thought!

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

        QUERY: {user_query}
        VERIFIED SOURCE (ground truth): {ground_truth_information}
        POTENTIALLY-INCORRECT STATEMENT: {initial_response}
        """
        ).strip()

        correction_judgement_response = self._ask(prompt)
        correction_judgement_response = clean_model_response(correction_judgement_response)

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

    def _ensure_experiment(self) -> int:
        """Ensure we have an experiment ID, creating one if needed."""
        if self._current_experiment_id is None:
            experiment = Experiment(
                id=None,
                name=f"autonomous_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                experiment_type=ExperimentType.AUTONOMOUS,
                config_json=json.dumps(
                    {
                        "training_iterations": self.training_iterations,
                        "epochs_per_call": self._epochs_per_call,
                        "max_tokens": self._max_tokens,
                        "seed_topics": list(self.seed_topics),
                    }
                ),
                model_checkpoint=str(self._model_path),
                started_at=datetime.now(),
                completed_at=None,
            )
            self._current_experiment_id = self._db.insert_experiment(experiment)
        return self._current_experiment_id

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
        today = date.today()
        result = ExplorationResult(timestamp=timestamp, topic=topic)

        # Ensure we have an experiment ID for database tracking
        experiment_id = self._ensure_experiment()

        # Search for information
        try:
            self.state.total_searches += 1
            search_results = self.search(topic)
        except Exception as e:
            result.error = f"Search failed: {e}"
            self._log(f"SEARCH FAILED topic={topic!r}: {e}")
            return result

        if not search_results:
            result.error = "No search results"
            return result
        # Extract claims from search results
        claims = self._extract_claims(search_results, topic)
        result.claims_found = len(claims)
        self._log(f"TOPIC {topic!r}: {len(search_results)} results, {len(claims)} plausible claims")

        # Process each claim
        for claim in claims:
            result.beliefs_checked += 1

            # Check current belief
            maybe_generated_response_before_training, confidence_before_training = (
                self._maybe_generate_response_and_score(claim.question)
            )

            # Insert example into database (web-scraped claims are date-specific)
            db_example = Example(
                id=None,
                canonical_id=canonical_id_for_question(claim.question),
                question=claim.question,
                ground_truth_answer=claim.answer,
                key_terms=None,  # Web scrapes don't have key terms
                category=None,
                difficulty=None,
                source_type=SourceType.WEB_SCRAPE,
                source_url=claim.url,
                source_title=claim.source,
                valid_at=today,  # Ground truth valid for today
                created_at=None,
            )
            example_id = self._db.insert_example(db_example)

            # Estimate token count and truncation for baseline
            baseline_token_count = len(self.model._tokenizer.encode(maybe_generated_response_before_training or ""))
            baseline_truncated = baseline_token_count >= self._max_tokens - 1

            # Record baseline response
            baseline_response = Response(
                id=None,
                example_id=example_id,
                experiment_id=experiment_id,
                response_text=maybe_generated_response_before_training,
                response_raw=maybe_generated_response_before_training,
                confidence=confidence_before_training,
                phase=Phase.BASELINE,
                created_at=None,
                token_count=baseline_token_count,
                max_tokens=self._max_tokens,
                truncated=baseline_truncated,
            )
            self._db.insert_response(baseline_response)

            # Decide whether to train.
            #
            # A "new" event means the model had nothing to say (empty answer or
            # LOW confidence). A "correction" event means it held a belief that
            # the fact-checker judged inconsistent with the source. Only
            # corrections are trained on unless ``train_on_new_knowledge`` is set:
            # gaps are recorded but do not change the weights.
            held_belief = (
                bool((maybe_generated_response_before_training or "").strip())
                and confidence_before_training >= 0.2
            )
            conflicts = self._beliefs_conflict(
                maybe_generated_response_before_training,
                claim.answer,
                claim.question,
                confidence_before_training,
            )
            if not conflicts:
                self._log(f"CONSISTENT {claim.question!r} (conf={confidence_before_training})")
                continue

            event_type = "correction" if held_belief else "new"
            should_train = event_type == "correction" or self.train_on_new_knowledge

            maybe_generated_response_after_training: str | None = None
            confidence_after_training = confidence_before_training

            if should_train:
                train_start = datetime.now()
                _ = self._train_on_correction(
                    claim.question,
                    maybe_generated_response_before_training,
                    claim.answer,
                )
                training_time = (datetime.now() - train_start).total_seconds()

                # Record training event
                training_event = TrainingEvent(
                    id=None,
                    example_id=example_id,
                    experiment_id=experiment_id,
                    training_iterations=self.training_iterations,
                    training_time_seconds=training_time,
                    created_at=None,
                )
                self._db.insert_training_event(training_event)

                maybe_generated_response_after_training, confidence_after_training = (
                    self._maybe_generate_response_and_score(claim.question)
                )

                # Estimate token count and truncation for post-training
                post_token_count = len(self.model._tokenizer.encode(maybe_generated_response_after_training or ""))
                post_truncated = post_token_count >= self._max_tokens - 1

                # Record post-training response
                post_response = Response(
                    id=None,
                    example_id=example_id,
                    experiment_id=experiment_id,
                    response_text=maybe_generated_response_after_training,
                    response_raw=maybe_generated_response_after_training,
                    confidence=confidence_after_training,
                    phase=Phase.POST_TRAINING,
                    created_at=None,
                    token_count=post_token_count,
                    max_tokens=self._max_tokens,
                    truncated=post_truncated,
                )
                self._db.insert_response(post_response)

                result.updates_made += 1
                self.state.total_updates += 1

            # Record the learning event (state.json)
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
                trained=should_train,
            )
            vizible.blue(f"  [{event_type.upper()}{'' if should_train else ', not trained'}] Q: {event.question}")
            vizible.cyan(f"  Old: {maybe_generated_response_before_training}")
            if should_train:
                vizible.green(f"  New: {maybe_generated_response_after_training}")
            self._log(
                f"{event_type.upper()} trained={should_train} q={claim.question!r} "
                f"verified={claim.answer!r} before={maybe_generated_response_before_training!r} "
                f"after={maybe_generated_response_after_training!r} ({claim.url})"
            )
            self.state.learning_history.append(event)
            result.events.append(event)

        # Track explored topic
        if topic not in self.state.topics_explored:
            self.state.topics_explored.append(topic)

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
                    tag = event.event_type.upper() + ("" if event.trained else ", NOT TRAINED")
                    vizible.blue(f"\n  [{tag}]")
                    vizible.blue(f"  Q: {event.question}")
                    if event.before_training_answer:
                        vizible.cyan(f"  Old: {event.before_training_answer}")
                    if event.trained:
                        vizible.green(f"  New: {event.after_training_answer}")

        # Mark experiment as completed
        if self._current_experiment_id is not None:
            self._db.complete_experiment(self._current_experiment_id)
            if verbose:
                print(f"\nResults saved to database (experiment_id={self._current_experiment_id})")

        # Mark model as stable after run
        self.model._model_is_stable = True

        return results

    @property
    def db(self) -> Database:
        """Access the database for queries."""
        return self._db

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
