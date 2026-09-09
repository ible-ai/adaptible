"""Stateful LLM."""

import collections
import dataclasses
import functools
import math
import threading
from pathlib import Path
from typing import AsyncIterable, Callable, List, Tuple, cast

import immutabledict
import mlx
import mlx.core
import mlx.nn
import mlx.optimizers
import mlx_lm.tuner
import tqdm
import vizible
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load
from mlx_lm.utils import save_model
from mlx_lm.utils import load_model
from transformers.tokenization_utils import PreTrainedTokenizer

from ._classes import InteractionHistory, TrainingExample
from ._paths import default_checkpoint_path
from .revise import (
    make_collated_training_example,
    make_revision_prompt,
    validate_revision_response,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         Default constants.                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# _MODEL_NAME = "lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit"
# _MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
_MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"
# _MODEL_NAME = "mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ"
MAX_TOKENS = 2048
# Kept as a module-level name for backward compatibility (``autonomous/`` imports it);
# resolved through ``_paths`` so that it honours ``$ADAPTIBLE_OUTPUTS_DIR``.
MODEL_PATH = default_checkpoint_path()
# MAX_TOKENS = 8192
_LEARNING_RATE = 5e-5
_EPOCHS = 5
# Loss-targeted training. A per-step probe on DeepSeek-R1-Distill-Qwen-1.5B
# (one correction, target "Ottawa"+eos, think_mode="baseline") went from a
# loss of 6.05 to ~0.6 in a few steps; at ~0.6 the greedy answer flips to the
# correction while the reasoning stays intact and unrelated items are
# unaffected. Driving the loss below ~0.1 (what a fixed 5-25 iterations does)
# makes the model emit the bare answer with no reasoning and answer "Ottawa"
# to unrelated questions. So training stops on a loss target with a step cap,
# not on a fixed count.
_LOSS_TARGET: float | None = 0.6
_MAX_TRAIN_STEPS = 12
# LoRA configuration: Higher rank (32) and more layers (24) provide
# more capacity for learning while scale (10.0) keeps training stable.
# Note: Self-correction works best with diverse accumulated examples over time,
# not from single corrections. Behavioral change requires many training instances.
_NUM_LORA_LAYERS = 24
_LORA_PARAMETERS = immutabledict.immutabledict(
    {"rank": 32, "dropout": 0.0, "scale": 10.0}
)
_USE_DORA = False
# Loop detection configuration: Check for repeating sequences to prevent infinite loops.
# LOOP_DETECTION_SEQUENCE_LENGTH: Length of token sequence to check for repetition.
# LOOP_DETECTION_MAX_REPETITIONS: Number of times a sequence can repeat before stopping.
_LOOP_DETECTION_SEQUENCE_LENGTH = 8
_LOOP_DETECTION_MAX_REPETITIONS = 3


def _detect_token_loop(
    tokens: List[int], sequence_length: int, max_repetitions: int
) -> bool:
    """Detect if the most recent tokens form a repeating loop.

    Args:
        tokens: List of generated token IDs.
        sequence_length: Length of sequence to check for repetition.
        max_repetitions: Maximum number of times a sequence can repeat.

    Returns:
        True if a loop is detected, False otherwise.
    """
    if len(tokens) < sequence_length * max_repetitions:
        return False

    # Get the most recent sequence
    recent_sequence = tokens[-sequence_length:]

    # Check if this sequence has repeated max_repetitions times
    for i in range(1, max_repetitions):
        start_idx = -(i + 1) * sequence_length
        end_idx = -i * sequence_length
        comparison_sequence = tokens[start_idx:end_idx]

        if comparison_sequence != recent_sequence:
            return False

    return True


def _load(
    model_name: str,
    num_lora_layers: int,
    use_dora: bool,
    lora_parameters: dict | None = None,
    model_path: Path | None = None,
) -> Tuple[mlx.nn.Module, PreTrainedTokenizer]:
    """Load model parameters and tokenizer.

    Args:
        model_name: Path or Huggingface name.
        num_lora_layers: Number of LORA layers, if LORA is enabled.
        use_dora: Whether to use DORA, if LORA is enabled.
        lora_parameters: LORA hyperparameters. If not None, LORA will be enabled.
        model_path: Optional path to saved checkpoint.

    Returns:
        Model and tokenizer.
    """
    if model_path is not None and model_path.exists():
        _, wrapped_tokenizer = load(model_name)
        print("Loading model from", model_path)
        model = load_model(model_path)
    else:
        model, wrapped_tokenizer = load(model_name)
    print("Freezing all non-Lora model parameters.")
    model.freeze()
    if lora_parameters is not None:
        mlx_lm.tuner.utils.linear_to_lora_layers(
            model=model,
            num_layers=num_lora_layers,
            config=lora_parameters,
            use_dora=use_dora,
        )
    return model, wrapped_tokenizer._tokenizer  # pylint: disable=protected-access


def _loss_fn(
    model: mlx.nn.Module,
    inputs: mlx.core.array,
    targets: mlx.core.array,
    mask: mlx.core.array,
) -> mlx.core.array:
    # The second positional argument of mlx_lm models is the KV cache, not an
    # attention mask; passing an array there crashes on current mlx_lm. With no
    # cache the model builds its own causal mask.
    logits: mlx.core.array = model(inputs)
    # Use reduction="none" to get per-token losses, then apply mask correctly.
    # Using reduction="mean" would return a scalar that broadcasts incorrectly
    # when multiplied by mask (every masked position gets the same mean value).
    loss = mlx.nn.losses.cross_entropy(logits, targets, reduction="none") * mask
    normalized_loss = loss.sum() / mask.sum()
    return normalized_loss


@dataclasses.dataclass
class TrainingStats:
    """What one ``train_on_example`` call did.

    Attributes:
        steps: Optimizer steps taken.
        initial_loss: Loss reported by the first step (NaN if no step ran).
        final_loss: Loss reported by the last step (NaN if no step ran).
        stopped_early: True if a step's loss fell below the loss target before
            the step cap was reached.
        losses: Every step's loss, in order.
    """

    steps: int
    initial_loss: float
    final_loss: float
    stopped_early: bool
    losses: list[float] = dataclasses.field(default_factory=list)

    @property
    def hit_cap(self) -> bool:
        """The loop ran out of steps without reaching the loss target."""
        return self.steps > 0 and not self.stopped_early


def should_stop(loss: float, loss_target: float | None) -> bool:
    """Whether a step whose loss is ``loss`` satisfies ``loss_target``.

    ``None`` never stops (a fixed number of steps); otherwise stop once the
    loss is strictly below the target.
    """
    return loss_target is not None and loss < loss_target


def run_training_steps(
    step_fn: Callable[[], float],
    max_steps: int,
    loss_target: float | None,
    verbose: bool = False,
) -> TrainingStats:
    """Call ``step_fn`` until its loss drops below ``loss_target`` or ``max_steps``.

    ``step_fn`` performs one optimizer step and returns that step's loss. The
    target is checked after each step, so the step that crosses it is the last
    one and its loss is ``final_loss``.

    Args:
        step_fn: Performs one gradient step and returns its loss.
        max_steps: Hard cap on steps.
        loss_target: Stop as soon as a step's loss is below this; ``None``
            runs exactly ``max_steps`` steps.
        verbose: Print each step's loss.

    Returns:
        Per-call ``TrainingStats``.
    """
    losses: list[float] = []
    stopped_early = False
    for step in range(max(max_steps, 0)):
        loss = float(step_fn())
        losses.append(loss)
        if verbose:
            vizible.green(f"Step: {step}\tLoss: {loss:.4f}")
        if should_stop(loss, loss_target):
            stopped_early = True
            break
    return TrainingStats(
        steps=len(losses),
        initial_loss=losses[0] if losses else math.nan,
        final_loss=losses[-1] if losses else math.nan,
        stopped_early=stopped_early,
        losses=losses,
    )


class StatefulLLM:
    """Model container that bundles revision, learning, and serving logic."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        learning_rate: float = _LEARNING_RATE,
        max_tokens: int = MAX_TOKENS,
        epochs: int = _EPOCHS,
        num_lora_layers: int = _NUM_LORA_LAYERS,
        lora_parameters: dict | None = {**_LORA_PARAMETERS},
        use_dora: bool = _USE_DORA,
        loop_detection_sequence_length: int = _LOOP_DETECTION_SEQUENCE_LENGTH,
        loop_detection_max_repetitions: int = _LOOP_DETECTION_MAX_REPETITIONS,
        model_path: Path | None = MODEL_PATH,
        loss_target: float | None = _LOSS_TARGET,
        max_train_steps: int = _MAX_TRAIN_STEPS,
    ) -> None:
        """Initializes the StatefulLLM

        Args:
            model_name: Path or Huggingface name.
            learning_rate: Backpropagation hyperparameter.
            max_tokens: Maximum number of tokens to decode in a single turn.
            epochs: Number of optimizer steps ``_train`` runs when called
                without a loss target (the legacy fixed-count path).
            num_lora_layers: Number of LORA layers, if LORA is enabled.
            lora_parameters: LORA hyperparameters. If not None, LORA will be enabled.
            use_dora: Whether to use DORA, if LORA is enabled.
            loop_detection_sequence_length: Length of token sequence to check for repetition.
            loop_detection_max_repetitions: Number of times a sequence can repeat before stopping.
            model_path: Checkpoint directory; loaded from if it exists.
            loss_target: ``self_correct_and_train`` stops training a revision as
                soon as a step's loss falls below this. ``None`` disables the
                target and trains ``max_train_steps`` steps.
            max_train_steps: Step cap for ``self_correct_and_train``.

        Returns: None
        """
        # Guards the model weights between generation and training. Re-entrant
        # because ``self_correct_and_train`` calls ``generate_response`` internally.
        self._lock = threading.RLock()
        with self._lock:
            self._messages: list[dict[str, str]] = []
            self._model, self._tokenizer = _load(
                model_name,
                num_lora_layers=num_lora_layers,
                lora_parameters=lora_parameters,
                use_dora=use_dora,
                model_path=model_path,
            )
        self._model_path = model_path
        self._model_name = model_name
        self._optimizer = mlx.optimizers.AdamW(learning_rate=learning_rate)
        self._epochs = epochs
        self._max_tokens = max_tokens
        self._bos_token = self._tokenizer.special_tokens_map.get("bos_token", "")
        self._eos_token = self._tokenizer.special_tokens_map.get("eos_token", "")
        self._loop_detection_sequence_length = loop_detection_sequence_length
        self._loop_detection_max_repetitions = loop_detection_max_repetitions
        self._loss_target = loss_target
        self._max_train_steps = max_train_steps

        self._model_is_stable = True
        self._response_stream = None

    @property
    def ok(self) -> bool:
        """Checks if the model is in a stable state (i.e. not in the middle of backprop)."""
        return self._model_is_stable

    def apply_chat_template(
        self,
        messages: List[dict[str, str]],
    ) -> List[int]:
        """Applies chat template to messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Formatted chat string.
        """
        # transformers >= 5 returns a BatchEncoding unless return_dict=False;
        # mlx_lm.stream_generate needs a plain list of token ids.
        tokenized_prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        )
        return cast(list[int], tokenized_prompt)

    def _messages_for_prompt(
        self, prompt: str, use_history: bool
    ) -> List[dict[str, str]]:
        """Builds the message list to feed the chat template for a new user turn.

        When ``use_history`` is set the user turn is appended to the persistent
        conversation and the whole conversation is returned; the matching
        assistant turn must be added afterwards via ``_record_turn``.

        Args:
            prompt: User input.
            use_history: Whether to include (and extend) the persistent history.

        Returns:
            Messages to serialize with the chat template.
        """
        message = {"role": "user", "content": prompt}
        if use_history:
            self._messages.append(message)
            return self._messages
        return [message]

    def _record_turn(self, user: str, assistant: str) -> None:
        """Records a completed user/assistant exchange in the conversation history.

        If the most recent history entry is already the ``user`` turn (as appended by
        ``_messages_for_prompt``) only the assistant reply is added, so the history
        alternates user/assistant as the chat template expects.

        Args:
            user: The user prompt for this turn.
            assistant: The model's reply to that prompt.
        """
        last = self._messages[-1] if self._messages else None
        if last is None or last.get("role") != "user" or last.get("content") != user:
            self._messages.append({"role": "user", "content": user})
        self._messages.append({"role": "assistant", "content": assistant})

    def generate_response(
        self,
        prompt: str,
        use_history: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """Generates model response, including handling tokenization and de-tokenization.

        Args:
            prompt: User input.
            use_history: Whether to include previous interaction history within the prompt.
            max_tokens: Maximum number of tokens to decode in a single turn.

        Returns:
            Model-generated response.
        """
        print(flush=True)
        vizible.blue("Generating response for:")
        vizible.blue(prompt)
        unique_lines_generated = collections.defaultdict(int)
        if max_tokens is None:
            max_tokens = self._max_tokens

        with self._lock:
            messages = self._messages_for_prompt(prompt, use_history)
            tokenized_prompt = self.apply_chat_template(messages)

            # Use streaming generation internally to enable loop detection
            generated_tokens = []
            full_response = []
            current_line = []

            for response in stream_generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=tokenized_prompt,
                max_tokens=max_tokens,
            ):
                print(response.text, end="", flush=True)
                # Keep the chunk before any loop check so the final piece of text is
                # never silently dropped when generation stops early.
                full_response.append(response.text)
                current_line.append(response.text)
                if "\n" in response.text:
                    most_recent_line = "".join(current_line)
                    current_line = []
                    if most_recent_line:
                        if unique_lines_generated[most_recent_line] > 1:
                            vizible.red(
                                f"⚠️  Loop detected! Stopping generation early. Found: {most_recent_line}"
                            )
                            break
                        unique_lines_generated[most_recent_line] += 1
                # Track generated tokens
                # Note: response.token is the most recent token ID
                if hasattr(response, "token"):
                    generated_tokens.append(response.token)

                    # Check for loop
                    if _detect_token_loop(
                        generated_tokens,
                        self._loop_detection_sequence_length,
                        self._loop_detection_max_repetitions,
                    ):
                        vizible.red("⚠️  Loop detected! Stopping generation early.")
                        vizible.red(
                            f"   Generated {len(generated_tokens)} tokens before loop detection."
                        )
                        break

            model_response = "".join(full_response).strip()
            if use_history:
                self._record_turn(prompt, model_response)

        return model_response

    async def stream_response(
        self,
        prompt: str,
        use_history: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterable[str]:
        """Stream generated response.

        Args:
            prompt: User input.
            use_history: Whether to include previous interaction history within the prompt.
            max_tokens: Maximum number of tokens to decode in a single turn.

        Yields:
            Text chunks as they are generated.
        """
        if max_tokens is None:
            max_tokens = self._max_tokens
        print(f"{self._messages = }")
        responses = []
        generated_tokens = []

        self._lock.acquire()
        try:
            messages = self._messages_for_prompt(prompt, use_history)
            tokenized_prompt = self.apply_chat_template(messages)
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=tokenized_prompt,
                max_tokens=max_tokens,
            ):
                # Track generated tokens for loop detection
                if hasattr(response, "token"):
                    generated_tokens.append(response.token)

                    # Check for loop
                    if _detect_token_loop(
                        generated_tokens,
                        self._loop_detection_sequence_length,
                        self._loop_detection_max_repetitions,
                    ):
                        vizible.red(f"⚠️  Loop detected! Stopping generation early.")
                        vizible.red(
                            f"   Generated {len(generated_tokens)} tokens before loop detection."
                        )
                        break

                text = response.text
                responses.append(text)
                yield text
        finally:
            self._response_stream = "".join(responses)
            if use_history:
                self._record_turn(prompt, self._response_stream.strip())
            self._lock.release()

    def _tokenize(
        self, inp: str, dtype: mlx.core.Dtype = mlx.core.int32
    ) -> mlx.core.array:
        return mlx.core.array(
            self._tokenizer.encode(inp, add_special_tokens=False), dtype=dtype
        )

    def _self_correct(
        self,
        interaction_history: List[InteractionHistory],
        indices_to_review: List[int] | None,
        verbose: bool,
    ) -> TrainingExample:
        self._model_is_stable = False
        vizible.green("\n--- Starting Self-Correction and Training Cycle ---")
        if indices_to_review is None:
            indices_to_review = list(range(len(interaction_history)))
        if not interaction_history or not indices_to_review:
            raise ValueError("No unreviewed interactions to process.")
        if verbose:
            vizible.magenta(
                f"Found {len(interaction_history)} unreviewed interactions."
            )
        interactions_to_review = []
        for idx in indices_to_review:
            # Mark as reviewed to skip re-processing in the next cycle.
            interaction_history[idx].reviewed = True
            interactions_to_review.append(interaction_history[idx])

        # 1. Have the model re-evaluate its past responses and try to improve upon one of its turns.
        review_prompt = make_revision_prompt(interactions_to_review, self._tokenizer)
        llm_rewrite_response = self.generate_response(review_prompt, use_history=False)
        if verbose:
            vizible.blue(f"  - Response: {llm_rewrite_response}")

        # 2. Validate the revision response before using it for training.
        validate_revision_response(
            llm_rewrite_response,
            num_interactions=len(interactions_to_review),
        )

        # 3. Prepare training data to train the model on how it should have responded in this
        #    situation. Pass the same list as make_revision_prompt above: the [[X]] index in
        #    the response is a position within interactions_to_review, not interaction_history.
        #    think_mode="baseline": the interaction's raw llm_response carries the model's
        #    own reasoning, which stays in the (unmasked) prefix so training does not
        #    teach the model to skip reasoning.
        example = make_collated_training_example(
            llm_rewrite_response,
            interactions_to_review,
            self._tokenizer,
            think_mode="baseline",
        )
        return example

    def _train(self, example: TrainingExample, verbose: bool) -> TrainingStats:
        """Runs ``self._epochs`` fixed optimizer steps on ``example``."""
        with self._lock:
            return self._train_locked(example, verbose)

    def _train_locked(
        self,
        example: TrainingExample,
        verbose: bool,
        max_steps: int | None = None,
        loss_target: float | None = None,
    ) -> TrainingStats:
        """Runs optimizer steps on ``example``; caller must hold ``_lock``.

        The compiled step is built once per call and then run one step at a
        time by ``run_training_steps`` so training can stop on ``loss_target``.

        Args:
            example: Collated input/label/mask arrays.
            verbose: Print per-step losses and device info.
            max_steps: Step cap; defaults to ``self._epochs``.
            loss_target: Stop once a step's loss is below this; ``None`` runs
                exactly ``max_steps`` steps.

        Returns:
            ``TrainingStats`` for this call.
        """
        if max_steps is None:
            max_steps = self._epochs
        state = [self._model.state, self._optimizer.state, mlx.core.random.state]
        mlx.core.eval(state)
        loss_and_grad_fn = mlx.nn.value_and_grad(self._model, _loss_fn)

        @functools.partial(mlx.core.compile, inputs=state, outputs=state)
        def _step(inputs, labels, mask):
            loss, grads = loss_and_grad_fn(self._model, inputs, labels, mask)
            self._optimizer.update(self._model, grads)
            return loss

        if verbose:
            vizible.green(f"During training: {mlx.core.metal.device_info() = }")
        max_recommended_working_set_size = mlx.core.metal.device_info()[
            "max_recommended_working_set_size"
        ]
        assert isinstance(max_recommended_working_set_size, int)

        mlx.core.set_wired_limit(max_recommended_working_set_size)
        max_buffer_length = mlx.core.metal.device_info()["max_buffer_length"]
        assert isinstance(max_buffer_length, int)

        mlx.core.set_cache_limit(max_buffer_length)
        world = mlx.core.distributed.init()
        world_size = world.size()
        rank = world.rank()
        if world_size > 1:
            tqdm.tqdm.write(f"Node {rank} of {world_size}")

        progress = tqdm.tqdm(desc="Training", total=max_steps)

        def one_step() -> float:
            loss = _step(example.input, example.label, example.mask)
            mlx.core.eval(state, loss)
            progress.update(1)
            return loss.item()

        self._model.train(True)
        try:
            stats = run_training_steps(one_step, max_steps, loss_target, verbose)
        finally:
            self._model.train(False)
            progress.close()
        if verbose:
            vizible.cyan(
                f"Trained {stats.steps} steps, loss {stats.initial_loss:.4f} → "
                f"{stats.final_loss:.4f}"
                + (" (loss target reached)" if stats.stopped_early else "")
            )
        return stats

    def train_on_example(
        self,
        example: TrainingExample,
        iterations: int = 25,
        verbose: bool = False,
        save_checkpoint: bool = False,
        loss_target: float | None = None,
        max_steps: int | None = None,
    ) -> TrainingStats:
        """Train on a pre-constructed training example.

        Runs single optimizer steps until a step's loss is below
        ``loss_target`` or the step cap is reached. With ``loss_target=None``
        (the default) exactly ``iterations`` steps run, which keeps existing
        callers' behaviour.

        Args:
            example: Pre-constructed TrainingExample with input, label, and mask.
            iterations: Number of gradient steps; the cap when ``max_steps`` is
                None.
            verbose: Enable verbose logging.
            save_checkpoint: Whether to save the model after training.
            loss_target: Stop as soon as a step's loss is below this. The
                crossing step is the last one and its loss is ``final_loss``.
            max_steps: Step cap; defaults to ``iterations``.

        Returns:
            ``TrainingStats`` describing the steps taken.
        """
        if max_steps is None:
            max_steps = iterations
        self._model_is_stable = False
        try:
            with self._lock:
                stats = self._train_locked(
                    example,
                    verbose=verbose,
                    max_steps=max_steps,
                    loss_target=loss_target,
                )
                if save_checkpoint and self._model_path is not None:
                    self._save_checkpoint()
        finally:
            self._model_is_stable = True
        return stats

    def _save_checkpoint(self) -> None:
        """Writes the current weights to ``self._model_path``; caller must hold ``_lock``."""
        assert self._model_path is not None
        vizible.green(f"Saving model to {self._model_path}")
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(self._model_path, self._model)

    def self_correct_and_train(
        self,
        interaction_history: List[InteractionHistory],
        indices_to_review: List[int] | None = None,
        verbose: bool = False,
    ) -> bool:
        """Cycle in which the model revises a previously unreviewed prompt and trains from its rewrite.

        Args:
            interaction_history: Past user and model messages.
            indices_to_review: Optional indices of relevant interactions to revise within
                               interaction_history. If not set, all interactions will be reviewed.
            verbose: Enable verbose logging.

        Returns:
            Whether the process completed successfully.
        """
        self._model_is_stable = False
        try:
            with self._lock:
                # Prepare training example from self-reflective revision of past dialog.
                example = self._self_correct(
                    interaction_history, indices_to_review, verbose
                )

                # Train the model on the new, improved examples (backward-pass),
                # stopping on the loss target so the correction lands without
                # collapsing the model's reasoning.
                self.train_on_example(
                    example,
                    verbose=verbose,
                    save_checkpoint=True,
                    loss_target=self._loss_target,
                    max_steps=self._max_train_steps,
                )
        finally:
            # ``ok`` must recover even if revision validation or training raised.
            self._model_is_stable = True
        return True
