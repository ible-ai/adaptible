"""Stateful LLM."""

import collections
import functools
import threading
from pathlib import Path
from typing import AsyncIterable, List, Tuple, cast

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
MODEL_PATH = Path("outputs/autonomous/checkpoint")
# MAX_TOKENS = 8192
_LEARNING_RATE = 5e-5
_EPOCHS = 5
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
        end_idx = -i * sequence_length if i > 0 else None
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
    logits: mlx.core.array = model(inputs, mlx.core.ones_like(inputs))
    # Use reduction="none" to get per-token losses, then apply mask correctly.
    # Using reduction="mean" would return a scalar that broadcasts incorrectly
    # when multiplied by mask (every masked position gets the same mean value).
    loss = mlx.nn.losses.cross_entropy(logits, targets, reduction="none") * mask
    normalized_loss = loss.sum() / mask.sum()
    return normalized_loss


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
    ) -> None:
        """Initializes the StatefulLLM

        Args:
            model_name: Path or Huggingface name.
            learning_rate: Backpropagation hyperparameter.
            max_tokens: Maximum number of tokens to decode in a single turn.
            epochs: Number of training epochs to perform on self-reflective model revisions.
            num_lora_layers: Number of LORA layers, if LORA is enabled.
            lora_parameters: LORA hyperparameters. If not None, LORA will be enabled.
            use_dora: Whether to use DORA, if LORA is enabled.
            loop_detection_sequence_length: Length of token sequence to check for repetition.
            loop_detection_max_repetitions: Number of times a sequence can repeat before stopping.

        Returns: None
        """
        self._lock = threading.Lock()
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
        tokenized_prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return cast(list[int], tokenized_prompt)

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
        message = {"role": "user", "content": prompt}
        if max_tokens is None:
            max_tokens = self._max_tokens
        if use_history:
            self._messages.append(message)
            messages = self._messages
        else:
            messages = [message]
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

            full_response.append(response.text)

        model_response = "".join(full_response)

        if model_response is None:
            return ""
        return model_response.strip()

    async def stream_response(
        self,
        prompt: str,
        use_history: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterable[str]:
        """Stream generated response"""
        message = {"role": "user", "content": prompt}
        if max_tokens is None:
            max_tokens = self._max_tokens
        print(f"{self._messages = }")
        if use_history:
            self._messages.append(message)
            messages = self._messages
        else:
            messages = [message]
        tokenized_prompt = self.apply_chat_template(messages)
        responses = []
        generated_tokens = []
        loop_detected = False

        try:
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
        #    situation.
        example = make_collated_training_example(
            llm_rewrite_response, interactions_to_review, self._tokenizer
        )
        return example

    def _train(self, example: TrainingExample, verbose: bool):
        state = [self._model.state, self._optimizer.state, mlx.core.random.state]
        mlx.core.eval(state)
        loss_and_grad_fn = mlx.nn.value_and_grad(self._model, _loss_fn)

        @functools.partial(mlx.core.compile, inputs=state, outputs=state)
        def _step(inputs, labels, mask):
            loss, grads = loss_and_grad_fn(self._model, inputs, labels, mask)
            self._optimizer.update(self._model, grads)
            return loss

        # 3. Train the model on the new, improved examples (backward-pass)
        losses = []

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

        self._model.train(True)
        for epoch in tqdm.tqdm(
            range(self._epochs), desc="Training", total=self._epochs
        ):
            loss = _step(example.input, example.label, example.mask)
            mlx.core.eval(state, loss)
            if verbose:
                vizible.green(f"Epoch: {epoch}\tLoss: {loss = }")
            losses.append(loss)

        self._model.train(False)
        if verbose:
            vizible.cyan(f"{losses = }")

    def train_on_example(
        self,
        example: TrainingExample,
        iterations: int = 25,
        verbose: bool = False,
        save_checkpoint: bool = False,
    ) -> None:
        """Train on a pre-constructed training example for N iterations.

        This is a convenience wrapper around _train() that handles the
        iteration loop and optional checkpointing.

        Args:
            example: Pre-constructed TrainingExample with input, label, and mask.
            iterations: Total number of training iterations (epochs * calls).
            verbose: Enable verbose logging.
            save_checkpoint: Whether to save the model after training.
        """
        self._model_is_stable = False
        calls = iterations // self._epochs
        for _ in range(calls):
            self._train(example, verbose=verbose)

        if save_checkpoint and self._model_path is not None:
            vizible.green(f"Saving model to {self._model_path}")
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(self._model_path, self._model)

        self._model_is_stable = True

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

        # Prepare training example from self-reflective revision of past dialog.
        example = self._self_correct(interaction_history, indices_to_review, verbose)

        # Train the model on the new, improved examples (backward-pass)
        self._train(example, verbose)
        if self._model_path is not None:
            vizible.green(f"Saving model to {self._model_path}")
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(self._model_path, self._model)

        self._model_is_stable = True
        return True
