"""Stateful LLM."""

from typing import AsyncIterable, cast, List, Tuple

import functools
import threading
import tqdm

import immutabledict
from mlx import optimizers
from mlx import nn
from mlx.nn import losses
from mlx.nn.utils import value_and_grad
import mlx.core as mx
from mlx_lm.generate import generate, stream_generate
from mlx_lm.utils import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from transformers.tokenization_utils import PreTrainedTokenizer
import vizible

from .libs import revise
from ._classes import InteractionHistory, TrainingExample


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         Default constants.                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
_MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"
# _MODEL_NAME = "mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ"
_MAX_TOKENS = 1024
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


def _load(
    model_name: str,
    num_lora_layers: int,
    use_dora: bool,
    lora_parameters: dict | None = None,
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """Load model parameters and tokenizer.

    Args:
        model_name: Path or Huggingface name.
        num_lora_layers: Number of LORA layers, if LORA is enabled.
        use_dora: Whether to use DORA, if LORA is enabled.
        lora_parameters: LORA hyperparameters. If not None, LORA will be enabled.

    Returns:
        Model and tokenizer.
    """
    model, wrapped_tokenizer = load(model_name)
    print("Freezing all non-Lora model parameters.")
    model.freeze()
    if lora_parameters is not None:
        linear_to_lora_layers(
            model=model,
            num_layers=num_lora_layers,
            config=lora_parameters,
            use_dora=use_dora,
        )
    return model, wrapped_tokenizer._tokenizer  # pylint: disable=protected-access


def _loss_fn(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> mx.array:
    logits: mx.array = model(inputs, mx.ones_like(inputs))
    loss = losses.cross_entropy(logits, targets, reduction="mean") * mask
    normalized_loss = loss.sum() / mask.sum()
    return normalized_loss


class StatefulLLM:
    """Model container that bundles revision, learning, and serving logic."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        learning_rate: float = _LEARNING_RATE,
        max_tokens: int = _MAX_TOKENS,
        epochs: int = _EPOCHS,
        num_lora_layers: int = _NUM_LORA_LAYERS,
        lora_parameters: dict | None = {**_LORA_PARAMETERS},
        use_dora: bool = _USE_DORA,
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
            )
        self._model_name = model_name
        self._optimizer = optimizers.AdamW(learning_rate=learning_rate)
        self._epochs = epochs
        self._max_tokens = max_tokens
        self._bos_token = self._tokenizer.special_tokens_map.get("bos_token", "")
        self._eos_token = self._tokenizer.special_tokens_map.get("eos_token", "")

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
        vizible.blue("Generating response for:")
        vizible.blue(prompt)
        message = {"role": "user", "content": prompt}
        if max_tokens is None:
            max_tokens = self._max_tokens
        if use_history:
            self._messages.append(message)
            messages = self._messages
        else:
            messages = [message]
        tokenized_prompt = self.apply_chat_template(messages)
        model_response = generate(
            self._model,
            self._tokenizer,
            prompt=tokenized_prompt,
            verbose=True,
            max_tokens=max_tokens,
        )
        if model_response is None:
            return None
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
        try:
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=tokenized_prompt,
                max_tokens=max_tokens,
            ):
                text = response.text
                responses.append(text)
                yield text
        finally:
            self._response_stream = "".join(responses)

    def _tokenize(self, inp: str, dtype: mx.Dtype = mx.int32) -> mx.array:
        return mx.array(
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
        review_prompt = revise.make_revision_prompt(
            interactions_to_review, self._tokenizer
        )
        llm_rewrite_response = self.generate_response(
            review_prompt, use_history=False, max_tokens=256
        )
        if verbose:
            vizible.blue(f"  - Response: {llm_rewrite_response}")

        # 2. Validate the revision response before using it for training.
        revise.validate_revision_response(
            llm_rewrite_response,
            num_interactions=len(interactions_to_review),
        )

        # 3. Prepare training data to train the model on how it should have responded in this
        #    situation.
        example = revise.make_collated_training_example(
            llm_rewrite_response, interactions_to_review, self._tokenizer
        )
        return example

    def _train(self, example: TrainingExample, verbose: bool):
        state = [self._model.state, self._optimizer.state, mx.random.state]
        mx.eval(state)
        loss_and_grad_fn = value_and_grad(self._model, _loss_fn)

        @functools.partial(mx.compile, inputs=state, outputs=state)
        def _step(inputs, labels, mask):
            loss, grads = loss_and_grad_fn(self._model, inputs, labels, mask)
            self._optimizer.update(self._model, grads)
            return loss

        # 3. Train the model on the new, improved examples (backward-pass)
        losses = []

        if verbose:
            vizible.green(f"During training: {mx.metal.device_info() = }")
        max_recommended_working_set_size = mx.metal.device_info()[
            "max_recommended_working_set_size"
        ]
        assert isinstance(max_recommended_working_set_size, int)

        mx.set_wired_limit(max_recommended_working_set_size)
        max_buffer_length = mx.metal.device_info()["max_buffer_length"]
        assert isinstance(max_buffer_length, int)

        mx.set_cache_limit(max_buffer_length)
        world = mx.distributed.init()
        world_size = world.size()
        rank = world.rank()
        if world_size > 1:
            tqdm.tqdm.write(f"Node {rank} of {world_size}")

        self._model.train(True)
        for epoch in tqdm.tqdm(
            range(self._epochs), desc="Training", total=self._epochs
        ):
            loss = _step(example.input, example.label, example.mask)
            mx.eval(state, loss)
            if verbose:
                vizible.green(f"Epoch: {epoch}\tLoss: {loss = }")
            losses.append(loss)

        self._model.train(False)
        if verbose:
            vizible.cyan(f"{losses = }")

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

        self._model_is_stable = True
        return True
