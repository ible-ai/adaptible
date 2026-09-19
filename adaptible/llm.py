"""Stateful LLM."""

import collections
import dataclasses
import re
import functools
import math
import threading
from pathlib import Path
from typing import Any, AsyncIterable, Callable, List, Sequence, Tuple, cast

import immutabledict
import mlx
import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
import mlx_lm.tuner
import tqdm
import vizible
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load
from transformers.tokenization_utils import PreTrainedTokenizer

from .classes import InteractionHistory, TrainingExample
from .paths import default_checkpoint_path
from .revise import (
    collate_training_examples,
    make_collated_training_example,
    make_revision_prompt,
    make_revision_training_example,
    rationale_from_output,
    template_opens_think,
    validate_revision_response,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         Default constants.                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# _MODEL_NAME = "lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit"
# _MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
_MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"
# _MODEL_NAME = "mlx-community/DeepSeek-R1-Qwen3-0528-8B-4bit-AWQ"
ADAPTER_FILE = "adapters.safetensors"
MAX_TOKENS = 2048
# Kept as a module-level name for backward compatibility (``autonomous/`` imports it);
# resolved through ``paths`` so that it honours ``$ADAPTIBLE_OUTPUTS_DIR``.
MODEL_PATH = default_checkpoint_path()
# MAX_TOKENS = 8192
_LEARNING_RATE = 2e-5
_EPOCHS = 5
# Training recipe. These are the settings of the self-repair cycle runs in
# ``results/``: a rank-8 adapter on the last 8 blocks, at most four steps toward
# an answer loss of 0.15, and the update kept only if it lands. Driving the loss
# lower, or training for a fixed count, makes the model emit the bare answer
# with no reasoning and answer the correction to unrelated questions. The loss
# target applies to the *answer* tokens (``TrainingExample.stop_mask``): with
# think_mode="rationale" the training loss also covers the reasoning, which is
# not what the stop rule is about.
_LOSS_TARGET: float | None = 0.15
_MAX_TRAIN_STEPS = 4
_NUM_LORA_LAYERS = 8
_LORA_PARAMETERS = immutabledict.immutabledict(
    {"rank": 8, "dropout": 0.0, "scale": 10.0}
)
# Repair (``StatefulLLM.repair``): candidates are the model's own answers to the
# question with the looked-up note attached, sampled from the adapter the
# process started with rather than the current one (sampling from the trained
# adapter drifted in three of four cycle runs; a frozen sampler did not).
_REPAIR_SAMPLES = 6
_REPAIR_CANDIDATES = 2
_REPAIR_TEMPERATURE = 0.7
_REPAIR_SAMPLE_TOKENS = 1024
_JUDGE_TOKENS = 768
# A kept repair must leave these answering (closed think, non-empty answer).
_CONTROL_PROMPTS = (
    "What is 12 times 12?",
    "What is the capital of France?",
    "How many days are in a week?",
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
    if model_path is not None and model_path.exists():
        # A checkpoint is the adapter only (``_save_checkpoint``); it is loaded
        # onto the freshly wrapped base model. Older checkpoints hold the whole
        # model; only their adapter tensors are taken.
        adapter_file = model_path / ADAPTER_FILE
        if adapter_file.exists():
            print("Loading adapter from", adapter_file)
            model.load_weights(str(adapter_file), strict=False)
        else:
            files = sorted(model_path.glob("*.safetensors"))
            weights = {
                k: v
                for f in files
                for k, v in mlx.core.load(str(f)).items()
                if "lora" in k
            }
            if weights:
                print(f"Loading {len(weights)} adapter tensors from {model_path}")
                model.load_weights(list(weights.items()), strict=False)
        mlx.core.eval(model.parameters())
    return model, wrapped_tokenizer._tokenizer  # pylint: disable=protected-access


def _loss_fn(
    model: mlx.nn.Module,
    inputs: mlx.core.array,
    targets: mlx.core.array,
    mask: mlx.core.array,
    stop_mask: mlx.core.array | None = None,
) -> Tuple[mlx.core.array, mlx.core.array]:
    """Masked-mean cross entropy: ``(train_loss, stop_loss)``.

    ``train_loss`` is the mean over ``mask`` and is what gradients are taken
    of. ``stop_loss`` is the mean over ``stop_mask`` (the answer tokens; see
    ``TrainingExample.stop_mask``) and is what the loss target is compared
    against; with ``stop_mask=None`` it is ``train_loss``. Both come from the
    same forward pass.

    ``mlx.core.value_and_grad`` (and so ``mlx.nn.value_and_grad``) accepts a
    function returning a tuple whose first element is the scalar to
    differentiate; the remaining elements are returned as auxiliary values
    untouched, so no second forward pass is needed for ``stop_loss``.
    """
    # The second positional argument of mlx_lm models is the KV cache, not an
    # attention mask; passing an array there crashes on current mlx_lm. With no
    # cache the model builds its own causal mask.
    logits: mlx.core.array = model(inputs)
    # Use reduction="none" to get per-token losses, then apply mask correctly.
    # Using reduction="mean" would return a scalar that broadcasts incorrectly
    # when multiplied by mask (every masked position gets the same mean value).
    per_token = mlx.nn.losses.cross_entropy(logits, targets, reduction="none")
    train_loss = (per_token * mask).sum() / mask.sum()
    if stop_mask is None:
        return train_loss, train_loss
    stop_loss = (per_token * stop_mask).sum() / stop_mask.sum()
    return train_loss, stop_loss


StepLoss = float | Tuple[float, float]
"""What a step function returns: one loss, or ``(train_loss, stop_loss)``."""


def _split_step_loss(loss: Any) -> Tuple[float, float]:
    """``(train_loss, stop_loss)`` from a scalar or a pair (scalar: both equal)."""
    if isinstance(loss, (tuple, list)):
        train_loss, stop_loss = loss
        return float(train_loss), float(stop_loss)
    return float(loss), float(loss)


@dataclasses.dataclass
class TrainingStats:
    """What one ``train_on_example`` call did.

    Attributes:
        steps: Optimizer steps taken.
        initial_loss: Answer (stop) loss reported by the first step (NaN if no
            step ran).
        final_loss: Answer (stop) loss reported by the last step (NaN if no
            step ran). This is the loss the stop rule compares against the
            target: the masked mean over ``TrainingExample.stop_mask`` (the
            answer tokens), or the training loss when there is no stop mask.
        stopped_early: True if a step's answer loss fell below the loss target
            before the step cap was reached.
        losses: Every step's answer loss, in order.
        final_train_loss: Training loss (masked mean over the whole target)
            at the last step; equals ``final_loss`` without a stop mask.
        train_losses: Every step's training loss, in order.
        rehearsal_final_loss: For ``train_on_examples``, the mean rehearsal
            loss at the last step; ``None`` when no rehearsal examples were
            trained (``train_on_example``, or an empty rehearsal list).
        rehearsal_count: Rehearsal examples in the joint objective (0 for
            ``train_on_example``).
        rehearsal_initial_loss: Mean rehearsal loss at the first step, before
            any update of this call; the anchor the rehearsal hinge measures
            drift from. ``None`` without rehearsal.
        rehearsal_active_steps: Number of ``(step, rehearsal example)`` pairs
            whose gradient was applied, i.e. where that example's loss had
            drifted more than ``rehearsal_margin`` above its initial loss
            (see ``active_rehearsal``). Out of ``steps * rehearsal_count``.

    All loss figures other than the rehearsal ones are the correction's: the
    stop rule only ever looks at the correction's answer loss.
    """

    steps: int
    initial_loss: float
    final_loss: float
    stopped_early: bool
    losses: list[float] = dataclasses.field(default_factory=list)
    rehearsal_final_loss: float | None = None
    rehearsal_count: int = 0
    final_train_loss: float = math.nan
    train_losses: list[float] = dataclasses.field(default_factory=list)
    rehearsal_initial_loss: float | None = None
    rehearsal_active_steps: int = 0

    @property
    def hit_cap(self) -> bool:
        """The loop ran out of steps without reaching the loss target."""
        return self.steps > 0 and not self.stopped_early

    @property
    def rehearsal_pairs(self) -> int:
        """``steps * rehearsal_count``: the pairs ``rehearsal_active_steps`` is out of."""
        return self.steps * self.rehearsal_count


def should_stop(loss: float, loss_target: float | None) -> bool:
    """Whether a step whose loss is ``loss`` satisfies ``loss_target``.

    ``None`` never stops (a fixed number of steps); otherwise stop once the
    loss is strictly below the target.
    """
    return loss_target is not None and loss < loss_target


def run_training_steps(
    step_fn: Callable[[], StepLoss],
    max_steps: int,
    loss_target: float | None,
    verbose: bool = False,
) -> TrainingStats:
    """Call ``step_fn`` until its answer loss drops below ``loss_target`` or ``max_steps``.

    ``step_fn`` performs one optimizer step and returns that step's loss,
    either a scalar or ``(train_loss, stop_loss)``. Only ``stop_loss`` (the
    answer-token loss; the scalar itself when only one is returned) is
    compared against the target. The target is checked after each step, so
    the step that crosses it is the last one and its loss is ``final_loss``.

    Args:
        step_fn: Performs one gradient step and returns its loss(es).
        max_steps: Hard cap on steps.
        loss_target: Stop as soon as a step's answer loss is below this;
            ``None`` runs exactly ``max_steps`` steps.
        verbose: Print each step's loss.

    Returns:
        Per-call ``TrainingStats``.
    """
    losses: list[float] = []
    train_losses: list[float] = []
    stopped_early = False
    for step in range(max(max_steps, 0)):
        train_loss, stop_loss = _split_step_loss(step_fn())
        losses.append(stop_loss)
        train_losses.append(train_loss)
        if verbose:
            extra = f"\tTrain loss: {train_loss:.4f}" if train_loss != stop_loss else ""
            vizible.green(f"Step: {step}\tLoss: {stop_loss:.4f}{extra}")
        if should_stop(stop_loss, loss_target):
            stopped_early = True
            break
    return TrainingStats(
        steps=len(losses),
        initial_loss=losses[0] if losses else math.nan,
        final_loss=losses[-1] if losses else math.nan,
        stopped_early=stopped_early,
        losses=losses,
        final_train_loss=train_losses[-1] if train_losses else math.nan,
        train_losses=train_losses,
    )


def active_rehearsal(
    losses: Sequence[float], initial_losses: Sequence[float], margin: float
) -> list[bool]:
    """Which rehearsal examples get a gradient this step (the rehearsal hinge).

    Example ``i`` is active when ``losses[i] > initial_losses[i] + margin``:
    its loss has drifted above where it started this training call by more
    than the margin. Rehearsal is meant to *anchor* the model, not to be
    minimised; with the gradient always on, a few hundred steps over a pool
    of a dozen items drove the rehearsal loss from ~0.4 to ~0.1 and the model
    memorised its own outputs. An inactive example contributes nothing that
    step.

    Args:
        losses: This step's per-example rehearsal losses.
        initial_losses: Each example's loss at the call's first step.
        margin: Allowed drift above the initial loss (>= 0).

    Returns:
        One flag per example.
    """
    if len(losses) != len(initial_losses):
        raise ValueError(
            f"losses and initial_losses differ in length: "
            f"{len(losses)} vs {len(initial_losses)}"
        )
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    return [
        float(loss) > float(initial) + margin
        for loss, initial in zip(losses, initial_losses)
    ]


JointStepResult = Tuple[StepLoss, float | None] | Tuple[StepLoss, float | None, int]


def run_joint_training_steps(
    step_fn: Callable[[], JointStepResult],
    max_steps: int,
    loss_target: float | None,
    rehearsal_count: int,
    verbose: bool = False,
) -> TrainingStats:
    """``run_training_steps`` for a joint correction + rehearsal objective.

    ``step_fn`` performs one optimizer step on the combined gradient and
    returns ``(correction_loss, mean_rehearsal_loss)`` or
    ``(correction_loss, mean_rehearsal_loss, active_count)``, where the
    correction loss is a scalar or a ``(train_loss, stop_loss)`` pair, the
    rehearsal loss is ``None`` when there are no rehearsal examples, and
    ``active_count`` is how many rehearsal examples' gradients the step
    applied (``active_rehearsal``; taken as 0 when omitted). Only the
    correction's answer loss is compared against ``loss_target``: rehearsal
    targets are the model's own baseline output, whose loss is already low,
    so stopping on it would end training before the correction lands.

    Args:
        step_fn: Performs one gradient step and returns the losses.
        max_steps: Hard cap on steps.
        loss_target: Stop as soon as the *correction's answer* loss is below
            this; ``None`` runs exactly ``max_steps`` steps.
        rehearsal_count: Rehearsal examples per step, recorded on the stats.
        verbose: Print each step's losses.

    Returns:
        ``TrainingStats`` whose loss fields are the correction's, whose
        ``rehearsal_initial_loss`` / ``rehearsal_final_loss`` are the first
        and last step's mean rehearsal loss, and whose
        ``rehearsal_active_steps`` sums the active counts.
    """
    first_rehearsal: list[float | None] = [None]
    last_rehearsal: list[float | None] = [None]
    active_total = [0]

    def correction_only() -> StepLoss:
        result = step_fn()
        loss_c, loss_r = result[0], result[1]
        active = int(result[2]) if len(result) > 2 else 0
        loss_r = None if loss_r is None else float(loss_r)
        if first_rehearsal[0] is None:
            first_rehearsal[0] = loss_r
        last_rehearsal[0] = loss_r
        active_total[0] += active
        if verbose and loss_r is not None:
            vizible.green(
                f"\tRehearsal loss: {loss_r:.4f} (active {active}/{rehearsal_count})"
            )
        return _split_step_loss(loss_c)

    stats = run_training_steps(correction_only, max_steps, loss_target, verbose)
    stats.rehearsal_initial_loss = first_rehearsal[0]
    stats.rehearsal_final_loss = last_rehearsal[0]
    stats.rehearsal_count = rehearsal_count
    stats.rehearsal_active_steps = active_total[0]
    return stats


def combine_grads(grads_c: Any, rehearsal_grads: list[Any], weight: float) -> Any:
    """``grads_c + weight * mean(rehearsal_grads)`` over nested trees of arrays.

    The trees must share the structure ``mlx.nn.value_and_grad`` produces for
    the model's trainable parameters. Rehearsal gradients are accumulated one
    tree at a time (a running sum, then scaled) so no padded batch is ever
    formed. With no rehearsal gradients or ``weight == 0`` the correction
    gradient is returned unchanged.

    Args:
        grads_c: Gradient tree of the correction example.
        rehearsal_grads: Gradient trees of the rehearsal examples.
        weight: Multiplier on the mean rehearsal gradient.

    Returns:
        The combined gradient tree.
    """
    if not rehearsal_grads or weight == 0:
        return grads_c
    total = rehearsal_grads[0]
    for grads_r in rehearsal_grads[1:]:
        total = mlx.utils.tree_map(lambda a, b: a + b, total, grads_r)
    scale = weight / len(rehearsal_grads)
    return mlx.utils.tree_map(lambda c, r: c + scale * r, grads_c, total)


def _prepare_training_device(verbose: bool) -> None:
    """Wire the Metal working set and cap the buffer cache before a training loop."""
    if verbose:
        vizible.green(f"During training: {mlx.core.metal.device_info() = }")
    max_recommended_working_set_size = mlx.core.metal.device_info()[
        "max_recommended_working_set_size"
    ]
    assert isinstance(max_recommended_working_set_size, int)

    mlx.core.set_wired_limit(max_recommended_working_set_size)
    max_buffer_length = mlx.core.metal.device_info()["max_buffer_length"]
    assert isinstance(max_buffer_length, int)

    # Cap the buffer cache: caching up to max_buffer_length (many GB) lets freed
    # activations accumulate across hundreds of training steps until Metal OOMs.
    mlx.core.set_cache_limit(min(max_buffer_length, 1 << 30))
    world = mlx.core.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.tqdm.write(f"Node {rank} of {world_size}")


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
        lookup: Callable[[str], str | None] | None = None,
        control_prompts: Sequence[str] = _CONTROL_PROMPTS,
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
            lookup: What the node consults about a flagged question (``repair``):
                a callable from the question to a short note, or ``None`` when it
                found nothing. ``DocStore.search`` is one; the autonomous node's
                web search is another. Without one, flagged interactions are
                left alone.
            control_prompts: Prompts that must still answer after a repair for
                the repair to be kept.

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
        self._learning_rate = learning_rate
        self._optimizer = mlx.optimizers.AdamW(learning_rate=learning_rate)
        self._epochs = epochs
        self._lookup = lookup
        self._control_prompts = tuple(control_prompts)
        self._max_tokens = max_tokens
        self._bos_token = self._tokenizer.special_tokens_map.get("bos_token", "")
        self._eos_token = self._tokenizer.special_tokens_map.get("eos_token", "")
        self._loop_detection_sequence_length = loop_detection_sequence_length
        self._loop_detection_max_repetitions = loop_detection_max_repetitions
        self._loss_target = loss_target
        self._max_train_steps = max_train_steps

        self._model_is_stable = True
        self._response_stream = None
        # The adapter the process started with: ``repair`` samples from it.
        self._base_adapter = self._snapshot()

    def reset_conversation(self) -> None:
        """Forgets the conversation so far; the next prompt starts a new chat."""
        with self._lock:
            self._messages.clear()

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
        #    think_mode="rationale": the revision generation's own reasoning becomes the
        #    reasoning in the target, so the model is trained on reasoning that actually
        #    concludes the revision rather than on "given the old reasoning, say X"
        #    (rationale_from_output: the think block, or the whole output when the model
        #    never closed the tag). There is no fallback without one: the empty-think
        #    target collapses the model, so an item with no rationale is not trained.
        rationale, _, _ = rationale_from_output(llm_rewrite_response, self._tokenizer)
        if not rationale and template_opens_think(
            self._tokenizer,
            [{"role": "user", "content": interactions_to_review[0].user_input}],
        ):
            raise ValueError("rationale required: the revision carried no reasoning")
        example = make_collated_training_example(
            llm_rewrite_response,
            interactions_to_review,
            self._tokenizer,
            think_mode="rationale",
            rationale=rationale,
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
        # ``_loss_fn`` returns ``(train_loss, stop_loss)``; value_and_grad
        # differentiates the first element and passes the tuple through.
        loss_and_grad_fn = mlx.nn.value_and_grad(self._model, _loss_fn)
        stop_mask = example.mask if example.stop_mask is None else example.stop_mask

        @functools.partial(mlx.core.compile, inputs=state, outputs=state)
        def _step(inputs, labels, mask, stop_mask):
            (train_loss, stop_loss), grads = loss_and_grad_fn(
                self._model, inputs, labels, mask, stop_mask
            )
            self._optimizer.update(self._model, grads)
            return train_loss, stop_loss

        _prepare_training_device(verbose)
        progress = tqdm.tqdm(desc="Training", total=max_steps)

        def one_step() -> Tuple[float, float]:
            train_loss, stop_loss = _step(
                example.input, example.label, example.mask, stop_mask
            )
            mlx.core.eval(state, train_loss, stop_loss)
            progress.update(1)
            return train_loss.item(), stop_loss.item()

        self._model.train(True)
        try:
            stats = run_training_steps(one_step, max_steps, loss_target, verbose)
        finally:
            self._model.train(False)
            progress.close()
        self._report_training(stats, verbose)
        return stats

    def _train_joint_locked(
        self,
        correction: TrainingExample,
        rehearsal: List[TrainingExample],
        verbose: bool,
        max_steps: int,
        loss_target: float | None,
        rehearsal_weight: float,
        rehearsal_margin: float = 0.05,
    ) -> TrainingStats:
        """Joint correction + rehearsal steps; caller must hold ``_lock``.

        Each step computes the correction's gradient and every rehearsal
        example's gradient as separate single-sequence forward/backward
        passes (never a padded batch, so peak memory stays that of one
        sequence), combines them with ``combine_grads``, and applies one
        optimizer update. Intermediate gradients are evaluated as they are
        produced so their activations are released before the next pass.

        Rehearsal is a hinge, not an objective: each rehearsal example's loss
        at the first step is its anchor, and on every step only the examples
        whose loss has drifted more than ``rehearsal_margin`` above their
        anchor contribute a gradient (``active_rehearsal``); the mean in
        ``combine_grads`` is over that active set. On the first step nothing
        has drifted, so no rehearsal gradient is applied.

        Args:
            correction: The corrected answer, whose answer loss drives the
                stop rule.
            rehearsal: Self-distillation examples anchoring the model.
            verbose: Print per-step losses and device info.
            max_steps: Step cap.
            loss_target: Stop once the correction's answer loss is below this.
            rehearsal_weight: Multiplier on the mean rehearsal gradient.
            rehearsal_margin: Drift above the initial loss that activates a
                rehearsal example's gradient (>= 0).

        Returns:
            ``TrainingStats`` for this call, rehearsal loss included.
        """
        state = [self._model.state, self._optimizer.state, mlx.core.random.state]
        mlx.core.eval(state)
        loss_and_grad_fn = mlx.nn.value_and_grad(self._model, _loss_fn)
        stop_mask_c = (
            correction.mask if correction.stop_mask is None else correction.stop_mask
        )

        _prepare_training_device(verbose)
        progress = tqdm.tqdm(desc="Training", total=max_steps)
        # Per-example rehearsal loss at the first step: the hinge's anchor.
        initial_losses: list[float] | None = None

        def one_step() -> Tuple[Tuple[float, float], float | None, int]:
            nonlocal initial_losses
            (train_c, stop_c), grads_c = loss_and_grad_fn(
                self._model,
                correction.input,
                correction.label,
                correction.mask,
                stop_mask_c,
            )
            mlx.core.eval(train_c, stop_c, grads_c)
            rehearsal_losses: list[float] = []
            rehearsal_grads: list[Any] = []
            for example in rehearsal:
                (loss_r, _), grads_r = loss_and_grad_fn(
                    self._model, example.input, example.label, example.mask
                )
                mlx.core.eval(loss_r, grads_r)
                rehearsal_losses.append(loss_r.item())
                rehearsal_grads.append(grads_r)
                del loss_r, grads_r
            if initial_losses is None:
                initial_losses = list(rehearsal_losses)
            active = active_rehearsal(
                rehearsal_losses, initial_losses, rehearsal_margin
            )
            active_grads = [g for g, on in zip(rehearsal_grads, active) if on]
            grads = combine_grads(grads_c, active_grads, rehearsal_weight)
            del grads_c, rehearsal_grads, active_grads
            self._optimizer.update(self._model, grads)
            mlx.core.eval(state)
            del grads
            progress.update(1)
            loss_r_mean = (
                sum(rehearsal_losses) / len(rehearsal_losses)
                if rehearsal_losses
                else None
            )
            return (train_c.item(), stop_c.item()), loss_r_mean, sum(active)

        self._model.train(True)
        try:
            stats = run_joint_training_steps(
                one_step, max_steps, loss_target, len(rehearsal), verbose
            )
        finally:
            self._model.train(False)
            progress.close()
        self._report_training(stats, verbose)
        return stats

    @staticmethod
    def _report_training(stats: TrainingStats, verbose: bool) -> None:
        if not verbose:
            return
        rehearsal = ""
        if stats.rehearsal_final_loss is not None:
            initial = (
                f"{stats.rehearsal_initial_loss:.4f}→"
                if stats.rehearsal_initial_loss is not None
                else ""
            )
            rehearsal = (
                f", rehearsal {initial}{stats.rehearsal_final_loss:.4f} "
                f"(k={stats.rehearsal_count}, active "
                f"{stats.rehearsal_active_steps}/{stats.rehearsal_pairs})"
            )
        train = (
            f", train loss {stats.final_train_loss:.4f}"
            if stats.final_train_loss != stats.final_loss
            else ""
        )
        vizible.cyan(
            f"Trained {stats.steps} steps, answer loss {stats.initial_loss:.4f} → "
            f"{stats.final_loss:.4f}{train}{rehearsal}"
            + (" (loss target reached)" if stats.stopped_early else "")
        )

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

        Runs single optimizer steps until a step's answer loss (the masked
        mean over ``example.stop_mask``, or the whole target without one) is
        below ``loss_target`` or the step cap is reached. With
        ``loss_target=None`` (the default) exactly ``iterations`` steps run,
        which keeps existing callers' behaviour.

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
            # Release cached activation buffers between items; otherwise a long
            # sequential run accumulates them until Metal reports OOM.
            mlx.core.clear_cache()
            self._model_is_stable = True
        return stats

    def train_on_examples(
        self,
        correction: TrainingExample,
        rehearsal: List[TrainingExample],
        *,
        loss_target: float | None,
        max_steps: int,
        rehearsal_weight: float = 1.0,
        rehearsal_margin: float = 0.05,
        verbose: bool = False,
        save_checkpoint: bool = False,
    ) -> TrainingStats:
        """Train on a correction jointly with rehearsal examples.

        Every optimizer step uses ``grad(correction) + rehearsal_weight *
        mean(grad(rehearsal_i) for active i)``, each gradient from its own
        single-sequence pass, and stops as soon as the *correction* loss is
        below ``loss_target`` or after ``max_steps`` steps. Training
        rehearsal examples in their own calls (``train_on_example`` each)
        does nothing under a loss target, since their loss already sits below
        it; folding them into the correction's step is what lets them anchor
        the model while the correction is being driven in.

        Rehearsal example ``i`` is *active* on a step only when its loss is
        more than ``rehearsal_margin`` above its loss at the call's first
        step (``active_rehearsal``). Rehearsal anchors; it is never minimised
        on its own: with the gradient always on, the rehearsal loss was driven
        from ~0.4 to ~0.1 over a run and the model memorised its own outputs.

        Args:
            correction: Pre-constructed example whose loss drives the stop rule.
            rehearsal: Self-distillation examples; may be empty, in which case
                this is ``train_on_example`` with the same target and cap.
            loss_target: Stop as soon as the correction loss is below this;
                ``None`` runs exactly ``max_steps`` steps.
            max_steps: Step cap.
            rehearsal_weight: Multiplier on the mean rehearsal gradient.
            rehearsal_margin: Drift above its initial loss that makes a
                rehearsal example's gradient count on a step (>= 0).
            verbose: Enable verbose logging.
            save_checkpoint: Whether to save the model after training.

        Returns:
            ``TrainingStats`` for the call; ``rehearsal_initial_loss`` /
            ``rehearsal_final_loss`` are the mean rehearsal loss at the first
            and last step and ``rehearsal_active_steps`` counts the
            ``(step, example)`` pairs whose gradient was applied.
        """
        if rehearsal_weight < 0:
            raise ValueError(f"rehearsal_weight must be >= 0, got {rehearsal_weight}")
        if rehearsal_margin < 0:
            raise ValueError(f"rehearsal_margin must be >= 0, got {rehearsal_margin}")
        self._model_is_stable = False
        try:
            with self._lock:
                if rehearsal:
                    stats = self._train_joint_locked(
                        correction,
                        list(rehearsal),
                        verbose=verbose,
                        max_steps=max_steps,
                        loss_target=loss_target,
                        rehearsal_weight=rehearsal_weight,
                        rehearsal_margin=rehearsal_margin,
                    )
                else:
                    stats = self._train_locked(
                        correction,
                        verbose=verbose,
                        max_steps=max_steps,
                        loss_target=loss_target,
                    )
                if save_checkpoint and self._model_path is not None:
                    self._save_checkpoint()
        finally:
            mlx.core.clear_cache()
            self._model_is_stable = True
        return stats

    # ------------------------------------------------------------------ repair
    def _checksum(self) -> float:
        return float(
            sum(
                mlx.core.abs(a).sum().item()
                for _, a in mlx.utils.tree_flatten(self._model.trainable_parameters())
            )
        )

    def _snapshot(self) -> Tuple[Any, float]:
        """Copies the adapter weights; ``_restore`` puts them back."""
        params = mlx.utils.tree_map(lambda x: x + 0, self._model.trainable_parameters())
        # Evaluate now: a lazy copy evaluated later from another thread (the
        # review runs in a worker thread) fails with "no Stream in current thread".
        mlx.core.eval(params)
        return params, self._checksum()

    def _restore(self, snap: Tuple[Any, float]) -> None:
        """Puts a snapshot's weights back, verified by checksum, with a fresh optimizer."""
        params, s = snap
        self._model.update(mlx.utils.tree_map(lambda x: x + 0, params))
        mlx.core.eval(self._model.parameters())
        self._optimizer = mlx.optimizers.AdamW(learning_rate=self._learning_rate)
        assert abs(self._checksum() - s) < 1e-3 * max(1.0, s), "restore failed"

    def _sample(self, prompt: str, seed: int, max_tokens: int) -> str:
        """One sampled (T=``_REPAIR_TEMPERATURE``) response, stopped on a token loop."""
        mlx.core.random.seed(seed)
        ids = self.apply_chat_template(self._messages_for_prompt(prompt, False))
        sampler = make_sampler(temp=_REPAIR_TEMPERATURE)
        toks: list[int] = []
        text: list[str] = []
        for r in stream_generate(
            self._model,
            self._tokenizer,
            prompt=ids,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            toks.append(r.token)
            text.append(r.text)
            if _detect_token_loop(
                toks,
                self._loop_detection_sequence_length,
                self._loop_detection_max_repetitions,
            ):
                break
        return "".join(text)

    @staticmethod
    def _answer_of(text: str) -> str:
        return text.split("</think>")[-1].strip()

    @staticmethod
    def _first_sentence(text: str) -> str:
        return re.split(r"(?<=[.!?])\s", text.strip(), 1)[0]

    def extract(self, question: str, text: str, label: str = "Reference note") -> str:
        """Names the answer to ``question`` using only ``text``; empty when it cannot.

        Used on what the node looked up and, with ``label="Reply"``, on the
        model's own answers, so a reply is judged by what it asserts rather
        than by which words it contains ("Proxima Centauri, the closest star
        to the Sun" contains "Sun"). A yes/no judge on this model said yes to
        wrong answers and no to right ones; a bare "what is the answer" read
        let its prior override a note that named a distractor. Quoting the
        answering sentence first and then naming the answer read every note
        right in a check. The name must occur in the text, so it cannot be
        invented. Greedy first, then a few sampled reads if that gave nothing.
        """
        prompt = (
            f"{label}: {text}\n\n"
            f"Question: {question}\n\n"
            f"First copy the one sentence of the {label.lower()} that answers the question. "
            "Then on a new line starting with 'Name:' write only the name that sentence gives."
        )
        for attempt in range(4):
            if attempt == 0:
                out = self.generate_response(
                    prompt, use_history=False, max_tokens=_JUDGE_TOKENS
                )
            else:
                out = self._sample(prompt, 7000 + attempt, _JUDGE_TOKENS)
            if "</think>" not in out:
                continue
            answer = self._answer_of(out)
            m = re.search(r"Name:\s*(.+)", answer)
            lines = [l for l in answer.splitlines() if l.strip()]
            name = (
                m.group(1)
                if m
                else (lines[-1] if lines and len(lines[-1].split()) <= 5 else "")
            )
            name = re.sub(r"[*_`\"']", "", name).strip().rstrip(".").strip()
            if name and len(name.split()) <= 5 and name.casefold() in text.casefold():
                return name
        return ""

    @staticmethod
    def _same_name(a: str, b: str) -> bool:
        norm = lambda x: re.sub(r"^(the|a|an)\s+", "", x.strip().casefold())
        return bool(a) and bool(b) and norm(a) == norm(b)

    def judge(self, question: str, reply: str, name: str) -> bool:
        """Whether ``reply`` asserts ``name`` as the answer, by the node's own reading of it.

        Only the first sentence counts: that is where the answer is asserted,
        and a later "Istanbul is the largest city" misleads the read. Cheap
        reject first (the name must occur in it), then the sentence is read
        like a note and the name it gives must match.
        """
        head = self._first_sentence(reply)
        if not head or name.casefold() not in head.casefold():
            return False
        return self._same_name(self.extract(question, head, label="Reply"), name)

    def rephrase(self, question: str, n: int = 3) -> list[str]:
        """The node's own rephrasings of ``question`` (greedy), up to ``n``.

        A repair is scored on the question plus these, so an update that fixes
        a rephrasing but not yet the original counts as progress and is kept;
        the cycle runs that produced the flagship result kept a third of their
        early updates that way, and the corrections accumulated.
        """
        prompt = (
            f"Rewrite this question in {n} different ways, one per line, with no numbering "
            f"and nothing else:\n{question}"
        )
        # The model tends to deliberate at length and not always close its think
        # block; the rephrasings it considers are quoted there, so take those too.
        # Greedy first, then sampled attempts until there are enough.
        seen, paras = {question.strip().casefold()}, []
        for attempt in range(4):
            out = (
                self.generate_response(prompt, use_history=False)
                if attempt == 0
                else self._sample(prompt, 8000 + attempt, self._max_tokens)
            )
            found = re.findall(r'["\u201c]([^"\u201d]{8,140}\?)["\u201d]', out)
            for line in self._answer_of(out).splitlines() if "</think>" in out else []:
                found.append(
                    re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip().strip('"')
                )
            for cand in found:
                cand = cand.strip()
                if (
                    cand.endswith("?")
                    and 3 <= len(cand.split()) <= 25
                    and cand.casefold() not in seen
                ):
                    seen.add(cand.casefold())
                    paras.append(cand)
                if len(paras) == n:
                    return paras
        return paras

    def _score_prompts(
        self, question: str, prompts: Sequence[str], name: str
    ) -> Tuple[int, str]:
        """Answers each prompt with the current weights; reads each answer under the frozen adapter."""
        answers = [self.generate_response(p, use_history=False) for p in prompts]
        cur = self._snapshot()
        try:
            self._restore(self._base_adapter)
            marks = [
                (
                    "✓"
                    if "</think>" in a
                    and self.judge(question, self._answer_of(a), name)
                    else "✗"
                )
                for a in answers
            ]
            return marks.count("✓"), "".join(marks)
        finally:
            self._restore(cur)

    def _controls_answer(self) -> bool:
        for q in self._control_prompts:
            out = self.generate_response(q, use_history=False, max_tokens=_JUDGE_TOKENS)
            if "</think>" not in out or not self._answer_of(out):
                return False
        return True

    def repair(
        self, interaction: InteractionHistory, note: str, verbose: bool = False
    ) -> bool:
        """Patches one flagged answer into the weights using what the node looked up.

        The model first names the answer from the note (``extract``) and writes
        three rephrasings of the question (``rephrase``). With the note attached
        to the question it then writes candidate answers (sampled from the
        adapter the process started with), keeps the ones that name that
        answer, trains a few steps on one, and re-asks the question and its
        rephrasings. The update stays only if the number of prompts answered
        with that name rises and the control prompts still answer; otherwise
        the weights are restored. Both candidates are tried; a kept update is
        the base for the next. Every step is logged with a ``REPAIR`` prefix.

        Args:
            interaction: The flagged exchange.
            note: What the lookup returned for its question.
            verbose: Print the sampled candidates.

        Returns:
            Whether a repair was kept.
        """
        q = interaction.user_input
        hinted = f"{q}\n\n(Reference note: {note})"
        with self._lock:
            trained = self._snapshot()
            try:
                # Candidates and the reading of the note come from the frozen starting adapter.
                self._restore(self._base_adapter)
                name = self.extract(q, note)
                print(f"REPAIR idx={interaction.idx} note_answer={name!r}", flush=True)
                if not name:
                    self._restore(trained)
                    return False
                cands: list[str] = []
                for i in range(_REPAIR_SAMPLES):
                    text = self._sample(
                        hinted, 1000 + 100 * interaction.idx + i, _REPAIR_SAMPLE_TOKENS
                    )
                    closed = "</think>" in text
                    first = (
                        self._first_sentence(self._answer_of(text)) if closed else ""
                    )
                    good = (
                        closed
                        and 0 < len(first) < 300
                        and self.judge(q, self._answer_of(text), name)
                    )
                    print(
                        f"REPAIR idx={interaction.idx} sample={i} clean={int(good)} | {first[:70]!r}",
                        flush=True,
                    )
                    if good:
                        cands.append(text)
                    if len(cands) == _REPAIR_CANDIDATES:
                        break
                prompts = [q] + self.rephrase(q)
                self._restore(trained)
                if not cands:
                    print(f"REPAIR idx={interaction.idx} NOCAND", flush=True)
                    return False
                best_n, best_marks = self._score_prompts(q, prompts, name)
                print(
                    f"REPAIR idx={interaction.idx} before={best_marks} prompts={len(prompts)}",
                    flush=True,
                )
                kept = False
                for k, text in enumerate(cands):
                    think = text.split("</think>")[0].strip()
                    ans = (
                        " ".join(re.split(r"(?<=[.!?])\s", self._answer_of(text))[:2])
                        .replace("\n", " ")
                        .strip()
                    )
                    example = make_revision_training_example(
                        f"[[0]] {ans} [[/0]]",
                        [InteractionHistory(idx=0, user_input=q, llm_response="")],
                        self._tokenizer,
                        think_mode="rationale",
                        rationale=think,
                    )
                    stats = self.train_on_examples(
                        collate_training_examples([example], self._tokenizer),
                        [],
                        loss_target=self._loss_target,
                        max_steps=self._max_train_steps,
                        rehearsal_weight=0.0,
                        verbose=verbose,
                    )
                    mlx.core.clear_cache()
                    n, marks = self._score_prompts(q, prompts, name)
                    ok = n > best_n and self._controls_answer()
                    print(
                        f"REPAIR idx={interaction.idx} cand={k} steps={stats.steps} loss={stats.final_loss:.2f} "
                        f"target={ans[:45]!r} | {marks} | {'KEEP' if ok else 'restore'}",
                        flush=True,
                    )
                    if ok:
                        kept = True
                        best_n, best_marks, trained = n, marks, self._snapshot()
                        if n == len(prompts):
                            break
                    else:
                        self._restore(trained)
                if kept and self._model_path is not None:
                    self._save_checkpoint()
            except BaseException:
                # Sampling temporarily swaps in the frozen adapter, and training
                # mutates weights before validation. Neither may leak on failure;
                # preserve only the last candidate that passed the keep checks.
                self._restore(trained)
                raise
        return kept

    def _save_checkpoint(self) -> None:
        """Writes the adapter weights to ``self._model_path``; caller must hold ``_lock``."""
        assert self._model_path is not None
        vizible.green(f"Saving adapter to {self._model_path}")
        self._model_path.mkdir(parents=True, exist_ok=True)
        weights = dict(mlx.utils.tree_flatten(self._model.trainable_parameters()))
        mlx.core.save_safetensors(str(self._model_path / ADAPTER_FILE), weights)

    def self_correct_and_train(
        self,
        interaction_history: List[InteractionHistory],
        indices_to_review: List[int] | None = None,
        verbose: bool = False,
    ) -> bool:
        """Reviews unreviewed interactions and trains on what it finds.

        Flagged interactions (a thumbs-down) go through ``repair``: the node
        looks the question up, writes a corrected answer, and patches it into
        the weights. With a ``lookup`` configured that is all a review does;
        unflagged interactions are marked reviewed and left alone. Without one,
        unflagged interactions take the older path: the model rewrites one of
        its own past turns and trains on the rewrite.

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
                if indices_to_review is None:
                    indices_to_review = list(range(len(interaction_history)))
                flagged = [
                    i for i in indices_to_review if interaction_history[i].flagged
                ]
                for i in flagged:
                    h = interaction_history[i]
                    h.reviewed = True
                    if not h.note and self._lookup is not None:
                        h.note = self._lookup(h.user_input) or ""
                    if not h.note:
                        print(f"REPAIR idx={h.idx} NONOTE", flush=True)
                        continue
                    self.repair(h, h.note, verbose=verbose)
                indices_to_review = [i for i in indices_to_review if i not in flagged]
                if not indices_to_review:
                    return True
                if self._lookup is not None:
                    # Repair mode: only what the user flagged is trained on.
                    for i in indices_to_review:
                        interaction_history[i].reviewed = True
                    return True
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
