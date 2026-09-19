"""Loss-driven training with a hard bound on optimizer updates."""

import math
import time


def fit_masked_target(
    model,
    ids,
    labels,
    optimizer,
    *,
    max_steps=64,
    target_loss=0.15,
    attention_mask=None,
    stop_labels=None,
    stop_at_target=True,
    stop_after_update=False,
):
    """Fit caller-supplied masked targets; report loss after the final update.

    The caller owns parameter freezing, the optimizer, and prompt masking. A low
    training loss is only a stopping condition, not evidence to accept an adapter.
    With stop_labels, gradients cover labels (rationale plus final answer), while
    the stopping rule measures only the final-answer labels from the same forward.
    Set stop_at_target=False to continue an existing candidate for exactly
    max_steps additional updates while retaining its optimizer state.
    The stopping-mask mode always takes at least one step: an easy teacher-forced answer does
    not imply that the model has learned the supplied rationale.
    """
    import torch

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(target_loss) or target_loss <= 0:
        raise ValueError("target_loss must be positive and finite")
    if stop_labels is not None:
        if stop_labels.shape != labels.shape:
            raise ValueError("Stopping labels must match training-label shape.")
        supervised = stop_labels != -100
        if not torch.all((~supervised) | (stop_labels == labels)):
            raise ValueError("Stopping labels must be a subset of training labels.")
        if not torch.all(supervised[:, 1:].any(dim=1)):
            raise ValueError(
                "Each example needs at least one final-answer stopping token."
            )
    if not isinstance(stop_at_target, bool):
        raise ValueError("stop_at_target must be boolean")
    if not isinstance(stop_after_update, bool):
        raise ValueError("stop_after_update must be boolean")
    started = time.monotonic()
    if stop_after_update:
        return _fit_the_experiments_way(
            model,
            ids,
            labels,
            optimizer,
            max_steps=max_steps,
            target_loss=target_loss,
            attention_mask=attention_mask,
            stop_labels=stop_labels,
            stop_at_target=stop_at_target,
            started=started,
        )
    steps, initial_loss, initial_training_loss = 0, None, None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    options = {} if attention_mask is None else dict(attention_mask=attention_mask)
    while True:
        # Measure the weights AFTER the last allowed update without another
        # backward graph. An already-learned answer-only target needs no update;
        # rationale training still requires at least one actual optimizer step.
        with torch.set_grad_enabled(steps < max_steps):
            output = model(input_ids=ids, labels=labels, **options)
            loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss.")
        training_loss = loss.detach().item()
        if stop_labels is None:
            current_loss = training_loss
        else:
            # Select only supervised answer positions before casting to float32;
            # avoid a second full-sequence vocabulary-sized allocation or forward.
            with torch.no_grad():
                shifted = stop_labels[:, 1:]
                selected = shifted != -100
                answer_loss = torch.nn.functional.cross_entropy(
                    output.logits[:, :-1][selected].float(), shifted[selected]
                )
            if not torch.isfinite(answer_loss):
                raise ValueError("Training produced a non-finite final-answer loss.")
            current_loss = answer_loss.item()
        if initial_loss is None:
            initial_loss = current_loss
            initial_training_loss = training_loss
        if (
            stop_at_target
            and current_loss <= target_loss
            and (stop_labels is None or steps > 0)
        ):
            stop_reason = "target_loss"
            break
        if steps == max_steps:
            stop_reason = "max_steps"
            break
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    return dict(
        steps=steps,
        initial_loss=initial_loss,
        final_loss=current_loss,
        loss=current_loss,
        initial_training_loss=initial_training_loss,
        final_training_loss=training_loss,
        stopping_scope="final_answer" if stop_labels is not None else "training_target",
        stop_reason=stop_reason,
        max_steps=max_steps,
        target_loss=target_loss,
        elapsed_seconds=time.monotonic() - started,
    )


def _fit_the_experiments_way(
    model,
    ids,
    labels,
    optimizer,
    *,
    max_steps,
    target_loss,
    attention_mask,
    stop_labels,
    stop_at_target,
    started,
):
    """`run_training_steps`' loop: update first, then test the pre-update loss.

    The two rules are not the same and do not train the same adapter. The
    original (`llm.py::run_training_steps` with `should_stop`) applies an
    optimizer step, records the loss computed *before* that step, and stops
    when that loss is strictly below the target -- so it always performs one
    update after the crossing, and its reported `final_loss` is measured at the
    weights before the last update. The wrapper's default measures at the
    current weights and breaks *before* updating, with `<=`.

    Driven over the loss trajectory [0.62, 0.41, 0.18, 0.09, 0.05] at floor
    0.15 with a cap of 4: the original applies 4 updates, the default applies
    3. Same quoted recipe -- "4 steps, 0.15 on the answer tokens" -- and a
    different adapter at the end of it, which is the whole thing the port is
    supposed to reproduce.

    The default stays the default: it exists so an already-learned answer-only
    target costs no update, which is the served product's behaviour and not
    something the experiment has an opinion about.
    """
    import torch

    steps, initial_loss, initial_training_loss = 0, None, None
    current_loss = training_loss = math.nan
    stop_reason = "max_steps"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    options = {} if attention_mask is None else dict(attention_mask=attention_mask)
    while steps < max_steps:
        output = model(input_ids=ids, labels=labels, **options)
        loss = output.loss
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss.")
        training_loss = loss.detach().item()
        if stop_labels is None:
            current_loss = training_loss
        else:
            with torch.no_grad():
                shifted = stop_labels[:, 1:]
                selected = shifted != -100
                answer_loss = torch.nn.functional.cross_entropy(
                    output.logits[:, :-1][selected].float(), shifted[selected]
                )
            if not torch.isfinite(answer_loss):
                raise ValueError("Training produced a non-finite final-answer loss.")
            current_loss = answer_loss.item()
        if initial_loss is None:
            initial_loss, initial_training_loss = current_loss, training_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        # Strictly below, and only after the update: `should_stop` is
        # `loss < loss_target`, and the original checks it having already
        # stepped.
        if stop_at_target and current_loss < target_loss:
            stop_reason = "target_loss"
            break
    return dict(
        steps=steps,
        initial_loss=initial_loss,
        final_loss=current_loss,
        loss=current_loss,
        initial_training_loss=initial_training_loss,
        final_training_loss=training_loss,
        stopping_scope="final_answer" if stop_labels is not None else "training_target",
        stop_reason=stop_reason,
        max_steps=max_steps,
        target_loss=target_loss,
        elapsed_seconds=time.monotonic() - started,
    )
