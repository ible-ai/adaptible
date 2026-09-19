"""Preserve the flagship optimizer's update rule for grounded reasoning."""

import math

import torch


class MLXAdamW(torch.optim.Optimizer):
    """AdamW without bias correction, as used by the original MLX recipe.

    This is the existing cycles_torch.MLXAdamW update: decoupled weight decay,
    then uncorrected first and second moments. Ordinary torch AdamW's bias
    correction materially reduces early updates with a fresh optimizer.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (
                group["lr"],
                group["betas"],
                group["eps"],
                group["weight_decay"],
            )
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                first, second = state["m"], state["v"]
                parameter.mul_(1 - lr * wd)
                first.mul_(b1).add_(parameter.grad, alpha=1 - b1)
                second.mul_(b2).addcmul_(parameter.grad, parameter.grad, value=1 - b2)
                parameter.addcdiv_(first, second.sqrt().add_(eps), value=-lr)


# The flagship recipe. Production never overrides these; the override exists so
# a test can drive the same code path at a scale its toy model can learn at.
LEARNING_RATE = 2e-5
THINKING_WEIGHT_DECAY = 0.01


def training_optimizer(parameters, *, thinking=False, learning_rate=None):
    """Build the optimizer for one candidate.

    Args:
        parameters: The trainable (LoRA) parameters.
        thinking: Use the original MLX-matched uncorrected AdamW.
        learning_rate: Override the flagship rate. Defaults to
            :data:`LEARNING_RATE`.

    Raises:
        ValueError: The learning rate is not a positive finite number.
    """
    rate = LEARNING_RATE if learning_rate is None else float(learning_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if thinking:
        return MLXAdamW(parameters, lr=rate, weight_decay=THINKING_WEIGHT_DECAY)
    return torch.optim.AdamW(parameters, lr=rate, weight_decay=0.0)
