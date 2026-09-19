"""Numerical portability of the original reasoning-training optimizer."""

import math
import unittest

import torch

from adaptible.wrap.training_optimizer import MLXAdamW, training_optimizer


class TrainingOptimizerTest(unittest.TestCase):
    def test_first_and_multiple_steps_match_uncorrected_scalar_formula(self):
        parameter = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float64))
        optimizer = MLXAdamW([parameter], lr=2e-5)
        expected, first, second = 0.25, 0.0, 0.0
        for gradient in (0.4, -0.2, 0.8, 0.0, -0.7):
            parameter.grad = torch.tensor([gradient], dtype=torch.float64)
            expected *= 1 - 2e-5 * 0.01
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            expected -= 2e-5 * first / (math.sqrt(second) + 1e-8)
            optimizer.step()
            self.assertAlmostEqual(parameter.item(), expected, places=13)

    def test_thinking_selects_original_optimizer_and_nonthinking_stays_torch(self):
        for thinking, optimizer_type, decay in (
            (True, MLXAdamW, 0.01),
            (False, torch.optim.AdamW, 0.0),
        ):
            optimizer = training_optimizer(
                [torch.nn.Parameter(torch.tensor([1.0]))], thinking=thinking
            )
            self.assertIs(type(optimizer), optimizer_type)
            self.assertEqual(optimizer.param_groups[0]["lr"], 2e-5)
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], decay)

    def test_missing_gradient_does_not_update_or_initialize_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = MLXAdamW([parameter], lr=2e-5)
        optimizer.step()
        self.assertEqual(parameter.item(), 1.0)
        self.assertEqual(len(optimizer.state), 0)

    def test_first_and_multiple_steps_match_installed_mlx_on_cpu(self):
        try:
            import mlx.core as mx
            from mlx.optimizers import AdamW
        except ImportError:
            self.skipTest(
                "MLX parity check requires Apple Silicon; scalar test is portable"
            )
        original_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            parameter = torch.nn.Parameter(torch.tensor([0.25, -0.75]))
            optimizer = MLXAdamW([parameter], lr=2e-5)
            mlx_optimizer = AdamW(learning_rate=2e-5)
            mlx_parameters = {"weight": mx.array([0.25, -0.75])}
            for gradient in ([0.4, -0.2], [-0.2, 0.8], [0.8, 0.0], [-0.7, 0.3]):
                parameter.grad = torch.tensor(gradient)
                optimizer.step()
                mlx_parameters = mlx_optimizer.apply_gradients(
                    {"weight": mx.array(gradient)}, mlx_parameters
                )
                mx.eval(mlx_parameters, mlx_optimizer.state)
                torch.testing.assert_close(
                    parameter.detach(),
                    torch.tensor(mlx_parameters["weight"].tolist()),
                    rtol=0,
                    atol=1e-7,
                )
        finally:
            mx.set_default_device(original_device)
