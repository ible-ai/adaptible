"""Tests to verify and demonstrate training bugs and their fixes.

This module tests two critical bugs in the training pipeline:

1. Mask slicing bug: The mask in make_collated_training_example is not sliced
   to match the input/label after [:-1] and [1:] slicing, causing misalignment.

2. Loss reduction bug: The loss function uses reduction="mean" which returns
   a scalar, then multiplies by mask (broadcasting incorrectly) instead of
   using reduction="none" for per-token losses.

Run with:
    python -m adaptible._src.tests.dev.training_bugs_test
"""

import inspect
import unittest

import mlx.core as mx
from mlx.nn import losses
from typing import cast, List

from adaptible._src.libs import revise
from adaptible._src import _llm


class MaskSlicingBugTest(unittest.TestCase):
    """Test demonstrating the mask slicing bug."""

    def test_mask_alignment_bug(self):
        """The mask should be sliced to match input/label dimensions."""
        # Simulate what make_collated_training_example does
        dialog_pre = mx.array([1, 2, 3, 4, 5])  # 5 tokens of prompt
        revision = mx.array([6, 7, 8])  # 3 tokens of revision

        full_sequence = mx.concat(
            [dialog_pre, revision]
        )  # [1,2,3,4,5,6,7,8] - 8 tokens
        mask = mx.concat(
            [mx.zeros_like(dialog_pre), mx.ones_like(revision)]
        )  # [0,0,0,0,0,1,1,1]

        # Current (buggy) behavior: input/label are sliced but mask is not
        input_buggy = full_sequence[:-1]  # [1,2,3,4,5,6,7] - 7 tokens
        label_buggy = full_sequence[1:]  # [2,3,4,5,6,7,8] - 7 tokens
        mask_buggy = mask  # [0,0,0,0,0,1,1,1] - 8 tokens (WRONG!)

        # The mask has 8 elements but input/label have 7
        self.assertEqual(len(input_buggy), 7)
        self.assertEqual(len(label_buggy), 7)
        self.assertEqual(len(mask_buggy), 8)  # BUG: should be 7

        # This means the mask is misaligned - position 5 in mask (first 1)
        # should correspond to position 5 in input (token 6), but the
        # sizes don't match so broadcasting/indexing will be wrong

        # Fixed behavior: mask should also be sliced
        # We want to mask the LABELS (what we're predicting), not the inputs
        # So mask should be sliced as [1:] to align with labels
        mask_fixed = mask[1:]  # [0,0,0,0,1,1,1] - 7 tokens

        self.assertEqual(len(input_buggy), len(mask_fixed))
        self.assertEqual(len(label_buggy), len(mask_fixed))

        # Verify the mask still covers the right positions
        # Label positions 4,5,6 correspond to tokens 6,7,8 (the revision)
        # Those should be masked (1), rest should be 0
        self.assertEqual(mask_fixed[4].item(), 1)  # predicting token 6
        self.assertEqual(mask_fixed[5].item(), 1)  # predicting token 7
        self.assertEqual(mask_fixed[6].item(), 1)  # predicting token 8
        self.assertEqual(mask_fixed[3].item(), 0)  # predicting token 5 (prompt)


class LossReductionBugTest(unittest.TestCase):
    """Test demonstrating the loss reduction bug."""

    def test_reduction_mean_bug(self):
        """Using reduction='mean' then multiplying by mask is incorrect."""
        # Simulate logits and targets
        batch_size = 1
        seq_len = 8
        vocab_size = 100

        # Random logits
        mx.random.seed(42)
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        # Mask: only train on last 3 positions
        mask = mx.array([[0, 0, 0, 0, 0, 1, 1, 1]])

        # BUGGY: reduction="mean" returns scalar, then * mask broadcasts wrong
        loss_buggy = losses.cross_entropy(logits, targets, reduction="mean") * mask
        # This creates a tensor of shape (1, 8) where EVERY position has the same
        # mean loss value (broadcast from scalar), then masked
        # The .sum() / mask.sum() gives wrong normalization

        # What's happening:
        # 1. cross_entropy with mean returns a single scalar (mean of ALL tokens)
        # 2. scalar * mask broadcasts to mask shape
        # 3. Each masked position gets the FULL mean, not its individual loss

        # CORRECT: reduction="none" gives per-token losses
        loss_per_token = losses.cross_entropy(logits, targets, reduction="none")
        # Shape: (1, 8) - one loss per position

        # Then apply mask and normalize correctly
        masked_loss = loss_per_token * mask
        correct_loss = masked_loss.sum() / mask.sum()

        # The buggy version computes wrong values
        buggy_result = loss_buggy.sum() / mask.sum()

        # They should differ (bug demonstration)
        # The buggy version essentially computes: mean_all_tokens * num_masked_tokens / num_masked_tokens
        # = mean_all_tokens (wrong - includes non-masked tokens in mean)
        #
        # The correct version computes: sum_masked_tokens / num_masked_tokens
        # = mean_masked_tokens (right - only masked tokens contribute)

        print(f"Buggy loss: {buggy_result.item():.6f}")
        print(f"Correct loss: {correct_loss.item():.6f}")

        # In general these will differ unless by coincidence
        # The buggy version includes all tokens in the mean, then "unmasks"
        # The correct version only averages masked tokens

    def test_correct_loss_only_trains_masked_positions(self):
        """Verify that correct loss function only considers masked positions."""
        batch_size = 1
        seq_len = 5
        vocab_size = 10

        # Logits where position 0-2 predict wrong, position 3-4 predict correct
        logits = mx.zeros((batch_size, seq_len, vocab_size))
        # Make position 3 and 4 have high logit for the correct token
        targets = mx.array([[5, 5, 5, 5, 5]])  # All targets are token 5

        # Set logits so positions 0-2 are wrong, 3-4 are right
        logits_np = logits.tolist()
        logits_np = cast(List[List[List[float]]], logits_np)
        for pos in range(3):
            logits_np[0][pos][0] = 10.0  # Wrong prediction
        for pos in range(3, 5):
            logits_np[0][pos][5] = 10.0  # Correct prediction
        logits = mx.array(logits_np)

        # Only mask positions 3-4 (the correct ones)
        mask_correct_only = mx.array([[0, 0, 0, 1, 1]], dtype=mx.float32)

        # Only mask positions 0-2 (the wrong ones)
        mask_wrong_only = mx.array([[1, 1, 1, 0, 0]], dtype=mx.float32)

        # Compute losses correctly
        loss_per_token = losses.cross_entropy(logits, targets, reduction="none")

        loss_correct_only = (
            loss_per_token * mask_correct_only
        ).sum() / mask_correct_only.sum()
        loss_wrong_only = (
            loss_per_token * mask_wrong_only
        ).sum() / mask_wrong_only.sum()

        # Loss on correct predictions should be low
        # Loss on wrong predictions should be high
        print(f"Loss on correct positions: {loss_correct_only.item():.6f}")
        print(f"Loss on wrong positions: {loss_wrong_only.item():.6f}")

        self.assertLess(cast(float, loss_correct_only.item()), 1.0)  # Low loss
        self.assertGreater(cast(float, loss_wrong_only.item()), 5.0)  # High loss


def run_bug_demonstration():
    """Run the bug demonstrations."""
    print("=" * 70)
    print("TRAINING BUG DEMONSTRATION")
    print("=" * 70)
    print()

    print("1. MASK SLICING BUG")
    print("-" * 70)
    print(
        """
The mask in make_collated_training_example has 8 elements but
input/label have 7 elements after slicing. This causes misalignment.

Current code:
    input = sequence[:-1]   # 7 elements
    label = sequence[1:]    # 7 elements
    mask = mask             # 8 elements (BUG!)

Fixed code:
    input = sequence[:-1]   # 7 elements
    label = sequence[1:]    # 7 elements
    mask = mask[1:]         # 7 elements (FIXED)
"""
    )

    print("\n2. LOSS REDUCTION BUG")
    print("-" * 70)
    print(
        """
The loss function uses reduction="mean" which returns a scalar,
then multiplies by mask. This broadcasts incorrectly.

Current code:
    loss = cross_entropy(logits, targets, reduction="mean") * mask
    # scalar * mask broadcasts: every masked position gets same loss value

Fixed code:
    loss = cross_entropy(logits, targets, reduction="none") * mask
    # per-token losses * mask: each position gets its own loss value
"""
    )

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(MaskSlicingBugTest)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(LossReductionBugTest))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class FixVerificationTest(unittest.TestCase):
    """Verify that the fixes are actually applied in the codebase."""

    def test_mask_slicing_fix_applied(self):
        """Verify make_collated_training_example uses mask[1:]."""
        source = inspect.getsource(revise.make_collated_training_example)
        # The fix should have mask=mask[1:] not mask=mask
        self.assertIn("mask[1:]", source, "Mask slicing fix not applied!")
        # Should NOT have the buggy version
        self.assertNotIn("mask=mask,", source, "Buggy mask assignment still present!")

    def test_loss_reduction_fix_applied(self):
        """Verify _loss_fn uses reduction='none'."""
        source = inspect.getsource(_llm._loss_fn)
        # The fix should have reduction="none" in the actual cross_entropy call
        self.assertIn(
            'cross_entropy(logits, targets, reduction="none")',
            source,
            "Loss reduction fix not applied!",
        )
        # Should NOT have the buggy version in the actual call
        self.assertNotIn(
            'cross_entropy(logits, targets, reduction="mean")',
            source,
            "Buggy reduction='mean' still present in cross_entropy call!",
        )


if __name__ == "__main__":
    run_bug_demonstration()

    # Also run fix verification
    print("\n" + "=" * 70)
    print("FIX VERIFICATION")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(FixVerificationTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✓ All fixes verified in codebase!")
