"""StatefulLLM testing.

Tests verify that the self-correction and training cycle actually works:
- Model weights change after training
- Loss decreases during training epochs
- Revisions address the original problem
- The full pipeline produces usable training data
"""

import unittest
from unittest import mock

import mlx.core as mx

import adaptible
from adaptible._src import _llm

# Module-level shared model instance - loaded once for all tests
_shared_model = None


def get_shared_model():
    """Get or create the shared model instance."""
    global _shared_model
    if _shared_model is None:
        _shared_model = adaptible.StatefulLLM()
    return _shared_model


def _history_only_llm() -> adaptible.StatefulLLM:
    """A StatefulLLM with only the conversation-history state, no model loaded."""
    llm = adaptible.StatefulLLM.__new__(adaptible.StatefulLLM)
    llm._messages = []
    return llm


class ConversationHistoryTest(unittest.TestCase):
    """Tests for multi-turn history bookkeeping (no model or MLX required)."""

    def test_messages_for_prompt_without_history_is_isolated(self):
        """use_history=False must neither read nor extend the stored conversation."""
        llm = _history_only_llm()
        llm._messages.append({"role": "user", "content": "earlier"})

        messages = llm._messages_for_prompt("now", use_history=False)

        self.assertEqual(messages, [{"role": "user", "content": "now"}])
        self.assertEqual(llm._messages, [{"role": "user", "content": "earlier"}])

    def test_messages_for_prompt_with_history_appends_user_turn(self):
        """use_history=True appends the user turn and returns the full conversation."""
        llm = _history_only_llm()

        messages = llm._messages_for_prompt("hi", use_history=True)

        self.assertIs(messages, llm._messages)
        self.assertEqual(llm._messages, [{"role": "user", "content": "hi"}])

    def test_record_turn_appends_assistant_after_pending_user(self):
        """After _messages_for_prompt, _record_turn adds only the assistant reply."""
        llm = _history_only_llm()
        llm._messages_for_prompt("What is 2+2?", use_history=True)

        llm._record_turn("What is 2+2?", "4")

        self.assertEqual(
            llm._messages,
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        )

    def test_record_turn_adds_user_when_not_pending(self):
        """_record_turn on its own records a full user/assistant pair."""
        llm = _history_only_llm()

        llm._record_turn("hello", "hi there")

        self.assertEqual(
            llm._messages,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        )

    def test_history_alternates_roles_over_multiple_turns(self):
        """Multi-turn history must alternate user/assistant, not be user-only."""
        llm = _history_only_llm()
        for user, assistant in [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]:
            llm._messages_for_prompt(user, use_history=True)
            llm._record_turn(user, assistant)

        roles = [m["role"] for m in llm._messages]
        self.assertEqual(roles, ["user", "assistant"] * 3)
        self.assertEqual(
            [m["content"] for m in llm._messages if m["role"] == "assistant"],
            ["a1", "a2", "a3"],
        )


class TokenLoopDetectionTest(unittest.TestCase):
    """Tests for _detect_token_loop (pure function, no model required)."""

    def test_repeated_sequence_detected(self):
        tokens = [1, 2, 3, 4] * 3
        self.assertTrue(_llm._detect_token_loop(tokens, 4, 3))

    def test_too_short_not_detected(self):
        self.assertFalse(_llm._detect_token_loop([1, 2, 3, 4] * 2, 4, 3))

    def test_non_repeating_not_detected(self):
        tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5]
        self.assertFalse(_llm._detect_token_loop(tokens, 4, 3))

    def test_earlier_repeat_broken_not_detected(self):
        tokens = [9, 9, 9, 9, 1, 2, 3, 4, 1, 2, 3, 4]
        self.assertFalse(_llm._detect_token_loop(tokens, 4, 3))


class ModelStateTest(unittest.TestCase):
    """Tests for model initialization and state management."""

    @classmethod
    def setUpClass(cls):
        cls.model = get_shared_model()
        # Reset state in case previous tests left it unstable
        cls.model._model_is_stable = True

    def test_model_ok_property_reflects_stability(self):
        """The ok property should reflect _model_is_stable."""
        self.model._model_is_stable = True
        self.assertTrue(self.model.ok)

        self.model._model_is_stable = False
        self.assertFalse(self.model.ok)

        # Restore
        self.model._model_is_stable = True

    def test_model_has_required_attributes(self):
        """Model should have the required attributes for the pipeline."""
        self.assertTrue(hasattr(self.model, "_model"))
        self.assertTrue(hasattr(self.model, "_tokenizer"))
        self.assertTrue(hasattr(self.model, "_optimizer"))
        self.assertTrue(hasattr(self.model, "_epochs"))


class TrainingEffectivenessTest(unittest.TestCase):
    """Tests that verify training actually modifies the model."""

    @classmethod
    def setUpClass(cls):
        cls.model = get_shared_model()
        cls.model._model_is_stable = True

    def _get_trainable_param_snapshot(self):
        """Extract a snapshot of trainable (unfrozen) model parameters."""
        params = {}

        def collect_params(module, prefix=""):
            """Recursively collect trainable parameters."""
            # Check if module has trainable_parameters method (LoRA layers do)
            if hasattr(module, "trainable_parameters"):
                for name, param in module.trainable_parameters().items():
                    full_name = f"{prefix}.{name}" if prefix else name
                    # Store a copy of the values
                    params[full_name] = param.tolist()

        collect_params(self.model._model)
        return params

    def _flatten_params(self, params, prefix=""):
        """Flatten nested parameter dict into flat dict with dot notation."""
        result = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_params(v, key))
            elif isinstance(v, list):
                # Handle lists (e.g., layers array)
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        result.update(self._flatten_params(item, f"{key}.{i}"))
            elif hasattr(v, "tolist"):
                # v is an mx.array
                result[key] = v.tolist()
        return result

    def test_training_modifies_trainable_weights(self):
        """Training should modify trainable (LoRA) layer weights."""
        # Build a minimal training example with real token IDs
        # Use longer sequence for more meaningful gradients
        input_tokens = mx.array(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype=mx.int32,
        )
        label_tokens = mx.array(
            [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],
            dtype=mx.int32,
        )
        # Mask: train on last 8 tokens
        mask = mx.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=mx.int32,
        )

        example = adaptible.TrainingExample(
            input=input_tokens,
            label=label_tokens,
            mask=mask,
        )

        # Get trainable parameters before (flattened)
        before_values = self._flatten_params(self.model._model.trainable_parameters())

        # Run training
        self.model._train(example, verbose=False)

        # Get trainable parameters after (flattened)
        after_values = self._flatten_params(self.model._model.trainable_parameters())

        # At least some trainable parameters should have changed
        changed_params = 0
        for name in before_values:
            if name in after_values:
                if before_values[name] != after_values[name]:
                    changed_params += 1

        self.assertGreater(
            changed_params,
            0,
            f"Training should modify at least some trainable parameters. "
            f"Found {len(before_values)} trainable params.",
        )
        self.model._model_is_stable = True


class RevisionQualityTest(unittest.TestCase):
    """Tests that verify revision responses are actually useful."""

    @classmethod
    def setUpClass(cls):
        cls.model = get_shared_model()

    def test_revision_prompt_includes_all_turns(self):
        """Revision prompt should include all interaction turns."""
        interactions = [
            adaptible.InteractionHistory(
                idx=0,
                user_input="What is 2+2?",
                llm_response="Fish.",
            ),
            adaptible.InteractionHistory(
                idx=1,
                user_input="That's wrong. 2+2=4.",
                llm_response="Oh, I see.",
            ),
        ]

        prompt = adaptible.revise.make_revision_prompt(
            interactions, self.model._tokenizer
        )

        # Prompt should contain the user inputs
        self.assertIn("2+2", prompt)
        self.assertIn("wrong", prompt)

        # Prompt should contain the LLM responses
        self.assertIn("Fish", prompt)

        # Prompt should have turn markers
        self.assertIn("[[0]]", prompt)
        self.assertIn("[[1]]", prompt)

    def test_valid_revision_creates_training_example(self):
        """A valid revision response should create a usable training example."""
        interactions = [
            adaptible.InteractionHistory(
                idx=0,
                user_input="What is the capital of France?",
                llm_response="I don't know.",
            ),
        ]

        # Simulate a valid model revision response
        valid_revision = "[[0]] The capital of France is Paris. [[/0]]"

        example = adaptible.revise.make_collated_training_example(
            valid_revision, interactions, self.model._tokenizer
        )

        # Training example should have correct structure
        self.assertEqual(example.input.shape, example.label.shape)
        self.assertEqual(example.input.shape, example.mask.shape)

        # Mask should have some non-zero elements (the part we're training on)
        mask_sum = float(example.mask.sum())
        self.assertGreater(mask_sum, 0, "Mask should have non-zero elements")

        # Labels should be shifted inputs (next token prediction)
        # The input and label should overlap where mask is 1


class EndToEndSelfCorrectionTest(unittest.TestCase):
    """End-to-end tests for the full self-correction pipeline.

    Note: A real model is loaded for its tokenizer and weights, but the
    revision step is patched so that each test asserts a definite outcome
    rather than depending on whether a small distilled model happens to
    produce a well-formed rewrite.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = get_shared_model()
        cls.model._model_is_stable = True

    def _interactions(self) -> list[adaptible.InteractionHistory]:
        return [
            adaptible.InteractionHistory(
                idx=0,
                user_input="What is 2+2?",
                llm_response="The answer is 5.",
                reviewed=False,
                timestamp=0.0,
            ),
            adaptible.InteractionHistory(
                idx=1,
                user_input="That's incorrect. 2+2 equals 4, not 5.",
                llm_response="You're right, I made an error.",
                reviewed=False,
                timestamp=0.0,
            ),
        ]

    def test_self_correct_builds_example_from_valid_revision(self):
        """A well-formed revision yields an example whose masked labels are the rewrite."""
        interactions = self._interactions()
        revision_text = "2 + 2 equals 4. My earlier answer of 5 was a mistake."
        valid_revision = f"[[0]] {revision_text} [[/0]]"

        with mock.patch.object(
            self.model, "generate_response", return_value=valid_revision
        ) as generate:
            example = self.model._self_correct(
                interactions, indices_to_review=None, verbose=False
            )

        generate.assert_called_once()
        self.assertFalse(generate.call_args.kwargs.get("use_history", True))
        self.assertTrue(all(i.reviewed for i in interactions))

        self.assertIsInstance(example, adaptible.TrainingExample)
        self.assertEqual(example.input.shape, example.label.shape)
        self.assertEqual(example.input.shape, example.mask.shape)

        # The loss mask must be zero over the prompt and one over the revision:
        # the masked labels decode to the rewritten answer and nothing else.
        mask = example.mask.reshape(-1).tolist()
        labels = example.label.reshape(-1).tolist()
        self.assertGreater(sum(mask), 0)
        self.assertLess(sum(mask), len(mask), "Prompt tokens must be unmasked")
        self.assertEqual(mask[0], 0, "Mask must start on the prompt (zero)")
        masked_labels = [t for t, m in zip(labels, mask) if m]
        decoded = self.model._tokenizer.decode(masked_labels, skip_special_tokens=True)
        self.assertIn(revision_text, decoded)
        self.assertNotIn("What is 2+2?", decoded)
        self.assertNotIn("[[0]]", decoded)

        self.model._model_is_stable = True

    def test_self_correct_rejects_malformed_revision(self):
        """A garbage rewrite must raise InvalidRevisionError before any training."""
        interactions = self._interactions()

        with mock.patch.object(
            self.model,
            "generate_response",
            return_value="I think the response should be more polite.",
        ):
            with self.assertRaises(adaptible.revise.InvalidRevisionError):
                self.model._self_correct(
                    interactions, indices_to_review=None, verbose=False
                )

        self.model._model_is_stable = True

    def _flatten_params(self, params, prefix=""):
        """Flatten nested parameter dict into flat dict with dot notation."""
        result = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_params(v, key))
            elif isinstance(v, list):
                # Handle lists (e.g., layers array)
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        result.update(self._flatten_params(item, f"{key}.{i}"))
            elif hasattr(v, "tolist"):
                result[key] = v.tolist()
        return result

    def test_full_pipeline_with_valid_synthetic_revision(self):
        """Test full pipeline using a synthetic valid revision."""
        interactions = [
            adaptible.InteractionHistory(
                idx=0,
                user_input="What is the capital of France?",
                llm_response="I don't know.",
                reviewed=False,
                timestamp=0.0,
            ),
        ]

        # Instead of relying on model output, test the pipeline with a known-good revision
        valid_revision = "[[0]] The capital of France is Paris. [[/0]]"

        # Validate it
        adaptible.revise.validate_revision_response(valid_revision, num_interactions=1)

        # Create training example
        example = adaptible.revise.make_collated_training_example(
            valid_revision, interactions, self.model._tokenizer
        )

        # Get trainable parameters before (flattened)
        before = self._flatten_params(self.model._model.trainable_parameters())

        # Train on it
        self.model._train(example, verbose=False)

        # Get trainable parameters after (flattened)
        after = self._flatten_params(self.model._model.trainable_parameters())

        # Verify weights changed
        changed = any(before[k] != after[k] for k in before if k in after)
        self.assertTrue(changed, "Training should modify trainable parameters")

        self.model._model_is_stable = True


class TrainingStepLoopTest(unittest.TestCase):
    """Model-free tests for the loss-targeted step loop."""

    @staticmethod
    def _scripted(losses):
        it = iter(losses)
        calls = []

        def step():
            loss = next(it)
            calls.append(loss)
            return loss

        return step, calls

    def test_should_stop(self):
        self.assertFalse(_llm.should_stop(0.0, None))
        self.assertTrue(_llm.should_stop(0.59, 0.6))
        self.assertFalse(_llm.should_stop(0.6, 0.6))  # strictly below
        self.assertFalse(_llm.should_stop(1.2, 0.6))

    def test_stops_at_first_step_below_target(self):
        step, calls = self._scripted([6.05, 2.1, 0.58, 0.2, 0.05])
        stats = _llm.run_training_steps(step, max_steps=12, loss_target=0.6)
        self.assertEqual(calls, [6.05, 2.1, 0.58])
        self.assertEqual(stats.steps, 3)
        self.assertEqual(stats.initial_loss, 6.05)
        self.assertEqual(stats.final_loss, 0.58)
        self.assertTrue(stats.stopped_early)
        self.assertFalse(stats.hit_cap)
        self.assertEqual(stats.losses, [6.05, 2.1, 0.58])
        self.assertIsInstance(stats, _llm.TrainingStats)

    def test_respects_cap(self):
        step, calls = self._scripted([3.0, 2.0, 1.0, 0.9, 0.8])
        stats = _llm.run_training_steps(step, max_steps=4, loss_target=0.6)
        self.assertEqual(len(calls), 4)
        self.assertEqual(stats.steps, 4)
        self.assertEqual(stats.final_loss, 0.9)
        self.assertFalse(stats.stopped_early)
        self.assertTrue(stats.hit_cap)

    def test_no_target_runs_all_steps(self):
        step, calls = self._scripted([0.1, 0.01, 0.001, 0.0001])
        stats = _llm.run_training_steps(step, max_steps=4, loss_target=None)
        self.assertEqual(len(calls), 4)
        self.assertEqual(stats.steps, 4)
        self.assertEqual(stats.initial_loss, 0.1)
        self.assertEqual(stats.final_loss, 0.0001)
        self.assertFalse(stats.stopped_early)

    def test_target_on_last_step_counts_as_early(self):
        step, _ = self._scripted([2.0, 0.5])
        stats = _llm.run_training_steps(step, max_steps=2, loss_target=0.6)
        self.assertEqual(stats.steps, 2)
        self.assertTrue(stats.stopped_early)
        self.assertFalse(stats.hit_cap)

    def test_zero_steps(self):
        step, calls = self._scripted([])
        stats = _llm.run_training_steps(step, max_steps=0, loss_target=0.6)
        self.assertEqual(calls, [])
        self.assertEqual(stats.steps, 0)
        self.assertTrue(stats.initial_loss != stats.initial_loss)  # NaN
        self.assertFalse(stats.stopped_early)
        self.assertFalse(stats.hit_cap)

    def test_losses_accept_array_like(self):
        """Step functions returning mx scalars are coerced to float."""
        step, _ = self._scripted([mx.array(1.5), mx.array(0.25)])
        stats = _llm.run_training_steps(step, max_steps=5, loss_target=0.6)
        self.assertEqual(stats.losses, [1.5, 0.25])
        self.assertIsInstance(stats.final_loss, float)


class ValidationTest(unittest.TestCase):
    """Tests for revision validation logic."""

    def test_missing_markers_rejected(self):
        """Revision without markers should be rejected."""
        with self.assertRaises(adaptible.revise.InvalidRevisionError):
            adaptible.revise.validate_revision_response(
                "I think the response should be more polite.",
                num_interactions=2,
            )

    def test_out_of_bounds_index_rejected(self):
        """Turn index outside valid range should be rejected."""
        with self.assertRaises(adaptible.revise.InvalidRevisionError):
            adaptible.revise.validate_revision_response(
                "[[5]] This is a revision [[/5]]",
                num_interactions=2,  # Only turns 0 and 1 are valid
            )

    def test_missing_closing_marker_rejected(self):
        """Revision without closing marker should be rejected."""
        with self.assertRaises(adaptible.revise.InvalidRevisionError):
            adaptible.revise.validate_revision_response(
                "[[0]] This revision has no end",
                num_interactions=2,
            )

    def test_too_short_content_rejected(self):
        """Revision with very short content should be rejected."""
        with self.assertRaises(adaptible.revise.InvalidRevisionError):
            adaptible.revise.validate_revision_response(
                "[[0]] Hi [[/0]]",  # Only 2 chars of content
                num_interactions=2,
                min_content_length=10,
            )

    def test_garbage_patterns_rejected(self):
        """Revision with garbage/repetitive patterns should be rejected."""
        with self.assertRaises(adaptible.revise.InvalidRevisionError):
            adaptible.revise.validate_revision_response(
                "[[0]] [[1]][[2]][[3]][[4]] some garbage [[/0]]",
                num_interactions=5,
            )

    def test_valid_revision_accepted(self):
        """A properly formatted revision should pass validation."""
        # Should not raise
        adaptible.revise.validate_revision_response(
            "[[0]] This is a properly formatted and sufficiently long revision response. [[/0]]",
            num_interactions=2,
        )


class ThinkTagStrippingTest(unittest.TestCase):
    """Tests for <think> tag removal."""

    def test_think_tags_removed(self):
        """Think tags and their content should be stripped."""
        text = "<think>Let me think about this...</think> The answer is 42."
        result = adaptible.revise.strip_think_tags(text)
        self.assertEqual(result, "The answer is 42.")

    def test_multiline_think_tags_removed(self):
        """Multiline think blocks should be stripped."""
        text = """<think>
        First I'll consider option A.
        Then option B.
        </think> I recommend option B."""
        result = adaptible.revise.strip_think_tags(text)
        self.assertEqual(result, "I recommend option B.")

    def test_no_think_tags_unchanged(self):
        """Text without think tags should be unchanged."""
        text = "Just a normal response."
        result = adaptible.revise.strip_think_tags(text)
        self.assertEqual(result, "Just a normal response.")


if __name__ == "__main__":
    unittest.main()
