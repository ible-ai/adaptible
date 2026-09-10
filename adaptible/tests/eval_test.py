"""Model-free tests for the evaluation harness and meta-learning experiment.

A ``FakeModel`` stands in for ``StatefulLLM``. It answers trivia from a
script, produces a scripted revision when handed a revision prompt, and
"learns" an item by decoding the masked training target it is trained on.
The tokenizer is a one-character-per-token fake so training examples built
by ``make_collated_training_example`` can be decoded back to text.
"""

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import warnings
from typing import Callable

import adaptible
from adaptible._src import _llm

eval_mod = adaptible.eval
harness = adaptible._src.eval.harness
meta = adaptible._src.eval.meta
TrainingStats = _llm.TrainingStats

EvaluationConfig = eval_mod.EvaluationConfig
EvaluationHarness = eval_mod.EvaluationHarness
MetaLearningConfig = eval_mod.MetaLearningConfig
MetaLearningExperiment = eval_mod.MetaLearningExperiment
MetaLearningResult = eval_mod.MetaLearningResult
SeedTrajectory = eval_mod.SeedTrajectory
Checkpoint = eval_mod.Checkpoint
TriviaDataset = eval_mod.TriviaDataset
TriviaItem = eval_mod.TriviaItem
contains_key_terms = eval_mod.contains_key_terms
extract_revision = harness.extract_revision
generate_html_report = eval_mod.generate_html_report
ItemResult = harness.ItemResult
EvaluationResult = harness.EvaluationResult

EOS = "<eos>"
DONT_KNOW = "I do not know."
# FakeModel's default per-step loss script: crosses a 0.6 target on step 3.
FAKE_LOSSES = [6.05, 2.1, 0.58, 0.2, 0.05]
# Rehearsal loss script: already under the target from step 1, so if it drove
# the stop rule training would end before the correction landed.
FAKE_REHEARSAL_LOSSES = [0.5, 0.45, 0.41, 0.4, 0.4]
LONG_ANSWER_TOKENS = 2000  # a FakeModel(long_at_baseline=...) answer, in fake tokens
REPO_ROOT = pathlib.Path(__file__).parents[2]


def fake_think(item: TriviaItem) -> str:
    """The reasoning FakeModel(think=True) emits before ``</think>``."""
    return f"Thinking about {item.id}."


def fake_revision_think(item: TriviaItem) -> str:
    """The think block FakeModel(think=True) puts in front of its revision."""
    return f"Revising {item.id}: it should say {item.correct_answer}."


def fake_rationale(item: TriviaItem) -> str:
    """The think block FakeModel(think=True) generates for the rationale prompt."""
    return f"Given the label, {item.id} must be {item.correct_answer}."


def fake_bare_rationale(item: TriviaItem) -> str:
    """FakeModel(rationale="bare"): reasoning that never emits ``</think>``."""
    return (
        f"Thinking about {item.id}. The label says {item.correct_answer}. "
        f"So it must be {item.correct_answer}. Still going on and on"
    )


RATIONALE_MARKER = "The correct answer is:"


class FakeTokenizer:
    """One token per character; token 0 is reserved for padding.

    Args:
        think: Mimic DeepSeek-R1-Distill, whose generation prompt ends with an
            open ``<think>\n``.
    """

    special_tokens_map = {"eos_token": EOS}

    def __init__(self, think: bool = False):
        self.think = think

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) + 1 for c in text]

    def decode(self, tokens) -> str:
        return "".join(chr(t - 1) for t in tokens if t > 0)

    def apply_chat_template(
        self,
        conversation=None,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        **kwargs,
    ) -> str:
        del tokenize, continue_final_message, kwargs
        text = "".join(
            f"<{m['role']}>{m['content']}</{m['role']}>" for m in conversation
        )
        if add_generation_prompt:
            text += "<assistant>"
            if self.think:
                text += "<think>\n"
        return text


class FakeModel:
    """Scripted stand-in for StatefulLLM.

    Args:
        dataset: Items the model knows the answers to (once trained).
        learns_after: Number of training calls that must happen before a
            training call actually teaches the item. ``0`` learns immediately;
            a value >= the number of training calls means it never learns.
        revision: "valid" -> a well-formed revision containing the answer;
            "invalid" -> text with no [[X]] markers; a callable
            ``(item) -> str`` for custom revisions; or a dict
            ``item_id -> revision text`` scripting each item individually
            (items missing from the dict get the "valid" revision).
        known_at_baseline: Item ids answered correctly before any training.
        think: Give the tokenizer a generation prompt ending in ``<think>\n``
            and make every trivia answer carry ``fake_think(item)`` before
            ``</think>``, the way DeepSeek-R1-Distill does.
        long_at_baseline: Item ids whose trivia answer is padded out to
            ``LONG_ANSWER_TOKENS`` tokens, the way a small model rambles to the
            generation cap. Used to exercise ``rehearsal_max_tokens``.
        train_losses: Per-step correction loss script every training call
            replays (the last value repeats past the end). The call stops
            through ``_llm.run_training_steps`` exactly as the real model does,
            so the steps it reports depend on ``loss_target``/``max_steps``.
        rehearsal_losses: Per-step mean rehearsal loss script for
            ``train_on_examples`` (last value repeats). Never consulted by the
            stop rule, exactly as in ``_llm.run_joint_training_steps``.
        rationale: What the model answers to the ground-truth rationale prompt
            (``harness.make_rationale_prompt``): "valid" -> a think block
            ``fake_rationale(item)`` then the answer (only with
            ``think=True``; without a think template the harness never asks);
            "bare" -> ``fake_bare_rationale(item)``, reasoning that never
            closes the think tag (the whole output is the rationale);
            "missing" -> an empty output (no rationale: the item is skipped);
            or a callable ``(item) -> str``.
        rehearsal_margin: Hinge margin the fake applies to its scripted mean
            rehearsal loss, exactly as ``_llm.active_rehearsal`` does with the
            first step's loss as the anchor: on a step where the scripted
            loss is more than the margin above the first step's, every
            rehearsal example counts as active. The call's ``rehearsal_margin``
            argument overrides it.
        learns_after_steps: Total optimizer *steps* (over every training
            call) the model must have received before a call teaches its
            target; ``None`` (default) leaves only ``learns_after`` in force.
            Exercises the verify-after-target loop, which keeps training the
            same example a few steps at a time until the answer comes out.
        regresses: Item ids whose answer is wrong once any training call has
            happened, however they were answered at baseline. Scripts
            interference: an untrained item that a neighbour's training broke.
    """

    def __init__(
        self,
        dataset: TriviaDataset,
        learns_after: int = 0,
        revision: str | Callable[[TriviaItem], str] | dict[str, str] = "valid",
        known_at_baseline: set[str] | None = None,
        think: bool = False,
        long_at_baseline: set[str] | None = None,
        train_losses: list[float] | None = None,
        rehearsal_losses: list[float] | None = None,
        rationale: str | Callable[[TriviaItem], str] = "valid",
        learns_after_steps: int | None = None,
        regresses: set[str] | None = None,
    ):
        self._tokenizer = FakeTokenizer(think=think)
        self._rationale = rationale
        self._learns_after_steps = learns_after_steps
        self._regresses = set(regresses or set())
        # Optimizer steps over every training call so far.
        self.total_steps = 0
        self._train_losses = list(train_losses or FAKE_LOSSES)
        self._rehearsal_losses = list(rehearsal_losses or FAKE_REHEARSAL_LOSSES)
        self._think = think
        self._long_at_baseline = set(long_at_baseline or set())
        self._max_tokens = 4096
        self._model_is_stable = True
        self._by_question = {item.question: item for item in dataset}
        self._by_id = {item.id: item for item in dataset}
        self._learns_after = learns_after
        self._revision = revision
        self.learned: set[str] = set(known_at_baseline or set())
        # Training calls of either kind; one per trained item.
        self.train_calls = 0
        # Masked target of every call's correction, in order.
        self.trained_targets: list[str] = []
        # Stop-masked (answer-only) target of every call's correction, in
        # order; None when the example carried no stop mask.
        self.trained_stop_targets: list[str | None] = []
        # Every example handed to a training call, in order (a joint call
        # contributes its correction and then each rehearsal example): the
        # rows of that example as (decoded masked target, decoded full sequence).
        self.trained_batches: list[list[tuple[str, str]]] = []
        self.batch_shapes: list[tuple[int, int]] = []
        self.revision_prompts: list[str] = []
        self.rationale_prompts: list[str] = []
        self.question_prompts: list[str] = []
        # (iterations, loss_target, max_steps) of every train_on_example call.
        self.train_kwargs: list[tuple[int, float | None, int | None]] = []
        # Keyword arguments of every train_on_examples call, plus the number of
        # rehearsal examples and their decoded masked targets.
        self.joint_calls: list[dict] = []
        self.training_stats: list[TrainingStats] = []

    def generate_response(
        self, prompt: str, use_history: bool = False, max_tokens: int | None = None
    ) -> str:
        del use_history, max_tokens
        if "<PAST_DIALOG>" in prompt:
            self.revision_prompts.append(prompt)
            # "default" dialogs go through FakeTokenizer.apply_chat_template;
            # "fewshot" dialogs are plain "User: ..." lines.
            item = next(
                it
                for q, it in self._by_question.items()
                if f"<user>{q}</user>" in prompt or f"User: {q}\n" in prompt
            )
            if isinstance(self._revision, dict) and item.id in self._revision:
                return self._revision[item.id]
            if callable(self._revision):
                return self._revision(item)
            if self._revision == "invalid":
                return "Here is a revision with no markers at all."
            revision = f"[[0]] The answer is {item.correct_answer}. [[/0]]"
            if self._think:
                # The revision generation reasons before it answers, like
                # every other DeepSeek-R1-Distill response.
                return f"{fake_revision_think(item)}\n</think>\n\n{revision}"
            return revision
        if RATIONALE_MARKER in prompt:
            self.rationale_prompts.append(prompt)
            item = next(
                it for q, it in self._by_question.items() if prompt.startswith(q)
            )
            if callable(self._rationale):
                return self._rationale(item)
            if self._rationale == "missing":
                return ""
            if self._rationale == "bare":
                return fake_bare_rationale(item)
            answer = f"So the answer is {item.correct_answer}."
            if not self._think:
                return answer
            return f"{fake_rationale(item)}\n</think>\n\n{answer}"
        self.question_prompts.append(prompt)
        item = self._by_question[prompt]
        knows = item.id in self.learned
        if item.id in self._regresses and self.train_calls > 0:
            knows = False
        answer = f"It is {item.correct_answer}." if knows else DONT_KNOW
        if item.id in self._long_at_baseline:
            answer = answer.ljust(LONG_ANSWER_TOKENS, ".")
        if self._think:
            return f"{fake_think(item)}\n</think>\n\n{answer}"
        return answer

    def train_on_example(
        self,
        example,
        iterations: int = 25,
        verbose: bool = False,
        save_checkpoint: bool = False,
        loss_target: float | None = None,
        max_steps: int | None = None,
    ) -> TrainingStats:
        del verbose, save_checkpoint
        self.train_kwargs.append((iterations, loss_target, max_steps))
        step = self._scripted_step(self._train_losses)
        stats = _llm.run_training_steps(
            step, iterations if max_steps is None else max_steps, loss_target
        )
        return self._finish_call(stats, example)

    def train_on_examples(
        self,
        correction,
        rehearsal,
        *,
        loss_target: float | None,
        max_steps: int,
        rehearsal_weight: float = 1.0,
        rehearsal_margin: float = 0.05,
        verbose: bool = False,
        save_checkpoint: bool = False,
    ) -> TrainingStats:
        del verbose, save_checkpoint
        rehearsal = list(rehearsal)
        rehearsal_rows = [self._decode_rows(ex) for ex in rehearsal]
        self.joint_calls.append(
            {
                "loss_target": loss_target,
                "max_steps": max_steps,
                "rehearsal_weight": rehearsal_weight,
                "rehearsal_margin": rehearsal_margin,
                "rehearsal_k": len(rehearsal),
                "rehearsal_targets": [rows[0][0] for rows in rehearsal_rows],
            }
        )
        step_c = self._scripted_step(self._train_losses)
        step_r = self._scripted_step(self._rehearsal_losses)
        initial: list[float] = []

        def step() -> tuple[float, float | None, int]:
            loss_c = step_c()
            if not rehearsal:
                return loss_c, None, 0
            loss_r = step_r()
            if not initial:
                initial.append(loss_r)
            # Every rehearsal example shares the scripted loss, so the hinge
            # fires for all k of them or none.
            active = _llm.active_rehearsal(
                [loss_r] * len(rehearsal), initial * len(rehearsal), rehearsal_margin
            )
            return loss_c, loss_r, sum(active)

        stats = _llm.run_joint_training_steps(
            step, max_steps, loss_target, len(rehearsal)
        )
        stats = self._finish_call(stats, correction)
        for ex, rows in zip(rehearsal, rehearsal_rows):
            self.batch_shapes.append(tuple(ex.mask.shape))
            self.trained_batches.append(rows)
        return stats

    @staticmethod
    def _scripted_step(losses: list[float]) -> Callable[[], float]:
        script = iter(losses)
        last = losses[-1]

        def step() -> float:
            nonlocal last
            last = next(script, last)
            return last

        return step

    def _decode_rows(self, example) -> list[tuple[str, str]]:
        rows = []
        for inputs, labels, mask in zip(
            example.input.tolist(), example.label.tolist(), example.mask.tolist()
        ):
            masked = self._tokenizer.decode(t for t, m in zip(labels, mask) if m)
            full = self._tokenizer.decode([inputs[0]] + labels)
            rows.append((masked, full))
        return rows

    def _decode_stop_target(self, example) -> str | None:
        stop_mask = getattr(example, "stop_mask", None)
        if stop_mask is None:
            return None
        labels = example.label.tolist()[0]
        return self._tokenizer.decode(
            t for t, m in zip(labels, stop_mask.tolist()[0]) if m
        )

    def _finish_call(self, stats: TrainingStats, example) -> TrainingStats:
        """Record the correction example of a call and "learn" its target."""
        self.training_stats.append(stats)
        self.batch_shapes.append(tuple(example.mask.shape))
        rows = self._decode_rows(example)
        self.trained_batches.append(rows)
        target = rows[0][0]
        self.trained_targets.append(target)
        self.trained_stop_targets.append(self._decode_stop_target(example))
        self.train_calls += 1
        self.total_steps += stats.steps
        if self.train_calls <= self._learns_after:
            return stats
        if (
            self._learns_after_steps is not None
            and self.total_steps < self._learns_after_steps
        ):
            return stats
        for item in self._by_id.values():
            if item.correct_answer.lower() in target.lower():
                self.learned.add(item.id)
        return stats


def make_dataset(n: int) -> TriviaDataset:
    items = [
        TriviaItem(
            id=f"q{i:02d}",
            category="test",
            question=f"What is fact number {i}?",
            correct_answer=f"Answer{i}",
            key_terms=[f"answer{i}"],
        )
        for i in range(n)
    ]
    return TriviaDataset(name="fake", version="1", items=items)


def training_groups(model: FakeModel, ks: list[int]) -> list[list[tuple[str, str]]]:
    """Split the model's single-row training examples into per-item groups.

    With joint rehearsal every item hands the model ``1 + k`` single-row
    examples in one call: the correction, then each rehearsal example.
    Returns, per item, the list of ``(masked target, full sequence)`` rows in
    that order.
    """
    groups, pos = [], 0
    for k in ks:
        examples = model.trained_batches[pos : pos + 1 + k]
        assert all(len(rows) == 1 for rows in examples), "every example is one row"
        groups.append([rows[0] for rows in examples])
        pos += 1 + k
    assert pos == len(model.trained_batches), (pos, len(model.trained_batches))
    return groups


class _TempDbTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.db = adaptible.Database(self.tmp_path / "test.db")

    def tearDown(self):
        self._tmp.cleanup()

    def _config_json_for(self, experiment_id: int) -> dict:
        return json.loads(self.db.get_experiment(experiment_id).config_json)

    def _latest_experiment_id(self) -> int:
        return max(e.id for e in self.db.get_experiments())


class JudgeTest(unittest.TestCase):
    """contains_key_terms is NFKC + casefold on both sides."""

    def test_subscript_matches_ascii_digit(self):
        self.assertTrue(contains_key_terms("Water is H₂O.", ["H2O"]))
        self.assertTrue(contains_key_terms("Water is H2O.", ["H₂O"]))

    def test_casefold_handles_sharp_s(self):
        self.assertTrue(contains_key_terms("The Straße is long", ["STRASSE"]))

    def test_fullwidth_matches(self):
        self.assertTrue(contains_key_terms("Ｔｏｋｙｏ is the capital", ["tokyo"]))

    def test_no_match_and_empty(self):
        self.assertFalse(contains_key_terms("Paris", ["london"]))
        self.assertFalse(contains_key_terms("", ["x"]))
        self.assertFalse(contains_key_terms("anything", []))


class ExtractRevisionTest(unittest.TestCase):
    """extract_revision returns the span make_collated_training_example trains on."""

    def test_basic(self):
        self.assertEqual(extract_revision("[[0]] Canberra. [[/0]]"), "Canberra.")

    def test_lowest_index_wins_and_uses_last_markers(self):
        text = "[[1]] junk [[/1]] [[0]] first [[0]] second [[/0]] tail [[/0]]"
        self.assertEqual(extract_revision(text), "second [[/0]] tail")

    def test_missing_closing_marker_runs_to_end(self):
        self.assertEqual(extract_revision("[[0]] open ended"), "open ended")

    def test_no_marker_raises(self):
        with self.assertRaises(ValueError):
            extract_revision("no markers here")


class TrainingSourceTest(_TempDbTest):
    def test_invalid_training_source_rejected(self):
        with self.assertRaises(ValueError):
            EvaluationConfig(training_source="labels")
        with self.assertRaises(ValueError):
            MetaLearningConfig(training_source="labels")
        with self.assertRaises(ValueError):
            MetaLearningConfig(repeats=0)

    def test_ground_truth_trains_on_label(self):
        dataset = make_dataset(5)
        model = FakeModel(dataset)
        config = EvaluationConfig(name="gt", train_ratio=0.6, shuffle=False)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )

        self.assertEqual(len(model.trained_targets), 3)
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"{item.correct_answer}{EOS}")
        self.assertEqual(model.revision_prompts, [])
        self.assertTrue(all(i.revision_text is None for i in result.items))
        self.assertEqual(result.revision_invalid_count, 0)
        self.assertEqual(result.to_dict()["config"]["training_source"], "ground_truth")
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["training_source"],
            "ground_truth",
        )
        # The trained items were learned, the holdout ones were not.
        self.assertEqual(result.train_post_accuracy, 1.0)
        self.assertEqual(result.holdout_accuracy, 0.0)

    def test_self_generated_trains_on_revision(self):
        dataset = make_dataset(5)
        model = FakeModel(dataset)
        config = EvaluationConfig(
            name="sg", train_ratio=0.6, training_source="self_generated"
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )

        self.assertEqual(len(model.revision_prompts), 3)
        self.assertEqual(len(model.trained_targets), 3)
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"The answer is {item.correct_answer}.{EOS}")
        trained = result.train_items
        self.assertEqual(len(trained), 3)
        for item in trained:
            self.assertIn("[[0]] The answer is", item.revision_text)
            self.assertFalse(item.revision_invalid)
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["training_source"],
            "self_generated",
        )
        item_dicts = result.to_dict()["items"]
        self.assertTrue(any(d["revision_text"] for d in item_dicts))

    def test_invalid_revision_prompt_rejected(self):
        with self.assertRaises(ValueError):
            EvaluationConfig(revision_prompt="zero_shot")
        with self.assertRaises(ValueError):
            MetaLearningConfig(revision_prompt="zero_shot")

    def test_default_revision_prompt_uses_chat_template(self):
        dataset = make_dataset(5)
        model = FakeModel(dataset)
        config = EvaluationConfig(
            name="sg-default", train_ratio=0.6, training_source="self_generated"
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        self.assertEqual(len(model.revision_prompts), 3)
        for item, prompt in zip(dataset.items[:3], model.revision_prompts):
            self.assertTrue(prompt.startswith(adaptible.revise.REWRITE_INSTRUCTIONS))
            self.assertIn(f"[[0]]<user>{item.question}</user><assistant>", prompt)
            self.assertNotIn(f"User: {item.question}", prompt)
            self.assertNotIn("Example 1", prompt)
        self.assertEqual(result.to_dict()["config"]["revision_prompt"], "default")
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["revision_prompt"],
            "default",
        )

    def test_fewshot_revision_prompt_is_plain_dialog(self):
        dataset = make_dataset(5)
        model = FakeModel(dataset)
        config = EvaluationConfig(
            name="sg-fewshot",
            train_ratio=0.6,
            training_source="self_generated",
            revision_prompt="fewshot",
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        self.assertEqual(len(model.revision_prompts), 3)
        for item, prompt in zip(dataset.items[:3], model.revision_prompts):
            self.assertTrue(
                prompt.startswith(adaptible.revise.REWRITE_INSTRUCTIONS_FEWSHOT)
            )
            self.assertIn("Example 1", prompt)
            self.assertIn("Example 2", prompt)
            self.assertIn(
                f"<PAST_DIALOG>\n[[0]] User: {item.question}\nAssistant: {DONT_KNOW}\n"
                "</PAST_DIALOG>",
                prompt,
            )
            # No chat-template markers from FakeTokenizer.apply_chat_template.
            self.assertNotIn("<user>", prompt)
            self.assertNotIn("<assistant>", prompt)
        # Training still happens on the revision, exactly as with "default".
        self.assertEqual(len(model.trained_targets), 3)
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"The answer is {item.correct_answer}.{EOS}")
        self.assertEqual(result.to_dict()["config"]["revision_prompt"], "fewshot")
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["revision_prompt"],
            "fewshot",
        )
        path = generate_html_report(result, self.tmp_path / "fewshot.html")
        self.assertIn("<code>fewshot</code>", pathlib.Path(path).read_text())

    def test_think_mode_rationale_is_default_and_trains_on_rationale(self):
        dataset = make_dataset(4)
        for source in ("ground_truth", "self_generated"):
            model = FakeModel(dataset, think=True)
            config = EvaluationConfig(name=f"tm-{source}", training_source=source)
            self.assertEqual(config.think_mode, "rationale")
            self.assertIs(config.close_think, True)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=False
            )
            self.assertEqual(len(model.trained_batches), 3)
            if source == "ground_truth":
                # One extra generation per trained item asks the model to
                # reason its way to the label; holdout items never get one.
                self.assertEqual(len(model.rationale_prompts), 3)
                for item, prompt in zip(dataset.items[:3], model.rationale_prompts):
                    self.assertEqual(
                        prompt, harness.make_rationale_prompt(item.question, item.correct_answer)
                    )
                    self.assertTrue(prompt.startswith(item.question))
                    self.assertIn(f"The correct answer is: {item.correct_answer}", prompt)
                body = "{answer}"
                rationale_for = fake_rationale
            else:
                # The revision's own think block is the rationale; no extra
                # generation happens.
                self.assertEqual(model.rationale_prompts, [])
                self.assertEqual(len(model.revision_prompts), 3)
                body = "The answer is {answer}."
                rationale_for = fake_revision_think
            by_id = {r.item_id: r for r in result.items}
            for item, (masked, full), stop in zip(
                dataset.items[:3],
                (b[0] for b in model.trained_batches),
                model.trained_stop_targets,
            ):
                answer = f"{body.format(answer=item.correct_answer)}{EOS}"
                target = f"{rationale_for(item)}\n</think>\n\n{answer}"
                # The rationale and the answer are both in the loss...
                self.assertEqual(masked, target)
                self.assertEqual(
                    full, f"<user>{item.question}</user><assistant><think>\n{target}"
                )
                # ...but the stop mask covers exactly the answer plus eos.
                self.assertEqual(stop, answer)
                # The rationale is recorded per item.
                self.assertEqual(by_id[item.id].rationale_text, rationale_for(item))
                self.assertFalse(by_id[item.id].rationale_missing)
            for item in dataset.items[3:]:
                self.assertIsNone(by_id[item.id].rationale_text)
            self.assertEqual(result.rationale_missing_count, 0)
            self.assertEqual(result.train_post_accuracy, 1.0)
            if source == "self_generated":
                # The raw revision (think block included) is what is kept.
                for item in result.train_items:
                    self.assertTrue(item.revision_text.startswith("Revising "))
                    self.assertIn("</think>", item.revision_text)
                    self.assertEqual(
                        item.revision_answer, f"The answer is {item.correct_answer}."
                    )
            d = result.to_dict()
            self.assertEqual(d["config"]["think_mode"], "rationale")
            self.assertIs(d["config"]["close_think"], True)
            self.assertEqual(d["metrics"]["rationale_missing_count"], 0)
            trained = [i for i in d["items"] if i["was_trained"]]
            self.assertTrue(all(i["rationale_text"] for i in trained))
            self.assertTrue(all(i["rationale_missing"] is False for i in trained))
            cfg = self._config_json_for(self._latest_experiment_id())
            self.assertEqual(cfg["think_mode"], "rationale")
            self.assertEqual(cfg["model_kwargs"], {})
            text = pathlib.Path(
                generate_html_report(result, self.tmp_path / f"{source}-tm.html")
            ).read_text()
            self.assertIn("Think mode: <code>rationale</code>", text)
            self.assertIn("rationales missing: <code>0</code>", text)
            self.assertIn("Rehearsal k: <code>0</code>", text)
            # The report card shows the rationale that went into the target.
            for item in dataset.items[:3]:
                self.assertIn(
                    f"<strong>Rationale (in target):</strong> {rationale_for(item)}", text
                )

    def test_think_mode_rationale_missing_skips_item_and_is_counted(self):
        dataset = make_dataset(4)
        # ground_truth: the rationale generation comes back empty. Nothing to
        # put in the think block, so the item is skipped: the empty-think
        # target it used to fall back to is the one that collapses the model.
        model = FakeModel(dataset, think=True, rationale="missing")
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="gt-missing"), verbose=False
        )
        self.assertEqual(len(model.rationale_prompts), 3)
        self.assertEqual(model.train_calls, 0)
        self.assertEqual(model.train_kwargs, [])
        self.assertEqual(model.joint_calls, [])
        self.assertEqual(model.trained_targets, [])
        by_id = {r.item_id: r for r in result.items}
        for item in dataset.items[:3]:
            self.assertTrue(by_id[item.id].rationale_missing)
            self.assertFalse(by_id[item.id].was_trained)
            self.assertIsNone(by_id[item.id].rationale_text)
            self.assertEqual(by_id[item.id].train_steps, 0)
        self.assertFalse(by_id["q03"].rationale_missing)
        # Skipped items are neither trained nor holdout, like invalid revisions.
        self.assertEqual(result.train_items, [])
        self.assertEqual([r.item_id for r in result.holdout_items], ["q03"])
        self.assertEqual(
            [r.item_id for r in result.rationale_missing_items], ["q00", "q01", "q02"]
        )
        self.assertEqual(result.rationale_missing_count, 3)
        self.assertEqual(result.to_dict()["metrics"]["rationale_missing_count"], 3)
        self.assertEqual(result.to_dict()["metrics"]["train_count"], 0)
        self.assertEqual(result.to_dict()["metrics"]["holdout_count"], 1)
        self.assertEqual(result.training_summary_text(), "Training: no items trained")
        self.assertEqual(self.db.get_training_events_for_experiment(1), [])
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "missing.html")
        ).read_text()
        self.assertIn("rationales missing: <code>3</code>", text)
        self.assertIn("<strong>Rationale:</strong> MISSING (item skipped, not trained)", text)

        # self_generated: the revision carries no </think> at all. The model
        # generates inside the open think block, so the whole output is its
        # reasoning and becomes the rationale; nothing is skipped.
        model = FakeModel(
            dataset,
            think=True,
            revision=lambda item: f"[[0]] The answer is {item.correct_answer}. [[/0]]",
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(name="sg-bare", training_source="self_generated"),
            verbose=False,
        )
        self.assertEqual(model.rationale_prompts, [])
        for item, target in zip(dataset.items[:3], model.trained_targets):
            revision = f"[[0]] The answer is {item.correct_answer}. [[/0]]"
            self.assertEqual(
                target, f"{revision}\n</think>\n\nThe answer is {item.correct_answer}.{EOS}"
            )
        self.assertEqual(result.rationale_missing_count, 0)
        self.assertEqual(len(result.train_items), 3)

        # A partial miss skips only that item and is counted per item; the
        # summary line carries the count.
        model = FakeModel(
            dataset,
            think=True,
            rationale=lambda item: (
                f"{fake_rationale(item)}\n</think>\n\nSo {item.correct_answer}."
                if item.id != "q01"
                else "   "
            ),
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, EvaluationConfig(name="partial"), verbose=True
            )
        self.assertEqual(result.rationale_missing_count, 1)
        self.assertEqual(model.train_calls, 2)
        self.assertEqual([r.item_id for r in result.train_items], ["q00", "q02"])
        self.assertEqual(
            [r.rationale_missing for r in result.items], [False, True, False, False]
        )
        # Train metrics are over the two trained items only.
        self.assertEqual(result.train_improvement_rate, 1.0)
        self.assertAlmostEqual(result.mean_train_steps, 3.0)
        out = buf.getvalue()
        self.assertIn("Skipped: no rationale", out)
        self.assertIn(
            "mean final answer loss 0.58 (train loss 0.58); 0 items hit the cap; "
            "1 rationales missing",
            out,
        )
        self.assertIn("1 skipped for no rationale, 0 truncated", out)

    def test_think_mode_rationale_takes_whole_output_without_close_tag(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, think=True, rationale="bare")
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="bare"), verbose=False
        )
        self.assertEqual(len(model.rationale_prompts), 3)
        by_id = {r.item_id: r for r in result.items}
        for item, target in zip(dataset.items[:3], model.trained_targets):
            rationale = fake_bare_rationale(item)
            self.assertEqual(
                target, f"{rationale}\n</think>\n\n{item.correct_answer}{EOS}"
            )
            r = by_id[item.id]
            self.assertEqual(r.rationale_text, rationale)
            self.assertEqual(r.rationale_tokens, len(rationale))  # char-per-token
            self.assertFalse(r.rationale_truncated)
            self.assertFalse(r.rationale_missing)
            self.assertTrue(r.was_trained)
        self.assertEqual(result.rationale_missing_count, 0)
        self.assertEqual(result.rationale_truncated_count, 0)

    def test_rationale_max_tokens_truncates_at_sentence_boundary(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, think=True, rationale="bare")
        # fake_bare_rationale(q00) is "Thinking about q00. The label says
        # Answer0. So it must be Answer0. Still going on and on": 60 chars
        # (= fake tokens) keeps "Thinking about q00. The label says Answer0.
        # So it must be Answer0." minus a partial last sentence.
        config = EvaluationConfig(name="trunc", rationale_max_tokens=60)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=True
            )
        by_id = {r.item_id: r for r in result.items}
        for item, target in zip(dataset.items[:3], model.trained_targets):
            full = fake_bare_rationale(item)
            expected = full[:60]
            expected = expected[: expected.rfind(". ") + 1]
            self.assertTrue(expected.endswith("."))
            self.assertLess(len(expected), 60)
            self.assertEqual(target, f"{expected}\n</think>\n\n{item.correct_answer}{EOS}")
            r = by_id[item.id]
            self.assertEqual(r.rationale_text, expected)
            self.assertEqual(r.rationale_tokens, len(expected))
            self.assertTrue(r.rationale_truncated)
        self.assertEqual(result.rationale_truncated_count, 3)
        d = result.to_dict()
        self.assertEqual(d["config"]["rationale_max_tokens"], 60)
        self.assertEqual(d["metrics"]["rationale_truncated_count"], 3)
        self.assertTrue(all(i["rationale_truncated"] for i in d["items"][:3]))
        self.assertEqual(self._config_json_for(1)["rationale_max_tokens"], 60)
        out = buf.getvalue()
        self.assertIn("Think mode: rationale (rationale max tokens: 60)", out)
        self.assertRegex(out, r"Rationale truncated to \d+ tokens \(cap 60\)")
        self.assertIn("0 skipped for no rationale, 3 truncated", out)
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "trunc.html")
        ).read_text()
        self.assertIn("rationale max tokens: <code>60</code>", text)
        self.assertIn("truncated: <code>3</code>", text)
        self.assertIn("tokens, truncated)</em>", text)
        # An untruncated run under the default cap records the default.
        result = EvaluationHarness(model=FakeModel(dataset, think=True), db=self.db).run(
            dataset, EvaluationConfig(name="default-cap"), verbose=False
        )
        self.assertEqual(result.config.rationale_max_tokens, 512)
        self.assertEqual(result.to_dict()["config"]["rationale_max_tokens"], 512)
        for bad in (0, -5, 2.5):
            with self.assertRaises(ValueError):
                EvaluationConfig(rationale_max_tokens=bad)
            with self.assertRaises(ValueError):
                MetaLearningConfig(rationale_max_tokens=bad)

    def test_meta_skips_items_without_rationale(self):
        dataset = make_dataset(10)
        model = FakeModel(
            dataset,
            think=True,
            rationale=lambda item: (
                "" if item.id in {"q02", "q05"}
                else f"{fake_rationale(item)}\n</think>\n\nSo {item.correct_answer}."
            ),
        )
        config = MetaLearningConfig(
            name="meta-missing", seeds=[1], checkpoint_interval=4, train_ratio=0.8,
            rationale_max_tokens=100,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
                dataset, config, verbose=True
            )
        traj = result.trajectories[1]
        trained_ids = traj_train_ids(dataset, seed=1)
        skipped = sum(1 for i in ("q02", "q05") if i in trained_ids)
        self.assertGreater(skipped, 0)
        self.assertEqual(traj.rationale_missing_count, skipped)
        self.assertEqual(model.train_calls, len(trained_ids) - skipped)
        self.assertEqual(len(traj.train_steps), len(trained_ids) - skipped)
        self.assertEqual(traj.checkpoints[-1].step, len(trained_ids) - skipped)
        self.assertIn("Skipped q0", buf.getvalue())
        self.assertIn(f"{skipped} rationales missing", buf.getvalue())
        self.assertEqual(self._config_json_for(traj.experiment_id)["rationale_max_tokens"], 100)
        self.assertEqual(
            MetaLearningConfig.from_dict(config.to_dict()).rationale_max_tokens, 100
        )
        path = self.tmp_path / "meta-missing.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(loaded.trajectories[1].rationale_missing_count, skipped)
        self.assertEqual(loaded.config.rationale_max_tokens, 100)

    def test_think_mode_baseline_keeps_reasoning_unmasked(self):
        dataset = make_dataset(4)
        for source in ("ground_truth", "self_generated"):
            model = FakeModel(dataset, think=True)
            config = EvaluationConfig(
                name=f"tm-{source}", training_source=source, think_mode="baseline"
            )
            self.assertIs(config.close_think, True)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=False
            )
            self.assertEqual(len(model.trained_batches), 3)
            self.assertEqual(model.rationale_prompts, [])
            body = "{answer}" if source == "ground_truth" else "The answer is {answer}."
            for item, (masked, full), stop in zip(
                dataset.items[:3],
                (b[0] for b in model.trained_batches),
                model.trained_stop_targets,
            ):
                target = f"{body.format(answer=item.correct_answer)}{EOS}"
                # Only the corrected answer is in the loss...
                self.assertEqual(masked, target)
                self.assertEqual(stop, target)
                # ...and the model's own reasoning sits in the unmasked prefix.
                self.assertEqual(
                    full,
                    f"<user>{item.question}</user><assistant><think>\n"
                    f"{fake_think(item)}\n</think>\n\n{target}",
                )
            self.assertEqual(result.train_post_accuracy, 1.0)
            self.assertTrue(all(r.rationale_text is None for r in result.items))
            self.assertEqual(result.rationale_missing_count, 0)
            d = result.to_dict()["config"]
            self.assertEqual(d["think_mode"], "baseline")
            self.assertIs(d["close_think"], True)
            self.assertEqual(d["rehearsal_k"], 0)
            cfg = self._config_json_for(self._latest_experiment_id())
            self.assertEqual(cfg["think_mode"], "baseline")
            self.assertEqual(cfg["rehearsal_k"], 0)
            text = pathlib.Path(
                generate_html_report(result, self.tmp_path / f"{source}-tm.html")
            ).read_text()
            self.assertIn("Think mode: <code>baseline</code>", text)
            self.assertIn("Rehearsal k: <code>0</code>", text)

    def test_think_mode_empty_closes_open_think_block(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, think=True)
        config = EvaluationConfig(name="empty", think_mode="empty")
        EvaluationHarness(model=model, db=self.db).run(dataset, config, verbose=False)
        self.assertEqual(model.rationale_prompts, [])
        for item, target, stop in zip(
            dataset.items[:3], model.trained_targets, model.trained_stop_targets
        ):
            self.assertEqual(target, f"</think>\n\n{item.correct_answer}{EOS}")
            # The stop mask never covers the think close, only the answer.
            self.assertEqual(stop, f"{item.correct_answer}{EOS}")
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["think_mode"], "empty"
        )

    def test_think_mode_none_and_deprecated_close_think_false(self):
        dataset = make_dataset(4)
        for kwargs in ({"think_mode": "none"}, {"close_think": False}):
            model = FakeModel(dataset, think=True)
            config = EvaluationConfig(name="none", **kwargs)
            self.assertEqual(config.think_mode, "none")
            self.assertIs(config.close_think, False)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=False
            )
            for item, (masked, full) in zip(
                dataset.items[:3], (b[0] for b in model.trained_batches)
            ):
                self.assertEqual(masked, f"{item.correct_answer}{EOS}")
                self.assertTrue(full.endswith(f"<think>\n{item.correct_answer}{EOS}"))
            self.assertEqual(result.to_dict()["config"]["think_mode"], "none")
            self.assertIs(result.to_dict()["config"]["close_think"], False)
            cfg = self._config_json_for(self._latest_experiment_id())
            self.assertEqual(cfg["think_mode"], "none")
            self.assertIs(cfg["close_think"], False)

    def test_think_mode_validation(self):
        with self.assertRaises(ValueError):
            EvaluationConfig(think_mode="reasoning")
        with self.assertRaises(ValueError):
            EvaluationConfig(think_mode="none", close_think=True)
        with self.assertRaises(ValueError):
            EvaluationConfig(rehearsal_k=-1)
        with self.assertRaises(ValueError):
            MetaLearningConfig(think_mode="reasoning")
        with self.assertRaises(ValueError):
            MetaLearningConfig(rehearsal_k=-1)

    def test_think_mode_baseline_falls_back_to_empty_without_reasoning(self):
        dataset = make_dataset(4)
        # Think template, but the model's baseline answers carry no reasoning.
        model = FakeModel(dataset, think=False)
        model._tokenizer = FakeTokenizer(think=True)
        EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="fallback", think_mode="baseline"), verbose=False
        )
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"</think>\n\n{item.correct_answer}{EOS}")

    def test_think_mode_noop_without_think_template(self):
        dataset = make_dataset(4)
        for mode in ("none", "empty", "baseline", "rationale"):
            model = FakeModel(dataset)  # template ends in <assistant>
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, EvaluationConfig(name="plain", think_mode=mode), verbose=False
            )
            for item, target, stop in zip(
                dataset.items[:3], model.trained_targets, model.trained_stop_targets
            ):
                self.assertEqual(target, f"{item.correct_answer}{EOS}")
                self.assertEqual(stop, target)
            # No think block to fill: no rationale is generated, and nothing
            # is counted as missing.
            self.assertEqual(model.rationale_prompts, [])
            self.assertEqual(result.rationale_missing_count, 0)

    def test_model_kwargs_are_recorded_and_used_lazily(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset)
        h = EvaluationHarness(model=model, db=self.db, model_kwargs={"learning_rate": 1e-5})
        h.run(dataset, EvaluationConfig(name="lr"), verbose=False)
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["model_kwargs"],
            {"learning_rate": 1e-5},
        )
        # The lazy path forwards the kwargs to StatefulLLM.
        seen = {}
        original = harness.StatefulLLM
        harness.StatefulLLM = lambda **kw: seen.update(kw) or model
        try:
            EvaluationHarness(db=self.db, model_kwargs={"learning_rate": 2e-5}).model
        finally:
            harness.StatefulLLM = original
        self.assertEqual(seen, {"learning_rate": 2e-5})

    def test_ground_truth_ignores_revision_prompt(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset)
        config = EvaluationConfig(name="gt-fs", revision_prompt="fewshot")
        EvaluationHarness(model=model, db=self.db).run(dataset, config, verbose=False)
        self.assertEqual(model.revision_prompts, [])
        self.assertEqual(len(model.trained_targets), 3)

    def test_invalid_revision_is_counted_and_skipped(self):
        dataset = make_dataset(5)
        model = FakeModel(dataset, revision="invalid")
        config = EvaluationConfig(
            name="bad", train_ratio=0.6, training_source="self_generated"
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )

        self.assertEqual(model.trained_targets, [])
        self.assertEqual(result.revision_invalid_count, 3)
        self.assertEqual(result.train_items, [])
        # Skipped items are neither trained nor holdout.
        self.assertEqual(len(result.holdout_items), 2)
        self.assertEqual(result.to_dict()["metrics"]["revision_invalid_count"], 3)
        invalid = [d for d in result.to_dict()["items"] if d["revision_invalid"]]
        self.assertEqual(len(invalid), 3)
        self.assertEqual(
            len(
                self.db.get_training_events_for_experiment(self._latest_experiment_id())
            ),
            0,
        )

    def test_ground_truth_has_no_revision_judgement(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="gt-rev"), verbose=False
        )
        for item in result.items:
            self.assertIsNone(item.revision_answer)
            self.assertIsNone(item.revision_has_key_terms)
            self.assertIsNone(item.revision_changed_text)
            self.assertIsNone(item.revision_changed_verdict)
        self.assertEqual(result.revision_valid_count, 0)
        self.assertEqual(result.revision_attempted_count, 0)
        self.assertEqual(
            result.revision_summary(),
            {
                "attempted": 0,
                "valid": 0,
                "invalid": 0,
                "correct": 0,
                "fixed": 0,
                "broke": 0,
                "unchanged_text": 0,
            },
        )
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "gt-rev.html")
        ).read_text()
        self.assertNotIn("Revisions:", text)
        self.assertNotIn("responses-container three", text)

    def test_revision_quality_is_judged_before_training(self):
        # 5 items, train_ratio 0.8 -> q00..q03 trained, q04 holdout.
        #   q00: baseline wrong, revision right      -> valid, fixed
        #   q01: baseline right, revision wrong      -> valid, broke
        #   q02: baseline wrong, revision restates it -> valid, wrong, verbatim
        #   q03: baseline wrong, revision unparseable -> invalid, skipped
        dataset = make_dataset(5)
        revisions = {
            "q00": "[[0]] The answer is Answer0. [[/0]]",
            "q01": "[[0]] Honestly I have no idea about this one. [[/0]]",
            "q02": f"[[0]] {DONT_KNOW} [[/0]]",
            "q03": "Here is a revision with no markers at all.",
        }
        model = FakeModel(dataset, revision=revisions, known_at_baseline={"q01"})
        config = EvaluationConfig(
            name="sg-quality",
            train_ratio=0.8,
            training_source="self_generated",
            train_correct_items=True,
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        by_id = {item.item_id: item for item in result.items}

        # Per-item instrumentation.
        q00, q01, q02, q03, q04 = (by_id[f"q0{i}"] for i in range(5))
        self.assertEqual(q00.revision_answer, "The answer is Answer0.")
        self.assertIs(q00.revision_has_key_terms, True)
        self.assertIs(q00.revision_changed_text, True)
        self.assertIs(q00.revision_changed_verdict, True)
        self.assertTrue(q00.revision_fixed)
        self.assertFalse(q00.revision_broke)

        self.assertEqual(q01.revision_answer, "Honestly I have no idea about this one.")
        self.assertIs(q01.revision_has_key_terms, False)
        self.assertIs(q01.revision_changed_text, True)
        self.assertIs(q01.revision_changed_verdict, True)
        self.assertTrue(q01.revision_broke)
        self.assertFalse(q01.revision_fixed)

        self.assertEqual(q02.revision_answer, DONT_KNOW)
        self.assertIs(q02.revision_has_key_terms, False)
        self.assertIs(q02.revision_changed_text, False)
        self.assertIs(q02.revision_changed_verdict, False)
        self.assertFalse(q02.revision_fixed or q02.revision_broke)

        self.assertTrue(q03.revision_invalid)
        self.assertFalse(q03.was_trained)
        self.assertIsNone(q03.revision_answer)
        self.assertIsNone(q03.revision_has_key_terms)

        self.assertFalse(q04.was_trained)
        self.assertIsNone(q04.revision_answer)

        # Aggregate counts: 4 attempted, 3 valid, 1 correct, 1 fixed, 1 broke.
        self.assertEqual(result.revision_attempted_count, 4)
        self.assertEqual(result.revision_valid_count, 3)
        self.assertEqual(result.revision_invalid_count, 1)
        self.assertEqual(result.revision_correct_count, 1)
        self.assertEqual(result.revision_fixed_count, 1)
        self.assertEqual(result.revision_broke_count, 1)
        self.assertEqual(result.revision_unchanged_text_count, 1)
        summary = result.revision_summary()
        self.assertEqual(
            summary,
            {
                "attempted": 4,
                "valid": 3,
                "invalid": 1,
                "correct": 1,
                "fixed": 1,
                "broke": 1,
                "unchanged_text": 1,
            },
        )
        self.assertEqual(
            result.revision_summary_text(),
            "Revisions: 4 attempted, 3 valid, of which 1 correct; fixed 1 wrong "
            "answers, broke 1 right ones (1 restated the baseline verbatim).",
        )
        d = result.to_dict()
        self.assertEqual(d["metrics"]["revision_summary"], summary)
        d_q00 = next(i for i in d["items"] if i["item_id"] == "q00")
        self.assertEqual(d_q00["revision_answer"], "The answer is Answer0.")
        self.assertIs(d_q00["revision_has_key_terms"], True)
        self.assertIs(d_q00["revision_changed_text"], True)
        self.assertIs(d_q00["revision_changed_verdict"], True)

        # Training actually used the parsed revisions (3 valid ones).
        self.assertEqual(
            model.trained_targets,
            [
                f"The answer is Answer0.{EOS}",
                f"Honestly I have no idea about this one.{EOS}",
                f"{DONT_KNOW}{EOS}",
            ],
        )

        # The report shows each revision with its own judge mark and the summary.
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "sg-quality.html")
        ).read_text()
        self.assertIn("Revisions: 4 attempted, 3 valid, of which 1 correct", text)
        self.assertIn("The answer is Answer0.", text)
        self.assertIn("Honestly I have no idea about this one.", text)
        self.assertIn("Revision ✓ <small>(fixes baseline)</small>", text)
        self.assertIn("Revision ✗ <small>(breaks baseline)</small>", text)
        self.assertIn("Revision ✗ <small>(restates baseline)</small>", text)
        self.assertEqual(text.count("responses-container three"), 3)

    def test_report_states_training_source(self):
        dataset = make_dataset(4)
        for source in ("ground_truth", "self_generated"):
            model = FakeModel(dataset)
            config = EvaluationConfig(name=source, training_source=source)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=False
            )
            path = generate_html_report(result, self.tmp_path / f"{source}.html")
            text = pathlib.Path(path).read_text()
            self.assertIn("Training source", text)
            self.assertIn(f"<code>{source}</code>", text)
        self.assertIn(
            "not</strong> self-correction",
            (self.tmp_path / "ground_truth.html").read_text(),
        )


class MetaLearningTest(_TempDbTest):
    def _run(self, dataset, config, model_kwargs=None, factory=None):
        model_kwargs = model_kwargs or {}
        factory = factory or (lambda: FakeModel(dataset, **model_kwargs))
        return MetaLearningExperiment(model_factory=factory, db=self.db).run(
            dataset, config, verbose=False
        )

    def test_holdout_is_stored(self):
        dataset = make_dataset(10)
        config = MetaLearningConfig(
            name="h", seeds=[7], checkpoint_interval=4, train_ratio=0.8
        )
        result = self._run(dataset, config)
        traj = result.trajectories[7]

        self.assertEqual(traj.holdout_total, 2)
        self.assertEqual(traj.holdout_correct, 0)
        self.assertEqual(traj.holdout_accuracy, 0.0)
        self.assertEqual(result.holdout_results, {7: 0.0})
        d = traj.to_dict()
        self.assertEqual(d["holdout_total"], 2)
        self.assertEqual(d["holdout_accuracy"], 0.0)
        # Default: no per-checkpoint holdout probe.
        self.assertTrue(all(c.holdout_total is None for c in traj.checkpoints))
        self.assertEqual(
            self._config_json_for(traj.experiment_id)["training_source"], "ground_truth"
        )

    def test_holdout_every_checkpoint(self):
        dataset = make_dataset(10)
        config = MetaLearningConfig(
            name="hc",
            seeds=[7],
            checkpoint_interval=4,
            train_ratio=0.8,
            holdout_every_checkpoint=True,
        )
        traj = self._run(dataset, config).trajectories[7]
        self.assertEqual(len(traj.checkpoints), 2)
        for cp in traj.checkpoints:
            self.assertEqual(cp.holdout_total, 2)
            self.assertEqual(cp.holdout_accuracy, 0.0)

    def test_self_generated_meta_records_invalid_revisions(self):
        dataset = make_dataset(10)
        config = MetaLearningConfig(
            name="sg",
            seeds=[1],
            checkpoint_interval=4,
            train_ratio=0.8,
            training_source="self_generated",
        )
        traj = self._run(dataset, config, {"revision": "invalid"}).trajectories[1]
        self.assertEqual(traj.revision_invalid_count, 8)
        self.assertEqual(sum(len(c.revision_invalid_ids) for c in traj.checkpoints), 8)
        self.assertEqual(traj.checkpoints[-1].step, 0)
        self.assertEqual(
            self._config_json_for(traj.experiment_id)["training_source"],
            "self_generated",
        )

    def test_meta_think_mode_threads_through(self):
        dataset = make_dataset(10)
        expectations = {
            "rationale": lambda item, answer: (
                f"{fake_rationale(item)}\n</think>\n\n{answer}{EOS}",
                f"<user>{item.question}</user><assistant><think>\n"
                f"{fake_rationale(item)}\n</think>\n\n{answer}{EOS}",
            ),
            "baseline": lambda item, answer: (
                f"{answer}{EOS}",
                f"<user>{item.question}</user><assistant><think>\n"
                f"{fake_think(item)}\n</think>\n\n{answer}{EOS}",
            ),
            "empty": lambda item, answer: (
                f"</think>\n\n{answer}{EOS}",
                f"<user>{item.question}</user><assistant><think>\n"
                f"</think>\n\n{answer}{EOS}",
            ),
            "none": lambda item, answer: (
                f"{answer}{EOS}",
                f"<user>{item.question}</user><assistant><think>\n{answer}{EOS}",
            ),
        }
        by_question = {item.question: item for item in dataset}
        for think_mode, expect in expectations.items():
            model = FakeModel(dataset, think=True)
            config = MetaLearningConfig(
                name=f"tm-{think_mode}",
                seeds=[1],
                checkpoint_interval=4,
                train_ratio=0.8,
                think_mode=think_mode,
            )
            result = self._run(dataset, config, factory=lambda: model)
            traj = result.trajectories[1]
            self.assertEqual(len(model.trained_batches), 8)
            for masked, full in (b[0] for b in model.trained_batches):
                item = next(i for q, i in by_question.items() if f"<user>{q}</user>" in full)
                self.assertEqual((masked, full), expect(item, item.correct_answer))
            for stop in model.trained_stop_targets:
                self.assertTrue(stop.endswith(EOS))
                self.assertNotIn("</think>", stop)
            self.assertEqual(traj.rationale_missing_count, 0)
            self.assertEqual(traj.to_dict()["rationale_missing_count"], 0)
            self.assertEqual(
                len(model.rationale_prompts), 8 if think_mode == "rationale" else 0
            )
            cfg = self._config_json_for(traj.experiment_id)
            self.assertEqual(cfg["think_mode"], think_mode)
            self.assertIs(cfg["close_think"], think_mode != "none")
            self.assertEqual(cfg["rehearsal_k"], 0)
            self.assertEqual(cfg["model_kwargs"], {})
            self.assertEqual(result.to_dict()["config"]["think_mode"], think_mode)
            self.assertEqual(
                MetaLearningConfig.from_dict(config.to_dict()).think_mode, think_mode
            )
        # Deprecated alias.
        self.assertEqual(MetaLearningConfig(close_think=False).think_mode, "none")
        # Legacy files: only close_think, or neither.
        base = {"name": "x", "seeds": [1], "checkpoint_interval": 1,
                "training_iterations": 1, "train_ratio": 0.5}
        self.assertEqual(MetaLearningConfig.from_dict(base).think_mode, "none")
        self.assertEqual(
            MetaLearningConfig.from_dict({**base, "close_think": True}).think_mode, "empty"
        )
        self.assertEqual(
            MetaLearningConfig.from_dict({**base, "close_think": False}).think_mode, "none"
        )

    def test_meta_rehearsal_and_model_kwargs(self):
        dataset = make_dataset(10)
        # seed 1 shuffle of 10 items: 8 trained, 2 holdout. Everything is known
        # at baseline so the rehearsal pool is the whole trained split.
        model = FakeModel(dataset, think=True, known_at_baseline={i.id for i in dataset})
        config = MetaLearningConfig(
            name="reh",
            seeds=[1],
            checkpoint_interval=4,
            train_ratio=0.8,
            rehearsal_k=2,
            think_mode="baseline",
            train_correct_items=True,
        )
        experiment = MetaLearningExperiment(
            model_factory=lambda: model, db=self.db, model_kwargs={"learning_rate": 3e-5}
        )
        result = experiment.run(dataset, config, verbose=False)
        traj = result.trajectories[1]
        # 8 items, each one joint call over 1 correction + 2 rehearsal rows.
        self.assertEqual(model.train_calls, 8)
        self.assertEqual(len(model.joint_calls), 8)
        self.assertEqual({c["rehearsal_k"] for c in model.joint_calls}, {2})
        self.assertEqual({c["rehearsal_weight"] for c in model.joint_calls}, {1.0})
        self.assertEqual(len(model.trained_batches), 8 * 3)
        self.assertTrue(all(shape[0] == 1 for shape in model.batch_shapes))
        self.assertEqual(traj.train_rehearsal_final_losses, [0.41] * 8)
        self.assertAlmostEqual(traj.mean_train_rehearsal_final_loss, 0.41)
        self.assertIn("(train loss 0.58, rehearsal 0.41)", traj.training_summary_text(12, 0.6))
        self.assertEqual(result.model_kwargs, {"learning_rate": 3e-5})
        holdout_questions = {
            dataset[idx].question for idx in traj_holdout_indices(dataset, seed=1)
        }
        for rows in training_groups(model, [2] * 8):
            self.assertRegex(rows[0][0], r"^Answer\d+<eos>$")  # correction first
            for masked, full in rows[1:]:
                q = full.split("<user>")[1].split("</user>")[0]
                self.assertNotIn(q, holdout_questions)
                self.assertRegex(masked, r"^Thinking about q\d\d\.\n</think>\n\nIt is Answer\d+\.<eos>$")
        cfg = self._config_json_for(traj.experiment_id)
        self.assertEqual(cfg["rehearsal_k"], 2)
        self.assertEqual(cfg["rehearsal_max_tokens"], 768)
        self.assertEqual(
            MetaLearningConfig.from_dict(config.to_dict()).rehearsal_max_tokens, 768
        )
        self.assertEqual(cfg["rehearsal_weight"], 1.0)
        self.assertEqual(cfg["model_kwargs"], {"learning_rate": 3e-5})
        # LoRA capacity defaults are recorded even when no flag set them.
        self.assertEqual(cfg["lora"], {"rank": 32, "layers": 24, "scale": 10.0})
        # The per-item rehearsal loss survives a save/load round trip.
        path = self.tmp_path / "reh.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(loaded.trajectories[1].train_rehearsal_final_losses, [0.41] * 8)
        self.assertEqual(loaded.config.rehearsal_weight, 1.0)
        self.assertEqual(loaded.model_kwargs, {"learning_rate": 3e-5})

    def test_meta_rehearsal_pool_honors_max_tokens(self):
        dataset = make_dataset(10)
        known = {i.id for i in dataset}
        model = FakeModel(
            dataset, think=True, known_at_baseline=known, long_at_baseline={"q00"}
        )
        config = MetaLearningConfig(
            name="cap",
            seeds=[1],
            checkpoint_interval=4,
            train_ratio=0.8,
            rehearsal_k=9,
            rehearsal_max_tokens=200,
            think_mode="baseline",
            train_correct_items=True,
        )
        MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
            dataset, config, verbose=False
        )
        # 8 trained items; every one is rehearsed on the other short trained
        # items only (7 others, minus q00 when it is in the trained split).
        long_q = f"<user>{dataset[0].question}</user>"
        rehearsal_rows = [
            rows[0] for rows in model.trained_batches if "</think>" in rows[0][0]
        ]
        self.assertTrue(rehearsal_rows)
        for _, full in rehearsal_rows:
            self.assertNotIn(long_q, full)
        self.assertTrue(all(shape[0] == 1 for shape in model.batch_shapes))

    def test_meta_fewshot_revision_prompt_threads_through(self):
        dataset = make_dataset(10)
        model = FakeModel(dataset)
        config = MetaLearningConfig(
            name="sg-fs",
            seeds=[1],
            checkpoint_interval=4,
            train_ratio=0.8,
            training_source="self_generated",
            revision_prompt="fewshot",
        )
        result = self._run(dataset, config, factory=lambda: model)
        traj = result.trajectories[1]
        self.assertEqual(len(model.revision_prompts), 8)
        for prompt in model.revision_prompts:
            self.assertIn("Example 2", prompt)
            self.assertIn("User: ", prompt)
            self.assertNotIn("<user>", prompt)
        self.assertEqual(
            self._config_json_for(traj.experiment_id)["revision_prompt"], "fewshot"
        )
        self.assertEqual(result.to_dict()["config"]["revision_prompt"], "fewshot")
        self.assertEqual(
            MetaLearningConfig.from_dict(config.to_dict()).revision_prompt, "fewshot"
        )

    def test_window_metrics_from_scripted_learning(self):
        # 20 items, 15 trained in windows of 5. The model learns nothing from
        # its first 5 training calls and everything after that.
        dataset = make_dataset(20)
        config = MetaLearningConfig(
            name="w", seeds=[3], checkpoint_interval=5, train_ratio=0.75
        )
        traj = self._run(dataset, config, {"learns_after": 5}).trajectories[3]

        self.assertEqual(traj.window_sizes, [5, 5, 5])
        cp1, cp2, cp3 = traj.checkpoints
        self.assertEqual((cp1.window_improved, cp1.window_stuck), (0, 5))
        self.assertEqual((cp2.window_improved, cp2.window_stuck), (5, 0))
        self.assertEqual((cp3.window_improved, cp3.window_stuck), (5, 0))
        # Cumulative fields keep the old meaning.
        self.assertEqual((cp3.improved, cp3.stuck), (10, 5))
        self.assertEqual(cp3.improved + cp3.stuck + cp3.retained + cp3.regressed, 15)
        self.assertAlmostEqual(cp3.improvement_rate, 10 / 15)
        self.assertEqual(cp3.window_improvement_rate, 1.0)
        # Score uses window rates: late (1.0) - early (0.0).
        self.assertIsNone(traj.meta_learning_score_reason)
        self.assertAlmostEqual(traj.meta_learning_score, 1.0)
        self.assertEqual(traj.final_trained_accuracy, 10 / 15)
        self.assertEqual(traj.total_net_learning, 10)
        # Window ids are disjoint across checkpoints and cover the trained set.
        seen = [i for c in traj.checkpoints for i in c.window_ids]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(cp3.improved_ids + cp3.stuck_ids))

    def test_small_windows_give_no_score_with_reason(self):
        dataset = make_dataset(12)
        config = MetaLearningConfig(
            name="tiny", seeds=[3], checkpoint_interval=3, train_ratio=0.75
        )
        traj = self._run(dataset, config).trajectories[3]
        self.assertEqual(traj.window_sizes, [3, 3, 3])
        self.assertIsNone(traj.meta_learning_score)
        self.assertIn("below 5", traj.meta_learning_score_reason)
        self.assertFalse(traj.has_meta_learning_score)
        self.assertIn("below 5", traj.to_dict()["meta_learning_score_reason"])

    def test_compute_checkpoint_window_vs_cumulative(self):
        experiment = MetaLearningExperiment(model_factory=lambda: None, db=self.db)
        baseline = {
            "a": ("", False),
            "b": ("", True),
            "c": ("", True),
            "d": ("", False),
        }
        post = {"a": ("", True), "b": ("", False), "c": ("", True), "d": ("", False)}
        cp = experiment._compute_checkpoint(
            step=4,
            baseline_responses=baseline,
            post_responses=post,
            trained_items=["a", "b", "c", "d"],
            window_ids=["c", "d"],
        )
        self.assertEqual(
            (cp.improved, cp.regressed, cp.retained, cp.stuck), (1, 1, 1, 1)
        )
        self.assertEqual(
            (
                cp.window_improved,
                cp.window_regressed,
                cp.window_retained,
                cp.window_stuck,
            ),
            (0, 0, 1, 1),
        )
        self.assertEqual(cp.window_size, 2)
        self.assertEqual(cp.forgetting_rate, 0.5)
        self.assertEqual(cp.window_forgetting_rate, 0.0)
        self.assertEqual(cp.window_improvement_rate, 0.0)

    def test_repeats_give_within_and_across_variance(self):
        dataset = make_dataset(20)
        # seed 1: repeats learn / never learn -> scores 1.0, 0.0
        # seed 2: both repeats learn         -> scores 1.0, 1.0
        schedule = iter([5, 15, 5, 5])
        factory = lambda: FakeModel(dataset, learns_after=next(schedule))
        config = MetaLearningConfig(
            name="rep", seeds=[1, 2], checkpoint_interval=5, train_ratio=0.75, repeats=2
        )
        result = self._run(dataset, config, factory=factory)

        self.assertEqual(
            sorted(result.all_trajectories), [(1, 0), (1, 1), (2, 0), (2, 1)]
        )
        self.assertEqual(sorted(result.trajectories), [1, 2])
        self.assertEqual(result.trajectories[1].repeat, 0)
        self.assertEqual(len(result.trajectories_for_seed(1)), 2)
        scores = {k: t.meta_learning_score for k, t in result.all_trajectories.items()}
        self.assertAlmostEqual(scores[(1, 0)], 1.0)
        self.assertAlmostEqual(scores[(1, 1)], 0.0)
        self.assertAlmostEqual(scores[(2, 0)], 1.0)
        self.assertAlmostEqual(scores[(2, 1)], 1.0)
        # within = mean(var([1,0]), var([1,1])) = mean(0.25, 0) = 0.125
        self.assertAlmostEqual(result.within_seed_variance, 0.125)
        # across = var([0.5, 1.0]) = 0.0625
        self.assertAlmostEqual(result.across_seed_variance, 0.0625)
        self.assertAlmostEqual(result.signal_to_noise, 0.5)
        # Repeats share the shuffle: identical training order per seed.
        r0, r1 = result.trajectories_for_seed(1)
        self.assertEqual(
            [c.window_ids for c in r0.checkpoints],
            [c.window_ids for c in r1.checkpoints],
        )
        # The DB experiment names distinguish repeats.
        names = {e.name for e in self.db.get_experiments()}
        self.assertIn("rep_seed1_r0", names)
        self.assertIn("rep_seed1_r1", names)

        # Round-trips through JSON with every repeat intact.
        path = self.tmp_path / "rep.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(
            sorted(loaded.all_trajectories), sorted(result.all_trajectories)
        )
        self.assertEqual(loaded.config.repeats, 2)
        self.assertAlmostEqual(loaded.within_seed_variance, 0.125)
        self.assertAlmostEqual(loaded.signal_to_noise, 0.5)

    def test_variance_undefined_without_repeats(self):
        dataset = make_dataset(20)
        config = MetaLearningConfig(
            name="one", seeds=[1, 2], checkpoint_interval=5, train_ratio=0.75
        )
        result = self._run(dataset, config, {"learns_after": 5})
        self.assertIsNone(result.within_seed_variance)
        self.assertIsNotNone(result.across_seed_variance)
        self.assertIsNone(result.signal_to_noise)

    def test_final_accuracy_alias_is_deprecated(self):
        traj = SeedTrajectory(seed=1)
        traj.checkpoints.append(
            Checkpoint(
                step=1,
                timestamp="t",
                baseline_correct=0,
                baseline_total=2,
                post_correct=1,
                post_total=2,
                improved=1,
                retained=0,
                regressed=0,
                stuck=1,
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(traj.final_accuracy, 0.5)
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        self.assertEqual(traj.final_trained_accuracy, 0.5)
        self.assertIn("final_trained_accuracy", traj.to_dict())


def traj_holdout_indices(dataset, seed: int) -> list[int]:
    """The holdout indices MetaLearningExperiment picks for ``seed`` (train_ratio 0.8)."""
    import random

    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices[int(len(indices) * 0.8) :]


def traj_train_ids(dataset, seed: int) -> list[str]:
    """The trained item ids, in order, MetaLearningExperiment picks for ``seed`` (train_ratio 0.8)."""
    import random

    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [dataset[i].id for i in indices[: int(len(indices) * 0.8)]]


class RehearsalTest(_TempDbTest):
    """rehearsal_k folds self-distillation rows into each correction's steps.

    Every trained item is one ``train_on_examples`` call: the correction plus
    its rehearsal examples (in sampled order), each a single row.
    """

    def test_rehearsal_joins_correction_with_rehearsal_rows(self):
        # 6 items, train_ratio 5/6 -> q00..q04 trained, q05 holdout.
        # q01, q03 and q05 are known at baseline: the pool is {q01, q03} (q05 is
        # holdout and must never be rehearsed).
        dataset = make_dataset(6)
        model = FakeModel(dataset, think=True, known_at_baseline={"q01", "q03", "q05"})
        config = EvaluationConfig(
            name="reh",
            train_ratio=5 / 6,
            rehearsal_k=2,
            seed=7,
            rehearsal_weight=0.5,
            think_mode="baseline",
            train_correct_items=True,
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        by_id = {item.item_id: item for item in result.items}
        trained = [by_id[item.id] for item in dataset.items[:5]]
        ks = [len(r.rehearsal_item_ids) for r in trained]

        # One joint call per item carrying k single-row rehearsal examples and
        # the config's target, cap, and weight; never a train_on_example call.
        self.assertEqual(ks, [2, 1, 2, 1, 2])  # k capped at the pool minus self
        self.assertEqual(model.train_calls, 5)
        self.assertEqual(model.train_kwargs, [])
        self.assertEqual([c["rehearsal_k"] for c in model.joint_calls], ks)
        for call in model.joint_calls:
            self.assertEqual(call["loss_target"], 0.6)
            self.assertEqual(call["max_steps"], 12)
            self.assertEqual(call["rehearsal_weight"], 0.5)
        self.assertEqual(len(model.trained_batches), sum(1 + k for k in ks))
        for shape in model.batch_shapes:
            self.assertEqual(shape[0], 1)
        # Single rows are never padded: the last label is <eos>, masked in.
        for rows in model.trained_batches:
            self.assertTrue(rows[0][0].endswith(EOS))
        # The stop rule saw only the correction loss: 3 steps to cross 0.6 even
        # though the rehearsal loss was under it from step 1, and the
        # rehearsal loss at that last step is reported per item.
        for r in trained:
            self.assertEqual(r.train_steps, 3)
            self.assertAlmostEqual(r.train_final_loss, 0.58)
            self.assertAlmostEqual(r.train_rehearsal_final_loss, 0.41)
        self.assertIsNone(by_id["q05"].train_rehearsal_final_loss)
        self.assertAlmostEqual(result.mean_train_rehearsal_final_loss, 0.41)
        self.assertEqual(
            result.training_summary_text(),
            "Training: mean 3.0 steps/item (cap 12, answer-loss target 0.60), "
            "mean final answer loss 0.58 (train loss 0.58, rehearsal 0.41); "
            "0 items hit the cap; 0 rationales missing; "
            "rehearsal active in 0% of steps",
        )
        groups = training_groups(model, ks)
        for item, r, rows in zip(dataset.items[:5], trained, groups):
            pool = [pid for pid in ("q01", "q03") if pid != item.id]
            # Row 0 of each item's group is the correction with the model's own
            # reasoning unmasked.
            masked, full = rows[0]
            self.assertEqual(masked, f"{item.correct_answer}{EOS}")
            self.assertIn(f"{fake_think(item)}\n</think>\n\n", full)
            # The following calls are correct-baseline items, never the item
            # itself, trained on their own full baseline output, in sample order.
            rehearsed = []
            for masked, full in rows[1:]:
                r_item = next(
                    it for it in dataset.items if f"<user>{it.question}</user>" in full
                )
                rehearsed.append(r_item.id)
                self.assertNotEqual(r_item.id, item.id)
                self.assertIn(r_item.id, pool)
                baseline_raw = by_id[r_item.id].initial_response_raw
                self.assertEqual(masked, f"{baseline_raw}{EOS}")
                self.assertEqual(
                    full,
                    f"<user>{r_item.question}</user><assistant><think>\n{baseline_raw}{EOS}",
                )
            self.assertEqual(sorted(rehearsed), sorted(pool))
            self.assertEqual(rehearsed, r.rehearsal_item_ids)
        # Holdout items are untouched and have no rehearsal ids.
        self.assertEqual(by_id["q05"].rehearsal_item_ids, [])
        self.assertNotIn("q05", {rid for r in result.items for rid in r.rehearsal_item_ids})
        # One training event per item, not per call.
        experiment_id = self._latest_experiment_id()
        events = self.db.get_training_events_for_experiment(experiment_id)
        self.assertEqual(len(events), 5)
        d = result.to_dict()
        self.assertEqual(d["config"]["rehearsal_k"], 2)
        self.assertEqual(d["config"]["rehearsal_max_tokens"], 768)
        self.assertEqual(d["config"]["rehearsal_weight"], 0.5)
        self.assertAlmostEqual(d["metrics"]["mean_train_rehearsal_final_loss"], 0.41)
        cfg = self._config_json_for(experiment_id)
        self.assertEqual(cfg["rehearsal_k"], 2)
        self.assertEqual(cfg["rehearsal_max_tokens"], 768)
        self.assertEqual(cfg["rehearsal_weight"], 0.5)
        self.assertTrue(
            any(i["rehearsal_item_ids"] for i in d["items"] if i["was_trained"])
        )
        self.assertEqual(
            {i["train_rehearsal_final_loss"] for i in d["items"] if i["was_trained"]},
            {0.41},
        )
        with self.assertRaises(ValueError):
            EvaluationConfig(rehearsal_weight=-0.1)
        with self.assertRaises(ValueError):
            MetaLearningConfig(rehearsal_weight=-1)

    def test_rehearsal_pool_honors_max_tokens(self):
        # Everything is correct at baseline, but q02 rambles for
        # LONG_ANSWER_TOKENS tokens: it is never rehearsed under the cap, and
        # the pool shrinks to the remaining items.
        dataset = make_dataset(6)
        known = {item.id for item in dataset}

        def run(max_tokens: int):
            model = FakeModel(
                dataset, think=True, known_at_baseline=known, long_at_baseline={"q02"}
            )
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset,
                EvaluationConfig(
                    name="cap",
                    train_ratio=1.0,
                    rehearsal_k=5,
                    rehearsal_max_tokens=max_tokens,
                    train_correct_items=True,
                ),
                verbose=False,
            )
            return model, {item.item_id: item for item in result.items}

        model, by_id = run(200)
        self.assertGreater(by_id["q02"].initial_token_count, 200)
        for item_id in known - {"q02"}:
            self.assertLessEqual(by_id[item_id].initial_token_count, 200)
        for item_id, r in by_id.items():
            self.assertNotIn("q02", r.rehearsal_item_ids)
            expected = sorted(known - {"q02", item_id})
            self.assertEqual(sorted(r.rehearsal_item_ids), expected)
        self.assertEqual(model.train_calls, len(by_id))
        self.assertEqual(
            [c["rehearsal_k"] for c in model.joint_calls],
            [len(r.rehearsal_item_ids) for r in by_id.values()],
        )
        self.assertTrue(all(shape[0] == 1 for shape in model.batch_shapes))
        cfg = self._config_json_for(self._latest_experiment_id())
        self.assertEqual(cfg["rehearsal_max_tokens"], 200)

        # Raising the cap above the long baseline lets q02 back into the pool.
        model, by_id = run(LONG_ANSWER_TOKENS + 100)
        self.assertTrue(any("q02" in r.rehearsal_item_ids for r in by_id.values()))
        self.assertTrue(all(shape[0] == 1 for shape in model.batch_shapes))

        with self.assertRaises(ValueError):
            EvaluationConfig(rehearsal_max_tokens=0)
        with self.assertRaises(ValueError):
            MetaLearningConfig(rehearsal_max_tokens=-5)

    def test_collate_pads_with_zero_mask(self):
        # collate_training_examples still pads a multi-row batch; padded
        # positions carry mask 0 and the pad id (eos here, since the fake
        # tokenizer has no pad id). The harness no longer builds such batches,
        # but StatefulLLM callers may.
        dataset = make_dataset(3)
        tok = FakeTokenizer(think=True)
        tok.eos_token_id = 99999
        rows = [
            harness.make_rehearsal_example(
                item,
                f"{fake_think(item)}\n</think>\n\nIt is {item.correct_answer}.{'!' * i}",
                tok,
            )
            for i, item in enumerate(dataset.items)
        ]
        ex = harness.collate_training_examples(rows, tok)
        n_rows, length = ex.mask.shape
        self.assertEqual(n_rows, 3)
        self.assertEqual(ex.input.shape, (n_rows, length))
        self.assertEqual(ex.label.shape, (n_rows, length))
        masks = ex.mask.tolist()
        labels = ex.label.tolist()
        inputs = ex.input.tolist()
        lengths = [
            max(i for i, t in enumerate(row) if t != 99999) + 1 for row in inputs
        ]
        self.assertTrue(any(l < length for l in lengths))
        for row_mask, row_label, row_len in zip(masks, labels, lengths):
            if row_len < length:
                self.assertEqual(row_mask[row_len:], [0] * (length - row_len))
                self.assertEqual(row_label[row_len:], [99999] * (length - row_len))
            # The last real label of every row is <eos> (masked in).
            self.assertEqual(row_mask[row_len - 1], 1)

    def test_rehearsal_with_empty_pool_trains_single_row(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, think=True)  # nothing correct at baseline
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="nopool", rehearsal_k=3), verbose=False
        )
        # With nothing to rehearse the harness falls back to train_on_example.
        self.assertEqual(model.train_calls, 3)  # one call per trained item
        self.assertEqual(model.joint_calls, [])
        self.assertEqual(model.train_kwargs, [(12, 0.6, 12)] * 3)
        self.assertTrue(all(shape[0] == 1 for shape in model.batch_shapes))
        self.assertTrue(
            all(i.train_rehearsal_final_loss is None for i in result.items)
        )
        self.assertIsNone(result.mean_train_rehearsal_final_loss)
        self.assertNotIn("rehearsal", result.training_summary_text())

    def test_rehearsal_k_zero_uses_single_example_calls(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, known_at_baseline={i.id for i in dataset})
        EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="k0", rehearsal_k=0, train_correct_items=True), verbose=False
        )
        self.assertEqual(model.joint_calls, [])
        self.assertEqual(model.train_kwargs, [(12, 0.6, 12)] * 3)

    def test_rehearsal_sampling_is_seeded(self):
        dataset = make_dataset(12)
        known = {f"q{i:02d}" for i in range(12)}
        ids = []
        for _ in range(2):
            model = FakeModel(dataset, think=True, known_at_baseline=known)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, EvaluationConfig(
                    name="seeded", rehearsal_k=2, seed=3, train_correct_items=True
                ),
                verbose=False,
            )
            ids.append([i.rehearsal_item_ids for i in result.items if i.was_trained])
        self.assertEqual(ids[0], ids[1])
        self.assertTrue(all(len(r) == 2 for r in ids[0]))


class CollapseSummaryTest(unittest.TestCase):
    """EvaluationResult token/empty-think/holdout summaries from scripted results."""

    @staticmethod
    def _item(i, trained, init_ok, post_ok, init_tok, post_tok, post_raw):
        return ItemResult(
            item_id=f"q{i}",
            question=f"Q{i}",
            correct_answer="A",
            key_terms=["a"],
            initial_response="x",
            initial_response_raw="x",
            initial_has_key_terms=init_ok,
            post_response="y",
            post_response_raw=post_raw,
            post_has_key_terms=post_ok,
            was_trained=trained,
            initial_token_count=init_tok,
            post_token_count=post_tok,
        )

    def test_summary_fields(self):
        result = EvaluationResult(
            config=EvaluationConfig(name="s"), dataset_name="d", timestamp="t"
        )
        result.items = [
            self._item(0, True, False, True, 700, 8, "</think>\n\nA"),
            self._item(1, True, True, True, 800, 12, "  </think>\n\nA"),
            self._item(2, False, True, False, 600, 10, "reason\n</think>\n\nB"),
            self._item(3, False, False, True, 500, 10, "</think>\n\nA"),
            self._item(4, False, True, True, 400, 10, "</think>\n\nA"),
        ]
        self.assertAlmostEqual(result.mean_baseline_tokens, 600.0)
        self.assertAlmostEqual(result.mean_post_tokens, 10.0)
        self.assertEqual(result.post_empty_think_count, 4)
        self.assertEqual(result.post_count, 5)
        self.assertEqual(
            result.collapse_summary_text(),
            "Response length: baseline 600 tok → post 10 tok; empty-think responses "
            "after training: 4/5",
        )
        self.assertEqual(result.holdout_total, 3)
        self.assertEqual(result.holdout_correct, 2)
        self.assertEqual(result.holdout_baseline_correct, 2)
        self.assertEqual(
            result.holdout_summary_text(),
            "Holdout accuracy: 66.7% (2/3, baseline 2/3)",
        )
        metrics = result.to_dict()["metrics"]
        self.assertEqual(metrics["mean_baseline_tokens"], 600.0)
        self.assertEqual(metrics["mean_post_tokens"], 10.0)
        self.assertEqual(metrics["post_empty_think_count"], 4)
        self.assertEqual(metrics["holdout_baseline_correct"], 2)

    def test_summary_before_post_pass(self):
        result = EvaluationResult(
            config=EvaluationConfig(name="s"), dataset_name="d", timestamp="t"
        )
        result.items = [
            ItemResult(
                item_id="q", question="Q", correct_answer="A", key_terms=["a"],
                initial_response="x", initial_response_raw="x",
                initial_has_key_terms=False, initial_token_count=42,
            )
        ]
        self.assertEqual(result.mean_baseline_tokens, 42.0)
        self.assertEqual(result.mean_post_tokens, 0.0)
        self.assertEqual(result.post_empty_think_count, 0)
        self.assertEqual(result.post_count, 0)

    def test_harness_records_token_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = adaptible.Database(pathlib.Path(tmp) / "t.db")
            dataset = make_dataset(3)
            model = FakeModel(dataset, think=True, known_at_baseline={"q02"})
            result = EvaluationHarness(model=model, db=db).run(
                dataset, EvaluationConfig(name="tok", train_ratio=2 / 3), verbose=False
            )
            for item in result.items:
                self.assertEqual(item.initial_token_count, len(item.initial_response_raw))
                self.assertEqual(item.post_token_count, len(item.post_response_raw))
            self.assertEqual(result.post_empty_think_count, 0)
            self.assertEqual(
                result.holdout_summary_text(), "Holdout accuracy: 100.0% (1/1, baseline 1/1)"
            )


class LossTargetTest(_TempDbTest):
    """The harness and meta experiment stop training on the loss target."""

    def test_defaults_and_normalization(self):
        config = EvaluationConfig(name="d")
        self.assertEqual(config.loss_target, 0.6)
        self.assertEqual(config.training_iterations, 12)
        self.assertIsNone(EvaluationConfig(name="off", loss_target=0).loss_target)
        self.assertIsNone(EvaluationConfig(name="off", loss_target=-1).loss_target)
        self.assertIsNone(EvaluationConfig(name="off", loss_target=None).loss_target)
        meta_config = MetaLearningConfig(name="m", loss_target=-1)
        self.assertIsNone(meta_config.loss_target)
        self.assertEqual(MetaLearningConfig(name="m").loss_target, 0.6)
        self.assertEqual(MetaLearningConfig(name="m").training_iterations, 12)
        # Round-trips, and legacy files without the key mean fixed-count.
        self.assertEqual(
            MetaLearningConfig.from_dict(MetaLearningConfig(name="m").to_dict()).loss_target,
            0.6,
        )
        legacy = {"name": "x", "seeds": [1], "checkpoint_interval": 1,
                  "training_iterations": 25, "train_ratio": 0.5}
        self.assertIsNone(MetaLearningConfig.from_dict(legacy).loss_target)

    def test_harness_passes_target_and_records_steps(self):
        dataset = make_dataset(5)
        model = FakeModel(dataset)
        config = EvaluationConfig(name="lt", train_ratio=0.6, training_iterations=7)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        # Every call carried the target and the cap.
        self.assertEqual(model.train_kwargs, [(7, 0.6, 7)] * 3)
        for item in result.train_items:
            self.assertEqual(item.train_steps, 3)  # FAKE_LOSSES crosses 0.6 on step 3
            self.assertAlmostEqual(item.train_initial_loss, 6.05)
            self.assertAlmostEqual(item.train_final_loss, 0.58)
            self.assertFalse(item.train_hit_cap)
        for item in result.holdout_items:
            self.assertEqual(item.train_steps, 0)
            self.assertIsNone(item.train_final_loss)
        self.assertAlmostEqual(result.mean_train_steps, 3.0)
        self.assertAlmostEqual(result.mean_train_final_loss, 0.58)
        self.assertEqual(result.train_cap_hit_count, 0)
        self.assertEqual(
            result.training_summary_text(),
            "Training: mean 3.0 steps/item (cap 7, answer-loss target 0.60), "
            "mean final answer loss 0.58 (train loss 0.58); 0 items hit the cap; "
            "0 rationales missing",
        )
        d = result.to_dict()
        self.assertEqual(d["config"]["loss_target"], 0.6)
        self.assertEqual(d["config"]["training_iterations"], 7)
        self.assertAlmostEqual(d["metrics"]["mean_train_steps"], 3.0)
        self.assertEqual(d["metrics"]["train_cap_hit_count"], 0)
        trained = [i for i in d["items"] if i["was_trained"]]
        self.assertEqual({i["train_steps"] for i in trained}, {3})
        self.assertAlmostEqual(trained[0]["train_initial_loss"], 6.05)
        self.assertAlmostEqual(trained[0]["train_final_loss"], 0.58)
        # The DB stores the steps actually taken, and the config json the target.
        experiment_id = self._latest_experiment_id()
        events = self.db.get_training_events_for_experiment(experiment_id)
        self.assertEqual([e.training_iterations for e in events], [3, 3, 3])
        cfg = self._config_json_for(experiment_id)
        self.assertEqual(cfg["loss_target"], 0.6)
        self.assertEqual(cfg["training_iterations"], 7)

    def test_cap_and_disabled_target(self):
        dataset = make_dataset(4)
        # Never reaches 0.6: every item hits the cap.
        model = FakeModel(dataset, train_losses=[3.0, 2.0, 1.0])
        config = EvaluationConfig(name="cap", train_ratio=0.5, training_iterations=4)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        for item in result.train_items:
            self.assertEqual(item.train_steps, 4)
            self.assertTrue(item.train_hit_cap)
            self.assertAlmostEqual(item.train_final_loss, 1.0)
        self.assertEqual(result.train_cap_hit_count, 2)
        self.assertIn("2 items hit the cap", result.training_summary_text())

        # loss_target=None: exactly training_iterations steps, cap reported.
        model = FakeModel(dataset)
        config = EvaluationConfig(
            name="fixed", train_ratio=0.5, training_iterations=5, loss_target=None
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        self.assertEqual(model.train_kwargs, [(5, None, 5)] * 2)
        self.assertEqual([i.train_steps for i in result.train_items], [5, 5])
        self.assertIn("answer-loss target off", result.training_summary_text())

    def test_rehearsal_joint_call_carries_target_cap_and_weight(self):
        dataset = make_dataset(6)
        model = FakeModel(dataset, known_at_baseline={i.id for i in dataset})
        config = EvaluationConfig(
            name="reh-lt",
            train_ratio=0.5,
            rehearsal_k=2,
            training_iterations=9,
            rehearsal_weight=2.0,
            train_correct_items=True,
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        # 3 trained items, one joint call each with 2 rehearsal rows.
        self.assertEqual(model.train_kwargs, [])
        self.assertEqual(len(model.joint_calls), 3)
        for call in model.joint_calls:
            self.assertEqual(
                (call["loss_target"], call["max_steps"], call["rehearsal_weight"]),
                (0.6, 9, 2.0),
            )
            self.assertEqual(call["rehearsal_k"], 2)
        # Rehearsal alone never stops a call: a rehearsal script under the
        # target from step 1 still runs the correction to its own crossing.
        for item in result.train_items:
            self.assertEqual(item.train_steps, 3)
            self.assertAlmostEqual(item.train_rehearsal_final_loss, 0.41)
        # A correction that never reaches the target hits the cap regardless
        # of how low the rehearsal loss is.
        model = FakeModel(
            dataset,
            known_at_baseline={i.id for i in dataset},
            train_losses=[3.0, 2.0, 1.0],
            rehearsal_losses=[0.1],
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(
                name="cap",
                train_ratio=0.5,
                rehearsal_k=2,
                training_iterations=4,
                train_correct_items=True,
            ),
            verbose=False,
        )
        for item in result.train_items:
            self.assertEqual(item.train_steps, 4)
            self.assertTrue(item.train_hit_cap)
            self.assertAlmostEqual(item.train_rehearsal_final_loss, 0.1)
        self.assertIn(
            "mean final answer loss 1.00 (train loss 1.00, rehearsal 0.10); "
            "3 items hit the cap",
            result.training_summary_text(),
        )

    def test_rehearsal_hinge_counts_active_steps_and_threads_margin(self):
        dataset = make_dataset(6)
        # Rehearsal loss script: anchor 0.40 on step 1; step 2 drifts to 0.47
        # (> 0.40 + 0.05: active), step 3 back to 0.43 (inactive). The
        # correction crosses the target on step 3.
        model = FakeModel(
            dataset,
            known_at_baseline={i.id for i in dataset},
            rehearsal_losses=[0.40, 0.47, 0.43],
        )
        config = EvaluationConfig(
            name="hinge",
            train_ratio=0.5,
            rehearsal_k=2,
            rehearsal_margin=0.05,
            train_correct_items=True,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=True
            )
        self.assertEqual({c["rehearsal_margin"] for c in model.joint_calls}, {0.05})
        for item in result.train_items:
            self.assertEqual(item.train_steps, 3)
            self.assertEqual(len(item.rehearsal_item_ids), 2)
            self.assertAlmostEqual(item.train_rehearsal_initial_loss, 0.40)
            self.assertAlmostEqual(item.train_rehearsal_final_loss, 0.43)
            # Step 2 only, both examples: 2 of the 3 * 2 pairs.
            self.assertEqual(item.train_rehearsal_active_steps, 2)
            self.assertEqual(item.train_rehearsal_pairs, 6)
        self.assertEqual(result.train_rehearsal_active_steps, 6)
        self.assertEqual(result.train_rehearsal_pairs, 18)
        self.assertEqual(
            result.training_summary_text(),
            "Training: mean 3.0 steps/item (cap 12, answer-loss target 0.60), "
            "mean final answer loss 0.58 (train loss 0.58, rehearsal 0.43); "
            "0 items hit the cap; 0 rationales missing; "
            "rehearsal active in 33% of steps",
        )
        out = buf.getvalue()
        self.assertRegex(
            out, r"\(rehearsal 0\.40→0\.43 \(active 2/6\)\), \d+\.\ds"
        )
        self.assertIn("rehearsal active in 33% of steps", out)
        d = result.to_dict()
        self.assertEqual(d["config"]["rehearsal_margin"], 0.05)
        self.assertEqual(d["metrics"]["train_rehearsal_active_steps"], 6)
        self.assertEqual(d["metrics"]["train_rehearsal_pairs"], 18)
        for i in d["items"][:3]:
            self.assertEqual(i["train_rehearsal_active_steps"], 2)
            self.assertAlmostEqual(i["train_rehearsal_initial_loss"], 0.40)
        self.assertEqual(self._config_json_for(1)["rehearsal_margin"], 0.05)
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "hinge.html")
        ).read_text()
        self.assertIn("margin: <code>0.05</code>", text)
        self.assertIn("rehearsal active in 33% of steps", text)

        # A wider margin keeps every step inactive; the margin is threaded.
        model = FakeModel(
            dataset,
            known_at_baseline={i.id for i in dataset},
            rehearsal_losses=[0.40, 0.47, 0.43],
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(
                name="wide",
                train_ratio=0.5,
                rehearsal_k=2,
                rehearsal_margin=0.1,
                train_correct_items=True,
            ),
            verbose=False,
        )
        self.assertEqual({c["rehearsal_margin"] for c in model.joint_calls}, {0.1})
        self.assertEqual(result.train_rehearsal_active_steps, 0)
        self.assertIn("rehearsal active in 0% of steps", result.training_summary_text())
        # Without rehearsal the summary has no active figure at all.
        result = EvaluationHarness(model=FakeModel(dataset), db=self.db).run(
            dataset, EvaluationConfig(name="none", train_ratio=0.5), verbose=False
        )
        self.assertNotIn("rehearsal active", result.training_summary_text())
        for bad in (-0.01, -1):
            with self.assertRaises(ValueError):
                EvaluationConfig(rehearsal_margin=bad)
            with self.assertRaises(ValueError):
                MetaLearningConfig(rehearsal_margin=bad)
        self.assertEqual(EvaluationConfig().rehearsal_margin, 0.05)
        self.assertEqual(EvaluationConfig(rehearsal_margin=0).rehearsal_margin, 0.0)

    def test_meta_threads_rehearsal_margin_and_records_active_steps(self):
        dataset = make_dataset(10)
        model = FakeModel(
            dataset,
            think=True,
            known_at_baseline={i.id for i in dataset},
            rehearsal_losses=[0.40, 0.47, 0.43],
        )
        config = MetaLearningConfig(
            name="meta-hinge", seeds=[1], checkpoint_interval=4, train_ratio=0.8,
            rehearsal_k=2, rehearsal_margin=0.02, think_mode="baseline",
            train_correct_items=True,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
                dataset, config, verbose=True
            )
        traj = result.trajectories[1]
        self.assertEqual({c["rehearsal_margin"] for c in model.joint_calls}, {0.02})
        # Margin 0.02: step 2 (0.47) and step 3 (0.43) both drift past 0.42.
        self.assertEqual(traj.train_rehearsal_active_steps, [4] * 8)
        self.assertEqual(traj.train_rehearsal_pairs, [6] * 8)
        self.assertIn(
            "rehearsal active in 67% of steps", traj.training_summary_text(12, 0.6)
        )
        self.assertIn("rehearsal active in 67% of steps", buf.getvalue())
        self.assertIn("weight: 1, margin: 0.02)", buf.getvalue())
        self.assertRegex(buf.getvalue(), r"rehearsal 0\.40→0\.43 \(active 4/6\)")
        cfg = self._config_json_for(traj.experiment_id)
        self.assertEqual(cfg["rehearsal_margin"], 0.02)
        self.assertEqual(cfg["rationale_max_tokens"], 512)
        self.assertEqual(MetaLearningConfig.from_dict(config.to_dict()).rehearsal_margin, 0.02)
        self.assertEqual(MetaLearningConfig.from_dict({**config.to_dict()}).rationale_max_tokens, 512)
        legacy = {k: v for k, v in config.to_dict().items() if k not in ("rehearsal_margin", "rationale_max_tokens")}
        self.assertEqual(MetaLearningConfig.from_dict(legacy).rehearsal_margin, 0.05)
        d = traj.to_dict()
        self.assertEqual(d["train_rehearsal_active_steps"], [4] * 8)
        self.assertEqual(d["train_rehearsal_pairs"], [6] * 8)
        path = self.tmp_path / "meta-hinge.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(loaded.trajectories[1].train_rehearsal_active_steps, [4] * 8)
        self.assertEqual(loaded.config.rehearsal_margin, 0.02)

    def test_verbose_output_shows_rehearsal_loss(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, known_at_baseline={i.id for i in dataset})
        config = EvaluationConfig(
            name="v-reh", train_ratio=0.5, rehearsal_k=1, train_correct_items=True
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            EvaluationHarness(model=model, db=self.db).run(dataset, config, verbose=True)
        out = buf.getvalue()
        self.assertIn("Rehearsal k: 1 (max tokens: 768, weight: 1, margin: 0.05)", out)
        self.assertIn("LoRA: rank 32, layers 24, scale 10", out)
        self.assertRegex(
            out,
            r"Trained \(3 steps, loss 6\.05 → 0\.58 "
            r"\(rehearsal 0\.50→0\.41 \(active 0/3\)\), \d+\.\ds",
        )
        self.assertIn(
            "mean final answer loss 0.58 (train loss 0.58, rehearsal 0.41); "
            "0 items hit the cap; 0 rationales missing; "
            "rehearsal active in 0% of steps",
            out,
        )
        self.assertIn("weight: 1, margin: 0.05)", out)

    def test_lora_model_kwargs_are_recorded(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset)
        kwargs = harness.lora_model_kwargs(rank=8, layers=4, scale=2.0)
        self.assertEqual(
            kwargs,
            {
                "num_lora_layers": 4,
                "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
            },
        )
        result = EvaluationHarness(model=model, db=self.db, model_kwargs=kwargs).run(
            dataset, EvaluationConfig(name="lora", train_ratio=0.5), verbose=False
        )
        cfg = self._config_json_for(self._latest_experiment_id())
        self.assertEqual(cfg["model_kwargs"], kwargs)
        self.assertEqual(cfg["lora"], {"rank": 8, "layers": 4, "scale": 2.0})
        self.assertEqual(result.model_kwargs, kwargs)
        self.assertEqual(result.lora_settings_text(), "LoRA: rank 8, layers 4, scale 2")
        self.assertEqual(result.to_dict()["lora"], {"rank": 8, "layers": 4, "scale": 2.0})
        # Defaults match StatefulLLM's when nothing was passed.
        self.assertEqual(harness.lora_settings({}), (32, 24, 10.0))
        self.assertEqual(harness.lora_model_kwargs(), harness.lora_model_kwargs(32, 24, 10.0))
        for bad in ({"rank": 0}, {"layers": -1}, {"scale": 0}):
            with self.assertRaises(ValueError):
                harness.lora_model_kwargs(**bad)
        # The lazy path hands the LoRA kwargs to StatefulLLM as-is.
        seen = {}
        original = harness.StatefulLLM
        harness.StatefulLLM = lambda **kw: seen.update(kw) or model
        try:
            EvaluationHarness(db=self.db, model_kwargs=kwargs).model
        finally:
            harness.StatefulLLM = original
        self.assertEqual(seen, kwargs)
        path = generate_html_report(result, str(self.tmp_path / "lora.html"))
        self.assertIn("LoRA: rank 8, layers 4, scale 2", pathlib.Path(path).read_text())

    def test_verbose_output(self):
        import contextlib
        import io

        dataset = make_dataset(4)
        model = FakeModel(dataset)
        config = EvaluationConfig(name="v", train_ratio=0.5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            EvaluationHarness(model=model, db=self.db).run(dataset, config, verbose=True)
        out = buf.getvalue()
        self.assertIn("Loss target: 0.6 (step cap: 12)", out)
        self.assertRegex(out, r"Trained \(3 steps, loss 6\.05 → 0\.58, \d+\.\ds\)")
        self.assertIn(
            "Training: mean 3.0 steps/item (cap 12, answer-loss target 0.60), "
            "mean final answer loss 0.58 (train loss 0.58); 0 items hit the cap; "
            "0 rationales missing",
            out,
        )

    def test_meta_records_and_prints_training_stats(self):
        import contextlib
        import io

        dataset = make_dataset(10)
        model = FakeModel(dataset)
        config = MetaLearningConfig(
            name="mlt", seeds=[7], checkpoint_interval=4, train_ratio=0.8,
            training_iterations=6,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = MetaLearningExperiment(
                model_factory=lambda: model, db=self.db
            ).run(dataset, config, verbose=True)
        out = buf.getvalue()
        traj = result.trajectories[7]
        self.assertEqual(model.train_kwargs, [(6, 0.6, 6)] * 8)
        self.assertEqual(traj.train_steps, [3] * 8)
        self.assertEqual(traj.train_cap_hits, 0)
        self.assertAlmostEqual(traj.mean_train_steps, 3.0)
        self.assertAlmostEqual(traj.mean_train_final_loss, 0.58)
        self.assertRegex(out, r"Trained q\d\d \(3 steps, loss 6\.05 → 0\.58, \d+\.\ds\)")
        self.assertIn(
            "Training: mean 3.0 steps/item (cap 6, answer-loss target 0.60), "
            "mean final answer loss 0.58 (train loss 0.58); 0 items hit the cap; "
            "0 rationales missing",
            out,
        )
        events = self.db.get_training_events_for_experiment(traj.experiment_id)
        self.assertEqual({e.training_iterations for e in events}, {3})
        self.assertEqual(self._config_json_for(traj.experiment_id)["loss_target"], 0.6)
        d = traj.to_dict()
        self.assertEqual(d["train_steps"], [3] * 8)
        self.assertAlmostEqual(d["mean_train_final_loss"], 0.58)
        # Round trip through save/load keeps the per-item stats.
        path = self.tmp_path / "mlt.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(loaded.trajectories[7].train_steps, [3] * 8)
        self.assertEqual(loaded.config.loss_target, 0.6)

    def test_report_header_shows_loss_target(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="rep", train_ratio=0.5), verbose=False
        )
        path = generate_html_report(result, str(self.tmp_path / "r.html"))
        text = pathlib.Path(path).read_text()
        self.assertIn("loss target: <code>0.60</code>", text)
        self.assertIn("Training: mean 3.0 steps/item (cap 12", text)
        self.assertIn("weight: <code>1</code>", text)
        self.assertIn("LoRA: rank 32, layers 24, scale 10", text)

    def test_report_shows_rehearsal_loss(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, known_at_baseline={i.id for i in dataset})
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(
                name="rep-reh", train_ratio=0.5, rehearsal_k=1, train_correct_items=True
            ),
            verbose=False,
        )
        path = generate_html_report(result, str(self.tmp_path / "r2.html"))
        text = pathlib.Path(path).read_text()
        self.assertIn("mean final answer loss 0.58 (train loss 0.58, rehearsal 0.41)", text)


class SkipCorrectTest(_TempDbTest):
    """train_correct_items=False skips train-split items already right at baseline."""

    def test_skip_is_default_and_trains_only_wrong_items(self):
        # 6 items, train_ratio 5/6 -> q00..q04 in the train split, q05 holdout.
        # q01 and q03 are right at baseline; q05 (holdout) too.
        dataset = make_dataset(6)
        model = FakeModel(dataset, known_at_baseline={"q01", "q03", "q05"})
        config = EvaluationConfig(name="skip", train_ratio=5 / 6)
        self.assertIs(config.train_correct_items, False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=True
            )
        by_id = {r.item_id: r for r in result.items}

        # Only the baseline-wrong train items reached the model.
        self.assertEqual(model.train_calls, 3)
        self.assertEqual(
            model.trained_targets, [f"Answer{i}{EOS}" for i in (0, 2, 4)]
        )
        self.assertEqual([r.item_id for r in result.train_items], ["q00", "q02", "q04"])
        # The skipped ones are their own bucket: not trained, not holdout.
        self.assertEqual(
            [r.item_id for r in result.skipped_correct_items], ["q01", "q03"]
        )
        self.assertEqual([r.item_id for r in result.holdout_items], ["q05"])
        for item_id in ("q01", "q03"):
            r = by_id[item_id]
            self.assertTrue(r.skipped_correct)
            self.assertFalse(r.was_trained)
            self.assertEqual(r.train_steps, 0)
            self.assertIsNone(r.train_final_loss)
            self.assertEqual(r.verify_attempts, 0)
            self.assertIsNone(r.verified)
            # Still re-inferred after training.
            self.assertIsNotNone(r.post_response)
            self.assertIs(r.post_has_key_terms, True)
        self.assertFalse(by_id["q05"].skipped_correct)
        self.assertFalse(by_id["q00"].skipped_correct)
        # Train metrics cover the trained (baseline-wrong) items only.
        self.assertEqual(result.train_improvement_rate, 1.0)
        self.assertEqual(result.train_retention_rate, 1.0)
        self.assertEqual(result.train_post_accuracy, 1.0)
        self.assertEqual(result.skipped_correct_count, 2)
        self.assertEqual(result.skipped_correct_regressed_count, 0)
        self.assertEqual(
            result.interference_summary_text(),
            "Skipped (baseline correct): 2; of which 0 regressed after other "
            "items' training",
        )
        self.assertEqual(result.holdout_summary_text(), "Holdout accuracy: 100.0% (1/1, baseline 1/1)")
        # One training event per trained item; none for the skipped ones.
        experiment_id = self._latest_experiment_id()
        self.assertEqual(
            len(self.db.get_training_events_for_experiment(experiment_id)), 3
        )
        cfg = self._config_json_for(experiment_id)
        self.assertIs(cfg["train_correct_items"], False)
        self.assertEqual(cfg["verify_steps"], 0)
        d = result.to_dict()
        self.assertIs(d["config"]["train_correct_items"], False)
        self.assertEqual(d["metrics"]["train_count"], 3)
        self.assertEqual(d["metrics"]["holdout_count"], 1)
        self.assertEqual(d["metrics"]["skipped_correct_count"], 2)
        self.assertEqual(d["metrics"]["skipped_correct_regressed_count"], 0)
        self.assertEqual(
            [i["item_id"] for i in d["items"] if i["skipped_correct"]], ["q01", "q03"]
        )
        out = buf.getvalue()
        self.assertIn("Train correct items: False; verify steps: 0", out)
        self.assertEqual(out.count("Skipped: baseline correct (not trained)"), 2)
        self.assertIn("✓ → ✓ (skipped: baseline correct)", out)
        self.assertIn("Skipped (baseline correct): 2; of which 0 regressed", out)
        self.assertNotIn("Verification:", out)
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "skip.html")
        ).read_text()
        self.assertIn("Train correct items: <code>False</code>", text)
        self.assertIn("Skipped (baseline correct): 2; of which 0 regressed", text)
        self.assertIn("2 items, 0 regressed", text)
        self.assertIn("<strong>Skipped:</strong> baseline already correct", text)

    def test_regression_among_skipped_items_counts_as_interference(self):
        dataset = make_dataset(6)
        # q03 is right at baseline but forgets it once anything is trained.
        model = FakeModel(
            dataset, known_at_baseline={"q01", "q03", "q05"}, regresses={"q03"}
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="interf", train_ratio=5 / 6), verbose=False
        )
        by_id = {r.item_id: r for r in result.items}
        self.assertTrue(by_id["q03"].skipped_correct)
        self.assertIs(by_id["q03"].post_has_key_terms, False)
        self.assertTrue(by_id["q03"].regressed)
        self.assertEqual(result.skipped_correct_count, 2)
        self.assertEqual(result.skipped_correct_regressed_count, 1)
        self.assertIn("of which 1 regressed", result.interference_summary_text())
        # The regression is not a train regression: retention stays 100%.
        self.assertEqual(result.train_retention_rate, 1.0)
        self.assertEqual(
            result.to_dict()["metrics"]["skipped_correct_regressed_count"], 1
        )
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "interf.html")
        ).read_text()
        self.assertIn("2 items, 1 regressed", text)

    def test_train_correct_items_true_restores_old_behaviour(self):
        dataset = make_dataset(6)
        model = FakeModel(dataset, known_at_baseline={"q01", "q03", "q05"})
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(name="all", train_ratio=5 / 6, train_correct_items=True),
            verbose=False,
        )
        self.assertEqual(model.train_calls, 5)
        self.assertEqual(len(result.train_items), 5)
        self.assertEqual(result.skipped_correct_items, [])
        self.assertEqual(result.skipped_correct_count, 0)
        self.assertEqual([r.item_id for r in result.holdout_items], ["q05"])
        self.assertIs(
            self._config_json_for(self._latest_experiment_id())["train_correct_items"],
            True,
        )
        self.assertIs(EvaluationConfig(train_correct_items=1).train_correct_items, True)

    def test_meta_skips_correct_items_and_reports_interference(self):
        dataset = make_dataset(10)
        known = {"q00", "q01", "q02", "q03"}
        model = FakeModel(dataset, known_at_baseline=known, regresses={"q01"})
        config = MetaLearningConfig(
            name="meta-skip", seeds=[1], checkpoint_interval=4, train_ratio=0.8
        )
        self.assertIs(config.train_correct_items, False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
                dataset, config, verbose=True
            )
        traj = result.trajectories[1]
        train_ids = traj_train_ids(dataset, seed=1)
        skipped = [i for i in train_ids if i in known]
        trained = [i for i in train_ids if i not in known]
        self.assertGreater(len(skipped), 0)
        # Skipped items never reach the model and are in no window.
        self.assertEqual(model.train_calls, len(trained))
        self.assertEqual(traj.skipped_correct_ids, skipped)
        self.assertEqual(traj.skipped_correct_count, len(skipped))
        self.assertEqual(len(traj.train_steps), len(trained))
        self.assertEqual(traj.checkpoints[-1].step, len(trained))
        window = [i for c in traj.checkpoints for i in c.window_ids]
        self.assertEqual(window, trained)
        self.assertEqual(
            [i for c in traj.checkpoints for i in c.skipped_correct_ids], skipped
        )
        # Not holdout either.
        self.assertEqual(traj.holdout_total, 2)
        # q01 was right, untrained, and broke: interference.
        expected_regressed = 1 if "q01" in skipped else 0
        self.assertEqual(traj.skipped_correct_regressed, expected_regressed)
        self.assertIn(
            f"Skipped (baseline correct): {len(skipped)}; of which "
            f"{expected_regressed} regressed",
            buf.getvalue(),
        )
        self.assertIn("Train correct items: False; verify steps: 0", buf.getvalue())
        self.assertIn("Skipped q0", buf.getvalue())
        cfg = self._config_json_for(traj.experiment_id)
        self.assertIs(cfg["train_correct_items"], False)
        self.assertEqual(cfg["verify_steps"], 0)
        # The skipped items got a post-training response row each.
        responses = self.db.get_responses_for_experiment(traj.experiment_id)
        post = [r for r in responses if r.phase == adaptible.Phase.POST_TRAINING]
        self.assertEqual(len(post), 2 + len(skipped))
        # Round trip.
        path = self.tmp_path / "meta-skip.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(loaded.trajectories[1].skipped_correct_ids, skipped)
        self.assertEqual(
            loaded.trajectories[1].skipped_correct_regressed, expected_regressed
        )
        self.assertIs(loaded.config.train_correct_items, False)
        self.assertEqual(
            loaded.trajectories[1].checkpoints[-1].skipped_correct_ids,
            traj.checkpoints[-1].skipped_correct_ids,
        )
        # Legacy files trained every item.
        legacy = {k: v for k, v in config.to_dict().items() if k != "train_correct_items"}
        self.assertIs(MetaLearningConfig.from_dict(legacy).train_correct_items, True)
        # With train_correct_items=True every train-split item is trained.
        model = FakeModel(dataset, known_at_baseline=known)
        traj = MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
            dataset,
            MetaLearningConfig(
                name="meta-all", seeds=[1], checkpoint_interval=4, train_ratio=0.8,
                train_correct_items=True,
            ),
            verbose=False,
        ).trajectories[1]
        self.assertEqual(model.train_calls, len(train_ids))
        self.assertEqual(traj.skipped_correct_ids, [])


class VerifyTest(_TempDbTest):
    """verify_steps: generate and judge after the target, train more while wrong."""

    def test_verify_loop_trains_until_the_answer_comes_out(self):
        # FAKE_LOSSES crosses the 0.6 target on step 3, but the model only
        # answers correctly once it has had 6 steps in total. With
        # verify_steps=2: call 1 (3 steps) -> check 1 wrong -> call 2 (2 steps,
        # 5 total) -> check 2 wrong -> call 3 (2 steps, 7 total) -> check 3 ok.
        dataset = make_dataset(3)
        model = FakeModel(dataset, learns_after_steps=6)
        config = EvaluationConfig(
            name="verify", train_ratio=1 / 3, verify_steps=2, training_iterations=12
        )
        self.assertEqual(config.verify_steps, 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=True
            )
        q00 = next(r for r in result.items if r.item_id == "q00")
        self.assertEqual(model.train_calls, 3)
        # Extra rounds halve the target (call 1 ended at 0.58 -> 0.29, then
        # 0.6/4 = 0.15 = VERIFY_LOSS_FLOOR) instead of training with none.
        self.assertEqual(model.train_kwargs, [(12, 0.6, 12), (2, 0.29, 2), (2, 0.15, 2)])
        self.assertEqual([s.steps for s in model.training_stats], [3, 2, 2])
        self.assertEqual(q00.train_steps, 7)
        self.assertEqual(q00.verify_attempts, 3)
        self.assertIs(q00.verified, True)
        self.assertFalse(q00.train_hit_cap)
        self.assertAlmostEqual(q00.train_initial_loss, 6.05)
        # Every check was a plain question generation: 3 baseline + 3 checks
        # + 3 post-training.
        self.assertEqual(len(model.question_prompts), 9)
        self.assertEqual(model.question_prompts.count(dataset[0].question), 5)
        # ...none of which was recorded as a response.
        experiment_id = self._latest_experiment_id()
        responses = self.db.get_responses_for_experiment(experiment_id)
        self.assertEqual(len(responses), 6)
        events = self.db.get_training_events_for_experiment(experiment_id)
        self.assertEqual([e.training_iterations for e in events], [7])
        self.assertEqual(result.verified_count, 1)
        self.assertEqual(result.verify_still_wrong_count, 0)
        self.assertAlmostEqual(result.mean_verify_attempts, 3.0)
        self.assertEqual(
            result.verification_summary_text(),
            "Verification: 1/1 items verified correct after training "
            "(mean 3.0 checks/item); 0 still wrong at the cap",
        )
        self.assertEqual(result.train_cap_hit_count, 0)
        self.assertEqual(self._config_json_for(experiment_id)["verify_steps"], 2)
        d = result.to_dict()
        self.assertEqual(d["config"]["verify_steps"], 2)
        self.assertEqual(d["metrics"]["verified_count"], 1)
        self.assertEqual(d["metrics"]["verify_still_wrong_count"], 0)
        self.assertAlmostEqual(d["metrics"]["mean_verify_attempts"], 3.0)
        item_d = next(i for i in d["items"] if i["item_id"] == "q00")
        self.assertEqual(item_d["verify_attempts"], 3)
        self.assertIs(item_d["verified"], True)
        self.assertEqual(item_d["train_steps"], 7)
        out = buf.getvalue()
        self.assertIn("Train correct items: False; verify steps: 2", out)
        self.assertRegex(
            # The fake replays its loss script per call, so the final loss is
            # the last call's second step.
            out,
            r"Trained \(7 steps, loss 6\.05 → 2\.10, verified ✓ \(3 checks\), \d+\.\ds\)",
        )
        self.assertIn(
            "Verification: 1/1 items verified correct after training "
            "(mean 3.0 checks/item); 0 still wrong at the cap",
            out,
        )
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "verify.html")
        ).read_text()
        self.assertIn("Verify steps: <code>2</code>", text)
        self.assertIn("Verification: 1/1 items verified correct", text)
        self.assertIn(
            "<strong>Verification:</strong> ✓ correct after 3 checks (7 steps total)",
            text,
        )

    def test_verify_stops_at_the_cap_and_reports_still_wrong(self):
        # Never answers correctly. Cap 6, verify_steps 2: 3 + 2 + 1 (clipped to
        # the remaining budget) steps and 3 checks, then still wrong.
        dataset = make_dataset(2)
        model = FakeModel(dataset, learns_after_steps=100)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(
                name="verify-cap", train_ratio=0.5, verify_steps=2, training_iterations=6
            ),
            verbose=False,
        )
        q00 = next(r for r in result.items if r.item_id == "q00")
        self.assertEqual(model.train_kwargs, [(6, 0.6, 6), (2, 0.29, 2), (1, 0.15, 1)])
        self.assertEqual(q00.train_steps, 6)
        self.assertEqual(q00.verify_attempts, 3)
        self.assertIs(q00.verified, False)
        self.assertTrue(q00.train_hit_cap)
        self.assertIs(q00.post_has_key_terms, False)
        self.assertEqual(result.verified_count, 0)
        self.assertEqual(result.verify_still_wrong_count, 1)
        self.assertEqual(result.train_cap_hit_count, 1)
        self.assertIn("1 items hit the cap", result.training_summary_text())
        self.assertEqual(
            result.verification_summary_text(),
            "Verification: 0/1 items verified correct after training "
            "(mean 3.0 checks/item); 1 still wrong at the cap",
        )
        self.assertEqual(
            [e.training_iterations for e in self.db.get_training_events_for_experiment(1)],
            [6],
        )
        text = pathlib.Path(
            generate_html_report(result, self.tmp_path / "verify-cap.html")
        ).read_text()
        self.assertIn("✗ still wrong at the step cap after 3 checks (6 steps total)", text)

        # An item already at the cap after the first call gets exactly one
        # check and no extra training.
        model = FakeModel(dataset, train_losses=[3.0, 2.0], learns_after_steps=100)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(
                name="verify-cap1", train_ratio=0.5, verify_steps=2, training_iterations=4
            ),
            verbose=False,
        )
        q00 = next(r for r in result.items if r.item_id == "q00")
        self.assertEqual(model.train_kwargs, [(4, 0.6, 4)])
        self.assertEqual((q00.train_steps, q00.verify_attempts, q00.verified), (4, 1, False))

        # A passing first check makes exactly one check and no extra call.
        model = FakeModel(dataset)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(name="verify-ok", train_ratio=0.5, verify_steps=2),
            verbose=False,
        )
        q00 = next(r for r in result.items if r.item_id == "q00")
        self.assertEqual(model.train_kwargs, [(12, 0.6, 12)])
        self.assertEqual((q00.train_steps, q00.verify_attempts, q00.verified), (3, 1, True))
        self.assertEqual(len(model.question_prompts), 2 * 2 + 1)

    def test_verify_off_makes_no_extra_generations(self):
        dataset = make_dataset(3)
        model = FakeModel(dataset, learns_after_steps=6)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="off", train_ratio=1 / 3), verbose=False
        )
        q00 = next(r for r in result.items if r.item_id == "q00")
        # Baseline + post-training only.
        self.assertEqual(len(model.question_prompts), 6)
        self.assertEqual(model.train_kwargs, [(12, 0.6, 12)])
        self.assertEqual(q00.train_steps, 3)
        self.assertEqual(q00.verify_attempts, 0)
        self.assertIsNone(q00.verified)
        self.assertFalse(q00.train_hit_cap)
        # Without verification the model never got its 6 steps.
        self.assertIs(q00.post_has_key_terms, False)
        self.assertEqual(result.verified_count, 0)
        self.assertEqual(result.verify_still_wrong_count, 0)
        self.assertEqual(result.mean_verify_attempts, 0.0)
        self.assertNotIn("Verification", pathlib.Path(
            generate_html_report(result, self.tmp_path / "off.html")
        ).read_text().split("<h2>Overall Metrics</h2>")[1].split("<div class=\"metrics-grid\">")[0])
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                EvaluationConfig(verify_steps=bad)
            with self.assertRaises(ValueError):
                MetaLearningConfig(verify_steps=bad)

    def test_verify_with_rehearsal_reuses_the_same_examples(self):
        # 4 items, all in the train split; q01..q03 are right at baseline and
        # form the rehearsal pool for q00, which needs 5 steps.
        dataset = make_dataset(4)
        model = FakeModel(
            dataset, known_at_baseline={"q01", "q02", "q03"}, learns_after_steps=5
        )
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset,
            EvaluationConfig(
                name="verify-reh", train_ratio=1.0, rehearsal_k=2, verify_steps=2
            ),
            verbose=False,
        )
        q00 = next(r for r in result.items if r.item_id == "q00")
        self.assertEqual(model.train_kwargs, [])
        self.assertEqual(
            [(c["loss_target"], c["max_steps"], c["rehearsal_k"]) for c in model.joint_calls],
            [(0.6, 12, 2), (0.29, 2, 2)],
        )
        # The same rehearsal rows went into both calls.
        self.assertEqual(
            model.joint_calls[0]["rehearsal_targets"],
            model.joint_calls[1]["rehearsal_targets"],
        )
        self.assertEqual((q00.train_steps, q00.verify_attempts, q00.verified), (5, 2, True))
        self.assertEqual(len(q00.rehearsal_item_ids), 2)
        self.assertEqual(q00.train_rehearsal_pairs, 10)
        # Last call's last (second) scripted rehearsal loss.
        self.assertAlmostEqual(q00.train_rehearsal_final_loss, 0.45)
        self.assertAlmostEqual(q00.train_rehearsal_initial_loss, 0.5)
        self.assertEqual(result.skipped_correct_count, 3)

    def test_meta_threads_verify_steps_and_records_per_item(self):
        dataset = make_dataset(10)
        model = FakeModel(dataset, learns_after_steps=4)
        config = MetaLearningConfig(
            name="meta-verify", seeds=[1], checkpoint_interval=4, train_ratio=0.8,
            verify_steps=3,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
                dataset, config, verbose=True
            )
        traj = result.trajectories[1]
        # First item: 3 steps -> wrong -> 3 more -> right (6 total, 2 checks).
        # Later items: the model already has its 4 steps, so 3 steps and one
        # check each.
        self.assertEqual(traj.train_steps, [6] + [3] * 7)
        self.assertEqual(traj.verify_attempts, [2] + [1] * 7)
        self.assertEqual(traj.verified, [True] * 8)
        self.assertEqual(traj.verified_count, 8)
        self.assertEqual(traj.verify_still_wrong_count, 0)
        self.assertAlmostEqual(traj.mean_verify_attempts, 9 / 8)
        self.assertEqual(traj.train_cap_hits, 0)
        self.assertEqual(
            traj.verification_summary_text(),
            "Verification: 8/8 items verified correct after training "
            "(mean 1.1 checks/item); 0 still wrong at the cap",
        )
        out = buf.getvalue()
        self.assertIn("Train correct items: False; verify steps: 3", out)
        self.assertIn("verified ✓ (2 checks)", out)
        self.assertIn("verified ✓ (1 check)", out)
        self.assertIn("Verification: 8/8 items verified correct", out)
        cfg = self._config_json_for(traj.experiment_id)
        self.assertEqual(cfg["verify_steps"], 3)
        self.assertEqual(MetaLearningConfig.from_dict(config.to_dict()).verify_steps, 3)
        legacy = {k: v for k, v in config.to_dict().items() if k != "verify_steps"}
        self.assertEqual(MetaLearningConfig.from_dict(legacy).verify_steps, 0)
        d = traj.to_dict()
        self.assertEqual(d["verify_attempts"], [2] + [1] * 7)
        self.assertEqual(d["verified_count"], 8)
        path = self.tmp_path / "meta-verify.json"
        result.save(path)
        loaded = MetaLearningResult.load(path)
        self.assertEqual(loaded.trajectories[1].verified, [True] * 8)
        self.assertEqual(loaded.trajectories[1].verify_attempts, [2] + [1] * 7)
        self.assertEqual(loaded.config.verify_steps, 3)
        # Off: nothing recorded as verified.
        model = FakeModel(dataset)
        traj = MetaLearningExperiment(model_factory=lambda: model, db=self.db).run(
            dataset,
            MetaLearningConfig(name="meta-off", seeds=[1], checkpoint_interval=4, train_ratio=0.8),
            verbose=False,
        ).trajectories[1]
        self.assertEqual(traj.verified, [None] * 8)
        self.assertEqual(traj.verify_attempts, [0] * 8)
        self.assertEqual(traj.verified_count, 0)


class CliFlagsTest(unittest.TestCase):
    """Both CLIs expose the new absl flags."""

    EXPECTED = (
        "--[no]train_correct_items",
        "--verify_steps",
        "--think_mode",
        "--rehearsal_k",
        "--rehearsal_max_tokens",
        "--rehearsal_weight",
        "--rehearsal_margin",
        "--rationale_max_tokens",
        "--learning_rate",
        "--loss_target",
        "--lora_rank",
        "--lora_layers",
        "--lora_scale",
        "--[no]close_think",
    )

    def _helpfull(self, *argv) -> str:
        proc = subprocess.run(
            [sys.executable, *argv, "--helpfull"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return proc.stdout + proc.stderr

    LORA_DEFAULTS = ("(default: '32')", "(default: '24')", "(default: '10.0')")
    HINGE_DEFAULTS = ("(default: '0.05')", "(default: '512')")

    def test_eval_cli_flags(self):
        text = self._helpfull("-m", "adaptible.eval")
        for flag in self.EXPECTED:
            self.assertIn(flag, text)
        self.assertIn("<none|empty|baseline|rationale>", text)
        self.assertIn("(default: 'rationale')", text)
        for default in self.LORA_DEFAULTS + self.HINGE_DEFAULTS:
            self.assertIn(default, text)

    def test_meta_cli_flags(self):
        text = self._helpfull("scripts/run_meta_experiment.py")
        for flag in self.EXPECTED:
            self.assertIn(flag, text)
        self.assertIn("<none|empty|baseline|rationale>", text)
        self.assertIn("(default: 'rationale')", text)
        for default in self.HINGE_DEFAULTS:
            self.assertIn(default, text)

    def test_think_mode_rationale_flag_value_is_accepted(self):
        # absl validates enum values at parse time; --think_mode rationale must
        # parse on both CLIs (a bad value fails before --helpfull prints).
        for argv in (("-m", "adaptible.eval"), ("scripts/run_meta_experiment.py",)):
            text = self._helpfull(*argv, "--think_mode", "rationale")
            self.assertIn("--think_mode", text)
            self.assertNotIn("FATAL", text)
            bad = self._helpfull(*argv, "--think_mode", "reasoning")
            self.assertIn("reasoning", bad)
            self.assertNotIn("--rehearsal_k", bad)
        for default in self.LORA_DEFAULTS:
            self.assertIn(default, text)


class LegacyLoadTest(_TempDbTest):
    """Result files written before the new fields existed still load."""

    def _legacy_checkpoint(self, step: int) -> dict:
        return {
            "step": step,
            "timestamp": "t",
            "baseline_correct": 5,
            "baseline_total": 10,
            "post_correct": 3,
            "post_total": step,
            "improved": 1,
            "retained": 2,
            "regressed": 0,
            "stuck": step - 3,
        }

    def test_load_legacy_format(self):
        data = {
            "config": {
                "name": "old",
                "seeds": [42],
                "checkpoint_interval": 10,
                "training_iterations": 25,
                "train_ratio": 0.8,
            },
            "dataset_name": "trivia",
            "timestamp": "2025-01-01T00:00:00",
            "trajectories": {
                "42": {
                    "total_time_seconds": 1.0,
                    "checkpoints": [self._legacy_checkpoint(s) for s in (10, 20, 30)],
                }
            },
        }
        path = self.tmp_path / "old.json"
        path.write_text(json.dumps(data))
        result = MetaLearningResult.load(path)

        self.assertEqual(result.config.training_source, "ground_truth")
        self.assertEqual(result.config.repeats, 1)
        traj = result.trajectories[42]
        self.assertEqual(traj.repeat, 0)
        self.assertIsNone(traj.holdout_accuracy)
        self.assertEqual(traj.window_sizes, [0, 0, 0])
        self.assertIsNone(traj.meta_learning_score)
        self.assertIn("below 5", traj.meta_learning_score_reason)
        self.assertEqual(traj.checkpoints[-1].improvement_rate, 1 / 28)
        self.assertIsNone(result.within_seed_variance)

    def test_load_shipped_results_file(self):
        shipped = (
            pathlib.Path(__file__).parents[2]
            / "outputs"
            / "meta"
            / "meta_experiment.json"
        )
        if not shipped.exists():
            self.skipTest("outputs/meta/meta_experiment.json not present")
        result = MetaLearningResult.load(shipped)
        self.assertEqual(sorted(result.trajectories), [42, 123, 456])
        self.assertEqual(result.config.training_source, "ground_truth")
        for traj in result.trajectories.values():
            self.assertGreater(len(traj.checkpoints), 0)
            self.assertIsNone(traj.holdout_accuracy)


if __name__ == "__main__":
    unittest.main()
