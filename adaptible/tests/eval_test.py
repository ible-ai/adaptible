"""Model-free tests for the evaluation harness and meta-learning experiment.

A ``FakeModel`` stands in for ``StatefulLLM``. It answers trivia from a
script, produces a scripted revision when handed a revision prompt, and
"learns" an item by decoding the masked training target it is trained on.
The tokenizer is a one-character-per-token fake so training examples built
by ``make_collated_training_example`` can be decoded back to text.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import warnings
from typing import Callable

import adaptible

eval_mod = adaptible.eval
harness = adaptible._src.eval.harness
meta = adaptible._src.eval.meta

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
REPO_ROOT = pathlib.Path(__file__).parents[2]


def fake_think(item: TriviaItem) -> str:
    """The reasoning FakeModel(think=True) emits before ``</think>``."""
    return f"Thinking about {item.id}."


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
    """

    def __init__(
        self,
        dataset: TriviaDataset,
        learns_after: int = 0,
        revision: str | Callable[[TriviaItem], str] | dict[str, str] = "valid",
        known_at_baseline: set[str] | None = None,
        think: bool = False,
    ):
        self._tokenizer = FakeTokenizer(think=think)
        self._think = think
        self._max_tokens = 4096
        self._model_is_stable = True
        self._by_question = {item.question: item for item in dataset}
        self._by_id = {item.id: item for item in dataset}
        self._learns_after = learns_after
        self._revision = revision
        self.learned: set[str] = set(known_at_baseline or set())
        self.train_calls = 0
        self.trained_targets: list[str] = []
        # Every row of every batch: (decoded masked target, decoded full sequence).
        self.trained_batches: list[list[tuple[str, str]]] = []
        self.batch_shapes: list[tuple[int, int]] = []
        self.revision_prompts: list[str] = []
        self.question_prompts: list[str] = []

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
            return f"[[0]] The answer is {item.correct_answer}. [[/0]]"
        self.question_prompts.append(prompt)
        item = self._by_question[prompt]
        answer = f"It is {item.correct_answer}." if item.id in self.learned else DONT_KNOW
        if self._think:
            return f"{fake_think(item)}\n</think>\n\n{answer}"
        return answer

    def train_on_example(self, example, iterations: int = 25, **kwargs) -> None:
        del iterations, kwargs
        self.batch_shapes.append(tuple(example.mask.shape))
        rows = []
        for inputs, labels, mask in zip(
            example.input.tolist(), example.label.tolist(), example.mask.tolist()
        ):
            masked = self._tokenizer.decode(t for t, m in zip(labels, mask) if m)
            full = self._tokenizer.decode([inputs[0]] + labels)
            rows.append((masked, full))
        self.trained_batches.append(rows)
        target = rows[0][0]
        self.trained_targets.append(target)
        self.train_calls += 1
        if self.train_calls <= self._learns_after:
            return
        for item in self._by_id.values():
            if item.correct_answer.lower() in target.lower():
                self.learned.add(item.id)


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

    def test_think_mode_baseline_is_default_and_keeps_reasoning_unmasked(self):
        dataset = make_dataset(4)
        for source in ("ground_truth", "self_generated"):
            model = FakeModel(dataset, think=True)
            config = EvaluationConfig(name=f"tm-{source}", training_source=source)
            self.assertEqual(config.think_mode, "baseline")
            self.assertIs(config.close_think, True)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=False
            )
            self.assertEqual(len(model.trained_batches), 3)
            body = "{answer}" if source == "ground_truth" else "The answer is {answer}."
            for item, (masked, full) in zip(
                dataset.items[:3], (b[0] for b in model.trained_batches)
            ):
                target = f"{body.format(answer=item.correct_answer)}{EOS}"
                # Only the corrected answer is in the loss...
                self.assertEqual(masked, target)
                # ...and the model's own reasoning sits in the unmasked prefix.
                self.assertEqual(
                    full,
                    f"<user>{item.question}</user><assistant><think>\n"
                    f"{fake_think(item)}\n</think>\n\n{target}",
                )
            self.assertEqual(result.train_post_accuracy, 1.0)
            d = result.to_dict()["config"]
            self.assertEqual(d["think_mode"], "baseline")
            self.assertIs(d["close_think"], True)
            self.assertEqual(d["rehearsal_k"], 0)
            cfg = self._config_json_for(self._latest_experiment_id())
            self.assertEqual(cfg["think_mode"], "baseline")
            self.assertEqual(cfg["rehearsal_k"], 0)
            self.assertEqual(cfg["model_kwargs"], {})
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
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"</think>\n\n{item.correct_answer}{EOS}")
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
            dataset, EvaluationConfig(name="fallback"), verbose=False
        )
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"</think>\n\n{item.correct_answer}{EOS}")

    def test_think_mode_noop_without_think_template(self):
        dataset = make_dataset(4)
        for mode in ("none", "empty", "baseline"):
            model = FakeModel(dataset)  # template ends in <assistant>
            EvaluationHarness(model=model, db=self.db).run(
                dataset, EvaluationConfig(name="plain", think_mode=mode), verbose=False
            )
            for item, target in zip(dataset.items[:3], model.trained_targets):
                self.assertEqual(target, f"{item.correct_answer}{EOS}")

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
            name="sg-quality", train_ratio=0.8, training_source="self_generated"
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
            name="reh", seeds=[1], checkpoint_interval=4, train_ratio=0.8, rehearsal_k=2
        )
        experiment = MetaLearningExperiment(
            model_factory=lambda: model, db=self.db, model_kwargs={"learning_rate": 3e-5}
        )
        result = experiment.run(dataset, config, verbose=False)
        traj = result.trajectories[1]
        self.assertEqual(len(model.batch_shapes), 8)
        self.assertTrue(all(shape[0] == 3 for shape in model.batch_shapes))
        holdout_questions = {
            dataset[idx].question for idx in traj_holdout_indices(dataset, seed=1)
        }
        for rows in model.trained_batches:
            for masked, full in rows[1:]:
                q = full.split("<user>")[1].split("</user>")[0]
                self.assertNotIn(q, holdout_questions)
                self.assertRegex(masked, r"^Thinking about q\d\d\.\n</think>\n\nIt is Answer\d+\.<eos>$")
        cfg = self._config_json_for(traj.experiment_id)
        self.assertEqual(cfg["rehearsal_k"], 2)
        self.assertEqual(cfg["model_kwargs"], {"learning_rate": 3e-5})

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


class RehearsalTest(_TempDbTest):
    """rehearsal_k batches self-distillation rows with each correction."""

    def test_rehearsal_batch_shape_rows_and_padding(self):
        # 6 items, train_ratio 5/6 -> q00..q04 trained, q05 holdout.
        # q01, q03 and q05 are known at baseline: the pool is {q01, q03} (q05 is
        # holdout and must never be rehearsed).
        dataset = make_dataset(6)
        model = FakeModel(dataset, think=True, known_at_baseline={"q01", "q03", "q05"})
        config = EvaluationConfig(name="reh", train_ratio=5 / 6, rehearsal_k=2, seed=7)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        by_id = {item.item_id: item for item in result.items}

        # One train_on_example call per trained item.
        self.assertEqual(model.train_calls, 5)
        self.assertEqual(len(model.batch_shapes), 5)
        for i, (item, rows, shape) in enumerate(
            zip(dataset.items[:5], model.trained_batches, model.batch_shapes)
        ):
            pool = [pid for pid in ("q01", "q03") if pid != item.id]
            self.assertEqual(shape[0], 1 + len(pool))  # (1+k, L), k capped at the pool
            # Row 0 is the correction with the model's own reasoning unmasked.
            masked, full = rows[0]
            self.assertEqual(masked, f"{item.correct_answer}{EOS}")
            self.assertIn(f"{fake_think(item)}\n</think>\n\n", full)
            # The other rows are correct-baseline items, never the item itself,
            # trained on their own full baseline output.
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
            self.assertEqual(sorted(by_id[item.id].rehearsal_item_ids), sorted(pool))
        # Holdout items are untouched and have no rehearsal ids.
        self.assertEqual(by_id["q05"].rehearsal_item_ids, [])
        self.assertNotIn("q05", {rid for r in result.items for rid in r.rehearsal_item_ids})
        d = result.to_dict()
        self.assertEqual(d["config"]["rehearsal_k"], 2)
        self.assertEqual(
            self._config_json_for(self._latest_experiment_id())["rehearsal_k"], 2
        )
        self.assertTrue(
            any(i["rehearsal_item_ids"] for i in d["items"] if i["was_trained"])
        )

    def test_rehearsal_padding_has_zero_mask(self):
        # Rows differ in length, so the collated batch is padded; padded
        # positions carry mask 0 and the pad id (eos here, since the fake
        # tokenizer has no pad id).
        dataset = make_dataset(3)
        model = FakeModel(dataset, think=True, known_at_baseline={"q01"})
        tok = model._tokenizer
        tok.eos_token_id = 99999
        captured = []
        original = model.train_on_example
        model.train_on_example = lambda ex, **kw: (captured.append(ex), original(ex, **kw))
        EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="pad", train_ratio=1.0, rehearsal_k=1),
            verbose=False,
        )
        self.assertEqual(len(captured), 3)
        for ex in captured:
            n_rows, length = ex.mask.shape
            self.assertEqual(ex.input.shape, (n_rows, length))
            self.assertEqual(ex.label.shape, (n_rows, length))
            masks = ex.mask.tolist()
            labels = ex.label.tolist()
            inputs = ex.input.tolist()
            lengths = [
                max(i for i, t in enumerate(row) if t != 99999) + 1 for row in inputs
            ]
            self.assertTrue(any(l < length for l in lengths) or n_rows == 1)
            for row_mask, row_label, row_len in zip(masks, labels, lengths):
                if row_len < length:
                    self.assertEqual(row_mask[row_len:], [0] * (length - row_len))
                    self.assertEqual(row_label[row_len:], [99999] * (length - row_len))
                # The last real label of every row is <eos> (masked in).
                self.assertEqual(row_mask[row_len - 1], 1)

    def test_rehearsal_with_empty_pool_trains_single_row(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, think=True)  # nothing correct at baseline
        EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="nopool", rehearsal_k=3), verbose=False
        )
        self.assertTrue(all(shape[0] == 1 for shape in model.batch_shapes))

    def test_rehearsal_sampling_is_seeded(self):
        dataset = make_dataset(12)
        known = {f"q{i:02d}" for i in range(12)}
        ids = []
        for _ in range(2):
            model = FakeModel(dataset, think=True, known_at_baseline=known)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, EvaluationConfig(name="seeded", rehearsal_k=2, seed=3),
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


class CliFlagsTest(unittest.TestCase):
    """Both CLIs expose the new absl flags."""

    EXPECTED = ("--think_mode", "--rehearsal_k", "--learning_rate", "--[no]close_think")

    def _helpfull(self, *argv) -> str:
        proc = subprocess.run(
            [sys.executable, *argv, "--helpfull"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return proc.stdout + proc.stderr

    def test_eval_cli_flags(self):
        text = self._helpfull("-m", "adaptible.eval")
        for flag in self.EXPECTED:
            self.assertIn(flag, text)
        self.assertIn("<none|empty|baseline>", text)

    def test_meta_cli_flags(self):
        text = self._helpfull("scripts/run_meta_experiment.py")
        for flag in self.EXPECTED:
            self.assertIn(flag, text)
        self.assertIn("<none|empty|baseline>", text)


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
