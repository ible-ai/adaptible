"""Model-free tests for the evaluation harness and meta-learning experiment.

A ``FakeModel`` stands in for ``StatefulLLM``. It answers trivia from a
script, produces a scripted revision when handed a revision prompt, and
"learns" an item by decoding the masked training target it is trained on.
The tokenizer is a one-character-per-token fake so training examples built
by ``make_collated_training_example`` can be decoded back to text.
"""

import json
import pathlib
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
generate_html_report = eval_mod.generate_html_report

EOS = "<eos>"
DONT_KNOW = "I do not know."


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
            "invalid" -> text with no [[X]] markers; or a callable
            ``(item) -> str`` for custom revisions.
        known_at_baseline: Item ids answered correctly before any training.
        think: Give the tokenizer a generation prompt ending in ``<think>\n``.
    """

    def __init__(
        self,
        dataset: TriviaDataset,
        learns_after: int = 0,
        revision: str | Callable[[TriviaItem], str] = "valid",
        known_at_baseline: set[str] | None = None,
        think: bool = False,
    ):
        self._tokenizer = FakeTokenizer(think=think)
        self._max_tokens = 4096
        self._model_is_stable = True
        self._by_question = {item.question: item for item in dataset}
        self._by_id = {item.id: item for item in dataset}
        self._learns_after = learns_after
        self._revision = revision
        self.learned: set[str] = set(known_at_baseline or set())
        self.train_calls = 0
        self.trained_targets: list[str] = []
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
            if callable(self._revision):
                return self._revision(item)
            if self._revision == "invalid":
                return "Here is a revision with no markers at all."
            return f"[[0]] The answer is {item.correct_answer}. [[/0]]"
        self.question_prompts.append(prompt)
        item = self._by_question[prompt]
        if item.id in self.learned:
            return f"It is {item.correct_answer}."
        return DONT_KNOW

    def train_on_example(self, example, iterations: int = 25, **kwargs) -> None:
        del iterations, kwargs
        labels = example.label.tolist()[0]
        mask = example.mask.tolist()[0]
        target = self._tokenizer.decode(t for t, m in zip(labels, mask) if m)
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

    def test_close_think_closes_open_think_block(self):
        dataset = make_dataset(4)
        for source in ("ground_truth", "self_generated"):
            model = FakeModel(dataset, think=True)
            config = EvaluationConfig(name=f"ct-{source}", training_source=source)
            result = EvaluationHarness(model=model, db=self.db).run(
                dataset, config, verbose=False
            )
            self.assertEqual(len(model.trained_targets), 3)
            body = "{answer}" if source == "ground_truth" else "The answer is {answer}."
            for item, target in zip(dataset.items[:3], model.trained_targets):
                self.assertEqual(
                    target,
                    f"</think>\n\n{body.format(answer=item.correct_answer)}{EOS}",
                )
            self.assertEqual(result.train_post_accuracy, 1.0)
            self.assertIs(result.to_dict()["config"]["close_think"], True)
            self.assertIs(
                self._config_json_for(self._latest_experiment_id())["close_think"], True
            )
            text = pathlib.Path(
                generate_html_report(result, self.tmp_path / f"{source}-ct.html")
            ).read_text()
            self.assertIn("Close think: <code>True</code>", text)

    def test_noclose_think_reproduces_old_target(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset, think=True)
        config = EvaluationConfig(name="noct", close_think=False)
        result = EvaluationHarness(model=model, db=self.db).run(
            dataset, config, verbose=False
        )
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"{item.correct_answer}{EOS}")
        self.assertIs(result.to_dict()["config"]["close_think"], False)
        self.assertIs(
            self._config_json_for(self._latest_experiment_id())["close_think"], False
        )

    def test_close_think_noop_without_think_template(self):
        dataset = make_dataset(4)
        model = FakeModel(dataset)  # template ends in <assistant>
        EvaluationHarness(model=model, db=self.db).run(
            dataset, EvaluationConfig(name="plain"), verbose=False
        )
        for item, target in zip(dataset.items[:3], model.trained_targets):
            self.assertEqual(target, f"{item.correct_answer}{EOS}")

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

    def test_meta_close_think_threads_through(self):
        dataset = make_dataset(10)
        for close_think in (True, False):
            model = FakeModel(dataset, think=True)
            config = MetaLearningConfig(
                name=f"ct{close_think}",
                seeds=[1],
                checkpoint_interval=4,
                train_ratio=0.8,
                close_think=close_think,
            )
            result = self._run(dataset, config, factory=lambda: model)
            traj = result.trajectories[1]
            self.assertEqual(len(model.trained_targets), 8)
            for target in model.trained_targets:
                self.assertEqual(target.startswith("</think>\n\n"), close_think)
            self.assertIs(self._config_json_for(traj.experiment_id)["close_think"], close_think)
            self.assertIs(result.to_dict()["config"]["close_think"], close_think)
            self.assertIs(
                MetaLearningConfig.from_dict(config.to_dict()).close_think, close_think
            )

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
