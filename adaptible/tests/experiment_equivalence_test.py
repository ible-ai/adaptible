"""Fixed input, identical output: the port against the experiment's own code.

Self-consistency within a runtime proves nothing about whether the port
measures what the experiment measures. What settles that is running the
*original's* functions and the port's functions over the same fixed inputs and
requiring the same answers.

`scripts/cycles_mlx.py` constructs a model at import, so its definitions are
lifted out of the source with `ast` and executed in isolation. That is
deliberate: the comparison has to be against the experiment's real code, not
against a copy of it in this file, or the test only proves I can copy.

Generation itself is excluded. Q4_K_M and bf16 are different arithmetic, so a
greedy decode cannot be expected to match across runtimes; only the components
that are pure functions of text can be held to exact equality.
"""

import ast
import re
import sys
import unittest
from pathlib import Path

from adaptible._src.eval.harness import contains_key_terms
from adaptible.eval import generate_default_dataset

sys.path.insert(0, "scripts")

ORIGINAL = Path(__file__).resolve().parents[2] / "scripts" / "cycles_mlx.py"
WANTED = ("closed", "answer_of", "ok", "clean")


def original_functions():
    """Execute the experiment's own definitions, without its model."""
    tree = ast.parse(ORIGINAL.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in WANTED
    ]
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ("WRONG", "PARA") for t in node.targets
        )
    ]
    namespace = {"re": re, "contains_key_terms": contains_key_terms}
    exec(  # noqa: S102 - the experiment's own source is the reference
        compile(
            ast.Module(body=assignments + wanted, type_ignores=[]),
            str(ORIGINAL),
            "exec",
        ),
        namespace,
    )
    missing = [name for name in WANTED if name not in namespace]
    if missing:
        raise AssertionError(f"cycles_mlx.py no longer defines {missing}")
    return namespace


ITEMS = {item.id: item for item in generate_default_dataset()}

# Fixed inputs covering every branch the judging path can take.
TEXTS = [
    "<think>\nThe note says Canberra.\n</think>\nThe capital of Australia is Canberra.",
    "<think>\nunterminated reasoning about Canberra",
    "The capital of Australia is Canberra.",
    "The capital of Australia is Sydney.",
    "<think>\nx\n</think>\nCanberra.",
    "<think>\nx\n</think>\n",
    "",
    "<think>\nCanberra\n</think>\n" + "Canberra is the capital. " * 60,
]
STAR_TEXTS = [
    "<think>\nx\n</think>\nThe nearest star to Earth is the Sun.",
    "<think>\nx\n</think>\nThe nearest star is Proxima Centauri.",
    "<think>\nx\n</think>\nThe Sun, though Proxima Centauri is nearby.",
    "<think>\nx\n</think>\nThe nearest star is Alpha Centauri, not the Sun.",
]


class JudgeEquivalenceTest(unittest.TestCase):
    """The port's judge must agree with the experiment's on every fixed input."""

    @classmethod
    def setUpClass(cls):
        import wrapper_cycles

        cls.original = original_functions()
        # Held as a module: a bare function assigned to a class attribute
        # would be bound as a method and receive `self` as its first argument.
        cls.port = wrapper_cycles

    def test_answer_extraction_agrees(self):
        for text in TEXTS + STAR_TEXTS:
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    self.port.answer_of(text),
                    self.original["answer_of"](text),
                    "answer_of diverged",
                )

    def test_closedness_agrees_wherever_the_experiment_applies(self):
        """The experiment's model always frames its thought; a runtime may not.

        Where a think block is present the two must agree exactly. A bare
        answer with no markers is the one case the port must handle and the
        experiment never sees, so it is asserted separately below.
        """
        for text in TEXTS + STAR_TEXTS:
            if "<think>" not in text:
                continue
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    self.port.closed(text),
                    self.original["closed"](text),
                    "closed diverged on a framed thought",
                )

    def test_a_bare_answer_is_the_ports_documented_extension(self):
        self.assertTrue(self.port.closed("Canberra."))
        self.assertFalse(self.original["closed"]("Canberra."))

    def test_judging_agrees_on_framed_answers(self):
        item = ITEMS["geo_001"]
        for text in TEXTS:
            if "<think>" not in text:
                continue
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    self.port.ok(item, text),
                    self.original["ok"](item, text),
                    "ok diverged",
                )

    def test_known_wrong_entities_agree(self):
        """sci_017 must not be credited for naming Proxima or Alpha Centauri."""
        item = ITEMS["sci_017"]
        for text in STAR_TEXTS:
            with self.subTest(text=text[:46]):
                self.assertEqual(
                    self.port.ok(item, text),
                    self.original["ok"](item, text),
                    "WRONG handling diverged",
                )


class CandidateAcceptanceEquivalenceTest(unittest.TestCase):
    """`clean` decides which samples may be trained on; it must agree exactly."""

    @classmethod
    def setUpClass(cls):
        cls.original = original_functions()

    def port_clean(self, item, text):
        """The acceptance the port actually applies, called, not restated.

        This used to reimplement `flagship_candidate`'s three filters here in
        the test. That is worth nothing: the port was missing the known-wrong
        clause entirely and this test passed anyway, because both sides of the
        comparison had it. It now calls `repair.flagship_clean`, which is the
        function `flagship_candidate` runs, and splits the thought off with the
        port's own `answer`.
        """
        from adaptible._src.wrap.repair import answer, flagship_clean

        if "</think>" not in text:
            return False
        return (
            flagship_clean(
                answer(text),
                item.correct_answer,
                self.original["WRONG"].get(item.id, []),
            )
            is not None
        )

    def test_candidate_acceptance_agrees(self):
        for item_id, texts in (("geo_001", TEXTS), ("sci_017", STAR_TEXTS)):
            item = ITEMS[item_id]
            for text in texts:
                with self.subTest(item=item_id, text=text[:40]):
                    self.assertEqual(
                        self.port_clean(item, text),
                        self.original["clean"](item, text),
                        "clean diverged",
                    )

    def test_a_known_wrong_entity_is_rejected_by_the_port_itself(self):
        """The case the reimplementation hid, pinned against the original.

        `sci_017`'s note is the Sun and its known-wrong entities are Proxima
        Centauri and Alpha Centauri. A first sentence naming both the right and
        a wrong one satisfies every other clause, so if the port ever drops the
        exclusion again this fails on its own, without needing the paired
        comparison above to happen to cover it.
        """
        item = ITEMS["sci_017"]
        text = (
            "Thinking about it.</think>The closest star to Earth is the Sun, "
            "not Proxima Centauri. It is about 150 million kilometres away."
        )
        self.assertFalse(self.original["clean"](item, text))
        self.assertFalse(self.port_clean(item, text))


class TrainingTargetEquivalenceTest(unittest.TestCase):
    """The trained text must be the one `example_from` would have built."""

    def port_target(self, text):
        from adaptible._src.wrap.repair import _FLAGSHIP_ANSWER_SENTENCES

        body = text.split("</think>")[-1].strip()
        sentences = re.split(r"(?<=[.!?])\s", body)
        return (
            " ".join(sentences[:_FLAGSHIP_ANSWER_SENTENCES]).replace("\n", " ").strip()
        )

    def original_target(self, text):
        """`example_from`'s answer, lifted verbatim from the experiment."""
        source = ORIGINAL.read_text()
        line = next(
            text for text in source.splitlines() if text.strip().startswith("ans =")
        )
        namespace = {
            "re": re,
            "text": text,
            "answer_of": lambda r: r.split("</think>")[-1].strip(),
        }
        exec(line.strip(), namespace)  # noqa: S102 - the experiment's own line
        return namespace["ans"]

    def test_training_target_agrees(self):
        for text in TEXTS + STAR_TEXTS:
            if "</think>" not in text:
                continue
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    self.port_target(text),
                    self.original_target(text),
                    "training target diverged",
                )


class StopRuleEquivalenceTest(unittest.TestCase):
    """The step count, driven through both real loops rather than read.

    The experiment applies an optimizer step, records the loss computed
    *before* it, and stops when that loss is strictly below the floor, so it
    always takes one update past the crossing. The wrapper's default measures
    at the current weights and breaks before updating, with `<=`. Both run the
    quoted recipe -- four steps, 0.15 on the answer tokens -- and end on
    different adapters, which is why this is driven rather than inspected.
    """

    TRAJECTORY = [0.62, 0.41, 0.18, 0.09, 0.05]
    FLOOR, CAP = 0.15, 4

    def original_steps(self):
        from adaptible._src._llm import run_training_steps

        losses = iter(self.TRAJECTORY)
        stats = run_training_steps(lambda: next(losses), self.CAP, self.FLOOR)
        return stats.steps, stats.final_loss

    def port_steps(self, stop_after_update):
        import types

        import torch

        from adaptible._src.wrap.training_budget import fit_masked_target

        trajectory = self.TRAJECTORY

        class Scripted(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))
                self.i = 0

            def train(self, mode=True):
                return self

            def forward(self, input_ids=None, labels=None, **kwargs):
                loss = self.w.sum() + trajectory[min(self.i, len(trajectory) - 1)]
                self.i += 1
                return types.SimpleNamespace(loss=loss, logits=None)

        model = Scripted()
        zeros = torch.zeros(1, 2, dtype=torch.long)
        stats = fit_masked_target(
            model,
            zeros,
            zeros,
            torch.optim.SGD(model.parameters(), lr=0.0),
            max_steps=self.CAP,
            target_loss=self.FLOOR,
            stop_after_update=stop_after_update,
        )
        return stats["steps"], stats["final_loss"]

    def test_the_flagship_stop_rule_matches_the_experiment(self):
        steps, loss = self.port_steps(True)
        original_steps, original_loss = self.original_steps()
        self.assertEqual(steps, original_steps)
        # The loss is the one measured before the last update in both. It is
        # compared loosely only because the port reads it back out of a float32
        # tensor; the step count is the exact claim.
        self.assertAlmostEqual(loss, original_loss, places=5)

    def test_the_served_default_is_the_one_that_differs(self):
        """Pins the difference, so the default cannot drift into the flagship
        path unnoticed and so removing `stop_after_update` fails loudly."""
        self.assertEqual(self.port_steps(False)[0], self.original_steps()[0] - 1)


class OptimizerEquivalenceTest(unittest.TestCase):
    """The flagship recipe must train with the experiment's optimizer.

    `mlx.optimizers.AdamW` applies no bias correction and couples its decoupled
    weight decay to the parameter before the moment update. `MLXAdamW`
    reproduces that; stock `torch.optim.AdamW` does not, and its first four
    steps are 0.32x, 0.24x, 0.20x and 0.18x the size of MLX's. Which one the
    trainer picks is keyed on `training_options["thinking"]`, which is derived
    from the model's detected mode -- so without this the flagship recipe ran
    the experiment's optimizer only by luck of detection.
    """

    def test_the_flagship_path_pins_the_mlx_matched_optimizer(self):
        """Driven, not read: the trainer's own selector on the flag the
        flagship path sets, against the selector on the flag it would have
        had if detection said this model does not reason."""
        from adaptible._src.wrap.training_optimizer import (
            MLXAdamW,
            training_optimizer,
        )

        import torch

        parameter = [torch.nn.Parameter(torch.zeros(2))]
        self.assertIsInstance(
            training_optimizer(parameter, thinking=True), MLXAdamW
        )
        self.assertNotIsInstance(
            training_optimizer(parameter, thinking=False), MLXAdamW
        )

        # And the flagship path sets it regardless of detection.
        import inspect

        from adaptible._src.wrap import repair

        source = inspect.getsource(repair.Controller.repair)
        self.assertIn("thinking or self.flagship_recipe", source)

    def test_the_two_optimizers_are_not_interchangeable(self):
        """Pins the difference, so the fallback cannot quietly become harmless."""
        import torch

        from adaptible._src.wrap.training_optimizer import MLXAdamW

        def walk(make):
            p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
            opt = make([p])
            for _ in range(4):
                p.grad = torch.full((4,), 0.1, dtype=torch.float32)
                opt.step()
            return float(p.detach().abs().sum())

        mlx_like = walk(lambda ps: MLXAdamW(ps, lr=2e-5, weight_decay=0.01))
        stock = walk(lambda ps: torch.optim.AdamW(ps, lr=2e-5, weight_decay=0.0))
        self.assertNotAlmostEqual(mlx_like, stock, places=7)


class InitialAdapterTest(unittest.TestCase):
    """A wrapper can start its first candidate from the original's own draw.

    The original draws its LoRA initialisation from `mx.random` unseeded, so it
    does not reproduce its own runs; measured, two MLX runs of one training call
    started from different adapters. Reproducing one particular MLX cycle end
    to end therefore needs the wrapper to start from the draw that run used.
    Given the same starting tensors, the two trainers agree to 2.6e-6 after four
    updates; given independent draws, to only 5.7e-3.
    """

    def controller(self, initial_adapter):
        from adaptible._src.wrap.repair import Controller

        controller = Controller.__new__(Controller)
        controller.initial_adapter = (
            Path(initial_adapter) if initial_adapter else None
        )
        return controller

    def test_the_first_candidate_starts_from_the_given_draw(self):
        c = self.controller("/runs/mlx-init")
        self.assertEqual(c.training_start(None), "/runs/mlx-init")

    def test_without_one_the_first_candidate_starts_fresh(self):
        self.assertIsNone(self.controller(None).training_start(None))

    def test_an_accepted_adapter_always_wins(self):
        """A later repair continues from what is served, never back to the
        initial draw -- that would discard an accepted correction."""
        c = self.controller("/runs/mlx-init")
        self.assertEqual(
            c.training_start({"directory": "/state/adapters/abc"}),
            "/state/adapters/abc/adapter",
        )
