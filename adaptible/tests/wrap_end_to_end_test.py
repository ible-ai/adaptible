"""The whole loop, with nothing about the outcome stipulated.

Real FastAPI app, real ``Controller``, real ``Store``, real ``Trainer``
subprocess, real Transformers model, real LoRA, real adapter directory. A
question is asked, the answer is wrong, it is flagged, the loop looks up its
fixture evidence, trains, and the acceptance checks re-ask the *actual model
with the actual adapter applied*.

The complement to ``wrap_test``, which drives the same loop with a fake trainer
and a runtime that returns a hardcoded "Canberra" for any adapter. That proves
the control flow; this proves the weights.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from adaptible._src.wrap.app import create_app
from adaptible._src.wrap.repair import Controller, Trainer
from adaptible._src.wrap.store import Store
from adaptible.tests import tiny_models as tiny
from adaptible.tests.model_backed_runtime import (
    FACT,
    LlamaCppHarness,
    LMStudioHarness,
    ModelBackedRuntime,
    OllamaHarness,
    TinyGenerator,
    VLLMHarness,
)


class EndToEndRepairTest(unittest.IsolatedAsyncioTestCase):
    """One base model for the class; each test gets its own wrapper state."""

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory()
        cls.blob = tiny.write_trained_checkpoint(Path(cls._base.name) / "base")

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runtime = ModelBackedRuntime(self.blob)
        self.store = Store(root / "state", "end-to-end")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=Trainer(),
            idle_seconds=60,
            documents={"Morocco": FACT["document"]},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    async def ask(self, question=None, *, history=False):
        """Ask in a fresh conversation by default.

        A learned scope is bound to the conversation it was recorded in, so the
        same question inside an ongoing chat is served by the frozen base. That
        is the documented rule, covered in ``RuntimeEndToEndMixin``.
        """
        response = await self.client.post(
            "/interact",
            json={"prompt": question or FACT["question"], "use_history": history},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def review(self):
        await self.client.post("/trigger_review")
        response = await self.client.get("/sync")
        self.assertEqual(response.status_code, 200)
        return response.json()["reviews"]

    async def test_a_flagged_answer_is_repaired_by_training_real_weights(self):
        first = await self.ask()
        baseline = first["response"]
        self.assertIn(
            tiny.WRONG_ANSWER.rstrip("."),
            baseline,
            "the fixture model must start out confidently wrong",
        )
        self.assertNotIn(FACT["answer"], baseline)

        flagged = await self.client.post(
            "/feedback",
            json=dict(interaction_idx=first["interaction_idx"], thumbs="down"),
        )
        self.assertEqual(flagged.status_code, 200)

        reviews = await self.review()
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        # The adapter directory the worker wrote is a real PEFT adapter.
        accepted = self.store.get("accepted")
        adapter = Path(accepted["directory"]) / "adapter"
        self.assertTrue((adapter / "adapter_model.safetensors").is_file())
        stats = json.loads((Path(accepted["directory"]) / "stats.json").read_text())
        self.assertGreaterEqual(stats["steps"], 1)
        self.assertLess(stats["final_loss"], stats["initial_loss"])

        repaired = await self.ask()
        self.assertIn(
            FACT["answer"],
            repaired["response"],
            "the adapted weights must now produce the corrected answer",
        )

    async def test_the_base_checkpoint_is_never_written_to(self):
        before = {
            path.name: path.read_bytes()
            for path in self.blob.iterdir()
            if path.suffix == ".safetensors"
        }
        first = await self.ask()
        await self.client.post(
            "/feedback",
            json=dict(interaction_idx=first["interaction_idx"], thumbs="down"),
        )
        await self.review()
        after = {
            path.name: path.read_bytes()
            for path in self.blob.iterdir()
            if path.suffix == ".safetensors"
        }
        self.assertEqual(before, after)

    async def test_an_accepted_repair_survives_a_restart(self):
        first = await self.ask()
        await self.client.post(
            "/feedback",
            json=dict(interaction_idx=first["interaction_idx"], thumbs="down"),
        )
        self.assertEqual((await self.review())[0]["status"], "kept")
        accepted = self.store.get("accepted")

        store = Store(Path(self.temp.name) / "state", "end-to-end")
        runtime = ModelBackedRuntime(self.blob)
        controller = Controller(runtime, store, trainer=Trainer())
        try:
            await controller.start()
            self.assertEqual(runtime.active, accepted["directory"])
            # Serving after the restart uses the reloaded adapter, not the base.
            self.assertIn(
                FACT["answer"], runtime.generate(FACT["question"], runtime.active)
            )
        finally:
            await controller.close()

    async def test_an_unrelated_question_still_uses_the_base_model(self):
        first = await self.ask()
        await self.client.post(
            "/feedback",
            json=dict(interaction_idx=first["interaction_idx"], thumbs="down"),
        )
        self.assertEqual((await self.review())[0]["status"], "kept")

        unrelated = await self.client.post(
            "/interact",
            json=dict(prompt="What is the capital of France?", use_history=False),
        )
        self.assertEqual(unrelated.headers["X-Adaptible-Adapter"], "base")


if __name__ == "__main__":
    unittest.main()


class WatchingTrainer(Trainer):
    """Records the runtime's state at the moment training actually starts."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.serving_at_training = []

    async def train(self, *args, **kwargs):
        self.serving_at_training.append(self.runtime.process)
        return await super().train(*args, **kwargs)


class LlamaCppEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """The same loop through the real ``LlamaCpp`` runtime and a GGUF base.

    Only the child process and the socket are stood in for. The command line,
    readiness poll, ``lora`` scale, adapter staging and the stop-before-training
    step are the shipped code.
    """

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory()
        root = Path(cls._base.name)
        source = tiny.write_trained_checkpoint(root / "hf")
        cls.gguf = tiny.write_gguf_checkpoint(
            root / "model.gguf", "qwen2", source=source
        )

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    async def asyncSetUp(self):
        from adaptible._src.wrap.runtime import LlamaCpp

        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runtime = LlamaCpp(
            self.gguf, root / "runtime", executable="/nonexistent/llama-server"
        )
        self.runtime.architecture = "qwen2"
        # The runtime declares its training recipe; a toy model needs a rate
        # suited to it, as the production default barely moves these weights.
        self.runtime.training_options = lambda: {"learning_rate": 1e-3}
        self.generator = TinyGenerator(root.parent, max_tokens=24, gguf_file=None)
        # Serve from the same GGUF the runtime was given.
        self.generator.blob = Path(self.gguf).parent
        self.generator.gguf_file = Path(self.gguf).name
        self.harness = LlamaCppHarness(self.runtime, self.generator)
        self.harness.__enter__()
        self.trainer = WatchingTrainer(self.runtime)
        self.store = Store(root / "state", "llama-cpp-end-to-end")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            idle_seconds=60,
            documents={"Morocco": FACT["document"]},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.harness.__exit__(None, None, None)
        self.temp.cleanup()

    async def test_a_gguf_model_is_repaired_and_served_through_its_adapter(self):
        first = await self.client.post(
            "/interact", json={"prompt": FACT["question"], "use_history": False}
        )
        self.assertEqual(first.status_code, 200)
        body = first.json()
        self.assertIn(tiny.WRONG_ANSWER.rstrip("."), body["response"])

        await self.client.post(
            "/feedback",
            json=dict(interaction_idx=body["interaction_idx"], thumbs="down"),
        )
        await self.client.post("/trigger_review")
        reviews = (await self.client.get("/sync")).json()["reviews"]
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        # The worker exported a GGUF adapter, and llama-server was relaunched
        # pointing at it.
        accepted = Path(self.store.get("accepted")["directory"])
        self.assertTrue((accepted / "adapter.gguf").is_file())
        self.assertEqual(
            self.runtime.loaded, str((accepted / "adapter.gguf").resolve())
        )

        launch = self.harness.launches[-1]
        self.assertIn("--lora", launch)
        self.assertIn(str((accepted / "adapter.gguf").resolve()), launch)
        self.assertIn("--lora-init-without-apply", launch)

        repaired = await self.client.post(
            "/interact", json={"prompt": FACT["question"], "use_history": False}
        )
        self.assertIn(FACT["answer"], repaired.json()["response"])

    async def test_the_managed_server_is_stopped_before_training_starts(self):
        first = (
            await self.client.post(
                "/interact", json={"prompt": FACT["question"], "use_history": False}
            )
        ).json()
        await self.client.post(
            "/feedback",
            json=dict(interaction_idx=first["interaction_idx"], thumbs="down"),
        )
        launches_before = len(self.harness.launches)
        await self.client.post("/trigger_review")
        (await self.client.get("/sync")).json()
        # Dense training must not run beside a resident serving copy.
        self.assertTrue(self.trainer.serving_at_training, "training never ran")
        self.assertEqual(
            self.trainer.serving_at_training,
            [None] * len(self.trainer.serving_at_training),
            "llama-server was still resident when the trainer started",
        )
        # And it is relaunched afterwards to validate the candidate.
        self.assertGreater(len(self.harness.launches), launches_before)

    async def test_the_source_gguf_is_never_modified(self):
        before = Path(self.gguf).read_bytes()
        first = (
            await self.client.post(
                "/interact", json={"prompt": FACT["question"], "use_history": False}
            )
        ).json()
        await self.client.post(
            "/feedback",
            json=dict(interaction_idx=first["interaction_idx"], thumbs="down"),
        )
        await self.client.post("/trigger_review")
        (await self.client.get("/sync")).json()
        self.assertEqual(Path(self.gguf).read_bytes(), before)


class RuntimeEndToEndMixin:
    """Ask, flag, review, re-ask -- through whichever runtime the case builds."""

    async def ask(self, question=None, *, history=False):
        """Ask in a fresh conversation by default.

        A learned scope is bound to the conversation it was recorded in, so a
        re-ask inside an ongoing chat deliberately falls back to the base model.
        ``test_a_repair_does_not_leak_into_an_unrelated_conversation`` covers
        that; every other case here asks the way the terminal's ``/new`` does.
        """
        response = await self.client.post(
            "/interact",
            json={"prompt": question or FACT["question"], "use_history": history},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def flag_and_review(self, interaction):
        response = await self.client.post(
            "/feedback",
            json=dict(interaction_idx=interaction["interaction_idx"], thumbs="down"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        await self.client.post("/trigger_review")
        return (await self.client.get("/sync")).json()["reviews"]

    async def test_a_flagged_answer_is_repaired_and_then_served(self):
        first = await self.ask()
        self.assertIn(tiny.WRONG_ANSWER.rstrip("."), first["response"])

        reviews = await self.flag_and_review(first)
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        repaired = await self.ask()
        self.assertIn(FACT["answer"], repaired["response"])

    async def test_serving_weights_are_released_before_training(self):
        first = await self.ask()
        await self.flag_and_review(first)
        self.assertTrue(self.trainer.released_at_training, "training never ran")
        self.assertTrue(
            all(self.trainer.released_at_training),
            "serving weights were still resident when the trainer started",
        )

    async def test_a_repair_does_not_leak_into_an_unrelated_conversation(self):
        first = await self.ask()
        reviews = await self.flag_and_review(first)
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        # Build a conversation, then ask the repaired question inside it. The
        # recorded scope was learned with no preceding turns, so it does not
        # match here and the frozen base answers.
        await self.client.post(
            "/interact",
            json={
                "prompt": "What is the capital of France? Reply with only the name.",
                "use_history": True,
            },
        )
        carried = await self.client.post(
            "/interact", json={"prompt": FACT["question"], "use_history": True}
        )
        self.assertEqual(carried.headers["X-Adaptible-Adapter"], "base")

    async def test_the_source_model_is_never_modified(self):
        before = self.source_fingerprint()
        first = await self.ask()
        await self.flag_and_review(first)
        self.assertEqual(self.source_fingerprint(), before)


class ReleaseWatchingTrainer(Trainer):
    """Records whether serving weights were released when training began."""

    def __init__(self, released):
        self.released = released
        self.released_at_training = []

    async def train(self, *args, **kwargs):
        self.released_at_training.append(bool(self.released()))
        return await super().train(*args, **kwargs)


class VLLMEndToEndTest(RuntimeEndToEndMixin, unittest.IsolatedAsyncioTestCase):
    """The real ``VLLM`` runtime: native dynamic LoRA against a local HF base."""

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory()
        cls.blob = tiny.write_trained_checkpoint(Path(cls._base.name) / "hf")

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    def source_fingerprint(self):
        return {
            path.name: path.read_bytes()
            for path in Path(self.blob).iterdir()
            if path.suffix == ".safetensors"
        }

    async def asyncSetUp(self):
        from adaptible._src.wrap.vllm import VLLM

        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runtime = VLLM(self.blob, root / "runtime", executable="/nonexistent/vllm")
        self.runtime.architecture = "qwen2"
        self.runtime.training_options = lambda: {"learning_rate": 1e-3}
        self.harness = VLLMHarness(self.runtime, TinyGenerator(self.blob))
        self.harness.__enter__()
        self.trainer = ReleaseWatchingTrainer(lambda: self.runtime.process is None)
        self.store = Store(root / "state", "vllm-end-to-end")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            idle_seconds=60,
            documents={"Morocco": FACT["document"]},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.harness.__exit__(None, None, None)
        self.temp.cleanup()

    async def test_the_adapter_is_registered_under_a_private_handle(self):
        first = await self.ask()
        reviews = await self.flag_and_review(first)
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        self.assertTrue(self.runtime.active.startswith(self.runtime.prefix))
        self.assertIn(self.runtime.active, self.harness.loaded)
        accepted = Path(self.store.get("accepted")["directory"]) / "adapter"
        self.assertEqual(self.harness.loaded[self.runtime.active], accepted)


class LMStudioEndToEndTest(RuntimeEndToEndMixin, unittest.IsolatedAsyncioTestCase):
    """The real ``LMStudio`` runtime, which has no LoRA API and must fuse.

    Fusion is not stood in for: ``stage`` runs the real GGUF fusion worker and
    the answers here are generated from the fused file it writes.
    """

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory()
        root = Path(cls._base.name)
        source = tiny.write_trained_checkpoint(root / "hf")
        cls.gguf = tiny.write_gguf_checkpoint(
            root / "model.gguf", "qwen2", source=source
        )

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    def source_fingerprint(self):
        return Path(self.gguf).read_bytes()

    async def asyncSetUp(self):
        from adaptible._src.wrap.lmstudio import LMStudio

        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runtime = LMStudio(
            self.gguf, root / "runtime", executable="/nonexistent/lms"
        )
        self.runtime.architecture = "qwen2"
        self.runtime.training_options = lambda: {"learning_rate": 1e-3}
        self.harness = LMStudioHarness(self.runtime, self.gguf)
        self.harness.__enter__()
        # serve() discovers before building the controller; LM Studio's discover
        # imports the base model and is what gives it a key to load.
        await self.runtime.discover()
        self.trainer = ReleaseWatchingTrainer(lambda: self.runtime.instance is None)
        self.store = Store(root / "state", "lm-studio-end-to-end")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            idle_seconds=60,
            documents={"Morocco": FACT["document"]},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.harness.__exit__(None, None, None)
        self.temp.cleanup()

    async def test_the_accepted_candidate_is_served_from_a_fused_gguf(self):
        first = await self.ask()
        reviews = await self.flag_and_review(first)
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        merged = Path(self.store.get("accepted")["directory"]) / "merged.gguf"
        self.assertTrue(merged.is_file(), "LM Studio needs a fused serving file")
        self.assertNotEqual(merged.read_bytes(), Path(self.gguf).read_bytes())
        # The imported model the runtime switched to is that fused file.
        self.assertEqual(
            Path(self.harness.models[self.runtime.active]).resolve(),
            merged.resolve(),
        )


class OllamaEndToEndTest(RuntimeEndToEndMixin, unittest.IsolatedAsyncioTestCase):
    """The real ``Ollama`` runtime: a private tag created from a Modelfile."""

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory()
        root = Path(cls._base.name)
        source = tiny.write_trained_checkpoint(root / "hf")
        cls.gguf = tiny.write_gguf_checkpoint(
            root / "model.gguf", "qwen2", source=source
        )

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    def source_fingerprint(self):
        return Path(self.gguf).read_bytes()

    async def asyncSetUp(self):
        from adaptible._src.wrap.runtime import Ollama

        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.generator = TinyGenerator(
            Path(self.gguf).parent, gguf_file=Path(self.gguf).name
        )
        with mock.patch(
            "adaptible._src.wrap.runtime.shutil.which", return_value="/usr/bin/ollama"
        ):
            self.runtime = Ollama("tiny-model", root / "runtime")
        self.runtime.architecture = "qwen2"
        self.runtime.blob = Path(self.gguf)
        self.runtime.training_options = lambda: {"learning_rate": 1e-3}
        self.harness = OllamaHarness(
            self.runtime, self.generator, modelfile=f'FROM "{self.gguf}"\n'
        )
        self.harness.__enter__()
        await self.runtime.discover()
        self.trainer = ReleaseWatchingTrainer(lambda: not self.harness.resident)
        self.store = Store(root / "state", "ollama-end-to-end")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=self.trainer,
            idle_seconds=60,
            documents={"Morocco": FACT["document"]},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.harness.__exit__(None, None, None)
        self.temp.cleanup()

    async def test_the_adapter_is_served_through_a_private_tag(self):
        first = await self.ask()
        reviews = await self.flag_and_review(first)
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        self.assertTrue(self.runtime.active.startswith(self.runtime.prefix))
        created = [c for c in self.harness.commands if c[1] == "create"]
        self.assertTrue(created, "a private tag must be created for the candidate")
        # The Modelfile pins the exported adapter and keeps the original base.
        accepted = Path(self.store.get("accepted")["directory"])
        modelfile = (accepted / "Modelfile").read_text()
        self.assertIn(str((accepted / "adapter.gguf").resolve()), modelfile)
        self.assertIn(str(self.gguf), modelfile)


class ThinkingEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """The reasoning path: the model must keep thinking *and* be corrected.

    This is the case the native runs have never completed. A repair here has to
    train on a grounded rationale and still leave the model emitting a completed
    thought and a correct short answer -- the acceptance rule that rejected
    every previous attempt.
    """

    @classmethod
    def setUpClass(cls):
        cls._base = tempfile.TemporaryDirectory()
        cls.blob = tiny.write_trained_checkpoint(
            Path(cls._base.name) / "hf",
            "qwen3",
            pairs=tiny.THINKING_PAIRS,
            steps=600,
            hidden_size=96,
        )

    @classmethod
    def tearDownClass(cls):
        cls._base.cleanup()

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        # The rationale target is an order of magnitude longer than a bare
        # answer, and the thinking optimizer is the uncorrected MLX AdamW, so
        # the rate that suits the non-thinking fixture diverges here.
        self.runtime = ModelBackedRuntime(self.blob, learning_rate=1e-3)
        self.runtime.architecture = "qwen3"
        self.runtime.max_tokens = 48
        self.runtime.generator.max_tokens = 48
        self.store = Store(root / "state", "thinking-end-to-end")
        self.controller = Controller(
            self.runtime,
            self.store,
            trainer=Trainer(),
            idle_seconds=60,
            documents={"Morocco": FACT["document"]},
            web_search=False,
        )
        await self.controller.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.controller)),
            base_url="http://wrapper",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    async def test_a_reasoning_model_is_repaired_without_losing_its_reasoning(self):
        first = await self.client.post(
            "/interact", json={"prompt": FACT["question"], "use_history": False}
        )
        self.assertEqual(first.status_code, 200, first.text)
        body = first.json()
        self.assertIn(tiny.WRONG_ANSWER.rstrip("."), body["response"])

        flagged = await self.client.post(
            "/feedback",
            json=dict(interaction_idx=body["interaction_idx"], thumbs="down"),
        )
        self.assertEqual(flagged.status_code, 200, flagged.text)
        await self.client.post("/trigger_review")
        reviews = (await self.client.get("/sync")).json()["reviews"]
        self.assertEqual(reviews[0]["status"], "kept", reviews[0].get("reason"))

        repaired = await self.client.post(
            "/interact", json={"prompt": FACT["question"], "use_history": False}
        )
        answer = repaired.json()["response"]
        self.assertIn(FACT["answer"], answer)
        self.assertIn("<think>", answer)
        self.assertIn("</think>", answer)
