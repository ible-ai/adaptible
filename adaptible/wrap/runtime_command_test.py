"""Owned subprocess cancellation, using tiny local Python children only."""

import asyncio
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from adaptible.wrap.runtime import LlamaCpp, run_command


class LlamaCppPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_loaded_adapter_uses_explicit_zero_for_every_base_route(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base.gguf"
            source.touch()
            runtime = LlamaCpp(source, directory, executable="unused-llama-server")
            try:
                runtime.loaded = "adapter.gguf"
                body = {"messages": [], "stream": True, "lora": [{"id": 0, "scale": 9}]}
                self.assertEqual(runtime.payload(body)["lora"], [{"id": 0, "scale": 0}])
                runtime.active = runtime.loaded
                self.assertEqual(runtime.payload(body)["lora"], [{"id": 0, "scale": 1}])
                self.assertEqual(
                    runtime.payload(body, frozen=True)["lora"], [{"id": 0, "scale": 0}]
                )
                self.assertEqual(
                    (await runtime.prepare_payload(body, frozen=True))["lora"],
                    [{"id": 0, "scale": 0}],
                )
                self.assertEqual(
                    runtime.payload(body, handle=runtime.loaded)["lora"],
                    [{"id": 0, "scale": 1}],
                )
                self.assertEqual(body["lora"], [{"id": 0, "scale": 9}])
                self.assertEqual(runtime.active, "adapter.gguf")
            finally:
                await runtime.close()

    async def test_unloaded_base_uses_empty_list_and_rejects_unknown_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base.gguf"
            source.touch()
            runtime = LlamaCpp(source, directory, executable="unused-llama-server")
            try:
                self.assertEqual(runtime.payload({})["lora"], [])
                self.assertEqual(runtime.payload({}, frozen=True)["lora"], [])
                with self.assertRaisesRegex(RuntimeError, "not loaded"):
                    runtime.payload({}, handle="unknown.gguf")
            finally:
                await runtime.close()


    async def test_prompt_cache_is_off_so_a_seeded_draw_reproduces(self):
        """Measured against llama-server: the same seed and prompt back to back
        returned 1961 then 1613 characters with the cache on, and the same 1961
        twice with `cache_prompt: false`. A reused prefix is evaluated in a
        different batch shape, the logits differ in the last bits, and a
        temperature-0.7 draw lands elsewhere -- so a correction sampled by the
        repair loop depended on what had been asked before it. The original
        re-runs the whole prompt every call and carries no such state.
        """
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base.gguf"
            source.touch()
            runtime = LlamaCpp(source, directory, executable="unused-llama-server")
            try:
                self.assertIs(runtime.payload({})["cache_prompt"], False)
                self.assertIs(
                    runtime.payload({}, frozen=True)["cache_prompt"], False
                )
                # An explicit request still wins; the default is a default.
                self.assertIs(
                    runtime.payload({"cache_prompt": True})["cache_prompt"], True
                )
            finally:
                await runtime.close()


class RuntimeCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_creation_reaps_child_after_handle_arrives(self):
        spawn = asyncio.create_subprocess_exec
        started, release = asyncio.Event(), asyncio.Event()
        children = []

        async def delayed_spawn(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            children.append(process)
            started.set()
            await release.wait()
            return process

        with patch(
            "adaptible.wrap.runtime.asyncio.create_subprocess_exec", delayed_spawn
        ):
            task = asyncio.create_task(
                run_command(sys.executable, "-c", "import time; time.sleep(60)")
            )
            try:
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                self.assertIsNotNone(children[0].returncode)
            finally:
                release.set()
                for process in children:
                    if process.returncode is None:
                        process.kill()
                    await process.communicate()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_cancel_force_kills_child_that_ignores_termination(self):
        spawn = asyncio.create_subprocess_exec
        children = []

        async def capture_spawn(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            children.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            program = (
                "import signal, sys, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "Path(sys.argv[1]).write_text('ready'); time.sleep(60)"
            )
            with (
                patch(
                    "adaptible.wrap.runtime.asyncio.create_subprocess_exec",
                    capture_spawn,
                ),
                patch("adaptible.wrap.runtime._COMMAND_SHUTDOWN_TIMEOUT", 0.05),
            ):
                task = asyncio.create_task(
                    run_command(sys.executable, "-c", program, ready)
                )
                try:
                    async with asyncio.timeout(2):
                        while not ready.exists():
                            await asyncio.sleep(0.005)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 2)
                    self.assertEqual(children[0].returncode, -signal.SIGKILL)
                finally:
                    for process in children:
                        if process.returncode is None:
                            process.kill()
                        await process.communicate()
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    async def test_success_and_bounded_failure_output_are_preserved(self):
        self.assertEqual(
            await run_command(sys.executable, "-c", "print('ready')"), "ready\n"
        )
        with self.assertRaises(RuntimeError) as caught:
            await run_command(
                sys.executable,
                "-c",
                "import sys; print('x' * 3000 + 'tail'); sys.exit(2)",
            )
        self.assertEqual(len(str(caught.exception)), 2000)
        self.assertTrue(str(caught.exception).endswith("tail\n"))

    async def test_missing_executable_retains_original_spawn_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                await run_command(Path(directory) / "missing-command")


if __name__ == "__main__":
    unittest.main()


class LlamaCppPrecisionTest(unittest.IsolatedAsyncioTestCase):
    """llama.cpp must compute at the precision it is given, not its defaults.

    Its Metal backend drops to reduced precision in three independent places by
    default -- f16-staged prefill matmuls, an f16 KV cache, flash attention --
    and each alone holds the gap to MLX at 1e-4 or worse. Measured on one model
    from bit-identical weights: 3.7e-4 at the defaults, 1.4e-6 with all three
    pinned. That residual is what flipped a greedy argmax at the same character
    on all three ggml runtimes.
    """

    async def test_every_launch_pins_all_three(self):
        from adaptible.wrap.runtime import LLAMA_CPP_FULL_PRECISION

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base.gguf"
            source.touch()
            runtime = LlamaCpp(source, directory, executable="unused-llama-server")
            captured = []

            async def fake_exec(*args, **kwargs):
                captured.append(args)
                raise RuntimeError("stop after capturing argv")

            try:
                with patch("asyncio.create_subprocess_exec", fake_exec):
                    for handle in (None, "adapter.gguf"):
                        with self.assertRaises(RuntimeError):
                            await runtime.launch(handle)
            finally:
                await runtime.close()

        self.assertEqual(len(captured), 2)
        for argv in captured:
            joined = list(argv)
            for flag, value in zip(
                LLAMA_CPP_FULL_PRECISION[::2], LLAMA_CPP_FULL_PRECISION[1::2]
            ):
                self.assertIn(flag, joined, flag)
                self.assertEqual(joined[joined.index(flag) + 1], value, flag)

    def test_the_three_settings_are_the_measured_ones(self):
        """Pins the values, so one cannot be dropped as a speed tweak without
        this failing: each alone leaves the gap to MLX near 1e-4."""
        from adaptible.wrap.runtime import LLAMA_CPP_FULL_PRECISION

        settings = dict(zip(LLAMA_CPP_FULL_PRECISION[::2], LLAMA_CPP_FULL_PRECISION[1::2]))
        self.assertEqual(
            settings,
            {
                "--ubatch-size": "8",
                "--cache-type-k": "f32",
                "--cache-type-v": "f32",
                "--flash-attn": "off",
            },
        )
