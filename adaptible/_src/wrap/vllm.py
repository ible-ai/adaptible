"""Managed vLLM with local weights and its documented dynamic LoRA API.

The private server binds to loopback because dynamic adapter loading is an
administrative operation. Only this wrapper's process group and adapters are
managed. Nothing here downloads a checkpoint or modifies a running server.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
from pathlib import Path

import httpx

from . import config
from .model_source import declared_precision
from .runtime import Runtime

logger = logging.getLogger(__name__)


class VLLM(Runtime):
    def __init__(
        self,
        model,
        directory,
        executable=None,
        context_size=4096,
        # Loading a checkpoint and building its graph is not a request: on a
        # CPU backend a 1.5B model takes several minutes, and a 180s bound made
        # the wrapper fail at startup rather than wait for a server that was
        # working normally.
        startup_timeout=1800,
        **kwargs,
    ):
        blob = Path(model).expanduser().resolve(strict=True)
        if not blob.is_dir():
            raise ValueError(
                "vLLM repair requires an existing local Hugging Face checkpoint "
                "directory, including its tokenizer; no checkpoint is downloaded."
            )
        executable = executable or shutil.which("vllm")
        if not executable:
            raise ValueError(
                "Install vLLM for this machine's supported GPU or CPU backend, "
                "then put vllm on PATH or pass --vllm-server /path/to/vllm. "
                "Apple Silicon's native CPU backend requires a source build."
            )
        super().__init__(blob.name, directory, **kwargs)
        self.blob, self.executable = blob, str(executable)
        self.context_size, self.startup_timeout = context_size, startup_timeout
        self.process = self.log = None
        self.adapters = {}
        self.loaded = set()
        self.prefix = (
            "adaptible-"
            + hashlib.sha256(str(self.directory.resolve()).encode()).hexdigest()[:12]
            + "-"
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    async def discover(self):
        return self.blob

    def prefills_thinking(self):
        """Whether this model's chat template opens a thought for every turn."""
        try:
            config = json.loads((self.blob / "tokenizer_config.json").read_text())
        except (OSError, ValueError):
            logger.debug("No readable tokenizer_config for %s", self.blob)
            return False
        template = config.get("chat_template")
        if not isinstance(template, str):
            return False
        return "<think>" in template and "</think>" in template

    def checkpoint_dtype(self):
        """The precision the checkpoint declares, so the server cannot pick."""
        # The weights, not `config.json`: this checkpoint declares `bfloat16`
        # and stores `F16`, and vLLM's loader goes by the tensors unless told
        # otherwise. Pinning the config's answer would be a second way to serve
        # a precision the checkpoint was never stored in.
        declared = declared_precision(self.blob)
        if declared is None:
            try:
                declared = json.loads((self.blob / "config.json").read_text()).get(
                    "torch_dtype"
                )
            except (OSError, ValueError):
                logger.debug("No readable config.json for %s", self.blob)
                return "auto"
        return declared if isinstance(declared, str) and declared else "auto"

    def command(self):
        args = [
            self.executable,
            "serve",
            str(self.blob),
            "--tokenizer",
            str(self.blob),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            self.name,
            "--max-model-len",
            str(self.context_size),
            "--max-num-seqs",
            "1",
            "--enable-lora",
            "--max-lora-rank",
            "8",
            "--max-loras",
            "2",
            "--max-cpu-loras",
            "2",
            # Avoid expensive graph compilation at every training handoff.
            "--enforce-eager",
            # vLLM's `--dtype auto` does not mean "the checkpoint's dtype": on
            # ARM it resolves f32 down to the platform's preferred bf16 and says
            # so in one log line, "Downcasting torch.float32 to torch.bfloat16".
            # That silently made vLLM the only runtime not serving the weights
            # it was given -- bf16 keeps 7 mantissa bits against f32's 23, and
            # a greedy decode against the identical checkpoint first disagreed
            # with the original about 35 tokens in. The checkpoint's own
            # `torch_dtype` is the answer; `auto` is only a fallback for a
            # config that does not state one.
            "--dtype",
            self.checkpoint_dtype(),
            # The analogue of llama.cpp's prompt cache, which was defect 11:
            # a reused prefix is evaluated in a different batch shape and the
            # logits move in the last bits. Greedy absorbs it, a temperature
            # draw does not, and the repair loop samples its corrections.
            "--no-enable-prefix-caching",
        ]
        if self.architecture == "qwen3":
            args += ["--reasoning-parser", "qwen3"]
        elif self.prefills_thinking():
            # A distilled reasoning model on another architecture still frames
            # its thought with <think>, and its template prefills the opening
            # tag. Without a parser vLLM leaves the whole thought in `content`,
            # so an unterminated ramble that mentions the answer somewhere
            # scores as an answer -- and the experiment counts exactly that as
            # a miss (its `closed(r)` requires `</think>`). The behaviour is
            # read from the template rather than the architecture name, and it
            # has to be known before launch, so the startup probe is too late.
            args += ["--reasoning-parser", "deepseek_r1"]
        return args

    async def launch(self):
        if self.process is not None and self.process.returncode is None:
            return
        await self.stop()
        self.log = (self.directory / "vllm-server.log").open("ab")
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            # Never inherit a remote adapter resolver from the user's shell.
            "VLLM_PLUGINS": "",
        }
        # vLLM's CPU default reserves 4 GiB for KV cache. One GiB is ample
        # for this wrapper's single short-context request; GPU backends ignore
        # the variable, and an explicit user setting takes precedence.
        env.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command(),
                env=env,
                stdout=self.log,
                stderr=self.log,
                start_new_session=True,
            )
            last_error = None
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            while asyncio.get_running_loop().time() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(
                        "vLLM exited before becoming ready. Its installed backend must "
                        "support this hardware and the selected model's LoRA; see "
                        f"{self.directory / 'vllm-server.log'}"
                    )
                try:
                    response = await self.client.get(self.url + "/health", timeout=1)
                    if response.status_code == 200:
                        models = await self.client.get(self.url + "/v1/models")
                        models.raise_for_status()
                        if self.name in {m["id"] for m in models.json()["data"]}:
                            return
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.debug("vLLM not ready yet: %s", last_error)
                await asyncio.sleep(config.READY_POLL_SECONDS)
            raise RuntimeError(
                f"vLLM did not become ready within {self.startup_timeout:g}s"
                + (f"; last error: {last_error}" if last_error else "")
                + f"; see {self.directory / 'vllm-server.log'}"
            )
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception:
            await self.stop()
            raise

    def payload(self, body, *, handle=None, frozen=False):
        desired = None if frozen else handle or self.active
        if desired and desired not in self.loaded:
            raise RuntimeError("Requested adapter is not loaded in managed vLLM.")
        return {**body, "model": desired or self.name}

    async def _load(self, handle):
        if handle in self.loaded:
            return
        if handle not in self.adapters:
            raise ValueError("Cannot load an adapter not owned by this wrapper.")
        response = await self.client.post(
            self.url + "/v1/load_lora_adapter",
            json={"lora_name": handle, "lora_path": str(self.adapters[handle])},
        )
        if response.is_error:
            raise RuntimeError(
                "vLLM could not load the local PEFT adapter. Its backend must support "
                "LoRA and runtime loading: "
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )
        self.loaded.add(handle)

    async def stage(self, directory):
        directory = Path(directory).resolve(strict=True)
        adapter = directory / "adapter"
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config.get("peft_type") != "LORA" or config.get("r") != 8:
            raise ValueError("vLLM repair expects the wrapper's rank-8 PEFT LoRA.")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError("The candidate has no local PEFT adapter weights.")
        handle = self.prefix + hashlib.sha256(str(directory).encode()).hexdigest()[:20]
        self.adapters[handle] = adapter
        await self.launch()
        try:
            await self._load(handle)
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception:
            # A lost HTTP response may follow a successful server-side load.
            # The controller has no handle until stage returns, so clean up our
            # known private name here while preserving the original error.
            try:
                await self.client.post(
                    self.url + "/v1/unload_lora_adapter",
                    json={"lora_name": handle},
                    timeout=5,
                )
            except httpx.HTTPError:
                logger.warning(
                    "Could not unload vLLM adapter %s after a failed load.",
                    handle,
                    exc_info=True,
                )
            self.loaded.discard(handle)
            self.adapters.pop(handle, None)
            raise
        return handle

    async def restore(self, handle):
        await self.launch()
        if handle:
            await self._load(handle)
        self.active = handle

    async def discard(self, handle):
        if handle == self.active or handle not in self.adapters:
            return
        if handle in self.loaded:
            response = await self.client.post(
                self.url + "/v1/unload_lora_adapter", json={"lora_name": handle}
            )
            response.raise_for_status()
            self.loaded.remove(handle)
        self.adapters.pop(handle, None)

    async def release_for_training(self):
        # Sleep mode retains CPU copies and is not supported on every backend.
        # Stopping our server frees the entire serving copy before dense training.
        await self.stop()

    async def stop(self):
        if self.process is not None:
            # The API process may have crashed while GPU workers survive.
            # Signal our whole group even when its leader has already exited.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                logger.debug("vLLM process group already gone before SIGTERM.")
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(
                        self.process.wait(), config.SHUTDOWN_GRACE_SECONDS
                    )
                except TimeoutError:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        logger.debug("vLLM group exited before the forced kill.")
                    await self.process.wait()
            # Do not hand memory to the training worker while orphaned engine
            # processes are still exiting after their API parent has returned.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.debug("No orphaned vLLM engine processes remained.")
            self.process = None
        self.loaded.clear()
        if self.log:
            self.log.close()
            self.log = None

    async def close(self):
        try:
            await self.stop()
        finally:
            await super().close()
