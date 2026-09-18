"""LM Studio serving with locally fused LoRA candidates.

LM Studio does not expose native LoRA loading. We preserve the existing GGUF,
merge only adapted matrices into a derived GGUF, and import it by symbolic link.
Generation always runs in LM Studio; no checkpoint is fetched or substituted.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from . import config
from .gguf_fusion import fuse_gguf  # compatibility re-export
from .runtime import NEUTRAL_SAMPLING, Runtime, run_command

# LM Studio is llama.cpp underneath, so it drops to reduced precision in the
# same three places llama.cpp does -- f16-staged prefill, an f16 KV cache, and
# flash attention -- which together put llama.cpp 3.7e-4 from MLX and, pinned,
# 1.4e-6 (see `runtime.LLAMA_CPP_FULL_PRECISION`).
#
# LM Studio has two load APIs and neither carries all three alone. Its REST
# endpoint accepts `flash_attention` and `physical_batch_size` but rejects every
# KV cache key by name. Its SDK API accepts `llama_{k,v}_cache_quantization_type`
# including 'f32', and `eval_batch_size` -- and llama.cpp caps the physical
# batch at the logical one, so a logical batch of 8 keeps prefill off the
# half-staged kernel too. So models are loaded through the SDK.
#
# Measured on the real checkpoint with a trained adapter: loaded with an f16 KV
# cache, LM Studio departs from the original at character 191 of one of eight
# generations -- byte-identical to llama.cpp run with an f16 KV cache. Loaded
# with these settings, that generation is identical to the original.
LM_STUDIO_FULL_PRECISION = {
    "flash_attention": False,
    "eval_batch_size": 8,
    "llama_k_cache_quantization_type": "f32",
    "llama_v_cache_quantization_type": "f32",
}

logger = logging.getLogger(__name__)


def _import_matches(key, expected):
    return key in (expected, expected.split("/", 1)[-1]) or key.startswith(
        expected + "/"
    )


class LMStudio(Runtime):
    def __init__(
        self,
        model,
        directory,
        url=config.LM_STUDIO_URL,
        executable=None,
        context_size=4096,
        **kwargs,
    ):
        super().__init__(Path(model).stem, directory, **kwargs)
        self.blob = Path(model).expanduser().resolve(strict=True)
        self.url = url.rstrip("/")
        if urlparse(self.url).hostname not in config.LOOPBACK_HOSTNAMES:
            raise ValueError(
                "LM Studio repair requires a local service with access to this GGUF."
            )
        self.executable = executable or shutil.which("lms")
        if not self.executable:
            candidate = Path.home() / ".lmstudio/bin/lms"
            self.executable = str(candidate) if candidate.is_file() else None
        if not self.executable:
            raise ValueError(
                "Install LM Studio and start its local server; lms must be on PATH (or use --lms)."
            )
        self.context_size = context_size
        self.prefix = (
            "adaptible-"
            + hashlib.sha256(str(self.directory.resolve()).encode()).hexdigest()[:12]
        )
        self.imports_path = self.directory / "lmstudio-imports.json"
        self.imports = (
            json.loads(self.imports_path.read_text())
            if self.imports_path.exists()
            else {}
        )
        self.loaded = None
        self.instance = None
        self.base_key = None
        token = os.environ.get("LM_API_TOKEN")
        if token:
            self.client.headers["Authorization"] = f"Bearer {token}"

    async def discover(self):
        response = await self.client.get(self.url + "/api/v1/models")
        response.raise_for_status()
        local = self.directory / "base.gguf"
        if (local.exists() or local.is_symlink()) and local.resolve() != self.blob:
            raise ValueError("Saved LM Studio base link belongs to another model.")
        self.base_key = await self._import(self.blob, "base")
        return self.blob

    async def _import(self, path, label):
        repo = f"{self.prefix}-{label}"
        expected = f"adaptible/{repo}"
        response = await self.client.get(self.url + "/api/v1/models")
        response.raise_for_status()
        matches = [
            m["key"]
            for m in response.json()["models"]
            if _import_matches(m["key"], expected)
        ]
        if not matches:
            output = await run_command(
                self.executable,
                "import",
                path,
                "--symbolic-link",
                "--user-repo",
                expected,
                "--yes",
            )
            clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
            match = re.search(r"Symbolic link created at\s+(.+)", clean)
            if match:
                self.imports[expected] = dict(
                    link=match[1].strip(), source=str(Path(path).absolute())
                )
                self.imports_path.write_text(json.dumps(self.imports))
            for _ in range(40):
                response = await self.client.get(self.url + "/api/v1/models")
                response.raise_for_status()
                matches = [
                    m["key"]
                    for m in response.json()["models"]
                    if _import_matches(m["key"], expected)
                ]
                if matches:
                    break
                await asyncio.sleep(0.25)
        if len(matches) != 1:
            raise ValueError(
                f"LM Studio did not uniquely discover imported model {expected}."
            )
        return matches[0]

    async def _ensure(self, key):
        if self.loaded == key and self.instance:
            return
        await self.release_for_training()
        self.instance = await self._load_instance(key)
        self.loaded = key

    async def _load_instance(self, key):
        """Load ``key`` through LM Studio's SDK API and return its instance id.

        The SDK rather than REST because only the SDK accepts a KV cache type
        (see ``LM_STUDIO_FULL_PRECISION``). A model it loads stays resident
        after the connection closes, so serving continues over REST unchanged.
        """
        import lmstudio

        async with lmstudio.AsyncClient(urlparse(self.url).netloc) as client:
            model = await client.llm.load_new_instance(
                key,
                config={
                    "context_length": self.context_size,
                    **LM_STUDIO_FULL_PRECISION,
                },
            )
        return model.identifier

    def payload(self, body, *, handle=None, frozen=False):
        desired = self.base_key if frozen else (handle or self.active or self.base_key)
        if self.loaded != desired or not self.instance:
            raise RuntimeError("Requested LM Studio model is not loaded.")
        # LM Studio is llama.cpp underneath and applies the same house
        # defaults, so an unpinned request is served under repeat_penalty 1.1.
        # It scored this item 4/4 where every other version scored 3/4, and the
        # wrapper then skipped the repair entirely as "already passes".
        #
        # `cache_prompt` for the same reason it is off for llama.cpp (defect
        # 11): a reused KV prefix is evaluated in a different batch shape, the
        # logits move in the last bits, and a seeded draw lands elsewhere.
        # Measured here as the same seed returning two different samples, which
        # is the one thing candidate selection cannot tolerate.
        return {
            "cache_prompt": False,
            **NEUTRAL_SAMPLING,
            **body,
            "model": self.instance,
        }

    async def prepare_payload(self, body, *, frozen=False):
        await self._ensure(self.base_key if frozen else (self.active or self.base_key))
        return self.payload(body, frozen=frozen)

    async def complete(self, messages, *, handle=None, frozen=False, **kwargs):
        previous = self.loaded
        desired = self.base_key if frozen else (handle or self.active or self.base_key)
        await self._ensure(desired)
        try:
            return await super().complete(
                messages, handle=handle, frozen=frozen, **kwargs
            )
        finally:
            if previous and previous != desired:
                await self._ensure(previous)

    async def stage(self, directory):
        directory = Path(directory)
        merged = directory / "merged.gguf"
        expected = f"adaptible/{self.prefix}-{directory.name}"
        try:
            if not merged.exists():
                # A separate worker makes cancellation terminate fusion and release
                # its tensor buffers; it cannot linger after wrapper shutdown.

                try:
                    await run_command(
                        sys.executable,
                        "-m",
                        "adaptible._src.wrap.gguf_fusion",
                        str(self.blob),
                        str(directory / "adapter.gguf"),
                        str(merged),
                    )
                finally:
                    # SIGTERM cannot run the fusion child's Python finally block.
                    merged.with_suffix(".gguf.partial").unlink(missing_ok=True)
            key = await self._import(merged, directory.name)
            await self._ensure(key)
            return key
        except BaseException:
            # No handle reached Controller, so it cannot discard this candidate.
            # Keep its small adapter for diagnosis/retry, but retire derived weights.
            try:
                if self.loaded and _import_matches(self.loaded, expected):
                    await self.release_for_training()
            finally:
                self._remove_import(expected, remove_merged=True)
                if not merged.is_symlink() and merged.resolve().is_relative_to(
                    (self.directory / "adapters").resolve()
                ):
                    merged.unlink(missing_ok=True)
            raise

    async def restore(self, handle):
        await self._ensure(handle or self.base_key)
        self.active = handle

    async def release_for_training(self):
        if self.instance:
            response = await self.client.post(
                self.url + "/api/v1/models/unload", json={"instance_id": self.instance}
            )
            response.raise_for_status()
            self.instance = self.loaded = None

    def _remove_import(self, key, *, remove_merged=False):
        for expected, entry in list(self.imports.items()):
            if not _import_matches(key, expected):
                continue
            link, source = Path(entry["link"]), Path(entry["source"])
            if (
                link.is_symlink()
                and link.resolve() == source.resolve()
                and link.parent.name.startswith(self.prefix + "-")
            ):
                link.unlink()
                try:
                    link.parent.rmdir()
                except OSError:
                    logger.debug(
                        "Left %s in place: not empty or not removable.", link.parent
                    )
            self.imports.pop(expected)
            # The small adapter is sufficient to recreate this local artifact.
            # Never remove the original checkpoint or follow a substituted link.
            if (
                remove_merged
                and source.name == "merged.gguf"
                and not source.is_symlink()
                and source.resolve().is_relative_to(
                    (self.directory / "adapters").resolve()
                )
            ):
                source.unlink(missing_ok=True)
        self.imports_path.write_text(json.dumps(self.imports))

    async def discard(self, handle):
        if (
            handle
            and handle != self.active
            and handle.removeprefix("adaptible/").startswith(self.prefix + "-")
        ):
            if self.loaded == handle:
                await self.release_for_training()
            self._remove_import(handle, remove_merged=True)

    async def cleanup_imports(self):
        """Optional demo cleanup; preserve state and imports during ordinary close."""
        await self.release_for_training()
        for key in list(self.imports):
            self._remove_import(key, remove_merged=True)

    async def close(self):
        try:
            await self.release_for_training()
        finally:
            await super().close()


if __name__ == "__main__":

    fuse_gguf(*sys.argv[1:])
