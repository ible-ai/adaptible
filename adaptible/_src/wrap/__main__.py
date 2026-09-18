"""One command to wrap an installed model and repair it from feedback."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import uvicorn

from .._paths import outputs_dir
from . import config
from .app import create_app
from .lmstudio import LMStudio
from .model_source import fingerprint_base, read_architecture, repairable
from .repair import Controller
from .runtime import LlamaCpp, Ollama
from .store import Store
from .vllm import VLLM

# Third-party packages from the optional ``wrap`` extra are imported inside
# ``main`` and ``serve`` on purpose: importing them here would turn a missing
# extra into an ImportError traceback instead of the install hint below.
_REQUIRED_EXTRAS = ("torch", "peft", "gguf", "httpx", "uvicorn", "filelock")

_LOG_LEVELS = ("debug", "info", "warning", "error")

logger = logging.getLogger(__name__)


def parser():
    """Builds the ``adaptible wrap`` argument parser."""
    p = argparse.ArgumentParser(prog="adaptible wrap", description=__doc__)
    p.add_argument("service", choices=("ollama", "llama-cpp", "lm-studio", "vllm"))
    p.add_argument(
        "model",
        help="Ollama model name, local GGUF (llama.cpp/LM Studio), or local HF directory (vLLM)",
    )
    p.add_argument(
        "--upstream",
        help=(
            "Existing Ollama or LM Studio URL. Defaults to $ADAPTIBLE_OLLAMA_URL "
            f"({config.OLLAMA_URL}) or $ADAPTIBLE_LM_STUDIO_URL "
            f"({config.LM_STUDIO_URL}). llama.cpp and vLLM are managed locally "
            "and do not accept it."
        ),
    )
    p.add_argument(
        "--host",
        default=config.DEFAULT_HOST,
        help="Address the wrapper listens on (default: %(default)s)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=config.DEFAULT_PORT,
        help="Port the wrapper listens on (default: %(default)s)",
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        help="History and small adapters (default: outputs/wrap/<model>)",
    )
    p.add_argument(
        "--documents",
        type=Path,
        help="Optional local reference passages (JSON object mapping titles to text)",
    )
    p.add_argument(
        "--no-web-search",
        action="store_true",
        help="Keep reference lookup offline; use only supplied documents or feedback notes",
    )
    p.add_argument(
        "--flagship-recipe",
        action="store_true",
        help="Match scripts/cycles_mlx.py: accepted weights answer every prompt, "
        "and an update is kept on the repaired item's own score, without the "
        "control prompts and prior-repair checks a served wrapper applies",
    )
    p.add_argument(
        "--flagship-candidate-temperature",
        type=float,
        default=None,
        help="Temperature for sampling a correction under --flagship-recipe "
        "(default 0.7, the experiment's). Set 0 to draw the candidate greedily: "
        "MLX, vLLM and llama.cpp use different generators, so the same seed "
        "means different draws, and greedy is the only way five runtimes can "
        "draw the same candidate and be compared for identical output",
    )
    p.add_argument(
        "--initial-adapter",
        default=None,
        help="PEFT adapter directory the first training candidate starts from "
        "instead of a fresh LoRA initialisation. Used to reproduce one particular "
        "run of the original, which draws its initialisation unseeded",
    )
    p.add_argument(
        "--idle-seconds",
        type=float,
        default=config.DEFAULT_IDLE_SECONDS,
        help="Wait after the last request before reviewing feedback (default: %(default)s)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=config.DEFAULT_MAX_TOKENS,
        help="Completion budget, reasoning included (default: %(default)s)",
    )
    p.add_argument("--llama-server", help="Path to llama-server when it is not on PATH")
    p.add_argument("--lms", help="Path to the LM Studio lms CLI")
    p.add_argument("--vllm-server", help="Path to the vllm executable")
    p.add_argument(
        "--context-size",
        type=int,
        default=config.DEFAULT_CONTEXT_SIZE,
        help="Managed runtime context size (default: %(default)s)",
    )
    p.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="info",
        help="Wrapper and runtime log verbosity (default: %(default)s)",
    )
    return p


def _load_documents(path):
    """Reads a JSON object of title: passage strings.

    Raises:
        ValueError: The file is not a flat object of strings.
    """
    documents = json.loads(path.read_text())
    if not isinstance(documents, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in documents.items()
    ):
        raise ValueError(
            "--documents must contain a JSON object of title: passage strings."
        )
    return documents


def _build_runtime(args, directory):
    """Constructs the runtime adapter for the selected service."""
    if args.service == "ollama":
        return Ollama(
            args.model,
            directory,
            url=config.upstream_url("ollama", args.upstream),
            max_tokens=args.max_tokens,
            context_size=args.context_size,
        )
    if args.service == "llama-cpp":
        return LlamaCpp(
            args.model,
            directory,
            executable=args.llama_server,
            context_size=args.context_size,
            max_tokens=args.max_tokens,
        )
    if args.service == "lm-studio":
        return LMStudio(
            args.model,
            directory,
            url=config.upstream_url("lm-studio", args.upstream),
            executable=args.lms,
            context_size=args.context_size,
            max_tokens=args.max_tokens,
        )
    return VLLM(
        args.model,
        directory,
        executable=args.vllm_server,
        context_size=args.context_size,
        max_tokens=args.max_tokens,
    )


async def serve(args):
    """Runs the wrapper until the server stops, then releases every resource."""
    from filelock import FileLock, Timeout

    key = hashlib.sha256((args.service + ":" + args.model).encode()).hexdigest()[:12]
    directory = (args.state_dir or outputs_dir() / "wrap" / key).expanduser().resolve()
    documents = _load_documents(args.documents) if args.documents else None
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(directory / ".wrapper.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise ValueError(
            "Another wrapper is already using this state directory."
        ) from exc
    runtime = controller = None
    try:
        runtime = _build_runtime(args, directory)
        blob = await runtime.discover()
        # Serving is a proxy: it must work for whatever model the runtime is
        # already running. Only repair needs an architecture it can train.
        runtime.architecture = await asyncio.to_thread(read_architecture, blob)
        can_repair, reason = await asyncio.to_thread(repairable, blob)
        # Reasoning is probed once the model is serving; see the app's lifespan.
        digest = await asyncio.to_thread(fingerprint_base, blob)
        store = Store(directory, args.service + ":" + args.model + ":" + digest)
        template_digest = getattr(runtime, "serving_template_digest", None)
        if template_digest:
            try:
                store.bind_serving_template(template_digest)
            except BaseException:
                store.close()
                raise
        controller = Controller(
            runtime,
            store,
            documents=documents,
            idle_seconds=args.idle_seconds,
            web_search=not args.no_web_search,
            flagship_recipe=args.flagship_recipe,
            flagship_candidate_temperature=args.flagship_candidate_temperature,
            initial_adapter=args.initial_adapter,
        )
        app = create_app(controller)
        address = config.server_url(args.host, args.port)
        print(
            f"Wrapping {args.service} model {runtime.name}; reusing {blob}\n"
            f"Chat: {address}/v1  |  Feedback and history: {address}/docs\n"
            f"State: {directory}\n"
            + (
                f"Thumbs-down feedback is reviewed after {args.idle_seconds:g}s idle."
                if can_repair
                else f"Chat only: {reason} Feedback is recorded but not trained on."
            ),
            flush=True,
        )
        await uvicorn.Server(
            uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_graceful_shutdown=5,
            )
        ).serve()
    finally:
        # Also cover discovery/startup failure, before FastAPI enters lifespan.
        try:
            if controller is not None and not controller.closing:
                await controller.close()
            elif controller is None and runtime is not None:
                await runtime.close()
        finally:
            lock.release()


def main(argv=None):
    """Entry point for ``adaptible wrap``. Returns a process exit code."""
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config.validate_server_options(args)
    except ValueError as exc:
        parser().error(str(exc))
    missing = [
        name for name in _REQUIRED_EXTRAS if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(
            f"Missing wrapper support: {', '.join(missing)}.\n"
            "Install it once: python -m pip install 'adaptible[wrap]'\n"
            "From this checkout: python -m pip install -e '.[wrap]'",
            file=sys.stderr,
        )
        return 2
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("Wrapper startup failed")
        print(f"Cannot run wrapper: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
