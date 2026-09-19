"""Deployment defaults and validation, independent of runtime dependencies.

Existing servers use explicit URLs or environment configuration. Managed child
servers allocate their own loopback ports; they do not reuse an existing server.
"""

import math
import os
from urllib import parse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_IDLE_SECONDS = 2.0
OLLAMA_URL = "http://127.0.0.1:11434"
LM_STUDIO_URL = "http://127.0.0.1:1234"

# Hostnames that mean "this machine". Repair needs the serving process to be
# able to read the same local model file, so non-loopback upstreams are refused.
LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

# Managed child servers: how long to wait for readiness, and how long a
# terminated server may take to exit before it is killed.
#
# Readiness is dominated by reading the weights, so the bound is set by the
# model and the disk, not by the server: an f32 1.5B GGUF off an external USB
# disk takes minutes. Sixty seconds silently made every such model look like a
# server that failed to start. vLLM's own path already allows 1800s for the
# same reason; this is the matching bound for the rest.
READY_POLL_SECONDS = 0.25
READY_TIMEOUT_SECONDS = 1800.0
SHUTDOWN_GRACE_SECONDS = 15.0

_UPSTREAMS = {
    "ollama": ("ADAPTIBLE_OLLAMA_URL", OLLAMA_URL),
    "lm-studio": ("ADAPTIBLE_LM_STUDIO_URL", LM_STUDIO_URL),
}


def upstream_url(service: str, explicit: str | None = None) -> str:
    """Resolve an existing server URL: argument, environment, then default.

    Raises:
        ValueError: The service is managed locally or the URL is malformed.
    """
    if service not in _UPSTREAMS:
        raise ValueError(
            f"{service} starts a managed server; --upstream is not supported."
        )
    variable, default = _UPSTREAMS[service]
    value = explicit if explicit is not None else os.environ.get(variable, default)
    parts = parse.urlsplit(value)
    # Accessing port also checks malformed and out-of-range port numbers.
    port = parts.port
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port == 0
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            f"{service} upstream must be an absolute HTTP(S) URL without "
            "credentials, query parameters, or a fragment."
        )
    return value.rstrip("/")


def validate_server_options(args) -> None:
    """Reject invalid deployment options before creating state or processes."""
    if not args.host or any(character.isspace() for character in args.host):
        raise ValueError("host must be a nonempty bind address without whitespace")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds < 0:
        raise ValueError("idle-seconds must be finite and >= 0")
    if args.max_tokens < 1 or args.context_size < 128:
        raise ValueError("max-tokens must be >= 1; context-size must be >= 128")
    if args.service in _UPSTREAMS or args.upstream is not None:
        upstream_url(args.service, args.upstream)


def server_url(host: str, port: int) -> str:
    """Format a listening address, including IPv6 literals, for CLI output."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"
