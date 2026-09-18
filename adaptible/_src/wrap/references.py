"""Bounded, model-free discovery of public web evidence for a flagged question.

Search snippets are untrusted evidence, not instructions. The caller must still
validate a correction against them before training. No page URLs are fetched.
"""

import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import os
import re
import socket
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from adaptible._src.wrap.source_lineage import attribution_conflicts

# DDGS 9.16 Yahoo joins nested result cards under the first URL. Never use
# auto/all fallback for training evidence while this extraction path is unsafe.
SEARCH_BACKENDS = ("duckduckgo", "brave", "google", "mojeek", "wikipedia")


logger = logging.getLogger(__name__)


class ReferenceSearchError(RuntimeError):
    """The search provider failed or exceeded the lookup deadline."""


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _plain(value, limit):
    if not isinstance(value, str):
        return ""
    parser = _PlainText()
    parser.feed(value[:20_000])
    return " ".join("".join(parser.parts).split())[:limit]


def _public_url(value):
    """Reject non-web links and explicitly local/private host forms.

    This checks syntax, not DNS: we retain source attribution but never fetch
    these URLs, so arbitrary provider results cannot cause local HTTP requests.
    """
    if not isinstance(value, str) or len(value) > 2048:
        return None
    if any(ord(character) < 33 for character in value) or "\\" in value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return None
        host = (parsed.hostname or "").rstrip(".").encode("idna").decode().lower()
        port = parsed.port
        if not host or "%" in host:
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # inet_aton also recognizes integer, octal and abbreviated IPv4.
            try:
                address = ipaddress.ip_address(socket.inet_aton(host))
            except OSError:
                address = None
            if address is None:
                if "." not in host or host.endswith(
                    (
                        ".localhost",
                        ".local",
                        ".internal",
                        ".lan",
                        ".home",
                        ".test",
                        ".invalid",
                    )
                ):
                    return None
                if any(
                    not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                    for label in host.split(".")
                ):
                    return None
        if address is not None and not address.is_global:
            return None
        netloc = f"[{host}]" if ":" in host else host
        if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
            netloc += f":{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path or "/", parsed.query, ""))
    except (ValueError, UnicodeError):
        return None


def _sources(rows, max_results):
    sources, seen, domains = [], set(), {}
    if not isinstance(rows, list):
        raise ValueError("Search provider returned an invalid result list")
    candidates = []
    for row in rows[:50]:
        if not isinstance(row, dict):
            continue
        url = _public_url(row.get("href") or row.get("url"))
        body = _plain(row.get("body") or row.get("text"), 20_000)
        if not url or len(body) < 20:
            continue
        candidates.append(
            {"title": _plain(row.get("title"), 20_000), "url": url, "text": body}
        )
    # Examine all bounded raw rows before truncation/selection hides the nested
    # card that proves a result's first-URL attribution is contaminated.
    rejected = attribution_conflicts(candidates)
    for index, candidate in enumerate(candidates):
        if index in rejected:
            continue
        url = candidate["url"]
        parsed = urlsplit(url)
        domain = parsed.hostname.removeprefix("www.")
        identity = (domain, parsed.path.rstrip("/"), parsed.query)
        if identity in seen or domains.get(domain, 0) >= 2:
            continue
        seen.add(identity)
        domains[domain] = domains.get(domain, 0) + 1
        sources.append(
            {
                "title": candidate["title"][:200] or domain,
                "url": url,
                "text": candidate["text"][:1800],
            }
        )
        if len(sources) >= max_results:
            break
    return sources


def _search_sync(question, max_results, timeout):
    # Import only in the short-lived worker: DDGS owns threads whose lifetime
    # cannot be bounded by cancelling an asyncio.to_thread call.
    from ddgs import DDGS
    from ddgs.engines import ENGINES
    from ddgs.exceptions import DDGSException

    # DDGS silently falls back to auto when no requested backend exists. Check
    # the installed registry first, so missing support fails closed instead.
    backends = [name for name in SEARCH_BACKENDS if name in ENGINES.get("text", {})]
    if not backends:
        raise ReferenceSearchError("No supported anonymous search backend is installed")

    with DDGS(timeout=max(1, min(8, math.ceil(timeout)))) as provider:
        try:
            rows = provider.text(
                question,
                max_results=min(15, max_results * 3),
                backend=",".join(backends),
            )
        except DDGSException as exc:
            # DDGS reports no usable results with this message, which can also
            # hide non-200 provider responses. It is not proof of an error-free
            # search. Other explicit provider exceptions remain failures.
            if str(exc) == "No results found.":
                return []
            raise
    return _sources(rows, max_results)


class WebReferences:
    """Look up evidence without API keys, another model, or manual files."""

    def __init__(self, *, timeout=20.0, max_results=5):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Search timeout must be positive and finite")
        if not isinstance(max_results, int) or not 1 <= max_results <= 5:
            raise ValueError("Search max_results must be between 1 and 5")
        self.timeout = timeout
        self.max_results = max_results

    async def search(self, question: str) -> list[dict[str, str]]:
        """Search only this question; timeout/errors differ from no results."""
        question = " ".join(question.split())[:600]
        if not question:
            return []
        process = None
        creation = None
        try:
            async with asyncio.timeout(self.timeout):
                creation = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                )
                process = await asyncio.shield(creation)
                stdout, _ = await process.communicate(
                    json.dumps(
                        {
                            "question": question,
                            "max_results": self.max_results,
                            "timeout": self.timeout,
                        }
                    ).encode()
                )
                if process.returncode:
                    raise ReferenceSearchError("Web reference search worker failed")
                if len(stdout) > 64_000:
                    raise ReferenceSearchError(
                        "Web reference search response exceeded its limit"
                    )
                response = json.loads(stdout)
                if response.get("error"):
                    raise ReferenceSearchError(response["error"])
                return _sources(response["results"], self.max_results)
        except TimeoutError as exc:
            raise ReferenceSearchError("Web reference search timed out") from exc
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ReferenceSearchError(
                "Web reference search returned an invalid response"
            ) from exc
        finally:
            # Shield creation so cancellation during spawn cannot lose the
            # process handle and leave its search threads running unattended.
            if process is None and creation is not None:
                try:
                    process = await creation
                except (OSError, asyncio.CancelledError):
                    logger.debug("Search helper never started; nothing to reap.")
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    logger.debug("Search helper already exited before kill().")
                await process.wait()


def _main():
    try:
        request = json.loads(sys.stdin.buffer.read(8192))
        # Keep dependency banners/logging out of the small JSON wire protocol.
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            results = _search_sync(**request)
        response = {"results": results}
    except Exception as exc:
        # Do not leak provider URLs, proxies, or the user's query in failures.
        response = {"error": f"Web reference search failed ({type(exc).__name__})"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    _main()
