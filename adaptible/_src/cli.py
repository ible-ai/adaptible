"""Terminal client for the local server.

    python -m adaptible.cli [--url http://127.0.0.1:8000] [line ...]

Typed text is a question, streamed back as the model writes it. Commands:
``/down`` and ``/up`` rate the last answer, ``/review`` repairs the flagged
answers and waits for it, ``/new`` starts a new chat, ``/quit`` exits. Lines
given as arguments run in order instead of the interactive loop.
"""

import argparse
import logging
import sys

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"
# Colour per kind of line, as an ANSI SGR code; used only when stdout is a TTY. The demo
# renderer mirrors this table.
COLORS = {
    "prompt": "32",  # what the user types
    "think": "2",  # the model's reasoning, dim
    "answer": "0",  # the final answer, default text
    "down": "31",
    "up": "32",
    "review": "33",
    "new": "36",
    "error": "31",
}
RESET = "\033[0m"
THINK_END = "</think>"
# Importing adaptible configures the root logger; keep httpx's request lines out of the chat.
logging.getLogger("httpx").setLevel(logging.WARNING)
HELP = "commands: /down  /up  /review  /new  /quit   (anything else is a question)"


class Client:
    """One session against the server; ``run`` handles a typed line."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        http: httpx.Client | None = None,
        out=None,
        color: bool | None = None,
    ):
        self.http = http or httpx.Client(base_url=url, timeout=httpx.Timeout(None, connect=5))
        self.url = url
        self.out = out or sys.stdout
        self.color = bool(getattr(self.out, "isatty", lambda: False)()) if color is None else color
        self.last_idx: int | None = None

    def paint(self, text: str, kind: str) -> str:
        """Wraps ``text`` in the colour for ``kind`` when colour is on."""
        if not self.color or kind == "answer" or not text:
            return text
        return f"\033[{COLORS[kind]}m{text}{RESET}"

    def _print(self, text: str = "", end: str = "\n", kind: str = "answer") -> None:
        self.out.write(self.paint(text, kind) + end)
        self.out.flush()

    def ask(self, prompt: str) -> str:
        """Streams the answer; reasoning (up to ``</think>``) is dim, the answer is plain."""
        chunks, pending, thinking = [], "", True
        with self.http.stream("POST", "/stream_interact", json={"prompt": prompt}) as r:
            r.raise_for_status()
            self.last_idx = int(r.headers["X-Interaction-Idx"])
            for chunk in r.iter_text():
                chunks.append(chunk)
                if not thinking:
                    self._print(chunk, end="")
                    continue
                pending += chunk
                if THINK_END in pending:
                    head, _, tail = pending.partition(THINK_END)
                    self._print(head + THINK_END, end="", kind="think")
                    self._print(tail, end="")
                    pending, thinking = "", False
                else:
                    # Hold back a partial tag so its colour is not split across chunks.
                    keep = next((n for n in range(len(THINK_END) - 1, 0, -1) if pending.endswith(THINK_END[:n])), 0)
                    self._print(pending[: len(pending) - keep], end="", kind="think")
                    pending = pending[len(pending) - keep :]
        self._print(pending, end="", kind="think" if thinking else "answer")
        self._print()
        return "".join(chunks)

    def rate(self, thumbs: str) -> None:
        if self.last_idx is None:
            self._print("nothing to rate yet", kind="error")
            return
        r = self.http.post("/feedback", json={"interaction_idx": self.last_idx, "thumbs": thumbs})
        r.raise_for_status()
        flagged = r.json()["flagged"]
        self._print(
            f"[{thumbs}] answer {self.last_idx} {'flagged for review' if flagged else 'unflagged'}",
            kind=thumbs,
        )

    def review(self) -> None:
        r = self.http.post("/trigger_review")
        r.raise_for_status()
        n = r.json()["unreviewed_count"]
        if n == 0:
            self._print("[review] nothing to review", kind="review")
            return
        self._print(f"[review] started on {n} answer(s), waiting...", kind="review")
        r = self.http.get("/sync")
        r.raise_for_status()
        self._print(f"[review] done in {r.json()['elapsed_time']:.0f} s", kind="review")

    def new_chat(self) -> None:
        self.http.post("/new_chat").raise_for_status()
        self.last_idx = None
        self._print("[new chat]", kind="new")

    def run(self, line: str) -> bool:
        """Handles one line; returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        if line in ("/quit", "/exit"):
            return False
        if line == "/down":
            self.rate("down")
        elif line == "/up":
            self.rate("up")
        elif line == "/review":
            self.review()
        elif line == "/new":
            self.new_chat()
        elif line == "/help":
            self._print(HELP)
        elif line.startswith("/"):
            self._print(f"unknown command {line}; {HELP}", kind="error")
        else:
            self.ask(line)
        return True


def prompt_kind(line: str) -> str:
    """Colour kind for a typed line: commands take their own colour, questions are prompts."""
    return line.strip()[1:] if line.strip() in ("/down", "/up", "/review", "/new") else "prompt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m adaptible.cli", description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="server address")
    parser.add_argument("lines", nargs="*", help="lines to run instead of reading stdin")
    args = parser.parse_args(argv)
    client = Client(args.url)
    try:
        client.http.get("/status").raise_for_status()
    except httpx.HTTPError:
        print(client.paint(f"no server at {args.url}; start one with `python -m adaptible.local`", "error"), file=sys.stderr)
        return 1
    try:
        if args.lines:
            for line in args.lines:
                print(client.paint(f"> {line}", prompt_kind(line)))
                if not client.run(line):
                    break
            return 0
        print(f"connected to {args.url}; {HELP}")
        while True:
            try:
                # The terminal echoes what is typed, so open the colour before reading.
                line = input(client.paint("> ", "prompt") + ("\033[%sm" % COLORS["prompt"] if client.color else ""))
            except EOFError:
                print()
                break
            finally:
                if client.color:
                    print(RESET, end="", flush=True)
            if not client.run(line):
                break
        return 0
    except httpx.HTTPError as e:
        print(client.paint(f"server error: {e}", "error"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
