"""Adaptible command line entry point."""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "wrap":
        from ._src.wrap.__main__ import main as wrap

        return wrap(sys.argv[2:])
    print("Usage: adaptible wrap {ollama,llama-cpp,lm-studio,vllm} MODEL [options]")
    return 0 if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help") else 2


if __name__ == "__main__":
    raise SystemExit(main())
