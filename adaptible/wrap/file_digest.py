"""SHA-256 of an open binary file, for Pythons without hashlib.file_digest (3.10)."""

import hashlib


def sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest
