"""Durable wrapper history and the currently accepted adapter."""

import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, directory: Path, identity: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.db = sqlite3.connect(directory / "history.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY, messages TEXT NOT NULL, response TEXT NOT NULL,
                created REAL NOT NULL, flagged INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'new', reason TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '', reference_data TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS accepted_repairs (
                interaction_idx INTEGER PRIMARY KEY, data TEXT NOT NULL
            );
        """
        )
        if self.get("identity") not in (None, identity):
            self.db.close()
            raise ValueError(
                "This state directory belongs to a different model. Use --state-dir for a new wrapper."
            )
        self.set("identity", identity)
        # Existing wrappers retain their history when automatic lookup is added.
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(interactions)")}
        if "reference_data" not in columns:
            self.db.execute(
                "ALTER TABLE interactions ADD COLUMN reference_data TEXT NOT NULL DEFAULT '{}'"
            )
        # Phrasings the caller wants this correction judged against, if any.
        if "reask_prompts" not in columns:
            self.db.execute(
                "ALTER TABLE interactions ADD COLUMN reask_prompts TEXT NOT NULL DEFAULT '[]'"
            )
        # Entities that contain the expected term but are still wrong.
        if "wrong_terms" not in columns:
            self.db.execute(
                "ALTER TABLE interactions ADD COLUMN wrong_terms TEXT NOT NULL DEFAULT '[]'"
            )
        for name in ("generation_mode", "response_details"):
            if name not in columns:
                self.db.execute(
                    f"ALTER TABLE interactions ADD COLUMN {name} TEXT NOT NULL DEFAULT '{{}}'"
                )
        # A stopped process must never silently lose a pending repair.
        self.db.execute(
            "UPDATE interactions SET status='pending', reason='Interrupted; queued again' WHERE status='reviewing'"
        )
        self.db.commit()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value))
            )

    def repairs(self):
        """Return durable accepted examples, independent of later feedback edits."""
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT data FROM accepted_repairs ORDER BY interaction_idx"
            )
        ]

    def bind_serving_template(self, digest):
        """Pin runtime-owned formatting that is outside the base checkpoint."""
        previous = self.get("serving_template_digest")
        if previous is None and self.get("accepted"):
            raise ValueError(
                "The saved adapter has no serving-template identity. Use a new "
                "--state-dir to establish a verified training/serving format."
            )
        if previous is not None and previous != digest:
            raise ValueError(
                "The model's serving template or system prompt changed. Use a "
                "new --state-dir; the saved adapter belongs to its previous format."
            )
        if previous is None:
            self.set("serving_template_digest", digest)

    def accept_repair(self, accepted, repair):
        """Commit weights and their retention evidence in the same transaction.

        A crash before the interaction's outcome is written must not leave the
        next review unaware of an adapter update that is already durable.
        """
        interaction_idx = repair["interaction_idx"]
        # Serialize before opening the transaction, including optional source
        # evidence and held-out prompts supplied by the controller.
        encoded_accepted, encoded_repair = json.dumps(accepted), json.dumps(repair)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('accepted', ?)",
                (encoded_accepted,),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO accepted_repairs VALUES (?, ?)",
                (interaction_idx, encoded_repair),
            )

    def record(self, messages, response, *, generation_mode=None):
        with self.db:
            cur = self.db.execute(
                "INSERT INTO interactions(messages,response,created,generation_mode) VALUES (?,?,?,?)",
                (
                    json.dumps(messages),
                    response,
                    time.time(),
                    json.dumps(generation_mode or {}),
                ),
            )
        return cur.lastrowid

    def rows(self):
        return [
            self.decode(r)
            for r in self.db.execute("SELECT * FROM interactions ORDER BY id")
        ]

    @staticmethod
    def decode(row):
        d = dict(row)
        d["messages"] = json.loads(d["messages"])
        d["references"] = json.loads(d.pop("reference_data", "{}"))
        for name in ("generation_mode", "response_details"):
            d[name] = json.loads(d.get(name, "{}"))
        d["reask_prompts"] = json.loads(d.get("reask_prompts") or "[]")
        d["wrong_terms"] = json.loads(d.get("wrong_terms") or "[]")
        d["flagged"] = bool(d["flagged"])
        d["interaction_idx"] = d["id"]
        return d

    def feedback(self, idx, down, note="", reask_prompts=None, wrong_terms=None):
        row = self.db.execute(
            "SELECT status FROM interactions WHERE id=?", (idx,)
        ).fetchone()
        if row is None:
            raise KeyError(idx)
        if row[0] == "reviewing":
            raise ValueError("This answer is already being reviewed.")
        with self.db:
            self.db.execute(
                "UPDATE interactions SET flagged=?, status=?, reason='', note=?, "
                "reference_data='{}', reask_prompts=?, wrong_terms=? WHERE id=?",
                (
                    int(down),
                    "pending" if down else "new",
                    note,
                    json.dumps(list(reask_prompts or [])),
                    json.dumps(list(wrong_terms or [])),
                    idx,
                ),
            )

    def pending(self):
        return [
            self.decode(r)
            for r in self.db.execute(
                "SELECT * FROM interactions WHERE flagged=1 AND status='pending' ORDER BY id"
            )
        ]

    def outcome(self, idx, status, reason):
        with self.db:
            self.db.execute(
                "UPDATE interactions SET status=?, reason=? WHERE id=?",
                (status, reason, idx),
            )

    def references(self, idx, data):
        with self.db:
            self.db.execute(
                "UPDATE interactions SET reference_data=? WHERE id=?",
                (json.dumps(data), idx),
            )

    def close(self):
        self.db.close()
