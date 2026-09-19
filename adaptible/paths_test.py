"""Tests that persisted state honors ``$ADAPTIBLE_OUTPUTS_DIR`` (model-free)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adaptible import paths
from adaptible.db import Database


class OutputsDirTest(unittest.TestCase):
    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"ADAPTIBLE_OUTPUTS_DIR": "/some/where"}):
            self.assertEqual(paths.outputs_dir(), Path("/some/where"))
            self.assertEqual(paths.default_db_path(), Path("/some/where/adaptible.db"))
            self.assertEqual(
                paths.default_checkpoint_path(),
                Path("/some/where/autonomous/checkpoint"),
            )
            self.assertEqual(
                paths.autonomous_state_path(),
                Path("/some/where/autonomous/state.json"),
            )
            self.assertEqual(
                paths.autonomous_log_dir(), Path("/some/where/autonomous/logs")
            )

    def test_defaults_to_cwd_outputs(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADAPTIBLE_OUTPUTS_DIR", None)
            self.assertEqual(paths.outputs_dir(), Path.cwd() / "outputs")

    def test_never_resolves_into_package_dir(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADAPTIBLE_OUTPUTS_DIR", None)
            package_dir = Path(paths.__file__).parent
            self.assertNotIn(package_dir, paths.default_db_path().parents)


class DatabaseDefaultPathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_database_default_uses_outputs_dir_at_construction(self):
        with mock.patch.dict(os.environ, {"ADAPTIBLE_OUTPUTS_DIR": str(self.tmp)}):
            db = Database()
        expected = self.tmp / "adaptible.db"
        self.assertEqual(db.db_path, expected)
        self.assertTrue(expected.exists(), "Database() should create the file")

    def test_explicit_db_path_wins(self):
        explicit = self.tmp / "explicit" / "custom.db"
        with mock.patch.dict(
            os.environ, {"ADAPTIBLE_OUTPUTS_DIR": str(self.tmp / "ignored")}
        ):
            db = Database(db_path=explicit)
        self.assertEqual(db.db_path, explicit)
        self.assertTrue(explicit.exists())
        self.assertFalse((self.tmp / "ignored" / "adaptible.db").exists())

    def test_env_change_after_import_is_honored(self):
        """The default is resolved per construction, not at import time."""
        first = self.tmp / "a"
        second = self.tmp / "b"
        with mock.patch.dict(os.environ, {"ADAPTIBLE_OUTPUTS_DIR": str(first)}):
            Database()
        with mock.patch.dict(os.environ, {"ADAPTIBLE_OUTPUTS_DIR": str(second)}):
            Database()
        self.assertTrue((first / "adaptible.db").exists())
        self.assertTrue((second / "adaptible.db").exists())


if __name__ == "__main__":
    unittest.main()
