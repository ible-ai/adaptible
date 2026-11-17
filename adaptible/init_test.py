"""Keeps the lazy runtime exports and the TYPE_CHECKING imports in `__init__.py` in sync."""

import ast
import pathlib
import subprocess
import sys
import unittest

import adaptible

_INIT = pathlib.Path(adaptible.__file__)


def _type_checking_exports() -> dict[str, tuple[str, str]]:
    """Maps each name imported under `if TYPE_CHECKING:` to its (module, attr) pair."""
    tree = ast.parse(_INIT.read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )
    exports = {}
    for stmt in block.body:
        assert isinstance(stmt, ast.ImportFrom), ast.dump(stmt)
        module = "." * stmt.level + (stmt.module or "")
        for alias in stmt.names:
            exports[alias.asname or alias.name] = (module, alias.name)
    return exports


class InitExportsTest(unittest.TestCase):

    def test_all_matches_runtime_exports(self):
        self.assertCountEqual(adaptible.__all__, adaptible._EXPORTS)

    def test_type_checking_block_matches_runtime_exports(self):
        self.assertEqual(_type_checking_exports(), adaptible._EXPORTS)

    def test_import_does_not_load_submodules(self):
        # `wrap` must import on machines without MLX, so the package import stays lazy.
        code = (
            "import sys, adaptible; "
            "print(sorted(m for m in sys.modules if m.startswith('adaptible.') or m == 'mlx'))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
