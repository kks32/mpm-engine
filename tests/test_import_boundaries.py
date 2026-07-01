"""Hard module boundary: ident/ must not import sim, warp, torch, jax, taichi.

Walks every file under ident/ and FAILS on any such import. The only
exception, Phase 3 only, is torch under ident/features/function_encoder_training/.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENT = ROOT / "src" / "ident"
FORBIDDEN = {"sim", "warp", "torch", "jax", "taichi"}
TORCH_EXCEPTION_DIR = IDENT / "features" / "function_encoder_training"


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def test_ident_never_imports_simulation_stacks():
    violations = []
    for path in sorted(IDENT.rglob("*.py")):
        allowed_exception = (
            TORCH_EXCEPTION_DIR in path.parents and path.suffix == ".py"
        )
        names = _imported_top_levels(path)
        bad = names & FORBIDDEN
        if allowed_exception:
            bad = bad - {"torch"}
        if bad:
            violations.append(f"{path.relative_to(ROOT)}: {sorted(bad)}")
    assert not violations, "forbidden imports in ident/:\n" + "\n".join(violations)


def test_walk_found_files():
    files = list(IDENT.rglob("*.py"))
    assert len(files) >= 10, "import-boundary walk found suspiciously few files"
