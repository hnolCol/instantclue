import importlib
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDED = {
    "codeExample.py",
    "chord_diagram.py",
    "main.py",
}


def collect_modules():
    modules = []
    for f in sorted(REPO_ROOT.rglob("*.py")):
        rel = f.relative_to(REPO_ROOT)
        if any(part in EXCLUDED or part.startswith((".", "env", "__pycache__")) for part in rel.parts):
            continue
        if f.name in EXCLUDED:
            continue
        modules.append(".".join(rel.with_suffix("").parts))
    return modules


MODULES = collect_modules()


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
