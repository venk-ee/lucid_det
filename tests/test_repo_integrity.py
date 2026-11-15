import importlib
import os
import pkgutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PACKAGES = [
    "assigners",
    "backbones",
    "core",
    "datasets",
    "engine",
    "heads",
    "losses",
    "models",
    "necks",
    "runners",
    "training_scripts",
    "transforms",
    "utils",
]


@pytest.mark.parametrize("package_name", PACKAGES)
def test_import_package_and_submodules(package_name: str) -> None:
    """Recursively import all submodules in package to detect syntax or import errors."""
    pkg = importlib.import_module(package_name)
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is not None:
        for _, module_name, is_pkg in pkgutil.walk_packages(pkg_path, prefix=f"{package_name}."):
            submod = importlib.import_module(module_name)
            assert submod is not None
