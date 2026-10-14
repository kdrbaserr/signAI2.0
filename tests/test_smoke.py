"""Check the installed distribution, not an injected source directory."""

from importlib.metadata import version
import subprocess
import sys

import signai


def test_installed_package_version() -> None:
    assert signai.__version__ == version("signai-2.0") == "0.1.0"


def test_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "signai"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "signai-2.0 0.1.0: environment ready"
