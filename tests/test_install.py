import subprocess
import sys
from pathlib import Path


def test_installer_help_works_without_site_packages():
    # Fresh users run this before NumPy or Qt is installed. -S removes site packages.
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", str(root / "scripts/install_runtime.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--ui-only" in result.stdout
