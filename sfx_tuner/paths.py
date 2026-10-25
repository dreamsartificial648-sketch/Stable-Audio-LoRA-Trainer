"""Local paths; settings never contain Hugging Face tokens."""
from pathlib import Path
import os
import sys


def python_in(folder):
    return Path(folder) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def default_root():
    return Path(__file__).resolve().parents[1]


def runtime_python(root):
    return python_in(Path(root) / "runtime/.venv")


def launch_environment():
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_PROGRESS_BARS="1", WANDB_MODE="disabled")
    # Respect the user's existing HF_HOME and cached authentication.
    return env


def base_python():
    return getattr(sys, "_base_executable", sys.executable)
