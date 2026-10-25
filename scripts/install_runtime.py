"""Install an isolated, pinned Small-SFX backend. No model weights are bundled."""
import argparse
from pathlib import Path
import os
import subprocess
import shutil
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sfx_tuner.paths import python_in
from sfx_tuner.constants import UPSTREAM_REVISION


def run(command, cwd=ROOT):
    print("\n> " + " ".join(str(x) for x in command), flush=True)
    subprocess.run([str(x) for x in command], cwd=str(cwd), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-only", action="store_true")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use Python 3.10, 3.11, or 3.12 for this release.")
    if args.ui_only:
        target = ROOT / ".venv"
        if not python_in(target).exists():
            venv.create(target, with_pip=True)
        run([python_in(target), "-m", "pip", "install", "--upgrade", "pip"])
        run([python_in(target), "-m", "pip", "install", "-e", str(ROOT)])
        print("Desktop installed. Run Launch.bat.", flush=True)
        return
    runtime = ROOT / "runtime"
    if not shutil.which("git"):
        raise RuntimeError("Git is required for the training runtime. Install it, add it to PATH, and restart the app.")
    runtime.mkdir(exist_ok=True)
    upstream = runtime / "upstream"
    if not upstream.exists():
        run(["git", "clone", "https://github.com/Stability-AI/stable-audio-3.git", str(upstream)])
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if actual != UPSTREAM_REVISION:
        # Never overwrite local modifications in the dependency checkout.
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream, text=True).strip()
        if dirty:
            raise RuntimeError("runtime/upstream has local changes. Restore them before installing another revision.")
        run(["git", "checkout", "--detach", UPSTREAM_REVISION], cwd=upstream)
    target = runtime / ".venv"
    if not python_in(target).exists():
        venv.create(target, with_pip=True)
    python = python_in(target)
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "torch==2.7.1", "torchaudio==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126"])
    run([python, "-m", "pip", "install", "-r", ROOT / "requirements-runtime.txt", "-e", str(upstream) + "[lora]"])
    run([python, "-m", "pip", "check"])
    run([python, "-c", "import torch, torchaudio; from stable_audio_3 import StableAudioModel; from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper; print('Runtime imports passed. CUDA:', torch.cuda.is_available())"])
    with (runtime / "installed-packages.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([str(python), "-m", "pip", "freeze"], stdout=handle, check=True)
    print("\nRuntime installed. Open Setup > Check runtime in the app.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
