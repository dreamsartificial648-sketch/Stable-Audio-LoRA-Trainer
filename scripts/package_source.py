"""Create a shareable source archive from an explicit file allowlist."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT / name for name in (
        "main.py", "pyproject.toml", "requirements.txt", "requirements-runtime.txt", ".gitignore",
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "Setup.bat", "Launch.bat",
        "Install training runtime.bat", "Login to Hugging Face.bat")]
    for folder in ("sfx_tuner", "scripts", "tests", "docs", ".github"):
        files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    target = ROOT / "dist/Stable-Audio-Fine-Tuner-v0.1.0-source.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "Stable-Audio-Fine-Tuner/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert all(not any(part in ("runtime", "workspace", ".venv", "local-settings.json") for part in Path(name).parts) for name in archive.namelist())
        assert archive.testzip() is None
    print(f"Source archive: {target}\nFiles: {len(files)}\nBytes: {target.stat().st_size}")


if __name__ == "__main__":
    main()
