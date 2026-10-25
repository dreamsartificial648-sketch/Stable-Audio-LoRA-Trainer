"""Explicit functional GPU check using synthetic data; writes diagnostic artifacts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sfx_tuner.audio import atomic_json, read_json, read_pcm, write_pcm
from sfx_tuner.paths import runtime_python
from sfx_tuner.protocol import PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-python", type=Path, default=runtime_python(ROOT))
    parser.add_argument("--hf-home", type=Path, help="Use an existing Hugging Face cache for this check")
    parser.add_argument("--offline", action="store_true", help="Fail instead of downloading uncached weights")
    args = parser.parse_args()
    run_root = ROOT / "workspace/diagnostics" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    run_root.mkdir(parents=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", WANDB_MODE="disabled", HF_HUB_DISABLE_PROGRESS_BARS="1")
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home.resolve())
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
    worker = ROOT / "sfx_tuner/worker.py"
    results = []

    def run(name, request, stop_after_first_batch=False):
        print(f"\n=== {name} ===", flush=True)
        request.update(stop_file=str(run_root / f"{name}.stop"), pause_file=str(run_root / f"{name}.pause"))
        path = run_root / f"{name}.json"
        atomic_json(path, request)
        if stop_after_first_batch:
            # The worker pauses after its first batch until this parent requests Stop.
            Path(request["pause_file"]).touch()
        events = []
        with (run_root / f"{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([str(args.runtime_python.resolve()), str(worker), "--request", str(path)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(PREFIX):
                        event = json.loads(line[len(PREFIX):])
                        events.append(event)
                        print(line.strip(), flush=True)
                        if stop_after_first_batch and event.get("event") == "step":
                            Path(request["stop_file"]).touch()
                code = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        if code or not events or events[-1].get("event") != "done":
            raise RuntimeError(f"{name} failed (exit {code}). Inspect {run_root / (name + '.log')}")
        results.append(dict(name=name, passed=True))

    def check_wav(path):
        values, rate = read_pcm(path)
        assert rate == 44100 and values.shape == (2, 88200)
        assert np.isfinite(values).all() and float(np.max(np.abs(values))) > .0001

    try:
        run("doctor", dict(operation="doctor"))
        gen = dict(operation="generate", model_path="stable-audio-3:base", model_name="Base diagnostic",
                   duration=2, clip_count=1, steps=10, seed=42,
                   prompt="Four short high pitched electronic beeps with silence between them.",
                   negative_prompt="", guidance=7., lora_strength=1.)
        run("base", {**gen, "output_dir": str(run_root / "base")})
        check_wav(run_root / "base/clip_001.wav")
        dataset = run_root / "dataset"
        dataset.mkdir()
        values = np.zeros((2, 88200), dtype=np.float32)
        time = np.arange(4410) / 44100
        for start in (0, 22050, 44100, 66150):
            values[:, start:start + len(time)] = .2 * np.sin(2 * np.pi * 880 * time) * np.exp(-40 * time)
        write_pcm(dataset / "beeps.wav", values, 44100)
        (dataset / "beeps.txt").write_text(gen["prompt"], encoding="utf-8")
        atomic_json(dataset / "dataset.json", dict(complete=True, name="Diagnostic beeps", sample_rate=44100,
                    channels=2, clip_seconds=2, training_clips=1, validation_clips=0,
                    clips=[dict(path="beeps.wav", split="train")]))
        train = dict(operation="train", dataset_dir=str(dataset), model_name="Diagnostic adapter", epochs=1,
                     batch_size=1, learning_rate=.0001, rank=4, save_every=1, seed=12345, resume_from="")
        run("train", {**train, "output_dir": str(run_root / "adapter")})
        info = read_json(run_root / "adapter/audio_model.json")
        assert info["epochs_completed"] == 1 and info["steps_completed"] == 1 and info["base_frozen"]
        assert (run_root / "adapter/adapter.safetensors").stat().st_size > 0
        run("continue", {**train, "resume_from": str(run_root / "adapter"), "rank": 8, "output_dir": str(run_root / "continued")})
        continued = read_json(run_root / "continued/audio_model.json")
        assert continued["epochs_completed"] == 2 and continued["rank"] == 4
        run("stop_save", {**train, "epochs": 3, "output_dir": str(run_root / "stopped")}, stop_after_first_batch=True)
        stopped = read_json(run_root / "stopped/audio_model.json")
        assert stopped["stopped"] and stopped["steps_completed"] == 1 and stopped["epochs_completed"] == 0
        assert (run_root / "stopped/adapter.safetensors").stat().st_size > 0
        run("adapter_generation", {**gen, "model_path": str(run_root / "continued"), "output_dir": str(run_root / "generated")})
        check_wav(run_root / "generated/clip_001.wav")
        atomic_json(run_root / "summary.json", dict(passed=True, checks=results, training_peak_vram_gib=info["peak_vram_bytes"] / 2**30))
        print(f"\nAll GPU checks passed. Peak allocated training VRAM: {info['peak_vram_bytes'] / 2**30:.2f} GiB\nArtifacts: {run_root}", flush=True)
    except Exception:
        atomic_json(run_root / "summary.json", dict(passed=False, checks=results))
        raise


if __name__ == "__main__":
    main()
