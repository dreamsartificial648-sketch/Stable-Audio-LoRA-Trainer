"""Process entrypoint. ML imports stay out of the desktop process."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx_tuner.audio import atomic_json, prepare_dataset, write_pcm
from sfx_tuner.contracts import ARCHITECTURE, BASE_MODEL, checkpoint_info
from sfx_tuner.protocol import emit, stopped, wait_if_paused
from sfx_tuner.requests import validate


def doctor(request):
    import torch
    import stable_audio_3
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper
    from huggingface_hub import get_token

    cuda = torch.cuda.is_available()
    result = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=cuda,
                  gpu=torch.cuda.get_device_name(0) if cuda else "No CUDA device",
                  vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else 0,
                  bf16=torch.cuda.is_bf16_supported() if cuda else False,
                  huggingface_login=bool(get_token()), ffmpeg=bool(shutil.which("ffmpeg")),
                  upstream=str(Path(stable_audio_3.__file__).resolve().parent),
                  lightning=importlib.metadata.version("pytorch-lightning"))
    emit(event="diagnostics", **result)
    emit(event="done", message="Runtime checked. Model access is verified when weights load.")


def generate(request):
    import torch
    from stable_audio_3 import StableAudioModel

    if not torch.cuda.is_available():
        raise ValueError("This release requires an NVIDIA CUDA GPU for generation.")
    emit(event="log", message="Loading Small-SFX base model and text encoder. First use downloads weights.")
    model = StableAudioModel.from_pretrained(BASE_MODEL, device="cuda", model_half=False)
    if request["model_path"] != "stable-audio-3:base":
        checkpoint_info(Path(request["model_path"]))
        model.load_lora([str(Path(request["model_path"]) / "adapter.safetensors")])
        model.set_lora_strength(request["lora_strength"])
    if stopped(request):
        raise InterruptedError("Stopped before generation.")
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    record = {key: value for key, value in request.items() if key not in ("stop_file", "pause_file")}
    record.update(architecture=ARCHITECTURE, base_model=BASE_MODEL, sample_rate=44100, channels=2,
                  files=[], created_at=datetime.now(timezone.utc).isoformat(), inference_precision="float32")
    frames = round(request["duration"] * 44100)
    for index in range(request["clip_count"]):
        wait_if_paused(request)
        if stopped(request):
            raise InterruptedError("Generation stopped.")

        def progress(payload):
            wait_if_paused(request)
            if stopped(request):
                raise InterruptedError("Generation stopped.")
            emit(event="generation_step", current=index * request["steps"] + int(payload["i"]) + 1,
                 total=request["clip_count"] * request["steps"])

        audio = model.generate(prompt=request["prompt"].strip(), negative_prompt=request.get("negative_prompt") or None,
                               duration=math.nextafter(frames / 44100, math.inf), steps=request["steps"],
                               cfg_scale=request["guidance"], seed=(request["seed"] + index) % 2147483648,
                               batch_size=1, chunked_decode=True, callback=progress, disable_tqdm=True)
        if not torch.isfinite(audio).all() or audio.shape[-1] < frames:
            raise ValueError("Model returned invalid or incomplete audio.")
        if stopped(request):
            raise InterruptedError("Stopped before WAV export.")
        path = output / f"clip_{index + 1:03d}.wav"
        write_pcm(path, audio[0, :, :frames].float().cpu().numpy(), 44100)
        record["files"].append(path.name)
        atomic_json(output / "generation.json", record)
        emit(event="preview", path=str(path))
    emit(event="done", output_dir=str(output), files=record["files"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    if os.environ.get("HF_HOME") and not os.environ.get("HF_TOKEN") and not os.environ.get("HF_TOKEN_PATH"):
        # Reuse the ordinary login if only the cache location was customized.
        normal_token = Path.home() / ".cache/huggingface/token"
        if normal_token.is_file():
            os.environ["HF_TOKEN_PATH"] = str(normal_token)
    try:
        request = validate(json.loads(Path(args.request).read_text(encoding="utf-8")))
        if stopped(request):
            raise InterruptedError("Stopped before starting.")
        if request["operation"] == "doctor":
            doctor(request)
        elif request["operation"] == "prepare":
            def gate():
                wait_if_paused(request)
                return stopped(request)
            prepare_dataset(request, emit, gate)
        elif request["operation"] == "train":
            from sfx_tuner.training import train
            train(request)
        else:
            generate(request)
        return 0
    except InterruptedError as exc:
        emit(event="stopped", message=str(exc))
        return 0
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        if "GatedRepo" in type(exc).__name__ or "401" in message or "403" in message:
            message = "Model access denied. Accept the Small-SFX model terms on Hugging Face, then run Login to Hugging Face.bat and retry."
        elif "out of memory" in message.lower():
            message = "GPU memory exhausted. Reduce batch size or training clip length, close other GPU applications, and retry."
        emit(event="error", message=message)
        return 1


if __name__ == "__main__":
    sys.exit(main())
