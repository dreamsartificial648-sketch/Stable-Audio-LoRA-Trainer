"""Validate requests before starting expensive operations."""
import math
from pathlib import Path

from sfx_tuner.contracts import checkpoint_info, training_rows


def number(request, key, low, high, integer=False):
    value = request[key]
    if isinstance(value, bool):
        raise ValueError(f"{key}: enter a number.")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high or (integer and value != int(value)):
        raise ValueError(f"{key}: expected {'an integer' if integer else 'a number'} from {low} to {high}.")
    request[key] = int(value) if integer else value


def validate(request):
    request = dict(request)
    operation = request.get("operation")
    if operation in ("doctor", "prepare"):
        return request
    if operation not in ("train", "generate"):
        raise ValueError("Unknown operation.")
    number(request, "seed", 0, 2147483647, True)
    if Path(request["output_dir"]).exists():
        raise ValueError("Choose a new output folder; existing results are never overwritten.")
    if operation == "train":
        for key, low, high in [("epochs", 1, 100000), ("batch_size", 1, 256), ("rank", 1, 64), ("save_every", 1, 1000)]:
            number(request, key, low, high, True)
        number(request, "learning_rate", 1e-7, .01)
        training_rows(Path(request["dataset_dir"]))
        if request.get("resume_from"):
            request["rank"] = checkpoint_info(Path(request["resume_from"]))["rank"]
    else:
        for key, low, high in [("duration", .5, 120), ("guidance", 1, 20), ("lora_strength", 0, 2)]:
            number(request, key, low, high)
        for key, low, high in [("clip_count", 1, 16), ("steps", 5, 250)]:
            number(request, key, low, high, True)
        if not request.get("prompt", "").strip():
            raise ValueError("Describe the sound you want to generate.")
        if request["model_path"] != "stable-audio-3:base":
            checkpoint_info(Path(request["model_path"]))
    return request
