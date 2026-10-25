from pathlib import Path

import pytest

from sfx_tuner.audio import atomic_json
from sfx_tuner.constants import ARCHITECTURE, BASE_MODEL
from sfx_tuner.contracts import checkpoint_info
from sfx_tuner.requests import validate
from test_data import make_dataset


def generate_request(tmp_path):
    return dict(operation="generate", output_dir=str(tmp_path / "new"), model_path="stable-audio-3:base",
                prompt="A bell rings", duration=2., guidance=7., lora_strength=1., steps=10, seed=0, clip_count=1)


@pytest.mark.parametrize("key,value", [("duration", float("nan")), ("duration", 121), ("steps", 1.5),
                                      ("steps", 0), ("seed", -1), ("seed", True), ("lora_strength", 3)])
def test_rejects_unsafe_or_unusable_generation_settings(tmp_path, key, value):
    request = generate_request(tmp_path)
    request[key] = value
    with pytest.raises(ValueError):
        validate(request)


def test_generation_requires_prompt_and_new_output_folder(tmp_path):
    request = generate_request(tmp_path)
    assert validate(request)["seed"] == 0
    with pytest.raises(ValueError, match="Describe"):
        validate({**request, "prompt": " "})
    Path(request["output_dir"]).mkdir()
    with pytest.raises(ValueError, match="never overwritten"):
        validate(request)


def test_continuation_uses_adapter_rank_and_rejects_incompatible_model(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    folder = tmp_path / "adapter"
    atomic_json(folder / "audio_model.json", dict(architecture=ARCHITECTURE, base_model=BASE_MODEL, rank=4))
    (folder / "adapter.safetensors").write_bytes(b"mock adapter for request validation")
    request = dict(operation="train", output_dir=str(tmp_path / "new"), dataset_dir=str(dataset),
                   epochs=1, batch_size=1, rank=16, save_every=1, seed=42, learning_rate=.0001, resume_from=str(folder))
    assert validate(request)["rank"] == 4
    atomic_json(folder / "audio_model.json", dict(architecture="other", base_model=BASE_MODEL, rank=4))
    with pytest.raises(ValueError, match="Small-SFX LoRA"):
        validate(request)


def test_corrupt_adapter_rank_is_rejected(tmp_path):
    atomic_json(tmp_path / "audio_model.json", dict(architecture=ARCHITECTURE, base_model=BASE_MODEL, rank="four"))
    (tmp_path / "adapter.safetensors").touch()
    with pytest.raises(ValueError, match="Small-SFX LoRA"):
        checkpoint_info(tmp_path)
