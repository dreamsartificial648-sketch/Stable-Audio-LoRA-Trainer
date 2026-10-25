import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest

from sfx_tuner.audio import atomic_json, prepare_dataset, read_json, read_pcm, write_pcm
from sfx_tuner.contracts import caption_rows, save_captions, training_rows


def make_dataset(root, caption="A metallic click"):
    root.mkdir(parents=True, exist_ok=True)
    write_pcm(root / "clip.wav", np.zeros((2, 22050), np.float32), 44100)
    (root / "clip.txt").write_text(caption, encoding="utf-8")
    atomic_json(root / "dataset.json", dict(name="Test clips", complete=True, clip_seconds=.5,
                sample_rate=44100, channels=2, training_clips=1, validation_clips=0,
                clips=[dict(path="clip.wav", split="train")]))
    return root


def test_captions_and_holdout_are_isolated(tmp_path):
    folder = make_dataset(tmp_path / "dataset", caption="")
    write_pcm(folder / "held.wav", np.zeros((2, 22050), np.float32), 44100)
    info = read_json(folder / "dataset.json")
    info["clips"].append(dict(path="held.wav", split="validation"))
    atomic_json(folder / "dataset.json", info)
    with pytest.raises(ValueError, match="Missing 1 caption"):
        training_rows(folder)
    save_captions(folder, [(str(folder / "clip.wav"), "A click")])
    assert [r["caption"] for r in training_rows(folder)] == ["A click"]
    with pytest.raises(ValueError, match="selected dataset"):
        save_captions(folder, [(str(tmp_path / "outside.wav"), "Outside")])


def test_clip_and_caption_paths_cannot_escape(tmp_path):
    folder = make_dataset(tmp_path / "dataset")
    write_pcm(tmp_path / "outside.wav", np.zeros((2, 22050), np.float32), 44100)
    info = read_json(folder / "dataset.json")
    info["clips"][0]["path"] = "../outside.wav"
    atomic_json(folder / "dataset.json", info)
    with pytest.raises(ValueError, match="outside"):
        caption_rows(folder)


def test_caption_symlink_is_rejected(tmp_path):
    folder = make_dataset(tmp_path / "dataset")
    (tmp_path / "private.txt").write_text("keep", encoding="utf-8")
    (folder / "clip.txt").unlink()
    try:
        (folder / "clip.txt").symlink_to(tmp_path / "private.txt")
    except OSError:
        pytest.skip("OS does not permit symlinks for this user")
    with pytest.raises(ValueError, match="outside"):
        save_captions(folder, [(str(folder / "clip.wav"), "replace")])
    assert (tmp_path / "private.txt").read_text() == "keep"


def test_ffmpeg_import_preserves_sources_inherits_captions_and_groups_splits(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.fail("FFmpeg is required for the dataset integration check")
    time = np.arange(16000) / 16000
    source = tmp_path / "tone.wav"
    other = tmp_path / "second.wav"
    duplicate = tmp_path / "duplicate.wav"
    write_pcm(source, .2 * np.sin(2 * np.pi * 440 * time), 16000)
    write_pcm(other, .2 * np.sin(2 * np.pi * 880 * time), 16000)
    shutil.copy2(source, duplicate)
    source.with_suffix(".txt").write_text("A clear tone", encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "prepared"
    events = []
    result = prepare_dataset(dict(output_dir=str(output), sources=[str(source), str(duplicate), str(other)],
                    sample_rate=44100, channels=2, clip_seconds=.5, overlap=0, caption="Fallback tone"),
                    lambda **e: events.append(e))
    assert result["complete"] and len(result["sources"]) == 2
    assert result["training_clips"] == result["validation_clips"] == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    splits = {}
    for clip in result["clips"]:
        splits.setdefault(clip["source_id"], set()).add(clip["split"])
        values, rate = read_pcm(output / clip["path"])
        assert rate == 44100 and values.shape == (2, 22050)
    assert all(len(s) == 1 for s in splits.values())
    assert {row["caption"] for row in caption_rows(output)} == {"A clear tone", "Fallback tone"}
    assert events[-1]["event"] == "done"
    with pytest.raises(ValueError, match="existing datasets"):
        prepare_dataset(dict(output_dir=str(output)), lambda **_: None)


def test_interrupted_dataset_is_not_trainable(tmp_path):
    source = tmp_path / "source.wav"
    write_pcm(source, np.ones((2, 22050), np.float32) * .1, 44100)
    output = tmp_path / "partial"
    with pytest.raises(InterruptedError):
        prepare_dataset(dict(output_dir=str(output), sources=[str(source)], sample_rate=44100,
                             channels=2, clip_seconds=.5, overlap=0), lambda **_: None, lambda: True)
    assert not read_json(output / "dataset.json")["complete"]
    with pytest.raises(ValueError, match="complete"):
        training_rows(output)
