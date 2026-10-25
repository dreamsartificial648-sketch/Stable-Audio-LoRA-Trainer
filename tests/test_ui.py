from pathlib import Path
import sys

from PySide6.QtCore import Qt
from sfx_tuner.app import MainWindow
from sfx_tuner.audio import read_json
from test_data import make_dataset


def test_all_pages_open_and_caption_edits_survive_switch(qapp, tmp_path):
    a = make_dataset(tmp_path / "workspace/datasets/a")
    b = make_dataset(tmp_path / "workspace/datasets/b", "Second caption")
    window = MainWindow(tmp_path)
    try:
        window.show()
        for i in range(5):
            window.switch_page(i)
            qapp.processEvents()
            assert window.pages.currentIndex() == i
        window.caption_dataset.setCurrentIndex(window.caption_dataset.findData(str(a)))
        window.captions.item(0, 2).setText("Edited first caption")
        window.caption_dataset.setCurrentIndex(window.caption_dataset.findData(str(b)))
        assert (a / "clip.txt").read_text() == "Edited first caption"
        assert window.captions.item(0, 2).text() == "Second caption"
        window.refresh()
        assert window.captions.item(0, 2).text() == "Second caption"
    finally:
        window.close()
        qapp.processEvents()


def test_ui_prepare_job_populates_caption_and_training_lists(qapp, wait_until, tmp_path):
    import numpy as np
    from sfx_tuner.audio import write_pcm
    source = tmp_path / "source.wav"
    write_pcm(source, np.ones((2, 22050), np.float32) * .1, 44100)
    window = MainWindow(tmp_path)
    try:
        window.add_sources([str(source)])
        window.clip_length.setValue(.5)
        window.shared_caption.setText("A short tone")
        window.prepare()
        wait_until(lambda: not window.runner.active)
        assert window.runner.result["event"] == "done"
        assert window.caption_dataset.count() == window.train_dataset.count() == 1
        assert window.captions.rowCount() == 1
        assert window.captions.item(0, 2).text() == "A short tone"
        assert window.progress.value() == 100
    finally:
        window.close()
        qapp.processEvents()


def test_training_request_saves_edits_before_launch(qapp, tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "workspace/datasets/test", "Old")
    window = MainWindow(tmp_path)
    requests = []
    try:
        window.captions.item(0, 2).setText("A newly captioned click")
        window.runtime.setText(sys.executable)
        monkeypatch.setattr(window.runner, "start", lambda python, request: requests.append(request))
        window.train()
        assert (dataset / "clip.txt").read_text() == "A newly captioned click"
        assert requests[0]["operation"] == "train"
        assert requests[0]["dataset_dir"] == str(dataset)
        assert requests[0]["rank"] == 8
        settings = read_json(tmp_path / "local-settings.json")
        assert "token" not in str(settings).lower()
    finally:
        window.close()
        qapp.processEvents()


def test_bad_dataset_does_not_prevent_closing(qapp, tmp_path):
    from sfx_tuner.audio import atomic_json
    atomic_json(tmp_path / "workspace/datasets/broken/dataset.json", dict(complete=True, clips=[dict(path="missing.wav")]))
    window = MainWindow(tmp_path)
    assert window.caption_folder is None
    assert "missing" in window.caption_summary.text()
    assert window.close()
    qapp.processEvents()
