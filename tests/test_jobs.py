import json
from pathlib import Path
import sys

from sfx_tuner.audio import read_json
from sfx_tuner.jobs import JobRunner


def test_split_event_stream_and_success_record(qapp, wait_until, tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text("import sys, time\nprint('loading')\nsys.stdout.write('SFX_EV'); sys.stdout.flush(); time.sleep(.05)\nprint('ENT:{\"event\":\"done\",\"message\":\"ok\"}')\n", encoding="utf-8")
    runner = JobRunner(tmp_path)
    runner.job_dir = tmp_path / "job"
    runner.job_dir.mkdir()
    events, logs, results = [], [], []
    runner.event.connect(events.append)
    runner.log.connect(logs.append)
    runner.finished.connect(lambda okay, message: results.append((okay, message)))
    runner._launch(sys.executable, [str(script)], "train")
    wait_until(lambda: not runner.active)
    assert results[0][0] is True
    assert events == [dict(event="done", message="ok")]
    assert "loading" in logs
    assert read_json(runner.job_dir / "result.json")["success"]


def test_zero_exit_without_done_event_is_failure(qapp, wait_until, tmp_path):
    runner = JobRunner(tmp_path)
    runner.job_dir = tmp_path / "job"
    runner.job_dir.mkdir()
    results = []
    runner.finished.connect(lambda okay, message: results.append(okay))
    runner._launch(sys.executable, ["-c", "print('no completion event')"], "generate")
    wait_until(lambda: not runner.active)
    assert results == [False]


def test_stop_unpauses_worker_and_preserves_stopped_status(qapp, wait_until, tmp_path):
    runner = JobRunner(tmp_path)
    runner.job_dir = tmp_path / "job"
    runner.job_dir.mkdir()
    script = tmp_path / "cooperative.py"
    script.write_text('''from pathlib import Path
import sys, time
root=Path(sys.argv[1])
print('SFX_EVENT:{"event":"log","message":"ready"}', flush=True)
while not (root/'stop.flag').exists():
    time.sleep(.01)
print('SFX_EVENT:{"event":"stopped","message":"saved"}', flush=True)
''', encoding="utf-8")
    events, results = [], []
    runner.event.connect(events.append)
    runner.finished.connect(lambda okay, message: results.append((okay, message)))
    runner._launch(sys.executable, [str(script), str(runner.job_dir)], "train")
    wait_until(lambda: bool(events))
    runner.toggle_pause()
    assert (runner.job_dir / "pause.flag").exists()
    runner.stop()
    assert not (runner.job_dir / "pause.flag").exists()
    assert (runner.job_dir / "stop.flag").exists()
    wait_until(lambda: not runner.active)
    result = read_json(runner.job_dir / "result.json")
    assert result["stopped"] and not result["success"] and results[0][0] is False


def test_real_dataset_worker_through_qprocess(qapp, wait_until, tmp_path):
    import numpy as np
    from sfx_tuner.audio import write_pcm
    source = tmp_path / "input.wav"
    write_pcm(source, np.ones((2, 22050), np.float32) * .1, 44100)
    runner = JobRunner(tmp_path)
    results = []
    runner.finished.connect(lambda okay, message: results.append((okay, message)))
    runner.start(sys.executable, dict(operation="prepare", output_dir=str(tmp_path / "prepared"),
                 sources=[str(source)], sample_rate=44100, channels=2, clip_seconds=.5, overlap=0, caption="A tone"))
    wait_until(lambda: not runner.active)
    assert results[0][0] is True
    assert read_json(tmp_path / "prepared/dataset.json")["complete"]
