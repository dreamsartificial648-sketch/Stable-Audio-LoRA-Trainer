"""One managed child process, with streamed logs and cooperative controls."""
from datetime import datetime, timezone
import codecs
import json
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from sfx_tuner.audio import atomic_json
from sfx_tuner.paths import launch_environment, default_root
from sfx_tuner.protocol import PREFIX


class JobRunner(QObject):
    log = Signal(str)
    event = Signal(dict)
    finished = Signal(bool, str)
    active_changed = Signal(bool)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.active = False
        self.paused = False
        self.stopping = False
        self.job_dir = None
        self.operation = ""
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.result = None
        self.error_message = ""
        self.hf_home = ""
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.force_stop)

    def start(self, python, request):
        if self.active:
            raise RuntimeError("Wait for the current job to finish.")
        self.job_dir = self.root / "workspace/jobs" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
        self.job_dir.mkdir(parents=True)
        request = {**request, "stop_file": str(self.job_dir / "stop.flag"), "pause_file": str(self.job_dir / "pause.flag")}
        atomic_json(self.job_dir / "request.json", request)
        self._launch(python, [str(Path(__file__).with_name("worker.py")), "--request", str(self.job_dir / "request.json")], request["operation"])

    def install(self):
        self.job_dir = self.root / "workspace/jobs" / ("setup_" + uuid4().hex[:8])
        self.job_dir.mkdir(parents=True)
        from sfx_tuner.paths import base_python
        self._launch(base_python(), [str(default_root() / "scripts/install_runtime.py")], "install")

    def _launch(self, python, arguments, operation):
        if self.active:
            raise RuntimeError("A job is already running.")
        if not Path(python).is_file():
            raise ValueError("Python executable was not found. Open Setup to install or select a runtime.")
        self.operation = operation
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.error_message = ""
        self.result = None
        self.stopping = self.paused = False
        self.active = True
        env = QProcessEnvironment()
        environment = launch_environment()
        if self.hf_home:
            environment["HF_HOME"] = self.hf_home
        for key, value in environment.items():
            env.insert(key, value)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(self.root))
        self.process.setProgram(str(python))
        self.process.setArguments(arguments)
        self.active_changed.emit(True)
        self.log.emit(f"Starting {operation}. Logs: {self.job_dir}")
        self.process.start()

    def _read(self):
        self.buffer += self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()
        for line in lines:
            self._line(line.rstrip("\r"))

    def _line(self, line):
        with (self.job_dir / "output.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if line.startswith(PREFIX):
            try:
                payload = json.loads(line[len(PREFIX):])
            except ValueError:
                self.log.emit(line)
                return
            kind = payload.get("event")
            if kind in ("done", "stopped"):
                self.result = payload
            elif kind == "error":
                self.error_message = payload.get("message", "Worker failed.")
            if kind in ("log", "error", "stopped"):
                self.log.emit(payload.get("message", ""))
            self.event.emit(payload)
        elif line.strip():
            self.log.emit(line)

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self.error_message = self.process.errorString()
            self._finished(-1, QProcess.CrashExit)

    def _finished(self, code, status):
        if not self.active:
            return
        self._read()
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""
        self.kill_timer.stop()
        was_stopped = self.stopping or (self.result or {}).get("event") == "stopped"
        okay = code == 0 and not self.error_message and (self.operation == "install" or self.result is not None)
        if self.error_message:
            message = self.error_message
        elif was_stopped:
            message = "Stopped. Any saved checkpoints and completed WAVs remain in the workspace."
        elif okay:
            message = "Runtime installed. Run Check runtime next." if self.operation == "install" else "Completed. Results saved to your workspace."
        else:
            message = f"Worker exited with code {code}. Open the job log for details."
        atomic_json(self.job_dir / "result.json", dict(success=okay and not was_stopped, stopped=was_stopped, message=message,
                    operation=self.operation, finished_at=datetime.now(timezone.utc).isoformat(), result=self.result))
        self.active = self.paused = False
        self.active_changed.emit(False)
        self.finished.emit(okay and not was_stopped, message)

    def toggle_pause(self):
        if not self.active or self.stopping or self.operation in ("doctor", "install"):
            return
        self.paused = not self.paused
        flag = self.job_dir / "pause.flag"
        if self.paused:
            flag.touch()
        else:
            flag.unlink(missing_ok=True)
        self.log.emit("Pause requested; takes effect at the next batch or sampling boundary." if self.paused else "Resumed.")

    def stop(self):
        if not self.active:
            return
        if self.operation == "install":
            self.force_stop()
            return
        self.stopping = True
        self.paused = False
        (self.job_dir / "pause.flag").unlink(missing_ok=True)
        (self.job_dir / "stop.flag").touch()
        self.log.emit("Stop requested. Waiting for the current operation to reach a safe boundary.")
        # The user can choose Force stop if a driver call or download is stalled.

    def force_stop(self):
        if not self.active:
            return
        self.stopping = True
        self.error_message = "Forced stop. The latest unfinished checkpoint or clip may be unavailable."
        # Kill descendants too (pip/git/ffmpeg), using process APIs without a shell.
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/PID", str(self.process.processId()), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.process.kill()
