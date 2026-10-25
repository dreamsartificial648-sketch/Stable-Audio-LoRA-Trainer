"""Standalone Small-SFX desktop workspace."""
import argparse
import math
from pathlib import Path
import shutil
import sys

from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)

from sfx_tuner import __version__
from sfx_tuner.audio import MEDIA_EXTENSIONS, atomic_json, new_folder, read_json
from sfx_tuner.contracts import caption_rows, checkpoint_info, discover_models, save_captions
from sfx_tuner.jobs import JobRunner
from sfx_tuner.paths import default_root, runtime_python
from sfx_tuner.requests import validate
from sfx_tuner.widgets import STYLE, AudioPlayer, Plot, SourceList, button, card, label

MODEL_URL = "https://huggingface.co/stabilityai/stable-audio-3-small-sfx"


def spin(low, high, value, decimals=None):
    result = QSpinBox() if decimals is None else QDoubleSpinBox()
    if decimals is not None:
        result.setDecimals(decimals)
    result.setRange(low, high)
    result.setValue(value)
    result.setMinimumWidth(120)
    result.setMaximumWidth(240)
    return result


class MainWindow(QMainWindow):
    def __init__(self, root=None, runtime=None, hf_home=None):
        super().__init__()
        self.root = Path(root or default_root()).resolve()
        self.settings_path = self.root / "local-settings.json"
        self.settings = read_json(self.settings_path)
        self.datasets = []
        self.adapters = []
        self.caption_folder = None
        self.caption_dirty = False
        self.players = []
        self.start_buttons = []
        self.close_pending = False
        self.runner = JobRunner(self.root, self)
        self.setWindowTitle("Stable Audio Fine-Tuner")
        self.resize(1310, 900)
        self.setMinimumSize(1020, 720)
        shell = QWidget()
        shell.setObjectName("Shell")
        self.setCentralWidget(shell)
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(225)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(18, 30, 18, 22)
        brand = label("SFX / STUDIO")
        brand.setObjectName("Brand")
        nav.addWidget(brand)
        nav.addWidget(label("STABLE AUDIO FINE-TUNER", True))
        nav.addSpacing(36)
        self.navigation = QButtonGroup(self)
        self.navigation.setExclusive(True)
        self.pages = QStackedWidget()
        for index, title in enumerate(("01   Setup", "02   Dataset", "03   Train", "04   Generate", "05   Library")):
            item = QPushButton(title)
            item.setProperty("nav", True)
            item.setCheckable(True)
            self.navigation.addButton(item, index)
            nav.addWidget(item)
        self.navigation.idClicked.connect(self.switch_page)
        nav.addStretch()
        f, box = card("Small-SFX", "Text to sound.\nYour sounds, your adapter.")
        box.addWidget(label("LoRA fine-tuning\n44.1 kHz stereo", True))
        nav.addWidget(f)
        nav.addSpacing(14)
        nav.addWidget(button("Open workspace", lambda: self.open_path(self.root / "workspace")))
        nav.addWidget(label(f"v{__version__}  ·  Derived from ADAM", True))
        outer.addWidget(sidebar)
        body = QVBoxLayout()
        body.setContentsMargins(30, 24, 30, 20)
        body.setSpacing(14)
        outer.addLayout(body, 1)
        top = QHBoxLayout()
        eyebrow = label("LOCAL AUDIO WORKSPACE")
        eyebrow.setObjectName("Eyebrow")
        eyebrow.setWordWrap(False)
        top.addWidget(eyebrow)
        top.addStretch()
        self.badge = label("Ready")
        self.badge.setObjectName("Badge")
        top.addWidget(self.badge)
        body.addLayout(top)
        body.addWidget(self.pages, 1)
        self._setup_page(runtime, hf_home)
        self._dataset_page()
        self._train_page()
        self._generate_page()
        self._library_page()
        self._footer(body)
        self.runner.log.connect(self.append_log)
        self.runner.event.connect(self._event)
        self.runner.active_changed.connect(self._active)
        self.runner.finished.connect(self._finished)
        self.refresh()
        self.switch_page(0)
        self.status.setText("Start in Setup to check your training environment.")

    def _page(self, title, subtitle):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 8, 8, 14)
        box.setSpacing(18)
        heading = label(title)
        heading.setObjectName("PageTitle")
        box.addWidget(heading)
        box.addWidget(label(subtitle, True))
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        return box

    def switch_page(self, index):
        # Pause playback when changing workspace pages.
        for player in self.players:
            player.player.pause()
        self.pages.setCurrentIndex(index)
        self.navigation.button(index).setChecked(True)

    def _action(self, text, callback, primary=True):
        item = button(text, lambda: self.guard(callback), primary)
        self.start_buttons.append(item)
        return item

    def _player(self):
        result = AudioPlayer()
        self.players.append(result)
        return result

    def _setup_page(self, runtime, hf_home):
        page = self._page("Make room for sound.", "A dedicated desktop for fine-tuning Stable Audio 3 Small-SFX.")
        f, box = card("01 / Set up your environment", "Install the desktop once, then install the separate training runtime. Model weights download on first use.")
        row = QHBoxLayout()
        self.install_button = self._action("Install training runtime", self.install_runtime)
        row.addWidget(self.install_button)
        row.addWidget(self._action("Check runtime", self.check_runtime, False))
        row.addStretch()
        box.addLayout(row)
        form = QFormLayout()
        self.runtime = QLineEdit(str(runtime or self.settings.get("runtime_python") or runtime_python(default_root())))
        self.runtime.setPlaceholderText("Select the training environment's Python executable")
        browse = QHBoxLayout()
        browse.addWidget(self.runtime, 1)
        browse.addWidget(button("Browse…", self.browse_runtime))
        form.addRow("Runtime Python", browse)
        self.hf_home = QLineEdit(str(hf_home or self.settings.get("hf_home") or ""))
        self.hf_home.setPlaceholderText("Optional: existing Hugging Face cache; leave blank for the default")
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.hf_home, 1)
        cache_row.addWidget(button("Browse…", self.browse_cache))
        form.addRow("Model cache", cache_row)
        box.addLayout(form)
        box.addWidget(label("Windows + NVIDIA CUDA · Python 3.10–3.12 · Training requires bfloat16 support.\nGPU memory usage depends on clip length and batch size; a minimum VRAM capacity has not been established for this app.", True))
        page.addWidget(f)
        f, box = card("02 / Connect model access", "Accept the Small-SFX terms on Hugging Face, then authenticate locally. A saved login is reused; tokens are never stored in app settings.")
        row = QHBoxLayout()
        row.addWidget(button("Open model access page", lambda: QDesktopServices.openUrl(QUrl(MODEL_URL))))
        row.addWidget(button("Show login instructions", lambda: self.notify("Hugging Face login", "After installing the runtime, run Login to Hugging Face.bat in this app folder.\n\nFor a custom runtime, run its hf executable with: auth login\n\nRestart Check runtime afterward. A login check does not guarantee access to gated weights.")))
        row.addStretch()
        box.addLayout(row)
        page.addWidget(f)
        f, box = card("Environment report")
        self.report = label("No runtime check yet.\n\nYou can prepare and caption datasets before installing the GPU runtime.", True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.report)
        page.addWidget(f)
        page.addStretch()

    def _dataset_page(self):
        page = self._page("Build your sound library.", "Import recordings, prepare clips, and describe what is audible in each one.")
        f, box = card("Source recordings", "Drop audio or video files here. Originals stay untouched. A matching .txt file supplies the source caption when available.")
        row = QHBoxLayout()
        row.addWidget(button("Add files…", self.add_files))
        row.addWidget(button("Add folder…", self.add_folder))
        row.addWidget(button("Remove selected", self.remove_sources))
        row.addStretch()
        box.addLayout(row)
        self.sources = SourceList()
        self.sources.dropped.connect(self.add_sources)
        box.addWidget(self.sources)
        form = QFormLayout()
        self.dataset_name = QLineEdit("My sound effects")
        self.clip_length = spin(.5, 120, self.settings.get("clip_seconds", 5), 1)
        self.clip_length.setSuffix(" s")
        self.overlap = spin(0, 75, 0)
        self.overlap.setSuffix(" %")
        self.shared_caption = QLineEdit()
        self.shared_caption.setPlaceholderText("Optional fallback description for recordings without a .txt caption")
        self.skip_silence = QCheckBox("Skip silent clips")
        self.skip_silence.setChecked(True)
        form.addRow("Dataset name", self.dataset_name)
        form.addRow("Clip length", self.clip_length)
        form.addRow("Overlap", self.overlap)
        form.addRow("Shared description", self.shared_caption)
        form.addRow("", self.skip_silence)
        box.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(self._action("Prepare dataset", self.prepare))
        row.addWidget(label("Prepared as 44.1 kHz stereo WAV", True))
        row.addStretch()
        box.addLayout(row)
        page.addWidget(f)
        f, box = card("Captions", "Specific descriptions help the adapter learn. Review inherited captions after splitting a longer recording.")
        row = QHBoxLayout()
        self.caption_dataset = QComboBox()
        self.caption_dataset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.caption_dataset, 1)
        row.addWidget(button("Open prepared dataset…", lambda: self.guard(self.import_dataset)))
        row.addWidget(button("Refresh", lambda: self.guard(self.refresh)))
        box.addLayout(row)
        self.caption_summary = label("Prepare a dataset to begin.", True)
        box.addWidget(self.caption_summary)
        self.captions = QTableWidget(0, 3)
        self.captions.setHorizontalHeaderLabels(("Clip", "Split", "Sound description"))
        self.captions.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.captions.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.captions.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.captions.setMinimumHeight(225)
        self.captions.itemSelectionChanged.connect(self._caption_selected)
        self.captions.itemChanged.connect(self._caption_edited)
        box.addWidget(self.captions)
        row = QHBoxLayout()
        self.save_captions_button = button("Save captions", lambda: self.guard(self.save_caption_table), True)
        row.addWidget(self.save_captions_button)
        row.addWidget(button("Fill blank descriptions", lambda: self.guard(self.fill_captions)))
        row.addStretch()
        box.addLayout(row)
        self.dataset_player = self._player()
        box.addWidget(self.dataset_player)
        page.addWidget(f)
        self.caption_dataset.currentIndexChanged.connect(self.load_caption_table)

    def _train_page(self):
        page = self._page("Teach it your sound.", "Train a compact LoRA adapter while the pretrained generator stays frozen.")
        f, box = card("Training session")
        form = QFormLayout()
        form.setVerticalSpacing(12)
        self.train_dataset = QComboBox()
        self.train_name = QLineEdit("My Small-SFX adapter")
        self.resume = QComboBox()
        self.resume.addItem("Start a new adapter", "")
        self.epochs = spin(1, 100000, self.settings.get("epochs", 20))
        self.batch = spin(1, 256, self.settings.get("batch_size", 1))
        self.lr = spin(.0000001, .01, self.settings.get("learning_rate", .0001), 7)
        self.lr.setSingleStep(.00001)
        self.rank = spin(1, 64, self.settings.get("rank", 8))
        self.save_every = spin(1, 1000, self.settings.get("save_every", 5))
        self.train_seed = spin(0, 2147483647, self.settings.get("train_seed", 12345))
        for text, widget in (("Dataset", self.train_dataset), ("Adapter name", self.train_name), ("Continue from", self.resume)):
            form.addRow(text, widget)
        box.addLayout(form)
        options = QHBoxLayout()
        left, right = QFormLayout(), QFormLayout()
        left.setVerticalSpacing(10)
        right.setVerticalSpacing(10)
        for text, widget in (("Epochs", self.epochs), ("Batch size", self.batch), ("Learning rate", self.lr)):
            left.addRow(text, widget)
        for text, widget in (("LoRA rank", self.rank), ("Save every N epochs", self.save_every), ("Seed", self.train_seed)):
            right.addRow(text, widget)
        options.addLayout(left, 1)
        options.addSpacing(24)
        options.addLayout(right, 1)
        box.addLayout(options)
        self.train_summary = label("Select a prepared and captioned dataset.", True)
        box.addWidget(self.train_summary)
        box.addWidget(label("Continue loads adapter weights with a fresh optimizer and saves to a new folder. Validation clips are held out; this version reports training loss only.", True))
        row = QHBoxLayout()
        row.addWidget(self._action("Start LoRA fine-tuning", self.train))
        row.addWidget(button("Open adapter folder…", lambda: self.guard(self.import_adapter)))
        row.addStretch()
        box.addLayout(row)
        page.addWidget(f)
        f, box = card("Training monitor")
        self.train_metrics = label("No training session yet.", True)
        box.addWidget(self.train_metrics)
        self.loss_chart = Plot(loss=True)
        self.loss_chart.setMinimumHeight(150)
        box.addWidget(self.loss_chart)
        box.addWidget(label("Loss tracks optimization, not perceived sound quality. Use Generate to listen to the saved adapter.", True))
        page.addWidget(f)
        self.resume.currentIndexChanged.connect(self.resume_changed)
        self.train_dataset.currentIndexChanged.connect(self.update_train_summary)
        self.batch.valueChanged.connect(self.update_train_summary)
        self.epochs.valueChanged.connect(self.update_train_summary)

    def _generate_page(self):
        page = self._page("Hear what you made.", "Generate sound effects with the base model or audition a trained adapter.")
        f, box = card("Sound direction")
        form = QFormLayout()
        self.gen_model = QComboBox()
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("A heavy wooden door creaks open, then closes with a deep thud in a quiet room.")
        self.prompt.setFixedHeight(85)
        self.negative = QLineEdit()
        self.negative.setPlaceholderText("Optional sounds to avoid")
        self.duration = spin(.5, 120, self.settings.get("duration", 3), 1)
        self.duration.setSuffix(" s")
        self.steps = spin(5, 250, self.settings.get("steps", 50))
        self.guidance = spin(1, 20, self.settings.get("guidance", 7), 1)
        self.strength = spin(0, 2, self.settings.get("lora_strength", 1), 2)
        self.strength.setSingleStep(.1)
        self.gen_count = spin(1, 16, 1)
        self.gen_seed = spin(0, 2147483647, self.settings.get("gen_seed", 42))
        for text, widget in (("Model", self.gen_model), ("Prompt", self.prompt), ("Negative prompt", self.negative)):
            form.addRow(text, widget)
        box.addLayout(form)
        options = QHBoxLayout()
        left, right = QFormLayout(), QFormLayout()
        left.setVerticalSpacing(10)
        right.setVerticalSpacing(10)
        for text, widget in (("Duration", self.duration), ("Sampling steps", self.steps), ("Guidance", self.guidance)):
            left.addRow(text, widget)
        for text, widget in (("LoRA strength", self.strength), ("Clips", self.gen_count), ("Seed", self.gen_seed)):
            right.addRow(text, widget)
        options.addLayout(left, 1)
        options.addSpacing(24)
        options.addLayout(right, 1)
        box.addLayout(options)
        box.addWidget(label("Uses small-sfx-base for both training and generation. Each extra clip uses seed + 1. Start with short clips and batch size 1 when testing your GPU.", True))
        row = QHBoxLayout()
        row.addWidget(self._action("Generate sound", self.generate))
        row.addStretch()
        box.addLayout(row)
        page.addWidget(f)
        f, box = card("Listen & export")
        self.gen_results = QComboBox()
        self.gen_results.currentIndexChanged.connect(self._gen_selected)
        box.addWidget(self.gen_results)
        self.gen_player = self._player()
        box.addWidget(self.gen_player)
        row = QHBoxLayout()
        row.addWidget(button("Save WAV…", lambda: self.guard(lambda: self.export_audio(self.gen_player))))
        row.addWidget(button("Show in folder", lambda: self.open_audio_folder(self.gen_player)))
        row.addStretch()
        box.addLayout(row)
        page.addWidget(f)
        self.gen_model.currentIndexChanged.connect(lambda: self.strength.setEnabled(self.gen_model.currentData() != "stable-audio-3:base"))

    def _library_page(self):
        page = self._page("Your sounds, collected.", "Browse saved adapters and generated WAVs. Every run keeps its settings and logs.")
        f, box = card("Adapters & generations")
        row = QHBoxLayout()
        row.addWidget(button("Refresh library", lambda: self.guard(self.refresh)))
        row.addWidget(button("Import adapter folder…", lambda: self.guard(self.import_adapter)))
        row.addStretch()
        box.addLayout(row)
        self.library = QListWidget()
        self.library.setMinimumHeight(220)
        self.library.currentItemChanged.connect(self.library_selected)
        box.addWidget(self.library)
        self.library_details = label("Select an adapter or a generated clip.", True)
        self.library_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self.library_details)
        self.library_player = self._player()
        box.addWidget(self.library_player)
        row = QHBoxLayout()
        self.use_adapter_button = button("Use in Generate", self.use_adapter)
        self.use_adapter_button.setEnabled(False)
        row.addWidget(self.use_adapter_button)
        row.addWidget(button("Save WAV…", lambda: self.guard(lambda: self.export_audio(self.library_player))))
        row.addWidget(button("Open selected folder", self.open_library_folder))
        row.addStretch()
        box.addLayout(row)
        page.addWidget(f)
        page.addStretch()

    def _footer(self, body):
        f, box = card("Session")
        box.setContentsMargins(16, 12, 16, 12)
        self.status = label("Ready.", True)
        box.addWidget(self.status)
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        row.addWidget(self.progress, 1)
        self.pause_button = button("Pause", self.pause)
        self.stop_button = button("Stop && save", self.runner.stop)
        self.force_button = button("Force stop", self.force_stop)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.force_button.setEnabled(False)
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.force_button)
        self.logs_button = button("Show logs", self.toggle_logs)
        row.addWidget(self.logs_button)
        box.addLayout(row)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(2500)
        self.logs.setFixedHeight(130)
        self.logs.setVisible(False)
        box.addWidget(self.logs)
        body.addWidget(f)

    def guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            if not self.runner.active:
                self.progress.setRange(0, 100)
                self.progress.setValue(0)
            self.status.setText(str(exc))
            self.append_log(str(exc))
            QMessageBox.warning(self, "Unable to continue", str(exc))

    def notify(self, title, text):
        QMessageBox.information(self, title, text)

    def append_log(self, message):
        self.logs.appendPlainText(message)

    def toggle_logs(self):
        self.logs.setVisible(not self.logs.isVisible())
        self.logs_button.setText("Hide logs" if self.logs.isVisible() else "Show logs")

    def open_path(self, path):
        path = Path(path)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            self.status.setText("This folder will be created when you run your first job.")

    def browse_runtime(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose runtime Python", str(self.root), "Python executable (python.exe python python3);;All files (*)")
        if path:
            self.runtime.setText(path)
            self.save_settings()

    def browse_cache(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Hugging Face cache root (folder containing hub)")
        if folder:
            self.hf_home.setText(folder)
            self.save_settings()

    def install_runtime(self):
        self.save_settings()
        self.runtime.setText(str(runtime_python(default_root())))
        self.save_settings()
        self.progress.setRange(0, 0)
        self.status.setText("Installing training runtime. This downloads PyTorch and other dependencies; see logs for progress.")
        self.runner.install()

    def start(self, request):
        if self.runner.active:
            raise ValueError("Wait for the active job to finish.")
        request = validate(request)
        self.save_settings()
        python = sys.executable if request["operation"] == "prepare" else self.runtime.text().strip()
        self.runner.hf_home = self.hf_home.text().strip()
        self.runner.start(python, request)
        self.progress.setRange(0, 0)
        self.status.setText(f"Starting {request['operation']}… Model loading and downloads can take several minutes.")

    def check_runtime(self):
        self.start(dict(operation="doctor"))

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Import recordings", "", "Audio / video (*.wav *.mp3 *.flac *.ogg *.opus *.m4a *.aac *.aif *.aiff *.wma *.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)")
        self.add_sources(files)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Import recordings folder")
        if folder:
            self.add_sources([str(path) for path in Path(folder).rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS])

    def add_sources(self, files):
        known = {self.sources.item(i).data(Qt.UserRole) for i in range(self.sources.count())}
        for filename in files:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and str(path) not in known:
                item = QListWidgetItem(path.name)
                item.setData(Qt.UserRole, str(path))
                item.setToolTip(str(path))
                self.sources.addItem(item)
                known.add(str(path))

    def remove_sources(self):
        for item in self.sources.selectedItems():
            self.sources.takeItem(self.sources.row(item))

    def prepare(self):
        files = [self.sources.item(i).data(Qt.UserRole) for i in range(self.sources.count())]
        if not files:
            raise ValueError("Add recordings before preparing a dataset.")
        if not shutil.which("ffmpeg"):
            raise ValueError("Dataset preparation needs ffmpeg on PATH. See README.md, then restart the app.")
        output = new_folder(self.root, "datasets", self.dataset_name.text())
        self.start(dict(operation="prepare", name=self.dataset_name.text().strip() or "Sound effects", sources=files,
                        output_dir=str(output), sample_rate=44100, channels=2, clip_seconds=self.clip_length.value(),
                        overlap=self.overlap.value() / 100., skip_silence=self.skip_silence.isChecked(), caption=self.shared_caption.text()))

    def import_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder containing dataset.json")
        if folder:
            info = read_json(Path(folder) / "dataset.json")
            if not info.get("complete"):
                raise ValueError("Select a completed prepared dataset containing dataset.json.")
            caption_rows(Path(folder))
            paths = self.settings.setdefault("external_datasets", [])
            if folder not in paths:
                paths.append(folder)
            self.refresh(dataset=folder)
            self.save_settings()

    def import_adapter(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder containing adapter.safetensors and audio_model.json")
        if folder:
            checkpoint_info(Path(folder))
            paths = self.settings.setdefault("external_adapters", [])
            if folder not in paths:
                paths.append(folder)
            self.refresh(adapter=folder)
            self.save_settings()

    def refresh(self, dataset=None, adapter=None):
        if self.caption_folder:
            self.save_caption_table(quiet=True)
        self.datasets = []
        folders = list((self.root / "workspace/datasets").glob("*/dataset.json"))
        folders += [Path(folder) / "dataset.json" for folder in self.settings.get("external_datasets", [])]
        seen = set()
        for path in sorted(folders):
            folder = str(path.parent.resolve())
            info = read_json(path)
            if info.get("complete") and folder not in seen:
                self.datasets.append({**info, "path": folder})
                seen.add(folder)
        self.adapters = discover_models(self.root)
        for folder in self.settings.get("external_adapters", []):
            try:
                info = checkpoint_info(Path(folder))
                if not any(m["path"] == folder for m in self.adapters):
                    self.adapters.append({**info, "path": folder})
            except ValueError:
                pass
        for combo in (self.caption_dataset, self.train_dataset):
            wanted = dataset or combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for item in self.datasets:
                combo.addItem(item.get("name", Path(item["path"]).name), item["path"])
            index = combo.findData(wanted)
            combo.setCurrentIndex(max(0, index) if combo.count() else -1)
            combo.blockSignals(False)
        for combo, first, first_data in ((self.resume, "Start a new adapter", ""), (self.gen_model, "Original Small-SFX base model", "stable-audio-3:base")):
            wanted = adapter or combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first, first_data)
            for item in self.adapters:
                combo.addItem(item.get("name", Path(item["path"]).name), item["path"])
            index = combo.findData(wanted)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
        self.load_caption_table()
        self.update_train_summary()
        self.resume_changed()
        self.strength.setEnabled(self.gen_model.currentData() != "stable-audio-3:base")
        self.library.clear()
        for item in self.adapters:
            entry = QListWidgetItem("LoRA  /  " + item.get("name", Path(item["path"]).name))
            entry.setData(Qt.UserRole, dict(kind="adapter", **item))
            self.library.addItem(entry)
        for info_path in sorted((self.root / "workspace/generations").glob("*/generation.json"), reverse=True):
            record = read_json(info_path)
            for filename in record.get("files", []):
                path = (info_path.parent / filename).resolve()
                if not path.is_relative_to(info_path.parent.resolve()) or not path.is_file():
                    continue
                item = QListWidgetItem("WAV  /  " + record.get("prompt", "Sound")[:70] + "  ·  " + path.name)
                item.setData(Qt.UserRole, dict(kind="audio", path=str(path), record=record))
                self.library.addItem(item)

    def load_caption_table(self):
        folder = self.caption_dataset.currentData()
        # Save the previous table before switching to another dataset.
        if self.caption_folder and str(self.caption_folder) != folder:
            try:
                self.save_caption_table(quiet=True)
            except Exception as exc:
                self.status.setText(f"Could not save previous captions: {exc}")
                self.caption_dataset.blockSignals(True)
                self.caption_dataset.setCurrentIndex(self.caption_dataset.findData(str(self.caption_folder)))
                self.caption_dataset.blockSignals(False)
                return
        self.caption_folder = Path(folder) if folder else None
        self.captions.setRowCount(0)
        if not folder:
            self.caption_summary.setText("Prepare a dataset to begin.")
            return
        try:
            rows = caption_rows(Path(folder))
            self.captions.setRowCount(len(rows))
            for i, row in enumerate(rows):
                name = QTableWidgetItem(Path(row["path"]).name)
                name.setData(Qt.UserRole, row["path"])
                name.setFlags(name.flags() & ~Qt.ItemIsEditable)
                split = QTableWidgetItem(row["split"])
                split.setFlags(split.flags() & ~Qt.ItemIsEditable)
                self.captions.setItem(i, 0, name)
                self.captions.setItem(i, 1, split)
                self.captions.setItem(i, 2, QTableWidgetItem(row["caption"]))
            missing = sum(not r["caption"] and r["split"] == "train" for r in rows)
            self.caption_summary.setText(f"{len(rows)} clips  ·  {missing} training captions missing  ·  Captions save on dataset switch, refresh, and training start.")
            self.caption_dirty = False
        except Exception as exc:
            self.caption_summary.setText(str(exc))
            self.caption_folder = None
            self.caption_dirty = False
            self.captions.setRowCount(0)

    def _caption_edited(self, item):
        if item.column() == 2:
            self.caption_dirty = True

    def save_caption_table(self, quiet=False):
        if not self.caption_folder or not self.caption_dirty:
            return
        rows = [(self.captions.item(i, 0).data(Qt.UserRole), self.captions.item(i, 2).text()) for i in range(self.captions.rowCount())]
        save_captions(self.caption_folder, rows)
        self.caption_dirty = False
        missing = sum(not self.captions.item(i, 2).text().strip() and self.captions.item(i, 1).text() == "train" for i in range(self.captions.rowCount()))
        self.caption_summary.setText(f"{self.captions.rowCount()} clips  ·  {missing} training captions missing  ·  Captions saved.")
        if not quiet:
            self.status.setText("Captions saved.")
            self.update_train_summary()

    def fill_captions(self):
        text = self.shared_caption.text().strip()
        if not text:
            raise ValueError("Enter a shared description in Source recordings first.")
        for i in range(self.captions.rowCount()):
            item = self.captions.item(i, 2)
            if not item.text().strip():
                item.setText(text)

    def _caption_selected(self):
        row = self.captions.currentRow()
        if row >= 0 and self.captions.item(row, 0):
            self.dataset_player.load(self.captions.item(row, 0).data(Qt.UserRole))

    def update_train_summary(self):
        folder = self.train_dataset.currentData()
        info = next((d for d in self.datasets if d["path"] == folder), None)
        if info:
            count = info.get("training_clips", 0)
            updates = math.ceil(count / self.batch.value()) * self.epochs.value()
            self.train_summary.setText(f"{count} training clips  ·  {info.get('validation_clips', 0)} held out  ·  {info.get('clip_seconds', '?')} s clips\n{updates:,} optimizer updates across {self.epochs.value():,} epochs")
        else:
            self.train_summary.setText("Prepare or open a dataset in Dataset first.")

    def resume_changed(self):
        path = self.resume.currentData()
        if path:
            info = checkpoint_info(Path(path))
            self.rank.setValue(info["rank"])
        self.rank.setEnabled(not bool(path))

    def train(self):
        self.save_caption_table(quiet=True)
        dataset = self.train_dataset.currentData()
        if not dataset:
            raise ValueError("Select a prepared and captioned dataset first.")
        name = self.train_name.text().strip() or "Small-SFX adapter"
        request = dict(operation="train", dataset_dir=dataset, model_name=name,
                       output_dir=str(new_folder(self.root, "models", name)), epochs=self.epochs.value(),
                       batch_size=self.batch.value(), learning_rate=self.lr.value(), rank=self.rank.value(),
                       save_every=self.save_every.value(), seed=self.train_seed.value(), resume_from=self.resume.currentData() or "")
        self.start(request)
        self.loss_chart.points = []
        self.loss_chart.update()
        self.train_metrics.setText("Loading model…")
        if self.pages.currentIndex() == 2:
            self.pages.currentWidget().ensureWidgetVisible(self.loss_chart)

    def generate(self):
        model_path = self.gen_model.currentData()
        self.start(dict(operation="generate", model_path=model_path, model_name=self.gen_model.currentText(),
                        output_dir=str(new_folder(self.root, "generations", "Sound")), duration=self.duration.value(),
                        clip_count=self.gen_count.value(), steps=self.steps.value(), seed=self.gen_seed.value(),
                        prompt=self.prompt.toPlainText().strip(), negative_prompt=self.negative.text().strip(),
                        guidance=self.guidance.value(), lora_strength=self.strength.value()))
        self.gen_results.clear()

    def _active(self, active):
        for item in self.start_buttons:
            item.setEnabled(not active)
        controls = active and self.runner.operation not in ("doctor", "install")
        self.pause_button.setEnabled(controls)
        self.stop_button.setEnabled(controls)
        self.force_button.setEnabled(active)
        self.badge.setText("Working" if active else "Ready")
        self.pause_button.setText("Pause")

    def pause(self):
        self.runner.toggle_pause()
        self.pause_button.setText("Resume" if self.runner.paused else "Pause")
        self.badge.setText("Pause requested" if self.runner.paused else "Working")

    def force_stop(self):
        if QMessageBox.question(self, "Force stop?", "Force stop the active process? Unsaved progress may be lost.") == QMessageBox.Yes:
            self.runner.force_stop()

    def _event(self, payload):
        kind = payload.get("event")
        if kind in ("prepare", "step", "generation_step"):
            current, total = payload["current"], payload["total"]
            self.progress.setRange(0, 100)
            self.progress.setValue(min(99, round(current * 100 / max(1, total))))
            self.status.setText(f"{self.runner.operation.title()}  ·  {current:,} / {total:,}")
            if kind == "step" and "loss" in payload:
                loss = payload["loss"]
                self.loss_chart.points.append(loss)
                # Keep rendering bounded for long sessions; epoch history is saved in metadata.
                if len(self.loss_chart.points) > 2000:
                    self.loss_chart.points.pop(0)
                self.loss_chart.update()
                self.train_metrics.setText(f"Batch {current:,} / {total:,}  ·  Training loss {loss:.5f}")
        elif kind == "epoch":
            self.append_log(f"Epoch {payload['epoch']} / {payload['total_epochs']}  ·  loss {payload['loss']:.5f}")
        elif kind == "diagnostics":
            self.report.setText(f"Python {payload['python']}  ·  PyTorch {payload['torch']}  ·  Lightning {payload['lightning']}\n\n"
                f"GPU: {payload['gpu']}  ·  {payload['vram_gb']} GiB VRAM\n"
                f"CUDA: {'available' if payload['cuda'] else 'unavailable'}  ·  bfloat16: {'supported' if payload['bf16'] else 'unsupported'}\n"
                f"Hugging Face login: {'found' if payload['huggingface_login'] else 'not found'}\n"
                f"FFmpeg: {'found' if payload['ffmpeg'] else 'missing'}\n\nModel access is checked when loading weights.")
        elif kind == "preview":
            path = payload.get("path")
            if path:
                self.gen_results.addItem(Path(path).name, path)
                self.gen_results.setCurrentIndex(self.gen_results.count() - 1)
                if self.pages.currentIndex() == 3:
                    self.pages.currentWidget().ensureWidgetVisible(self.gen_player)
        elif kind == "log":
            self.status.setText(payload.get("message", "Working…"))

    def _finished(self, success, message):
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.status.setText(message)
        self.append_log(message)
        if not success:
            self.logs.setVisible(True)
            self.logs_button.setText("Hide logs")
        result = self.runner.result or {}
        folder = result.get("output_dir")
        if self.runner.operation == "prepare":
            self.guard(lambda: self.refresh(dataset=folder))
        elif self.runner.operation == "train":
            self.guard(lambda: self.refresh(adapter=folder))
        else:
            self.guard(self.refresh)
        if self.close_pending:
            QTimer.singleShot(0, self.close)

    def _gen_selected(self):
        path = self.gen_results.currentData()
        if path:
            self.gen_player.load(path)

    def export_audio(self, player):
        if not player.path or not player.path.is_file():
            raise ValueError("Select a generated WAV first.")
        filename, _ = QFileDialog.getSaveFileName(self, "Save WAV", player.path.name, "WAV audio (*.wav)")
        if filename:
            target = Path(filename)
            if target.suffix.lower() != ".wav":
                target = target.with_suffix(".wav")
            if target.resolve() != player.path.resolve():
                shutil.copy2(player.path, target)
            self.status.setText(f"Saved {target.name}")

    def open_audio_folder(self, player):
        if player.path:
            self.open_path(player.path.parent)

    def library_selected(self, item, previous=None):
        self.use_adapter_button.setEnabled(False)
        self.library_player.player.stop()
        self.library_player.path = None
        self.library_player.play.setEnabled(False)
        self.library_player.waveform.points = []
        self.library_player.waveform.update()
        if not item:
            return
        entry = item.data(Qt.UserRole)
        if entry["kind"] == "adapter":
            self.use_adapter_button.setEnabled(True)
            self.library_details.setText(f"{entry.get('name', 'LoRA adapter')}\nRank {entry.get('rank', '?')}  ·  {entry.get('epochs_completed', 0)} completed epochs\n{entry['path']}")
        else:
            record = entry["record"]
            self.library_details.setText(f"{record.get('prompt', '')}\nSeed {record.get('seed', '?')}  ·  {record.get('steps', '?')} steps  ·  {record.get('duration', '?')} seconds\n{entry['path']}")
            self.library_player.load(entry["path"])

    def use_adapter(self):
        item = self.library.currentItem()
        if item:
            entry = item.data(Qt.UserRole)
            if entry["kind"] == "adapter":
                self.gen_model.setCurrentIndex(self.gen_model.findData(entry["path"]))
                self.switch_page(3)

    def open_library_folder(self):
        item = self.library.currentItem()
        if item:
            entry = item.data(Qt.UserRole)
            path = Path(entry["path"])
            self.open_path(path if entry["kind"] == "adapter" else path.parent)

    def save_settings(self):
        self.settings.update(runtime_python=self.runtime.text().strip(), hf_home=self.hf_home.text().strip(), clip_seconds=self.clip_length.value(),
            epochs=self.epochs.value(), batch_size=self.batch.value(), learning_rate=self.lr.value(), rank=self.rank.value(),
            save_every=self.save_every.value(), train_seed=self.train_seed.value(), duration=self.duration.value(),
            steps=self.steps.value(), guidance=self.guidance.value(), lora_strength=self.strength.value(), gen_seed=self.gen_seed.value())
        atomic_json(self.settings_path, self.settings)

    def closeEvent(self, event):
        if self.runner.active:
            answer = QMessageBox.question(self, "Job is running", "Stop the current job and close after it finishes?\n\nTraining saves at the next safe batch boundary.")
            if answer == QMessageBox.Yes:
                self.close_pending = True
                self.runner.stop()
                self.status.setText("Stopping before closing. Wait for the current operation, or use Force stop if it stalls.")
            event.ignore()
            return
        try:
            self.save_caption_table(quiet=True)
            self.save_settings()
        except Exception as exc:
            answer = QMessageBox.question(self, "Unable to save", f"{exc}\n\nDiscard unsaved changes and close?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        for player in self.players:
            player.shutdown()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Stable Audio Fine-Tuner desktop")
    parser.add_argument("--workspace-root", type=Path, help="App data root (defaults to the repository folder)")
    parser.add_argument("--runtime-python", type=Path, help="Use an existing compatible Stable Audio environment")
    parser.add_argument("--hf-home", type=Path, help="Optional Hugging Face cache root (containing hub)")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Stable Audio Fine-Tuner")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow(args.workspace_root, args.runtime_python, args.hf_home)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
