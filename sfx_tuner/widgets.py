"""Qt widgets and a small, self-contained desktop theme."""
from pathlib import Path
import wave

import numpy as np
from PySide6.QtCore import Qt, QUrl, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QLabel, QListWidget,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

STYLE = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }
QMainWindow, QWidget#Shell { background: #10141d; color: #e8edf4; }
QWidget { color: #e8edf4; }
QFrame#Sidebar { background: #141a25; border-right: 1px solid #293141; }
QFrame[card="true"] { background: #191f2b; border: 1px solid #2b3444; border-radius: 12px; }
QFrame[card="true"] QLabel { background: transparent; }
QLabel#Brand { font-size: 23px; font-weight: 800; letter-spacing: 2px; color: #ffb47b; }
QLabel#PageTitle { font-size: 28px; font-weight: 700; }
QLabel#CardTitle { font-size: 16px; font-weight: 650; }
QLabel[muted="true"] { color: #94a2b8; }
QLabel#Eyebrow { color: #ffb47b; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
QLabel#Badge { color: #7ee2be; background: #19372f; padding: 7px 12px; border-radius: 14px; }
QPushButton { background: #263043; border: 1px solid #344159; padding: 9px 14px; border-radius: 7px; font-weight: 600; }
QPushButton:hover { background: #344159; }
QPushButton:pressed { background: #1e2635; }
QPushButton:disabled { color: #657187; background: #1d2431; border-color: #293141; }
QPushButton[primary="true"] { background: #f8ad76; color: #151922; border-color: #f8ad76; }
QPushButton[primary="true"]:hover { background: #ffc79f; }
QPushButton[primary="true"]:disabled { background: #534333; color: #9b8b7b; border-color: #534333; }
QPushButton[nav="true"] { text-align: left; background: transparent; border: 0; padding: 13px 16px; color: #a7b4c9; }
QPushButton[nav="true"]:checked { background: #2b303a; color: #ffc49a; border-left: 3px solid #f8ad76; }
QPushButton[nav="true"]:hover { background: #222b3b; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QListWidget, QTableWidget {
    background: #111722; border: 1px solid #344159; border-radius: 6px; padding: 7px; selection-background-color: #3a5365;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: #f8ad76; }
QComboBox QAbstractItemView { background: #1c2535; color: #e8edf4; selection-background-color: #35445c; }
QHeaderView::section { background: #222c3c; color: #a7b4c9; padding: 8px; border: 0; }
QTableWidget { gridline-color: #293141; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: #10141d; }
QScrollBar:vertical { width: 9px; background: #141a25; }
QScrollBar::handle:vertical { background: #354159; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar { background: #111722; border: 1px solid #293141; border-radius: 5px; text-align: center; min-height: 14px; }
QProgressBar::chunk { background: #73d8b3; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #344159; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; background: #f8ad76; border-radius: 6px; margin: -4px 0; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #53657d; background: #111722; border-radius: 3px; }
QCheckBox::indicator:checked { background: #73d8b3; }
QToolTip { background: #263043; color: #e8edf4; border: 1px solid #53657d; padding: 6px; }
QMessageBox { background: #191f2b; }
"""


def label(text, muted=False):
    result = QLabel(text)
    result.setWordWrap(True)
    result.setProperty("muted", muted)
    return result


def button(text, callback, primary=False):
    result = QPushButton(text)
    result.setProperty("primary", primary)
    result.clicked.connect(callback)
    return result


def card(title, description=""):
    frame = QFrame()
    frame.setProperty("card", True)
    box = QVBoxLayout(frame)
    box.setContentsMargins(20, 18, 20, 18)
    box.setSpacing(13)
    heading = label(title)
    heading.setObjectName("CardTitle")
    box.addWidget(heading)
    if description:
        box.addWidget(label(description, True))
    return frame, box


class SourceList(QListWidget):
    dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setFixedHeight(125)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.count():
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#8797af"))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter, "Drop recordings here, or choose Add files")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.dropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()


class WaveformReader(QThread):
    ready = Signal(str, list, str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = str(path)

    def run(self):
        try:
            with wave.open(self.path, "rb") as handle:
                if handle.getsampwidth() != 2:
                    raise ValueError("Waveform preview supports prepared 16-bit WAV audio.")
                rate, channels, count = handle.getframerate(), handle.getnchannels(), handle.getnframes()
                block = max(1, (count + 599) // 600)
                peaks = []
                while not self.isInterruptionRequested():
                    data = handle.readframes(block)
                    if not data:
                        break
                    values = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.
                    peaks.append((float(values.min()), float(values.max())))
            self.ready.emit(self.path, peaks, f"{count / rate:.2f} seconds  ·  {rate / 1000:g} kHz  ·  {'Stereo' if channels == 2 else 'Mono'}")
        except Exception as exc:
            self.ready.emit(self.path, [], str(exc))


class Plot(QWidget):
    def __init__(self, loss=False):
        super().__init__()
        self.loss = loss
        self.points = []
        self.position = 0
        self.setMinimumHeight(130)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#111722"))
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor("#253043"), 1))
        for i in range(1, 5):
            p.drawLine(0, i * h // 5, w, i * h // 5)
        if not self.points:
            p.setPen(QColor("#8797af"))
            p.drawText(self.rect(), Qt.AlignCenter, "Training loss will appear here" if self.loss else "Select a clip to see its waveform")
            return
        if self.loss:
            left, right, top, bottom = 48, w - 18, 18, h - 24
            # Include zero so small changes do not look like large jumps.
            high = max(max(self.points) * 1.05, 1e-9)
            p.setPen(QColor("#8797af"))
            p.drawText(6, top + 8, f"{high:.3g}")
            p.drawText(12, bottom, "0")
            p.drawText(left, h - 5, "Batch")
            path = QPainterPath()
            for i, value in enumerate(self.points):
                x, y = left + i * (right - left) / max(1, len(self.points) - 1), bottom - value / high * (bottom - top)
                if i:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
            p.setPen(QPen(QColor("#73d8b3"), 2))
            p.drawPath(path)
        else:
            p.setPen(QPen(QColor("#73d8b3"), 1.2))
            for i, (low, high) in enumerate(self.points):
                x = round(i * (w - 1) / max(1, len(self.points) - 1))
                p.drawLine(x, round(h / 2 - high * h * .42), x, round(h / 2 - low * h * .42))
            p.setPen(QPen(QColor("#f8ad76"), 1))
            x = round(self.position * (w - 1))
            p.drawLine(x, 0, x, h)


class AudioPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.path = None
        self.readers = []
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.title = label("No clip selected", True)
        box.addWidget(self.title)
        self.waveform = Plot()
        box.addWidget(self.waveform)
        row = QHBoxLayout()
        self.play = button("Play", self.toggle)
        self.play.setEnabled(False)
        row.addWidget(self.play)
        self.seek = QSlider(Qt.Horizontal)
        row.addWidget(self.seek, 1)
        self.clock = label("0:00 / 0:00", True)
        row.addWidget(self.clock)
        box.addLayout(row)
        self.details = label("WAV preview", True)
        box.addWidget(self.details)
        self.output = QAudioOutput(self)
        self.output.setVolume(.65)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.positionChanged.connect(self._position)
        self.player.durationChanged.connect(lambda value: self.seek.setRange(0, value))
        self.player.playbackStateChanged.connect(lambda state: self.play.setText("Pause" if state == QMediaPlayer.PlayingState else "Play"))
        self.player.errorOccurred.connect(lambda *_: self.details.setText(self.player.errorString()))
        self.seek.sliderMoved.connect(self.player.setPosition)

    def load(self, path):
        self.player.stop()
        self.path = Path(path)
        self.title.setText(self.path.name)
        self.player.setSource(QUrl.fromLocalFile(str(self.path)))
        self.play.setEnabled(self.path.is_file())
        self.waveform.points = []
        self.waveform.position = 0
        self.waveform.update()
        for active in self.readers:
            active.requestInterruption()
        reader = WaveformReader(self.path, self)
        self.readers.append(reader)
        reader.ready.connect(self._ready)
        reader.finished.connect(lambda r=reader: self._retire(r))
        reader.start()

    def _retire(self, reader):
        if reader in self.readers:
            self.readers.remove(reader)
        reader.deleteLater()

    def _ready(self, path, points, details):
        if self.path and path == str(self.path):
            self.waveform.points = points
            self.waveform.update()
            self.details.setText(details)

    def _position(self, value):
        duration = self.player.duration()
        if not self.seek.isSliderDown():
            self.seek.setValue(value)
        def clock(ms):
            seconds = ms // 1000
            return f"{seconds // 60}:{seconds % 60:02d}"
        self.clock.setText(f"{clock(value)} / {clock(duration)}")
        self.waveform.position = value / max(1, duration)
        self.waveform.update()

    def toggle(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.position() >= self.player.duration():
                self.player.setPosition(0)
            self.player.play()

    def shutdown(self):
        self.player.stop()
        for reader in list(self.readers):
            reader.requestInterruption()
            reader.wait()
