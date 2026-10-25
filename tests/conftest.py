import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def wait_until(qapp):
    def wait(predicate, timeout=12):
        end = time.monotonic() + timeout
        while not predicate() and time.monotonic() < end:
            qapp.processEvents()
            time.sleep(.01)
        qapp.processEvents()
        assert predicate(), "Timed out waiting for Qt operation"
    return wait
