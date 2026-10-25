"""Render actual desktop pages for visual inspection; no mock training results."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from sfx_tuner.app import MainWindow
from sfx_tuner.widgets import STYLE


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    # Windows' offscreen plugin does not enumerate installed fonts.
    if not QFontDatabase.families():
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf"):
            if (fonts / name).is_file():
                QFontDatabase.addApplicationFont(str(fonts / name))
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    with TemporaryDirectory() as folder:
        window = MainWindow(Path(folder))
        # Display repository-relative placeholder paths in public screenshots.
        window.runtime.setText("runtime/.venv/Scripts/python.exe")
        window.show()
        for index, name in enumerate(("setup", "dataset", "train", "generate", "library")):
            window.switch_page(index)
            app.processEvents()
            output = ROOT / "docs" / f"{name}-preview.png"
            window.grab().save(str(output))
            print(output.name)
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
