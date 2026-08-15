from __future__ import annotations

import threading
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Slot

from .config import AppSettings
from .media import CancelledError
from .pipeline import DatasetPipeline


class PipelineWorker(QObject):
    log = Signal(str)
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.cancel_event = threading.Event()

    def _log(self, text: str) -> None:
        """Tee runtime diagnostics to GUI and processing.log."""
        self.log.emit(text)
        try:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.settings.output_dir / "processing.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")
        except Exception:
            # Logging must never be able to break dataset processing.
            pass

    @Slot()
    def run(self):
        try:
            self._log("\n" + "=" * 72)
            self._log(f"Voice Dataset Forge run started: {datetime.now().isoformat(timespec='seconds')}")
            pipeline = DatasetPipeline(
                self.settings,
                log=self._log,
                progress=self.progress.emit,
                cancel=self.cancel_event,
            )
            result = pipeline.run()
            if result.failed_files:
                self._log(f"Run finished with errors: {result.failed_files} source file(s) failed; already committed clips were preserved.")
            else:
                self._log("Run finished successfully.")
            self.finished.emit(result)
        except CancelledError:
            self._log("Run cancelled by user.")
            self.cancelled.emit()
        except Exception as exc:
            self._log("FATAL: " + str(exc))
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self):
        self.cancel_event.set()
