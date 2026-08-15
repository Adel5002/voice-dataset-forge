from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import AppSettings, QUALITY_PRESETS
from .models import PipelineSummary
from .worker import PipelineWorker
from .pipeline import shutdown_process_speaker_encoder


QUALITY_HELP = {
    "Быстро": "Без удаления музыки. Memory-safe diarization чанками до 5 минут; подходит для локальной проверки.",
    "Сбалансированно": "Demucs htdemucs + строгая speaker-фильтрация. На малой VRAM чанки и Demucs segment уменьшаются автоматически.",
    "Максимум": "htdemucs_ft + повторная проверка клипов. При CUDA OOM тяжёлый этап автоматически повторяется на CPU без снижения качества.",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Dataset Forge 1.1")
        self.resize(980, 720)
        self.settings_store = QSettings("VoiceDatasetForge", "VoiceDatasetForge")
        self.thread: QThread | None = None
        self.worker: PipelineWorker | None = None
        self._build_ui()
        self._restore()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Voice Dataset Forge")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        subtitle = QLabel("Стримы → только твой голос → единый готовый датасет для Voice Conversion")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        form_box = QFrame()
        form_box.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(form_box)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(11)

        self.streams_edit = QLineEdit()
        self.refs_edit = QLineEdit()
        self.output_edit = QLineEdit()
        form.addRow("Папка со стримами", self._path_row(self.streams_edit, self._pick_streams))
        form.addRow("Папка с референсом", self._path_row(self.refs_edit, self._pick_refs))
        form.addRow("Папка датасета", self._path_row(self.output_edit, self._pick_output))

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(list(QUALITY_PRESETS.keys()))
        self.quality_combo.currentTextChanged.connect(self._quality_changed)
        form.addRow("Качество", self.quality_combo)
        self.quality_help = QLabel()
        self.quality_help.setWordWrap(True)
        form.addRow("", self.quality_help)

        self.token_edit = QLineEdit(os.getenv("HUGGINGFACE_TOKEN", ""))
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("hf_…  (не сохраняется приложением)")
        form.addRow("Hugging Face token", self.token_edit)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("Устройство", self.device_combo)

        self.language_edit = QLineEdit("auto")
        self.language_edit.setPlaceholderText("auto / ru / en …")
        form.addRow("Язык транскрипции", self.language_edit)

        self.transcribe_check = QCheckBox("Создавать транскрипции для metadata.csv")
        self.transcribe_check.setChecked(True)
        form.addRow("", self.transcribe_check)
        self.rejected_check = QCheckBox("Сохранять отбракованные кандидатные клипы")
        self.rejected_check.setChecked(True)
        form.addRow("", self.rejected_check)
        layout.addWidget(form_box)

        note = QLabel(
            "В режимах Сбалансированно/Максимум музыка автоматически удаляется не только из стримов, "
            "но и из референсных записей перед построением voice embedding. "
            "Hugging Face token не записывается в settings.json. "
            "v1.1 самовосстанавливается после ручного удаления output/cache-артефактов; Demucs/pyannote работают на GPU, а один CPU-WeSpeaker переиспользуется между последовательными запусками приложения без повторной загрузки checkpoint."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Собрать датасет")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        buttons.addWidget(self.start_btn, 1)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = QLabel("Готов к запуску")
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(4000)
        layout.addWidget(self.log_edit, 1)

        self.setCentralWidget(root)
        self._quality_changed(self.quality_combo.currentText())

    def _path_row(self, edit: QLineEdit, callback):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Обзор…")
        btn.clicked.connect(callback)
        row.addWidget(edit, 1)
        row.addWidget(btn)
        return widget

    def _pick_streams(self):
        self._pick_folder(self.streams_edit, "Выбери папку со стримами")

    def _pick_refs(self):
        self._pick_folder(self.refs_edit, "Выбери папку с референсами")

    def _pick_output(self):
        self._pick_folder(self.output_edit, "Выбери папку проекта датасета")

    def _pick_folder(self, edit: QLineEdit, title: str):
        start = edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, title, start)
        if path:
            edit.setText(path)

    def _quality_changed(self, name: str):
        self.quality_help.setText(QUALITY_HELP.get(name, ""))

    def _collect_settings(self) -> AppSettings:
        return AppSettings(
            streams_dir=Path(self.streams_edit.text().strip()),
            references_dir=Path(self.refs_edit.text().strip()),
            output_dir=Path(self.output_edit.text().strip()),
            quality=self.quality_combo.currentText(),
            hf_token=self.token_edit.text().strip(),
            device=self.device_combo.currentText(),
            transcribe=self.transcribe_check.isChecked(),
            language=self.language_edit.text().strip() or "auto",
            keep_rejected=self.rejected_check.isChecked(),
        )

    def _start(self):
        try:
            if not self.streams_edit.text().strip():
                raise ValueError("Укажи папку со стримами")
            if not self.refs_edit.text().strip():
                raise ValueError("Укажи папку с референсами")
            if not self.output_edit.text().strip():
                raise ValueError("Укажи папку датасета")
            cfg = self._collect_settings()
            if not cfg.hf_token:
                raise ValueError("Вставь Hugging Face token для pyannote")
        except ValueError as exc:
            QMessageBox.warning(self, "Не хватает настроек", str(exc))
            return

        self._persist()
        self.log_edit.clear()
        self.progress.setValue(0)
        self._set_running(True)

        self.thread = QThread(self)
        self.worker = PipelineWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _cancel(self):
        if self.worker:
            self.status.setText("Запрошена отмена…")
            self.worker.cancel()

    def _on_progress(self, value: int, message: str):
        self.progress.setValue(max(0, min(100, value)))
        self.status.setText(message)

    def _append_log(self, text: str):
        self.log_edit.appendPlainText(text.rstrip())
        bar = self.log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_finished(self, summary: PipelineSummary):
        self.progress.setValue(100)
        partial = summary.failed_files > 0
        self.status.setText("Завершено с ошибками" if partial else "Датасет готов")
        hours = summary.accepted_seconds / 3600.0
        title = "Завершено с ошибками" if partial else "Готово"
        text = (
            f"Добавлено клипов: {summary.accepted_clips}\n"
            f"Чистого аудио: {hours:.2f} ч\n"
            f"Ошибок файлов: {summary.failed_files}"
        )
        if partial:
            text += "\n\nУже сохранённые WAV не потеряны. Подробности — processing.log."
            QMessageBox.warning(self, title, text)
        else:
            QMessageBox.information(self, title, text)

    def _on_failed(self, error: str):
        self.status.setText("Ошибка")
        self._append_log("FATAL: " + error)
        QMessageBox.critical(self, "Ошибка", error)

    def _on_cancelled(self):
        self.status.setText("Обработка отменена")
        self._append_log("Обработка отменена.")

    def _thread_finished(self):
        self._set_running(False)
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        for widget in [
            self.streams_edit, self.refs_edit, self.output_edit, self.quality_combo,
            self.token_edit, self.device_combo, self.language_edit,
            self.transcribe_check, self.rejected_check,
        ]:
            widget.setEnabled(not running)

    def _persist(self):
        self.settings_store.setValue("streams", self.streams_edit.text())
        self.settings_store.setValue("refs", self.refs_edit.text())
        self.settings_store.setValue("output", self.output_edit.text())
        self.settings_store.setValue("quality", self.quality_combo.currentText())
        self.settings_store.setValue("device", self.device_combo.currentText())
        self.settings_store.setValue("language", self.language_edit.text())
        self.settings_store.setValue("transcribe", self.transcribe_check.isChecked())
        self.settings_store.setValue("rejected", self.rejected_check.isChecked())

    def _restore(self):
        self.streams_edit.setText(str(self.settings_store.value("streams", "")))
        self.refs_edit.setText(str(self.settings_store.value("refs", "")))
        self.output_edit.setText(str(self.settings_store.value("output", "")))
        quality = str(self.settings_store.value("quality", "Максимум"))
        idx = self.quality_combo.findText(quality)
        if idx >= 0:
            self.quality_combo.setCurrentIndex(idx)
        device = str(self.settings_store.value("device", "auto"))
        idx = self.device_combo.findText(device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.language_edit.setText(str(self.settings_store.value("language", "auto")))
        self.transcribe_check.setChecked(_as_bool(self.settings_store.value("transcribe", True)))
        self.rejected_check.setChecked(_as_bool(self.settings_store.value("rejected", True)))


    def closeEvent(self, event):
        # The session-level WeSpeaker cache intentionally survives individual
        # dataset runs. If no worker is active, free it explicitly; during an
        # active run the OS will reclaim it on process exit and we must not tear
        # the model out from under the worker thread.
        if not self.thread or not self.thread.isRunning():
            shutdown_process_speaker_encoder()
        super().closeEvent(event)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def run_app():
    app = QApplication(sys.argv)
    app.setApplicationName("Voice Dataset Forge")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
