from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import threading
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .config import AppSettings, SUPPORTED_MEDIA
from .diarization import Diarizer
from .media import (
    CancelledError,
    analyze_pcm16_wav,
    concatenate_pcm_segments,
    convert_analysis_wav,
    cut_analysis_chunk,
    cut_wav,
    extract_highres_audio,
    probe_duration,
    require_ffmpeg,
)
from .metadata import DatasetWriter, repair_dataset_metadata
from .models import ClipCandidate, DiarizedSegment, PipelineSummary, StagedClip
from .resources import (
    is_cuda_oom,
    log_memory,
    release_accelerator_memory,
    resolve_demucs_segment,
    resolve_diarization_window,
    resolve_torch_device,
)
from .separation import demucs_available, separate_vocals
from .speaker import SpeakerEncoder
from .state import ProjectState
from .transcription import Transcriber

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, str], None]

DATASET_ALGORITHM_VERSION = 2

# WeSpeaker/Lightning proved sensitive to repeated Model.from_pretrained calls
# inside one long-lived Windows process. Keep one CPU encoder for the whole GUI
# session and reuse it across independent dataset runs. GUI runs are serialized.
_PROCESS_SPEAKER_ENCODER: SpeakerEncoder | None = None
_PROCESS_SPEAKER_LOCK = threading.RLock()


def shutdown_process_speaker_encoder() -> None:
    global _PROCESS_SPEAKER_ENCODER
    with _PROCESS_SPEAKER_LOCK:
        if _PROCESS_SPEAKER_ENCODER is not None:
            try:
                _PROCESS_SPEAKER_ENCODER.close()
            finally:
                _PROCESS_SPEAKER_ENCODER = None


class DatasetPipeline:
    """Memory-safe staged dataset builder.

    v1.1 keeps GPU-heavy stages isolated and deliberately runs the WeSpeaker
    verification model on CPU. One WeSpeaker instance is created before diarization
    and reused for the whole application run. This avoids a second pyannote/Lightning
    checkpoint load after a long diarization pass, which proved unstable on Windows.
    Results are cached to disk between phases.
    """

    def __init__(self, settings: AppSettings, log: LogFn | None = None,
                 progress: ProgressFn | None = None, cancel: threading.Event | None = None):
        self.settings = settings
        self.log = log or (lambda _: None)
        self.progress = progress or (lambda _p, _m: None)
        self.cancel = cancel or threading.Event()
        self.output = settings.output_dir
        self.cache = self.output / ".cache"
        self.state = ProjectState(self.output)
        self.summary = PipelineSummary()
        self.reference_centroid = None
        self._speaker_encoder: SpeakerEncoder | None = None

    def run(self) -> PipelineSummary:
        self._validate()
        self._prepare_output()
        streams = self._discover(self.settings.streams_dir)
        references = self._discover(self.settings.references_dir)
        if not streams:
            raise RuntimeError("В папке со стримами не найдено поддерживаемых медиафайлов.")
        if not references:
            raise RuntimeError("В папке с референсами не найдено поддерживаемых файлов.")

        self.summary.source_files = len(streams)
        self.log(f"Найдено стримов: {len(streams)}; референсов: {len(references)}")
        self.log("v1.1 self-healing: missing/stale project artifacts are rebuilt; one CPU WeSpeaker instance is reused for the whole app session.")
        log_memory(self.log, "старт")

        self.progress(1, "Подготовка референсов")
        prepared_refs = self._prepare_references(references)
        self.progress(6, "Построение voice embedding референса")
        self.reference_centroid = self._build_reference_centroid(prepared_refs)
        self.progress(10, "Референсный голос готов")
        log_memory(self.log, "после референсов")

        writer = DatasetWriter(self.output, self.state.next_audio_index())
        baseline_count, baseline_seconds = _dataset_totals(self.output / "metadata_extended.csv")
        work_start = 10.0
        work_span = 90.0
        per_stream = work_span / max(1, len(streams))

        for index, source in enumerate(streams, start=1):
            self._check_cancel()
            fp = self.state.fingerprint(source)
            base_progress = work_start + (index - 1) * per_stream
            self.progress(int(base_progress), f"{index}/{len(streams)} — {source.name}")
            if self.state.is_done(fp):
                self.summary.skipped_files += 1
                self.log(f"SKIP: уже обработан — {source.name}")
                continue
            try:
                committed_count, committed_seconds, source_duration, rejected_seconds, rejected_count = self._process_stream(
                    source, fp, writer, base_progress, per_stream, index, len(streams)
                )
                self.summary.source_seconds += source_duration
                self.summary.accepted_seconds += committed_seconds
                self.summary.accepted_clips += committed_count
                self.summary.rejected_seconds += rejected_seconds
                self.summary.rejected_clips += rejected_count
                self.state.mark_done(fp, source, committed_count, committed_seconds)
                self.log(
                    f"DONE: {source.name}: добавлено {committed_count} новых клипов, "
                    f"{_fmt_seconds(committed_seconds)} нового чистого голоса"
                )
                release_accelerator_memory()
                log_memory(self.log, f"после {source.name}")
            except CancelledError:
                raise
            except Exception as exc:
                release_accelerator_memory()
                self.summary.failed_files += 1
                self.state.mark_failed(fp, source, str(exc))
                self.log(f"ERROR: {source.name}: {exc}")
                self.log(traceback.format_exc(limit=6))
                log_memory(self.log, f"после ошибки {source.name}")

        # Accepted audio is append-only and may already be committed even if a
        # later chunk of the same source fails. Derive run totals from disk so
        # the GUI/report never claims 0 clips after a partial failure.
        final_count, final_seconds = _dataset_totals(self.output / "metadata_extended.csv")
        self.summary.accepted_clips = max(0, final_count - baseline_count)
        self.summary.accepted_seconds = max(0.0, final_seconds - baseline_seconds)

        self.progress(100, "Готово" if self.summary.failed_files == 0 else "Завершено с ошибками")
        self._write_statistics()
        self._close_speaker_encoder()
        release_accelerator_memory()
        return self.summary

    def _validate(self) -> None:
        require_ffmpeg()
        for path, title in [
            (self.settings.streams_dir, "папка со стримами"),
            (self.settings.references_dir, "папка с референсами"),
        ]:
            if not Path(path).is_dir():
                raise RuntimeError(f"Не найдена {title}: {path}")
        if self.settings.preset.use_separation and not demucs_available():
            raise RuntimeError(
                "Режим Сбалансированно/Максимум требует Demucs. "
                "Установи зависимости из requirements.txt или временно выбери 'Быстро'."
            )

    def _prepare_output(self) -> None:
        # Make manual cleanup safe. If the user deletes final WAV/metadata but
        # leaves hidden state/cache files, stale bookkeeping must not cause SKIP
        # or duplicate decisions. Cache is intentionally preserved and reused only
        # when the concrete artifact it points to still exists.
        self.output.mkdir(parents=True, exist_ok=True)
        existing_settings = self.output / "settings.json"
        existing_audio = self.output / "audio"
        repair = repair_dataset_metadata(self.output)
        metadata_exists = (self.output / "metadata.csv").exists() or (self.output / "metadata_extended.csv").exists()
        audio_exists = existing_audio.exists() and any(existing_audio.glob("*.wav"))
        if repair.get("removed", 0) > 0:
            self.log(f"SELF-HEAL: удалено {repair['removed']} metadata-записей, чьи WAV больше не существуют.")
            if self.state.has_done_files():
                self.state.reset_files()
                self.log("SELF-HEAL: completion state сброшен — исходники будут обработаны заново.")
        elif self.state.has_done_files() and not audio_exists and not metadata_exists:
            self.state.reset_files()
            self.log("SELF-HEAL: итоговый датасет очищен вручную; старый completion state сброшен.")
        if existing_settings.exists() and existing_audio.exists() and any(existing_audio.glob("*.wav")):
            try:
                old = json.loads(existing_settings.read_text(encoding="utf-8"))
                old_quality = old.get("quality")
                old_rate = int(old.get("output_sample_rate", self.settings.output_sample_rate))
                old_algorithm = int(old.get("dataset_algorithm_version", 1))
                if old_algorithm != DATASET_ALGORITHM_VERSION:
                    raise RuntimeError(
                        "В этой папке уже есть WAV, созданные старой логикой speaker selection. "
                        "Для v1.1 выбери новую пустую папку датасета — старые клипы автоматически не удаляются."
                    )
                if old_quality and old_quality != self.settings.quality:
                    raise RuntimeError(
                        f"Этот dataset project уже создан в режиме '{old_quality}'. "
                        f"Для режима '{self.settings.quality}' выбери новую папку вывода, "
                        "чтобы не смешивать клипы разного качества."
                    )
                if old_rate != self.settings.output_sample_rate:
                    raise RuntimeError(
                        "Нельзя менять sample rate внутри уже существующего dataset project. "
                        "Выбери новую папку вывода."
                    )
            except json.JSONDecodeError:
                self.log("WARNING: существующий settings.json повреждён; продолжаю с текущими настройками")
        for folder in [self.output, self.output / "audio", self.output / "rejected", self.cache]:
            folder.mkdir(parents=True, exist_ok=True)
        public_settings = self.settings.public_dict()
        public_settings["dataset_algorithm_version"] = DATASET_ALGORITHM_VERSION
        (self.output / "settings.json").write_text(
            json.dumps(public_settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _discover(self, folder: Path) -> list[Path]:
        return sorted(
            [p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA],
            key=lambda p: str(p).lower(),
        )

    # ---------- references ----------

    def _prepare_references(self, references: list[Path]) -> list[Path]:
        ref_cache = self.cache / "references"
        ref_cache.mkdir(parents=True, exist_ok=True)
        preset = self.settings.preset
        if preset.use_separation:
            self.log(
                f"Референсы автоматически очищаются от музыки через Demucs "
                f"({preset.demucs_model}) перед speaker embedding."
            )
        else:
            self.log("Режим 'Быстро': референсы используются без source separation.")

        prepared: list[Path] = []
        for idx, src in enumerate(references, start=1):
            self._check_cancel()
            self.progress(1 + int(4 * idx / max(1, len(references))),
                          f"Референс {idx}/{len(references)}")
            prepared.append(self._prepare_reference(src, idx, len(references), ref_cache))
        return prepared

    def _build_reference_centroid(self, prepared: list[Path]):
        import numpy as np

        # Cache the target centroid, but keep one CPU WeSpeaker instance alive for
        # the entire run. The same encoder is reused later for speaker verification,
        # avoiding a second Model.from_pretrained/Lightning checkpoint load after
        # pyannote diarization.
        signature_parts = ["wespeaker-voxceleb-resnet34-LM|v1.0"]
        for path in prepared:
            stat = path.stat()
            signature_parts.append(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
        key = hashlib.sha1("\n".join(signature_parts).encode()).hexdigest()[:20]
        centroid_dir = self.cache / "references" / "centroids"
        centroid_dir.mkdir(parents=True, exist_ok=True)
        centroid_path = centroid_dir / f"{key}.npy"

        # Load the encoder before diarization even when the centroid is cached.
        # Holding this CPU model costs RAM, not VRAM, and guarantees verification
        # will not trigger another checkpoint load later in the run.
        encoder = self._get_speaker_encoder("референс + speaker verification")

        if centroid_path.exists():
            try:
                centroid = np.load(centroid_path, allow_pickle=False).astype("float32")
                if centroid.size == 0 or not np.isfinite(centroid).all():
                    raise ValueError("пустой/нечисловой centroid")
                self.log(f"Использую кэш voice centroid: {centroid_path.name}")
                return centroid
            except Exception as exc:
                self.log(f"SELF-HEAL: voice centroid кэш повреждён/не читается ({exc}); пересчитываю.")
                centroid_path.unlink(missing_ok=True)

        self.log(f"Строю voice centroid из {len(prepared)} подготовленных референсов…")
        centroid = encoder.centroid(prepared)
        np.save(centroid_path, centroid, allow_pickle=False)
        self.log(f"Voice centroid сохранён в кэш: {centroid_path.name}")
        return centroid

    def _prepare_reference(self, src: Path, index: int, total: int, ref_cache: Path) -> Path:
        preset = self.settings.preset
        stat = src.stat()
        cache_signature = (
            f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"separation={preset.use_separation}|model={preset.demucs_model}"
        )
        key = hashlib.sha1(cache_signature.encode()).hexdigest()[:16]
        item_dir = ref_cache / key
        item_dir.mkdir(parents=True, exist_ok=True)

        highres = item_dir / "source_48k.wav"
        if not highres.exists():
            self.log(f"  reference {index}/{total}: извлекаю аудио — {src.name}")
            extract_highres_audio(src, highres, self.cancel)

        processed = highres
        flavor = "raw"
        if preset.use_separation:
            flavor = preset.demucs_model
            cleaned = item_dir / f"vocals_{preset.demucs_model}.wav"
            if not cleaned.exists():
                self.log(f"  reference {index}/{total}: убираю музыку ({preset.demucs_model}) — {src.name}")
                demucs_vocals = self._separate_memory_safe(highres, item_dir, preset.demucs_model)
                shutil.copy2(demucs_vocals, cleaned)
            else:
                self.log(f"  reference {index}/{total}: использую кэш очистки — {src.name}")
            processed = cleaned

        analysis = item_dir / f"analysis_{flavor}.wav"
        if not analysis.exists():
            convert_analysis_wav(processed, analysis, self.cancel)
        return analysis

    # ---------- stream phases ----------

    def _process_stream(self, source: Path, fingerprint: str, writer: DatasetWriter,
                        progress_base: float = 10.0, progress_span: float = 90.0,
                        stream_index: int = 1, stream_total: int = 1
                        ) -> tuple[int, float, float, float, int]:
        preset = self.settings.preset
        work = self.cache / "streams" / fingerprint
        work.mkdir(parents=True, exist_ok=True)
        stage = work / "stage"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)

        # Phase A: extraction / source separation. No pyannote/Whisper model is resident.
        highres = work / "source_48k.wav"
        self.progress(int(progress_base + progress_span * 0.03),
                      f"{stream_index}/{stream_total} — извлечение аудио")
        if not highres.exists():
            self.log(f"Извлекаю аудио: {source.name}")
            extract_highres_audio(source, highres, self.cancel)

        processed = highres
        if preset.use_separation:
            cached_vocals = work / f"vocals_{preset.demucs_model}.wav"
            if not cached_vocals.exists():
                self.progress(int(progress_base + progress_span * 0.10),
                              f"{stream_index}/{stream_total} — Demucs: отделение голоса")
                self.log(f"Отделяю голос от музыки ({preset.demucs_model})…")
                log_memory(self.log, "перед Demucs")
                demucs_vocals = self._separate_memory_safe(highres, work, preset.demucs_model)
                shutil.copy2(demucs_vocals, cached_vocals)
                log_memory(self.log, "после Demucs")
            processed = cached_vocals

        analysis = work / f"analysis_{preset.demucs_model if preset.use_separation else 'raw'}.wav"
        self.progress(int(progress_base + progress_span * 0.27),
                      f"{stream_index}/{stream_total} — подготовка анализа")
        if not analysis.exists():
            convert_analysis_wav(processed, analysis, self.cancel)

        duration = probe_duration(analysis)
        window = resolve_diarization_window(preset.diarization_window_seconds, self.settings.device)
        self.log(
            f"Memory-safe diarization window: {window}s "
            f"(preset limit {preset.diarization_window_seconds}s)."
        )

        # Phase B: diarization only. Results are cached, then pyannote is released.
        chunks = self._diarize_stream(
            analysis, work, duration, window, progress_base, progress_span,
            stream_index, stream_total,
        )

        # Phase C: speaker verification + clip quality only. Diarizer is gone.
        committed_count, committed_seconds, committed_for_transcription, rejected_seconds, rejected_count = self._filter_and_commit(
            source, processed, fingerprint, chunks, writer,
            progress_base, progress_span, stream_index, stream_total,
        )

        # Phase D: Whisper only, and only after speaker model is gone. Audio is
        # already visible in output/audio at this point. Transcription merely
        # patches metadata for those committed files.
        if self.settings.transcribe and committed_for_transcription:
            pending = [clip for clip in committed_for_transcription if not clip.text]
            if pending:
                self._transcribe_staged(pending, progress_base, progress_span, stream_index, stream_total)
                writer.update_texts(pending)

        self.progress(int(progress_base + progress_span * 0.98),
                      f"{stream_index}/{stream_total} — финализация клипов")
        return committed_count, committed_seconds, duration, rejected_seconds, rejected_count

    def _diarize_stream(self, analysis: Path, work: Path, duration: float, window: int,
                        progress_base: float, progress_span: float,
                        stream_index: int, stream_total: int) -> list[tuple[int, float, Path, Path, Path]]:
        preset = self.settings.preset
        total_chunks = max(1, math.ceil(duration / window))
        flavor = preset.demucs_model if preset.use_separation else "raw"
        chunk_root = work / f"chunks_{_safe_name(self.settings.quality)}_{flavor}_{window}s"
        chunk_root.mkdir(parents=True, exist_ok=True)
        chunks: list[tuple[int, float, Path, Path, Path]] = []

        missing_diarization = False
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * window
            chunk_duration = min(window, duration - chunk_start)
            if chunk_duration <= 0.05:
                break
            chunk_dir = chunk_root / f"{chunk_idx:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_wav = chunk_dir / "analysis.wav"
            diar_json = chunk_dir / "diarization.json"
            if not chunk_wav.exists():
                cut_analysis_chunk(analysis, chunk_wav, chunk_start, chunk_duration, self.cancel)
            if diar_json.exists():
                try:
                    Diarizer.load(diar_json)
                except Exception as exc:
                    self.log(f"SELF-HEAL: битый diarization cache chunk {chunk_idx + 1}: {exc}; пересчитываю.")
                    diar_json.unlink(missing_ok=True)
            if not diar_json.exists():
                missing_diarization = True
            chunks.append((chunk_idx, chunk_start, chunk_dir, chunk_wav, diar_json))

        if not missing_diarization:
            self.log(f"  diarization: весь результат найден в кэше ({len(chunks)} chunks).")
            return chunks

        release_accelerator_memory()
        diarizer = self._load_diarizer("diarization")
        try:
            for pos, (chunk_idx, _chunk_start, _chunk_dir, chunk_wav, diar_json) in enumerate(chunks):
                self._check_cancel()
                if diar_json.exists():
                    continue
                frac = (pos + 1) / max(1, len(chunks))
                self.progress(
                    int(progress_base + progress_span * (0.30 + 0.27 * frac)),
                    f"{stream_index}/{stream_total} — diarization {pos + 1}/{len(chunks)}",
                )
                chunk_duration = probe_duration(chunk_wav)
                self.log(f"  diarization {pos + 1}/{len(chunks)} ({_fmt_seconds(chunk_duration)}) [{diarizer.device}]")
                log_memory(self.log, f"перед pyannote chunk {pos + 1}/{len(chunks)}")
                try:
                    segments = diarizer.run(chunk_wav)
                except RuntimeError as exc:
                    if diarizer.device == "cuda" and is_cuda_oom(exc):
                        self.log("WARNING: pyannote получил CUDA OOM; переключаю diarization на CPU и повторяю chunk.")
                        diarizer.close()
                        diarizer = SpeakerlessDiarizerFactory.create(self.settings.hf_token, "cpu")
                        log_memory(self.log, "pyannote CPU fallback")
                        segments = diarizer.run(chunk_wav)
                    else:
                        raise
                Diarizer.save(diar_json, segments)
                log_memory(self.log, f"после pyannote chunk {pos + 1}/{len(chunks)}")
        finally:
            diarizer.close()
            log_memory(self.log, "pyannote выгружен")
        return chunks

    def _filter_and_commit(self, source: Path, processed: Path, fingerprint: str,
                           chunks: list[tuple[int, float, Path, Path, Path]], writer: DatasetWriter,
                           progress_base: float, progress_span: float,
                           stream_index: int, stream_total: int
                           ) -> tuple[int, float, list[StagedClip], float, int]:
        """Filter diarized chunks and commit accepted audio immediately.

        Unlike v0.6, accepted clips are moved into the final dataset after each
        internal chunk, so a two-hour source never has to finish before the user
        sees files in output/audio. DatasetWriter deduplicates source/start/end on
        restart, making this safe across crashes.
        """
        preset = self.settings.preset
        stage = self.cache / "streams" / fingerprint / "stage"
        committed_count = 0
        committed_seconds = 0.0
        transcription_queue: list[StagedClip] = []
        rejected_seconds = 0.0
        rejected_count = 0

        release_accelerator_memory()
        encoder = self._get_speaker_encoder("speaker verification")
        self.log("Переиспользую уже загруженный WeSpeaker — повторной загрузки checkpoint нет.")
        for pos, (chunk_idx, chunk_start, chunk_dir, chunk_wav, diar_json) in enumerate(chunks):
            self._check_cancel()
            frac = (pos + 1) / max(1, len(chunks))
            self.progress(
                int(progress_base + progress_span * (0.58 + 0.27 * frac)),
                f"{stream_index}/{stream_total} — speaker filter {pos + 1}/{len(chunks)}",
            )
            segments = Diarizer.load(diar_json)
            self.log(f"  speaker filter {pos + 1}/{len(chunks)}: {len(segments)} diarized segments")
            candidates_json = chunk_dir / "candidates_v2.json"
            candidates = None
            if candidates_json.exists():
                try:
                    raw = json.loads(candidates_json.read_text(encoding="utf-8"))
                    candidates = [ClipCandidate(**item) for item in raw]
                except Exception as exc:
                    self.log(f"SELF-HEAL: битый candidates cache chunk {pos + 1}: {exc}; пересчитываю.")
                    candidates_json.unlink(missing_ok=True)
            if candidates is None:
                candidates = self._identify_candidates(
                    encoder, chunk_wav, segments, chunk_start, chunk_dir
                )
                candidates_json.write_text(
                    json.dumps([c.__dict__ for c in candidates], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            self.summary.candidate_seconds += sum(c.duration for c in candidates)

            chunk_staged: list[StagedClip] = []
            for cand_idx, cand in enumerate(candidates):
                self._check_cancel()
                temp = stage / f"c{chunk_idx:04d}_{cand_idx:06d}.wav"
                cut_wav(processed, temp, cand.start, cand.end, self.settings.output_sample_rate, self.cancel)
                metrics = analyze_pcm16_wav(
                    temp,
                    preset.min_rms_dbfs,
                    preset.max_clipping_ratio,
                    preset.max_silence_ratio,
                    preset.min_clip_seconds,
                    preset.max_clip_seconds,
                )
                similarity = cand.speaker_similarity
                reason = metrics.reason
                accepted = metrics.accepted

                if accepted and preset.segment_verify:
                    similarity = encoder.similarity_file(temp, self.reference_centroid)
                    if similarity < preset.segment_threshold:
                        accepted = False
                        reason = "speaker_low_confidence"

                if not accepted:
                    rejected_seconds += metrics.duration
                    rejected_count += 1
                    if self.settings.keep_rejected:
                        self._store_rejected(temp, source, cand, reason or "quality", fingerprint)
                    else:
                        temp.unlink(missing_ok=True)
                    continue

                chunk_staged.append(StagedClip(
                    path=temp,
                    text="",
                    source=str(source.name),
                    start=cand.start,
                    end=cand.end,
                    speaker_similarity=similarity,
                    metrics=metrics,
                ))

            new_count, new_seconds, usable = writer.commit_detailed(chunk_staged)
            committed_count += new_count
            committed_seconds += new_seconds
            if self.settings.transcribe:
                transcription_queue.extend(usable)
            self.log(
                f"    chunk {pos + 1}/{len(chunks)} → audio/: "
                f"{new_count} новых клипов, {_fmt_seconds(new_seconds)}"
            )
        return committed_count, committed_seconds, transcription_queue, rejected_seconds, rejected_count

    def _transcribe_staged(self, staged: list[StagedClip], progress_base: float, progress_span: float,
                           stream_index: int, stream_total: int) -> None:
        model_name = self.settings.preset.whisper_model
        self.progress(int(progress_base + progress_span * 0.87),
                      f"{stream_index}/{stream_total} — загрузка Whisper {model_name}")
        release_accelerator_memory()
        transcriber = self._load_transcriber(model_name)
        try:
            for i, clip in enumerate(staged, start=1):
                self._check_cancel()
                if i == 1 or i % 25 == 0 or i == len(staged):
                    frac = i / max(1, len(staged))
                    self.progress(
                        int(progress_base + progress_span * (0.88 + 0.09 * frac)),
                        f"{stream_index}/{stream_total} — транскрипция {i}/{len(staged)}",
                    )
                try:
                    clip.text = transcriber.transcribe(clip.path)
                except RuntimeError as exc:
                    if transcriber.device == "cuda" and is_cuda_oom(exc):
                        self.log("WARNING: Whisper получил CUDA OOM; продолжаю транскрипцию на CPU.")
                        transcriber.close()
                        transcriber = Transcriber(model_name, "cpu", self.settings.language)
                        clip.text = transcriber.transcribe(clip.path)
                    else:
                        raise
        finally:
            transcriber.close()
            log_memory(self.log, "Whisper выгружен")

    # ---------- model/runtime helpers ----------

    def _load_diarizer(self, label: str) -> Diarizer:
        release_accelerator_memory()
        device = resolve_torch_device(self.settings.device)
        self.log(f"Загружаю pyannote community-1 [{device}] — {label}")
        log_memory(self.log, "перед загрузкой pyannote")
        try:
            return SpeakerlessDiarizerFactory.create(self.settings.hf_token, device)
        except RuntimeError as exc:
            if device == "cuda" and is_cuda_oom(exc):
                self.log("WARNING: pyannote не поместился в VRAM при загрузке; использую CPU.")
                release_accelerator_memory()
                return SpeakerlessDiarizerFactory.create(self.settings.hf_token, "cpu")
            raise

    def _load_speaker_encoder(self, label: str) -> SpeakerEncoder:
        # CPU-only by design. Keep one process-wide instance because repeated
        # pyannote/Lightning Model.from_pretrained calls in the same Windows GUI
        # process can be unstable even when separate DatasetPipeline objects are used.
        global _PROCESS_SPEAKER_ENCODER
        release_accelerator_memory()
        device = "cpu"
        with _PROCESS_SPEAKER_LOCK:
            if _PROCESS_SPEAKER_ENCODER is not None:
                self.log(f"WeSpeaker session cache [cpu] — {label}; checkpoint повторно не загружается.")
                return _PROCESS_SPEAKER_ENCODER
            self.log(f"Загружаю WeSpeaker ResNet34 [{device}] — {label} (app-session single-load)")
            log_memory(self.log, "перед загрузкой speaker encoder")
            _PROCESS_SPEAKER_ENCODER = SpeakerEncoder(self.settings.hf_token, device)
            return _PROCESS_SPEAKER_ENCODER

    def _get_speaker_encoder(self, label: str) -> SpeakerEncoder:
        if self._speaker_encoder is None:
            self._speaker_encoder = self._load_speaker_encoder(label)
        else:
            self.log(f"WeSpeaker уже подключён к текущему run [cpu] — {label}")
        return self._speaker_encoder

    def _close_speaker_encoder(self) -> None:
        # Detach only. The process-wide CPU model stays resident until the GUI
        # closes so a second run/mode switch does not reload the checkpoint.
        if self._speaker_encoder is not None:
            self._speaker_encoder = None
            self.log("WeSpeaker оставлен в session cache для следующего запуска приложения.")

    def _load_transcriber(self, model_name: str) -> Transcriber:
        release_accelerator_memory()
        device = resolve_torch_device(self.settings.device)
        self.log(f"Загружаю faster-whisper {model_name} [{device}]")
        log_memory(self.log, "перед загрузкой Whisper")
        try:
            return Transcriber(model_name, device, self.settings.language)
        except RuntimeError as exc:
            if device == "cuda" and is_cuda_oom(exc):
                self.log("WARNING: Whisper не поместился в VRAM при загрузке; использую CPU.")
                release_accelerator_memory()
                return Transcriber(model_name, "cpu", self.settings.language)
            raise

    def _separate_memory_safe(self, source: Path, work_dir: Path, model_name: str) -> Path:
        device = resolve_torch_device(self.settings.device)
        segment = resolve_demucs_segment(self.settings.device, model_name)
        if device == "cuda" and segment:
            self.log(f"Demucs memory-safe: CUDA segment={segment}s.")
        # Demucs' internal --segment bounds model inference VRAM, but not the
        # full-track host result tensor. Bound the actual input-file length too.
        outer_chunk = 600 if model_name == "htdemucs_ft" else 900
        return separate_vocals(
            source,
            work_dir,
            model_name,
            self.cancel,
            device=self.settings.device,
            segment_seconds=segment,
            retry_cpu_on_oom=True,
            log=self.log,
            outer_chunk_seconds=outer_chunk,
            guard_seconds=2.0,
        )

    def _identify_candidates(self, encoder: SpeakerEncoder, chunk_wav: Path,
                             segments: list[DiarizedSegment], absolute_offset: float,
                             chunk_dir: Path) -> list[ClipCandidate]:
        preset = self.settings.preset
        by_speaker: dict[str, list[DiarizedSegment]] = defaultdict(list)
        for seg in segments:
            if seg.duration >= 0.25:
                by_speaker[seg.speaker].append(seg)
        if not by_speaker:
            return []

        similarities: dict[str, float] = {}
        for speaker, speaker_segments in by_speaker.items():
            profile = chunk_dir / f"profile_{_safe_name(speaker)}.wav"
            if not profile.exists():
                seconds = concatenate_pcm_segments(chunk_wav, speaker_segments, profile, max_seconds=60.0)
                if seconds < 1.0:
                    profile.unlink(missing_ok=True)
                    continue
            try:
                self.log(f"    encoding profile {speaker} ({probe_duration(profile):.1f}s) [{encoder.device}]")
                similarities[speaker] = encoder.similarity_file(profile, self.reference_centroid)
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    raise
                self.log(f"    speaker profile {speaker}: ошибка: {exc}")
            except Exception as exc:
                self.log(f"    speaker profile {speaker}: ошибка: {exc}")

        if not similarities:
            return []
        ranked = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
        self.log("    speakers: " + ", ".join(f"{k}={v:.3f}" for k, v in ranked))
        target_speakers, decision = _select_target_speakers(
            similarities,
            threshold=preset.speaker_threshold,
            margin=preset.speaker_margin,
        )
        if not target_speakers:
            self.log(f"    chunk пропущен: {decision}")
            return []
        chosen = next(iter(target_speakers))
        self.log(f"    target: {chosen} ({similarities[chosen]:.3f}); {decision}")

        clean: list[tuple[DiarizedSegment, float]] = []
        for seg in segments:
            if seg.speaker not in target_speakers:
                continue
            overlap = _overlap_with_foreign(seg, segments, target_speakers)
            if overlap > preset.max_overlap_seconds:
                continue
            clean.append((seg, similarities[seg.speaker]))

        merged = _merge_segments(
            clean,
            all_segments=segments,
            target_speakers=target_speakers,
            max_gap=preset.merge_gap_seconds,
            min_duration=preset.min_clip_seconds,
            max_duration=preset.max_clip_seconds,
        )
        return [
            ClipCandidate(
                start=c.start + absolute_offset,
                end=c.end + absolute_offset,
                speaker_similarity=c.speaker_similarity,
            ) for c in merged
        ]

    def _store_rejected(self, temp: Path, source: Path, cand: ClipCandidate, reason: str,
                        fingerprint: str) -> None:
        reason_dir = self.output / "rejected" / _safe_name(reason)
        reason_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(f"{fingerprint}|{cand.start:.3f}|{cand.end:.3f}|{reason}".encode()).hexdigest()[:12]
        dest = reason_dir / f"{_safe_name(source.stem)[:50]}_{tag}.wav"
        shutil.move(str(temp), str(dest))

    def _write_statistics(self) -> None:
        total_count, total_seconds = _dataset_totals(self.output / "metadata_extended.csv")
        data = {
            "run": self.summary.__dict__,
            "dataset_total": {"clips": total_count, "seconds": total_seconds},
        }
        (self.output / "statistics.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = (
            "Voice Dataset Forge — dataset report\n"
            "====================================\n"
            f"Стримов найдено: {self.summary.source_files}\n"
            f"Пропущено уже обработанных: {self.summary.skipped_files}\n"
            f"Ошибок файлов: {self.summary.failed_files}\n"
            f"Добавлено клипов за запуск: {self.summary.accepted_clips}\n"
            f"Добавлено аудио за запуск: {_fmt_seconds(self.summary.accepted_seconds)}\n"
            f"Отбраковано клипов за запуск: {self.summary.rejected_clips}\n"
            f"ИТОГО в датасете: {total_count} клипов / {_fmt_seconds(total_seconds)}\n"
        )
        (self.output / "dataset_report.txt").write_text(report, encoding="utf-8")

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise CancelledError("Обработка отменена пользователем")


class SpeakerlessDiarizerFactory:
    """Tiny indirection that makes CPU fallback easy to unit-test."""

    @staticmethod
    def create(hf_token: str, device: str) -> Diarizer:
        return Diarizer(hf_token, device)


def _select_target_speakers(similarities: dict[str, float], threshold: float, margin: float) -> tuple[set[str], str]:
    """Choose at most one target cluster from a diarization chunk.

    Speaker labels are local to each chunk, so labels such as SPEAKER_00 cannot
    be tracked across the whole stream. We therefore select the best reference
    match per chunk. A runner-up that is too close makes the chunk ambiguous and
    is rejected rather than mixing another voice into the training set.
    """
    if not similarities:
        return set(), "нет speaker embeddings"
    ranked = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]
    if best_score < threshold:
        return set(), f"лучший score {best_score:.3f} ниже порога {threshold:.2f}"
    if len(ranked) > 1:
        runner_name, runner_score = ranked[1]
        gap = best_score - runner_score
        if gap < margin:
            return set(), (
                f"неоднозначно: {best_name}={best_score:.3f}, "
                f"{runner_name}={runner_score:.3f}, отрыв {gap:.3f} < {margin:.2f}"
            )
        return {best_name}, f"отрыв от второго {gap:.3f} ≥ {margin:.2f}"
    return {best_name}, "единственный найденный speaker cluster"


def _overlap_with_foreign(seg: DiarizedSegment, all_segments: list[DiarizedSegment],
                          target_speakers: set[str]) -> float:
    total = 0.0
    for other in all_segments:
        if other.speaker in target_speakers:
            continue
        overlap = max(0.0, min(seg.end, other.end) - max(seg.start, other.start))
        total += overlap
    return total


def _foreign_in_gap(start: float, end: float, all_segments: list[DiarizedSegment],
                    target_speakers: set[str]) -> bool:
    if end <= start:
        return False
    for other in all_segments:
        if other.speaker in target_speakers:
            continue
        if min(end, other.end) - max(start, other.start) > 0.03:
            return True
    return False


def _merge_segments(items: list[tuple[DiarizedSegment, float]], all_segments: list[DiarizedSegment],
                    target_speakers: set[str], max_gap: float, min_duration: float,
                    max_duration: float) -> list[ClipCandidate]:
    if not items:
        return []
    items = sorted(items, key=lambda x: (x[0].start, x[0].end))
    raw: list[ClipCandidate] = []
    cur_start = items[0][0].start
    cur_end = items[0][0].end
    cur_sim = items[0][1]

    def flush(start: float, end: float, sim: float) -> None:
        duration = end - start
        pos = start
        while duration > max_duration:
            raw.append(ClipCandidate(pos, pos + max_duration, sim))
            pos += max_duration
            duration = end - pos
        if duration >= min_duration:
            raw.append(ClipCandidate(pos, end, sim))

    for seg, sim in items[1:]:
        gap = seg.start - cur_end
        combined = seg.end - cur_start
        can_merge = (
            gap <= max_gap and combined <= max_duration and
            not _foreign_in_gap(cur_end, seg.start, all_segments, target_speakers)
        )
        if can_merge:
            cur_end = max(cur_end, seg.end)
            cur_sim = min(cur_sim, sim)
        else:
            flush(cur_start, cur_end, cur_sim)
            cur_start, cur_end, cur_sim = seg.start, seg.end, sim
    flush(cur_start, cur_end, cur_sim)
    return raw


def _dataset_totals(path: Path) -> tuple[int, float]:
    if not path.exists():
        return 0, 0.0
    count = 0
    seconds = 0.0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            count += 1
            try:
                seconds += float(row.get("duration", 0) or 0)
            except ValueError:
                pass
    return count, seconds


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "item"


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
