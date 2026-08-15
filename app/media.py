from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import DiarizedSegment, QualityMetrics


class CancelledError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg/ffprobe не найдены в PATH. Установи FFmpeg и перезапусти приложение.")


def run_command(args: list[str], cancel: threading.Event | None = None) -> None:
    # Do not leave stdout/stderr in PIPE without draining them. Tools such as Demucs
    # can emit enough progress output to fill a Windows pipe buffer and deadlock.
    # A temporary file gives the child an effectively unbounded sink while still
    # letting us include the tail of the output in an error message.
    with tempfile.TemporaryFile(mode="w+b") as output:
        proc = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if cancel and cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise CancelledError("Обработка отменена пользователем")
            time.sleep(0.20)

        output.flush()
        output.seek(0, 2)
        size = output.tell()
        output.seek(max(0, size - 12000))
        tail = output.read().decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"Команда завершилась с кодом {proc.returncode}:\n{tail[-4000:]}")


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_highres_audio(source: Path, dest: Path, cancel: threading.Event | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "2", "-ar", "48000",
        "-c:a", "pcm_s16le", str(dest),
    ], cancel)


def convert_analysis_wav(source: Path, dest: Path, cancel: threading.Event | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(dest),
    ], cancel)


def cut_wav(source: Path, dest: Path, start: float, end: float, sample_rate: int = 48000,
            cancel: threading.Event | None = None) -> None:
    duration = max(0.01, end - start)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.4f}", "-t", f"{duration:.4f}", "-i", str(source),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dest),
    ], cancel)


def cut_analysis_chunk(source: Path, dest: Path, start: float, duration: float,
                       cancel: threading.Event | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest),
    ], cancel)


def concatenate_pcm_segments(source: Path, segments: Iterable[DiarizedSegment], dest: Path,
                             max_seconds: float = 60.0) -> float:
    """Concatenate ranges from a mono PCM16 WAV without decoding the full file to RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0.0
    with wave.open(str(source), "rb") as src:
        if src.getsampwidth() != 2:
            raise RuntimeError("Для speaker profile ожидается PCM16 WAV")
        rate = src.getframerate()
        channels = src.getnchannels()
        params = src.getparams()
        with wave.open(str(dest), "wb") as out:
            out.setparams(params)
            for seg in segments:
                if written >= max_seconds:
                    break
                remaining = max_seconds - written
                dur = min(seg.duration, remaining)
                if dur <= 0:
                    continue
                start_frame = max(0, int(seg.start * rate))
                frame_count = max(1, int(dur * rate))
                src.setpos(min(start_frame, max(0, src.getnframes() - 1)))
                data = src.readframes(frame_count)
                if channels > 0 and data:
                    out.writeframes(data)
                    written += len(data) / (2 * channels * rate)
    return written


def analyze_pcm16_wav(path: Path, min_rms_dbfs: float, max_clipping_ratio: float,
                      max_silence_ratio: float, min_seconds: float, max_seconds: float) -> QualityMetrics:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        if width != 2:
            return QualityMetrics(0, -120, -120, 1, 1, False, "unsupported_sample_width")
        raw = wf.readframes(frames)

    # NumPy is deliberately used here instead of a Python per-sample loop.
    # Besides being much faster on long batches, this avoids a Windows/Python
    # runtime edge case observed in v0.7 where the scalar accumulator path
    # crashed with ``unsupported operand type(s) for *: 'type' and 'int'``.
    # Convert to a wide integer before abs/squaring so -32768 and x*x cannot
    # overflow int16.
    samples = np.frombuffer(raw, dtype=np.dtype("<i2"))
    if samples.size == 0:
        return QualityMetrics(0, -120, -120, 1, 1, False, "empty")

    duration = frames / float(rate)
    values = samples.astype(np.int32, copy=False)
    abs_values = np.abs(values)
    silence_amp = int(32768 * (10 ** (-50 / 20)))
    clip_amp = int(32767 * 0.995)

    # float64 here is intentional: it keeps the RMS reduction stable and
    # avoids integer overflow even for clipped material.
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    peak = int(abs_values.max(initial=0))
    clipping_ratio = float(np.mean(abs_values >= clip_amp))
    silence_ratio = float(np.mean(abs_values <= silence_amp))
    rms_dbfs = 20 * math.log10(max(rms, 1.0) / 32768.0)
    peak_dbfs = 20 * math.log10(max(peak, 1) / 32768.0)

    reason = ""
    if duration < min_seconds:
        reason = "too_short"
    elif duration > max_seconds + 0.5:
        reason = "too_long"
    elif rms_dbfs < min_rms_dbfs:
        reason = "too_quiet"
    elif clipping_ratio > max_clipping_ratio:
        reason = "clipping"
    elif silence_ratio > max_silence_ratio:
        reason = "too_much_silence"

    return QualityMetrics(
        duration=duration,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        accepted=not reason,
        reason=reason,
    )
