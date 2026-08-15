from __future__ import annotations

import importlib.util
import math
import shutil
import threading
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .media import probe_duration, run_command
from .resources import is_cuda_oom, resolve_torch_device

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class SeparationChunk:
    index: int
    core_start: float
    core_duration: float
    input_start: float
    input_duration: float
    trim_start: float


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def _demucs_command(source: Path, out_root: Path, model_name: str,
                    device: str | None = None, segment_seconds: int | None = None) -> list[str]:
    cmd = [
        sys.executable, "-m", "demucs.separate", "--two-stems", "vocals",
        "--float32", "-n", model_name, "-o", str(out_root),
    ]
    if device:
        cmd += ["-d", device]
    if segment_seconds:
        cmd += ["--segment", str(int(segment_seconds))]
    cmd.append(str(source))
    return cmd


def plan_separation_chunks(duration: float, chunk_seconds: float,
                           guard_seconds: float = 2.0) -> list[SeparationChunk]:
    """Plan externally bounded Demucs jobs while preserving context at boundaries.

    Demucs' own ``--segment`` limits model inference windows, but the upstream
    implementation still materializes a full-track multi-stem output tensor on
    the host. For multi-hour streams that tensor alone can consume many GiB.
    We therefore give Demucs short *files*, with a small context guard on both
    sides, and trim each separated stem back to its exact core interval.
    """
    duration = max(0.0, float(duration))
    chunk_seconds = max(30.0, float(chunk_seconds))
    guard_seconds = max(0.0, float(guard_seconds))
    if duration <= 0.0:
        return []

    total = max(1, math.ceil(duration / chunk_seconds))
    chunks: list[SeparationChunk] = []
    for idx in range(total):
        core_start = idx * chunk_seconds
        core_end = min(duration, core_start + chunk_seconds)
        if core_end - core_start <= 0.01:
            break
        input_start = max(0.0, core_start - guard_seconds)
        input_end = min(duration, core_end + guard_seconds)
        chunks.append(SeparationChunk(
            index=idx,
            core_start=core_start,
            core_duration=core_end - core_start,
            input_start=input_start,
            input_duration=input_end - input_start,
            trim_start=core_start - input_start,
        ))
    return chunks


def _single_pass_vocals(source: Path, work_dir: Path, model_name: str,
                        cancel: threading.Event | None = None, *,
                        device: str = "auto", segment_seconds: int | None = None,
                        retry_cpu_on_oom: bool = True, log: LogFn | None = None) -> Path:
    log = log or (lambda _m: None)
    out_root = work_dir / "demucs"
    out_root.mkdir(parents=True, exist_ok=True)
    resolved = resolve_torch_device(device)

    try:
        run_command(
            _demucs_command(source, out_root, model_name, resolved, segment_seconds),
            cancel,
        )
    except RuntimeError as exc:
        if resolved == "cuda" and retry_cpu_on_oom and is_cuda_oom(exc):
            log("WARNING: Demucs не поместился в VRAM; повторяю этот chunk на CPU (качество не меняется, только скорость).")
            partial = out_root / model_name / source.stem
            if partial.exists():
                shutil.rmtree(partial, ignore_errors=True)
            run_command(
                _demucs_command(source, out_root, model_name, "cpu", None),
                cancel,
            )
        else:
            raise

    expected = out_root / model_name / source.stem / "vocals.wav"
    if expected.exists():
        return expected
    matches = list(out_root.glob(f"*/{source.stem}/vocals.wav"))
    if not matches:
        raise RuntimeError("Demucs завершился без ожидаемого vocals.wav")
    return matches[0]


def _cut_stereo_pcm(source: Path, dest: Path, start: float, duration: float,
                    cancel: threading.Event | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.4f}", "-t", f"{max(0.01, duration):.4f}", "-i", str(source),
        "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(dest),
    ], cancel)


def _trim_stem(source: Path, dest: Path, start: float, duration: float,
               cancel: threading.Event | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.4f}", "-t", f"{max(0.01, duration):.4f}", "-i", str(source),
        "-vn", "-c:a", "pcm_f32le", str(dest),
    ], cancel)


def _ffconcat_quote(path: Path) -> str:
    # ffconcat accepts forward slashes on Windows. Single quotes are escaped
    # using the same close/escape/reopen form documented by ffmpeg utilities.
    text = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{text}'"


def _concat_stems(parts: list[Path], dest: Path,
                  cancel: threading.Event | None = None) -> None:
    if not parts:
        raise RuntimeError("Нет частей vocals для объединения")
    if len(parts) == 1:
        shutil.copy2(parts[0], dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = dest.with_suffix(".ffconcat.txt")
    manifest.write_text("ffconcat version 1.0\n" + "\n".join(_ffconcat_quote(p) for p in parts) + "\n",
                        encoding="utf-8")
    try:
        # Re-encode to float PCM rather than stream-copying WAV chunks. This is
        # deterministic across ffmpeg builds and avoids container timestamp/header
        # quirks while keeping the separated signal unquantized.
        run_command([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(manifest),
            "-vn", "-c:a", "pcm_f32le", str(dest),
        ], cancel)
    finally:
        manifest.unlink(missing_ok=True)


def separate_vocals(source: Path, work_dir: Path, model_name: str,
                    cancel: threading.Event | None = None, *,
                    device: str = "auto", segment_seconds: int | None = None,
                    retry_cpu_on_oom: bool = True, log: LogFn | None = None,
                    outer_chunk_seconds: int | None = 900,
                    guard_seconds: float = 2.0) -> Path:
    """Separate vocals with bounded host memory for long recordings.

    ``segment_seconds`` controls Demucs' *internal* inference split. For long
    streams we additionally split the input file into bounded outer jobs because
    upstream Demucs keeps a full-track multi-source result tensor in host RAM.
    Each outer job includes a small context guard, then its vocal stem is trimmed
    back to the exact core interval and all cores are concatenated.
    """
    if not demucs_available():
        raise RuntimeError(
            "Для выбранного режима нужен Demucs. Установи зависимости из requirements.txt "
            "или выбери режим 'Быстро'."
        )
    log = log or (lambda _m: None)

    duration = probe_duration(source)
    if not outer_chunk_seconds or duration <= float(outer_chunk_seconds) + 0.5:
        return _single_pass_vocals(
            source, work_dir, model_name, cancel,
            device=device, segment_seconds=segment_seconds,
            retry_cpu_on_oom=retry_cpu_on_oom, log=log,
        )

    final = work_dir / f"vocals_chunked_{model_name}.wav"
    if final.exists():
        try:
            if abs(probe_duration(final) - duration) <= 1.0:
                log(f"Demucs outer-chunk cache: использую готовый vocals ({duration / 60:.1f} мин).")
                return final
            log("SELF-HEAL: merged Demucs cache имеет неверную длительность; пересобираю.")
        except Exception as exc:
            log(f"SELF-HEAL: merged Demucs cache не читается ({exc}); пересобираю.")
        final.unlink(missing_ok=True)

    chunks = plan_separation_chunks(duration, float(outer_chunk_seconds), guard_seconds)
    chunk_root = work_dir / f"demucs_outer_{model_name}_{int(outer_chunk_seconds)}s"
    chunk_root.mkdir(parents=True, exist_ok=True)
    log(
        f"Demucs long-stream mode: {duration / 60:.1f} мин → {len(chunks)} outer chunks "
        f"по ~{int(outer_chunk_seconds // 60)} мин; это ограничивает RAM независимо от длины стрима."
    )

    core_parts: list[Path] = []
    for pos, plan in enumerate(chunks, start=1):
        if cancel and cancel.is_set():
            from .media import CancelledError
            raise CancelledError("Обработка отменена пользователем")

        item_dir = chunk_root / f"{plan.index:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        input_wav = item_dir / "input_48k.wav"
        core_wav = item_dir / "vocals_core.wav"

        if core_wav.exists():
            try:
                if abs(probe_duration(core_wav) - plan.core_duration) <= 0.75:
                    log(f"  Demucs outer {pos}/{len(chunks)}: кэш ({plan.core_duration / 60:.1f} мин)")
                    core_parts.append(core_wav)
                    continue
                log(f"  SELF-HEAL: Demucs outer {pos}/{len(chunks)} cache неверной длительности; пересчитываю.")
            except Exception as exc:
                log(f"  SELF-HEAL: Demucs outer {pos}/{len(chunks)} cache не читается ({exc}); пересчитываю.")
            core_wav.unlink(missing_ok=True)

        log(
            f"  Demucs outer {pos}/{len(chunks)}: "
            f"{plan.core_start / 60:.1f}–{(plan.core_start + plan.core_duration) / 60:.1f} мин"
        )
        if not input_wav.exists():
            _cut_stereo_pcm(source, input_wav, plan.input_start, plan.input_duration, cancel)

        demucs_work = item_dir / "run"
        vocals = _single_pass_vocals(
            input_wav, demucs_work, model_name, cancel,
            device=device, segment_seconds=segment_seconds,
            retry_cpu_on_oom=retry_cpu_on_oom, log=log,
        )
        _trim_stem(vocals, core_wav, plan.trim_start, plan.core_duration, cancel)
        core_parts.append(core_wav)

        # The Demucs run output and source slice duplicate data we no longer need.
        # Keep only the trimmed core for resume until the final concat succeeds.
        shutil.rmtree(demucs_work, ignore_errors=True)
        input_wav.unlink(missing_ok=True)

    _concat_stems(core_parts, final, cancel)
    final_duration = probe_duration(final)
    if abs(final_duration - duration) > max(1.0, len(chunks) * 0.08):
        raise RuntimeError(
            f"После chunked Demucs длительность vocals отличается от исходника: "
            f"{final_duration:.2f}s vs {duration:.2f}s"
        )
    # Once the merged stem is validated, the resumable per-chunk cache is only
    # duplicate disk usage. A failed run keeps it; a successful run cleans it.
    shutil.rmtree(chunk_root, ignore_errors=True)
    return final
