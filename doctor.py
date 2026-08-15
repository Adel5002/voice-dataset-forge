from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import wave
from pathlib import Path


def check(label: str, ok: bool, detail: str = ""):
    mark = "OK" if ok else "FAIL"
    print(f"[{mark:4}] {label}{': ' + detail if detail else ''}")
    return ok


def main() -> int:
    good = True
    version_ok = (3, 10) <= sys.version_info[:2] < (3, 13)
    good &= check("Python", version_ok, sys.version.split()[0] + " (recommended: 3.11)")
    good &= check("ffmpeg", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "not found")
    good &= check("ffprobe", bool(shutil.which("ffprobe")), shutil.which("ffprobe") or "not found")

    for module in ["PySide6", "torch", "torchaudio", "pyannote.audio", "faster_whisper", "demucs", "soundfile"]:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "installed")
            check(module, True, str(version))
        except Exception as exc:
            check(module, False, str(exc))
            good = False

    try:
        import psutil
        vm = psutil.virtual_memory()
        check("RAM", True, f"{vm.total / (1024**3):.1f} GB total; {vm.available / (1024**3):.1f} GB free")
    except Exception:
        check("RAM telemetry", True, "psutil not installed; optional")

    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            props = torch.cuda.get_device_properties(0)
            detail = f"{torch.cuda.get_device_name(0)}; {props.total_memory/(1024**3):.1f} GB VRAM"
        else:
            detail = "CUDA unavailable (CPU fallback is possible but slow)"
        good &= check("CUDA", cuda, detail)
    except Exception:
        pass

    try:
        from app.resources import memory_snapshot, resolve_demucs_segment, resolve_diarization_window
        check("Memory-safe runtime", True, memory_snapshot())
        print(f"       Suggested diarization chunk: {resolve_diarization_window(600, 'auto')} s")
        print(f"       Suggested Demucs segment: {resolve_demucs_segment('auto', 'htdemucs') or 'default'}")
    except Exception as exc:
        check("Memory-safe runtime", False, str(exc))
        good = False

    # Verify the exact decoder path used by Dataset Forge. This does not use TorchCodec.
    try:
        from app.audioio import load_audio_mapping
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "doctor.wav"
            with wave.open(str(wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 1600)
            audio = load_audio_mapping(wav)
            ok = tuple(audio["waveform"].shape) == (1, 1600) and audio["sample_rate"] == 16000
            good &= check("Dataset Forge audio decoder", ok, "soundfile -> Tensor (TorchCodec bypassed)")
    except Exception as exc:
        good &= check("Dataset Forge audio decoder", False, str(exc))

    try:
        tc = importlib.import_module("torchcodec")
        check("TorchCodec (optional)", True, str(getattr(tc, "__version__", "installed")))
    except Exception:
        check("TorchCodec (optional)", True, "not usable; intentionally bypassed")

    print("\nResult:", "environment looks ready" if good else "fix FAIL items before a full run")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
