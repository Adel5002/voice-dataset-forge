from __future__ import annotations

import gc
from pathlib import Path

from .resources import release_accelerator_memory


class Transcriber:
    def __init__(self, model_name: str, device: str = "auto", language: str = "auto"):
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Не установлен faster-whisper.") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.device = device
        self.language = None if language.strip().lower() in {"", "auto"} else language.strip()
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, path: Path) -> str:
        if self.model is None:
            raise RuntimeError("Whisper уже выгружен из памяти")
        segments, _ = self.model.transcribe(
            str(path),
            language=self.language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return " ".join(text.split())

    def close(self) -> None:
        # CTranslate2 exposes unload_model on the inner Whisper object. Use it when
        # available, then drop Python references. This is best-effort: old builds
        # may not expose the method.
        try:
            inner = getattr(self.model, "model", None)
            if inner is not None and hasattr(inner, "unload_model"):
                inner.unload_model()
        except Exception:
            pass
        self.model = None
        gc.collect()
        release_accelerator_memory()
