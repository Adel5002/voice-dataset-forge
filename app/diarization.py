from __future__ import annotations

import gc
import json
from pathlib import Path

from .models import DiarizedSegment
from .audioio import load_audio_mapping
from .resources import release_accelerator_memory


class Diarizer:
    def __init__(self, hf_token: str, device: str = "auto"):
        if not hf_token:
            raise RuntimeError(
                "Нужен Hugging Face token для pyannote community-1. "
                "Прими условия модели pyannote/speaker-diarization-community-1 и вставь token в приложение."
            )
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("Не установлены pyannote.audio/PyTorch. Выполни установку зависимостей.") from exc

        self.torch = torch
        self.device = self._resolve_device(device)
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        self.pipeline.to(torch.device(self.device))

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            if requested == "cuda" and not self.torch.cuda.is_available():
                return "cpu"
            return requested
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def run(self, wav_path: Path) -> list[DiarizedSegment]:
        if self.pipeline is None:
            raise RuntimeError("Diarizer уже выгружен из памяти")
        # Feed an in-memory waveform so pyannote does not invoke TorchCodec.
        audio = load_audio_mapping(wav_path, self.torch)
        output = self.pipeline(audio)
        # Drop the large input mapping before parsing annotation.
        del audio
        annotation = getattr(output, "speaker_diarization", output)
        segments: list[DiarizedSegment] = []

        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(DiarizedSegment(float(turn.start), float(turn.end), str(speaker)))
        else:
            for item in annotation:
                if len(item) == 2:
                    turn, speaker = item
                elif len(item) == 3:
                    turn, _, speaker = item
                else:
                    continue
                segments.append(DiarizedSegment(float(turn.start), float(turn.end), str(speaker)))
        del output
        gc.collect()
        return sorted(segments, key=lambda s: (s.start, s.end, s.speaker))

    def close(self) -> None:
        self.pipeline = None
        release_accelerator_memory()

    @staticmethod
    def save(path: Path, segments: list[DiarizedSegment]) -> None:
        path.write_text(json.dumps([s.__dict__ for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> list[DiarizedSegment]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [DiarizedSegment(**item) for item in data]
