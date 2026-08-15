from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

QualityName = Literal["Быстро", "Сбалансированно", "Максимум"]


@dataclass(frozen=True)
class QualityPreset:
    name: QualityName
    use_separation: bool
    demucs_model: str
    speaker_threshold: float
    speaker_margin: float
    segment_verify: bool
    segment_threshold: float
    min_clip_seconds: float
    max_clip_seconds: float
    merge_gap_seconds: float
    max_overlap_seconds: float
    min_rms_dbfs: float
    max_clipping_ratio: float
    max_silence_ratio: float
    whisper_model: str
    diarization_window_seconds: int = 1800


QUALITY_PRESETS: dict[QualityName, QualityPreset] = {
    "Быстро": QualityPreset(
        name="Быстро",
        use_separation=False,
        demucs_model="htdemucs",
        speaker_threshold=0.34,
        speaker_margin=0.06,
        segment_verify=False,
        segment_threshold=0.30,
        min_clip_seconds=1.0,
        max_clip_seconds=18.0,
        merge_gap_seconds=0.45,
        max_overlap_seconds=0.20,
        min_rms_dbfs=-46.0,
        max_clipping_ratio=0.015,
        max_silence_ratio=0.78,
        whisper_model="small",
        diarization_window_seconds=300,
    ),
    "Сбалансированно": QualityPreset(
        name="Сбалансированно",
        use_separation=True,
        demucs_model="htdemucs",
        speaker_threshold=0.42,
        speaker_margin=0.08,
        segment_verify=False,
        segment_threshold=0.38,
        min_clip_seconds=1.2,
        max_clip_seconds=14.0,
        merge_gap_seconds=0.35,
        max_overlap_seconds=0.10,
        min_rms_dbfs=-42.0,
        max_clipping_ratio=0.006,
        max_silence_ratio=0.68,
        whisper_model="turbo",
        diarization_window_seconds=600,
    ),
    "Максимум": QualityPreset(
        name="Максимум",
        use_separation=True,
        demucs_model="htdemucs_ft",
        speaker_threshold=0.50,
        speaker_margin=0.10,
        segment_verify=True,
        segment_threshold=0.44,
        min_clip_seconds=1.5,
        max_clip_seconds=12.0,
        merge_gap_seconds=0.25,
        max_overlap_seconds=0.05,
        min_rms_dbfs=-39.0,
        max_clipping_ratio=0.0025,
        max_silence_ratio=0.58,
        whisper_model="large-v3",
        diarization_window_seconds=600,
    ),
}


@dataclass
class AppSettings:
    streams_dir: Path
    references_dir: Path
    output_dir: Path
    quality: QualityName = "Максимум"
    hf_token: str = ""
    device: str = "auto"
    transcribe: bool = True
    language: str = "auto"
    keep_rejected: bool = True
    output_sample_rate: int = 48000

    @property
    def preset(self) -> QualityPreset:
        return QUALITY_PRESETS[self.quality]

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("hf_token", None)
        for key in ("streams_dir", "references_dir", "output_dir"):
            data[key] = str(data[key])
        data["preset"] = asdict(self.preset)
        return data


SUPPORTED_MEDIA = {
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".mka",
}
