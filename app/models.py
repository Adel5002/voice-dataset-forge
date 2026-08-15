from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiarizedSegment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class ClipCandidate:
    start: float
    end: float
    speaker_similarity: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class QualityMetrics:
    duration: float
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    accepted: bool
    reason: str = ""


@dataclass
class StagedClip:
    path: Path
    text: str
    source: str
    start: float
    end: float
    speaker_similarity: float
    metrics: QualityMetrics


@dataclass
class PipelineSummary:
    source_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    source_seconds: float = 0.0
    candidate_seconds: float = 0.0
    accepted_seconds: float = 0.0
    rejected_seconds: float = 0.0
    accepted_clips: int = 0
    rejected_clips: int = 0
