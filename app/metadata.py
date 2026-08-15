from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from .models import StagedClip


def repair_dataset_metadata(output_dir: Path) -> dict[str, int]:
    """Drop metadata rows whose referenced WAV no longer exists.

    Manual cleanup in Explorer can easily leave metadata/state behind while the
    ``audio`` directory has been emptied. Treat those rows as stale instead of
    letting DatasetWriter believe the missing clips are valid duplicates.
    Returns unique referenced-file counters for pipeline reconciliation.
    """
    output_dir = Path(output_dir)
    removed_files: set[str] = set()
    kept_files: set[str] = set()

    def exists_rel(rel: str) -> bool:
        rel = (rel or "").strip()
        return bool(rel) and (output_dir / rel).is_file()

    metadata_path = output_dir / "metadata.csv"
    if metadata_path.exists():
        rows: list[list[str]] = []
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="|"):
                if not row:
                    continue
                rel = row[0].strip()
                if exists_rel(rel):
                    rows.append(row)
                    kept_files.add(rel)
                else:
                    removed_files.add(rel)
        tmp = metadata_path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL).writerows(rows)
        os.replace(tmp, metadata_path)

    extended_path = output_dir / "metadata_extended.csv"
    if extended_path.exists():
        rows: list[list[str]] = []
        header: list[str] | None = None
        with extended_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")
            for idx, row in enumerate(reader):
                if idx == 0 and row and row[0] == "file":
                    header = row
                    continue
                if not row:
                    continue
                rel = row[0].strip()
                if exists_rel(rel):
                    rows.append(row)
                    kept_files.add(rel)
                else:
                    removed_files.add(rel)
        tmp = extended_path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp, extended_path)

    jsonl_path = output_dir / "metadata.jsonl"
    if jsonl_path.exists():
        lines: list[str] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                rel = str(obj.get("file", "")).strip()
                if exists_rel(rel):
                    kept_files.add(rel)
                    lines.append(json.dumps(obj, ensure_ascii=False))
                elif rel:
                    removed_files.add(rel)
            except Exception:
                # Broken cache-like metadata should never block a new dataset run.
                continue
        tmp = jsonl_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, jsonl_path)

    return {"removed": len(removed_files), "kept": len(kept_files)}


class DatasetWriter:
    """Append-only dataset writer with idempotent clip commits.

    Incremental mode commits accepted audio as soon as an internal stream chunk has been
    filtered. Existing source/start/end rows are treated as duplicates so a
    restart cannot create a second copy of the same clip.
    """

    def __init__(self, output_dir: Path, next_index: int):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.rejected_dir = self.output_dir / "rejected"
        self.metadata_path = self.output_dir / "metadata.csv"
        self.extended_path = self.output_dir / "metadata_extended.csv"
        self.jsonl_path = self.output_dir / "metadata.jsonl"
        self.next_index = next_index
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        repair_dataset_metadata(self.output_dir)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self._existing = self._load_existing()

    @staticmethod
    def _key(source: str, start: float, end: float) -> tuple[str, int, int]:
        # Millisecond precision matches what metadata_extended.csv stores.
        return (str(source), int(round(float(start) * 1000)), int(round(float(end) * 1000)))

    def _load_existing(self) -> dict[tuple[str, int, int], tuple[str, str, float]]:
        out: dict[tuple[str, int, int], tuple[str, str, float]] = {}
        if not self.extended_path.exists() or self.extended_path.stat().st_size == 0:
            return out
        try:
            with self.extended_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="|")
                for row in reader:
                    try:
                        key = self._key(row.get("source", ""), float(row.get("start", 0)), float(row.get("end", 0)))
                        out[key] = (
                            row.get("file", ""),
                            row.get("text", "") or "",
                            float(row.get("duration", 0) or 0),
                        )
                    except Exception:
                        continue
        except Exception:
            return {}
        return out

    def commit(self, clips: list[StagedClip]) -> tuple[int, float]:
        new_count, new_seconds, _ = self.commit_detailed(clips)
        return new_count, new_seconds

    def commit_detailed(self, clips: list[StagedClip]) -> tuple[int, float, list[StagedClip]]:
        """Commit clips immediately.

        Returns (new_count, new_seconds, usable_clips). ``usable_clips`` contains
        both newly committed clips and already-existing duplicates, with ``path``
        rewritten to the final audio path. This lets transcription resume after a
        crash without duplicating audio.
        """
        new_count = 0
        new_seconds = 0.0
        usable: list[StagedClip] = []
        new_extended = not self.extended_path.exists() or self.extended_path.stat().st_size == 0
        with self.metadata_path.open("a", encoding="utf-8", newline="") as meta_f,              self.extended_path.open("a", encoding="utf-8", newline="") as ext_f,              self.jsonl_path.open("a", encoding="utf-8") as jsonl_f:
            meta = csv.writer(meta_f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            ext = csv.writer(ext_f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            if new_extended:
                ext.writerow([
                    "file", "text", "source", "start", "end", "duration",
                    "speaker_similarity", "rms_dbfs", "peak_dbfs",
                    "clipping_ratio", "silence_ratio",
                ])
            for clip in clips:
                key = self._key(clip.source, clip.start, clip.end)
                existing = self._existing.get(key)
                if existing:
                    rel, text, _duration = existing
                    clip.path.unlink(missing_ok=True)
                    clip.path = self.output_dir / rel
                    clip.text = text
                    if clip.path.exists():
                        usable.append(clip)
                    continue

                safe_text = " ".join(clip.text.replace("|", " ").split())
                name = f"{self.next_index:06d}.wav"
                dest = self.audio_dir / name
                shutil.move(str(clip.path), str(dest))
                clip.path = dest
                rel = f"audio/{name}"
                meta.writerow([rel, safe_text])
                ext.writerow([
                    rel, safe_text, clip.source, f"{clip.start:.3f}", f"{clip.end:.3f}",
                    f"{clip.metrics.duration:.3f}", f"{clip.speaker_similarity:.5f}",
                    f"{clip.metrics.rms_dbfs:.3f}", f"{clip.metrics.peak_dbfs:.3f}",
                    f"{clip.metrics.clipping_ratio:.7f}", f"{clip.metrics.silence_ratio:.7f}",
                ])
                record = {
                    "file": rel,
                    "text": safe_text,
                    "source": clip.source,
                    "start": clip.start,
                    "end": clip.end,
                    "duration": clip.metrics.duration,
                    "speaker_similarity": clip.speaker_similarity,
                    "rms_dbfs": clip.metrics.rms_dbfs,
                    "peak_dbfs": clip.metrics.peak_dbfs,
                    "clipping_ratio": clip.metrics.clipping_ratio,
                    "silence_ratio": clip.metrics.silence_ratio,
                }
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._existing[key] = (rel, safe_text, clip.metrics.duration)
                self.next_index += 1
                new_count += 1
                new_seconds += clip.metrics.duration
                usable.append(clip)
        return new_count, new_seconds, usable

    def update_texts(self, clips: list[StagedClip]) -> None:
        """Patch transcripts for already-committed clips atomically."""
        updates: dict[str, str] = {}
        for clip in clips:
            try:
                rel = clip.path.relative_to(self.output_dir).as_posix()
            except ValueError:
                continue
            updates[rel] = " ".join((clip.text or "").replace("|", " ").split())
        if not updates:
            return

        if self.metadata_path.exists():
            rows: list[list[str]] = []
            with self.metadata_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f, delimiter="|"):
                    if row and row[0] in updates:
                        if len(row) < 2:
                            row.append(updates[row[0]])
                        else:
                            row[1] = updates[row[0]]
                    rows.append(row)
            tmp = self.metadata_path.with_suffix(".csv.tmp")
            with tmp.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL).writerows(rows)
            os.replace(tmp, self.metadata_path)

        if self.extended_path.exists():
            rows = []
            with self.extended_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f, delimiter="|"):
                    if row and row[0] in updates and row[0] != "file":
                        if len(row) < 2:
                            row.append(updates[row[0]])
                        else:
                            row[1] = updates[row[0]]
                    rows.append(row)
            tmp = self.extended_path.with_suffix(".csv.tmp")
            with tmp.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL).writerows(rows)
            os.replace(tmp, self.extended_path)

        if self.jsonl_path.exists():
            lines: list[str] = []
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("file") in updates:
                        obj["text"] = updates[obj["file"]]
                    lines.append(json.dumps(obj, ensure_ascii=False))
                except Exception:
                    lines.append(line)
            tmp = self.jsonl_path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            os.replace(tmp, self.jsonl_path)

        # Refresh in-memory duplicate map with the new text.
        for key, (rel, _text, duration) in list(self._existing.items()):
            if rel in updates:
                self._existing[key] = (rel, updates[rel], duration)
