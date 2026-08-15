from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class ProjectState:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / ".vdf_state.json"
        self.data = {"version": 1, "files": {}}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {"version": 1, "files": {}}

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def fingerprint(path: Path) -> str:
        """Fast content fingerprint: stable even if a stream file is moved/renamed."""
        stat = path.stat()
        h = hashlib.sha1()
        h.update(str(stat.st_size).encode())
        with path.open("rb") as f:
            head = f.read(1024 * 1024)
            h.update(head)
            if stat.st_size > 1024 * 1024:
                f.seek(max(0, stat.st_size - 1024 * 1024))
                h.update(f.read(1024 * 1024))
        return h.hexdigest()

    def is_done(self, fingerprint: str) -> bool:
        return self.data.get("files", {}).get(fingerprint, {}).get("status") == "done"

    def has_done_files(self) -> bool:
        return any(
            item.get("status") == "done"
            for item in self.data.get("files", {}).values()
            if isinstance(item, dict)
        )

    def reset_files(self) -> None:
        """Forget per-source completion state without touching cached media.

        This is used when the user manually clears the final dataset files while
        leaving ``.vdf_state.json``/``.cache`` behind. Heavy cache artifacts may
        still be reusable, but sources must no longer be treated as completed.
        """
        self.data = {"version": 1, "files": {}}
        self.save()

    def mark_done(self, fingerprint: str, path: Path, clips: int, seconds: float) -> None:
        self.data.setdefault("files", {})[fingerprint] = {
            "path": str(path.resolve()),
            "status": "done",
            "clips": clips,
            "seconds": seconds,
        }
        self.save()

    def mark_failed(self, fingerprint: str, path: Path, error: str) -> None:
        self.data.setdefault("files", {})[fingerprint] = {
            "path": str(path.resolve()),
            "status": "failed",
            "error": error,
        }
        self.save()

    def next_audio_index(self) -> int:
        audio_dir = self.output_dir / "audio"
        if not audio_dir.exists():
            return 1
        maximum = 0
        for path in audio_dir.glob("*.wav"):
            try:
                maximum = max(maximum, int(path.stem))
            except ValueError:
                pass
        return maximum + 1
