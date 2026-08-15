# Changelog

## 1.1.0

- Added self-healing project state after manual output cleanup.
- Missing WAV references are pruned from metadata automatically instead of being treated as valid duplicates.
- Missing or corrupt voice-centroid, diarization, candidate, and long-stream Demucs cache artifacts are rebuilt automatically.
- WeSpeaker is now cached for the whole desktop application session, avoiding repeated `Model.from_pretrained(...)` calls between consecutive runs/mode switches.
- Added explicit session-model cleanup when the GUI closes.
- Added GitHub-friendly `.gitignore` so datasets, caches, media, local settings, and secrets are not committed accidentally.
- Updated package/application version metadata to 1.1.0.
