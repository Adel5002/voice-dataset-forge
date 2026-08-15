from __future__ import annotations

from pathlib import Path


def load_audio_mapping(path: Path, torch_module=None) -> dict:
    """Load WAV/audio with libsndfile and return a pyannote in-memory AudioFile mapping.

    Passing {"waveform": tensor, "sample_rate": int} to pyannote bypasses its
    TorchCodec file decoder. This is particularly useful on Windows where
    TorchCodec otherwise needs a matching wheel plus a shared-library FFmpeg build.
    The Dataset Forge pipeline already normalizes analysis audio to WAV, so there
    is no reason to decode the same file a second time through TorchCodec.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Не установлен soundfile. Переустанови зависимости приложения.") from exc

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("Не установлен PyTorch.") from exc

    # always_2d -> [time, channels], pyannote expects [channels, time]
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0:
        raise RuntimeError(f"Пустой аудиофайл: {path}")
    waveform = torch_module.from_numpy(data.T.copy())
    return {"waveform": waveform, "sample_rate": int(sample_rate)}
