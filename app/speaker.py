from __future__ import annotations

from pathlib import Path

from .audioio import load_audio_mapping
from .resources import release_accelerator_memory


class SpeakerEncoder:
    """Speaker embeddings using pyannote's WeSpeaker VoxCeleb ResNet34 model."""

    def __init__(self, hf_token: str, device: str = "auto"):
        try:
            import numpy as np
            import torch
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise RuntimeError("Не установлены pyannote.audio/PyTorch/numpy.") from exc

        self.np = np
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            token=hf_token or None,
        )
        self.model.to(torch.device(device))
        self.model.eval()
        self.inference = Inference(self.model, window="whole")

    def encode_file(self, path: Path):
        if self.inference is None:
            raise RuntimeError("Speaker encoder уже выгружен из памяти")
        # In-memory input bypasses TorchCodec file decoding on Windows.
        audio = load_audio_mapping(path, self.torch)
        vec = self.inference(audio)
        del audio
        return self._normalize(self.np.asarray(vec, dtype="float32").reshape(-1))

    def centroid(self, paths: list[Path]):
        if not paths:
            raise RuntimeError("В папке референсов не найдено ни одного подходящего файла.")
        vectors = [self.encode_file(p) for p in paths]
        return self._normalize(self.np.mean(self.np.stack(vectors, axis=0), axis=0))

    def similarity(self, embedding, centroid) -> float:
        a = self._normalize(embedding)
        b = self._normalize(centroid)
        return float(self.np.dot(a, b))

    def similarity_file(self, path: Path, centroid) -> float:
        return self.similarity(self.encode_file(path), centroid)

    def close(self) -> None:
        self.inference = None
        self.model = None
        release_accelerator_memory()

    def _normalize(self, vec):
        norm = float(self.np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm
