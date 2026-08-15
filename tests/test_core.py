from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

from app.media import analyze_pcm16_wav
from app.models import DiarizedSegment
from app.pipeline import _merge_segments
from app.state import ProjectState


def test_merge_target_segments():
    items = [
        (DiarizedSegment(0.0, 2.0, "me"), 0.9),
        (DiarizedSegment(2.1, 4.0, "me"), 0.8),
    ]
    out = _merge_segments(
        items,
        all_segments=[x[0] for x in items],
        target_speakers={"me"},
        max_gap=0.3,
        min_duration=1.0,
        max_duration=10.0,
    )
    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 4.0
    assert out[0].speaker_similarity == 0.8


def test_does_not_merge_across_foreign_speech():
    a = DiarizedSegment(0.0, 2.0, "me")
    foreign = DiarizedSegment(2.02, 2.18, "other")
    b = DiarizedSegment(2.2, 4.0, "me")
    out = _merge_segments(
        [(a, 0.9), (b, 0.9)],
        all_segments=[a, foreign, b],
        target_speakers={"me"},
        max_gap=0.3,
        min_duration=1.0,
        max_duration=10.0,
    )
    assert len(out) == 2


def test_quality_metrics_accept_clean_tone(tmp_path: Path):
    path = tmp_path / "tone.wav"
    rate = 16000
    samples = array("h")
    for i in range(rate * 2):
        samples.append(int(6000 * math.sin(2 * math.pi * 220 * i / rate)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    metrics = analyze_pcm16_wav(path, -45, 0.01, 0.9, 1.0, 10.0)
    assert metrics.accepted
    assert 1.9 < metrics.duration < 2.1


def test_project_state_numbering(tmp_path: Path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "000001.wav").write_bytes(b"")
    (audio / "000017.wav").write_bytes(b"")
    state = ProjectState(tmp_path)
    assert state.next_audio_index() == 18


def test_fingerprint_survives_rename(tmp_path: Path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 10000 + b"tail")
    first = ProjectState.fingerprint(a)
    b = tmp_path / "renamed.bin"
    a.rename(b)
    assert ProjectState.fingerprint(b) == first


def test_balanced_reference_is_cleaned_before_embedding(tmp_path: Path, monkeypatch):
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline
    import app.pipeline as pipeline_module

    streams = tmp_path / "streams"
    refs = tmp_path / "refs"
    output = tmp_path / "out"
    streams.mkdir()
    refs.mkdir()
    source = refs / "reference.mp3"
    source.write_bytes(b"fake")

    settings = AppSettings(
        streams_dir=streams,
        references_dir=refs,
        output_dir=output,
        quality="Сбалансированно",
    )
    pipe = DatasetPipeline(settings)
    pipe.cache.mkdir(parents=True, exist_ok=True)

    calls = {"extract": 0, "separate": 0, "convert": 0}

    def fake_extract(src, dest, cancel=None):
        calls["extract"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wav")

    def fake_separate(src, work_dir, model_name, cancel=None, **kwargs):
        calls["separate"] += 1
        out = work_dir / "demucs_fake_vocals.wav"
        out.write_bytes(b"vocals")
        return out

    def fake_convert(src, dest, cancel=None):
        calls["convert"] += 1
        dest.write_bytes(b"analysis")

    monkeypatch.setattr(pipeline_module, "extract_highres_audio", fake_extract)
    monkeypatch.setattr(pipeline_module, "separate_vocals", fake_separate)
    monkeypatch.setattr(pipeline_module, "convert_analysis_wav", fake_convert)

    result = pipe._prepare_reference(source, 1, 1, pipe.cache / "references")
    assert result.exists()
    assert "htdemucs" in result.name
    assert calls == {"extract": 1, "separate": 1, "convert": 1}

    # Cached second pass must not invoke the heavy stages again.
    result2 = pipe._prepare_reference(source, 1, 1, pipe.cache / "references")
    assert result2 == result
    assert calls == {"extract": 1, "separate": 1, "convert": 1}


def test_fast_reference_skips_music_separation(tmp_path: Path, monkeypatch):
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline
    import app.pipeline as pipeline_module

    streams = tmp_path / "streams"
    refs = tmp_path / "refs"
    output = tmp_path / "out"
    streams.mkdir()
    refs.mkdir()
    source = refs / "reference.wav"
    source.write_bytes(b"fake")

    settings = AppSettings(
        streams_dir=streams,
        references_dir=refs,
        output_dir=output,
        quality="Быстро",
    )
    pipe = DatasetPipeline(settings)
    pipe.cache.mkdir(parents=True, exist_ok=True)

    separated = {"called": False}

    def fake_extract(src, dest, cancel=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wav")

    def fail_separate(*args, **kwargs):
        separated["called"] = True
        raise AssertionError("Fast mode must not separate references")

    def fake_convert(src, dest, cancel=None):
        dest.write_bytes(b"analysis")

    monkeypatch.setattr(pipeline_module, "extract_highres_audio", fake_extract)
    monkeypatch.setattr(pipeline_module, "separate_vocals", fail_separate)
    monkeypatch.setattr(pipeline_module, "convert_analysis_wav", fake_convert)

    result = pipe._prepare_reference(source, 1, 1, pipe.cache / "references")
    assert result.exists()
    assert result.name == "analysis_raw.wav"
    assert not separated["called"]


def test_run_command_does_not_deadlock_on_large_child_output():
    import sys
    from app.media import run_command

    # > typical Windows anonymous-pipe buffer; the old implementation could stall
    # because it piped stderr/stdout but never drained them until process exit.
    code = "import sys; sys.stderr.write('x'*300000); sys.stderr.flush()"
    run_command([sys.executable, "-c", code])



def test_audio_mapping_uses_soundfile_not_torchcodec(tmp_path: Path):
    from app.audioio import load_audio_mapping
    path = tmp_path / "input.wav"
    rate = 16000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * rate)
    audio = load_audio_mapping(path)
    assert audio["sample_rate"] == rate
    assert tuple(audio["waveform"].shape) == (1, rate)


def test_diarizer_passes_in_memory_mapping(monkeypatch, tmp_path: Path):
    import app.diarization as diarization_module

    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    class DummyPipeline:
        def __init__(self):
            self.received = None
        def __call__(self, value):
            self.received = value
            return []

    d = object.__new__(diarization_module.Diarizer)
    import torch
    d.torch = torch
    d.pipeline = DummyPipeline()
    out = d.run(path)
    assert out == []
    assert isinstance(d.pipeline.received, dict)
    assert "waveform" in d.pipeline.received
    assert d.pipeline.received["sample_rate"] == 16000



def test_memory_safe_preset_chunk_limits():
    from app.config import QUALITY_PRESETS
    assert QUALITY_PRESETS["Быстро"].diarization_window_seconds == 300
    assert QUALITY_PRESETS["Сбалансированно"].diarization_window_seconds == 600
    assert QUALITY_PRESETS["Максимум"].diarization_window_seconds == 600


def test_low_vram_caps_diarization_window(monkeypatch):
    import app.resources as resources
    monkeypatch.setattr(resources, "resolve_torch_device", lambda _requested: "cuda")
    monkeypatch.setattr(resources, "cuda_total_gb", lambda: 6.0)
    assert resources.resolve_diarization_window(600, "auto") == 300
    assert resources.resolve_diarization_window(300, "auto") == 300


def test_demucs_command_uses_memory_safe_segment(tmp_path: Path):
    from app.separation import _demucs_command
    cmd = _demucs_command(
        tmp_path / "input.wav", tmp_path / "out", "htdemucs_ft", "cuda", 7
    )
    assert "-d" in cmd and cmd[cmd.index("-d") + 1] == "cuda"
    assert "--segment" in cmd and cmd[cmd.index("--segment") + 1] == "7"


def test_cuda_oom_detection():
    from app.resources import is_cuda_oom
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 512 MiB"))
    assert not is_cuda_oom(RuntimeError("file not found"))


def test_transcriber_close_unloads_inner_model(monkeypatch):
    from app.transcription import Transcriber

    class Inner:
        def __init__(self):
            self.called = False
        def unload_model(self):
            self.called = True

    class Outer:
        def __init__(self):
            self.model = Inner()

    t = object.__new__(Transcriber)
    outer = Outer()
    inner = outer.model
    t.model = outer
    t.close()
    assert inner.called
    assert t.model is None


def test_hybrid_demucs_segment_stays_under_model_limit(monkeypatch):
    import app.resources as resources
    monkeypatch.setattr(resources, "resolve_torch_device", lambda _requested: "cuda")
    monkeypatch.setattr(resources, "cuda_total_gb", lambda: 6.0)
    assert resources.resolve_demucs_segment("auto", "htdemucs_ft") == 6
    monkeypatch.setattr(resources, "cuda_total_gb", lambda: 8.0)
    assert resources.resolve_demucs_segment("auto", "htdemucs") == 7


def test_v06_speaker_encoder_is_forced_to_cpu(tmp_path, monkeypatch):
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline
    import app.pipeline as pipeline_module

    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "out"
    settings = AppSettings(streams, refs, out, quality="Быстро", device="cuda")
    pipe = DatasetPipeline(settings)

    seen = {}
    class DummyEncoder:
        def __init__(self, token, device):
            seen["device"] = device
            self.device = device
        def close(self):
            pass

    monkeypatch.setattr(pipeline_module, "SpeakerEncoder", DummyEncoder)
    monkeypatch.setattr(pipeline_module, "release_accelerator_memory", lambda: None)
    monkeypatch.setattr(pipeline_module, "log_memory", lambda *args, **kwargs: None)

    enc = pipe._load_speaker_encoder("test")
    assert enc.device == "cpu"
    assert seen["device"] == "cpu"


def test_v06_reference_centroid_cache_avoids_second_encoder_run(tmp_path, monkeypatch):
    import numpy as np
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline

    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "out"
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"prepared-reference")

    pipe = DatasetPipeline(AppSettings(streams, refs, out, quality="Быстро"))
    pipe.cache.mkdir(parents=True, exist_ok=True)
    calls = {"load": 0, "centroid": 0}

    class DummyEncoder:
        device = "cpu"
        def centroid(self, paths):
            calls["centroid"] += 1
            return np.array([0.25, 0.75], dtype="float32")
        def close(self):
            pass

    def fake_load(label):
        calls["load"] += 1
        return DummyEncoder()

    monkeypatch.setattr(pipe, "_load_speaker_encoder", fake_load)
    first = pipe._build_reference_centroid([prepared])
    second = pipe._build_reference_centroid([prepared])
    assert np.allclose(first, second)
    assert calls == {"load": 1, "centroid": 1}


def _dummy_staged_clip(path: Path, source: str = "stream.m4a", start: float = 1.0, end: float = 3.0):
    from app.models import QualityMetrics, StagedClip
    return StagedClip(
        path=path,
        text="",
        source=source,
        start=start,
        end=end,
        speaker_similarity=0.8,
        metrics=QualityMetrics(
            duration=end-start,
            rms_dbfs=-20.0,
            peak_dbfs=-3.0,
            clipping_ratio=0.0,
            silence_ratio=0.1,
            accepted=True,
            reason="",
        ),
    )


def test_v07_writer_commits_audio_immediately_and_deduplicates(tmp_path: Path):
    from app.metadata import DatasetWriter

    stage = tmp_path / "stage.wav"
    stage.write_bytes(b"fake-wav")
    writer = DatasetWriter(tmp_path / "dataset", 1)
    clip = _dummy_staged_clip(stage)
    count, seconds, usable = writer.commit_detailed([clip])
    assert count == 1
    assert seconds == 2.0
    assert len(usable) == 1
    assert (tmp_path / "dataset" / "audio" / "000001.wav").exists()

    # Reprocessing the same source/start/end must not create 000002.wav.
    stage2 = tmp_path / "stage2.wav"
    stage2.write_bytes(b"fake-wav-2")
    duplicate = _dummy_staged_clip(stage2)
    count2, seconds2, usable2 = writer.commit_detailed([duplicate])
    assert count2 == 0
    assert seconds2 == 0.0
    assert len(usable2) == 1
    assert not (tmp_path / "dataset" / "audio" / "000002.wav").exists()
    assert usable2[0].path.name == "000001.wav"


def test_v07_writer_can_patch_transcript_after_audio_commit(tmp_path: Path):
    import csv
    from app.metadata import DatasetWriter

    stage = tmp_path / "stage.wav"
    stage.write_bytes(b"fake-wav")
    writer = DatasetWriter(tmp_path / "dataset", 1)
    clip = _dummy_staged_clip(stage)
    _count, _seconds, usable = writer.commit_detailed([clip])
    usable[0].text = "привет мир"
    writer.update_texts(usable)

    with (tmp_path / "dataset" / "metadata.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="|"))
    assert rows == [["audio/000001.wav", "привет мир"]]



def test_v08_quality_scan_handles_full_scale_pcm_without_scalar_overflow(tmp_path: Path):
    path = tmp_path / "fullscale.wav"
    rate = 16000
    samples = array("h", [-32768, 32767] * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    metrics = analyze_pcm16_wav(path, -120, 1.0, 1.0, 1.0, 10.0)
    assert 1.9 < metrics.duration < 2.1
    assert metrics.peak_dbfs > -0.1
    assert metrics.rms_dbfs > -0.1


def test_v08_best_speaker_only_when_runner_up_is_lower_enough():
    from app.pipeline import _select_target_speakers
    target, reason = _select_target_speakers(
        {"SPEAKER_01": 0.769, "SPEAKER_00": 0.637}, threshold=0.34, margin=0.06
    )
    assert target == {"SPEAKER_01"}
    assert "отрыв" in reason


def test_v08_ambiguous_speakers_reject_chunk():
    from app.pipeline import _select_target_speakers
    target, reason = _select_target_speakers(
        {"SPEAKER_00": 0.610, "SPEAKER_01": 0.590}, threshold=0.34, margin=0.06
    )
    assert target == set()
    assert "неоднозначно" in reason


def test_v08_single_speaker_above_threshold_is_accepted():
    from app.pipeline import _select_target_speakers
    target, _reason = _select_target_speakers({"SPEAKER_00": 0.780}, threshold=0.34, margin=0.06)
    assert target == {"SPEAKER_00"}



def test_v08_refuses_mixing_old_algorithm_audio(tmp_path: Path):
    import json
    import pytest
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline

    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "dataset"; (out / "audio").mkdir(parents=True)
    (out / "audio" / "000001.wav").write_bytes(b"old")
    (out / "settings.json").write_text(json.dumps({"quality": "Быстро", "output_sample_rate": 48000}), encoding="utf-8")
    pipe = DatasetPipeline(AppSettings(streams, refs, out, quality="Быстро"))
    with pytest.raises(RuntimeError, match="старой логикой speaker selection"):
        pipe._prepare_output()


def test_demucs_outer_chunk_plan_bounds_long_stream():
    from app.separation import plan_separation_chunks

    chunks = plan_separation_chunks(6867.0, 900.0, 2.0)
    assert len(chunks) == 8
    assert chunks[0].core_start == 0.0
    assert chunks[0].input_start == 0.0
    assert chunks[0].core_duration == 900.0
    # Middle chunks receive context on both sides but are trimmed back to core.
    assert chunks[1].input_start == 898.0
    assert chunks[1].input_duration == 904.0
    assert chunks[1].trim_start == 2.0
    assert abs(sum(c.core_duration for c in chunks) - 6867.0) < 1e-6
    assert chunks[-1].core_start + chunks[-1].core_duration == 6867.0


def test_long_demucs_uses_outer_chunks_and_resume_cache(tmp_path: Path, monkeypatch):
    import app.separation as sep

    source = tmp_path / "two_hours.wav"
    source.write_bytes(b"source")
    work = tmp_path / "work"
    work.mkdir()

    durations = {str(source): 6867.0}
    calls = {"single": 0, "cut": 0, "trim": 0, "concat": 0}

    monkeypatch.setattr(sep, "demucs_available", lambda: True)
    monkeypatch.setattr(sep, "probe_duration", lambda p: durations.get(str(p), 900.0))

    def fake_cut(src, dest, start, duration, cancel=None):
        calls["cut"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"input")
        durations[str(dest)] = duration

    def fake_single(src, work_dir, model_name, cancel=None, **kwargs):
        calls["single"] += 1
        out = work_dir / "demucs" / model_name / src.stem / "vocals.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"vocals")
        durations[str(out)] = durations[str(src)]
        return out

    def fake_trim(src, dest, start, duration, cancel=None):
        calls["trim"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"core")
        durations[str(dest)] = duration

    def fake_concat(parts, dest, cancel=None):
        calls["concat"] += 1
        dest.write_bytes(b"joined")
        durations[str(dest)] = sum(durations[str(p)] for p in parts)

    monkeypatch.setattr(sep, "_cut_stereo_pcm", fake_cut)
    monkeypatch.setattr(sep, "_single_pass_vocals", fake_single)
    monkeypatch.setattr(sep, "_trim_stem", fake_trim)
    monkeypatch.setattr(sep, "_concat_stems", fake_concat)

    out = sep.separate_vocals(source, work, "htdemucs", outer_chunk_seconds=900)
    assert out.exists()
    assert calls == {"single": 8, "cut": 8, "trim": 8, "concat": 1}
    assert abs(durations[str(out)] - 6867.0) < 1e-6

    # A second call should reuse the completed merged stem without rerunning Demucs.
    out2 = sep.separate_vocals(source, work, "htdemucs", outer_chunk_seconds=900)
    assert out2 == out
    assert calls == {"single": 8, "cut": 8, "trim": 8, "concat": 1}


def test_v11_speaker_encoder_single_load_is_reused_within_run(tmp_path, monkeypatch):
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline
    import app.pipeline as pipeline_module

    pipeline_module.shutdown_process_speaker_encoder()
    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "out"
    pipe = DatasetPipeline(AppSettings(streams, refs, out, quality="Максимум", device="cuda"))

    calls = {"init": 0, "close": 0}
    class DummyEncoder:
        device = "cpu"
        def __init__(self, token, device):
            calls["init"] += 1
            assert device == "cpu"
        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(pipeline_module, "SpeakerEncoder", DummyEncoder)
    monkeypatch.setattr(pipeline_module, "release_accelerator_memory", lambda: None)
    monkeypatch.setattr(pipeline_module, "log_memory", lambda *args, **kwargs: None)

    first = pipe._get_speaker_encoder("reference")
    second = pipe._get_speaker_encoder("verification after diarization")
    assert first is second
    assert calls["init"] == 1
    pipe._close_speaker_encoder()
    # Per-run close only detaches: the app-session encoder stays alive.
    assert calls["close"] == 0
    pipeline_module.shutdown_process_speaker_encoder()
    assert calls["close"] == 1


def test_v10_cached_centroid_still_preloads_encoder_before_diarization(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline

    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "out"
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"prepared-reference")
    pipe = DatasetPipeline(AppSettings(streams, refs, out, quality="Максимум"))
    pipe.cache.mkdir(parents=True, exist_ok=True)

    stat = prepared.stat()
    signature = "\n".join([
        "wespeaker-voxceleb-resnet34-LM|v1.0",
        f"{prepared.resolve()}|{stat.st_size}|{stat.st_mtime_ns}",
    ])
    key = hashlib.sha1(signature.encode()).hexdigest()[:20]
    centroid_dir = pipe.cache / "references" / "centroids"
    centroid_dir.mkdir(parents=True, exist_ok=True)
    np.save(centroid_dir / f"{key}.npy", np.array([0.2, 0.8], dtype="float32"), allow_pickle=False)

    calls = {"get": 0}
    sentinel = object()
    def fake_get(label):
        calls["get"] += 1
        return sentinel
    monkeypatch.setattr(pipe, "_get_speaker_encoder", fake_get)

    centroid = pipe._build_reference_centroid([prepared])
    assert np.allclose(centroid, [0.2, 0.8])
    assert calls["get"] == 1


def test_v11_process_speaker_encoder_reused_across_two_pipeline_runs(tmp_path, monkeypatch):
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline
    import app.pipeline as pipeline_module

    pipeline_module.shutdown_process_speaker_encoder()
    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "out"
    calls = {"init": 0, "close": 0}

    class DummyEncoder:
        device = "cpu"
        def __init__(self, token, device):
            calls["init"] += 1
        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(pipeline_module, "SpeakerEncoder", DummyEncoder)
    monkeypatch.setattr(pipeline_module, "release_accelerator_memory", lambda: None)
    monkeypatch.setattr(pipeline_module, "log_memory", lambda *args, **kwargs: None)

    p1 = DatasetPipeline(AppSettings(streams, refs, out, quality="Быстро"))
    e1 = p1._get_speaker_encoder("run1")
    p1._close_speaker_encoder()

    p2 = DatasetPipeline(AppSettings(streams, refs, out, quality="Сбалансированно"))
    e2 = p2._get_speaker_encoder("run2")
    p2._close_speaker_encoder()

    assert e1 is e2
    assert calls["init"] == 1
    assert calls["close"] == 0
    pipeline_module.shutdown_process_speaker_encoder()
    assert calls["close"] == 1


def test_v11_manual_output_cleanup_resets_done_state(tmp_path: Path, monkeypatch):
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline

    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "dataset"; out.mkdir()
    source = streams / "stream.m4a"
    source.write_bytes(b"stream")

    state = ProjectState(out)
    fp = state.fingerprint(source)
    state.mark_done(fp, source, 10, 20.0)
    # Simulate Explorer cleanup of final output while dot-state/cache survived.
    assert state.is_done(fp)

    pipe = DatasetPipeline(AppSettings(streams, refs, out, quality="Быстро"))
    pipe._prepare_output()
    assert not pipe.state.is_done(fp)


def test_v11_missing_audio_rows_are_pruned_and_can_be_recreated(tmp_path: Path):
    import csv
    from app.metadata import DatasetWriter, repair_dataset_metadata

    out = tmp_path / "dataset"
    audio = out / "audio"
    audio.mkdir(parents=True)
    (out / "metadata.csv").write_text("audio/000001.wav|old text\n", encoding="utf-8")
    (out / "metadata_extended.csv").write_text(
        "file|text|source|start|end|duration|speaker_similarity|rms_dbfs|peak_dbfs|clipping_ratio|silence_ratio\n"
        "audio/000001.wav|old text|stream.m4a|1.000|3.000|2.000|0.8|-20|-3|0|0.1\n",
        encoding="utf-8",
    )
    # WAV was manually deleted.
    repaired = repair_dataset_metadata(out)
    assert repaired["removed"] == 1

    stage = tmp_path / "new.wav"
    stage.write_bytes(b"new")
    writer = DatasetWriter(out, 1)
    clip = _dummy_staged_clip(stage)
    count, _seconds, _usable = writer.commit_detailed([clip])
    assert count == 1
    assert (audio / "000001.wav").exists()


def test_v11_corrupt_centroid_cache_is_recomputed(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from app.config import AppSettings
    from app.pipeline import DatasetPipeline
    import app.pipeline as pipeline_module

    streams = tmp_path / "streams"; streams.mkdir()
    refs = tmp_path / "refs"; refs.mkdir()
    out = tmp_path / "out"
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"prepared-reference")
    pipe = DatasetPipeline(AppSettings(streams, refs, out, quality="Быстро"))
    pipe.cache.mkdir(parents=True, exist_ok=True)

    stat = prepared.stat()
    signature = "\n".join([
        "wespeaker-voxceleb-resnet34-LM|v1.0",
        f"{prepared.resolve()}|{stat.st_size}|{stat.st_mtime_ns}",
    ])
    key = hashlib.sha1(signature.encode()).hexdigest()[:20]
    centroid_dir = pipe.cache / "references" / "centroids"
    centroid_dir.mkdir(parents=True, exist_ok=True)
    (centroid_dir / f"{key}.npy").write_bytes(b"not-a-numpy-file")

    class DummyEncoder:
        device = "cpu"
        def centroid(self, paths):
            return np.array([0.3, 0.7], dtype="float32")

    monkeypatch.setattr(pipe, "_get_speaker_encoder", lambda label: DummyEncoder())
    centroid = pipe._build_reference_centroid([prepared])
    assert np.allclose(centroid, [0.3, 0.7])
