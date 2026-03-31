from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from src.services.model_artifact_service import ModelArtifactStatus
from src.services.topic_model_runtime import _load_topic_model, apply_optional_topic_model, build_bertopic_inference_text
from src.utils.bertopic_compat import load_bertopic_with_cpu_fallback


def test_build_bertopic_inference_text_applies_cleaning() -> None:
    text, token_count, is_sparse = build_bertopic_inference_text(
        "How To Fix 3 Thumbnail Problems",
        "Subscribe here!\nThis breaks down packaging and click through rate. https://example.com",
        "thumbnail|ctr|packaging",
    )

    assert "subscribe" not in text
    assert "https" not in text
    assert "thumbnail" in text
    assert "3" not in text
    assert token_count >= 5
    assert is_sparse is False


def test_apply_optional_topic_model_returns_rows(monkeypatch) -> None:
    sample_df = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "video_title": "How To Improve Packaging",
                "video_description": "packaging and ctr ideas",
                "video_tags": "packaging|ctr",
            }
        ]
    )

    class _FakeTopicModel:
        def transform(self, texts):
            assert len(texts) == 1
            return [18], None

        def get_topic(self, topic_id: int):
            assert topic_id == 18
            return [("packaging", 0.4), ("ctr", 0.3), ("thumbnail", 0.2), ("hooks", 0.1)]

    monkeypatch.setattr(
        "src.services.topic_model_runtime.ensure_bertopic_artifact_ready",
        lambda: ModelArtifactStatus(
            state="ready",
            enabled=True,
            configured=True,
            ready=True,
            model_type="bertopic_global",
            bundle_version="2026.03.27",
            local_model_path="/tmp/bertopic_model.pkl",
        ),
    )
    monkeypatch.setattr("src.services.topic_model_runtime._load_topic_model", lambda path: _FakeTopicModel())

    result = apply_optional_topic_model(sample_df)

    assert result.success is True
    assert result.bundle_version == "2026.03.27"
    assert result.topic_rows[0]["model_topic_id"] == 18
    assert result.topic_rows[0]["model_topic_label"] == "Packaging / Ctr / Thumbnail / Hooks"


def test_load_topic_model_retries_with_cpu_for_mps_storage(monkeypatch) -> None:
    torch_load_calls = []
    restore_calls = []

    def _fake_torch_load(buffer, *args, **kwargs):
        torch_load_calls.append(kwargs.get("map_location"))
        return {"ok": True}

    def _fake_default_restore_location(storage, location):
        restore_calls.append((storage, location))
        return {"storage": storage, "location": location}

    original_default_restore_location = _fake_default_restore_location

    fake_torch = SimpleNamespace(
        load=_fake_torch_load,
        serialization=SimpleNamespace(default_restore_location=_fake_default_restore_location),
        storage=SimpleNamespace(_load_from_bytes=lambda _blob: {"raw": True}),
    )

    class _FakeBERTopic:
        attempts = 0

        @classmethod
        def load(cls, model_path: str):
            cls.attempts += 1
            if cls.attempts == 1:
                raise RuntimeError("torch.UntypedStorage(): Storage device not recognized: mps")
            restored = fake_torch.serialization.default_restore_location("storage", "mps")
            loaded = fake_torch.storage._load_from_bytes(b"payload")
            return {"restored": restored, "loaded": loaded, "path": model_path}

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bertopic", SimpleNamespace(BERTopic=_FakeBERTopic))

    result = _load_topic_model("/tmp/bertopic_model")

    assert _FakeBERTopic.attempts == 2
    assert restore_calls == [("storage", "cpu")]
    assert torch_load_calls == ["cpu"]
    assert result["path"] == "/tmp/bertopic_model"
    assert result["restored"]["location"] == "cpu"
    assert result["loaded"] == {"ok": True}
    assert fake_torch.serialization.default_restore_location is original_default_restore_location


def test_load_bertopic_with_cpu_fallback_surfaces_actionable_error(monkeypatch) -> None:
    class _FailingBERTopic:
        attempts = 0

        @classmethod
        def load(cls, _model_path: str):
            cls.attempts += 1
            raise RuntimeError("torch.UntypedStorage(): Storage device not recognized: mps")

    fake_torch = SimpleNamespace(
        load=lambda *_args, **_kwargs: {"ok": True},
        serialization=SimpleNamespace(default_restore_location=lambda storage, location: {"storage": storage, "location": location}),
        storage=SimpleNamespace(_load_from_bytes=lambda _blob: {"raw": True}),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bertopic", SimpleNamespace(BERTopic=_FailingBERTopic))

    try:
        load_bertopic_with_cpu_fallback("/tmp/bertopic_model")
    except RuntimeError as exc:
        assert "CPU-safe artifact bundle" in str(exc)
        assert _FailingBERTopic.attempts == 2
    else:
        raise AssertionError("Expected a CPU-safe artifact guidance error when both BERTopic load attempts fail.")
