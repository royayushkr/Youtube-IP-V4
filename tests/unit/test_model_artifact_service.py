from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from src.services.model_artifact_service import ensure_bertopic_artifact_ready, get_bertopic_artifact_status


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_payload=None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._json_payload = json_payload
        self._content = content

    def json(self):
        if self._json_payload is None:
            raise ValueError("No JSON payload configured")
        return self._json_payload

    def iter_content(self, chunk_size: int = 1024 * 1024):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]


def test_artifact_status_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_ARTIFACTS_ENABLED", raising=False)
    monkeypatch.delenv("MODEL_ARTIFACTS_MANIFEST_URL", raising=False)

    status = get_bertopic_artifact_status()

    assert status.state == "disabled"
    assert status.ready is False


def test_artifact_status_unconfigured_when_manifest_missing(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ARTIFACTS_ENABLED", "true")
    monkeypatch.delenv("MODEL_ARTIFACTS_MANIFEST_URL", raising=False)

    status = get_bertopic_artifact_status()

    assert status.state == "unconfigured"
    assert status.ready is False


def test_artifact_status_ready_when_cached_bundle_exists(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "bundle_version": "2026.03.27",
        "artifact_url": "https://example.com/bertopic_bundle.zip",
        "sha256": "a" * 64,
        "size_bytes": 1234,
        "model_type": "bertopic_global",
        "bertopic_version": "0.16.4",
        "python_version": "3.13",
        "load_subpath": "bundle/bertopic_model.pkl",
    }

    def fake_get(url: str, **kwargs):
        return _FakeResponse(json_payload=manifest)

    monkeypatch.setenv("MODEL_ARTIFACTS_ENABLED", "true")
    monkeypatch.setenv("MODEL_ARTIFACTS_MANIFEST_URL", "https://example.com/manifest.json")
    monkeypatch.setenv("MODEL_ARTIFACTS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("src.services.model_artifact_service.requests.get", fake_get)

    model_path = tmp_path / manifest["bundle_version"] / manifest["load_subpath"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"ready")

    status = get_bertopic_artifact_status()

    assert status.state == "ready"
    assert status.ready is True
    assert status.local_model_path.endswith("bertopic_model.pkl")


def test_ensure_artifact_ready_downloads_and_extracts_zip(monkeypatch, tmp_path: Path) -> None:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("bundle/bertopic_model.pkl", b"serialized-model")
    zip_bytes = zip_buffer.getvalue()

    import hashlib

    manifest = {
        "bundle_version": "2026.03.27",
        "artifact_url": "https://example.com/bertopic_bundle.zip",
        "sha256": hashlib.sha256(zip_bytes).hexdigest(),
        "size_bytes": len(zip_bytes),
        "model_type": "bertopic_global",
        "bertopic_version": "0.16.4",
        "python_version": "3.13",
        "load_subpath": "bundle/bertopic_model.pkl",
    }

    def fake_get(url: str, **kwargs):
        if url.endswith("manifest.json"):
            return _FakeResponse(json_payload=manifest)
        return _FakeResponse(content=zip_bytes)

    monkeypatch.setenv("MODEL_ARTIFACTS_ENABLED", "true")
    monkeypatch.setenv("MODEL_ARTIFACTS_MANIFEST_URL", "https://example.com/manifest.json")
    monkeypatch.setenv("MODEL_ARTIFACTS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("src.services.model_artifact_service.requests.get", fake_get)

    status = ensure_bertopic_artifact_ready()

    assert status.state == "ready"
    assert status.ready is True
    assert Path(status.local_model_path).exists()
    assert json.loads((tmp_path / manifest["bundle_version"] / "_artifact_manifest.json").read_text())["bundle_version"] == "2026.03.27"
