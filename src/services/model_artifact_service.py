from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
import streamlit as st


DEFAULT_CACHE_DIR = Path("outputs") / "models" / "runtime"
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 300
DEFAULT_MAX_SIZE_MB = 512
MODEL_TYPE_BERTOPIC_GLOBAL = "bertopic_global"
_LOCAL_MANIFEST_NAME = "_artifact_manifest.json"


@dataclass(frozen=True)
class ModelArtifactManifest:
    bundle_version: str
    artifact_url: str
    sha256: str
    size_bytes: int
    model_type: str
    bertopic_version: str
    python_version: str
    load_subpath: str


@dataclass(frozen=True)
class ModelArtifactStatus:
    state: str
    enabled: bool
    configured: bool
    ready: bool
    model_type: str
    bundle_version: str = ""
    manifest_url: str = ""
    artifact_url: str = ""
    sha256: str = ""
    size_bytes: int = 0
    load_subpath: str = ""
    cache_dir: str = ""
    bundle_dir: str = ""
    local_model_path: str = ""
    message: str = ""
    failure_reason: str = ""


def _secret_values() -> Dict[str, Any]:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def _read_setting(name: str) -> Any:
    secrets_map = _secret_values()
    if name in secrets_map:
        return secrets_map.get(name)
    return os.getenv(name)


def _read_text_setting(name: str, default: str = "") -> str:
    value = _read_setting(name)
    return str(value or default).strip()


def _read_bool_setting(name: str, default: bool = False) -> bool:
    value = _read_setting(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _read_int_setting(name: str, default: int) -> int:
    value = _read_setting(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return max(1, int(value))
    except Exception:
        return default


def model_artifacts_enabled() -> bool:
    return _read_bool_setting("MODEL_ARTIFACTS_ENABLED", default=False)


def get_model_artifact_manifest_url() -> str:
    return _read_text_setting("MODEL_ARTIFACTS_MANIFEST_URL")


def get_model_artifact_cache_dir() -> Path:
    custom_dir = _read_text_setting("MODEL_ARTIFACTS_CACHE_DIR")
    return Path(custom_dir) if custom_dir else DEFAULT_CACHE_DIR


def get_model_artifact_download_timeout_seconds() -> int:
    return _read_int_setting("MODEL_ARTIFACTS_DOWNLOAD_TIMEOUT_SECONDS", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)


def get_model_artifact_max_size_bytes() -> int:
    return _read_int_setting("MODEL_ARTIFACTS_MAX_SIZE_MB", DEFAULT_MAX_SIZE_MB) * 1024 * 1024


def _validate_manifest(payload: Dict[str, Any]) -> ModelArtifactManifest:
    required_fields = (
        "bundle_version",
        "artifact_url",
        "sha256",
        "size_bytes",
        "model_type",
        "bertopic_version",
        "python_version",
        "load_subpath",
    )
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        raise RuntimeError(f"Artifact manifest is missing required fields: {', '.join(missing)}")

    model_type = str(payload.get("model_type", "")).strip()
    if model_type != MODEL_TYPE_BERTOPIC_GLOBAL:
        raise RuntimeError(f"Unsupported model_type in manifest: {model_type}")

    sha256 = str(payload.get("sha256", "")).strip().lower()
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise RuntimeError("Artifact manifest contains an invalid sha256 checksum.")

    try:
        size_bytes = int(payload.get("size_bytes", 0))
    except Exception as exc:
        raise RuntimeError("Artifact manifest contains an invalid size_bytes value.") from exc
    if size_bytes <= 0:
        raise RuntimeError("Artifact manifest size_bytes must be greater than zero.")

    return ModelArtifactManifest(
        bundle_version=str(payload["bundle_version"]).strip(),
        artifact_url=str(payload["artifact_url"]).strip(),
        sha256=sha256,
        size_bytes=size_bytes,
        model_type=model_type,
        bertopic_version=str(payload["bertopic_version"]).strip(),
        python_version=str(payload["python_version"]).strip(),
        load_subpath=str(payload["load_subpath"]).strip(),
    )


def fetch_bertopic_manifest() -> ModelArtifactManifest:
    manifest_url = get_model_artifact_manifest_url()
    if not manifest_url:
        raise RuntimeError("MODEL_ARTIFACTS_MANIFEST_URL is not configured.")

    response = requests.get(manifest_url, timeout=min(get_model_artifact_download_timeout_seconds(), 60))
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to fetch model artifact manifest ({response.status_code}).")
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("Model artifact manifest is not valid JSON.") from exc
    return _validate_manifest(payload)


def _bundle_dir(manifest: ModelArtifactManifest) -> Path:
    return get_model_artifact_cache_dir() / manifest.bundle_version


def _local_model_path(manifest: ModelArtifactManifest) -> Path:
    return _bundle_dir(manifest) / manifest.load_subpath


def _local_manifest_path(manifest: ModelArtifactManifest) -> Path:
    return _bundle_dir(manifest) / _LOCAL_MANIFEST_NAME


def _write_local_manifest(manifest: ModelArtifactManifest) -> None:
    manifest_path = _local_manifest_path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_artifact(manifest: ModelArtifactManifest) -> Path:
    timeout = get_model_artifact_download_timeout_seconds()
    max_size = get_model_artifact_max_size_bytes()
    if manifest.size_bytes > max_size:
        raise RuntimeError(
            f"Artifact bundle is too large for the configured limit ({manifest.size_bytes} bytes > {max_size} bytes)."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="bertopic_bundle_"))
    parsed = urlparse(manifest.artifact_url)
    artifact_name = Path(parsed.path).name or "artifact_bundle"
    download_path = temp_dir / artifact_name

    response = requests.get(manifest.artifact_url, stream=True, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to download BERTopic artifact bundle ({response.status_code}).")

    downloaded = 0
    with download_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_size:
                raise RuntimeError("Artifact bundle exceeded the configured maximum size during download.")
            handle.write(chunk)

    if manifest.size_bytes and downloaded != manifest.size_bytes:
        raise RuntimeError(
            f"Downloaded artifact size does not match manifest ({downloaded} bytes != {manifest.size_bytes} bytes)."
        )

    checksum = _sha256(download_path)
    if checksum != manifest.sha256:
        raise RuntimeError("Downloaded BERTopic artifact failed checksum validation.")
    return download_path


def _extract_artifact(download_path: Path, bundle_dir: Path, manifest: ModelArtifactManifest) -> Path:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    lower_name = download_path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(download_path, "r") as archive:
            archive.extractall(bundle_dir)
    elif lower_name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(download_path, "r:*") as archive:
            archive.extractall(bundle_dir)
    else:
        target_path = bundle_dir / manifest.load_subpath
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(download_path), str(target_path))

    _write_local_manifest(manifest)
    model_path = _local_model_path(manifest)
    if not model_path.exists():
        raise RuntimeError(
            f"Artifact bundle was downloaded, but the configured load_subpath was not found: {manifest.load_subpath}"
        )
    return model_path


def _build_status_from_manifest(manifest: ModelArtifactManifest, *, state: str, ready: bool, message: str = "", failure_reason: str = "") -> ModelArtifactStatus:
    bundle_dir = _bundle_dir(manifest)
    model_path = _local_model_path(manifest)
    return ModelArtifactStatus(
        state=state,
        enabled=True,
        configured=True,
        ready=ready,
        model_type=manifest.model_type,
        bundle_version=manifest.bundle_version,
        manifest_url=get_model_artifact_manifest_url(),
        artifact_url=manifest.artifact_url,
        sha256=manifest.sha256,
        size_bytes=manifest.size_bytes,
        load_subpath=manifest.load_subpath,
        cache_dir=str(get_model_artifact_cache_dir()),
        bundle_dir=str(bundle_dir),
        local_model_path=str(model_path) if model_path.exists() else "",
        message=message,
        failure_reason=failure_reason,
    )


def get_bertopic_artifact_status() -> ModelArtifactStatus:
    if not model_artifacts_enabled():
        return ModelArtifactStatus(
            state="disabled",
            enabled=False,
            configured=False,
            ready=False,
            model_type=MODEL_TYPE_BERTOPIC_GLOBAL,
            cache_dir=str(get_model_artifact_cache_dir()),
            message="Model artifacts are disabled. Enable MODEL_ARTIFACTS_ENABLED to turn on the beta BERTopic workflow.",
        )

    manifest_url = get_model_artifact_manifest_url()
    if not manifest_url:
        return ModelArtifactStatus(
            state="unconfigured",
            enabled=True,
            configured=False,
            ready=False,
            model_type=MODEL_TYPE_BERTOPIC_GLOBAL,
            cache_dir=str(get_model_artifact_cache_dir()),
            message="MODEL_ARTIFACTS_MANIFEST_URL is missing, so the beta BERTopic workflow is unavailable.",
        )

    try:
        manifest = fetch_bertopic_manifest()
    except Exception as exc:
        return ModelArtifactStatus(
            state="invalid",
            enabled=True,
            configured=True,
            ready=False,
            model_type=MODEL_TYPE_BERTOPIC_GLOBAL,
            manifest_url=manifest_url,
            cache_dir=str(get_model_artifact_cache_dir()),
            message="The BERTopic manifest could not be loaded.",
            failure_reason=str(exc),
        )

    model_path = _local_model_path(manifest)
    if model_path.exists():
        return _build_status_from_manifest(
            manifest,
            state="ready",
            ready=True,
            message="BERTopic bundle is available locally and ready to use.",
        )

    return _build_status_from_manifest(
        manifest,
        state="download_required",
        ready=False,
        message="BERTopic bundle is configured but not cached locally yet. It will download on the next beta analysis refresh.",
    )


def ensure_bertopic_artifact_ready() -> ModelArtifactStatus:
    status = get_bertopic_artifact_status()
    if status.ready or status.state in {"disabled", "unconfigured", "invalid"}:
        return status

    try:
        manifest = fetch_bertopic_manifest()
        download_path = _download_artifact(manifest)
        try:
            model_path = _extract_artifact(download_path, _bundle_dir(manifest), manifest)
        finally:
            download_root = download_path.parent
            if download_root.exists():
                shutil.rmtree(download_root, ignore_errors=True)
    except Exception as exc:
        return ModelArtifactStatus(
            state="invalid",
            enabled=True,
            configured=True,
            ready=False,
            model_type=MODEL_TYPE_BERTOPIC_GLOBAL,
            manifest_url=get_model_artifact_manifest_url(),
            cache_dir=str(get_model_artifact_cache_dir()),
            message="The BERTopic bundle could not be prepared.",
            failure_reason=str(exc),
        )

    return ModelArtifactStatus(
        state="ready",
        enabled=True,
        configured=True,
        ready=True,
        model_type=manifest.model_type,
        bundle_version=manifest.bundle_version,
        manifest_url=get_model_artifact_manifest_url(),
        artifact_url=manifest.artifact_url,
        sha256=manifest.sha256,
        size_bytes=manifest.size_bytes,
        load_subpath=manifest.load_subpath,
        cache_dir=str(get_model_artifact_cache_dir()),
        bundle_dir=str(_bundle_dir(manifest)),
        local_model_path=str(model_path),
        message="BERTopic bundle downloaded and prepared successfully.",
    )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
    "DEFAULT_MAX_SIZE_MB",
    "MODEL_TYPE_BERTOPIC_GLOBAL",
    "ModelArtifactManifest",
    "ModelArtifactStatus",
    "ensure_bertopic_artifact_ready",
    "fetch_bertopic_manifest",
    "get_model_artifact_cache_dir",
    "get_bertopic_artifact_status",
    "get_model_artifact_download_timeout_seconds",
    "get_model_artifact_manifest_url",
    "get_model_artifact_max_size_bytes",
    "model_artifacts_enabled",
]
