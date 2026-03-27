#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_bundle(model_path: Path, output_dir: Path, bundle_version: str) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"bertopic_bundle_{bundle_version}.zip"
    bundle_path = output_dir / bundle_name
    load_subpath = f"bundle/{model_path.name}"

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, arcname=load_subpath)
    return bundle_path, load_subpath


def _verify_bundle(bundle_path: Path, load_subpath: str) -> None:
    with tempfile.TemporaryDirectory(prefix="bertopic_bundle_verify_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with zipfile.ZipFile(bundle_path, "r") as archive:
            archive.extractall(temp_dir)
        model_path = temp_dir / load_subpath
        if not model_path.exists():
            raise RuntimeError(f"Bundle verification failed: {load_subpath} was not found after extraction.")

        from bertopic import BERTopic

        BERTopic.load(str(model_path))


def _release_asset_url(repo: str, tag: str, asset_name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def _build_manifest(
    *,
    bundle_path: Path,
    load_subpath: str,
    bundle_version: str,
    bertopic_version: str,
    python_version: str,
    artifact_url: str,
) -> Dict[str, Any]:
    return {
        "bundle_version": bundle_version,
        "artifact_url": artifact_url,
        "sha256": _sha256(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "model_type": "bertopic_global",
        "bertopic_version": bertopic_version,
        "python_version": python_version,
        "load_subpath": load_subpath,
    }


def _github_headers(token: str, *, accept: str = "application/vnd.github+json") -> Dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _ensure_release(repo: str, tag: str, token: str) -> Dict[str, Any]:
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    response = requests.get(api_url, headers=_github_headers(token), timeout=60)
    if response.status_code == 200:
        return response.json()
    if response.status_code != 404:
        raise RuntimeError(f"Failed to look up GitHub release ({response.status_code}): {response.text[:300]}")

    create_response = requests.post(
        f"https://api.github.com/repos/{repo}/releases",
        headers=_github_headers(token),
        json={"tag_name": tag, "name": tag, "draft": False, "prerelease": True},
        timeout=60,
    )
    if create_response.status_code >= 400:
        raise RuntimeError(f"Failed to create GitHub release ({create_response.status_code}): {create_response.text[:300]}")
    return create_response.json()


def _delete_existing_asset(repo: str, asset_name: str, release: Dict[str, Any], token: str) -> None:
    for asset in release.get("assets", []):
        if asset.get("name") != asset_name:
            continue
        delete_response = requests.delete(
            asset.get("url", ""),
            headers=_github_headers(token),
            timeout=60,
        )
        if delete_response.status_code >= 400:
            raise RuntimeError(
                f"Failed to delete existing release asset {asset_name} ({delete_response.status_code}): {delete_response.text[:300]}"
            )


def _upload_release_asset(repo: str, release: Dict[str, Any], asset_path: Path, token: str) -> None:
    _delete_existing_asset(repo, asset_path.name, release, token)
    upload_template = str(release.get("upload_url", ""))
    upload_url = upload_template.split("{", 1)[0]
    upload_url = f"{upload_url}?name={asset_path.name}"
    mime_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"

    with asset_path.open("rb") as handle:
        response = requests.post(
            upload_url,
            headers=_github_headers(token, accept="application/vnd.github+json") | {"Content-Type": mime_type},
            data=handle.read(),
            timeout=300,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to upload {asset_path.name} ({response.status_code}): {response.text[:300]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a BERTopic model artifact for GitHub Release delivery.")
    parser.add_argument("--model-path", required=True, help="Path to the locally trained BERTopic model file.")
    parser.add_argument("--output-dir", default="dist/bertopic_release", help="Directory for the generated bundle and manifest.")
    parser.add_argument("--bundle-version", required=True, help="Version string for the release bundle.")
    parser.add_argument("--bertopic-version", required=True, help="BERTopic version that produced the model.")
    parser.add_argument("--python-version", default=f"{sys.version_info.major}.{sys.version_info.minor}", help="Python version used to verify the bundle.")
    parser.add_argument("--repo", help="GitHub repo in owner/name form, for example royayushkr/Youtube-IP-V4.")
    parser.add_argument("--tag", help="Git tag / release tag to publish assets under.")
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN", ""), help="GitHub token with release upload access.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip local BERTopic.load verification of the packaged bundle.")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    bundle_path, load_subpath = _build_bundle(model_path, output_dir, args.bundle_version)

    if not args.skip_verify:
        _verify_bundle(bundle_path, load_subpath)

    artifact_url = ""
    if args.repo and args.tag:
        artifact_url = _release_asset_url(args.repo, args.tag, bundle_path.name)
    manifest = _build_manifest(
        bundle_path=bundle_path,
        load_subpath=load_subpath,
        bundle_version=args.bundle_version,
        bertopic_version=args.bertopic_version,
        python_version=args.python_version,
        artifact_url=artifact_url,
    )

    manifest_path = output_dir / f"bertopic_manifest_{args.bundle_version}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.repo and args.tag:
        if not args.github_token:
            raise SystemExit("A GitHub token is required to upload release assets.")
        release = _ensure_release(args.repo, args.tag, args.github_token)
        _upload_release_asset(args.repo, release, bundle_path, args.github_token)
        release = _ensure_release(args.repo, args.tag, args.github_token)
        _upload_release_asset(args.repo, release, manifest_path, args.github_token)

    print(f"Bundle:   {bundle_path}")
    print(f"Manifest: {manifest_path}")
    if args.repo and args.tag:
        print(f"Release asset URL: {_release_asset_url(args.repo, args.tag, bundle_path.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
