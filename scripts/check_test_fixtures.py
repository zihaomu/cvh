#!/usr/bin/env python3
"""Validate fixture manifests and upstream test registries without third-party deps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_relative_strings(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_relative_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_relative_strings(child, f"{location}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        raise RuntimeError(f"Absolute path in manifest at {location}: {value}")


def validate_core_fixtures() -> int:
    manifest = load_json("test/core/data/manifest.json")
    manifest_path = "test/core/data/manifest.json"
    assert_relative_strings(manifest, manifest_path)

    entries = manifest["fixtures"]
    expected_paths = {entry["path"] for entry in entries}
    actual_paths = {
        path.relative_to(REPO_ROOT / "test/core/data").as_posix()
        for path in (REPO_ROOT / "test/core/data/npy").glob("*.npy")
    }
    if expected_paths != actual_paths:
        raise RuntimeError(
            "Core fixture manifest mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unmanaged={sorted(actual_paths - expected_paths)}"
        )

    for entry in entries:
        fixture = REPO_ROOT / "test/core/data" / entry["path"]
        if sha256(fixture) != entry["sha256"]:
            raise RuntimeError(f"Core fixture hash mismatch: {fixture}")
        for field in ("generator", "consumer"):
            target = REPO_ROOT / entry[field]
            if not target.is_file():
                raise RuntimeError(
                    f"Core fixture {entry['path']} has missing {field}: {target}"
                )
    return len(entries)


def validate_imgproc_fixtures() -> int:
    manifest = load_json("test/imgproc/data/manifest.json")
    manifest_path = "test/imgproc/data/manifest.json"
    assert_relative_strings(manifest, manifest_path)

    for entry in manifest["fixtures"]:
        fixture = REPO_ROOT / entry["snapshot_file"]
        if not fixture.is_file():
            raise RuntimeError(f"Missing imgproc fixture: {fixture}")
        if fixture.stat().st_size != entry["size"]:
            raise RuntimeError(f"Imgproc fixture size mismatch: {fixture}")
        if sha256(fixture) != entry["sha256"]:
            raise RuntimeError(f"Imgproc fixture hash mismatch: {fixture}")
        for consumer in entry.get("consumers", []):
            if not (REPO_ROOT / consumer).is_file():
                raise RuntimeError(
                    f"Imgproc fixture has missing consumer: {consumer}"
                )
    return len(manifest["fixtures"])


def validate_upstream_manifest(relative: str) -> int:
    manifest = load_json(relative)
    assert_relative_strings(manifest, relative)
    allowed_statuses = {"PASS", "PENDING", "OUT_OF_SCOPE", "REPLACED"}
    seen_ids: set[str] = set()
    for case in manifest["cases"]:
        case_id = case["id"]
        if case_id in seen_ids:
            raise RuntimeError(f"Duplicate upstream case id in {relative}: {case_id}")
        seen_ids.add(case_id)
        if case["status"] not in allowed_statuses:
            raise RuntimeError(
                f"Unsupported upstream status in {relative}: "
                f"{case_id}={case['status']}"
            )
        snapshot = REPO_ROOT / case["snapshot_file"]
        if not snapshot.is_file():
            raise RuntimeError(f"Missing upstream snapshot: {snapshot}")
        local_test = case.get("local_test")
        if case["status"] == "PASS":
            if not local_test or not (REPO_ROOT / local_test).is_file():
                raise RuntimeError(
                    f"PASS upstream case has no active local test: {case_id}"
                )
    return len(manifest["cases"])


def main() -> None:
    core_count = validate_core_fixtures()
    imgproc_count = validate_imgproc_fixtures()
    core_upstream_count = validate_upstream_manifest(
        "test/upstream/opencv/core/channel_manifest.json"
    )
    imgproc_upstream_count = validate_upstream_manifest(
        "test/upstream/opencv/imgproc/case_manifest.json"
    )
    print(
        "fixture/upstream manifests valid: "
        f"core={core_count}, imgproc={imgproc_count}, "
        f"core_upstream={core_upstream_count}, "
        f"imgproc_upstream={imgproc_upstream_count}"
    )


if __name__ == "__main__":
    main()
