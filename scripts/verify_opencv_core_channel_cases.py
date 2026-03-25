#!/usr/bin/env python3
"""
Verify channel test snapshots against hashes recorded in channel_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Dict, Sequence


TEST_RE = re.compile(
    r"^\s*TEST(?:_P)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)

ALLOWED_STATUS = {"PASS_NOW", "PENDING_CHANNEL"}
REQUIRED_MIN_CASES_PER_SOURCE: Dict[str, int] = {
    "modules/core/test/test_mat.cpp": 1,
    "modules/core/test/test_arithm.cpp": 1,
    "modules/core/test/test_operations.cpp": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify upstream channel case snapshots.")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument(
        "--min-pass-now",
        type=int,
        default=int(os.environ.get("CVH_MIN_PASS_NOW", "0")),
        help="Require at least N PASS_NOW cases in manifest (default from CVH_MIN_PASS_NOW, fallback 0)",
    )
    return parser.parse_args()


def find_case_block(lines: Sequence[str], suite: str, name: str) -> str:
    for idx, line in enumerate(lines):
        m = TEST_RE.match(line)
        if not m:
            continue
        if m.group(1) != suite or m.group(2) != name:
            continue

        start = idx
        open_found = False
        brace_depth = 0
        end = -1
        for j in range(idx, len(lines)):
            text = lines[j]
            for ch in text:
                if ch == "{":
                    brace_depth += 1
                    open_found = True
                elif ch == "}":
                    brace_depth -= 1
                    if open_found and brace_depth == 0:
                        end = j
                        break
            if end != -1:
                break

        if end == -1:
            raise RuntimeError(f"Cannot close TEST block for {suite}.{name}")
        return "".join(lines[start : end + 1])

    raise RuntimeError(f"TEST block not found for {suite}.{name}")


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(args.repo_root).resolve()
    manifest_path = repo_root / "test" / "upstream" / "opencv" / "core" / "channel_manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    source_case_counts = {k: 0 for k in REQUIRED_MIN_CASES_PER_SOURCE}
    pass_now_count = 0
    pending_count = 0

    for case in manifest.get("cases", []):
        source = case.get("source_file", "")
        if source in source_case_counts:
            source_case_counts[source] += 1

        status = case.get("status", "")
        if status not in ALLOWED_STATUS:
            failures.append(
                f"invalid status for {source}:{case.get('suite')}.{case.get('name')} -> {status}"
            )

        if status == "PASS_NOW":
            pass_now_count += 1
        elif status == "PENDING_CHANNEL":
            pending_count += 1
            reason = str(case.get("reason", "")).strip()
            unblock_by = str(case.get("unblock_by", "")).strip()
            if not reason:
                failures.append(
                    f"missing pending reason for {source}:{case.get('suite')}.{case.get('name')}"
                )
            if not unblock_by:
                failures.append(
                    f"missing pending unblock_by for {source}:{case.get('suite')}.{case.get('name')}"
                )

        snap_path = repo_root / case["snapshot_file"]
        if not snap_path.exists():
            failures.append(f"missing snapshot: {snap_path}")
            continue

        lines = snap_path.read_text(encoding="utf-8").splitlines(keepends=True)
        try:
            block = find_case_block(lines, case["suite"], case["name"])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures.append(f"{snap_path}: {case['suite']}.{case['name']} parse error: {exc}")
            continue

        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        if digest != case["sha256"]:
            failures.append(
                f"{snap_path}: {case['suite']}.{case['name']} hash mismatch "
                f"(expected {case['sha256']}, got {digest})"
            )

    for source, required_min in REQUIRED_MIN_CASES_PER_SOURCE.items():
        found = source_case_counts.get(source, 0)
        if found < required_min:
            failures.append(
                f"coverage policy violation: {source} has {found} tracked cases, require >= {required_min}"
            )

    if pass_now_count < args.min_pass_now:
        failures.append(
            f"PASS_NOW policy violation: found {pass_now_count}, require >= {args.min_pass_now}"
        )

    if failures:
        print("verification failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "verified "
        f"{len(manifest.get('cases', []))} upstream channel cases "
        f"(PASS_NOW={pass_now_count}, PENDING_CHANNEL={pending_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
