#!/usr/bin/env python3

"""Validate the maintained cvh documentation contract."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
DOC_INDEX = ROOT / "doc" / "README.md"
RESULTS_DIR = ROOT / "benchmark" / "opencv_compare" / "results"

STALE_IDENTIFIERS = (
    "cvh::headers_fast",
    "CVH_ENABLE_" + "OPENCV_INTRIN",
    "CVH_BUILD_NATIVE_BACKEND",
    "CVH_BUILD_FULL_BACKEND",
    "CVH_USE_OPENMP",
    "CVH_" + "LITE",
    "CVH_" + "NATIVE",
    "CVH_" + "FULL",
    "gemm_native",
    "native_intrinsics",
    "ci_native_all",
    "ci_full_all",
    "ci_lite_all",
)

INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
LATEST_REPORT_RE = re.compile(
    r"\*\*Latest performance report:\*\*[^\n]*\]\(([^)]+\.en\.md)\)"
)
LOCAL_USER_PATH_RE = re.compile(r"/Users/[^/\s]+/")


def maintained_markdown() -> list[Path]:
    files = []
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0].startswith("build-"):
            continue
        if relative.parts[:3] == ("benchmark", "opencv_compare", "results"):
            continue
        if relative.parts[:2] == ("test", "3rdparty"):
            continue
        files.append(path)
    return sorted(files)


def normalize_link_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    elif " " in target:
        target = target.split(" ", 1)[0]

    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None

    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target or target.startswith("/"):
        return None
    return target


def check_links(files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        raw_targets = INLINE_LINK_RE.findall(text) + REFERENCE_LINK_RE.findall(text)
        for raw_target in raw_targets:
            target = normalize_link_target(raw_target)
            if target is None:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"broken link: {path.relative_to(ROOT)} -> {raw_target}"
                )
    return errors


def check_doc_index() -> list[str]:
    index = DOC_INDEX.read_text(encoding="utf-8")
    errors = []
    for path in sorted((ROOT / "doc").glob("*.md")):
        if path == DOC_INDEX:
            continue
        if f"]({path.name})" not in index:
            errors.append(f"doc index missing: {path.relative_to(ROOT)}")
    return errors


def check_current_vocabulary(files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for identifier in STALE_IDENTIFIERS:
            if identifier in text:
                errors.append(
                    f"stale identifier {identifier!r}: {path.relative_to(ROOT)}"
                )
        if LOCAL_USER_PATH_RE.search(text):
            errors.append(f"local absolute path: {path.relative_to(ROOT)}")
    return errors


def check_current_report() -> list[str]:
    errors = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = LATEST_REPORT_RE.search(readme)
    if not match:
        return ["README.md does not link an English latest performance report"]

    report = (ROOT / match.group(1)).resolve()
    if not report.is_file():
        return [f"latest performance report does not exist: {match.group(1)}"]
    if report.parent != RESULTS_DIR.resolve():
        errors.append("latest performance report is outside the result archive")

    report_text = report.read_text(encoding="utf-8")
    if "`cvh_auto`" not in report_text and "`cvh_ui`" not in report_text:
        errors.append("latest performance report does not identify cvh_auto or cvh_ui")

    name = report.name.removesuffix(".en.md")
    csv_path = report.with_name(f"{name}.csv")
    meta_path = report.with_name(f"{name}.meta.json")
    if not csv_path.is_file():
        errors.append(f"latest performance CSV is missing: {csv_path.name}")
    if not meta_path.is_file():
        errors.append(f"latest performance metadata is missing: {meta_path.name}")

    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if not rows:
            errors.append("latest performance CSV has no rows")
        required_route_fields = {"algorithm_path", "dispatch_path", "isa_observed"}
        missing_route_fields = sorted(
            required_route_fields - set(reader.fieldnames or [])
        )
        if missing_route_fields:
            errors.append(
                "latest performance CSV is missing dispatch route fields: "
                + ", ".join(missing_route_fields)
            )
        for row_index, row in enumerate(rows, start=2):
            empty_route_fields = sorted(
                field for field in required_route_fields if not row.get(field, "")
            )
            if empty_route_fields:
                errors.append(
                    f"latest performance CSV row {row_index} has empty dispatch "
                    "route fields: " + ", ".join(empty_route_fields)
                )
                break
        report_impls = {row.get("impl", "") for row in rows}
        if report_impls not in ({"cvh_auto"}, {"cvh_ui"}):
            errors.append(
                "latest performance CSV must contain exactly one accepted "
                "product-report implementation: "
                + ", ".join(sorted(report_impls))
            )
    return errors


def main() -> int:
    files = maintained_markdown()
    errors = []
    errors.extend(check_links(files))
    errors.extend(check_doc_index())
    errors.extend(check_current_vocabulary(files))
    errors.extend(check_current_report())

    if errors:
        print("Documentation contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Documentation contract check passed: "
        f"{len(files)} maintained Markdown files; current report uses an accepted product mode."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
