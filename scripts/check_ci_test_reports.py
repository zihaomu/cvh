#!/usr/bin/env python3

"""Validate the CI test inventory and GTest execution counts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86-64": "x86_64",
        "x86_64": "x86_64",
    }
    return aliases.get(normalized, normalized)


def read_cache_value(build_dir: Path, name: str) -> Optional[str]:
    prefix = f"{name}:"
    for line in (build_dir / "CMakeCache.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    return None


def read_ctest_inventory(build_dir: Path) -> tuple[list[str], list[dict]]:
    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--show-only=json-v1",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)
    tests = payload.get("tests", [])
    return sorted(test["name"] for test in tests), tests


def read_gtest_report(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    test_cases = root.findall(".//testcase")

    tests = int(root.attrib.get("tests", len(test_cases)))
    failures = int(root.attrib.get("failures", 0))
    errors = int(root.attrib.get("errors", 0))
    disabled = int(root.attrib.get("disabled", 0))
    skipped = int(
        root.attrib.get(
            "skipped",
            sum(1 for case in test_cases if case.find("skipped") is not None),
        )
    )
    return {
        "executed": tests - disabled,
        "failed": failures + errors,
        "skipped": skipped,
    }


def validate_mapping(
    label: str, actual: dict[str, int], expected: dict[str, int]
) -> list[str]:
    failures = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            failures.append(
                f"{label}.{key}: expected {expected_value}, got {actual_value}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--profile", choices=("ui-on", "ui-off"), required=True)
    parser.add_argument("--architecture", default=platform.machine())
    parser.add_argument("--core-report", type=Path, required=True)
    parser.add_argument("--imgproc-report", type=Path, required=True)
    args = parser.parse_args()

    architecture = normalize_architecture(args.architecture)
    expectations = json.loads(
        args.expectations.read_text(encoding="utf-8")
    )

    try:
        expected = expectations["profiles"][args.profile][architecture]
    except KeyError:
        print(
            "No CI expectation for "
            f"profile={args.profile}, architecture={architecture}",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    expected_option = "ON" if args.profile == "ui-on" else "OFF"
    actual_option = read_cache_value(
        args.build_dir, "CVH_ENABLE_OPENCV_INTRIN"
    )
    if actual_option != expected_option:
        failures.append(
            "CVH_ENABLE_OPENCV_INTRIN: "
            f"expected {expected_option}, got {actual_option}"
        )

    actual_ctest_names, ctest_tests = read_ctest_inventory(args.build_dir)
    expected_ctest_names = sorted(expected["ctest"])
    if actual_ctest_names != expected_ctest_names:
        missing = sorted(set(expected_ctest_names) - set(actual_ctest_names))
        unexpected = sorted(set(actual_ctest_names) - set(expected_ctest_names))
        failures.append(
            "CTest inventory mismatch: "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )

    avx2_decision = "not-applicable"
    if args.profile == "ui-on" and architecture == "x86_64":
        avx2_value = read_cache_value(
            args.build_dir, "CVH_X86_AVX2_RUNTIME_SUPPORTED"
        )
        if avx2_value is None:
            failures.append("runtime AVX2 decision is missing from CMakeCache.txt")
            avx2_decision = "missing"
        else:
            avx2_supported = avx2_value.upper() in ("1", "ON", "TRUE", "YES")
            avx2_decision = "run" if avx2_supported else "compile-only"
            x86_test = next(
                (
                    test
                    for test in ctest_tests
                    if test["name"] == "cvh_opencv_intrin_x86_smoke"
                ),
                None,
            )
            command = x86_test.get("command", []) if x86_test else []
            command_text = " ".join(command)
            if avx2_supported:
                if not any(
                    Path(part).name == "cvh_opencv_intrin_x86_smoke"
                    for part in command
                ):
                    failures.append(
                        "AVX2-capable host does not run the x86 UI smoke"
                    )
            elif "compile-only" not in command_text:
                failures.append(
                    "AVX2-incapable host lacks an explicit compile-only result"
                )

    core_stats = read_gtest_report(args.core_report)
    imgproc_stats = read_gtest_report(args.imgproc_report)
    failures.extend(
        validate_mapping("core", core_stats, expected["gtest"]["core"])
    )
    failures.extend(
        validate_mapping(
            "imgproc", imgproc_stats, expected["gtest"]["imgproc"]
        )
    )

    print("ci_test_report_begin")
    print(f"profile: {args.profile}")
    print(f"architecture: {architecture}")
    print(f"opencv_intrin: {actual_option}")
    print(f"runtime_avx2: {avx2_decision}")
    print(f"ctest_registered: {len(actual_ctest_names)}")
    print(
        "core: "
        f"executed={core_stats['executed']} "
        f"failed={core_stats['failed']} "
        f"skipped={core_stats['skipped']}"
    )
    print(
        "imgproc: "
        f"executed={imgproc_stats['executed']} "
        f"failed={imgproc_stats['failed']} "
        f"skipped={imgproc_stats['skipped']}"
    )
    print("ci_test_report_end")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
