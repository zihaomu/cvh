#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.3.5"]
# ///
"""Generate every versioned core NumPy fixture from deterministic inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[1]
NPY_ROOT = DATA_ROOT / "npy"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
GENERATOR_PATH = "test/core/data/generators/generate_fixtures.py"


def make_case(
    name: str,
    batch_a: tuple[int, ...],
    batch_b: tuple[int, ...],
    m: int,
    k: int,
    n: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    scale: float = 1.0,
) -> dict[str, object]:
    return {
        "name": name,
        "batch_a": batch_a,
        "batch_b": batch_b,
        "m": m,
        "k": k,
        "n": n,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "scale": scale,
    }


GEMM_CASES = [
    make_case("nn_small_odd", (), (), 3, 5, 4),
    make_case("nn_tail_rect", (), (), 7, 11, 13),
    make_case("nn_rank3", (2,), (2,), 4, 6, 5),
    make_case("nn_rank4", (2, 3), (2, 3), 3, 7, 4),
    make_case("nn_broadcast_a", (1, 3), (2, 3), 5, 9, 4),
    make_case("nn_broadcast_b", (2, 1), (2, 4), 6, 8, 3),
    make_case("nn_rank_mismatch_a", (3,), (2, 3), 4, 5, 6),
    make_case("nn_rank_mismatch_b", (2, 3), (3,), 4, 5, 6),
    make_case("nt_small_odd", (), (), 3, 5, 4, False, True),
    make_case("nt_tail_rect", (), (), 9, 15, 7, False, True),
    make_case("nt_rank3", (2,), (2,), 5, 7, 6, False, True),
    make_case("nt_broadcast_a", (1, 3), (2, 3), 4, 9, 5, False, True),
    make_case("nt_broadcast_b", (2, 1), (2, 4), 4, 8, 5, False, True),
    make_case("nt_rank_mismatch", (3,), (2, 3), 6, 10, 4, False, True),
    make_case("tn_basic", (), (), 5, 7, 4, True, False),
    make_case("tn_rank3", (2,), (2,), 4, 6, 5, True, False),
    make_case("tn_broadcast", (1, 3), (2, 3), 4, 6, 5, True, False),
    make_case("tt_basic", (), (), 3, 5, 4, True, True),
    make_case("tt_rank3", (2,), (2,), 4, 9, 6, True, True),
    make_case("tt_broadcast", (2, 1), (1, 3), 5, 7, 4, True, True),
    make_case("nn_large_value", (), (), 8, 16, 9, False, False, 100.0),
    make_case("nt_small_value", (), (), 8, 16, 9, False, True, 1e-3),
]


def save(name: str, array: np.ndarray) -> None:
    np.save(
        NPY_ROOT / name,
        np.array(array, dtype=np.float32, order="C", copy=True),
    )


def generate_npy_reader_fixture() -> list[str]:
    values = np.random.RandomState(0).rand(10, 12).astype(np.float32)
    name = "random10x12.npy"
    save(name, values)
    return [name]


def generate_transpose_fixtures() -> list[str]:
    rng = np.random.default_rng(20260306)
    generated: list[str] = []

    source_3d = rng.standard_normal((2, 5, 7), dtype=np.float32)
    save("transpose_last2_3d_i.npy", source_3d)
    save("transpose_last2_3d_o.npy", np.swapaxes(source_3d, -1, -2))
    generated.extend(
        ["transpose_last2_3d_i.npy", "transpose_last2_3d_o.npy"]
    )

    source_4d = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
    save("transpose_perm_4d_i.npy", source_4d)
    save(
        "transpose_perm_4d_o.npy",
        np.transpose(source_4d, (0, 2, 1, 3)),
    )
    generated.extend(
        ["transpose_perm_4d_i.npy", "transpose_perm_4d_o.npy"]
    )
    return generated


def generate_gemm_fixtures() -> tuple[list[str], list[dict[str, object]]]:
    rng = np.random.default_rng(2026)
    generated: list[str] = []
    case_metadata: list[dict[str, object]] = []

    for case in GEMM_CASES:
        batch_a = tuple(case["batch_a"])
        batch_b = tuple(case["batch_b"])
        m = int(case["m"])
        k = int(case["k"])
        n = int(case["n"])
        transpose_a = bool(case["transpose_a"])
        transpose_b = bool(case["transpose_b"])
        scale = np.float32(case["scale"])

        shape_a = batch_a + ((k, m) if transpose_a else (m, k))
        shape_b = batch_b + ((n, k) if transpose_b else (k, n))
        first = rng.standard_normal(shape_a).astype(np.float32) * scale
        second = rng.standard_normal(shape_b).astype(np.float32) * scale
        effective_a = np.swapaxes(first, -1, -2) if transpose_a else first
        effective_b = np.swapaxes(second, -1, -2) if transpose_b else second
        output = np.matmul(effective_a, effective_b)

        prefix = f"gemm_{case['name']}"
        names = [f"{prefix}_a.npy", f"{prefix}_b.npy", f"{prefix}_o.npy"]
        save(names[0], first)
        save(names[1], second)
        save(names[2], output)
        generated.extend(names)
        case_metadata.append(
            {
                "name": case["name"],
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "shape_a": list(shape_a),
                "shape_b": list(shape_b),
                "shape_output": list(output.shape),
            }
        )

    return generated, case_metadata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_entry(
    name: str,
    oracle: str,
    consumer: str,
) -> dict[str, object]:
    path = NPY_ROOT / name
    array = np.load(path, allow_pickle=False)
    return {
        "path": f"npy/{name}",
        "sha256": sha256(path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "generator": GENERATOR_PATH,
        "oracle": oracle,
        "consumer": consumer,
    }


def write_manifest(
    reader_files: list[str],
    transpose_files: list[str],
    gemm_files: list[str],
    gemm_cases: list[dict[str, object]],
) -> None:
    fixtures = [
        fixture_entry(
            name,
            "NumPy RandomState(0) fixed sample",
            "test/core/support/npy_reader_test.cpp",
        )
        for name in reader_files
    ]
    fixtures.extend(
        fixture_entry(
            name,
            "numpy.swapaxes / numpy.transpose",
            "test/core/operations/transpose_fixture_test.cpp",
        )
        for name in transpose_files
    )
    fixtures.extend(
        fixture_entry(
            name,
            "numpy.matmul with explicit last-two-axis transpose",
            "test/core/operations/gemm_fixture_test.cpp",
        )
        for name in gemm_files
    )
    manifest = {
        "schema_version": 1,
        "generated_by": GENERATOR_PATH,
        "fixtures": sorted(fixtures, key=lambda item: item["path"]),
        "gemm_cases": gemm_cases,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    NPY_ROOT.mkdir(parents=True, exist_ok=True)
    reader_files = generate_npy_reader_fixture()
    transpose_files = generate_transpose_fixtures()
    gemm_files, gemm_cases = generate_gemm_fixtures()

    expected = set(reader_files + transpose_files + gemm_files)
    actual = {path.name for path in NPY_ROOT.glob("*.npy")}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise RuntimeError(
            "Unmanaged core fixture(s): " + ", ".join(unexpected)
        )

    write_manifest(reader_files, transpose_files, gemm_files, gemm_cases)
    print(f"Generated {len(expected)} fixtures in {NPY_ROOT}")


if __name__ == "__main__":
    main()
