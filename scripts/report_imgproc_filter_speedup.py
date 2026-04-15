#!/usr/bin/env python3
"""
Report speedup between two imgproc filter benchmark CSV files.

Expected columns:
op,kernel,depth,channels,shape,layout,border,dispatch_path,ms_per_iter
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


BenchKey = Tuple[str, str, str, str, str, str, str]


@dataclass
class BenchRow:
    key: BenchKey
    dispatch_path: str
    ms_per_iter: float


@dataclass
class SpeedupRow:
    key: BenchKey
    baseline_ms: float
    candidate_ms: float
    speedup: float
    baseline_dispatch: str
    candidate_dispatch: str


def load_csv(path: pathlib.Path) -> Dict[BenchKey, BenchRow]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    rows: Dict[BenchKey, BenchRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["op"].strip(),
                row["kernel"].strip(),
                row["depth"].strip(),
                row["channels"].strip(),
                row["shape"].strip(),
                row["layout"].strip(),
                row["border"].strip(),
            )
            if key in rows:
                raise ValueError(f"duplicate case in CSV: {key}")
            rows[key] = BenchRow(
                key=key,
                dispatch_path=row["dispatch_path"].strip(),
                ms_per_iter=float(row["ms_per_iter"]),
            )
    return rows


def format_key(key: BenchKey) -> str:
    op, kernel, depth, channels, shape, layout, border = key
    return (
        f"{op:<14} {kernel:<4} {depth:<7} c={channels:<2} "
        f"shape={shape:<10} layout={layout:<17} border={border}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Report imgproc filter benchmark speedup")
    parser.add_argument("--baseline", required=True, type=pathlib.Path, help="baseline CSV (typically fallback)")
    parser.add_argument("--candidate", required=True, type=pathlib.Path, help="candidate CSV (typically optimized)")
    parser.add_argument("--top", type=int, default=12, help="how many fastest/slowest cases to print")
    parser.add_argument(
        "--fail-below-speedup",
        type=float,
        default=1.0,
        help="fail if any compared case speedup is below this value (default: 1.0)",
    )
    args = parser.parse_args()

    baseline_rows = load_csv(args.baseline)
    candidate_rows = load_csv(args.candidate)

    speedups: List[SpeedupRow] = []
    missing_in_candidate: List[BenchKey] = []
    for key, base in baseline_rows.items():
        cand = candidate_rows.get(key)
        if cand is None:
            missing_in_candidate.append(key)
            continue
        if base.ms_per_iter <= 0.0 or cand.ms_per_iter <= 0.0:
            continue
        speedups.append(
            SpeedupRow(
                key=key,
                baseline_ms=base.ms_per_iter,
                candidate_ms=cand.ms_per_iter,
                speedup=base.ms_per_iter / cand.ms_per_iter,
                baseline_dispatch=base.dispatch_path,
                candidate_dispatch=cand.dispatch_path,
            )
        )

    missing_in_baseline = [key for key in candidate_rows.keys() if key not in baseline_rows]

    if missing_in_candidate:
        print("[filter-speedup] missing cases in candidate:")
        for key in missing_in_candidate[:20]:
            print("  -", format_key(key))
        if len(missing_in_candidate) > 20:
            print(f"  ... and {len(missing_in_candidate) - 20} more")

    if missing_in_baseline:
        print("[filter-speedup] extra cases in candidate (not in baseline):")
        for key in missing_in_baseline[:20]:
            print("  -", format_key(key))
        if len(missing_in_baseline) > 20:
            print(f"  ... and {len(missing_in_baseline) - 20} more")

    if not speedups:
        print("[filter-speedup] no comparable cases found")
        return 2

    speedup_values = [row.speedup for row in speedups]
    geomean_speedup = math.exp(sum(math.log(v) for v in speedup_values) / len(speedup_values))
    median_speedup = statistics.median(speedup_values)
    min_speedup = min(speedup_values)
    max_speedup = max(speedup_values)
    improved = sum(1 for v in speedup_values if v > 1.0)
    regressed = sum(1 for v in speedup_values if v < 1.0)

    print(
        "[filter-speedup] "
        f"compared={len(speedups)}, improved={improved}, regressed={regressed}, "
        f"geomean={geomean_speedup:.3f}x, median={median_speedup:.3f}x, "
        f"min={min_speedup:.3f}x, max={max_speedup:.3f}x"
    )

    speedups_sorted = sorted(speedups, key=lambda r: r.speedup, reverse=True)
    top_n = max(1, args.top)

    print(f"[filter-speedup] top {min(top_n, len(speedups_sorted))} fastest cases:")
    for row in speedups_sorted[:top_n]:
        print(
            f"  + {format_key(row.key)} | speedup={row.speedup:.3f}x "
            f"| baseline={row.baseline_ms:.6f}ms ({row.baseline_dispatch}) "
            f"| candidate={row.candidate_ms:.6f}ms ({row.candidate_dispatch})"
        )

    print(f"[filter-speedup] top {min(top_n, len(speedups_sorted))} slowest cases:")
    for row in speedups_sorted[-top_n:]:
        print(
            f"  - {format_key(row.key)} | speedup={row.speedup:.3f}x "
            f"| baseline={row.baseline_ms:.6f}ms ({row.baseline_dispatch}) "
            f"| candidate={row.candidate_ms:.6f}ms ({row.candidate_dispatch})"
        )

    below_threshold = [row for row in speedups if row.speedup < args.fail_below_speedup]
    if below_threshold:
        below_threshold.sort(key=lambda r: r.speedup)
        print(
            f"[filter-speedup] FAIL: {len(below_threshold)} case(s) below threshold "
            f"{args.fail_below_speedup:.3f}x"
        )
        for row in below_threshold[:20]:
            print(f"  * {format_key(row.key)} | speedup={row.speedup:.3f}x")
        if len(below_threshold) > 20:
            print(f"  ... and {len(below_threshold) - 20} more")
        return 1

    print("[filter-speedup] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
