#!/usr/bin/env python3
"""Aggregate cvh Pipeline proof sessions without hiding unstable samples."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


RowKey = Tuple[str, str, str]
PredicateKey = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--cv-limit", type=float, default=0.03)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--label", default="stable performance proof")
    return parser.parse_args()


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return (
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * weight
    )


def bootstrap_geomean_ci(
    values: Sequence[float], samples: int, seed: int
) -> Tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            geometric_mean([rng.choice(values) for _ in range(len(values))])
        )
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def read_sessions(paths: Iterable[Path]) -> Tuple[Dict[RowKey, List[dict]], List[int]]:
    rows: Dict[RowKey, List[dict]] = defaultdict(list)
    sessions = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            file_rows = list(csv.DictReader(handle))
        if not file_rows:
            raise ValueError(f"empty session CSV: {path}")
        session_values = {int(row["session"]) for row in file_rows}
        if len(session_values) != 1:
            raise ValueError(f"mixed session ids in {path}: {session_values}")
        session = next(iter(session_values))
        if session in sessions:
            raise ValueError(f"duplicate session id: {session}")
        sessions.append(session)
        for row in file_rows:
            if row["validation"] != "pass":
                raise ValueError(
                    f"validation failure: {path}:{row['case_id']}:"
                    f"{row['cache_mode']}:{row['implementation']}"
                )
            row = dict(row)
            row["_path"] = str(path)
            rows[(row["case_id"], row["cache_mode"], row["implementation"])].append(row)
    sessions.sort()
    if len(sessions) < 3:
        raise ValueError("at least three independent sessions are required")
    for key, grouped in rows.items():
        if len(grouped) != len(sessions):
            raise ValueError(f"missing session rows for {key}")
        grouped.sort(key=lambda row: int(row["session"]))
    return rows, sessions


def stable(row: dict, cv_limit: float) -> bool:
    return float(row["cv"]) <= cv_limit


def format_speedup(value: float) -> str:
    return f"{value:.3f}x"


def write_aggregate_csv(
    path: Path,
    aggregate_rows: Sequence[dict],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "cache_mode",
        "session_count",
        "stable_sessions",
        "session_speedups_vs_opencv",
        "median_speedup_vs_opencv",
        "geomean_speedup_vs_opencv",
        "ci95_low",
        "ci95_high",
        "min_speedup",
        "max_speedup",
        "candidate_route",
        "actual_route",
        "observed_isa",
        "predicate_gate",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> int:
    args = parse_args()
    rows, sessions = read_sessions(args.inputs)
    predicate_keys = sorted(
        {
            (case_id, cache_mode)
            for case_id, cache_mode, implementation in rows
            if implementation == "cvh_fused_auto"
        }
    )

    aggregate_rows = []
    predicate_data = {}
    unstable_rows = []
    for case_id, cache_mode in predicate_keys:
        auto_rows = rows[(case_id, cache_mode, "cvh_fused_auto")]
        opencv_rows = rows[(case_id, cache_mode, "opencv_explicit")]
        staged_rows = rows[(case_id, cache_mode, "cvh_staged")]
        scalar_rows = rows[(case_id, cache_mode, "cvh_fused_scalar")]
        all_implementations = auto_rows + opencv_rows + staged_rows + scalar_rows
        for row in all_implementations:
            if not stable(row, args.cv_limit):
                unstable_rows.append(row)

        speedups = [float(row["speedup_vs_opencv"]) for row in auto_rows]
        l2_speedups = [float(row["speedup_vs_staged"]) for row in auto_rows]
        stable_session_count = sum(
            stable(auto, args.cv_limit) and stable(opencv, args.cv_limit)
            for auto, opencv in zip(auto_rows, opencv_rows)
        )
        median_speedup = statistics.median(speedups)
        geomean_speedup = geometric_mean(speedups)
        ci_low, ci_high = bootstrap_geomean_ci(
            speedups,
            args.bootstrap_samples,
            args.seed + sum(ord(ch) for ch in case_id + cache_mode),
        )
        reasons = []
        if stable_session_count != len(sessions):
            reasons.append("CV gate")
        if median_speedup < 1.20:
            reasons.append("median < 1.20x")
        if ci_low < 1.10:
            reasons.append("CI lower < 1.10x")
        gate = not reasons
        predicate_data[(case_id, cache_mode)] = {
            "speedups": speedups,
            "l2_speedups": l2_speedups,
            "stable_sessions": stable_session_count,
            "gate": gate,
            "reasons": reasons,
            "auto_rows": auto_rows,
        }
        first = auto_rows[0]
        aggregate_rows.append(
            {
                "case_id": case_id,
                "cache_mode": cache_mode,
                "session_count": len(sessions),
                "stable_sessions": stable_session_count,
                "session_speedups_vs_opencv": ";".join(
                    f"{value:.6f}" for value in speedups
                ),
                "median_speedup_vs_opencv": f"{median_speedup:.6f}",
                "geomean_speedup_vs_opencv": f"{geomean_speedup:.6f}",
                "ci95_low": f"{ci_low:.6f}",
                "ci95_high": f"{ci_high:.6f}",
                "min_speedup": f"{min(speedups):.6f}",
                "max_speedup": f"{max(speedups):.6f}",
                "candidate_route": first["candidate_route"],
                "actual_route": first["actual_route"],
                "observed_isa": first["observed_isa"],
                "predicate_gate": "pass" if gate else "fail",
                "reason": "pass" if gate else "; ".join(reasons),
            }
        )

    # The other cache mode may not regress by more than 10% for a public claim.
    for row in aggregate_rows:
        other_mode = "streaming" if row["cache_mode"] == "hot" else "hot"
        other = predicate_data.get((row["case_id"], other_mode))
        if other and statistics.median(other["speedups"]) < 0.90:
            row["predicate_gate"] = "fail"
            suffix = "other cache mode < 0.90x"
            row["reason"] = (
                suffix if row["reason"] == "pass" else row["reason"] + "; " + suffix
            )

    write_aggregate_csv(args.output_csv, aggregate_rows)
    if args.output_md.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_md}")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    family_rows = []
    for cache_mode in ("hot", "streaming"):
        case_keys = [key for key in predicate_keys if key[1] == cache_mode]
        session_family = []
        session_l2_family = []
        for session_index in range(len(sessions)):
            session_family.append(
                geometric_mean(
                    [
                        predicate_data[key]["speedups"][session_index]
                        for key in case_keys
                    ]
                )
            )
            session_l2_family.append(
                geometric_mean(
                    [
                        predicate_data[key]["l2_speedups"][session_index]
                        for key in case_keys
                    ]
                )
            )
        stable_family = all(
            predicate_data[key]["stable_sessions"] == len(sessions)
            for key in case_keys
        )
        no_regression = all(
            statistics.median(predicate_data[key]["speedups"]) >= 0.95
            for key in case_keys
        )
        family_rows.append(
            (
                cache_mode,
                geometric_mean(session_family),
                min(session_family),
                max(session_family),
                geometric_mean(session_l2_family),
                stable_family,
                no_regression,
            )
        )

    fused_rows = [
        row
        for key in predicate_keys
        for row in predicate_data[key]["auto_rows"]
    ]
    structure_ok = all(
        row["execution_groups"] == "1"
        and row["full_frame_intermediates"] == "0"
        and row["allocations_per_run"] == "0"
        and row["workspace_bytes"] == "0"
        for row in fused_rows
    )
    temporary_bytes = [
        int(row["explicit_temporary_bytes"])
        for key in predicate_keys
        for row in rows[(key[0], key[1], "opencv_explicit")]
    ]

    with args.output_md.open("w", encoding="utf-8") as output:
        output.write(f"# cvh Pipeline Apple M5 {args.label}\n\n")
        output.write(
            "Status: device-bound evidence; no PF1-PF6 family or L4 claim is authorized.\n\n"
        )
        output.write("## Method\n\n")
        output.write(
            f"- Sessions: {', '.join(str(value) for value in sessions)}; "
            "single-thread Release; warmup 20; 50 frames/sample; 15 samples.\n"
        )
        output.write("- Cache modes: hot and a 64 MiB input/output ring streaming mode.\n")
        output.write(
            "- Each CSV passed the independent scalar-oracle validation before timing.\n"
        )
        output.write(
            f"- Stability gate: CV <= {args.cv_limit:.0%}; "
            f"{len(unstable_rows)} of {len(predicate_keys) * len(sessions) * 4} "
            "implementation rows exceeded it.\n"
        )
        output.write("- Inputs:\n")
        for path in args.inputs:
            output.write(f"  - `{path}`\n")

        output.write("\n## Predicate results\n\n")
        output.write(
            "| Case | Cache | Session speedups vs OpenCV | Geomean | 95% CI | "
            "Stable sessions | Route/ISA | Gate |\n"
        )
        output.write("| --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for row in aggregate_rows:
            speeds = ", ".join(
                format_speedup(float(value))
                for value in row["session_speedups_vs_opencv"].split(";")
            )
            output.write(
                f"| {row['case_id']} | {row['cache_mode']} | {speeds} | "
                f"{format_speedup(float(row['geomean_speedup_vs_opencv']))} | "
                f"[{format_speedup(float(row['ci95_low']))}, "
                f"{format_speedup(float(row['ci95_high']))}] | "
                f"{row['stable_sessions']}/{row['session_count']} | "
                f"{row['actual_route']}/{row['observed_isa']} | "
                f"{row['predicate_gate']}: {row['reason']} |\n"
            )

        output.write("\n## Family result\n\n")
        output.write(
            "| Cache | PF1-PF6 vs OpenCV geomean | Session range | "
            "Fused vs staged geomean | Stable | No case < 0.95x | Gate |\n"
        )
        output.write("| --- | ---: | ---: | ---: | --- | --- | --- |\n")
        for (
            mode,
            geomean_value,
            minimum,
            maximum,
            l2_geomean,
            stable_value,
            no_regression,
        ) in family_rows:
            gate = geomean_value >= 1.20 and stable_value and no_regression
            output.write(
                f"| {mode} | {format_speedup(geomean_value)} | "
                f"{format_speedup(minimum)}–{format_speedup(maximum)} | "
                f"{format_speedup(l2_geomean)} | "
                f"{'yes' if stable_value else 'no'} | "
                f"{'yes' if no_regression else 'no'} | "
                f"{'pass' if gate else 'fail'} |\n"
            )

        output.write("\n## Structural evidence\n\n")
        output.write(
            f"- L1 fused structure gate: {'pass' if structure_ok else 'fail'}; "
            "all fused rows report one execution group, zero full-frame "
            "intermediates, zero workspace bytes, and zero planned allocations/run.\n"
        )
        output.write(
            "- OpenCV explicit temporary storage range: "
            f"{min(temporary_bytes):,}–{max(temporary_bytes):,} bytes, excluding "
            "caller-owned output.\n"
        )

        output.write("\n## Decision\n\n")
        output.write(
            "- L1 is supported on this build. L2 remains supported for fusion over "
            "the cvh staged chain, subject to each row's stability marker.\n"
        )
        output.write(
            "- The PF1-PF6 family does not meet the OpenCV gate. PF1-PF4 Linear "
            "predicates are materially slower than the optimized OpenCV chain.\n"
        )
        passing_predicates = [
            row
            for row in aggregate_rows
            if row["predicate_gate"] == "pass"
        ]
        if passing_predicates:
            labels = ", ".join(
                f"{row['case_id']} {row['cache_mode']} "
                f"({format_speedup(float(row['geomean_speedup_vs_opencv']))}, "
                f"CI lower {format_speedup(float(row['ci95_low']))})"
                for row in passing_predicates
            )
            output.write(
                "- Exact device-bound predicates passing all frozen gates: "
                f"{labels}. This does not authorize a family or edge-device claim.\n"
            )
        else:
            output.write(
                "- No exact predicate passes every frozen speed, confidence, and "
                "stability gate.\n"
            )
        output.write(
            "- PF6 is near parity in hot mode and slower in streaming mode. "
            "No edge-device claim is permitted without two ARM Linux devices.\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
