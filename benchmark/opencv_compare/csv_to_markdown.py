#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render cvh vs OpenCV compare CSV to Markdown")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output Markdown path")
    p.add_argument("--meta", default="", help="Optional metadata JSON path")
    p.add_argument("--title", default="cvh vs OpenCV Benchmark Report", help="Markdown title")
    return p.parse_args()


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def row_suite(row: dict) -> str:
    suite = (row.get("suite", "") or "").strip().lower()
    if suite:
        return suite
    return "core_mat" if (row.get("op", "") or "").startswith("MAT_") else "imgproc"


def geometric_mean(values) -> float:
    positive = [x for x in values if x > 0.0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(x) for x in positive) / len(positive))


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def render_report(rows, title: str, input_path: Path, meta_path: Optional[Path] = None) -> str:
    supported = [r for r in rows if r.get("status", "") == "OK"]
    unsupported = [r for r in rows if r.get("status", "") != "OK"]
    meta = {}
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    speedups = [to_float(r.get("speedup", "0")) for r in supported if to_float(r.get("speedup", "0")) > 0.0]
    cvh_faster = sum(1 for x in speedups if x > 1.0)
    opencv_faster_or_equal = sum(1 for x in speedups if x <= 1.0)
    geo_speedup = geometric_mean(speedups)
    median_speedup = statistics.median(speedups) if speedups else 0.0
    def group_result(op_names) -> str:
        values = [
            to_float(r.get("speedup", "0"))
            for r in supported
            if r.get("op", "") in op_names
            and to_float(r.get("speedup", "0")) > 0.0
        ]
        ratio = geometric_mean(values)
        if ratio <= 0.0:
            return "no valid cases"
        if ratio > 1.0:
            return f"CVH `~{ratio:.2f}x`"
        return f"OpenCV `~{(1.0 / ratio):.2f}x`"

    match_template_has_ui = any(
        r.get("op", "") == "MATCH_TEMPLATE"
        and r.get("dispatch_path", "") == "opencv_ui"
        for r in supported
    )
    histogram_has_fastpath = any(
        r.get("op", "") in {"CALC_HIST", "COMPARE_HIST"}
        and r.get("dispatch_path", "") == "header_fastpath"
        for r in supported
    )
    if match_template_has_ui and histogram_has_fastpath:
        histogram_template_cause = (
            "Template matching uses UI correlation and squared-window integrals; histogram paths use typed scans and method-specialized double reductions"
        )
        histogram_template_follow_up = (
            "Keep histogram/template numeric and dispatch coverage stable, then continue with random fills"
        )
    elif match_template_has_ui:
        histogram_template_cause = (
            "Template matching now uses a method-specialized UI correlation path and squared-window integral; histogram construction and comparison remain scalar"
        )
        histogram_template_follow_up = (
            "Keep template matching correctness and dispatch coverage stable, then prioritize histogram construction"
        )
    else:
        histogram_template_cause = (
            "Histogram and direct-spatial template kernels have no CVH UI fast path in v0.1"
        )
        histogram_template_follow_up = (
            "Prioritize template matching by absolute runtime, then histogram construction"
        )
    point_transform_ratios = [
        to_float(r.get("speedup", "0"))
        for r in supported
        if r.get("op", "") in {"TRANSFORM", "PERSPECTIVE_TRANSFORM"}
    ]
    point_transforms_optimized = (
        point_transform_ratios and min(point_transform_ratios) >= 0.25
    )
    random_has_fastpath = any(
        r.get("op", "") in {"RANDU", "RANDN"}
        and r.get("dispatch_path", "") == "header_fastpath"
        for r in supported
    )
    if point_transforms_optimized and random_has_fastpath:
        random_transform_cause = (
            "Point transforms use prepacked channel-specialized spans; random fills use a lightweight 64-bit engine, hoisted distributions, and typed row kernels"
        )
        random_transform_follow_up = (
            "Keep transform and random statistical/dispatch coverage stable"
        )
    elif point_transforms_optimized:
        random_transform_cause = (
            "Point transforms use prepacked coefficients and channel-specialized continuous spans; random fills remain scalar public-header paths"
        )
        random_transform_follow_up = (
            "Keep point-transform numeric coverage stable and prioritize random-fill loop structure"
        )
    else:
        random_transform_cause = (
            "The v0.1 implementations are scalar public-header paths; upstream uses optimized RNG and transform kernels"
        )
        random_transform_follow_up = (
            "Treat the focused P2-P0 report as optimization prioritization, not a release gate"
        )
    connected_component_ratios = [
        to_float(r.get("speedup", "0"))
        for r in supported
        if r.get("op", "")
        in {"CONNECTED_COMPONENTS", "CONNECTED_COMPONENTS_WITH_STATS"}
    ]
    connected_components_optimized = (
        connected_component_ratios
        and min(connected_component_ratios) >= 0.25
    )
    contour_ratios = [
        to_float(r.get("speedup", "0"))
        for r in supported
        if r.get("op", "") == "FIND_CONTOURS"
    ]
    contours_optimized = contour_ratios and min(contour_ratios) >= 0.25
    if connected_components_optimized and contours_optimized:
        regions_cause = (
            "Connected components use row-pointer union-find and fused statistics; contour discovery uses a mode-specialized row-indexed workspace"
        )
        regions_follow_up = (
            "Keep label/statistics and contour ordering fixed, then continue with point transforms"
        )
    elif connected_components_optimized:
        regions_cause = (
            "Connected components now use row-pointer union-find, vector canonicalization, and fused statistics; contour discovery remains the scan-heavy scalar hotspot"
        )
        regions_follow_up = (
            "Keep label/statistics ordering fixed and prioritize contour workspace and discovery scans"
        )
    else:
        regions_cause = (
            "CVH currently favors explicit scalar correctness and deterministic ordering over specialized scans"
        )
        regions_follow_up = (
            "Separate scan-heavy region work from micro shape primitives when selecting fast paths"
        )

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated at (UTC): `{generated_at}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- `cvh` is a pure header-only library; every CVH row enters through the public `cvh::headers` API.")
    lines.append("- The benchmark forces `OpenCVUIOnly`, and every CVH implementation label must therefore be `cvh_ui`.")
    lines.append("- Direct architecture-specific dispatch is rejected. Operators without a Universal Intrinsics kernel use their normal public-header fallback, whose actual path is recorded in `CVH dispatch`.")
    lines.append("- The reference is the upstream OpenCV build recorded in the metadata, running on the same host with matching inputs and parameters.")
    lines.append("")

    lines.append("## Comparison Model")
    lines.append("")
    lines.append("| Layer | Current Implementation | Meaning in This Report |")
    lines.append("| --- | --- | --- |")
    lines.append("| Public candidate | `cvh::headers` | Built with `OpenCVUIOnly`; implementation label `cvh_ui` |")
    lines.append("| Vector dialect | OpenCV Universal Intrinsics | Portable UI kernels selected by the compiler and intrinsics layer |")
    lines.append("| Public fallback | Header-only scalar or generic fast path | Same product target; actual path recorded by `dispatch_path` |")
    lines.append("| Reference | Upstream OpenCV `core` / `imgproc` | Same input, dimensions, borders, parameters, and thread setting |")
    lines.append("")

    if meta:
        lines.append("## Run Configuration")
        lines.append("")
        lines.append(f"- Profile: `{meta.get('profile', 'unknown')}`")
        meta_impls = meta.get("impls", [])
        if isinstance(meta_impls, list) and meta_impls:
            lines.append(f"- CVH implementation: `{', '.join(meta_impls)}`")
        lines.append(
            f"- Sampling: `warmup={meta.get('warmup', 'n/a')}, iters={meta.get('iters', 'n/a')}, repeats={meta.get('repeats', 'n/a')}`"
        )
        lines.append(f"- Threads: `{meta.get('threads', 'n/a')}`")
        lines.append(
            f"- OpenMP: `dynamic={meta.get('omp_dynamic', 'n/a')}, proc_bind={meta.get('omp_proc_bind', 'n/a')}`"
        )
        lines.append(f"- Host: `{meta.get('system', 'n/a')} {meta.get('arch', 'n/a')}`")
        lines.append(f"- CPU: `{meta.get('cpu_model', 'n/a')}`")
        lines.append(f"- Compiler: `{meta.get('compiler', 'n/a')}`")
        lines.append(f"- Build type: `{meta.get('build_type', 'n/a')}`")
        lines.append(
            f"- CVH commit: `{meta.get('repo_git_commit', 'unknown')}`"
            f"{' + dirty' if meta.get('repo_git_dirty', False) else ''}"
        )
        lines.append(
            f"- OpenCV: `{meta.get('opencv_version', 'unknown')}`, commit "
            f"`{meta.get('opencv_git_commit', 'unknown')}`"
            f"{' + dirty' if meta.get('opencv_git_dirty', False) else ''}"
        )
        lines.append(f"- Raw data: `{input_path.name}`; metadata: `{meta_path.name if meta_path else 'n/a'}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: `{len(rows)}`; valid: `{len(supported)}`; unsupported: `{len(unsupported)}`.")
    lines.append(f"- `OpenCV/CVH` geometric mean: `{geo_speedup:.4f}`; median: `{median_speedup:.4f}`.")
    lines.append(f"- CVH faster: `{cvh_faster}`; OpenCV faster or equal: `{opencv_faster_or_equal}`.")
    lines.append("")

    suite_summary_rows = []
    for suite in ("core_mat", "imgproc"):
        suite_speedups = [
            to_float(r.get("speedup", "0"))
            for r in supported
            if row_suite(r) == suite and to_float(r.get("speedup", "0")) > 0.0
        ]
        if suite_speedups:
            suite_summary_rows.append(
                [
                    suite,
                    str(len(suite_speedups)),
                    f"{geometric_mean(suite_speedups):.4f}",
                    f"{statistics.median(suite_speedups):.4f}",
                    str(sum(1 for x in suite_speedups if x > 1.0)),
                    str(sum(1 for x in suite_speedups if x <= 1.0)),
                ]
            )
    if suite_summary_rows:
        lines.append(
            md_table(
                ["Suite", "Cases", "geometric mean OpenCV/CVH", "median", "CVH faster", "OpenCV faster/equal"],
                suite_summary_rows,
            )
        )
        lines.append("")

    lines.append("## Performance Priorities")
    lines.append("")
    lines.append("The multipliers below are within-group geometric means for this run. They prioritize follow-up work and do not indicate API support status.")
    lines.append("")
    lines.append(
        md_table(
            ["Area", "This Report", "Primary Cause", "Follow-up Boundary"],
            [
                [
                    "`GEMM`",
                    group_result({"GEMM"}),
                    "The default upstream build can use Accelerate/LAPACK; this is not a pure SIMD comparison against built-in OpenCV UI kernels",
                    "Keep the header-only boundary explicit when evaluating future improvements",
                ],
                [
                    "filter / derivative",
                    group_result({"BOX_FILTER", "FILTER2D", "GAUSSIAN", "SEP_FILTER2D", "SCHARR", "LAPLACIAN", "SPATIAL_GRADIENT"}),
                    "CVH still has generic filter dispatch, border materialization, and intermediate-row processing; upstream specializes more deeply by type and kernel size",
                    "Prioritize shared row/column work and fused U8-to-S16/F32 kernels",
                ],
                [
                    "nonlinear",
                    group_result({"BILATERAL_FILTER", "MEDIAN_BLUR", "STACK_BLUR"}),
                    "Repeated window scans are gone, but bilateral weight accumulation, the median lane network, and large-image cache behavior still lag",
                    "Separate pixel-kernel cost from memory-access cost using absolute runtime",
                ],
                [
                    "pyramid",
                    group_result({"PYR_DOWN", "PYR_UP", "BUILD_PYRAMID"}),
                    "The ring workspace and UI are in place, but C3 interleaving, boundary rows, and up/downsample writeback still trail specialized upstream kernels",
                    "Reuse the current ring infrastructure and avoid full-image temporaries",
                ],
                [
                    "geometry",
                    group_result({"CONVERT_MAPS", "REMAP", "WARP_AFFINE", "WARP_PERSPECTIVE"}),
                    "Coordinate blocks are shared, but interpolation, border masks, and multi-channel gather/store still contain substantial scalar work",
                    "Extend U8 C1/C3/C4 interior SIMD without duplicating public kernels",
                ],
                [
                    "reduction",
                    group_result({"MEAN_STD_DEV", "NORM", "REDUCE"}),
                    "Fast paths mainly cover F32 C1; the matrix also includes multi-channel, dual-input, and high-precision paths",
                    "Split gates by variant; do not trade precision for a better aggregate ratio",
                ],
                [
                    "P2 random / point transform",
                    group_result({"RANDU", "RANDN", "TRANSFORM", "PERSPECTIVE_TRANSFORM"}),
                    random_transform_cause,
                    random_transform_follow_up,
                ],
                [
                    "P2 regions / contours / shape",
                    group_result({"CONNECTED_COMPONENTS", "CONNECTED_COMPONENTS_WITH_STATS", "FIND_CONTOURS", "BOUNDING_RECT", "CONTOUR_AREA", "ARC_LENGTH", "APPROX_POLY_DP", "CONVEX_HULL", "IS_CONTOUR_CONVEX", "MOMENTS"}),
                    regions_cause,
                    regions_follow_up,
                ],
                [
                    "P2 histogram / template",
                    group_result({"CALC_HIST", "COMPARE_HIST", "MATCH_TEMPLATE"}),
                    histogram_template_cause,
                    histogram_template_follow_up,
                ],
            ],
        )
    )
    lines.append("")

    lines.append("## Operator-Level Overview")
    lines.append("")
    for suite in ("core_mat", "imgproc"):
        suite_rows = [r for r in supported if row_suite(r) == suite]
        if not suite_rows:
            continue
        lines.append(f"### `{suite}`")
        lines.append("")
        op_names = sorted({r.get("op", "") for r in suite_rows})
        op_rows = []
        for op in op_names:
            values = [
                to_float(r.get("speedup", "0"))
                for r in suite_rows
                if r.get("op", "") == op and to_float(r.get("speedup", "0")) > 0.0
            ]
            dispatches = sorted({r.get("dispatch_path", "") or "unknown" for r in suite_rows if r.get("op", "") == op})
            ratio = geometric_mean(values)
            winner = f"CVH `{ratio:.2f}x`" if ratio > 1.0 else f"OpenCV `{(1.0 / ratio):.2f}x`"
            op_rows.append(
                [
                    op,
                    ", ".join(dispatches),
                    str(len(values)),
                    f"{ratio:.4f}",
                    winner,
                ]
            )
        lines.append(
            md_table(
                ["Op", "CVH dispatch", "Cases", "geometric mean OpenCV/CVH", "Leader"],
                op_rows,
            )
        )
        lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    for suite in ("core_mat", "imgproc"):
        suite_rows = [r for r in supported if row_suite(r) == suite]
        if not suite_rows:
            continue
        lines.append(f"### `{suite}`")
        lines.append("")
        supported_sorted = sorted(
            suite_rows,
            key=lambda r: (
                r.get("op", ""),
                r.get("variant", ""),
                r.get("depth", ""),
                int(r.get("channels", "0")),
                r.get("layout", ""),
                r.get("shape", ""),
            ),
        )
        table_rows = []
        for r in supported_sorted:
            table_rows.append(
                [
                    r.get("op", ""),
                    r.get("variant", "") or "default",
                    r.get("dispatch_path", "") or "unknown",
                    r.get("depth", ""),
                    r.get("channels", ""),
                    r.get("layout", "continuous") or "continuous",
                    r.get("shape", ""),
                    f"{to_float(r.get('cvh_ms', '0')):.6f}",
                    f"{to_float(r.get('opencv_ms', '0')):.6f}",
                    f"{to_float(r.get('speedup', '0')):.4f}",
                    r.get("note", ""),
                ]
            )
        lines.append(
            md_table(
                ["Op", "Variant", "CVH dispatch", "Depth", "Ch", "Layout", "Shape", "CVH ms", "OpenCV ms", "OpenCV/CVH", "Note"],
                table_rows,
            )
        )
        lines.append("")

    if unsupported:
        lines.append("## Unsupported Cases")
        lines.append("")
        unsupported_rows = []
        for r in unsupported:
            unsupported_rows.append(
                [
                    row_suite(r),
                    r.get("op", ""),
                    r.get("variant", "") or "default",
                    r.get("shape", ""),
                    r.get("status", ""),
                    r.get("note", ""),
                ]
            )
        lines.append(md_table(["Suite", "Op", "Variant", "Shape", "Status", "Note"], unsupported_rows))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Ratios use `OpenCV time / CVH time`: values above `1` mean CVH is faster, and values below `1` mean OpenCV is faster.")
    lines.append("- Table timings use the minimum per-iteration time across repeats to reduce system-noise effects; this report is not a cross-machine ranking.")
    lines.append("- Mat cases compare matching allocation/reuse semantics; imgproc cases align input dimensions, types, kernels, borders, and primary parameters.")
    lines.append("- `headers_baseline` describes a public header fallback for an operator without a UI kernel; it is not a separate target or implementation profile.")
    lines.append("- Raw CSV and metadata files are reproducible run artifacts; date-named Markdown files are milestone snapshots.")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"input CSV not found: {in_path}")

    rows = read_rows(in_path)
    meta_path = Path(args.meta) if args.meta else None
    report = render_report(rows, args.title, in_path, meta_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"markdown_report_written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
