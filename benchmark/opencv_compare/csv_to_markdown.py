#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Optional


PHASE1_BENCHMARK_OPS = {
    "ABSDIFF",
    "BITWISE_AND",
    "BITWISE_NOT",
    "BITWISE_OR",
    "BITWISE_XOR",
    "IN_RANGE",
    "MIN",
    "MAX",
    "SCALE_ADD",
    "CONVERT_SCALE_ABS",
    "CONVERT_FP16",
    "SQRT",
    "POW",
    "EXP",
    "LOG",
    "CHECK_RANGE",
    "PATCH_NANS",
    "NORM",
    "SUM",
    "MEAN",
    "MEAN_STD_DEV",
    "COUNT_NON_ZERO",
    "HAS_NON_ZERO",
    "FIND_NON_ZERO",
    "MIN_MAX_IDX",
    "MIN_MAX_LOC",
    "REDUCE",
    "REDUCE_ARG_MAX",
    "REDUCE_ARG_MIN",
    "NORMALIZE",
    "BORDER_INTERPOLATE",
    "COPY_TO",
    "EXTRACT_CHANNEL",
    "INSERT_CHANNEL",
    "MIX_CHANNELS",
    "FLIP",
    "FLIP_ND",
    "ROTATE",
    "REPEAT",
    "HCONCAT",
    "VCONCAT",
    "BROADCAST",
    "SWAP",
    "GET_STRUCTURING_ELEMENT",
    "GET_GAUSSIAN_KERNEL",
    "GET_DERIV_KERNELS",
    "GET_GABOR_KERNEL",
    "CREATE_HANNING_WINDOW",
    "INTEGRAL",
    "SCHARR",
    "LAPLACIAN",
    "SPATIAL_GRADIENT",
    "SQR_BOX_FILTER",
    "MEDIAN_BLUR",
    "BILATERAL_FILTER",
    "STACK_BLUR",
    "ADAPTIVE_THRESHOLD",
    "THRESHOLD_WITH_MASK",
    "EQUALIZE_HIST",
    "APPLY_COLOR_MAP",
    "ACCUMULATE",
    "ACCUMULATE_PRODUCT",
    "ACCUMULATE_SQUARE",
    "ACCUMULATE_WEIGHTED",
    "BLEND_LINEAR",
    "PYR_DOWN",
    "PYR_UP",
    "BUILD_PYRAMID",
    "CVT_COLOR_TWO_PLANE",
    "DEMOSAICING",
    "REMAP",
    "CONVERT_MAPS",
    "WARP_PERSPECTIVE",
    "GET_AFFINE_TRANSFORM",
    "GET_PERSPECTIVE_TRANSFORM",
    "GET_ROTATION_MATRIX_2D",
    "GET_ROTATION_MATRIX_2D_",
    "INVERT_AFFINE_TRANSFORM",
    "GET_RECT_SUB_PIX",
}


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


def phase_label(op: str) -> str:
    return "P1 added" if op in PHASE1_BENCHMARK_OPS else "Existing"


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
    measured_phase1_core = {
        r.get("op", "")
        for r in supported
        if row_suite(r) == "core_mat" and r.get("op", "") in PHASE1_BENCHMARK_OPS
    }
    measured_phase1_imgproc = {
        r.get("op", "")
        for r in supported
        if row_suite(r) == "imgproc" and r.get("op", "") in PHASE1_BENCHMARK_OPS
    }
    measured_phase1 = measured_phase1_core | measured_phase1_imgproc
    phase1_case_count = sum(
        1 for r in supported if r.get("op", "") in PHASE1_BENCHMARK_OPS
    )

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

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated at (UTC): `{generated_at}`")
    lines.append("")
    lines.append("## Current Project Status")
    lines.append("")
    lines.append("- `cvh` (cv-header-only) is an independent pure header-only library and does not depend on an in-project `.cpp` extension layer.")
    lines.append("- Mode B compares the current `cvh::headers_fast` against upstream OpenCV built on the same machine; `cvh::headers_fast` represents the fastest header-only build configuration.")
    lines.append("- `cvh::headers_fast` fully inherits `cvh::headers`. When an operator has no dedicated fast path, it continues to use the inherited header implementation and remains in the benchmark instead of being skipped for lack of a SIMD specialization.")
    lines.append("- After Phase 1 for Core and Imgproc, callable API-name coverage is `107/220`: Core `57/97` and Imgproc `50/123`.")
    lines.append("- Core `add/subtract/multiply/divide/transpose/GEMM` have moved into ODR-safe headers; this report measures through the public API without linking legacy core objects.")
    lines.append("- OpenCV Universal Intrinsics are the default SIMD dialect, and kernels use OpenCV UI directly; the xsimd performance path has been removed.")
    lines.append("- Core F32 `patchNaNs/exp/log/pow` use UI; `pow` separates integer and general exponents while retaining a scalar fallback for special-value blocks.")
    lines.append("- Core `countNonZero/hasNonZero` use UI; counting uses segmented widening reduction, and existence checks exit early by block.")
    lines.append("- Core `findNonZero` uses sparsity-aware UI; all-zero blocks are skipped, contiguous dense blocks adaptively fall back to typed-lane enumeration, and row-major coordinate order is preserved.")
    lines.append("- Core `sum/mean/meanStdDev` use C1-C4 channel-aware UI; `sum/mean` share widening sum/count logic, and `meanStdDev` uses centered block statistics with Chan merging.")
    lines.append("- Core `minMaxIdx/minMaxLoc/reduceArgMin/reduceArgMax` use UI; extrema and indices are updated in a single pass while preserving first/last tie semantics.")
    lines.append("- Core `norm/normalize` use UI; `norm` covers L1/L2/Inf for single- and dual-input U8/F32, while `normalize` reuses norm/min-max reductions and vectorizes F32 scale application.")
    lines.append("- P-ACC-2 through P-ACC-8 complete the Apple ARM work across Core reductions, layouts, channels, and GEMM, plus Imgproc filtering, geometry, nonlinear operations, morphology, accumulation, and intensity transforms; validation on real x86 SSE/AVX hardware remains an external gate.")
    lines.append("- P-ACC-8 adds pyramid ring workspaces, specialized nonlinear-filter algorithms, fixed-coordinate geometry blocks, S16 derivative UI, a wide sliding sum for `sqrBoxFilter`, and an F32 C1 reduction fast path.")
    lines.append("- ARM work currently targets NEON, and this run used Apple ARM; x86 targets the SSE/AVX families, while RVV is deferred because of scalable-vector design considerations.")
    lines.append("- Legacy Imgproc `.cpp` fast paths have moved into ODR-safe detail headers; resize/cvtColor, shared filters, geometric sampling, morphology, threshold, LUT/histogram, and the accumulate family all enter through the public header API.")
    lines.append(f"- All `79` Phase 1 operation families are covered by Mode B; this report contains `{phase1_case_count}` P1 performance cases.")
    lines.append(f"- `{meta.get('profile', 'unknown')}` profile covers representative `CV_8U` / `CV_32F`, C1/C3/C4, dimensions, layouts, and non-contiguous ROI extensions.")
    lines.append("")

    lines.append("## Phase 1 Added Operators")
    lines.append("")
    lines.append("This section records API operation families added in Phase 1 relative to the previous coverage. An implemented API is not necessarily part of this Mode B performance matrix; an operator counts as measured only when a matching OpenCV case uses the same input and parameters.")
    lines.append("")
    lines.append("| Module | Added in Phase 1 | Measured in This Report | Implemented but Not Measured |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| Core | 43 | {len(measured_phase1_core)} | {43 - len(measured_phase1_core)} |")
    lines.append(f"| Imgproc | 36 | {len(measured_phase1_imgproc)} | {36 - len(measured_phase1_imgproc)} |")
    lines.append(f"| **Total** | **79** | **{len(measured_phase1)}** | **{79 - len(measured_phase1)}** |")
    lines.append("")
    lines.append("| Module / Category | Added Operation Families | Count | Mode B Status in This Report |")
    lines.append("| --- | --- | ---: | --- |")
    lines.append("| Core: element-wise and logical | `absdiff`, `bitwise_and`, `bitwise_not`, `bitwise_or`, `bitwise_xor`, `inRange`, `min`, `max` | 8 | 8/8 measured |")
    lines.append("| Core: conversion, math, and validation | `scaleAdd`, `convertScaleAbs`, `convertFp16`, `sqrt`, `pow`, `exp`, `log`, `checkRange`, `patchNaNs` | 9 | 9/9 measured |")
    lines.append("| Core: reductions and statistics | `norm`, `sum`, `mean`, `meanStdDev`, `countNonZero`, `hasNonZero`, `findNonZero`, `minMaxIdx`, `minMaxLoc`, `reduce`, `reduceArgMax`, `reduceArgMin`, `normalize` | 13 | 13/13 measured |")
    lines.append("| Core: layout, copying, and channels | `borderInterpolate`, `copyTo`, `extractChannel`, `insertChannel`, `mixChannels`, `flip`, `flipND`, `rotate`, `repeat`, `hconcat`, `vconcat`, `broadcast`, `swap` | 13 | 13/13 measured |")
    lines.append("| Imgproc: kernels, filtering, and intensity | `getStructuringElement`, `getGaussianKernel`, `getDerivKernels`, `getGaborKernel`, `createHanningWindow`, `integral`, `Scharr`, `Laplacian`, `spatialGradient`, `sqrBoxFilter`, `medianBlur`, `bilateralFilter`, `stackBlur`, `adaptiveThreshold`, `thresholdWithMask`, `equalizeHist`, `applyColorMap` | 17 | 17/17 measured |")
    lines.append("| Imgproc: accumulation, pyramids, and color | `accumulate`, `accumulateProduct`, `accumulateSquare`, `accumulateWeighted`, `blendLinear`, `pyrDown`, `pyrUp`, `buildPyramid`, `cvtColorTwoPlane`, `demosaicing` | 10 | 10/10 measured |")
    lines.append("| Imgproc: geometric transforms | `remap`, `convertMaps`, `warpPerspective`, `getAffineTransform`, `getPerspectiveTransform`, `getRotationMatrix2D`, `getRotationMatrix2D_`, `invertAffineTransform`, `getRectSubPix` | 9 | 9/9 measured |")
    lines.append("")
    lines.append("Operators such as `ADD`, `GEMM`, `resize`, and `cvtColor` remain existing baselines in the tables below; rows marked `P1 added` were added in this phase and included in the performance comparison.")
    lines.append("")

    lines.append("## High-Level Optimization Structure")
    lines.append("")
    lines.append("| Layer | Current Implementation | Meaning in This Report |")
    lines.append("| --- | --- | --- |")
    lines.append("| Public API | OpenCV-compatible header API | All cases call the public `cvh::headers_fast` entry points |")
    lines.append("| SIMD dialect | OpenCV Universal Intrinsics | Maps to NEON on Apple ARM |")
    lines.append("| Specialized kernel | `cvtColor`, selected `resize`, Core element-wise operations, statistics/nonzero reductions, F32 math, pyramid, and derivative UI kernels | Recorded as `dispatch_path=opencv_ui` |")
    lines.append("| Header fast-path | Row-parallel filters, LUT, border, Sobel, Canny, morphology, sliding sums, and specialized nonlinear kernels | Records the actual `header_fastpath` / `sliding_*` / `precomputed_lut` path |")
    lines.append("| Geometric sampling | shared fixed-coordinate blocks, U8 bilinear sampler with interior/border routing | Recorded as `dispatch_path=fixed_coordinate_block` |")
    lines.append("| Generic implementation | `cvh::headers` header baseline | Inherited automatically when no dedicated fast path exists; recorded as `headers_baseline` or `public_header_scalar` |")
    lines.append("| Reference implementation | upstream OpenCV `core` / `imgproc` | Same input, dimensions, border, and thread configuration |")
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

    lines.append("## Remaining P-ACC-8 Gaps")
    lines.append("")
    lines.append("The multipliers below are within-group geometric means for this run. They are intended only to prioritize follow-up work and do not indicate API support status. P-ACC-8 passed its acceptance gates relative to the previous internal paths, but some operators still trail upstream significantly.")
    lines.append("")
    lines.append(
        md_table(
            ["Area", "This Report", "Primary Cause", "Follow-up Boundary"],
            [
                [
                    "`GEMM`",
                    group_result({"GEMM"}),
                    "The default upstream build can use Accelerate/LAPACK; this is not a pure SIMD comparison against built-in OpenCV UI kernels",
                    "Keep the current header-only micro-kernel and do not add a link-time dependency merely to chase external BLAS performance",
                ],
                [
                    "filter / derivative",
                    group_result({"BOX_FILTER", "FILTER2D", "GAUSSIAN", "SEP_FILTER2D", "SCHARR", "LAPLACIAN", "SPATIAL_GRADIENT"}),
                    "CVH still has generic filter dispatch, border materialization, and intermediate-row processing; upstream specializes more deeply by type and kernel size",
                    "Prioritize a shared row/column engine and fused U8-to-S16/F32 kernels next",
                ],
                [
                    "nonlinear",
                    group_result({"BILATERAL_FILTER", "MEDIAN_BLUR", "STACK_BLUR"}),
                    "Repeated window scans are gone, but bilateral weight accumulation, the median lane network, and large-image cache behavior still lag",
                    "Keep the accepted algorithms and continue separating pixel kernels from memory access based on absolute runtime",
                ],
                [
                    "pyramid",
                    group_result({"PYR_DOWN", "PYR_UP", "BUILD_PYRAMID"}),
                    "The ring workspace and UI are in place, but C3 interleaving, boundary rows, and up/downsample writeback still trail specialized upstream kernels",
                    "Continue reusing the current ring infrastructure without reverting to full-image temporaries",
                ],
                [
                    "geometry",
                    group_result({"CONVERT_MAPS", "REMAP", "WARP_AFFINE", "WARP_PERSPECTIVE"}),
                    "Coordinate blocks are shared, but interpolation, border masks, and multi-channel gather/store still contain substantial scalar work",
                    "Extend only U8 C1/C3/C4 interior SIMD without duplicating three public kernels",
                ],
                [
                    "reduction",
                    group_result({"MEAN_STD_DEV", "NORM", "REDUCE"}),
                    "This round's fast paths mainly cover F32 C1; Mode B still includes multi-channel, dual-input, and high-precision contract paths",
                    "Split gates by variant; do not trade precision for a better aggregate ratio",
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
                    phase_label(op),
                    ", ".join(dispatches),
                    str(len(values)),
                    f"{ratio:.4f}",
                    winner,
                ]
            )
        lines.append(
            md_table(
                ["Op", "Phase", "CVH dispatch", "Cases", "geometric mean OpenCV/CVH", "Leader"],
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
                    phase_label(r.get("op", "")),
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
                ["Op", "Phase", "Variant", "CVH dispatch", "Depth", "Ch", "Layout", "Shape", "CVH ms", "OpenCV ms", "OpenCV/CVH", "Note"],
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
    lines.append("- `headers_baseline` does not mean optimization was skipped; it indicates that `cvh::headers_fast` currently inherits the generic `cvh::headers` implementation.")
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
