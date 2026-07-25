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
    return "P1 新增" if op in PHASE1_BENCHMARK_OPS else "既有"


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
            return "无有效 case"
        if ratio > 1.0:
            return f"CVH `~{ratio:.2f}x`"
        return f"OpenCV `~{(1.0 / ratio):.2f}x`"

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"生成时间（UTC）：`{generated_at}`")
    lines.append("")
    lines.append("## 当前项目状态")
    lines.append("")
    lines.append("- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。")
    lines.append("- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。")
    lines.append("- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。")
    lines.append("- Core/Imgproc 第一阶段完成后，名称级可调用覆盖为 `107/220`：Core `57/97`，Imgproc `50/123`。")
    lines.append("- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。")
    lines.append("- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。")
    lines.append("- Core F32 `patchNaNs/exp/log/pow` 已接入 UI；`pow` 分离整数指数与通用指数，特殊值 block 保留 scalar fallback。")
    lines.append("- Core `countNonZero/hasNonZero` 已接入 UI；计数使用分段 widen 归约，存在性检测按块 early-exit。")
    lines.append("- Core `findNonZero` 已接入稀疏感知 UI；全零 block 直接跳过，连续稠密 block 自适应切回 typed lane 枚举，并保持 row-major 坐标顺序。")
    lines.append("- Core `sum/mean/meanStdDev` 已接入 C1-C4 channel-aware UI；`sum/mean` 共享 widen sum/count，`meanStdDev` 使用中心化 block statistics 与 Chan merge。")
    lines.append("- Core `minMaxIdx/minMaxLoc/reduceArgMin/reduceArgMax` 已接入 UI；极值与索引在同一遍扫描中更新，并保留 first/last tie 语义。")
    lines.append("- Core `norm/normalize` 已接入 UI；`norm` 覆盖 U8/F32 单/双输入的 L1/L2/Inf，`normalize` 复用 norm/minmax 归约并向量化 F32 apply-scale。")
    lines.append("- P-ACC-2 至 P-ACC-8 已完成 Apple ARM 收尾，覆盖 Core 归约、布局、通道与 GEMM，以及 Imgproc 滤波、几何、非线性、形态学、累积和强度变换；真实 x86 SSE/AVX 运行验证仍是外部 gate。")
    lines.append("- P-ACC-8 新增 pyramid ring workspace、非线性滤波专用算法、几何 fixed-coordinate block、S16 derivative UI、sqrBoxFilter wide sliding sum 与 F32 C1 reduction fast-path。")
    lines.append("- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。")
    lines.append("- Imgproc legacy `.cpp` fast-path 已迁入 ODR-safe detail headers；resize/cvtColor、共享 filter、几何采样、morphology、threshold、LUT/histogram 和 accumulate family 均从公共 header API 进入。")
    lines.append(f"- 第一阶段新增的 `79` 个操作族已全部进入 Mode B，本报告包含 `{phase1_case_count}` 个 P1 性能 case。")
    lines.append(f"- `{meta.get('profile', 'unknown')}` profile 覆盖代表性的 `CV_8U` / `CV_32F`、C1/C3/C4、尺寸、布局与非连续 ROI 扩展。")
    lines.append("")

    lines.append("## 第一阶段新增算子")
    lines.append("")
    lines.append("本节记录第一阶段相对原有覆盖新增的 API 操作族。API 已实现不等于已经进入本次 Mode B 性能矩阵；只有建立了同输入、同参数 OpenCV 对照 case 的算子才计为“本报告实测”。")
    lines.append("")
    lines.append("| 模块 | 第一阶段新增 | 本报告实测 | 已实现但本报告未测 |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| Core | 43 | {len(measured_phase1_core)} | {43 - len(measured_phase1_core)} |")
    lines.append(f"| Imgproc | 36 | {len(measured_phase1_imgproc)} | {36 - len(measured_phase1_imgproc)} |")
    lines.append(f"| **合计** | **79** | **{len(measured_phase1)}** | **{79 - len(measured_phase1)}** |")
    lines.append("")
    lines.append("| 模块/类别 | 新增操作族 | 数量 | 本报告 Mode B 状态 |")
    lines.append("| --- | --- | ---: | --- |")
    lines.append("| Core：逐元素与逻辑 | `absdiff`、`bitwise_and`、`bitwise_not`、`bitwise_or`、`bitwise_xor`、`inRange`、`min`、`max` | 8 | 8/8 已实测 |")
    lines.append("| Core：转换、数学与校验 | `scaleAdd`、`convertScaleAbs`、`convertFp16`、`sqrt`、`pow`、`exp`、`log`、`checkRange`、`patchNaNs` | 9 | 9/9 已实测 |")
    lines.append("| Core：归约与统计 | `norm`、`sum`、`mean`、`meanStdDev`、`countNonZero`、`hasNonZero`、`findNonZero`、`minMaxIdx`、`minMaxLoc`、`reduce`、`reduceArgMax`、`reduceArgMin`、`normalize` | 13 | 13/13 已实测 |")
    lines.append("| Core：布局、复制与通道 | `borderInterpolate`、`copyTo`、`extractChannel`、`insertChannel`、`mixChannels`、`flip`、`flipND`、`rotate`、`repeat`、`hconcat`、`vconcat`、`broadcast`、`swap` | 13 | 13/13 已实测 |")
    lines.append("| Imgproc：核、滤波与强度 | `getStructuringElement`、`getGaussianKernel`、`getDerivKernels`、`getGaborKernel`、`createHanningWindow`、`integral`、`Scharr`、`Laplacian`、`spatialGradient`、`sqrBoxFilter`、`medianBlur`、`bilateralFilter`、`stackBlur`、`adaptiveThreshold`、`thresholdWithMask`、`equalizeHist`、`applyColorMap` | 17 | 17/17 已实测 |")
    lines.append("| Imgproc：累积、金字塔与颜色 | `accumulate`、`accumulateProduct`、`accumulateSquare`、`accumulateWeighted`、`blendLinear`、`pyrDown`、`pyrUp`、`buildPyramid`、`cvtColorTwoPlane`、`demosaicing` | 10 | 10/10 已实测 |")
    lines.append("| Imgproc：几何变换 | `remap`、`convertMaps`、`warpPerspective`、`getAffineTransform`、`getPerspectiveTransform`、`getRotationMatrix2D`、`getRotationMatrix2D_`、`invertAffineTransform`、`getRectSubPix` | 9 | 9/9 已实测 |")
    lines.append("")
    lines.append("后续表中的 `ADD`、`GEMM`、`resize`、`cvtColor` 等仍是既有算子基线；带有 `P1 新增` 标记的行是本轮新增并已进入性能对比的算子。")
    lines.append("")

    lines.append("## 高层优化结构")
    lines.append("")
    lines.append("| 层次 | 当前实现 | 本报告中的含义 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |")
    lines.append("| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |")
    lines.append("| 专用 kernel | `cvtColor`、特定 `resize`、core 逐元素、统计/非零归约、F32 数学、pyramid 与 derivative UI kernel | 实际命中时记录为 `dispatch_path=opencv_ui` |")
    lines.append("| Header fast-path | 行并行 filter、LUT、border、Sobel、Canny、morphology、sliding sum 与 nonlinear 专用 kernel | 记录实际 `header_fastpath` / `sliding_*` / `precomputed_lut` 路径 |")
    lines.append("| 几何采样 | 共享定点坐标 block、U8 bilinear sampler 与 interior/border 分流 | 记录为 `dispatch_path=fixed_coordinate_block` |")
    lines.append("| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` 或 `public_header_scalar` |")
    lines.append("| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |")
    lines.append("")

    if meta:
        lines.append("## 运行配置")
        lines.append("")
        lines.append(f"- Profile：`{meta.get('profile', 'unknown')}`")
        meta_impls = meta.get("impls", [])
        if isinstance(meta_impls, list) and meta_impls:
            lines.append(f"- CVH 实现：`{', '.join(meta_impls)}`")
        lines.append(
            f"- 采样：`warmup={meta.get('warmup', 'n/a')}, iters={meta.get('iters', 'n/a')}, repeats={meta.get('repeats', 'n/a')}`"
        )
        lines.append(f"- 线程数：`{meta.get('threads', 'n/a')}`")
        lines.append(
            f"- OpenMP：`dynamic={meta.get('omp_dynamic', 'n/a')}, proc_bind={meta.get('omp_proc_bind', 'n/a')}`"
        )
        lines.append(f"- 主机：`{meta.get('system', 'n/a')} {meta.get('arch', 'n/a')}`")
        lines.append(f"- CPU：`{meta.get('cpu_model', 'n/a')}`")
        lines.append(f"- 编译器：`{meta.get('compiler', 'n/a')}`")
        lines.append(f"- 构建类型：`{meta.get('build_type', 'n/a')}`")
        lines.append(
            f"- CVH commit：`{meta.get('repo_git_commit', 'unknown')}`"
            f"{' + dirty' if meta.get('repo_git_dirty', False) else ''}"
        )
        lines.append(
            f"- OpenCV：`{meta.get('opencv_version', 'unknown')}`，commit "
            f"`{meta.get('opencv_git_commit', 'unknown')}`"
            f"{' + dirty' if meta.get('opencv_git_dirty', False) else ''}"
        )
        lines.append(f"- 原始数据：`{input_path.name}`；元数据：`{meta_path.name if meta_path else 'n/a'}`")
        lines.append("")

    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- 总 case：`{len(rows)}`；有效：`{len(supported)}`；不支持：`{len(unsupported)}`。")
    lines.append(f"- `OpenCV/CVH` 几何平均：`{geo_speedup:.4f}`；中位数：`{median_speedup:.4f}`。")
    lines.append(f"- CVH 更快：`{cvh_faster}` 个；OpenCV 更快或相当：`{opencv_faster_or_equal}` 个。")
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
                ["Suite", "Cases", "几何平均 OpenCV/CVH", "中位数", "CVH 更快", "OpenCV 更快/相当"],
                suite_summary_rows,
            )
        )
        lines.append("")

    lines.append("## P-ACC-8 未收敛差距")
    lines.append("")
    lines.append("以下倍率是本次 case 的组内几何平均，只用于定位后续工作，不等同于 API 支持状态。P-ACC-8 已通过相对自身旧路径的接受 gate，但部分算子仍明显落后 upstream。")
    lines.append("")
    lines.append(
        md_table(
            ["范围", "本报告", "主要原因", "后续边界"],
            [
                [
                    "`GEMM`",
                    group_result({"GEMM"}),
                    "默认 upstream 可进入 Accelerate/LAPACK；这不是 OpenCV UI 内建 kernel 的纯 SIMD 对比",
                    "保留现有 header-only micro-kernel，不为追赶外部 BLAS 引入链接依赖",
                ],
                [
                    "filter / derivative",
                    group_result({"BOX_FILTER", "FILTER2D", "GAUSSIAN", "SEP_FILTER2D", "SCHARR", "LAPLACIAN", "SPATIAL_GRADIENT"}),
                    "CVH 仍有通用 filter 调度、border materialization 和中间行处理；upstream 的类型/核尺寸专用化更深",
                    "下一批优先做共享 row/column engine 与 U8-to-S16/F32 fused kernel",
                ],
                [
                    "nonlinear",
                    group_result({"BILATERAL_FILTER", "MEDIAN_BLUR", "STACK_BLUR"}),
                    "已消除重复窗口扫描，但 bilateral 权重累计、median lane network 和大尺寸 cache 行为仍有差距",
                    "保留已接受算法，后续按绝对耗时继续拆像素内核与内存访问",
                ],
                [
                    "pyramid",
                    group_result({"PYR_DOWN", "PYR_UP", "BUILD_PYRAMID"}),
                    "ring workspace 和 UI 已落地，C3 interleave、边界行与上下采样写回仍未达到 upstream 专用 kernel",
                    "继续复用当前 ring 基础设施，不回退到整图 temporary",
                ],
                [
                    "geometry",
                    group_result({"CONVERT_MAPS", "REMAP", "WARP_AFFINE", "WARP_PERSPECTIVE"}),
                    "坐标 block 已共享，但插值、border mask 和多通道 gather/store 仍包含较多标量工作",
                    "后续只扩 U8 C1/C3/C4 interior SIMD，不复制三套公共内核",
                ],
                [
                    "reduction",
                    group_result({"MEAN_STD_DEV", "NORM", "REDUCE"}),
                    "本轮 fast-path 主要覆盖 F32 C1；Mode B 仍包含多通道、双输入和高精度合同路径",
                    "按 variant 拆 gate，不能用降低精度换取汇总倍率",
                ],
            ],
        )
    )
    lines.append("")

    lines.append("## 算子级概览")
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
                ["Op", "阶段", "CVH dispatch", "Cases", "几何平均 OpenCV/CVH", "领先方"],
                op_rows,
            )
        )
        lines.append("")

    lines.append("## 详细结果")
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
                ["Op", "阶段", "Variant", "CVH dispatch", "Depth", "Ch", "Layout", "Shape", "CVH ms", "OpenCV ms", "OpenCV/CVH", "Note"],
                table_rows,
            )
        )
        lines.append("")

    if unsupported:
        lines.append("## 不支持用例")
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

    lines.append("## 说明")
    lines.append("")
    lines.append("- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。")
    lines.append("- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。")
    lines.append("- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。")
    lines.append("- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。")
    lines.append("- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。")

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
