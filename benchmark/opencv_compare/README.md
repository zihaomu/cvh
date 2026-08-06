# OpenCV Compare Mode

This directory is the current workspace for comparing `cvh` with official
OpenCV. It belongs to Mode B in [../readme.md](../readme.md): current
header-only `cvh` versus OpenCV upstream.

## Target Design

Mode B uses the single public compute target. Product reports use automatic
runtime dispatch; forced UI/scalar modes remain available for diagnostics:

| Implementation | Meaning |
|---|---|
| `cvh_auto` | Current `cvh::headers` with product `Auto` dispatch; eligible direct NEON/AVX2 kernels take priority, followed by OpenCV UI and scalar fallbacks. This is the default report mode. |
| `cvh_ui` | Current `cvh::headers` with `OpenCVUIOnly` forced; specialized NEON/AVX2 dispatch is rejected. |
| `cvh_scalar` | The same public headers with `ScalarOnly` forced, used to verify fallback correctness and dispatch observability. |
| `opencv` | Official OpenCV `core` / `imgproc` built on the same machine. |

`opencv_ui` is a vector-dialect observation, not an ISA observation. On an
ARM64 build the UI operations normally compile to NEON instructions, but the
report only claims a concrete ISA when `isa_observed` can record it. In
particular, a `GEMM` row tagged `opencv_ui` in this mode did not use CVH's
specialized direct-NEON GEMM backend because `OpenCVUIOnly` rejects that path.

The compare report is for visibility and prioritization. It is log-only by
default and should not block every PR.

Required metadata for every run:

- `cvh` git commit
- OpenCV git commit
- compiler and build type
- OS, arch, CPU
- thread count and runtime flags
- profile, warmup, iters, repeats
- OpenCV source/build directory
- focused `ops` filter and OpenCV acceleration variant
- OpenCV `WITH_LAPACK/WITH_IPP/WITH_KLEIDICV/WITH_CAROTENE`
  values when a CMake cache is available

## Sampling Policy

Regular image and matrix cases use the selected profile or explicit CLI
values for `warmup`, `iters`, and `repeats`.

P1 micro cases use a fixed `warmup=2`, `iters=100`, `repeats=3` policy. These
operations are too short for the regular image-case sampling counts, while
three repeated samples provide basic timing-noise resistance without making
micro cases dominate the total benchmark runtime.
CVH and upstream OpenCV always receive the same effective values. Each result
row records the effective micro settings in its `note` field.

P2-P0 shape-vector and 256-bin histogram comparison cases use the same fixed
micro policy. Image, random, transform, connected-region, contour, histogram
construction, and template-matching cases use the selected profile values.
The five random rows align type, shape, layout, range/mean, and standard
deviation, but CVH and OpenCV advance independent random streams because CVH
does not expose public RNG seed/state control.

## Local OpenCV Source

Provide an existing OpenCV/slim checkout with:

```text
CVH_OPENCV_DIR=/path/to/opencv-bench-slim
```

The runner default is:

```text
benchmark/opencv_compare/opencv-bench-slim
```

The full OpenCV tree should be built separately and passed to the compare
runner by environment variables or future CLI flags. Do not point the legacy
`setup_opencv_bench_slim.sh` clone/update flow at the full local OpenCV
checkout unless you explicitly want that script to manage the repo.

## Current Harness

Existing files:

- `setup_opencv_bench_slim.sh`: historical helper for a slim OpenCV clone.
- `run_compare.sh`: one-command runner for product-auto `cvh::headers` versus
  upstream OpenCV, with optional forced UI/scalar diagnostics.
- `csv_to_markdown.py`: render compare CSV into Markdown.
- `opencv_compare_header_benchmark.cpp`: pure header-only `cvh` compare cases.
- `opencv_compare_opencv_backend.cpp`: OpenCV-side implementation, compiled
  without `cvh::headers` include paths.
- `opencv_compare_phase2_header_benchmark.cpp`: CVH-side P2-P0 timing and case
  metadata for 26 representative rows.
- `opencv_compare_phase2_opencv_backend.cpp`: matching upstream OpenCV P2-P0
  kernels, compiled without `cvh::headers` include paths.

Current CSV observability fields:

| Field | Meaning |
|---|---|
| `algorithm_path` | Operator-level algorithm or data-flow choice, such as `gauss_separable` or `morph_rect3x3`. |
| `dispatch_path` | Actual last kernel dispatch reported by cvh, such as `scalar` or `opencv_ui`. |
| `isa_observed` | Directly observable specialized ISA; `unknown` when the UI backend does not expose it. The host architecture is never used as a substitute. |
| `kernel_route` | Stage-level route actually used by the case. Composite kernels use semicolon-separated stages; historical rows fall back to the main dispatch label. |

Historical CSV files without the two new fields remain supported by the
Markdown renderer.

Current caveats:

- Mode B has one product target with three runtime observations: `cvh_auto`
  (`Auto`), `cvh_ui` (`OpenCVUIOnly`), and `cvh_scalar` (`ScalarOnly`). Product
  reports default to `cvh_auto`; the benchmark rejects direct ISA in `cvh_ui`
  and rejects every accelerated dispatch in `cvh_scalar`.
- Rolling `current_*` reports are generated artifacts. A reviewed date-named
  snapshot may commit its English Markdown, raw CSV, and metadata together
  under `benchmark/opencv_compare/results/`.
- New benchmark reports are written in English by default. Optional
  translations use a locale suffix such as `.zh-CN.md`; the English report is
  the canonical version linked from the project README.
- A case without a UI kernel remains in the report through its public header
  fallback; `headers_baseline` is a dispatch description, not another product
  target.

## Dated Snapshots

- [2026-08-06 v0.1 RC product-auto OpenCV upstream performance (English)](results/2026-08-06-v0.1-rc-auto-opencv-upstream-performance.en.md):
  clean `cbd5076` cvh snapshot on Apple M5, single-threaded full profile; 369
  valid rows, overall geometric mean `0.7406`, Core `0.6539`, and Imgproc
  `0.8371`. Every case records its algorithm, dispatch, and observed ISA; all
  10 GEMM cases selected direct NEON.
- [2026-08-04 v0.1 RC forced-UI OpenCV upstream performance (English)](results/2026-08-04-v0.1-rc-opencv-upstream-performance.en.md):
  clean `f94f2d8` cvh snapshot on Apple M5, single-threaded full profile; this
  is historical diagnostic evidence that excluded direct NEON/AVX2 dispatch.
- [2026-08-03 OpenCV upstream performance (English)](results/2026-08-03-opencv-upstream-performance.en.md):
  current Apple M5, single-threaded full profile; all CVH rows use `cvh_ui`.
- [2026-07-25 OpenCV upstream performance (English)](results/2026-07-25-opencv-upstream-performance.en.md):
  historical Apple M5, single-threaded full profile from before the current
  single-target CPU configuration.
  A [Chinese translation](results/2026-07-25-opencv-upstream-performance.md)
  is also available.
- [2026-07-24 OpenCV upstream performance](results/2026-07-24-opencv-upstream-performance.md):
  historical Apple M5, single-threaded full profile.
- [2026-07-23 OpenCV upstream performance](results/2026-07-23-opencv-upstream-performance.md):
  historical Apple M5, single-threaded stable profile.

See the [result index](results/README.md) for raw artifacts and the snapshot
retention rule.

## Current Commands

Header-only quick run:

```bash
./benchmark/opencv_compare/run_compare.sh --profile quick
```

Use an existing local OpenCV build:

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
./benchmark/opencv_compare/run_compare.sh --profile quick
```

Stable baseline:

```bash
./benchmark/opencv_compare/run_compare.sh --profile stable --baseline
```

Explicit implementation:

```bash
./benchmark/opencv_compare/run_compare.sh --profile quick --impls ui
```

`cvh_ui` is accepted as an alias for `ui`.

Forced scalar dispatch verification:

```bash
./benchmark/opencv_compare/run_compare.sh \
  --profile quick --impls scalar --ops IMGPROC_FLOOR
```

Focused v0.1 Imgproc performance-floor matrix:

```bash
./benchmark/opencv_compare/run_compare.sh \
  --profile stable --impls ui --ops IMGPROC_FLOOR
```

`IMGPROC_FLOOR` freezes the families owned by
[`cvh-v0.1-imgproc-performance-floor-acceleration-plan.md`](../../doc/cvh-v0.1-imgproc-performance-floor-acceleration-plan.md).

Focused v0.1 direct-NEON hot-kernel matrix:

```bash
./benchmark/opencv_compare/run_compare.sh \
  --profile stable --impls auto,ui,scalar --ops V01_NEON_HOT
```

`V01_NEON_HOT` contains only `CVTCOLOR`, `RESIZE`, `SOBEL`, `SCHARR`, and
`SPATIAL_GRADIENT`. Stable runs add the target resolutions and odd-width ROI
needed by
[`cvh-v0.1-neon-hot-kernel-acceleration-plan.md`](../../doc/cvh-v0.1-neon-hot-kernel-acceleration-plan.md).

Focused P2-P0 operator comparison:

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
./benchmark/opencv_compare/run_compare.sh \
  --profile quick --ops PHASE2_P0 \
  --warmup 1 --iters 5 --repeats 1 --threads 1
```

This produces 26 rows covering all 17 P2-P0 operation families. Rows without
a CVH UI kernel remain valid comparisons and record
`dispatch_path=public_header_scalar` plus `no_ui_fastpath` in the note.

Focused GEMM attribution:

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
CVH_COMPARE_OPENCV_VARIANT=default_lapack_on \
./benchmark/opencv_compare/run_compare.sh \
  --profile quick --ops GEMM \
  --warmup 2 --iters 100 --repeats 3 --threads 1
```

Build the separate upstream built-in CPU configuration and run the same
focused matrix:

```bash
./benchmark/opencv_compare/configure_opencv_cpu_only.sh

CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-cpu-only \
CVH_COMPARE_OPENCV_VARIANT=builtin_cpu_no_lapack_ipp_hal \
./benchmark/opencv_compare/run_compare.sh \
  --profile quick --ops GEMM \
  --warmup 2 --iters 100 --repeats 3 --threads 1
```

The CPU-only configuration disables LAPACK, IPP, KleidiCV, Carotene, and
OpenCL. It is an attribution build, not the upstream product-performance
configuration used by the regular Mode B report.

## Coverage Status

- Stable covers the core compute matrix plus representative imgproc U8/F32
  C1/C3/C4 cases.
- P2-P0 contributes 7 Core and 19 Imgproc rows in every profile; the focused
  `PHASE2_P0` filter runs only those rows.
- Full adds odd-width and non-contiguous ROI cases plus representative
  I420/YUY2/NV12 layouts.
- Raw CSV and metadata stay generated under
  `benchmark/opencv_compare/results/`; a reviewed date-named snapshot tracks
  its English Markdown, CSV, and metadata together.
- Missing upstream operations remain explicit `UNSUPPORTED` rows.
