# OpenCV Compare Mode

This directory is the current workspace for comparing `cvh` with official
OpenCV. It belongs to Mode B in [../readme.md](../readme.md): current
header-only `cvh` versus OpenCV upstream.

## Target Design

Mode B intentionally uses only the fastest header-only profile on the `cvh`
side:

| Implementation | Meaning |
|---|---|
| `cvh_headers_fast` | Current `cvh::headers_fast`, representing the fastest header-only implementation. |
| `opencv` | Official OpenCV `core` / `imgproc` built on the same machine. |

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

## Local OpenCV Source

For this workspace, the preferred OpenCV source tree is:

```text
/Users/zmu/work/my_project/ocvh/opencv
```

From the `opencv-header-only` repository root this is:

```text
../opencv
```

The full OpenCV tree should be built separately and passed to the compare
runner by environment variables or future CLI flags. Do not point the legacy
`setup_opencv_bench_slim.sh` clone/update flow at the full local OpenCV
checkout unless you explicitly want that script to manage the repo.

## Current Harness

Existing files:

- `setup_opencv_bench_slim.sh`: historical helper for a slim OpenCV clone.
- `run_compare.sh`: one-command runner for `cvh::headers_fast` versus
  upstream OpenCV.
- `csv_to_markdown.py`: render compare CSV into Markdown.
- `opencv_compare_header_benchmark.cpp`: pure header-only `cvh` compare cases.
- `opencv_compare_opencv_backend.cpp`: OpenCV-side implementation, compiled
  without `cvh::headers` include paths.

Current caveats:

- `cvh::headers` is intentionally not a Mode B compare implementation. It is
  useful for default header-only validation and internal regression, while
  Mode B should stay easy to read: fastest header-only `cvh` versus upstream
  OpenCV.
- Raw CSV/metadata and rolling `current_*` reports are generated artifacts.
  Curated date-named `*-opencv-upstream-performance.md` snapshots may be
  committed under `benchmark/opencv_compare/results/`.
- New benchmark reports are written in English by default. Optional
  translations use a locale suffix such as `.zh-CN.md`; the English report is
  the canonical version linked from the project README.
- A missing `headers_fast` specialization is not an unsupported case:
  `cvh::headers_fast` inherits the `cvh::headers` implementation and the case
  remains in the report as `dispatch_path=headers_baseline`.

## Dated Snapshots

- [2026-07-25 OpenCV upstream performance (English)](results/2026-07-25-opencv-upstream-performance.en.md):
  Apple M5, single-threaded full profile, with all Phase 1 benchmark families.
  A [Chinese translation](results/2026-07-25-opencv-upstream-performance.md)
  is also available.
- [2026-07-24 OpenCV upstream performance](results/2026-07-24-opencv-upstream-performance.md):
  Apple M5, single-threaded full profile, including the current Core UI
  acceleration batches.
- [2026-07-23 OpenCV upstream performance](results/2026-07-23-opencv-upstream-performance.md):
  Apple M5, single-threaded stable profile, `core_mat` plus `imgproc`.

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
./benchmark/opencv_compare/run_compare.sh --profile quick --impls headers_fast
```

`cvh_headers_fast` is accepted as an alias for `headers_fast`.

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
- Full adds odd-width and non-contiguous ROI cases plus representative
  I420/YUY2/NV12 layouts.
- Raw CSV and metadata stay generated under
  `benchmark/opencv_compare/results/`; date-named Markdown snapshots may be
  tracked.
- Missing upstream operations remain explicit `UNSUPPORTED` rows.
