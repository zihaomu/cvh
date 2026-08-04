# cvh vs OpenCV Benchmark Report (stable)

Generated at (UTC): `2026-08-04 04:03:50Z`

## Scope

- `cvh` is a pure header-only library; every CVH row enters through the public `cvh::headers` API.
- The benchmark forces `OpenCVUIOnly`, and every CVH implementation label must therefore be `cvh_ui`.
- Direct architecture-specific dispatch is rejected. Operators without a Universal Intrinsics kernel use their normal public-header fallback, whose actual path is recorded in `CVH dispatch`.
- The reference is the upstream OpenCV build recorded in the metadata, running on the same host with matching inputs and parameters.

## Comparison Model

| Layer | Current Implementation | Meaning in This Report |
| --- | --- | --- |
| Public candidate | `cvh::headers` | Built with `OpenCVUIOnly`; implementation label `cvh_ui` |
| Vector dialect | OpenCV Universal Intrinsics | Portable UI kernels selected by the compiler and intrinsics layer |
| Public fallback | Header-only scalar or generic fast path | Same product target; actual path recorded by `dispatch_path` |
| Reference | Upstream OpenCV `core` / `imgproc` | Same input, dimensions, borders, parameters, and thread setting |

## Run Configuration

- Profile: `stable`
- CVH implementation: `cvh_ui`
- Sampling: `warmup=2, iters=20, repeats=5`
- Threads: `1`
- OpenMP: `dynamic=false, proc_bind=close`
- Host: `Darwin arm64`
- CPU: `Apple M5`
- Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102)`
- Build type: `Release`
- CVH commit: `8360e586d8c004954a2cfd0b22ce1a1476cf9af9` + dirty
- OpenCV: `4.14.0`, commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- Raw data: `2026-08-04-phase2-p0-a1-opencv-upstream-performance.csv`; metadata: `2026-08-04-phase2-p0-a1-opencv-upstream-performance.meta.json`

## Summary

- Total cases: `26`; valid: `26`; unsupported: `0`.
- `OpenCV/CVH` geometric mean: `0.1904`; median: `0.1846`.
- CVH faster: `3`; OpenCV faster or equal: `23`.

| Suite | Cases | geometric mean OpenCV/CVH | median | CVH faster | OpenCV faster/equal |
| --- | --- | --- | --- | --- | --- |
| core_mat | 7 | 0.0812 | 0.1249 | 0 | 7 |
| imgproc | 19 | 0.2607 | 0.6153 | 3 | 16 |

## Performance Priorities

The multipliers below are within-group geometric means for this run. They prioritize follow-up work and do not indicate API support status.

| Area | This Report | Primary Cause | Follow-up Boundary |
| --- | --- | --- | --- |
| `GEMM` | no valid cases | The default upstream build can use Accelerate/LAPACK; this is not a pure SIMD comparison against built-in OpenCV UI kernels | Keep the header-only boundary explicit when evaluating future improvements |
| filter / derivative | no valid cases | CVH still has generic filter dispatch, border materialization, and intermediate-row processing; upstream specializes more deeply by type and kernel size | Prioritize shared row/column work and fused U8-to-S16/F32 kernels |
| nonlinear | no valid cases | Repeated window scans are gone, but bilateral weight accumulation, the median lane network, and large-image cache behavior still lag | Separate pixel-kernel cost from memory-access cost using absolute runtime |
| pyramid | no valid cases | The ring workspace and UI are in place, but C3 interleaving, boundary rows, and up/downsample writeback still trail specialized upstream kernels | Reuse the current ring infrastructure and avoid full-image temporaries |
| geometry | no valid cases | Coordinate blocks are shared, but interpolation, border masks, and multi-channel gather/store still contain substantial scalar work | Extend U8 C1/C3/C4 interior SIMD without duplicating public kernels |
| reduction | no valid cases | Fast paths mainly cover F32 C1; the matrix also includes multi-channel, dual-input, and high-precision paths | Split gates by variant; do not trade precision for a better aggregate ratio |
| P2 random / point transform | OpenCV `~12.31x` | The v0.1 implementations are scalar public-header paths; upstream uses optimized RNG and transform kernels | Treat the focused P2-P0 report as optimization prioritization, not a release gate |
| P2 regions / contours / shape | OpenCV `~2.54x` | CVH currently favors explicit scalar correctness and deterministic ordering over specialized scans | Separate scan-heavy region work from micro shape primitives when selecting fast paths |
| P2 histogram / template | OpenCV `~6.07x` | Template matching now uses a method-specialized UI correlation path and squared-window integral; histogram construction and comparison remain scalar | Keep template matching correctness and dispatch coverage stable, then prioritize histogram construction |

## Operator-Level Overview

### `core_mat`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| PERSPECTIVE_TRANSFORM | public_header_scalar | 1 | 0.0142 | OpenCV `70.22x` |
| RANDN | public_header_scalar | 2 | 0.1251 | OpenCV `7.99x` |
| RANDU | public_header_scalar | 3 | 0.1804 | OpenCV `5.54x` |
| TRANSFORM | public_header_scalar | 1 | 0.0178 | OpenCV `56.11x` |

### `imgproc`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| APPROX_POLY_DP | public_header_scalar | 1 | 0.6335 | OpenCV `1.58x` |
| ARC_LENGTH | public_header_scalar | 1 | 0.8899 | OpenCV `1.12x` |
| BOUNDING_RECT | public_header_scalar | 1 | 2.5250 | CVH `2.52x` |
| CALC_HIST | public_header_scalar | 1 | 0.0130 | OpenCV `76.87x` |
| COMPARE_HIST | public_header_scalar | 4 | 0.0778 | OpenCV `12.86x` |
| CONNECTED_COMPONENTS | public_header_scalar | 1 | 0.0182 | OpenCV `55.03x` |
| CONNECTED_COMPONENTS_WITH_STATS | public_header_scalar | 1 | 0.0165 | OpenCV `60.50x` |
| CONTOUR_AREA | public_header_scalar | 1 | 0.9775 | OpenCV `1.02x` |
| CONVEX_HULL | public_header_scalar | 1 | 0.6153 | OpenCV `1.63x` |
| FIND_CONTOURS | public_header_scalar | 1 | 0.0780 | OpenCV `12.82x` |
| IS_CONTOUR_CONVEX | public_header_scalar | 1 | 4.5000 | CVH `4.50x` |
| MATCH_TEMPLATE | opencv_ui | 4 | 0.6575 | OpenCV `1.52x` |
| MOMENTS | public_header_scalar | 1 | 1.0041 | CVH `1.00x` |

## Detailed Results

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PERSPECTIVE_TRANSFORM | F32_C3 | public_header_scalar | CV_32F | 3 | continuous | 4096x1 | 0.556100 | 0.007919 | 0.0142 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| RANDN | C3 | public_header_scalar | CV_32F | 3 | continuous | 320x240 | 3.571360 | 0.447573 | 0.1253 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDN | C3 | public_header_scalar | CV_8U | 3 | continuous | 320x240 | 3.627673 | 0.452990 | 0.1249 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDU | C1 | public_header_scalar | CV_8U | 1 | roi | 320x240 | 0.360923 | 0.027602 | 0.0765 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDU | C3 | public_header_scalar | CV_32F | 3 | continuous | 320x240 | 0.793635 | 0.249763 | 0.3147 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDU | C3 | public_header_scalar | CV_8U | 3 | continuous | 320x240 | 1.023890 | 0.249608 | 0.2438 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| TRANSFORM | F32_C3_TO_C4 | public_header_scalar | CV_32F | 3 | continuous | 4096x1 | 0.541946 | 0.009658 | 0.0178 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| APPROX_POLY_DP | EPS_1_CLOSED | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.145640 | 0.092268 | 0.6335 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| ARC_LENGTH | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.002410 | 0.002145 | 0.8899 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| BOUNDING_RECT | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.000367 | 0.000928 | 2.5250 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CALC_HIST | U8C1_256 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 1.174713 | 0.015281 | 0.0130 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| COMPARE_HIST | METHOD_0 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001359 | 0.000144 | 0.1058 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COMPARE_HIST | METHOD_1 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001376 | 0.000122 | 0.0887 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COMPARE_HIST | METHOD_2 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001274 | 0.000057 | 0.0445 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COMPARE_HIST | METHOD_3 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001351 | 0.000118 | 0.0876 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CONNECTED_COMPONENTS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 1.646075 | 0.029910 | 0.0182 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONNECTED_COMPONENTS_WITH_STATS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 7.432546 | 0.122846 | 0.0165 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONTOUR_AREA | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.002403 | 0.002348 | 0.9775 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CONVEX_HULL | CCW_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.020970 | 0.012902 | 0.6153 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| FIND_CONTOURS | RETR_LIST_SIMPLE | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.784004 | 0.061162 | 0.0780 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| IS_CONTOUR_CONVEX | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.000003 | 0.000011 | 4.5000 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| MATCH_TEMPLATE | METHOD_0 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.427096 | 1.030410 | 0.7220 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_1 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.484846 | 1.043577 | 0.7028 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_2 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.287933 | 0.755929 | 0.5869 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_3 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.486858 | 0.932885 | 0.6274 | phase2_p0_representative_case;correctness=upstream_pass |
| MOMENTS | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.008432 | 0.008466 | 1.0041 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |

## Notes

- Ratios use `OpenCV time / CVH time`: values above `1` mean CVH is faster, and values below `1` mean OpenCV is faster.
- Table timings use the minimum per-iteration time across repeats to reduce system-noise effects; this report is not a cross-machine ranking.
- Mat cases compare matching allocation/reuse semantics; imgproc cases align input dimensions, types, kernels, borders, and primary parameters.
- `headers_baseline` describes a public header fallback for an operator without a UI kernel; it is not a separate target or implementation profile.
- Raw CSV and metadata files are reproducible run artifacts; date-named Markdown files are milestone snapshots.
