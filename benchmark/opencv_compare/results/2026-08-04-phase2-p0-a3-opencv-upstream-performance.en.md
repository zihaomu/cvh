# cvh vs OpenCV Benchmark Report (stable)

Generated at (UTC): `2026-08-04 04:33:45Z`

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
- Raw data: `2026-08-04-phase2-p0-a3-opencv-upstream-performance.csv`; metadata: `2026-08-04-phase2-p0-a3-opencv-upstream-performance.meta.json`

## Summary

- Total cases: `26`; valid: `26`; unsupported: `0`.
- `OpenCV/CVH` geometric mean: `0.3542`; median: `0.5732`.
- CVH faster: `4`; OpenCV faster or equal: `22`.

| Suite | Cases | geometric mean OpenCV/CVH | median | CVH faster | OpenCV faster/equal |
| --- | --- | --- | --- | --- | --- |
| core_mat | 7 | 0.2670 | 0.2465 | 1 | 6 |
| imgproc | 19 | 0.3930 | 0.6312 | 3 | 16 |

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
| P2 random / point transform | OpenCV `~3.75x` | Point transforms use prepacked coefficients and channel-specialized continuous spans; random fills remain scalar public-header paths | Keep point-transform numeric coverage stable and prioritize random-fill loop structure |
| P2 regions / contours / shape | OpenCV `~1.14x` | CVH currently favors explicit scalar correctness and deterministic ordering over specialized scans | Separate scan-heavy region work from micro shape primitives when selecting fast paths |
| P2 histogram / template | OpenCV `~6.18x` | Template matching now uses a method-specialized UI correlation path and squared-window integral; histogram construction and comparison remain scalar | Keep template matching correctness and dispatch coverage stable, then prioritize histogram construction |

## Operator-Level Overview

### `core_mat`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| PERSPECTIVE_TRANSFORM | header_fastpath | 1 | 0.9967 | OpenCV `1.00x` |
| RANDN | public_header_scalar | 2 | 0.1203 | OpenCV `8.31x` |
| RANDU | public_header_scalar | 3 | 0.1861 | OpenCV `5.37x` |
| TRANSFORM | header_fastpath | 1 | 1.0396 | CVH `1.04x` |

### `imgproc`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| APPROX_POLY_DP | public_header_scalar | 1 | 0.6312 | OpenCV `1.58x` |
| ARC_LENGTH | public_header_scalar | 1 | 0.8910 | OpenCV `1.12x` |
| BOUNDING_RECT | public_header_scalar | 1 | 2.5570 | CVH `2.56x` |
| CALC_HIST | public_header_scalar | 1 | 0.0129 | OpenCV `77.26x` |
| COMPARE_HIST | public_header_scalar | 4 | 0.0745 | OpenCV `13.42x` |
| CONNECTED_COMPONENTS | public_header_scalar | 1 | 0.5261 | OpenCV `1.90x` |
| CONNECTED_COMPONENTS_WITH_STATS | public_header_scalar | 1 | 0.2359 | OpenCV `4.24x` |
| CONTOUR_AREA | public_header_scalar | 1 | 0.9786 | OpenCV `1.02x` |
| CONVEX_HULL | public_header_scalar | 1 | 0.5640 | OpenCV `1.77x` |
| FIND_CONTOURS | public_header_scalar | 1 | 0.7487 | OpenCV `1.34x` |
| IS_CONTOUR_CONVEX | public_header_scalar | 1 | 3.5000 | CVH `3.50x` |
| MATCH_TEMPLATE | opencv_ui | 4 | 0.6604 | OpenCV `1.51x` |
| MOMENTS | public_header_scalar | 1 | 1.0035 | CVH `1.00x` |

## Detailed Results

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PERSPECTIVE_TRANSFORM | F32_C3 | header_fastpath | CV_32F | 3 | continuous | 4096x1 | 0.008240 | 0.008213 | 0.9967 | phase2_p0_representative_case;correctness=upstream_pass;coefficients=prepacked;channels=specialized;span=continuous |
| RANDN | C3 | public_header_scalar | CV_32F | 3 | continuous | 320x240 | 3.811263 | 0.437721 | 0.1148 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDN | C3 | public_header_scalar | CV_8U | 3 | continuous | 320x240 | 3.923267 | 0.494183 | 0.1260 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDU | C1 | public_header_scalar | CV_8U | 1 | roi | 320x240 | 0.376133 | 0.029827 | 0.0793 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDU | C3 | public_header_scalar | CV_32F | 3 | continuous | 320x240 | 0.809252 | 0.267071 | 0.3300 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| RANDU | C3 | public_header_scalar | CV_8U | 3 | continuous | 320x240 | 1.016735 | 0.250577 | 0.2465 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;no_ui_fastpath |
| TRANSFORM | F32_C3_TO_C4 | header_fastpath | CV_32F | 3 | continuous | 4096x1 | 0.009638 | 0.010019 | 1.0396 | phase2_p0_representative_case;correctness=upstream_pass;coefficients=prepacked;channels=specialized;span=continuous |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| APPROX_POLY_DP | EPS_1_CLOSED | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.157055 | 0.099140 | 0.6312 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| ARC_LENGTH | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.002596 | 0.002313 | 0.8910 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| BOUNDING_RECT | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.000391 | 0.001000 | 2.5570 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CALC_HIST | U8C1_256 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 1.276110 | 0.016517 | 0.0129 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| COMPARE_HIST | METHOD_0 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001498 | 0.000157 | 0.1048 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COMPARE_HIST | METHOD_1 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001582 | 0.000131 | 0.0830 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COMPARE_HIST | METHOD_2 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001439 | 0.000061 | 0.0423 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COMPARE_HIST | METHOD_3 | public_header_scalar | CV_32F | 1 | continuous | 256 bins | 0.001506 | 0.000126 | 0.0838 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CONNECTED_COMPONENTS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.061471 | 0.032340 | 0.5261 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONNECTED_COMPONENTS_WITH_STATS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.575460 | 0.135765 | 0.2359 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONTOUR_AREA | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.002590 | 0.002534 | 0.9786 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CONVEX_HULL | CCW_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.022615 | 0.012755 | 0.5640 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| FIND_CONTOURS | RETR_LIST_SIMPLE | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.093079 | 0.069687 | 0.7487 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| IS_CONTOUR_CONVEX | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.000003 | 0.000009 | 3.5000 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| MATCH_TEMPLATE | METHOD_0 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.440675 | 1.047367 | 0.7270 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_1 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.495140 | 1.063083 | 0.7110 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_2 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.298546 | 0.756275 | 0.5824 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_3 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 1.497160 | 0.945656 | 0.6316 | phase2_p0_representative_case;correctness=upstream_pass |
| MOMENTS | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.009092 | 0.009124 | 1.0035 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |

## Notes

- Ratios use `OpenCV time / CVH time`: values above `1` mean CVH is faster, and values below `1` mean OpenCV is faster.
- Table timings use the minimum per-iteration time across repeats to reduce system-noise effects; this report is not a cross-machine ranking.
- Mat cases compare matching allocation/reuse semantics; imgproc cases align input dimensions, types, kernels, borders, and primary parameters.
- `headers_baseline` describes a public header fallback for an operator without a UI kernel; it is not a separate target or implementation profile.
- Raw CSV and metadata files are reproducible run artifacts; date-named Markdown files are milestone snapshots.
