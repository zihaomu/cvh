# cvh vs OpenCV Benchmark Report (stable)

Generated at (UTC): `2026-08-04 07:30:48Z`

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
- Raw data: `2026-08-04-phase2-p0-a7-opencv-upstream-performance.csv`; metadata: `2026-08-04-phase2-p0-a7-opencv-upstream-performance.meta.json`

## Summary

- Total cases: `26`; valid: `26`; unsupported: `0`.
- `OpenCV/CVH` geometric mean: `0.7722`; median: `0.7331`.
- CVH faster: `5`; OpenCV faster or equal: `21`.

| Suite | Cases | geometric mean OpenCV/CVH | median | CVH faster | OpenCV faster/equal |
| --- | --- | --- | --- | --- | --- |
| core_mat | 7 | 0.5714 | 0.7179 | 1 | 6 |
| imgproc | 19 | 0.8628 | 0.7680 | 4 | 15 |

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
| P2 random / point transform | OpenCV `~1.75x` | Point transforms use prepacked channel-specialized spans; random fills use a lightweight 64-bit engine, hoisted distributions, and typed row kernels | Keep transform and random statistical/dispatch coverage stable |
| P2 regions / contours / shape | OpenCV `~1.02x` | Connected components use row-pointer union-find and fused statistics; contour discovery uses a mode-specialized row-indexed workspace | Keep label/statistics and contour ordering fixed, then continue with point transforms |
| P2 histogram / template | OpenCV `~1.33x` | Template matching uses UI correlation and squared-window integrals; histogram paths use typed scans and method-specialized double reductions | Keep histogram/template numeric and dispatch coverage stable, then continue with random fills |

## Operator-Level Overview

### `core_mat`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| PERSPECTIVE_TRANSFORM | header_fastpath | 1 | 0.9947 | OpenCV `1.01x` |
| RANDN | header_fastpath | 2 | 0.4001 | OpenCV `2.50x` |
| RANDU | header_fastpath | 3 | 0.4934 | OpenCV `2.03x` |
| TRANSFORM | header_fastpath | 1 | 1.0399 | CVH `1.04x` |

### `imgproc`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| APPROX_POLY_DP | public_header_scalar | 1 | 0.6622 | OpenCV `1.51x` |
| ARC_LENGTH | public_header_scalar | 1 | 0.8906 | OpenCV `1.12x` |
| BOUNDING_RECT | public_header_scalar | 1 | 2.5473 | CVH `2.55x` |
| CALC_HIST | header_fastpath | 1 | 0.6488 | OpenCV `1.54x` |
| COMPARE_HIST | header_fastpath | 4 | 0.8426 | OpenCV `1.19x` |
| CONNECTED_COMPONENTS | public_header_scalar | 1 | 0.4491 | OpenCV `2.23x` |
| CONNECTED_COMPONENTS_WITH_STATS | public_header_scalar | 1 | 0.7286 | OpenCV `1.37x` |
| CONTOUR_AREA | public_header_scalar | 1 | 0.9772 | OpenCV `1.02x` |
| CONVEX_HULL | public_header_scalar | 1 | 0.5844 | OpenCV `1.71x` |
| FIND_CONTOURS | public_header_scalar | 1 | 0.7171 | OpenCV `1.39x` |
| IS_CONTOUR_CONVEX | public_header_scalar | 1 | 3.7216 | CVH `3.72x` |
| MATCH_TEMPLATE | opencv_ui | 4 | 0.6944 | OpenCV `1.44x` |
| MOMENTS | public_header_scalar | 1 | 1.0644 | CVH `1.06x` |

## Detailed Results

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PERSPECTIVE_TRANSFORM | F32_C3 | header_fastpath | CV_32F | 3 | continuous | 4096x1 | 0.008617 | 0.008571 | 0.9947 | phase2_p0_representative_case;correctness=upstream_pass;coefficients=prepacked;channels=specialized;span=continuous |
| RANDN | C3 | header_fastpath | CV_32F | 3 | continuous | 320x240 | 1.237208 | 0.501938 | 0.4057 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDN | C3 | header_fastpath | CV_8U | 3 | continuous | 320x240 | 1.137348 | 0.448819 | 0.3946 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDU | C1 | header_fastpath | CV_8U | 1 | roi | 320x240 | 0.132013 | 0.029946 | 0.2268 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDU | C3 | header_fastpath | CV_32F | 3 | continuous | 320x240 | 0.340938 | 0.244744 | 0.7179 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDU | C3 | header_fastpath | CV_8U | 3 | continuous | 320x240 | 0.339850 | 0.250644 | 0.7375 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| TRANSFORM | F32_C3_TO_C4 | header_fastpath | CV_32F | 3 | continuous | 4096x1 | 0.010031 | 0.010431 | 1.0399 | phase2_p0_representative_case;correctness=upstream_pass;coefficients=prepacked;channels=specialized;span=continuous |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| APPROX_POLY_DP | EPS_1_CLOSED | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.198125 | 0.131200 | 0.6622 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| ARC_LENGTH | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.002973 | 0.002648 | 0.8906 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| BOUNDING_RECT | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.000449 | 0.001144 | 2.5473 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CALC_HIST | U8C1_256 | header_fastpath | CV_8U | 1 | continuous | 320x240 | 0.031140 | 0.020202 | 0.6488 | phase2_p0_representative_case;correctness=upstream_pass;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_0 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000186 | 0.000192 | 1.0359 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_1 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000199 | 0.000159 | 0.7987 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_2 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000110 | 0.000075 | 0.6806 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_3 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000175 | 0.000156 | 0.8950 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| CONNECTED_COMPONENTS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.073490 | 0.033006 | 0.4491 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONNECTED_COMPONENTS_WITH_STATS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.219854 | 0.160190 | 0.7286 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONTOUR_AREA | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.002961 | 0.002894 | 0.9772 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CONVEX_HULL | CCW_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.029090 | 0.017000 | 0.5844 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| FIND_CONTOURS | RETR_LIST_SIMPLE | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.109519 | 0.078535 | 0.7171 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| IS_CONTOUR_CONVEX | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.000003 | 0.000011 | 3.7216 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| MATCH_TEMPLATE | METHOD_0 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.171200 | 1.667460 | 0.7680 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_1 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.352329 | 2.316587 | 0.9848 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_2 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 3.414771 | 1.831046 | 0.5362 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_3 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.615396 | 1.499152 | 0.5732 | phase2_p0_representative_case;correctness=upstream_pass |
| MOMENTS | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 4096 points | 0.011059 | 0.011770 | 1.0644 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |

## Notes

- Ratios use `OpenCV time / CVH time`: values above `1` mean CVH is faster, and values below `1` mean OpenCV is faster.
- Table timings use the minimum per-iteration time across repeats to reduce system-noise effects; this report is not a cross-machine ranking.
- Mat cases compare matching allocation/reuse semantics; imgproc cases align input dimensions, types, kernels, borders, and primary parameters.
- `headers_baseline` describes a public header fallback for an operator without a UI kernel; it is not a separate target or implementation profile.
- Raw CSV and metadata files are reproducible run artifacts; date-named Markdown files are milestone snapshots.
