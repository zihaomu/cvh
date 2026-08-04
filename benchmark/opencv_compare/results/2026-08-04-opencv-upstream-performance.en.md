# cvh vs OpenCV Benchmark Report (full)

Generated at (UTC): `2026-08-04 07:32:02Z`

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

- Profile: `full`
- CVH implementation: `cvh_ui`
- Sampling: `warmup=1, iters=10, repeats=3`
- Threads: `1`
- OpenMP: `dynamic=false, proc_bind=close`
- Host: `Darwin arm64`
- CPU: `Apple M5`
- Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102)`
- Build type: `Release`
- CVH commit: `8360e586d8c004954a2cfd0b22ce1a1476cf9af9` + dirty
- OpenCV: `4.14.0`, commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- Raw data: `2026-08-04-opencv-upstream-performance.csv`; metadata: `2026-08-04-opencv-upstream-performance.meta.json`

## Summary

- Total cases: `370`; valid: `369`; unsupported: `1`.
- `OpenCV/CVH` geometric mean: `0.4808`; median: `0.6210`.
- CVH faster: `51`; OpenCV faster or equal: `318`.

| Suite | Cases | geometric mean OpenCV/CVH | median | CVH faster | OpenCV faster/equal |
| --- | --- | --- | --- | --- | --- |
| core_mat | 183 | 0.6273 | 0.7469 | 33 | 150 |
| imgproc | 186 | 0.3701 | 0.4608 | 18 | 168 |

## Performance Priorities

The multipliers below are within-group geometric means for this run. They prioritize follow-up work and do not indicate API support status.

| Area | This Report | Primary Cause | Follow-up Boundary |
| --- | --- | --- | --- |
| `GEMM` | OpenCV `~7.41x` | The default upstream build can use Accelerate/LAPACK; this is not a pure SIMD comparison against built-in OpenCV UI kernels | Keep the header-only boundary explicit when evaluating future improvements |
| filter / derivative | OpenCV `~8.32x` | CVH still has generic filter dispatch, border materialization, and intermediate-row processing; upstream specializes more deeply by type and kernel size | Prioritize shared row/column work and fused U8-to-S16/F32 kernels |
| nonlinear | OpenCV `~5.56x` | Repeated window scans are gone, but bilateral weight accumulation, the median lane network, and large-image cache behavior still lag | Separate pixel-kernel cost from memory-access cost using absolute runtime |
| pyramid | OpenCV `~4.82x` | The ring workspace and UI are in place, but C3 interleaving, boundary rows, and up/downsample writeback still trail specialized upstream kernels | Reuse the current ring infrastructure and avoid full-image temporaries |
| geometry | OpenCV `~3.33x` | Coordinate blocks are shared, but interpolation, border masks, and multi-channel gather/store still contain substantial scalar work | Extend U8 C1/C3/C4 interior SIMD without duplicating public kernels |
| reduction | OpenCV `~2.29x` | Fast paths mainly cover F32 C1; the matrix also includes multi-channel, dual-input, and high-precision paths | Split gates by variant; do not trade precision for a better aggregate ratio |
| P2 random / point transform | OpenCV `~1.81x` | Point transforms use prepacked channel-specialized spans; random fills use a lightweight 64-bit engine, hoisted distributions, and typed row kernels | Keep transform and random statistical/dispatch coverage stable |
| P2 regions / contours / shape | OpenCV `~1.06x` | Connected components use row-pointer union-find and fused statistics; contour discovery uses a mode-specialized row-indexed workspace | Keep label/statistics and contour ordering fixed, then continue with point transforms |
| P2 histogram / template | OpenCV `~1.38x` | Template matching uses UI correlation and squared-window integrals; histogram paths use typed scans and method-specialized double reductions | Keep histogram/template numeric and dispatch coverage stable, then continue with random fills |

## Operator-Level Overview

### `core_mat`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| ABSDIFF | opencv_ui | 1 | 0.7738 | OpenCV `1.29x` |
| ADD | opencv_ui | 16 | 0.7901 | OpenCV `1.27x` |
| BITWISE_AND | opencv_ui | 1 | 0.5761 | OpenCV `1.74x` |
| BITWISE_NOT | opencv_ui | 1 | 0.4755 | OpenCV `2.10x` |
| BITWISE_OR | opencv_ui | 1 | 0.7169 | OpenCV `1.39x` |
| BITWISE_XOR | opencv_ui | 1 | 0.5533 | OpenCV `1.81x` |
| BORDER_INTERPOLATE | public_header_baseline | 1 | 1.0310 | CVH `1.03x` |
| BROADCAST | scalar | 1 | 0.7762 | OpenCV `1.29x` |
| CHECK_RANGE | public_header_baseline | 1 | 0.6490 | OpenCV `1.54x` |
| CONVERT_FP16 | opencv_ui | 1 | 1.2219 | CVH `1.22x` |
| CONVERT_SCALE_ABS | opencv_ui | 1 | 0.5985 | OpenCV `1.67x` |
| COPY_TO | opencv_ui | 1 | 0.6210 | OpenCV `1.61x` |
| COUNT_NON_ZERO | opencv_ui | 1 | 1.0048 | CVH `1.00x` |
| DIVIDE | opencv_ui, scalar | 16 | 0.5035 | OpenCV `1.99x` |
| EXP | opencv_ui | 1 | 0.4817 | OpenCV `2.08x` |
| EXTRACT_CHANNEL | opencv_ui | 1 | 2.9128 | CVH `2.91x` |
| FIND_NON_ZERO | opencv_ui | 3 | 2.6488 | CVH `2.65x` |
| FLIP | opencv_ui | 1 | 0.9997 | OpenCV `1.00x` |
| FLIP_ND | opencv_ui | 1 | 7.9893 | CVH `7.99x` |
| GEMM | opencv_ui | 10 | 0.1350 | OpenCV `7.41x` |
| HAS_NON_ZERO | opencv_ui | 1 | 1.2530 | CVH `1.25x` |
| HCONCAT | scalar | 1 | 1.3684 | CVH `1.37x` |
| INSERT_CHANNEL | opencv_ui | 1 | 1.7097 | CVH `1.71x` |
| IN_RANGE | opencv_ui | 1 | 0.6154 | OpenCV `1.63x` |
| LOG | opencv_ui | 1 | 0.5822 | OpenCV `1.72x` |
| MAT_CLONE | headers_baseline | 4 | 0.9874 | OpenCV `1.01x` |
| MAT_CONVERTTO | headers_baseline | 4 | 1.0180 | CVH `1.02x` |
| MAT_COPYTO | headers_baseline | 4 | 0.9714 | OpenCV `1.03x` |
| MAT_CREATE | headers_baseline | 4 | 0.0737 | OpenCV `13.57x` |
| MAT_RESHAPE | headers_baseline | 4 | 0.3448 | OpenCV `2.90x` |
| MAT_SETTO | headers_baseline | 4 | 1.0026 | CVH `1.00x` |
| MAX | opencv_ui | 1 | 0.6754 | OpenCV `1.48x` |
| MEAN | opencv_ui | 1 | 2.0176 | CVH `2.02x` |
| MEAN_STD_DEV | opencv_ui | 1 | 0.3097 | OpenCV `3.23x` |
| MIN | opencv_ui | 1 | 0.6740 | OpenCV `1.48x` |
| MIN_MAX_IDX | opencv_ui | 1 | 0.7232 | OpenCV `1.38x` |
| MIN_MAX_LOC | opencv_ui | 1 | 0.7063 | OpenCV `1.42x` |
| MIX_CHANNELS | opencv_ui | 1 | 3.7924 | CVH `3.79x` |
| MULTIPLY | opencv_ui | 16 | 0.7733 | OpenCV `1.29x` |
| NORM | opencv_ui | 6 | 0.3043 | OpenCV `3.29x` |
| NORMALIZE | opencv_ui | 4 | 0.4692 | OpenCV `2.13x` |
| PATCH_NANS | opencv_ui | 1 | 0.8032 | OpenCV `1.25x` |
| PERSPECTIVE_TRANSFORM | header_fastpath | 1 | 1.0163 | CVH `1.02x` |
| POW | opencv_ui | 1 | 0.5957 | OpenCV `1.68x` |
| RANDN | header_fastpath | 2 | 0.3728 | OpenCV `2.68x` |
| RANDU | header_fastpath | 3 | 0.4724 | OpenCV `2.12x` |
| REDUCE | opencv_ui | 10 | 0.5625 | OpenCV `1.78x` |
| REDUCE_ARG_MAX | opencv_ui | 1 | 1.0048 | CVH `1.00x` |
| REDUCE_ARG_MIN | opencv_ui | 1 | 0.9480 | OpenCV `1.05x` |
| REPEAT | scalar | 1 | 1.0666 | CVH `1.07x` |
| ROTATE | opencv_ui | 1 | 0.4171 | OpenCV `2.40x` |
| SCALE_ADD | scalar | 1 | 0.8267 | OpenCV `1.21x` |
| SQRT | scalar | 1 | 0.9808 | OpenCV `1.02x` |
| SUBTRACT | opencv_ui | 16 | 0.7725 | OpenCV `1.29x` |
| SUM | opencv_ui | 1 | 1.9799 | CVH `1.98x` |
| SWAP | public_header_baseline | 1 | 1.0544 | CVH `1.05x` |
| TRANSFORM | header_fastpath | 1 | 1.0440 | CVH `1.04x` |
| TRANSPOSE | opencv_ui, scalar | 16 | 0.6324 | OpenCV `1.58x` |
| VCONCAT | scalar | 1 | 1.0451 | CVH `1.05x` |

### `imgproc`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| ACCUMULATE | opencv_ui | 1 | 0.2902 | OpenCV `3.45x` |
| ACCUMULATE_PRODUCT | opencv_ui | 1 | 0.3418 | OpenCV `2.93x` |
| ACCUMULATE_SQUARE | opencv_ui | 1 | 0.3549 | OpenCV `2.82x` |
| ACCUMULATE_WEIGHTED | opencv_ui | 1 | 0.3705 | OpenCV `2.70x` |
| ADAPTIVE_THRESHOLD | opencv_ui | 1 | 0.5982 | OpenCV `1.67x` |
| APPLY_COLOR_MAP | public_header_baseline | 1 | 0.3233 | OpenCV `3.09x` |
| APPROX_POLY_DP | public_header_scalar | 1 | 0.4895 | OpenCV `2.04x` |
| ARC_LENGTH | public_header_scalar | 1 | 0.8745 | OpenCV `1.14x` |
| BILATERAL_FILTER | public_header_baseline | 1 | 0.1478 | OpenCV `6.77x` |
| BLEND_LINEAR | public_header_baseline | 1 | 0.4053 | OpenCV `2.47x` |
| BOUNDING_RECT | public_header_scalar | 1 | 2.5437 | CVH `2.54x` |
| BOX_FILTER | box3x3, header_fastpath | 10 | 0.2042 | OpenCV `4.90x` |
| BUILD_PYRAMID | opencv_ui | 1 | 0.2353 | OpenCV `4.25x` |
| CALC_HIST | header_fastpath | 1 | 0.6459 | OpenCV `1.55x` |
| CANNY | header_fastpath | 4 | 0.5228 | OpenCV `1.91x` |
| COMPARE_HIST | header_fastpath | 4 | 0.8352 | OpenCV `1.20x` |
| CONNECTED_COMPONENTS | public_header_scalar | 1 | 0.5196 | OpenCV `1.92x` |
| CONNECTED_COMPONENTS_WITH_STATS | public_header_scalar | 1 | 0.7584 | OpenCV `1.32x` |
| CONTOUR_AREA | public_header_scalar | 1 | 0.9660 | OpenCV `1.04x` |
| CONVERT_MAPS | opencv_ui | 1 | 0.5774 | OpenCV `1.73x` |
| CONVEX_HULL | public_header_scalar | 1 | 0.4989 | OpenCV `2.00x` |
| COPY_MAKE_BORDER | header_fastpath | 9 | 0.9347 | OpenCV `1.07x` |
| CREATE_HANNING_WINDOW | opencv_ui | 1 | 1.9286 | CVH `1.93x` |
| CVTCOLOR | header_fastpath, opencv_ui | 17 | 0.2962 | OpenCV `3.38x` |
| CVT_COLOR_TWO_PLANE | public_header_baseline | 1 | 0.1949 | OpenCV `5.13x` |
| DEMOSAICING | public_header_baseline | 1 | 0.0900 | OpenCV `11.11x` |
| DILATE | header_fastpath | 6 | 0.2354 | OpenCV `4.25x` |
| EQUALIZE_HIST | opencv_ui | 1 | 0.9656 | OpenCV `1.04x` |
| ERODE | header_fastpath | 6 | 0.2371 | OpenCV `4.22x` |
| FILTER2D | header_fastpath | 10 | 0.0824 | OpenCV `12.13x` |
| FIND_CONTOURS | public_header_scalar | 1 | 0.7296 | OpenCV `1.37x` |
| GAUSSIAN | gauss_separable, header_fastpath | 10 | 0.0799 | OpenCV `12.52x` |
| GET_AFFINE_TRANSFORM | public_header_baseline | 1 | 2.0528 | CVH `2.05x` |
| GET_DERIV_KERNELS | public_header_baseline | 1 | 0.6159 | OpenCV `1.62x` |
| GET_GABOR_KERNEL | public_header_baseline | 1 | 0.9135 | OpenCV `1.09x` |
| GET_GAUSSIAN_KERNEL | public_header_baseline | 1 | 4.1104 | CVH `4.11x` |
| GET_PERSPECTIVE_TRANSFORM | public_header_baseline | 1 | 2.4840 | CVH `2.48x` |
| GET_RECT_SUB_PIX | public_header_scalar | 4 | 11.3142 | CVH `11.31x` |
| GET_ROTATION_MATRIX_2D | public_header_baseline | 1 | 0.9999 | OpenCV `1.00x` |
| GET_ROTATION_MATRIX_2D_ | public_header_baseline | 1 | 1.0364 | CVH `1.04x` |
| GET_STRUCTURING_ELEMENT | public_header_baseline | 1 | 0.7843 | OpenCV `1.28x` |
| INTEGRAL | opencv_ui | 1 | 0.5605 | OpenCV `1.78x` |
| INVERT_AFFINE_TRANSFORM | public_header_baseline | 1 | 1.5542 | CVH `1.55x` |
| IS_CONTOUR_CONVEX | public_header_scalar | 1 | 3.8880 | CVH `3.89x` |
| LAPLACIAN | opencv_ui | 1 | 0.0661 | OpenCV `15.14x` |
| LUT | header_fastpath | 6 | 0.7843 | OpenCV `1.28x` |
| MATCH_TEMPLATE | opencv_ui | 4 | 0.6442 | OpenCV `1.55x` |
| MEDIAN_BLUR | opencv_ui | 1 | 0.1756 | OpenCV `5.70x` |
| MOMENTS | public_header_scalar | 1 | 0.9886 | OpenCV `1.01x` |
| PYR_DOWN | opencv_ui | 1 | 0.1850 | OpenCV `5.41x` |
| PYR_UP | opencv_ui | 1 | 0.2050 | OpenCV `4.88x` |
| REMAP | fixed_coordinate_block | 8 | 0.3988 | OpenCV `2.51x` |
| RESIZE | header_fastpath, headers_baseline, opencv_ui | 10 | 0.4963 | OpenCV `2.01x` |
| SCHARR | opencv_ui | 1 | 0.0626 | OpenCV `15.98x` |
| SEP_FILTER2D | header_fastpath | 10 | 0.1690 | OpenCV `5.92x` |
| SOBEL | header_fastpath | 6 | 0.4424 | OpenCV `2.26x` |
| SPATIAL_GRADIENT | opencv_ui | 1 | 0.1799 | OpenCV `5.56x` |
| SQR_BOX_FILTER | opencv_ui | 1 | 0.4167 | OpenCV `2.40x` |
| STACK_BLUR | public_header_baseline | 1 | 0.2242 | OpenCV `4.46x` |
| THRESHOLD | header_fastpath, headers_baseline | 5 | 0.9735 | OpenCV `1.03x` |
| THRESHOLD_WITH_MASK | public_header_baseline | 1 | 1.0427 | CVH `1.04x` |
| WARP_AFFINE | fixed_coordinate_block, headers_baseline | 9 | 0.1800 | OpenCV `5.56x` |
| WARP_PERSPECTIVE | fixed_coordinate_block | 4 | 0.4579 | OpenCV `2.18x` |

## Detailed Results

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.049296 | 0.038146 | 0.7738 | phase1_representative_case |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.516387 | 0.415679 | 0.8050 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.057612 | 0.039083 | 0.6784 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.041283 | 0.030125 | 0.7297 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.209421 | 0.160971 | 0.7686 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 1.297925 | 1.085758 | 0.8365 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.162729 | 0.125588 | 0.7718 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.119146 | 0.091333 | 0.7666 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.777646 | 0.622796 | 0.8009 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.307683 | 0.286029 | 0.9296 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.013329 | 0.010021 | 0.7518 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009433 | 0.007046 | 0.7469 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.028725 | 0.028633 | 0.9968 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.790775 | 0.491175 | 0.6211 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.040983 | 0.031117 | 0.7593 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027825 | 0.022075 | 0.7934 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.097517 | 0.095196 | 0.9762 | correctness=upstream_pass |
| BITWISE_AND | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.054087 | 0.031158 | 0.5761 | phase1_representative_case |
| BITWISE_NOT | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.052492 | 0.024958 | 0.4755 | phase1_representative_case |
| BITWISE_OR | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.054004 | 0.038717 | 0.7169 | phase1_representative_case |
| BITWISE_XOR | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.053975 | 0.029862 | 0.5533 | phase1_representative_case |
| BORDER_INTERPOLATE | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.015197 | 0.015669 | 1.0310 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| BROADCAST | row_to_image_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.006721 | 0.005217 | 0.7762 | phase1_representative_case |
| CHECK_RANGE | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.259071 | 0.168129 | 0.6490 | phase1_representative_case |
| CONVERT_FP16 | f32c1_to_fp16 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.025271 | 0.030879 | 1.2219 | phase1_representative_case |
| CONVERT_SCALE_ABS | f32c3_to_u8c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.174167 | 0.104242 | 0.5985 | phase1_representative_case |
| COPY_TO | masked_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.061921 | 0.038450 | 0.6210 | phase1_representative_case |
| COUNT_NON_ZERO | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.014779 | 0.014850 | 1.0048 | phase1_representative_case |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.499692 | 0.418238 | 0.8370 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.056517 | 0.041608 | 0.7362 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.043083 | 0.030250 | 0.7021 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.232000 | 0.183138 | 0.7894 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 1.326225 | 1.109004 | 0.8362 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.161679 | 0.128062 | 0.7921 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.117500 | 0.098375 | 0.8372 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 1.356079 | 1.173600 | 0.8654 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 1080x1920 | 6.650471 | 1.632629 | 0.2455 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 479x641 | 0.373392 | 0.094588 | 0.2533 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 480x640 | 0.263217 | 0.068304 | 0.2595 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 720x1280 | 0.938433 | 0.231817 | 0.2470 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 9.257346 | 3.442596 | 0.3719 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.746971 | 0.284150 | 0.3804 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.538675 | 0.203012 | 0.3769 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 2.289271 | 1.122142 | 0.4902 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | bounded_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.265092 | 0.127683 | 0.4817 | phase1_representative_case |
| EXTRACT_CHANNEL | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.026471 | 0.077104 | 2.9128 | phase1_representative_case |
| FIND_NON_ZERO | all_zero_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018204 | 0.126279 | 6.9369 | phase1_representative_case |
| FIND_NON_ZERO | random_dense_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.876596 | 0.334263 | 0.3813 | phase1_representative_case |
| FIND_NON_ZERO | sparse_tail_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018233 | 0.128108 | 7.0261 | phase1_representative_case |
| FLIP | horizontal_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.028375 | 0.028367 | 0.9997 | phase1_representative_case |
| FLIP_ND | axis1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.028333 | 0.226363 | 7.9893 | phase1_representative_case |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.078029 | 0.004900 | 0.0628 | correctness=upstream_pass;shape=square_128;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.664504 | 0.029025 | 0.0437 | correctness=upstream_pass;shape=square_256;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x32x256 | 0.062417 | 0.006925 | 0.1109 | correctness=upstream_pass;shape=wide_m256_k32_n256;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 32x512x64 | 0.042533 | 0.152667 | 3.5893 | correctness=upstream_pass;shape=skinny_m32_k512_n64;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 5.630166 | 0.234959 | 0.0417 | correctness=upstream_pass;shape=square_512;component=public_end_to_end;iters=1 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.077642 | 0.004771 | 0.0614 | correctness=upstream_pass;shape=square_128;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.668292 | 0.028675 | 0.0429 | correctness=upstream_pass;shape=square_256;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x32x256 | 0.062575 | 0.007225 | 0.1155 | correctness=upstream_pass;shape=wide_m256_k32_n256;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 32x512x64 | 0.042550 | 0.152954 | 3.5947 | correctness=upstream_pass;shape=skinny_m32_k512_n64;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 5.502250 | 0.222208 | 0.0404 | correctness=upstream_pass;shape=square_512;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.000017 | 0.000021 | 1.2530 | phase1_representative_case |
| HCONCAT | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.007850 | 0.010742 | 1.3684 | phase1_representative_case |
| INSERT_CHANNEL | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.043396 | 0.074196 | 1.7097 | phase1_representative_case |
| IN_RANGE | scalar_bounds_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.305971 | 0.188287 | 0.6154 | phase1_representative_case |
| LOG | positive_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.372688 | 0.216983 | 0.5822 | phase1_representative_case |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024633 | 0.025879 | 1.0506 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003846 | 0.003679 | 0.9567 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.003983 | 0.003833 | 0.9623 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.012079 | 0.011871 | 0.9828 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.081658 | 0.082225 | 1.0069 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.012308 | 0.013596 | 1.1046 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.012600 | 0.011967 | 0.9497 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.035867 | 0.036462 | 1.0166 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.025492 | 0.024588 | 0.9645 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003354 | 0.003463 | 1.0323 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.003758 | 0.003438 | 0.9146 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.010675 | 0.010437 | 0.9778 |  |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000015 | 0.000001 | 0.0641 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000014 | 0.000001 | 0.0815 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000017 | 0.000001 | 0.0685 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000014 | 0.000001 | 0.0824 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000044 | 0.000015 | 0.3509 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000044 | 0.000016 | 0.3484 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000046 | 0.000016 | 0.3363 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000045 | 0.000016 | 0.3439 | micro_iters_x1000 |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.025771 | 0.024833 | 0.9636 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004225 | 0.003933 | 0.9310 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.004038 | 0.004462 | 1.1053 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.011542 | 0.011762 | 1.0191 |  |
| MAX | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.056612 | 0.038233 | 0.6754 | phase1_representative_case |
| MEAN | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.166796 | 0.336521 | 2.0176 | phase1_representative_case |
| MEAN_STD_DEV | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.713029 | 0.220842 | 0.3097 | phase1_representative_case |
| MIN | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.056600 | 0.038150 | 0.6740 | phase1_representative_case |
| MIN_MAX_IDX | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.077446 | 0.056008 | 0.7232 | phase1_representative_case |
| MIN_MAX_LOC | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.078242 | 0.055263 | 0.7063 | phase1_representative_case |
| MIX_CHANNELS | reverse_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.062104 | 0.235525 | 3.7924 | phase1_representative_case |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.505317 | 0.387358 | 0.7666 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.057904 | 0.041833 | 0.7225 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.040583 | 0.030212 | 0.7445 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.210629 | 0.176700 | 0.8389 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 1.324054 | 1.077683 | 0.8139 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.160083 | 0.132183 | 0.8257 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.121242 | 0.093404 | 0.7704 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 1.197904 | 0.856667 | 0.7151 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.285283 | 0.246758 | 0.8650 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.013496 | 0.009967 | 0.7385 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009562 | 0.007054 | 0.7377 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.030300 | 0.021596 | 0.7127 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.694796 | 0.632271 | 0.9100 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.039117 | 0.031150 | 0.7963 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027221 | 0.022104 | 0.8120 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.137467 | 0.088837 | 0.6462 | correctness=upstream_pass |
| NORM | inf_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.154383 | 0.034162 | 0.2213 | phase1_representative_case |
| NORM | inf_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.064433 | 0.018325 | 0.2844 | phase1_representative_case |
| NORM | l1_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.154417 | 0.045371 | 0.2938 | phase1_representative_case |
| NORM | l1_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.153954 | 0.038317 | 0.2489 | phase1_representative_case |
| NORM | l2_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.154367 | 0.058262 | 0.3774 | phase1_representative_case |
| NORM | l2_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.126829 | 0.058008 | 0.4574 | phase1_representative_case |
| NORMALIZE | inf_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.124608 | 0.050454 | 0.4049 | phase1_representative_case |
| NORMALIZE | l1_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.184413 | 0.071946 | 0.3901 | phase1_representative_case |
| NORMALIZE | l2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.215963 | 0.092658 | 0.4290 | phase1_representative_case |
| NORMALIZE | minmax_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.138629 | 0.099167 | 0.7153 | phase1_representative_case |
| PATCH_NANS | one_nan_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.032517 | 0.026117 | 0.8032 | phase1_representative_case |
| PERSPECTIVE_TRANSFORM | F32_C3 | header_fastpath | CV_32F | 3 | continuous | 16384x1 | 0.065933 | 0.067008 | 1.0163 | phase2_p0_representative_case;correctness=upstream_pass;coefficients=prepacked;channels=specialized;span=continuous |
| POW | power_1_75_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.838512 | 0.499525 | 0.5957 | phase1_representative_case |
| RANDN | C3 | header_fastpath | CV_32F | 3 | continuous | 320x240 | 2.460133 | 0.971508 | 0.3949 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDN | C3 | header_fastpath | CV_8U | 3 | continuous | 320x240 | 2.740221 | 0.964171 | 0.3519 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDU | C1 | header_fastpath | CV_8U | 1 | roi | 320x240 | 0.250683 | 0.057671 | 0.2301 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDU | C3 | header_fastpath | CV_32F | 3 | continuous | 320x240 | 0.736996 | 0.529846 | 0.7189 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| RANDU | C3 | header_fastpath | CV_8U | 3 | continuous | 320x240 | 1.088200 | 0.693646 | 0.6374 | phase2_p0_representative_case;correctness=upstream_pass;random_streams=independent;shape_and_range=aligned;engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed |
| REDUCE | axis0_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.116179 | 0.025688 | 0.2211 | phase1_representative_case |
| REDUCE | axis0_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.065221 | 0.029567 | 0.4533 | phase1_representative_case |
| REDUCE | axis0_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.064662 | 0.029517 | 0.4565 | phase1_representative_case |
| REDUCE | axis0_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.111658 | 0.029675 | 0.2658 | phase1_representative_case |
| REDUCE | axis0_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.113483 | 0.024417 | 0.2152 | phase1_representative_case |
| REDUCE | axis1_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.086292 | 0.017638 | 0.2044 | phase1_representative_case |
| REDUCE | axis1_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.095179 | 0.262742 | 2.7605 | phase1_representative_case |
| REDUCE | axis1_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.094412 | 0.262429 | 2.7796 | phase1_representative_case |
| REDUCE | axis1_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.116550 | 0.424096 | 3.6387 | phase1_representative_case |
| REDUCE | axis1_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.086225 | 0.018304 | 0.2123 | phase1_representative_case |
| REDUCE_ARG_MAX | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.249562 | 0.250762 | 1.0048 | phase1_representative_case |
| REDUCE_ARG_MIN | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.244533 | 0.231813 | 0.9480 | phase1_representative_case |
| REPEAT | two_by_two_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.007879 | 0.008404 | 1.0666 | phase1_representative_case |
| ROTATE | clockwise90_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.407271 | 0.169879 | 0.4171 | phase1_representative_case |
| SCALE_ADD | f32c3 | scalar | CV_32F | 3 | continuous | 480x640 | 0.186583 | 0.154246 | 0.8267 | phase1_representative_case |
| SQRT | positive_f32c1 | scalar | CV_32F | 1 | continuous | 480x640 | 0.064633 | 0.063392 | 0.9808 | phase1_representative_case |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.493179 | 0.396100 | 0.8032 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.056458 | 0.042171 | 0.7469 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.039987 | 0.029925 | 0.7484 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.205304 | 0.144467 | 0.7037 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 1.335133 | 1.086858 | 0.8140 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.162221 | 0.125883 | 0.7760 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.118971 | 0.098583 | 0.8286 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.762717 | 0.635158 | 0.8328 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.278471 | 0.247871 | 0.8901 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.014142 | 0.009963 | 0.7045 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009571 | 0.007275 | 0.7601 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.038208 | 0.023950 | 0.6268 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.750171 | 0.891979 | 1.1890 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.041688 | 0.030179 | 0.7239 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027758 | 0.022200 | 0.7998 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.138996 | 0.078846 | 0.5673 | correctness=upstream_pass |
| SUM | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.164596 | 0.325875 | 1.9799 | phase1_representative_case |
| SWAP | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000008 | 0.000008 | 1.0544 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| TRANSFORM | F32_C3_TO_C4 | header_fastpath | CV_32F | 3 | continuous | 16384x1 | 0.077012 | 0.080404 | 1.0440 | phase2_p0_representative_case;correctness=upstream_pass;coefficients=prepacked;channels=specialized;span=continuous |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 1.162483 | 1.085163 | 0.9335 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.044942 | 0.044446 | 0.9890 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.075350 | 0.072808 | 0.9663 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.562883 | 0.557417 | 0.9903 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 1080x1920 | 4.351325 | 2.137329 | 0.4912 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 479x641 | 0.397725 | 0.166187 | 0.4178 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 480x640 | 0.353025 | 0.165087 | 0.4676 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 720x1280 | 4.323775 | 2.154358 | 0.4983 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.260779 | 0.389404 | 1.4932 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.014467 | 0.013167 | 0.9101 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.007208 | 0.006696 | 0.9289 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.056921 | 0.029988 | 0.5268 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 3.778842 | 1.464142 | 0.3875 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.336654 | 0.114704 | 0.3407 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.240542 | 0.083538 | 0.3473 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 1.362117 | 0.692583 | 0.5085 | correctness=upstream_pass |
| VCONCAT | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.005917 | 0.006183 | 1.0451 | phase1_representative_case |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.169063 | 0.049067 | 0.2902 | phase1_representative_case |
| ACCUMULATE_PRODUCT | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.158883 | 0.054300 | 0.3418 | phase1_representative_case |
| ACCUMULATE_SQUARE | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.157679 | 0.055963 | 0.3549 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | alpha0_1_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.157329 | 0.058296 | 0.3705 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | mean11_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.965546 | 0.577571 | 0.5982 | phase1_representative_case |
| APPLY_COLOR_MAP | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.913025 | 0.618454 | 0.3233 | phase1_representative_case |
| APPROX_POLY_DP | EPS_1_CLOSED | public_header_scalar | CV_32S | 2 | vector | 16384 points | 3.087329 | 1.511116 | 0.4895 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| ARC_LENGTH | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 16384 points | 0.023378 | 0.020444 | 0.8745 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| BILATERAL_FILTER | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 18.614325 | 2.751346 | 0.1478 | phase1_representative_case |
| BLEND_LINEAR | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 1.176208 | 0.476667 | 0.4053 | phase1_representative_case |
| BOUNDING_RECT | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 16384 points | 0.003319 | 0.008443 | 2.5437 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 3.335971 | 0.395621 | 0.1186 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.474196 | 0.064208 | 0.1354 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.433133 | 0.059583 | 0.1376 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.614050 | 0.200825 | 0.1244 |  |
| BOX_FILTER | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.762342 | 0.189392 | 0.2484 |  |
| BOX_FILTER | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 1.230992 | 0.654504 | 0.5317 |  |
| BOX_FILTER | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 1.356092 | 0.833225 | 0.6144 |  |
| BOX_FILTER | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 1.144800 | 0.185542 | 0.1621 |  |
| BOX_FILTER | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 1.303767 | 0.241333 | 0.1851 |  |
| BOX_FILTER | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 1.292408 | 0.243617 | 0.1885 |  |
| BUILD_PYRAMID | levels3_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.175912 | 0.041396 | 0.2353 | phase1_representative_case |
| CALC_HIST | U8C1_256 | header_fastpath | CV_8U | 1 | continuous | 320x240 | 0.040987 | 0.026475 | 0.6459 | phase2_p0_representative_case;correctness=upstream_pass;method=split;rows=typed;accumulator=local |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 67.167525 | 34.522179 | 0.5140 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 11.529804 | 5.892971 | 0.5111 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 11.173308 | 6.473013 | 0.5793 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 37.132575 | 18.227321 | 0.4909 |  |
| COMPARE_HIST | METHOD_0 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000245 | 0.000250 | 1.0221 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_1 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000266 | 0.000209 | 0.7853 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_2 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000145 | 0.000099 | 0.6859 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| COMPARE_HIST | METHOD_3 | header_fastpath | CV_32F | 1 | continuous | 256 bins | 0.000230 | 0.000203 | 0.8840 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;method=split;rows=typed;accumulator=local |
| CONNECTED_COMPONENTS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.124279 | 0.064579 | 0.5196 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONNECTED_COMPONENTS_WITH_STATS | SPARSE_8 | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.384183 | 0.291371 | 0.7584 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| CONTOUR_AREA | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 16384 points | 0.022482 | 0.021717 | 0.9660 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| CONVERT_MAPS | f32_pair_to_fixed | opencv_ui | CV_32F | 2 | continuous | 480x640 | 0.269008 | 0.155325 | 0.5774 | phase1_representative_case |
| CONVEX_HULL | CCW_POINTS | public_header_scalar | CV_32S | 2 | vector | 16384 points | 0.152439 | 0.076055 | 0.4989 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.063779 | 0.061713 | 0.9676 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.010446 | 0.009471 | 0.9067 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.009663 | 0.008775 | 0.9082 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.032333 | 0.031183 | 0.9644 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.049771 | 0.048638 | 0.9772 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.197408 | 0.219246 | 1.1106 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.230708 | 0.218225 | 0.9459 |  |
| COPY_MAKE_BORDER | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.033983 | 0.025704 | 0.7564 |  |
| COPY_MAKE_BORDER | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.040150 | 0.036658 | 0.9130 |  |
| CREATE_HANNING_WINDOW | 64x64_f32 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.001050 | 0.002025 | 1.9286 | phase1_representative_case |
| CVTCOLOR | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.433292 | 0.031042 | 0.0716 |  |
| CVTCOLOR | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.137829 | 0.066025 | 0.4790 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.356567 | 0.356329 | 0.9993 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.054837 | 0.053983 | 0.9844 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.042612 | 0.042608 | 0.9999 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.153250 | 0.139008 | 0.9071 |  |
| CVTCOLOR | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.052267 | 0.052675 | 1.0078 |  |
| CVTCOLOR | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.607575 | 0.092196 | 0.1517 |  |
| CVTCOLOR | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.239117 | 0.101671 | 0.4252 |  |
| CVTCOLOR | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.203696 | 0.025921 | 0.1273 |  |
| CVTCOLOR | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.481529 | 0.120950 | 0.2512 |  |
| CVTCOLOR | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.549971 | 0.098296 | 0.1787 |  |
| CVTCOLOR | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.279983 | 0.061650 | 0.2202 |  |
| CVTCOLOR | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 1.470375 | 0.113458 | 0.0772 |  |
| CVTCOLOR | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.666908 | 0.114250 | 0.1713 |  |
| CVTCOLOR | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.499408 | 0.091213 | 0.1826 |  |
| CVTCOLOR | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.654533 | 0.112637 | 0.1721 |  |
| CVT_COLOR_TWO_PLANE | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.785517 | 0.153113 | 0.1949 | phase1_representative_case |
| DEMOSAICING | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.988108 | 0.088908 | 0.0900 | phase1_representative_case |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.219529 | 0.154537 | 0.7039 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.050562 | 0.030483 | 0.6029 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.059612 | 0.036571 | 0.6135 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.124900 | 0.095725 | 0.7664 |  |
| DILATE | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 4.572463 | 0.139021 | 0.0304 |  |
| DILATE | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 4.857225 | 0.136325 | 0.0281 |  |
| EQUALIZE_HIST | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.243250 | 0.234879 | 0.9656 | phase1_representative_case |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.222425 | 0.156104 | 0.7018 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.050958 | 0.030437 | 0.5973 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.059917 | 0.037325 | 0.6229 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.124267 | 0.086417 | 0.6954 |  |
| ERODE | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 4.784825 | 0.148363 | 0.0310 |  |
| ERODE | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 5.194783 | 0.163896 | 0.0316 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 14.790513 | 1.750671 | 0.1184 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 2.302579 | 0.131408 | 0.0571 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 2.034758 | 0.121500 | 0.0597 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 7.433467 | 0.426837 | 0.0574 |  |
| FILTER2D | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 3.702125 | 0.143079 | 0.0386 |  |
| FILTER2D | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 5.469958 | 0.477879 | 0.0874 |  |
| FILTER2D | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 4.566246 | 0.518492 | 0.1135 |  |
| FILTER2D | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 3.316371 | 0.353362 | 0.1066 |  |
| FILTER2D | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 4.579546 | 0.437917 | 0.0956 |  |
| FILTER2D | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 3.758704 | 0.602187 | 0.1602 |  |
| FIND_CONTOURS | RETR_LIST_SIMPLE | public_header_scalar | CV_8U | 1 | continuous | 320x240 | 0.197175 | 0.143863 | 0.7296 | phase2_p0_representative_case;correctness=upstream_pass;no_ui_fastpath |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 11.054721 | 0.342617 | 0.0310 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 1.499350 | 0.044229 | 0.0295 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 1.352796 | 0.039217 | 0.0290 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 4.483167 | 0.146237 | 0.0326 |  |
| GAUSSIAN | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 2.986758 | 0.203063 | 0.0680 |  |
| GAUSSIAN | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 2.537192 | 0.678462 | 0.2674 |  |
| GAUSSIAN | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 1.957854 | 0.893117 | 0.4562 |  |
| GAUSSIAN | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 2.229658 | 0.135121 | 0.0606 |  |
| GAUSSIAN | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 2.798704 | 0.628175 | 0.2245 |  |
| GAUSSIAN | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 1.540058 | 0.166762 | 0.1083 |  |
| GET_AFFINE_TRANSFORM | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000340 | 0.000697 | 2.0528 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_DERIV_KERNELS | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000241 | 0.000148 | 0.6159 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GABOR_KERNEL | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.001817 | 0.001660 | 0.9135 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GAUSSIAN_KERNEL | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000400 | 0.001644 | 4.1104 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_PERSPECTIVE_TRANSFORM | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000704 | 0.001749 | 2.4840 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 0.157133 | 1.904896 | 12.1228 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 0.026308 | 0.223300 | 8.4878 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 0.018817 | 0.250271 | 13.3005 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 0.055850 | 0.668729 | 11.9737 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000183 | 0.000183 | 0.9999 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_ROTATION_MATRIX_2D_ | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000011 | 0.000012 | 1.0364 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_STRUCTURING_ELEMENT | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000170 | 0.000133 | 0.7843 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| INTEGRAL | u8c1_to_s32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.124679 | 0.069883 | 0.5605 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000100 | 0.000155 | 1.5542 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| IS_CONTOUR_CONVEX | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 16384 points | 0.000004 | 0.000015 | 3.8880 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| LAPLACIAN | ksize3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 2.937192 | 0.194008 | 0.0661 | phase1_representative_case |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.336725 | 0.271904 | 0.8075 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.051263 | 0.039500 | 0.7705 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.046383 | 0.036275 | 0.7821 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.168404 | 0.132829 | 0.7888 |  |
| LUT | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.235025 | 0.184025 | 0.7830 |  |
| LUT | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.260038 | 0.201350 | 0.7743 |  |
| MATCH_TEMPLATE | METHOD_0 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.493821 | 1.815108 | 0.7278 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_1 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.603000 | 1.831617 | 0.7037 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_2 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.233408 | 1.193688 | 0.5345 | phase2_p0_representative_case;correctness=upstream_pass |
| MATCH_TEMPLATE | METHOD_3 | opencv_ui | CV_8U | 1 | continuous | 320x240/16x16 | 2.310283 | 1.453367 | 0.6291 | phase2_p0_representative_case;correctness=upstream_pass |
| MEDIAN_BLUR | ksize5_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 2.542300 | 0.446321 | 0.1756 | phase1_representative_case |
| MOMENTS | S32_POINTS | public_header_scalar | CV_32S | 2 | vector | 16384 points | 0.059198 | 0.058523 | 0.9886 | phase2_p0_representative_case;correctness=upstream_pass;micro_warmup=2;micro_iterations=100;micro_repeats=3;no_ui_fastpath |
| PYR_DOWN | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.764042 | 0.141354 | 0.1850 | phase1_representative_case |
| PYR_UP | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 1.657463 | 0.339838 | 0.2050 | phase1_representative_case |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 17.379646 | 8.015521 | 0.4612 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 3.308604 | 1.300621 | 0.3931 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 3.072867 | 1.141042 | 0.3713 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 7.761408 | 3.311121 | 0.4266 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 16.374508 | 6.118421 | 0.3737 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 3.376263 | 1.527013 | 0.4523 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 3.612338 | 1.266604 | 0.3506 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 8.599192 | 3.232671 | 0.3759 | Shared fixed coordinate block and U8 bilinear sampling path |
| RESIZE | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.303775 | 0.141338 | 0.4653 |  |
| RESIZE | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.682492 | 0.425971 | 0.6241 |  |
| RESIZE | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.580704 | 0.098392 | 0.1694 |  |
| RESIZE | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.576213 | 0.096317 | 0.1672 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.189233 | 0.124321 | 0.6570 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.034875 | 0.022729 | 0.6517 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.025704 | 0.016996 | 0.6612 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.093762 | 0.058046 | 0.6191 |  |
| RESIZE | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.204783 | 0.158363 | 0.7733 |  |
| RESIZE | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.204808 | 0.166554 | 0.8132 |  |
| SCHARR | dx1_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 2.960583 | 0.185300 | 0.0626 | phase1_representative_case |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 13.415592 | 3.612733 | 0.2693 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.930433 | 0.131183 | 0.1410 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.832329 | 0.120383 | 0.1446 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 3.053088 | 0.430408 | 0.1410 |  |
| SEP_FILTER2D | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 2.047779 | 0.166525 | 0.0813 |  |
| SEP_FILTER2D | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 3.439462 | 0.544854 | 0.1584 |  |
| SEP_FILTER2D | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 2.574579 | 0.567504 | 0.2204 |  |
| SEP_FILTER2D | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 1.765696 | 0.321317 | 0.1820 |  |
| SEP_FILTER2D | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 2.569608 | 0.432767 | 0.1684 |  |
| SEP_FILTER2D | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.043971 | 0.575108 | 0.2814 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 4.877958 | 2.245754 | 0.4604 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.370821 | 0.173896 | 0.4689 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.343988 | 0.152933 | 0.4446 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.411146 | 0.483342 | 0.3425 |  |
| SOBEL | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 1.463042 | 0.696275 | 0.4759 |  |
| SOBEL | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 1.716575 | 0.822458 | 0.4791 |  |
| SPATIAL_GRADIENT | ksize3_u8_to_s16 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.315525 | 0.056750 | 0.1799 | phase1_representative_case |
| SQR_BOX_FILTER | 3x3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.749858 | 0.312442 | 0.4167 | phase1_representative_case |
| STACK_BLUR | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.644496 | 0.368717 | 0.2242 | phase1_representative_case |
| THRESHOLD | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.116958 | 0.102842 | 0.8793 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.058092 | 0.057783 | 0.9947 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.008346 | 0.008375 | 1.0035 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.006917 | 0.006825 | 0.9867 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.022404 | 0.022621 | 1.0097 |  |
| THRESHOLD_WITH_MASK | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.434825 | 0.453387 | 1.0427 | phase1_representative_case |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 1080x1920 | 13.685704 | 2.459608 | 0.1797 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 479x641 | 2.386383 | 0.438154 | 0.1836 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 480x640 | 2.873787 | 0.502025 | 0.1747 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 720x1280 | 6.789413 | 1.228817 | 0.1810 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 6.036421 | 1.077992 | 0.1786 | F32 path remains the public header baseline |
| WARP_AFFINE | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 16.876650 | 1.584875 | 0.0939 | F32 path remains the public header baseline |
| WARP_AFFINE | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 18.216592 | 1.961950 | 0.1077 | F32 path remains the public header baseline |
| WARP_AFFINE | linear_inverse_replicate_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.587842 | 1.151963 | 0.4451 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate_u8c4 | fixed_coordinate_block | CV_8U | 4 | continuous | 480x640 | 4.278108 | 1.011579 | 0.2365 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 20.326675 | 9.472458 | 0.4660 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 3.768554 | 1.608138 | 0.4267 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 3.580237 | 1.679942 | 0.4692 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 9.557142 | 4.503213 | 0.4712 | Shared fixed coordinate block and U8 bilinear sampling path |

## Unsupported Cases

| Suite | Op | Variant | Shape | Status | Note |
| --- | --- | --- | --- | --- | --- |
| imgproc | CVTCOLOR | BGR2NV12_u8 | 480x640 | UNSUPPORTED | upstream OpenCV has NV12 decode but no single-call BGR-to-NV12 encoder |

## Notes

- Ratios use `OpenCV time / CVH time`: values above `1` mean CVH is faster, and values below `1` mean OpenCV is faster.
- Table timings use the minimum per-iteration time across repeats to reduce system-noise effects; this report is not a cross-machine ranking.
- Mat cases compare matching allocation/reuse semantics; imgproc cases align input dimensions, types, kernels, borders, and primary parameters.
- `headers_baseline` describes a public header fallback for an operator without a UI kernel; it is not a separate target or implementation profile.
- Raw CSV and metadata files are reproducible run artifacts; date-named Markdown files are milestone snapshots.
