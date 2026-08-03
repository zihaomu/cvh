# cvh vs OpenCV Benchmark Report (full)

Generated at (UTC): `2026-08-03 07:34:27Z`

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
- CVH commit: `d96bfdeb53398872c3251fcd0d21637210ca952f` + dirty
- OpenCV: `4.14.0`, commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- Raw data: `2026-08-03-opencv-upstream-performance.csv`; metadata: `2026-08-03-opencv-upstream-performance.meta.json`

## Summary

- Total cases: `344`; valid: `343`; unsupported: `1`.
- `OpenCV/CVH` geometric mean: `0.4614`; median: `0.6214`.
- CVH faster: `49`; OpenCV faster or equal: `294`.

| Suite | Cases | geometric mean OpenCV/CVH | median | CVH faster | OpenCV faster/equal |
| --- | --- | --- | --- | --- | --- |
| core_mat | 176 | 0.6316 | 0.7535 | 33 | 143 |
| imgproc | 167 | 0.3313 | 0.3915 | 16 | 151 |

## Performance Priorities

The multipliers below are within-group geometric means for this run. They prioritize follow-up work and do not indicate API support status.

| Area | This Report | Primary Cause | Follow-up Boundary |
| --- | --- | --- | --- |
| `GEMM` | OpenCV `~7.31x` | The default upstream build can use Accelerate/LAPACK; this is not a pure SIMD comparison against built-in OpenCV UI kernels | Keep the header-only boundary explicit when evaluating future improvements |
| filter / derivative | OpenCV `~8.70x` | CVH still has generic filter dispatch, border materialization, and intermediate-row processing; upstream specializes more deeply by type and kernel size | Prioritize shared row/column work and fused U8-to-S16/F32 kernels |
| nonlinear | OpenCV `~6.77x` | Repeated window scans are gone, but bilateral weight accumulation, the median lane network, and large-image cache behavior still lag | Separate pixel-kernel cost from memory-access cost using absolute runtime |
| pyramid | OpenCV `~4.80x` | The ring workspace and UI are in place, but C3 interleaving, boundary rows, and up/downsample writeback still trail specialized upstream kernels | Reuse the current ring infrastructure and avoid full-image temporaries |
| geometry | OpenCV `~3.46x` | Coordinate blocks are shared, but interpolation, border masks, and multi-channel gather/store still contain substantial scalar work | Extend U8 C1/C3/C4 interior SIMD without duplicating public kernels |
| reduction | OpenCV `~2.22x` | Fast paths mainly cover F32 C1; the matrix also includes multi-channel, dual-input, and high-precision paths | Split gates by variant; do not trade precision for a better aggregate ratio |

## Operator-Level Overview

### `core_mat`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| ABSDIFF | opencv_ui | 1 | 0.7923 | OpenCV `1.26x` |
| ADD | opencv_ui | 16 | 0.7630 | OpenCV `1.31x` |
| BITWISE_AND | opencv_ui | 1 | 0.6749 | OpenCV `1.48x` |
| BITWISE_NOT | opencv_ui | 1 | 0.4812 | OpenCV `2.08x` |
| BITWISE_OR | opencv_ui | 1 | 0.7115 | OpenCV `1.41x` |
| BITWISE_XOR | opencv_ui | 1 | 0.7112 | OpenCV `1.41x` |
| BORDER_INTERPOLATE | public_header_baseline | 1 | 1.0102 | CVH `1.01x` |
| BROADCAST | scalar | 1 | 0.7522 | OpenCV `1.33x` |
| CHECK_RANGE | public_header_baseline | 1 | 0.6650 | OpenCV `1.50x` |
| CONVERT_FP16 | opencv_ui | 1 | 1.2692 | CVH `1.27x` |
| CONVERT_SCALE_ABS | opencv_ui | 1 | 0.5948 | OpenCV `1.68x` |
| COPY_TO | opencv_ui | 1 | 1.0275 | CVH `1.03x` |
| COUNT_NON_ZERO | opencv_ui | 1 | 0.9961 | OpenCV `1.00x` |
| DIVIDE | opencv_ui, scalar | 16 | 0.5034 | OpenCV `1.99x` |
| EXP | opencv_ui | 1 | 0.4865 | OpenCV `2.06x` |
| EXTRACT_CHANNEL | opencv_ui | 1 | 2.6576 | CVH `2.66x` |
| FIND_NON_ZERO | opencv_ui | 3 | 2.6548 | CVH `2.65x` |
| FLIP | opencv_ui | 1 | 1.0520 | CVH `1.05x` |
| FLIP_ND | opencv_ui | 1 | 7.9931 | CVH `7.99x` |
| GEMM | opencv_ui | 10 | 0.1368 | OpenCV `7.31x` |
| HAS_NON_ZERO | opencv_ui | 1 | 1.5060 | CVH `1.51x` |
| HCONCAT | scalar | 1 | 1.4509 | CVH `1.45x` |
| INSERT_CHANNEL | opencv_ui | 1 | 1.7128 | CVH `1.71x` |
| IN_RANGE | opencv_ui | 1 | 0.6214 | OpenCV `1.61x` |
| LOG | opencv_ui | 1 | 0.5793 | OpenCV `1.73x` |
| MAT_CLONE | headers_baseline | 4 | 1.1159 | CVH `1.12x` |
| MAT_CONVERTTO | headers_baseline | 4 | 0.9990 | OpenCV `1.00x` |
| MAT_COPYTO | headers_baseline | 4 | 0.9636 | OpenCV `1.04x` |
| MAT_CREATE | headers_baseline | 4 | 0.0615 | OpenCV `16.27x` |
| MAT_RESHAPE | headers_baseline | 4 | 0.2816 | OpenCV `3.55x` |
| MAT_SETTO | headers_baseline | 4 | 0.9674 | OpenCV `1.03x` |
| MAX | opencv_ui | 1 | 0.6719 | OpenCV `1.49x` |
| MEAN | opencv_ui | 1 | 2.0213 | CVH `2.02x` |
| MEAN_STD_DEV | opencv_ui | 1 | 0.2821 | OpenCV `3.55x` |
| MIN | opencv_ui | 1 | 0.6438 | OpenCV `1.55x` |
| MIN_MAX_IDX | opencv_ui | 1 | 0.6825 | OpenCV `1.47x` |
| MIN_MAX_LOC | opencv_ui | 1 | 0.7309 | OpenCV `1.37x` |
| MIX_CHANNELS | opencv_ui | 1 | 3.7102 | CVH `3.71x` |
| MULTIPLY | opencv_ui | 16 | 0.7803 | OpenCV `1.28x` |
| NORM | opencv_ui | 6 | 0.3063 | OpenCV `3.26x` |
| NORMALIZE | opencv_ui | 4 | 0.4933 | OpenCV `2.03x` |
| PATCH_NANS | opencv_ui | 1 | 0.8063 | OpenCV `1.24x` |
| POW | opencv_ui | 1 | 0.6277 | OpenCV `1.59x` |
| REDUCE | opencv_ui | 10 | 0.5930 | OpenCV `1.69x` |
| REDUCE_ARG_MAX | opencv_ui | 1 | 0.9826 | OpenCV `1.02x` |
| REDUCE_ARG_MIN | opencv_ui | 1 | 0.9620 | OpenCV `1.04x` |
| REPEAT | scalar | 1 | 0.9431 | OpenCV `1.06x` |
| ROTATE | opencv_ui | 1 | 0.4034 | OpenCV `2.48x` |
| SCALE_ADD | scalar | 1 | 0.8168 | OpenCV `1.22x` |
| SQRT | scalar | 1 | 0.9757 | OpenCV `1.02x` |
| SUBTRACT | opencv_ui | 16 | 0.7437 | OpenCV `1.34x` |
| SUM | opencv_ui | 1 | 1.9256 | CVH `1.93x` |
| SWAP | public_header_baseline | 1 | 1.0893 | CVH `1.09x` |
| TRANSPOSE | opencv_ui, scalar | 16 | 0.6666 | OpenCV `1.50x` |
| VCONCAT | scalar | 1 | 1.1064 | CVH `1.11x` |

### `imgproc`

| Op | CVH dispatch | Cases | geometric mean OpenCV/CVH | Leader |
| --- | --- | --- | --- | --- |
| ACCUMULATE | opencv_ui | 1 | 0.3200 | OpenCV `3.12x` |
| ACCUMULATE_PRODUCT | opencv_ui | 1 | 0.2642 | OpenCV `3.78x` |
| ACCUMULATE_SQUARE | opencv_ui | 1 | 0.3109 | OpenCV `3.22x` |
| ACCUMULATE_WEIGHTED | opencv_ui | 1 | 0.4085 | OpenCV `2.45x` |
| ADAPTIVE_THRESHOLD | opencv_ui | 1 | 0.6296 | OpenCV `1.59x` |
| APPLY_COLOR_MAP | public_header_baseline | 1 | 0.3268 | OpenCV `3.06x` |
| BILATERAL_FILTER | public_header_baseline | 1 | 0.0802 | OpenCV `12.47x` |
| BLEND_LINEAR | public_header_baseline | 1 | 0.4027 | OpenCV `2.48x` |
| BOX_FILTER | box3x3, header_fastpath | 10 | 0.1978 | OpenCV `5.05x` |
| BUILD_PYRAMID | opencv_ui | 1 | 0.2298 | OpenCV `4.35x` |
| CANNY | header_fastpath | 4 | 0.5230 | OpenCV `1.91x` |
| CONVERT_MAPS | opencv_ui | 1 | 0.5731 | OpenCV `1.74x` |
| COPY_MAKE_BORDER | header_fastpath | 9 | 0.9664 | OpenCV `1.03x` |
| CREATE_HANNING_WINDOW | opencv_ui | 1 | 1.9228 | CVH `1.92x` |
| CVTCOLOR | header_fastpath, opencv_ui | 17 | 0.2982 | OpenCV `3.35x` |
| CVT_COLOR_TWO_PLANE | public_header_baseline | 1 | 0.2000 | OpenCV `5.00x` |
| DEMOSAICING | public_header_baseline | 1 | 0.0891 | OpenCV `11.22x` |
| DILATE | header_fastpath | 6 | 0.2329 | OpenCV `4.29x` |
| EQUALIZE_HIST | opencv_ui | 1 | 1.0641 | CVH `1.06x` |
| ERODE | header_fastpath | 6 | 0.2307 | OpenCV `4.33x` |
| FILTER2D | header_fastpath | 10 | 0.0778 | OpenCV `12.85x` |
| GAUSSIAN | gauss_separable, header_fastpath | 10 | 0.0779 | OpenCV `12.83x` |
| GET_AFFINE_TRANSFORM | public_header_baseline | 1 | 1.9380 | CVH `1.94x` |
| GET_DERIV_KERNELS | public_header_baseline | 1 | 0.6655 | OpenCV `1.50x` |
| GET_GABOR_KERNEL | public_header_baseline | 1 | 0.9290 | OpenCV `1.08x` |
| GET_GAUSSIAN_KERNEL | public_header_baseline | 1 | 3.9808 | CVH `3.98x` |
| GET_PERSPECTIVE_TRANSFORM | public_header_baseline | 1 | 2.5063 | CVH `2.51x` |
| GET_RECT_SUB_PIX | public_header_scalar | 4 | 11.4543 | CVH `11.45x` |
| GET_ROTATION_MATRIX_2D | public_header_baseline | 1 | 0.9774 | OpenCV `1.02x` |
| GET_ROTATION_MATRIX_2D_ | public_header_baseline | 1 | 1.2213 | CVH `1.22x` |
| GET_STRUCTURING_ELEMENT | public_header_baseline | 1 | 0.7480 | OpenCV `1.34x` |
| INTEGRAL | opencv_ui | 1 | 0.5597 | OpenCV `1.79x` |
| INVERT_AFFINE_TRANSFORM | public_header_baseline | 1 | 1.4592 | CVH `1.46x` |
| LAPLACIAN | opencv_ui | 1 | 0.0645 | OpenCV `15.50x` |
| LUT | header_fastpath | 6 | 0.7745 | OpenCV `1.29x` |
| MEDIAN_BLUR | opencv_ui | 1 | 0.1729 | OpenCV `5.78x` |
| PYR_DOWN | opencv_ui | 1 | 0.1815 | OpenCV `5.51x` |
| PYR_UP | opencv_ui | 1 | 0.2162 | OpenCV `4.63x` |
| REMAP | fixed_coordinate_block | 8 | 0.3753 | OpenCV `2.66x` |
| RESIZE | header_fastpath, headers_baseline, opencv_ui | 10 | 0.5005 | OpenCV `2.00x` |
| SCHARR | opencv_ui | 1 | 0.0612 | OpenCV `16.35x` |
| SEP_FILTER2D | header_fastpath | 10 | 0.1572 | OpenCV `6.36x` |
| SOBEL | header_fastpath | 6 | 0.4538 | OpenCV `2.20x` |
| SPATIAL_GRADIENT | opencv_ui | 1 | 0.1805 | OpenCV `5.54x` |
| SQR_BOX_FILTER | opencv_ui | 1 | 0.3850 | OpenCV `2.60x` |
| STACK_BLUR | public_header_baseline | 1 | 0.2325 | OpenCV `4.30x` |
| THRESHOLD | header_fastpath, headers_baseline | 5 | 0.9889 | OpenCV `1.01x` |
| THRESHOLD_WITH_MASK | public_header_baseline | 1 | 0.9370 | OpenCV `1.07x` |
| WARP_AFFINE | fixed_coordinate_block, headers_baseline | 9 | 0.1717 | OpenCV `5.83x` |
| WARP_PERSPECTIVE | fixed_coordinate_block | 4 | 0.4637 | OpenCV `2.16x` |

## Detailed Results

### `core_mat`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027967 | 0.022158 | 0.7923 | phase1_representative_case |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.251262 | 0.208962 | 0.8317 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.040079 | 0.029696 | 0.7409 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.041629 | 0.030671 | 0.7368 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.114987 | 0.089358 | 0.7771 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.749900 | 0.663408 | 0.8847 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.114321 | 0.093654 | 0.8192 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120079 | 0.098117 | 0.8171 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.337517 | 0.294321 | 0.8720 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.060975 | 0.049188 | 0.8067 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009408 | 0.004763 | 0.5062 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009996 | 0.007212 | 0.7215 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.032550 | 0.021096 | 0.6481 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.190292 | 0.157483 | 0.8276 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.029375 | 0.022037 | 0.7502 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027425 | 0.021375 | 0.7794 | correctness=upstream_pass |
| ADD | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.087413 | 0.068442 | 0.7830 | correctness=upstream_pass |
| BITWISE_AND | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.032642 | 0.022029 | 0.6749 | phase1_representative_case |
| BITWISE_NOT | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.030646 | 0.014746 | 0.4812 | phase1_representative_case |
| BITWISE_OR | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.031154 | 0.022167 | 0.7115 | phase1_representative_case |
| BITWISE_XOR | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.031175 | 0.022171 | 0.7112 | phase1_representative_case |
| BORDER_INTERPOLATE | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.005392 | 0.005447 | 1.0102 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| BROADCAST | row_to_image_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004254 | 0.003200 | 0.7522 | phase1_representative_case |
| CHECK_RANGE | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.147012 | 0.097758 | 0.6650 | phase1_representative_case |
| CONVERT_FP16 | f32c1_to_fp16 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.014858 | 0.018858 | 1.2692 | phase1_representative_case |
| CONVERT_SCALE_ABS | f32c3_to_u8c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.101142 | 0.060154 | 0.5948 | phase1_representative_case |
| COPY_TO | masked_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.022417 | 0.023033 | 1.0275 | phase1_representative_case |
| COUNT_NON_ZERO | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.008554 | 0.008521 | 0.9961 | phase1_representative_case |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.254167 | 0.206979 | 0.8143 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.041275 | 0.029812 | 0.7223 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.040512 | 0.030346 | 0.7490 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.114167 | 0.092262 | 0.8081 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.781083 | 0.784679 | 1.0046 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.115450 | 0.092133 | 0.7980 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.114296 | 0.092721 | 0.8112 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.328163 | 0.296779 | 0.9044 | correctness=upstream_pass |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 1080x1920 | 1.809742 | 0.453437 | 0.2506 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 479x641 | 0.265608 | 0.066679 | 0.2510 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 480x640 | 0.270429 | 0.068783 | 0.2543 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 720x1280 | 0.806179 | 0.205592 | 0.2550 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 3.613454 | 1.372442 | 0.3798 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.531613 | 0.199954 | 0.3761 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.539175 | 0.202225 | 0.3751 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 1.638404 | 0.608175 | 0.3712 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | bounded_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.152854 | 0.074358 | 0.4865 | phase1_representative_case |
| EXTRACT_CHANNEL | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.015904 | 0.042267 | 2.6576 | phase1_representative_case |
| FIND_NON_ZERO | all_zero_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.008875 | 0.074525 | 8.3972 | phase1_representative_case |
| FIND_NON_ZERO | random_dense_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.499450 | 0.191037 | 0.3825 | phase1_representative_case |
| FIND_NON_ZERO | sparse_tail_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009137 | 0.053229 | 5.8254 | phase1_representative_case |
| FLIP | horizontal_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.016337 | 0.017188 | 1.0520 | phase1_representative_case |
| FLIP_ND | axis1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.016379 | 0.130921 | 7.9931 | phase1_representative_case |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.055063 | 0.003450 | 0.0627 | correctness=upstream_pass;shape=square_128;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.481092 | 0.020808 | 0.0433 | correctness=upstream_pass;shape=square_256;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x32x256 | 0.043283 | 0.005267 | 0.1217 | correctness=upstream_pass;shape=wide_m256_k32_n256;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 32x512x64 | 0.031433 | 0.111483 | 3.5467 | correctness=upstream_pass;shape=skinny_m32_k512_n64;component=public_end_to_end;iters=10 |
| GEMM | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 4.099292 | 0.160958 | 0.0393 | correctness=upstream_pass;shape=square_512;component=public_end_to_end;iters=1 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.056046 | 0.003492 | 0.0623 | correctness=upstream_pass;shape=square_128;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.468883 | 0.020754 | 0.0443 | correctness=upstream_pass;shape=square_256;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x32x256 | 0.043871 | 0.005237 | 0.1194 | correctness=upstream_pass;shape=wide_m256_k32_n256;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 32x512x64 | 0.030075 | 0.109133 | 3.6287 | correctness=upstream_pass;shape=skinny_m32_k512_n64;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=10 |
| GEMM | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 3.936417 | 0.165166 | 0.0420 | correctness=upstream_pass;shape=square_512;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.000008 | 0.000013 | 1.5060 | phase1_representative_case |
| HCONCAT | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004325 | 0.006275 | 1.4509 | phase1_representative_case |
| INSERT_CHANNEL | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.025175 | 0.043121 | 1.7128 | phase1_representative_case |
| IN_RANGE | scalar_bounds_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.177254 | 0.110137 | 0.6214 | phase1_representative_case |
| LOG | positive_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.209817 | 0.121554 | 0.5793 | phase1_representative_case |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.026217 | 0.025867 | 0.9867 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003750 | 0.003683 | 0.9822 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.009062 | 0.008558 | 0.9444 |  |
| MAT_CLONE | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.013879 | 0.023512 | 1.6941 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.080763 | 0.083596 | 1.0351 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.012554 | 0.013137 | 1.0465 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.033746 | 0.030825 | 0.9134 |  |
| MAT_CONVERTTO | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.034908 | 0.035146 | 1.0068 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.026604 | 0.025112 | 0.9439 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003488 | 0.003379 | 0.9689 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.008042 | 0.008112 | 1.0088 |  |
| MAT_COPYTO | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.026258 | 0.024538 | 0.9345 |  |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000015 | 0.000001 | 0.0772 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000015 | 0.000001 | 0.0701 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000047 | 0.000002 | 0.0469 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000019 | 0.000001 | 0.0562 | cvh_ui_uses_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000046 | 0.000016 | 0.3423 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000045 | 0.000015 | 0.3358 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000112 | 0.000021 | 0.1854 | micro_iters_x1000 |
| MAT_RESHAPE | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000053 | 0.000016 | 0.2950 | micro_iters_x1000 |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.025517 | 0.025038 | 0.9812 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004204 | 0.004625 | 1.1001 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.007133 | 0.010383 | 1.4556 |  |
| MAT_SETTO | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.019904 | 0.011096 | 0.5575 |  |
| MAX | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.034537 | 0.023204 | 0.6719 | phase1_representative_case |
| MEAN | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.094121 | 0.190242 | 2.0213 | phase1_representative_case |
| MEAN_STD_DEV | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.454067 | 0.128075 | 0.2821 | phase1_representative_case |
| MIN | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.034554 | 0.022246 | 0.6438 | phase1_representative_case |
| MIN_MAX_IDX | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.046579 | 0.031792 | 0.6825 | phase1_representative_case |
| MIN_MAX_LOC | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.044758 | 0.032712 | 0.7309 | phase1_representative_case |
| MIX_CHANNELS | reverse_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.035838 | 0.132963 | 3.7102 | phase1_representative_case |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.253871 | 0.206842 | 0.8148 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.040983 | 0.029625 | 0.7229 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.040113 | 0.030279 | 0.7549 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.114437 | 0.090717 | 0.7927 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.792925 | 0.653904 | 0.8247 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.117508 | 0.090550 | 0.7706 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.114708 | 0.092404 | 0.8056 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.331117 | 0.280646 | 0.8476 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062613 | 0.050633 | 0.8087 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009546 | 0.007042 | 0.7377 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009758 | 0.007021 | 0.7195 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.030675 | 0.022517 | 0.7340 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.192567 | 0.153921 | 0.7993 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.027987 | 0.022083 | 0.7890 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027146 | 0.022067 | 0.8129 | correctness=upstream_pass |
| MULTIPLY | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.087383 | 0.066775 | 0.7642 | correctness=upstream_pass |
| NORM | inf_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.088950 | 0.019179 | 0.2156 | phase1_representative_case |
| NORM | inf_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.037433 | 0.010004 | 0.2673 | phase1_representative_case |
| NORM | l1_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.088954 | 0.026258 | 0.2952 | phase1_representative_case |
| NORM | l1_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.073071 | 0.022196 | 0.3038 | phase1_representative_case |
| NORM | l2_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.080021 | 0.033779 | 0.4221 | phase1_representative_case |
| NORM | l2_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.088663 | 0.033567 | 0.3786 | phase1_representative_case |
| NORMALIZE | inf_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.071054 | 0.030425 | 0.4282 | phase1_representative_case |
| NORMALIZE | l1_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.106646 | 0.041879 | 0.3927 | phase1_representative_case |
| NORMALIZE | l2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.106971 | 0.053308 | 0.4983 | phase1_representative_case |
| NORMALIZE | minmax_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.080142 | 0.056625 | 0.7066 | phase1_representative_case |
| PATCH_NANS | one_nan_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.018717 | 0.015092 | 0.8063 | phase1_representative_case |
| POW | power_1_75_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.488725 | 0.306792 | 0.6277 | phase1_representative_case |
| REDUCE | axis0_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.064621 | 0.012587 | 0.1948 | phase1_representative_case |
| REDUCE | axis0_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.037250 | 0.017083 | 0.4586 | phase1_representative_case |
| REDUCE | axis0_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.037304 | 0.017542 | 0.4702 | phase1_representative_case |
| REDUCE | axis0_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.064658 | 0.017112 | 0.2647 | phase1_representative_case |
| REDUCE | axis0_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.066250 | 0.014271 | 0.2154 | phase1_representative_case |
| REDUCE | axis1_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.050575 | 0.010558 | 0.2088 | phase1_representative_case |
| REDUCE | axis1_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.054275 | 0.274729 | 5.0618 | phase1_representative_case |
| REDUCE | axis1_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.055175 | 0.150279 | 2.7237 | phase1_representative_case |
| REDUCE | axis1_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.067271 | 0.245575 | 3.6505 | phase1_representative_case |
| REDUCE | axis1_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.049612 | 0.010608 | 0.2138 | phase1_representative_case |
| REDUCE_ARG_MAX | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.144250 | 0.141738 | 0.9826 | phase1_representative_case |
| REDUCE_ARG_MIN | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.140525 | 0.135183 | 0.9620 | phase1_representative_case |
| REPEAT | two_by_two_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004613 | 0.004350 | 0.9431 | phase1_representative_case |
| ROTATE | clockwise90_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.253433 | 0.102233 | 0.4034 | phase1_representative_case |
| SCALE_ADD | f32c3 | scalar | CV_32F | 3 | continuous | 480x640 | 0.108912 | 0.088958 | 0.8168 | phase1_representative_case |
| SQRT | positive_f32c1 | scalar | CV_32F | 1 | continuous | 480x640 | 0.037167 | 0.036262 | 0.9757 | phase1_representative_case |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.255850 | 0.208104 | 0.8134 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.041479 | 0.029746 | 0.7171 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.041163 | 0.023312 | 0.5664 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.116946 | 0.091508 | 0.7825 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.788979 | 0.631225 | 0.8001 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.114317 | 0.088667 | 0.7756 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.114696 | 0.086087 | 0.7506 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.339246 | 0.279508 | 0.8239 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.066587 | 0.049513 | 0.7436 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.010971 | 0.007192 | 0.6555 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009892 | 0.007071 | 0.7148 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.030608 | 0.023325 | 0.7620 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.191771 | 0.148892 | 0.7764 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.027758 | 0.019600 | 0.7061 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.028817 | 0.022758 | 0.7898 | correctness=upstream_pass |
| SUBTRACT | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.086304 | 0.066254 | 0.7677 | correctness=upstream_pass |
| SUM | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.097037 | 0.186854 | 1.9256 | phase1_representative_case |
| SWAP | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000005 | 0.000005 | 1.0893 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.588008 | 0.558504 | 0.9498 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.031929 | 0.032442 | 1.0161 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.072142 | 0.082187 | 1.1393 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.285771 | 0.290508 | 1.0166 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 1080x1920 | 2.756167 | 1.616229 | 0.5864 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 479x641 | 0.273100 | 0.117608 | 0.4306 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 480x640 | 0.283021 | 0.154883 | 0.5473 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_32F | 3 | continuous | 720x1280 | 0.967442 | 0.660283 | 0.6825 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.094542 | 0.117154 | 1.2392 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.010225 | 0.009317 | 0.9112 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.007229 | 0.006667 | 0.9222 | correctness=upstream_pass |
| TRANSPOSE | continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.053004 | 0.028533 | 0.5383 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 1.757796 | 0.727917 | 0.4141 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.240529 | 0.086546 | 0.3598 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.241667 | 0.084200 | 0.3484 | correctness=upstream_pass |
| TRANSPOSE | continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 0.738121 | 0.365446 | 0.4951 | correctness=upstream_pass |
| VCONCAT | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.003408 | 0.003771 | 1.1064 | phase1_representative_case |

### `imgproc`

| Op | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.058746 | 0.018800 | 0.3200 | phase1_representative_case |
| ACCUMULATE_PRODUCT | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.075650 | 0.019987 | 0.2642 | phase1_representative_case |
| ACCUMULATE_SQUARE | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.058425 | 0.018167 | 0.3109 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | alpha0_1_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.049417 | 0.020188 | 0.4085 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | mean11_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.325396 | 0.204863 | 0.6296 | phase1_representative_case |
| APPLY_COLOR_MAP | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.655517 | 0.214237 | 0.3268 | phase1_representative_case |
| BILATERAL_FILTER | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 10.907687 | 0.874683 | 0.0802 | phase1_representative_case |
| BLEND_LINEAR | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.509733 | 0.205262 | 0.4027 | phase1_representative_case |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 2.126508 | 0.255725 | 0.1203 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.308175 | 0.042588 | 0.1382 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.305608 | 0.041621 | 0.1362 |  |
| BOX_FILTER | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.924983 | 0.115425 | 0.1248 |  |
| BOX_FILTER | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.379192 | 0.094471 | 0.2491 |  |
| BOX_FILTER | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.534917 | 0.274467 | 0.5131 |  |
| BOX_FILTER | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.583108 | 0.363867 | 0.6240 |  |
| BOX_FILTER | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.736896 | 0.114308 | 0.1551 |  |
| BOX_FILTER | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 0.748754 | 0.114229 | 0.1526 |  |
| BOX_FILTER | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 0.907658 | 0.156454 | 0.1724 |  |
| BUILD_PYRAMID | levels3_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.079954 | 0.018371 | 0.2298 | phase1_representative_case |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 53.821563 | 28.409996 | 0.5279 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 7.845158 | 4.099208 | 0.5225 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 7.980325 | 4.151767 | 0.5202 |  |
| CANNY | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 24.126550 | 12.579746 | 0.5214 |  |
| CONVERT_MAPS | f32_pair_to_fixed | opencv_ui | CV_32F | 2 | continuous | 480x640 | 0.101267 | 0.058033 | 0.5731 | phase1_representative_case |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.041029 | 0.040858 | 0.9958 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.006683 | 0.006204 | 0.9283 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.006838 | 0.006237 | 0.9122 |  |
| COPY_MAKE_BORDER | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.018658 | 0.018092 | 0.9696 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.025983 | 0.024221 | 0.9322 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.067979 | 0.077108 | 1.1343 |  |
| COPY_MAKE_BORDER | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.095379 | 0.094533 | 0.9911 |  |
| COPY_MAKE_BORDER | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.019592 | 0.018258 | 0.9319 |  |
| COPY_MAKE_BORDER | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.025738 | 0.023704 | 0.9210 |  |
| CREATE_HANNING_WINDOW | 64x64_f32 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.000596 | 0.001146 | 1.9228 | phase1_representative_case |
| CVTCOLOR | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.243704 | 0.017904 | 0.0735 |  |
| CVTCOLOR | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.080192 | 0.038558 | 0.4808 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.205204 | 0.203529 | 0.9918 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.030246 | 0.030171 | 0.9975 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.030912 | 0.030150 | 0.9753 |  |
| CVTCOLOR | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.091208 | 0.090438 | 0.9915 |  |
| CVTCOLOR | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.031088 | 0.030179 | 0.9708 |  |
| CVTCOLOR | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.345221 | 0.052554 | 0.1522 |  |
| CVTCOLOR | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.136712 | 0.058813 | 0.4302 |  |
| CVTCOLOR | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.117850 | 0.015058 | 0.1278 |  |
| CVTCOLOR | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.273550 | 0.069546 | 0.2542 |  |
| CVTCOLOR | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.324417 | 0.056658 | 0.1746 |  |
| CVTCOLOR | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.161300 | 0.035646 | 0.2210 |  |
| CVTCOLOR | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 0.841850 | 0.065371 | 0.0777 |  |
| CVTCOLOR | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.390575 | 0.065662 | 0.1681 |  |
| CVTCOLOR | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.274763 | 0.052629 | 0.1915 |  |
| CVTCOLOR | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.380254 | 0.065854 | 0.1732 |  |
| CVT_COLOR_TWO_PLANE | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.328225 | 0.065638 | 0.2000 | phase1_representative_case |
| DEMOSAICING | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.434267 | 0.038687 | 0.0891 | phase1_representative_case |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.193138 | 0.132571 | 0.6864 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.036350 | 0.021654 | 0.5957 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.034883 | 0.020779 | 0.5957 |  |
| DILATE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.087371 | 0.064075 | 0.7334 |  |
| DILATE | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 2.053971 | 0.062279 | 0.0303 |  |
| DILATE | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.689363 | 0.079287 | 0.0295 |  |
| EQUALIZE_HIST | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.081417 | 0.086638 | 1.0641 | phase1_representative_case |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.189387 | 0.131546 | 0.6946 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.037112 | 0.021992 | 0.5926 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.035683 | 0.021529 | 0.6033 |  |
| ERODE | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.088079 | 0.062371 | 0.7081 |  |
| ERODE | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 2.128537 | 0.061967 | 0.0291 |  |
| ERODE | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.712242 | 0.079946 | 0.0295 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 9.625529 | 0.589100 | 0.0612 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 1.452421 | 0.085933 | 0.0592 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 1.445329 | 0.087663 | 0.0607 |  |
| FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 4.263958 | 0.246850 | 0.0579 |  |
| FILTER2D | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 1.697146 | 0.065125 | 0.0384 |  |
| FILTER2D | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 2.041950 | 0.178079 | 0.0872 |  |
| FILTER2D | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 1.974196 | 0.224496 | 0.1137 |  |
| FILTER2D | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 2.356662 | 0.264483 | 0.1122 |  |
| FILTER2D | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 2.324317 | 0.248925 | 0.1071 |  |
| FILTER2D | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.340354 | 0.328029 | 0.1402 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 6.445521 | 0.198142 | 0.0307 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.953763 | 0.028600 | 0.0300 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.959817 | 0.028187 | 0.0294 |  |
| GAUSSIAN | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 2.856242 | 0.084588 | 0.0296 |  |
| GAUSSIAN | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 1.527587 | 0.103696 | 0.0679 |  |
| GAUSSIAN | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 1.273958 | 0.290725 | 0.2282 |  |
| GAUSSIAN | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.905763 | 0.426750 | 0.4712 |  |
| GAUSSIAN | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 1.456971 | 0.089575 | 0.0615 |  |
| GAUSSIAN | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 1.441013 | 0.301292 | 0.2091 |  |
| GAUSSIAN | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 1.101154 | 0.120775 | 0.1097 |  |
| GET_AFFINE_TRANSFORM | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000127 | 0.000247 | 1.9380 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_DERIV_KERNELS | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000127 | 0.000085 | 0.6655 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GABOR_KERNEL | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.001010 | 0.000938 | 0.9290 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GAUSSIAN_KERNEL | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000239 | 0.000950 | 3.9808 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_PERSPECTIVE_TRANSFORM | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000265 | 0.000664 | 2.5063 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 0.080533 | 0.991979 | 12.3176 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 0.017004 | 0.146517 | 8.6165 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 0.011558 | 0.144962 | 12.5419 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 0.033388 | 0.431750 | 12.9315 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000073 | 0.000072 | 0.9774 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_ROTATION_MATRIX_2D_ | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000004 | 0.000005 | 1.2213 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_STRUCTURING_ELEMENT | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000102 | 0.000077 | 0.7480 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| INTEGRAL | u8c1_to_s32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.072112 | 0.040363 | 0.5597 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000041 | 0.000060 | 1.4592 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| LAPLACIAN | ksize3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 1.715675 | 0.110721 | 0.0645 | phase1_representative_case |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.221708 | 0.173646 | 0.7832 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.033563 | 0.025567 | 0.7618 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.034179 | 0.026279 | 0.7689 |  |
| LUT | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.098242 | 0.076542 | 0.7791 |  |
| LUT | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.105933 | 0.082804 | 0.7817 |  |
| LUT | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.132738 | 0.102571 | 0.7727 |  |
| MEDIAN_BLUR | ksize5_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 1.432733 | 0.247750 | 0.1729 | phase1_representative_case |
| PYR_DOWN | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.321946 | 0.058433 | 0.1815 | phase1_representative_case |
| PYR_UP | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.668954 | 0.144613 | 0.2162 | phase1_representative_case |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 12.058329 | 5.049596 | 0.4188 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 1.808158 | 0.719975 | 0.3982 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 1.807471 | 0.714754 | 0.3954 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 5.390133 | 2.110362 | 0.3915 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 14.352975 | 5.250863 | 0.3658 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 2.096608 | 0.731588 | 0.3489 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.091388 | 0.700846 | 0.3351 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 6.258767 | 2.228242 | 0.3560 | Shared fixed coordinate block and U8 bilinear sampling path |
| RESIZE | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.166013 | 0.082192 | 0.4951 |  |
| RESIZE | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.381646 | 0.237879 | 0.6233 |  |
| RESIZE | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.334408 | 0.055821 | 0.1669 |  |
| RESIZE | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.338258 | 0.056900 | 0.1682 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.126200 | 0.080642 | 0.6390 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.018446 | 0.012654 | 0.6860 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018204 | 0.012079 | 0.6635 |  |
| RESIZE | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.057117 | 0.037167 | 0.6507 |  |
| RESIZE | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.116917 | 0.088229 | 0.7546 |  |
| RESIZE | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.116192 | 0.092537 | 0.7964 |  |
| SCHARR | dx1_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 1.747558 | 0.106900 | 0.0612 | phase1_representative_case |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 4.017346 | 0.586821 | 0.1461 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.591538 | 0.084954 | 0.1436 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.597662 | 0.084096 | 0.1407 |  |
| SEP_FILTER2D | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.752779 | 0.245250 | 0.1399 |  |
| SEP_FILTER2D | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.941517 | 0.073467 | 0.0780 |  |
| SEP_FILTER2D | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 1.332671 | 0.192921 | 0.1448 |  |
| SEP_FILTER2D | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 1.168150 | 0.255633 | 0.2188 |  |
| SEP_FILTER2D | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 1.501433 | 0.269388 | 0.1794 |  |
| SEP_FILTER2D | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 1.329392 | 0.248558 | 0.1870 |  |
| SEP_FILTER2D | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 1.223796 | 0.329871 | 0.2695 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.685067 | 0.801246 | 0.4755 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.233896 | 0.107208 | 0.4584 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.243358 | 0.106992 | 0.4396 |  |
| SOBEL | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.818763 | 0.309100 | 0.3775 |  |
| SOBEL | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.628454 | 0.321858 | 0.5121 |  |
| SOBEL | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.861471 | 0.406167 | 0.4715 |  |
| SPATIAL_GRADIENT | ksize3_u8_to_s16 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.181308 | 0.032733 | 0.1805 | phase1_representative_case |
| SQR_BOX_FILTER | 3x3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.438371 | 0.168792 | 0.3850 | phase1_representative_case |
| STACK_BLUR | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.518242 | 0.120467 | 0.2325 | phase1_representative_case |
| THRESHOLD | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.064942 | 0.054679 | 0.8420 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.033033 | 0.034433 | 1.0424 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004775 | 0.004871 | 1.0201 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.004775 | 0.004892 | 1.0244 |  |
| THRESHOLD | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.014700 | 0.015154 | 1.0309 |  |
| THRESHOLD_WITH_MASK | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.166383 | 0.155908 | 0.9370 | phase1_representative_case |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 1080x1920 | 11.313925 | 1.992979 | 0.1762 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 479x641 | 1.673979 | 0.299433 | 0.1789 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 480x640 | 1.672637 | 0.287675 | 0.1720 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 720x1280 | 5.007579 | 0.882413 | 0.1762 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 2.631079 | 0.461538 | 0.1754 | F32 path remains the public header baseline |
| WARP_AFFINE | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 6.794892 | 0.702767 | 0.1034 | F32 path remains the public header baseline |
| WARP_AFFINE | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 8.917054 | 0.782808 | 0.0878 | F32 path remains the public header baseline |
| WARP_AFFINE | linear_inverse_replicate_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.110808 | 0.789650 | 0.3741 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | linear_inverse_replicate_u8c4 | fixed_coordinate_block | CV_8U | 4 | continuous | 480x640 | 2.150471 | 0.489304 | 0.2275 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 14.254129 | 6.642375 | 0.4660 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 2.082208 | 0.968983 | 0.4654 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.101787 | 0.968325 | 0.4607 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 6.270158 | 2.900913 | 0.4627 | Shared fixed coordinate block and U8 bilinear sampling path |

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
