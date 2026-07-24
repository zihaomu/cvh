# cvh vs OpenCV Benchmark Report (full)

生成时间（UTC）：`2026-07-24 08:30:55Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core/Imgproc 第一阶段完成后，名称级可调用覆盖为 `107/220`：Core `57/97`，Imgproc `50/123`。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- Imgproc legacy `.cpp` fast-path 已迁入 ODR-safe detail headers；resize/cvtColor UI、filter、LUT、border、Sobel、Canny 和 morphology 均从公共 header API 进入。
- 第一阶段新增的 `79` 个操作族已全部进入 Mode B，本报告包含 `92` 个 P1 性能 case。
- `full` profile 覆盖代表性的 `CV_8U` / `CV_32F`、C1/C3/C4、尺寸、布局与非连续 ROI 扩展。

## 第一阶段新增算子

本节记录第一阶段相对原有覆盖新增的 API 操作族。API 已实现不等于已经进入本次 Mode B 性能矩阵；只有建立了同输入、同参数 OpenCV 对照 case 的算子才计为“本报告实测”。

| 模块 | 第一阶段新增 | 本报告实测 | 已实现但本报告未测 |
| --- | ---: | ---: | ---: |
| Core | 43 | 43 | 0 |
| Imgproc | 36 | 36 | 0 |
| **合计** | **79** | **79** | **0** |

| 模块/类别 | 新增操作族 | 数量 | 本报告 Mode B 状态 |
| --- | --- | ---: | --- |
| Core：逐元素与逻辑 | `absdiff`、`bitwise_and`、`bitwise_not`、`bitwise_or`、`bitwise_xor`、`inRange`、`min`、`max` | 8 | 8/8 已实测 |
| Core：转换、数学与校验 | `scaleAdd`、`convertScaleAbs`、`convertFp16`、`sqrt`、`pow`、`exp`、`log`、`checkRange`、`patchNaNs` | 9 | 9/9 已实测 |
| Core：归约与统计 | `norm`、`sum`、`mean`、`meanStdDev`、`countNonZero`、`hasNonZero`、`findNonZero`、`minMaxIdx`、`minMaxLoc`、`reduce`、`reduceArgMax`、`reduceArgMin`、`normalize` | 13 | 13/13 已实测 |
| Core：布局、复制与通道 | `borderInterpolate`、`copyTo`、`extractChannel`、`insertChannel`、`mixChannels`、`flip`、`flipND`、`rotate`、`repeat`、`hconcat`、`vconcat`、`broadcast`、`swap` | 13 | 13/13 已实测 |
| Imgproc：核、滤波与强度 | `getStructuringElement`、`getGaussianKernel`、`getDerivKernels`、`getGaborKernel`、`createHanningWindow`、`integral`、`Scharr`、`Laplacian`、`spatialGradient`、`sqrBoxFilter`、`medianBlur`、`bilateralFilter`、`stackBlur`、`adaptiveThreshold`、`thresholdWithMask`、`equalizeHist`、`applyColorMap` | 17 | 17/17 已实测 |
| Imgproc：累积、金字塔与颜色 | `accumulate`、`accumulateProduct`、`accumulateSquare`、`accumulateWeighted`、`blendLinear`、`pyrDown`、`pyrUp`、`buildPyramid`、`cvtColorTwoPlane`、`demosaicing` | 10 | 10/10 已实测 |
| Imgproc：几何变换 | `remap`、`convertMaps`、`warpPerspective`、`getAffineTransform`、`getPerspectiveTransform`、`getRotationMatrix2D`、`getRotationMatrix2D_`、`invertAffineTransform`、`getRectSubPix` | 9 | 9/9 已实测 |

后续表中的 `ADD`、`GEMM`、`resize`、`cvtColor` 等仍是既有算子基线；带有 `P1 新增` 标记的行是本轮新增并已进入性能对比的算子。

## 高层优化结构

| 层次 | 当前实现 | 本报告中的含义 |
| --- | --- | --- |
| 公共 API | OpenCV-compatible header API | 所有 case 均从 `cvh::headers_fast` 公共入口调用 |
| SIMD 方言 | OpenCV Universal Intrinsics | 在 Apple ARM 上映射到 NEON |
| 专用 kernel | `cvtColor`、特定 `resize`、已验证的 core 逐元素 UI kernel | 实际命中时记录为 `dispatch_path=opencv_ui` |
| Header fast-path | 行并行 filter、LUT、border、Sobel、Canny、morphology | 记录为 `dispatch_path=header_fastpath` |
| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` 或 `public_header_scalar` |
| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |

## 运行配置

- Profile：`full`
- CVH 实现：`cvh_headers_fast`
- 采样：`warmup=1, iters=10, repeats=3`
- 线程数：`1`
- OpenMP：`dynamic=false, proc_bind=close`
- 主机：`Darwin arm64`
- CPU：`Apple M5`
- 编译器：`Apple clang version 21.0.0 (clang-2100.0.123.102)`
- 构建类型：`Release`
- CVH commit：`6a349e762fbf530085624f4f252a0dac92a54b98` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-24-opencv-upstream-performance.csv`；元数据：`2026-07-24-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`321`；有效：`320`；不支持：`1`。
- `OpenCV/CVH` 几何平均：`0.2479`；中位数：`0.4078`。
- CVH 更快：`33` 个；OpenCV 更快或相当：`287` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 153 | 0.2894 | 0.7470 | 14 | 139 |
| imgproc | 167 | 0.2150 | 0.2581 | 19 | 148 |

## 算子级概览

### `core_mat`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | opencv_ui | 1 | 0.7992 | OpenCV `1.25x` |
| ADD | 既有 | opencv_ui | 16 | 0.7830 | OpenCV `1.28x` |
| BITWISE_AND | P1 新增 | opencv_ui | 1 | 0.7675 | OpenCV `1.30x` |
| BITWISE_NOT | P1 新增 | opencv_ui | 1 | 0.4744 | OpenCV `2.11x` |
| BITWISE_OR | P1 新增 | opencv_ui | 1 | 0.7568 | OpenCV `1.32x` |
| BITWISE_XOR | P1 新增 | opencv_ui | 1 | 0.7661 | OpenCV `1.31x` |
| BORDER_INTERPOLATE | P1 新增 | public_header_baseline | 1 | 0.8477 | OpenCV `1.18x` |
| BROADCAST | P1 新增 | public_header_baseline | 1 | 0.0002 | OpenCV `4854.37x` |
| CHECK_RANGE | P1 新增 | public_header_baseline | 1 | 0.9588 | OpenCV `1.04x` |
| CONVERT_FP16 | P1 新增 | opencv_ui | 1 | 1.2121 | CVH `1.21x` |
| CONVERT_SCALE_ABS | P1 新增 | opencv_ui | 1 | 0.7937 | OpenCV `1.26x` |
| COPY_TO | P1 新增 | public_header_baseline | 1 | 0.0062 | OpenCV `161.76x` |
| COUNT_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.0292 | OpenCV `34.28x` |
| DIVIDE | 既有 | opencv_ui, scalar | 16 | 0.4986 | OpenCV `2.01x` |
| EXP | P1 新增 | public_header_baseline | 1 | 0.2262 | OpenCV `4.42x` |
| EXTRACT_CHANNEL | P1 新增 | public_header_baseline | 1 | 0.0174 | OpenCV `57.56x` |
| FIND_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.3553 | OpenCV `2.81x` |
| FLIP | P1 新增 | public_header_baseline | 1 | 0.0048 | OpenCV `208.86x` |
| FLIP_ND | P1 新增 | public_header_baseline | 1 | 0.0108 | OpenCV `92.63x` |
| GEMM | 既有 | headers_baseline | 6 | 0.0127 | OpenCV `78.59x` |
| HAS_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.0000 | OpenCV `25000.00x` |
| HCONCAT | P1 新增 | public_header_baseline | 1 | 0.5126 | OpenCV `1.95x` |
| INSERT_CHANNEL | P1 新增 | public_header_baseline | 1 | 0.0185 | OpenCV `53.94x` |
| IN_RANGE | P1 新增 | scalar | 1 | 0.0754 | OpenCV `13.26x` |
| LOG | P1 新增 | public_header_baseline | 1 | 0.2827 | OpenCV `3.54x` |
| MAT_CLONE | 既有 | headers_baseline | 4 | 0.9681 | OpenCV `1.03x` |
| MAT_CONVERTTO | 既有 | headers_baseline | 4 | 1.0380 | CVH `1.04x` |
| MAT_COPYTO | 既有 | headers_baseline | 4 | 0.9954 | OpenCV `1.00x` |
| MAT_CREATE | 既有 | headers_baseline | 4 | 0.0700 | OpenCV `14.28x` |
| MAT_RESHAPE | 既有 | headers_baseline | 4 | 0.3027 | OpenCV `3.30x` |
| MAT_SETTO | 既有 | headers_baseline | 4 | 0.0131 | OpenCV `76.25x` |
| MAX | P1 新增 | opencv_ui | 1 | 0.6113 | OpenCV `1.64x` |
| MEAN | P1 新增 | public_header_baseline | 1 | 0.2480 | OpenCV `4.03x` |
| MEAN_STD_DEV | P1 新增 | public_header_baseline | 1 | 0.1147 | OpenCV `8.72x` |
| MIN | P1 新增 | opencv_ui | 1 | 0.6494 | OpenCV `1.54x` |
| MIN_MAX_IDX | P1 新增 | public_header_baseline | 1 | 0.0647 | OpenCV `15.45x` |
| MIN_MAX_LOC | P1 新增 | public_header_baseline | 1 | 0.0629 | OpenCV `15.91x` |
| MIX_CHANNELS | P1 新增 | public_header_baseline | 1 | 0.0182 | OpenCV `55.09x` |
| MULTIPLY | 既有 | opencv_ui | 16 | 0.7826 | OpenCV `1.28x` |
| NORM | P1 新增 | public_header_baseline | 1 | 0.0766 | OpenCV `13.05x` |
| NORMALIZE | P1 新增 | public_header_baseline | 1 | 0.0521 | OpenCV `19.19x` |
| PATCH_NANS | P1 新增 | public_header_baseline | 1 | 0.3878 | OpenCV `2.58x` |
| POW | P1 新增 | public_header_baseline | 1 | 0.2211 | OpenCV `4.52x` |
| REDUCE | P1 新增 | public_header_baseline | 1 | 0.0193 | OpenCV `51.94x` |
| REDUCE_ARG_MAX | P1 新增 | public_header_baseline | 1 | 0.1680 | OpenCV `5.95x` |
| REDUCE_ARG_MIN | P1 新增 | public_header_baseline | 1 | 0.1507 | OpenCV `6.64x` |
| REPEAT | P1 新增 | public_header_baseline | 1 | 0.0013 | OpenCV `759.88x` |
| ROTATE | P1 新增 | public_header_baseline | 1 | 0.0291 | OpenCV `34.41x` |
| SCALE_ADD | P1 新增 | scalar | 1 | 0.8234 | OpenCV `1.21x` |
| SQRT | P1 新增 | public_header_baseline | 1 | 0.9685 | OpenCV `1.03x` |
| SUBTRACT | 既有 | opencv_ui | 16 | 0.7821 | OpenCV `1.28x` |
| SUM | P1 新增 | public_header_baseline | 1 | 0.2488 | OpenCV `4.02x` |
| SWAP | P1 新增 | public_header_baseline | 1 | 0.2052 | OpenCV `4.87x` |
| TRANSPOSE | 既有 | headers_baseline | 16 | 0.5343 | OpenCV `1.87x` |
| VCONCAT | P1 新增 | public_header_baseline | 1 | 0.5192 | OpenCV `1.93x` |

### `imgproc`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | public_header_baseline | 1 | 0.0680 | OpenCV `14.70x` |
| ACCUMULATE_PRODUCT | P1 新增 | public_header_baseline | 1 | 0.0557 | OpenCV `17.94x` |
| ACCUMULATE_SQUARE | P1 新增 | public_header_baseline | 1 | 0.0703 | OpenCV `14.23x` |
| ACCUMULATE_WEIGHTED | P1 新增 | public_header_baseline | 1 | 0.0700 | OpenCV `14.28x` |
| ADAPTIVE_THRESHOLD | P1 新增 | public_header_baseline | 1 | 0.5282 | OpenCV `1.89x` |
| APPLY_COLOR_MAP | P1 新增 | public_header_baseline | 1 | 0.3303 | OpenCV `3.03x` |
| BILATERAL_FILTER | P1 新增 | public_header_baseline | 1 | 0.0416 | OpenCV `24.05x` |
| BLEND_LINEAR | P1 新增 | public_header_baseline | 1 | 0.3972 | OpenCV `2.52x` |
| BOX_FILTER | 既有 | box3x3, header_fastpath | 10 | 0.2916 | OpenCV `3.43x` |
| BUILD_PYRAMID | P1 新增 | public_header_baseline | 1 | 0.0062 | OpenCV `161.45x` |
| CANNY | 既有 | header_fastpath | 4 | 0.9653 | OpenCV `1.04x` |
| CONVERT_MAPS | P1 新增 | public_header_baseline | 1 | 0.0047 | OpenCV `213.17x` |
| COPY_MAKE_BORDER | 既有 | header_fastpath | 9 | 0.3776 | OpenCV `2.65x` |
| CREATE_HANNING_WINDOW | P1 新增 | public_header_baseline | 1 | 0.0299 | OpenCV `33.42x` |
| CVTCOLOR | 既有 | header_fastpath, opencv_ui | 17 | 0.5557 | OpenCV `1.80x` |
| CVT_COLOR_TWO_PLANE | P1 新增 | public_header_baseline | 1 | 0.1555 | OpenCV `6.43x` |
| DEMOSAICING | P1 新增 | public_header_baseline | 1 | 0.0014 | OpenCV `725.16x` |
| DILATE | 既有 | header_fastpath | 6 | 0.1133 | OpenCV `8.83x` |
| EQUALIZE_HIST | P1 新增 | public_header_baseline | 1 | 0.4735 | OpenCV `2.11x` |
| ERODE | 既有 | header_fastpath | 6 | 0.1130 | OpenCV `8.85x` |
| FILTER2D | 既有 | header_fastpath | 10 | 0.3641 | OpenCV `2.75x` |
| GAUSSIAN | 既有 | gauss_separable, header_fastpath | 10 | 0.2808 | OpenCV `3.56x` |
| GET_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.0019 | CVH `2.00x` |
| GET_DERIV_KERNELS | P1 新增 | public_header_baseline | 1 | 0.4005 | OpenCV `2.50x` |
| GET_GABOR_KERNEL | P1 新增 | public_header_baseline | 1 | 0.3221 | OpenCV `3.10x` |
| GET_GAUSSIAN_KERNEL | P1 新增 | public_header_baseline | 1 | 3.9534 | CVH `3.95x` |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.4260 | CVH `2.43x` |
| GET_RECT_SUB_PIX | P1 新增 | public_header_scalar | 4 | 0.0221 | OpenCV `45.28x` |
| GET_ROTATION_MATRIX_2D | P1 新增 | public_header_baseline | 1 | 1.0144 | CVH `1.01x` |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | public_header_baseline | 1 | 1.1820 | CVH `1.18x` |
| GET_STRUCTURING_ELEMENT | P1 新增 | public_header_baseline | 1 | 0.1922 | OpenCV `5.20x` |
| INTEGRAL | P1 新增 | public_header_baseline | 1 | 0.0256 | OpenCV `39.07x` |
| INVERT_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 0.4608 | OpenCV `2.17x` |
| LAPLACIAN | P1 新增 | public_header_baseline | 1 | 0.0126 | OpenCV `79.28x` |
| LUT | 既有 | header_fastpath | 6 | 0.6150 | OpenCV `1.63x` |
| MEDIAN_BLUR | P1 新增 | public_header_baseline | 1 | 0.0048 | OpenCV `209.34x` |
| PYR_DOWN | P1 新增 | public_header_baseline | 1 | 0.0076 | OpenCV `131.79x` |
| PYR_UP | P1 新增 | public_header_baseline | 1 | 0.0075 | OpenCV `134.14x` |
| REMAP | P1 新增 | public_header_scalar | 8 | 0.0501 | OpenCV `19.97x` |
| RESIZE | 既有 | header_fastpath, headers_baseline, opencv_ui | 10 | 0.6629 | OpenCV `1.51x` |
| SCHARR | P1 新增 | public_header_baseline | 1 | 0.0122 | OpenCV `82.24x` |
| SEP_FILTER2D | 既有 | header_fastpath | 10 | 0.4601 | OpenCV `2.17x` |
| SOBEL | 既有 | header_fastpath | 6 | 1.5918 | CVH `1.59x` |
| SPATIAL_GRADIENT | P1 新增 | public_header_baseline | 1 | 0.1579 | OpenCV `6.33x` |
| SQR_BOX_FILTER | P1 新增 | public_header_baseline | 1 | 0.0229 | OpenCV `43.59x` |
| STACK_BLUR | P1 新增 | public_header_baseline | 1 | 0.0769 | OpenCV `13.01x` |
| THRESHOLD | 既有 | header_fastpath, headers_baseline | 5 | 0.0341 | OpenCV `29.34x` |
| THRESHOLD_WITH_MASK | P1 新增 | public_header_baseline | 1 | 0.9271 | OpenCV `1.08x` |
| WARP_AFFINE | 既有 | headers_baseline | 9 | 0.0868 | OpenCV `11.52x` |
| WARP_PERSPECTIVE | P1 新增 | public_header_scalar | 4 | 0.0969 | OpenCV `10.32x` |

## 详细结果

### `core_mat`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.031637 | 0.025283 | 0.7992 | phase1_representative_case |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.272829 | 0.214721 | 0.7870 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043304 | 0.031988 | 0.7387 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.039679 | 0.029788 | 0.7507 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.123683 | 0.097062 | 0.7848 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.801817 | 0.678396 | 0.8461 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.122167 | 0.097638 | 0.7992 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120746 | 0.096158 | 0.7964 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.354292 | 0.289363 | 0.8167 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062467 | 0.050667 | 0.8111 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009421 | 0.007054 | 0.7488 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009421 | 0.007037 | 0.7470 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.028437 | 0.022879 | 0.8045 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.203633 | 0.160646 | 0.7889 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.029746 | 0.022204 | 0.7465 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027521 | 0.022071 | 0.8020 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.092596 | 0.071112 | 0.7680 | correctness=upstream_pass |
| BITWISE_AND | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.034138 | 0.026200 | 0.7675 | phase1_representative_case |
| BITWISE_NOT | P1 新增 | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036221 | 0.017183 | 0.4744 | phase1_representative_case |
| BITWISE_OR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.034258 | 0.025925 | 0.7568 | phase1_representative_case |
| BITWISE_XOR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.034225 | 0.026221 | 0.7661 | phase1_representative_case |
| BORDER_INTERPOLATE | P1 新增 | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.006267 | 0.005312 | 0.8477 | phase1_representative_case;micro_iterations=10000 |
| BROADCAST | P1 新增 | row_to_image_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 11.656846 | 0.002396 | 0.0002 | phase1_representative_case |
| CHECK_RANGE | P1 新增 | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.165342 | 0.158533 | 0.9588 | phase1_representative_case |
| CONVERT_FP16 | P1 新增 | f32c1_to_fp16 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.016462 | 0.019954 | 1.2121 | phase1_representative_case |
| CONVERT_SCALE_ABS | P1 新增 | f32c3_to_u8c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.084717 | 0.067237 | 0.7937 | phase1_representative_case |
| COPY_TO | P1 新增 | masked_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.596983 | 0.022238 | 0.0062 | phase1_representative_case |
| COUNT_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.315083 | 0.009192 | 0.0292 | phase1_representative_case |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.271321 | 0.215308 | 0.7936 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.041617 | 0.030562 | 0.7344 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.039600 | 0.029504 | 0.7451 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.123358 | 0.098937 | 0.8020 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.737483 | 0.639971 | 0.8678 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.122825 | 0.095458 | 0.7772 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120242 | 0.097846 | 0.8137 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.356000 | 0.295283 | 0.8294 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 1080x1920 | 1.791454 | 0.455508 | 0.2543 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 479x641 | 0.260637 | 0.068375 | 0.2623 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 480x640 | 0.260033 | 0.066717 | 0.2566 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 720x1280 | 0.811425 | 0.215713 | 0.2658 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 3.712075 | 1.384863 | 0.3731 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.532096 | 0.202983 | 0.3815 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.526554 | 0.201229 | 0.3822 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 1.637717 | 0.609213 | 0.3720 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | P1 新增 | bounded_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.358887 | 0.081167 | 0.2262 | phase1_representative_case |
| EXTRACT_CHANNEL | P1 新增 | channel1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 2.455342 | 0.042654 | 0.0174 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.578996 | 0.205708 | 0.3553 | phase1_representative_case |
| FLIP | P1 新增 | horizontal_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.712354 | 0.017775 | 0.0048 | phase1_representative_case |
| FLIP_ND | P1 新增 | axis1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 11.885508 | 0.128317 | 0.0108 | phase1_representative_case |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.207417 | 0.003656 | 0.0176 | correctness=upstream_pass;iters=8 |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 2.066625 | 0.027041 | 0.0131 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.237917 | 0.165166 | 0.0096 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.208500 | 0.003687 | 0.0177 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 2.222667 | 0.025250 | 0.0114 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.736125 | 0.169542 | 0.0096 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.312096 | 0.000013 | 0.0000 | phase1_representative_case |
| HCONCAT | P1 新增 | two_halves_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.012721 | 0.006521 | 0.5126 | phase1_representative_case |
| INSERT_CHANNEL | P1 新增 | channel1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 2.492637 | 0.046208 | 0.0185 | phase1_representative_case |
| IN_RANGE | P1 新增 | scalar_bounds_u8c3 | scalar | CV_8U | 3 | continuous | 480x640 | 1.667121 | 0.125700 | 0.0754 | phase1_representative_case |
| LOG | P1 新增 | positive_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.479292 | 0.135508 | 0.2827 | phase1_representative_case |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024825 | 0.024754 | 0.9971 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004108 | 0.003846 | 0.9361 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.009437 | 0.009450 | 1.0013 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.015742 | 0.014796 | 0.9399 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.078308 | 0.087933 | 1.1229 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.013029 | 0.013287 | 1.0198 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.020362 | 0.020321 | 0.9980 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.037425 | 0.038017 | 1.0158 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024242 | 0.024696 | 1.0187 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003646 | 0.003458 | 0.9486 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.009042 | 0.009112 | 1.0078 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.014363 | 0.014475 | 1.0078 |  |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000016 | 0.000001 | 0.0741 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000015 | 0.000001 | 0.0782 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000045 | 0.000002 | 0.0530 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000023 | 0.000002 | 0.0782 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000047 | 0.000015 | 0.3267 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000045 | 0.000015 | 0.3387 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000065 | 0.000016 | 0.2414 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000051 | 0.000016 | 0.3143 | micro_iters_x1000 |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.951621 | 0.024754 | 0.0127 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.292154 | 0.004058 | 0.0139 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.574058 | 0.007446 | 0.0130 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.908404 | 0.011758 | 0.0129 |  |
| MAX | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.040737 | 0.024904 | 0.6113 | phase1_representative_case |
| MEAN | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 0.879537 | 0.218117 | 0.2480 | phase1_representative_case |
| MEAN_STD_DEV | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 1.199700 | 0.137546 | 0.1147 | phase1_representative_case |
| MIN | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.040171 | 0.026088 | 0.6494 | phase1_representative_case |
| MIN_MAX_IDX | P1 新增 | f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.530254 | 0.034313 | 0.0647 | phase1_representative_case |
| MIN_MAX_LOC | P1 新增 | f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.545875 | 0.034317 | 0.0629 | phase1_representative_case |
| MIX_CHANNELS | P1 新增 | reverse_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 7.924004 | 0.143842 | 0.0182 | phase1_representative_case |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.271687 | 0.219029 | 0.8062 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043100 | 0.030546 | 0.7087 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.039725 | 0.029579 | 0.7446 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.124092 | 0.098638 | 0.7949 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.758867 | 0.630537 | 0.8309 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.123642 | 0.096263 | 0.7786 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120438 | 0.098654 | 0.8191 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.353367 | 0.291700 | 0.8255 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.061892 | 0.049725 | 0.8034 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009546 | 0.007046 | 0.7381 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009425 | 0.007037 | 0.7467 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.028842 | 0.022879 | 0.7933 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.206892 | 0.161296 | 0.7796 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.028600 | 0.022342 | 0.7812 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027567 | 0.022179 | 0.8046 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.093617 | 0.072846 | 0.7781 | correctness=upstream_pass |
| NORM | P1 新增 | l2_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.486446 | 0.037275 | 0.0766 | phase1_representative_case |
| NORMALIZE | P1 新增 | l2_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 1.042571 | 0.054325 | 0.0521 | phase1_representative_case |
| PATCH_NANS | P1 新增 | one_nan_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.043500 | 0.016871 | 0.3878 | phase1_representative_case |
| POW | P1 新增 | power_1_75_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 1.455417 | 0.321758 | 0.2211 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.782342 | 0.015063 | 0.0193 | phase1_representative_case |
| REDUCE_ARG_MAX | P1 新增 | axis0_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.934238 | 0.156921 | 0.1680 | phase1_representative_case |
| REDUCE_ARG_MIN | P1 新增 | axis0_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.931867 | 0.140396 | 0.1507 | phase1_representative_case |
| REPEAT | P1 新增 | two_by_two_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 3.476163 | 0.004575 | 0.0013 | phase1_representative_case |
| ROTATE | P1 新增 | clockwise90_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.471067 | 0.100875 | 0.0291 | phase1_representative_case |
| SCALE_ADD | P1 新增 | f32c3 | scalar | CV_32F | 3 | continuous | 480x640 | 0.125283 | 0.103154 | 0.8234 | phase1_representative_case |
| SQRT | P1 新增 | positive_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.041771 | 0.040454 | 0.9685 | phase1_representative_case |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.270883 | 0.214675 | 0.7925 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043112 | 0.032296 | 0.7491 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.039617 | 0.030492 | 0.7697 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.125488 | 0.096421 | 0.7684 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.801258 | 0.691725 | 0.8633 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.121808 | 0.095525 | 0.7842 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120388 | 0.096842 | 0.8044 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.353904 | 0.288096 | 0.8141 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062267 | 0.050463 | 0.8104 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009733 | 0.007046 | 0.7239 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009437 | 0.007029 | 0.7448 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.030667 | 0.022908 | 0.7470 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.204050 | 0.163100 | 0.7993 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.028646 | 0.022492 | 0.7852 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027525 | 0.022075 | 0.8020 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.093012 | 0.071262 | 0.7662 | correctness=upstream_pass |
| SUM | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 0.877533 | 0.218288 | 0.2488 | phase1_representative_case |
| SWAP | P1 新增 | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000024 | 0.000005 | 0.2052 | phase1_representative_case;micro_iterations=10000 |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.799867 | 0.632925 | 0.7913 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.171479 | 0.033875 | 0.1975 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.181767 | 0.075375 | 0.4147 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.365800 | 0.318821 | 0.8716 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 0.687979 | 1.497229 | 2.1763 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.118462 | 0.125779 | 1.0618 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.121512 | 0.164958 | 1.3575 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.280283 | 0.581904 | 2.0761 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.569021 | 0.131862 | 0.2317 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.148929 | 0.009492 | 0.0637 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.153629 | 0.006825 | 0.0444 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.284021 | 0.029208 | 0.1028 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 0.408629 | 0.753879 | 1.8449 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.117671 | 0.088821 | 0.7548 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.115671 | 0.082158 | 0.7103 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.218138 | 0.391429 | 1.7944 | correctness=upstream_pass |
| VCONCAT | P1 新增 | two_halves_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.007071 | 0.003671 | 0.5192 | phase1_representative_case |

### `imgproc`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.287283 | 0.019542 | 0.0680 | phase1_representative_case |
| ACCUMULATE_PRODUCT | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.386796 | 0.021563 | 0.0557 | phase1_representative_case |
| ACCUMULATE_SQUARE | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.279221 | 0.019621 | 0.0703 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | P1 新增 | alpha0_1_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.310646 | 0.021754 | 0.0700 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | P1 新增 | mean11_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.423362 | 0.223629 | 0.5282 | phase1_representative_case |
| APPLY_COLOR_MAP | P1 新增 | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.705342 | 0.232979 | 0.3303 | phase1_representative_case |
| BILATERAL_FILTER | P1 新增 | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 23.246725 | 0.966421 | 0.0416 | phase1_representative_case |
| BLEND_LINEAR | P1 新增 | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.558858 | 0.221954 | 0.3972 | phase1_representative_case |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.548150 | 0.285950 | 0.1847 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.275342 | 0.045963 | 0.1669 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.273900 | 0.046238 | 0.1688 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.731117 | 0.134017 | 0.1833 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.181554 | 0.107783 | 0.5937 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.321904 | 0.312783 | 0.9717 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.383292 | 0.400079 | 1.0438 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.656604 | 0.132742 | 0.2022 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 0.681479 | 0.127342 | 0.1869 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 0.867279 | 0.177575 | 0.2047 |  |
| BUILD_PYRAMID | P1 新增 | levels3_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 2.996329 | 0.018558 | 0.0062 | phase1_representative_case |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 29.413433 | 28.073062 | 0.9544 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 4.587917 | 4.551675 | 0.9921 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 4.659042 | 4.542287 | 0.9749 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 13.166967 | 12.381921 | 0.9404 |  |
| CONVERT_MAPS | P1 新增 | f32_pair_to_fixed | public_header_baseline | CV_32F | 2 | continuous | 480x640 | 12.342471 | 0.057904 | 0.0047 | phase1_representative_case |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.074637 | 0.044838 | 0.6007 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.071858 | 0.006933 | 0.0965 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.060513 | 0.006988 | 0.1155 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.068383 | 0.021012 | 0.3073 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.069187 | 0.026508 | 0.3831 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.073446 | 0.088829 | 1.2095 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.085375 | 0.105621 | 1.2371 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.064029 | 0.021138 | 0.3301 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.068896 | 0.027621 | 0.4009 |  |
| CREATE_HANNING_WINDOW | P1 新增 | 64x64_f32 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.038150 | 0.001142 | 0.0299 | phase1_representative_case |
| CVTCOLOR | 既有 | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.110646 | 0.020004 | 0.1808 |  |
| CVTCOLOR | 既有 | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.084596 | 0.043367 | 0.5126 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.219492 | 0.219179 | 0.9986 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.033867 | 0.033721 | 0.9957 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.031300 | 0.031296 | 0.9999 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.097571 | 0.097417 | 0.9984 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.035246 | 0.035254 | 1.0002 |  |
| CVTCOLOR | 既有 | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.131713 | 0.061196 | 0.4646 |  |
| CVTCOLOR | 既有 | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.088283 | 0.065571 | 0.7427 |  |
| CVTCOLOR | 既有 | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.084829 | 0.016554 | 0.1951 |  |
| CVTCOLOR | 既有 | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.125050 | 0.079588 | 0.6364 |  |
| CVTCOLOR | 既有 | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.125308 | 0.065992 | 0.5266 |  |
| CVTCOLOR | 既有 | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.095192 | 0.039863 | 0.4188 |  |
| CVTCOLOR | 既有 | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 0.233525 | 0.076092 | 0.3258 |  |
| CVTCOLOR | 既有 | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.133992 | 0.076433 | 0.5704 |  |
| CVTCOLOR | 既有 | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.120104 | 0.061271 | 0.5101 |  |
| CVTCOLOR | 既有 | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.134500 | 0.074975 | 0.5574 |  |
| CVT_COLOR_TWO_PLANE | P1 新增 | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.427446 | 0.066479 | 0.1555 | phase1_representative_case |
| DEMOSAICING | P1 新增 | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 28.082658 | 0.038717 | 0.0014 | phase1_representative_case |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.105971 | 0.147592 | 0.1335 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.325000 | 0.024338 | 0.0749 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.268621 | 0.023621 | 0.0879 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.564775 | 0.067708 | 0.1199 |  |
| DILATE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.496100 | 0.067200 | 0.1355 |  |
| DILATE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.595246 | 0.088275 | 0.1483 |  |
| EQUALIZE_HIST | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.195171 | 0.092412 | 0.4735 | phase1_representative_case |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.106483 | 0.147821 | 0.1336 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.271896 | 0.023187 | 0.0853 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.278717 | 0.022725 | 0.0815 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.607967 | 0.066875 | 0.1100 |  |
| ERODE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.493238 | 0.067363 | 0.1366 |  |
| ERODE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.592925 | 0.088475 | 0.1492 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 2.464821 | 0.636071 | 0.2581 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.450321 | 0.094167 | 0.2091 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.414783 | 0.098046 | 0.2364 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.073838 | 0.286917 | 0.2672 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.406542 | 0.073633 | 0.1811 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.456163 | 0.209238 | 0.4587 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.423408 | 0.268062 | 0.6331 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.513146 | 0.286592 | 0.5585 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.521258 | 0.287408 | 0.5514 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.543837 | 0.403362 | 0.7417 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.633450 | 0.226562 | 0.1387 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.342375 | 0.032942 | 0.0962 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.354212 | 0.031342 | 0.0885 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.758512 | 0.098921 | 0.1304 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 0.298733 | 0.115512 | 0.3867 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 0.398338 | 0.335208 | 0.8415 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.372592 | 0.427646 | 1.1478 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 0.490854 | 0.108358 | 0.2208 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 0.491021 | 0.362392 | 0.7380 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 0.423917 | 0.137767 | 0.3250 |  |
| GET_AFFINE_TRANSFORM | P1 新增 | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000126 | 0.000253 | 2.0019 | phase1_representative_case;micro_iterations=10000 |
| GET_DERIV_KERNELS | P1 新增 | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000220 | 0.000088 | 0.4005 | phase1_representative_case;micro_iterations=10000 |
| GET_GABOR_KERNEL | P1 新增 | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.002985 | 0.000962 | 0.3221 | phase1_representative_case;micro_iterations=10000 |
| GET_GAUSSIAN_KERNEL | P1 新增 | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000242 | 0.000957 | 3.9534 | phase1_representative_case;micro_iterations=10000 |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000269 | 0.000653 | 2.4260 | phase1_representative_case;micro_iterations=10000 |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 48.995571 | 1.084512 | 0.0221 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 6.971188 | 0.155071 | 0.0222 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 7.209325 | 0.155037 | 0.0215 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 20.681225 | 0.464600 | 0.0225 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | P1 新增 | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000071 | 0.000072 | 1.0144 | phase1_representative_case;micro_iterations=10000 |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000004 | 0.000005 | 1.1820 | phase1_representative_case;micro_iterations=10000 |
| GET_STRUCTURING_ELEMENT | P1 新增 | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000417 | 0.000080 | 0.1922 | phase1_representative_case;micro_iterations=10000 |
| INTEGRAL | P1 新增 | u8c1_to_s32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.582925 | 0.040512 | 0.0256 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | P1 新增 | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000129 | 0.000060 | 0.4608 | phase1_representative_case;micro_iterations=10000 |
| LAPLACIAN | P1 新增 | ksize3_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 9.437354 | 0.119046 | 0.0126 | phase1_representative_case |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.185292 | 0.193042 | 1.0418 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.082733 | 0.028629 | 0.3460 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.084346 | 0.028592 | 0.3390 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.117908 | 0.085742 | 0.7272 |  |
| LUT | 既有 | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.121546 | 0.085800 | 0.7059 |  |
| LUT | 既有 | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.132742 | 0.114467 | 0.8623 |  |
| MEDIAN_BLUR | P1 新增 | ksize5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 52.328867 | 0.249958 | 0.0048 | phase1_representative_case |
| PYR_DOWN | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 7.717163 | 0.058554 | 0.0076 | phase1_representative_case |
| PYR_UP | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 19.932654 | 0.148596 | 0.0075 | phase1_representative_case |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 98.898683 | 5.170096 | 0.0523 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 15.035875 | 0.741079 | 0.0493 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 14.201725 | 0.720796 | 0.0508 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 44.182996 | 2.208300 | 0.0500 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 102.928392 | 5.272683 | 0.0512 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 15.239900 | 0.711729 | 0.0467 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 15.375254 | 0.752017 | 0.0489 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 45.353017 | 2.348467 | 0.0518 | no qualified SIMD fast path |
| RESIZE | 既有 | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.191917 | 0.089129 | 0.4644 |  |
| RESIZE | 既有 | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.430212 | 0.264667 | 0.6152 |  |
| RESIZE | 既有 | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.145108 | 0.062433 | 0.4303 |  |
| RESIZE | 既有 | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.135179 | 0.064537 | 0.4774 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.130258 | 0.087179 | 0.6693 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.020646 | 0.014042 | 0.6801 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018983 | 0.012508 | 0.6589 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.058221 | 0.040300 | 0.6922 |  |
| RESIZE | 既有 | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.087583 | 0.098613 | 1.1259 |  |
| RESIZE | 既有 | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.086813 | 0.103625 | 1.1937 |  |
| SCHARR | P1 新增 | dx1_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 9.462667 | 0.115062 | 0.0122 | phase1_representative_case |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.619475 | 0.646667 | 0.3993 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.407254 | 0.100554 | 0.2469 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.331242 | 0.098629 | 0.2978 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.776854 | 0.284058 | 0.3657 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.357588 | 0.084508 | 0.2363 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.438071 | 0.226796 | 0.5177 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.405308 | 0.290671 | 0.7172 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.447987 | 0.301279 | 0.6725 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.446617 | 0.302571 | 0.6775 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.407600 | 0.404063 | 0.9913 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.375404 | 0.867104 | 2.3098 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.110325 | 0.126383 | 1.1456 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.119388 | 0.124750 | 1.0449 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.209921 | 0.343946 | 1.6385 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.176113 | 0.335113 | 1.9028 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.234504 | 0.442625 | 1.8875 |  |
| SPATIAL_GRADIENT | P1 新增 | ksize3_u8_to_s16 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.230329 | 0.036362 | 0.1579 | phase1_representative_case |
| SQR_BOX_FILTER | P1 新增 | 3x3_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 8.699125 | 0.199575 | 0.0229 | phase1_representative_case |
| STACK_BLUR | P1 新增 | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.667829 | 0.128233 | 0.0769 | phase1_representative_case |
| THRESHOLD | 既有 | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.402229 | 0.069142 | 0.1719 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.506563 | 0.034388 | 0.0228 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.240525 | 0.005400 | 0.0225 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.221763 | 0.005033 | 0.0227 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.687787 | 0.015817 | 0.0230 |  |
| THRESHOLD_WITH_MASK | P1 新增 | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.179633 | 0.166529 | 0.9271 | phase1_representative_case |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 23.566842 | 2.001362 | 0.0849 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 479x641 | 3.480721 | 0.321596 | 0.0924 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 480x640 | 3.507900 | 0.308829 | 0.0880 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 9.898904 | 0.863158 | 0.0872 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 2.927958 | 0.512554 | 0.1751 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 7.540021 | 0.740217 | 0.0982 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 9.902679 | 0.799638 | 0.0808 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c3 | headers_baseline | CV_8U | 3 | continuous | 480x640 | 9.821371 | 0.821858 | 0.0837 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c4 | headers_baseline | CV_8U | 4 | continuous | 480x640 | 12.911188 | 0.518429 | 0.0402 |  |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 70.742850 | 6.731242 | 0.0952 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 10.676704 | 1.058496 | 0.0991 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 10.540379 | 1.059167 | 0.1005 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 31.440908 | 2.923429 | 0.0930 | no qualified SIMD fast path |

## 不支持用例

| Suite | Op | Variant | Shape | Status | Note |
| --- | --- | --- | --- | --- | --- |
| imgproc | CVTCOLOR | BGR2NV12_u8 | 480x640 | UNSUPPORTED | upstream OpenCV has NV12 decode but no single-call BGR-to-NV12 encoder |

## 说明

- 比值统一为 `OpenCV耗时 / CVH耗时`：大于 `1` 表示 CVH 更快，小于 `1` 表示 OpenCV 更快。
- 表内耗时取各 repeat 的最小单次耗时，用于降低系统抖动影响；本报告不是跨机器排名。
- Mat case 对比相同的分配/复用语义；imgproc case 对齐输入尺寸、类型、kernel、border 和主要参数。
- `headers_baseline` 不等于跳过优化，它表示 `cvh::headers_fast` 当前继承了 `cvh::headers` 的通用实现。
- 原始 CSV 和 metadata 是可再生成的运行产物，日期命名 Markdown 是阶段性快照。
