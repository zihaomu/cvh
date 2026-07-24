# cvh vs OpenCV Benchmark Report (full)

生成时间（UTC）：`2026-07-24 09:24:36Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core/Imgproc 第一阶段完成后，名称级可调用覆盖为 `107/220`：Core `57/97`，Imgproc `50/123`。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- Core F32 `patchNaNs/exp/log/pow` 已接入 UI；`pow` 分离整数指数与通用指数，特殊值 block 保留 scalar fallback。
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
| 专用 kernel | `cvtColor`、特定 `resize`、core 逐元素与 F32 数学 UI kernel | 实际命中时记录为 `dispatch_path=opencv_ui` |
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
- CVH commit：`81044e01064a2a5867b8da80f7181552ffa09860` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-24-opencv-upstream-performance.csv`；元数据：`2026-07-24-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`321`；有效：`320`；不支持：`1`。
- `OpenCV/CVH` 几何平均：`0.2509`；中位数：`0.4653`。
- CVH 更快：`30` 个；OpenCV 更快或相当：`290` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 153 | 0.2943 | 0.7401 | 14 | 139 |
| imgproc | 167 | 0.2168 | 0.2738 | 16 | 151 |

## 算子级概览

### `core_mat`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | opencv_ui | 1 | 0.7673 | OpenCV `1.30x` |
| ADD | 既有 | opencv_ui | 16 | 0.7725 | OpenCV `1.29x` |
| BITWISE_AND | P1 新增 | opencv_ui | 1 | 0.5290 | OpenCV `1.89x` |
| BITWISE_NOT | P1 新增 | opencv_ui | 1 | 0.4972 | OpenCV `2.01x` |
| BITWISE_OR | P1 新增 | opencv_ui | 1 | 0.7093 | OpenCV `1.41x` |
| BITWISE_XOR | P1 新增 | opencv_ui | 1 | 0.7091 | OpenCV `1.41x` |
| BORDER_INTERPOLATE | P1 新增 | public_header_baseline | 1 | 0.8862 | OpenCV `1.13x` |
| BROADCAST | P1 新增 | public_header_baseline | 1 | 0.0002 | OpenCV `4329.00x` |
| CHECK_RANGE | P1 新增 | public_header_baseline | 1 | 0.5965 | OpenCV `1.68x` |
| CONVERT_FP16 | P1 新增 | opencv_ui | 1 | 1.2128 | CVH `1.21x` |
| CONVERT_SCALE_ABS | P1 新增 | opencv_ui | 1 | 0.8271 | OpenCV `1.21x` |
| COPY_TO | P1 新增 | public_header_baseline | 1 | 0.0066 | OpenCV `152.02x` |
| COUNT_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.0333 | OpenCV `29.99x` |
| DIVIDE | 既有 | opencv_ui, scalar | 16 | 0.5007 | OpenCV `2.00x` |
| EXP | P1 新增 | opencv_ui | 1 | 0.4785 | OpenCV `2.09x` |
| EXTRACT_CHANNEL | P1 新增 | public_header_baseline | 1 | 0.0138 | OpenCV `72.71x` |
| FIND_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.3399 | OpenCV `2.94x` |
| FLIP | P1 新增 | public_header_baseline | 1 | 0.0048 | OpenCV `206.65x` |
| FLIP_ND | P1 新增 | public_header_baseline | 1 | 0.0103 | OpenCV `97.45x` |
| GEMM | 既有 | headers_baseline | 6 | 0.0127 | OpenCV `78.84x` |
| HAS_NON_ZERO | P1 新增 | public_header_baseline | 1 | 0.0000 | OpenCV `22222.22x` |
| HCONCAT | P1 新增 | public_header_baseline | 1 | 0.5305 | OpenCV `1.89x` |
| INSERT_CHANNEL | P1 新增 | public_header_baseline | 1 | 0.0137 | OpenCV `72.83x` |
| IN_RANGE | P1 新增 | opencv_ui | 1 | 0.6192 | OpenCV `1.61x` |
| LOG | P1 新增 | opencv_ui | 1 | 0.5823 | OpenCV `1.72x` |
| MAT_CLONE | 既有 | headers_baseline | 4 | 0.9826 | OpenCV `1.02x` |
| MAT_CONVERTTO | 既有 | headers_baseline | 4 | 0.9878 | OpenCV `1.01x` |
| MAT_COPYTO | 既有 | headers_baseline | 4 | 1.0038 | CVH `1.00x` |
| MAT_CREATE | 既有 | headers_baseline | 4 | 0.0683 | OpenCV `14.63x` |
| MAT_RESHAPE | 既有 | headers_baseline | 4 | 0.3380 | OpenCV `2.96x` |
| MAT_SETTO | 既有 | headers_baseline | 4 | 0.0119 | OpenCV `83.90x` |
| MAX | P1 新增 | opencv_ui | 1 | 0.6540 | OpenCV `1.53x` |
| MEAN | P1 新增 | public_header_baseline | 1 | 0.1756 | OpenCV `5.70x` |
| MEAN_STD_DEV | P1 新增 | public_header_baseline | 1 | 0.1049 | OpenCV `9.53x` |
| MIN | P1 新增 | opencv_ui | 1 | 0.6467 | OpenCV `1.55x` |
| MIN_MAX_IDX | P1 新增 | public_header_baseline | 1 | 0.0558 | OpenCV `17.92x` |
| MIN_MAX_LOC | P1 新增 | public_header_baseline | 1 | 0.0556 | OpenCV `17.98x` |
| MIX_CHANNELS | P1 新增 | public_header_baseline | 1 | 0.0136 | OpenCV `73.64x` |
| MULTIPLY | 既有 | opencv_ui | 16 | 0.7689 | OpenCV `1.30x` |
| NORM | P1 新增 | public_header_baseline | 1 | 0.0673 | OpenCV `14.85x` |
| NORMALIZE | P1 新增 | public_header_baseline | 1 | 0.0467 | OpenCV `21.40x` |
| PATCH_NANS | P1 新增 | opencv_ui | 1 | 0.8092 | OpenCV `1.24x` |
| POW | P1 新增 | opencv_ui | 1 | 0.6147 | OpenCV `1.63x` |
| REDUCE | P1 新增 | public_header_baseline | 1 | 0.0199 | OpenCV `50.29x` |
| REDUCE_ARG_MAX | P1 新增 | public_header_baseline | 1 | 0.1562 | OpenCV `6.40x` |
| REDUCE_ARG_MIN | P1 新增 | public_header_baseline | 1 | 0.1621 | OpenCV `6.17x` |
| REPEAT | P1 新增 | public_header_baseline | 1 | 0.0015 | OpenCV `667.56x` |
| ROTATE | P1 新增 | public_header_baseline | 1 | 0.0290 | OpenCV `34.48x` |
| SCALE_ADD | P1 新增 | scalar | 1 | 0.8201 | OpenCV `1.22x` |
| SQRT | P1 新增 | scalar | 1 | 0.9686 | OpenCV `1.03x` |
| SUBTRACT | 既有 | opencv_ui | 16 | 0.7735 | OpenCV `1.29x` |
| SUM | P1 新增 | public_header_baseline | 1 | 0.1598 | OpenCV `6.26x` |
| SWAP | P1 新增 | public_header_baseline | 1 | 0.2077 | OpenCV `4.81x` |
| TRANSPOSE | 既有 | headers_baseline | 16 | 0.5511 | OpenCV `1.81x` |
| VCONCAT | P1 新增 | public_header_baseline | 1 | 0.5137 | OpenCV `1.95x` |

### `imgproc`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | public_header_baseline | 1 | 0.0781 | OpenCV `12.81x` |
| ACCUMULATE_PRODUCT | P1 新增 | public_header_baseline | 1 | 0.0689 | OpenCV `14.52x` |
| ACCUMULATE_SQUARE | P1 新增 | public_header_baseline | 1 | 0.0706 | OpenCV `14.16x` |
| ACCUMULATE_WEIGHTED | P1 新增 | public_header_baseline | 1 | 0.0701 | OpenCV `14.27x` |
| ADAPTIVE_THRESHOLD | P1 新增 | public_header_baseline | 1 | 0.5265 | OpenCV `1.90x` |
| APPLY_COLOR_MAP | P1 新增 | public_header_baseline | 1 | 0.3266 | OpenCV `3.06x` |
| BILATERAL_FILTER | P1 新增 | public_header_baseline | 1 | 0.0412 | OpenCV `24.28x` |
| BLEND_LINEAR | P1 新增 | public_header_baseline | 1 | 0.3942 | OpenCV `2.54x` |
| BOX_FILTER | 既有 | box3x3, header_fastpath | 10 | 0.2961 | OpenCV `3.38x` |
| BUILD_PYRAMID | P1 新增 | public_header_baseline | 1 | 0.0065 | OpenCV `153.42x` |
| CANNY | 既有 | header_fastpath | 4 | 0.9434 | OpenCV `1.06x` |
| CONVERT_MAPS | P1 新增 | public_header_baseline | 1 | 0.0049 | OpenCV `203.50x` |
| COPY_MAKE_BORDER | 既有 | header_fastpath | 9 | 0.3703 | OpenCV `2.70x` |
| CREATE_HANNING_WINDOW | P1 新增 | public_header_baseline | 1 | 0.0286 | OpenCV `35.00x` |
| CVTCOLOR | 既有 | header_fastpath, opencv_ui | 17 | 0.5534 | OpenCV `1.81x` |
| CVT_COLOR_TWO_PLANE | P1 新增 | public_header_baseline | 1 | 0.1589 | OpenCV `6.29x` |
| DEMOSAICING | P1 新增 | public_header_baseline | 1 | 0.0014 | OpenCV `723.59x` |
| DILATE | 既有 | header_fastpath | 6 | 0.1166 | OpenCV `8.58x` |
| EQUALIZE_HIST | P1 新增 | public_header_baseline | 1 | 0.4953 | OpenCV `2.02x` |
| ERODE | 既有 | header_fastpath | 6 | 0.1131 | OpenCV `8.84x` |
| FILTER2D | 既有 | header_fastpath | 10 | 0.3689 | OpenCV `2.71x` |
| GAUSSIAN | 既有 | gauss_separable, header_fastpath | 10 | 0.2800 | OpenCV `3.57x` |
| GET_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 1.9186 | CVH `1.92x` |
| GET_DERIV_KERNELS | P1 新增 | public_header_baseline | 1 | 0.4383 | OpenCV `2.28x` |
| GET_GABOR_KERNEL | P1 新增 | public_header_baseline | 1 | 0.3271 | OpenCV `3.06x` |
| GET_GAUSSIAN_KERNEL | P1 新增 | public_header_baseline | 1 | 3.8778 | CVH `3.88x` |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.3457 | CVH `2.35x` |
| GET_RECT_SUB_PIX | P1 新增 | public_header_scalar | 4 | 0.0215 | OpenCV `46.51x` |
| GET_ROTATION_MATRIX_2D | P1 新增 | public_header_baseline | 1 | 0.9722 | OpenCV `1.03x` |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | public_header_baseline | 1 | 1.1173 | CVH `1.12x` |
| GET_STRUCTURING_ELEMENT | P1 新增 | public_header_baseline | 1 | 0.1842 | OpenCV `5.43x` |
| INTEGRAL | P1 新增 | public_header_baseline | 1 | 0.0276 | OpenCV `36.23x` |
| INVERT_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 0.4635 | OpenCV `2.16x` |
| LAPLACIAN | P1 新增 | public_header_baseline | 1 | 0.0128 | OpenCV `77.97x` |
| LUT | 既有 | header_fastpath | 6 | 0.5922 | OpenCV `1.69x` |
| MEDIAN_BLUR | P1 新增 | public_header_baseline | 1 | 0.0048 | OpenCV `208.07x` |
| PYR_DOWN | P1 新增 | public_header_baseline | 1 | 0.0076 | OpenCV `132.26x` |
| PYR_UP | P1 新增 | public_header_baseline | 1 | 0.0071 | OpenCV `139.92x` |
| REMAP | P1 新增 | public_header_scalar | 8 | 0.0494 | OpenCV `20.25x` |
| RESIZE | 既有 | header_fastpath, headers_baseline, opencv_ui | 10 | 0.6733 | OpenCV `1.49x` |
| SCHARR | P1 新增 | public_header_baseline | 1 | 0.0121 | OpenCV `82.75x` |
| SEP_FILTER2D | 既有 | header_fastpath | 10 | 0.4564 | OpenCV `2.19x` |
| SOBEL | 既有 | header_fastpath | 6 | 1.5897 | CVH `1.59x` |
| SPATIAL_GRADIENT | P1 新增 | public_header_baseline | 1 | 0.1571 | OpenCV `6.37x` |
| SQR_BOX_FILTER | P1 新增 | public_header_baseline | 1 | 0.0221 | OpenCV `45.29x` |
| STACK_BLUR | P1 新增 | public_header_baseline | 1 | 0.0800 | OpenCV `12.51x` |
| THRESHOLD | 既有 | header_fastpath, headers_baseline | 5 | 0.0449 | OpenCV `22.25x` |
| THRESHOLD_WITH_MASK | P1 新增 | public_header_baseline | 1 | 1.0440 | CVH `1.04x` |
| WARP_AFFINE | 既有 | headers_baseline | 9 | 0.0862 | OpenCV `11.59x` |
| WARP_PERSPECTIVE | P1 新增 | public_header_scalar | 4 | 0.0925 | OpenCV `10.81x` |

## 详细结果

### `core_mat`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.033642 | 0.025812 | 0.7673 | phase1_representative_case |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.280062 | 0.214517 | 0.7660 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043075 | 0.031950 | 0.7417 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042754 | 0.031329 | 0.7328 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.127446 | 0.101638 | 0.7975 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.801962 | 0.644042 | 0.8031 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.126575 | 0.099192 | 0.7837 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120571 | 0.092271 | 0.7653 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.355783 | 0.288587 | 0.8111 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062854 | 0.050238 | 0.7993 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009417 | 0.006829 | 0.7252 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009783 | 0.007050 | 0.7206 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.030067 | 0.023767 | 0.7905 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.203554 | 0.160871 | 0.7903 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.032317 | 0.024429 | 0.7559 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027204 | 0.022150 | 0.8142 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.092633 | 0.071546 | 0.7724 | correctness=upstream_pass |
| BITWISE_AND | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.048696 | 0.025758 | 0.5290 | phase1_representative_case |
| BITWISE_NOT | P1 新增 | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.034987 | 0.017396 | 0.4972 | phase1_representative_case |
| BITWISE_OR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036396 | 0.025817 | 0.7093 | phase1_representative_case |
| BITWISE_XOR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036383 | 0.025800 | 0.7091 | phase1_representative_case |
| BORDER_INTERPOLATE | P1 新增 | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.006288 | 0.005572 | 0.8862 | phase1_representative_case;micro_iterations=10000 |
| BROADCAST | P1 新增 | row_to_image_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 12.355787 | 0.002850 | 0.0002 | phase1_representative_case |
| CHECK_RANGE | P1 新增 | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.166558 | 0.099354 | 0.5965 | phase1_representative_case |
| CONVERT_FP16 | P1 新增 | f32c1_to_fp16 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.017092 | 0.020729 | 1.2128 | phase1_representative_case |
| CONVERT_SCALE_ABS | P1 新增 | f32c3_to_u8c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.085588 | 0.070792 | 0.8271 | phase1_representative_case |
| COPY_TO | P1 新增 | masked_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.639454 | 0.023942 | 0.0066 | phase1_representative_case |
| COUNT_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.286433 | 0.009550 | 0.0333 | phase1_representative_case |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.270737 | 0.219367 | 0.8103 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043146 | 0.031725 | 0.7353 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042729 | 0.032050 | 0.7501 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.123675 | 0.099854 | 0.8074 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.777433 | 0.644617 | 0.8292 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.122417 | 0.094925 | 0.7754 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120417 | 0.098825 | 0.8207 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.353538 | 0.291996 | 0.8259 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 1080x1920 | 1.923642 | 0.486058 | 0.2527 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 479x641 | 0.279963 | 0.074529 | 0.2662 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 480x640 | 0.260517 | 0.067054 | 0.2574 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 720x1280 | 0.812237 | 0.204767 | 0.2521 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 3.857200 | 1.473575 | 0.3820 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.590208 | 0.215529 | 0.3652 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.546687 | 0.216079 | 0.3953 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 1.635046 | 0.674146 | 0.4123 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | P1 新增 | bounded_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.170333 | 0.081513 | 0.4785 | phase1_representative_case |
| EXTRACT_CHANNEL | P1 新增 | channel1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.320725 | 0.045671 | 0.0138 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.598533 | 0.203412 | 0.3399 | phase1_representative_case |
| FLIP | P1 新增 | horizontal_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.413933 | 0.016521 | 0.0048 | phase1_representative_case |
| FLIP_ND | P1 新增 | axis1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 12.479850 | 0.128071 | 0.0103 | phase1_representative_case |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.210547 | 0.003661 | 0.0174 | correctness=upstream_pass;iters=8 |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 1.997291 | 0.025208 | 0.0126 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_end_to_end | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.384375 | 0.170000 | 0.0098 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 128x128x128 | 0.215859 | 0.003703 | 0.0172 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 256x256x256 | 2.297709 | 0.025208 | 0.0110 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | headers_baseline | CV_32F | 1 | continuous | 512x512x512 | 17.147583 | 0.176709 | 0.0103 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.280704 | 0.000013 | 0.0000 | phase1_representative_case |
| HCONCAT | P1 新增 | two_halves_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.013196 | 0.007000 | 0.5305 | phase1_representative_case |
| INSERT_CHANNEL | P1 新增 | channel1_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.187396 | 0.043767 | 0.0137 | phase1_representative_case |
| IN_RANGE | P1 新增 | scalar_bounds_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.206033 | 0.127575 | 0.6192 | phase1_representative_case |
| LOG | P1 新增 | positive_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.233167 | 0.135783 | 0.5823 | phase1_representative_case |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024800 | 0.025450 | 1.0262 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003829 | 0.004054 | 1.0588 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.009200 | 0.008567 | 0.9312 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.015867 | 0.014617 | 0.9212 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.087908 | 0.078425 | 0.8921 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.011371 | 0.013112 | 1.1532 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.020338 | 0.020421 | 1.0041 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.037604 | 0.034658 | 0.9217 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024054 | 0.024438 | 1.0159 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003767 | 0.003379 | 0.8971 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.008217 | 0.009004 | 1.0958 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.014183 | 0.014417 | 1.0164 |  |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000014 | 0.000001 | 0.0750 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000014 | 0.000001 | 0.0807 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000046 | 0.000002 | 0.0537 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000024 | 0.000002 | 0.0672 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000046 | 0.000015 | 0.3361 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000046 | 0.000015 | 0.3347 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000068 | 0.000024 | 0.3443 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000045 | 0.000015 | 0.3368 | micro_iters_x1000 |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.994700 | 0.024413 | 0.0122 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.288475 | 0.003858 | 0.0134 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.617629 | 0.007367 | 0.0119 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 1.173492 | 0.012133 | 0.0103 |  |
| MAX | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.039904 | 0.026096 | 0.6540 | phase1_representative_case |
| MEAN | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 1.189979 | 0.208913 | 0.1756 | phase1_representative_case |
| MEAN_STD_DEV | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 1.358846 | 0.142596 | 0.1049 | phase1_representative_case |
| MIN | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.040250 | 0.026029 | 0.6467 | phase1_representative_case |
| MIN_MAX_IDX | P1 新增 | f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.616596 | 0.034404 | 0.0558 | phase1_representative_case |
| MIN_MAX_LOC | P1 新增 | f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.617758 | 0.034354 | 0.0556 | phase1_representative_case |
| MIX_CHANNELS | P1 新增 | reverse_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 9.602342 | 0.130392 | 0.0136 | phase1_representative_case |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.271017 | 0.216771 | 0.7998 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043267 | 0.031721 | 0.7331 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042996 | 0.031854 | 0.7409 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.123029 | 0.097950 | 0.7962 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.797596 | 0.665942 | 0.8349 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.122500 | 0.096113 | 0.7846 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120358 | 0.098062 | 0.8148 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.353596 | 0.303925 | 0.8595 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062583 | 0.047829 | 0.7642 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009575 | 0.007042 | 0.7354 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009429 | 0.007046 | 0.7472 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.029592 | 0.023813 | 0.8047 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.205908 | 0.159696 | 0.7756 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.031179 | 0.018546 | 0.5948 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027192 | 0.021988 | 0.8086 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.090038 | 0.067117 | 0.7454 | correctness=upstream_pass |
| NORM | P1 新增 | l2_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.557163 | 0.037512 | 0.0673 | phase1_representative_case |
| NORMALIZE | P1 新增 | l2_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 1.233183 | 0.057637 | 0.0467 | phase1_representative_case |
| PATCH_NANS | P1 新增 | one_nan_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.020838 | 0.016862 | 0.8092 | phase1_representative_case |
| POW | P1 新增 | power_1_75_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.548958 | 0.337462 | 0.6147 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.774913 | 0.015408 | 0.0199 | phase1_representative_case |
| REDUCE_ARG_MAX | P1 新增 | axis0_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.978446 | 0.152846 | 0.1562 | phase1_representative_case |
| REDUCE_ARG_MIN | P1 新增 | axis0_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.929617 | 0.150721 | 0.1621 | phase1_representative_case |
| REPEAT | P1 新增 | two_by_two_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 3.430013 | 0.005137 | 0.0015 | phase1_representative_case |
| ROTATE | P1 新增 | clockwise90_u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 3.495613 | 0.101383 | 0.0290 | phase1_representative_case |
| SCALE_ADD | P1 新增 | f32c3 | scalar | CV_32F | 3 | continuous | 480x640 | 0.125754 | 0.103125 | 0.8201 | phase1_representative_case |
| SQRT | P1 新增 | positive_f32c1 | scalar | CV_32F | 1 | continuous | 480x640 | 0.043546 | 0.042179 | 0.9686 | phase1_representative_case |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.271246 | 0.215271 | 0.7936 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043158 | 0.032000 | 0.7415 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042812 | 0.032262 | 0.7536 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.127575 | 0.101096 | 0.7924 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.805521 | 0.617596 | 0.7667 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.126700 | 0.095113 | 0.7507 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.120171 | 0.097725 | 0.8132 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.354783 | 0.288304 | 0.8126 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062646 | 0.050296 | 0.8029 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009417 | 0.007071 | 0.7509 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009408 | 0.007033 | 0.7476 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.032250 | 0.023767 | 0.7370 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.209267 | 0.160867 | 0.7687 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.030762 | 0.024658 | 0.8016 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027217 | 0.022033 | 0.8096 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.092992 | 0.068992 | 0.7419 | correctness=upstream_pass |
| SUM | P1 新增 | f32c3 | public_header_baseline | CV_32F | 3 | continuous | 480x640 | 1.307183 | 0.208883 | 0.1598 | phase1_representative_case |
| SWAP | P1 新增 | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000024 | 0.000005 | 0.2077 | phase1_representative_case;micro_iterations=10000 |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 1080x1920 | 0.787963 | 0.583175 | 0.7401 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 479x641 | 0.164917 | 0.035133 | 0.2130 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.157858 | 0.075533 | 0.4785 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 1 | continuous | 720x1280 | 0.377904 | 0.317304 | 0.8396 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 1080x1920 | 0.698125 | 1.463533 | 2.0964 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 479x641 | 0.119225 | 0.126412 | 1.0603 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.120633 | 0.162121 | 1.3439 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_32F | 3 | continuous | 720x1280 | 0.279829 | 0.564050 | 2.0157 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.575071 | 0.131412 | 0.2285 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.145608 | 0.010583 | 0.0727 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.140187 | 0.006729 | 0.0480 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.297463 | 0.028571 | 0.0960 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 1080x1920 | 0.407712 | 0.778746 | 1.9100 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 479x641 | 0.109508 | 0.087358 | 0.7977 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 480x640 | 0.108363 | 0.088908 | 0.8205 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | headers_baseline | CV_8U | 3 | continuous | 720x1280 | 0.206062 | 0.408083 | 1.9804 | correctness=upstream_pass |
| VCONCAT | P1 新增 | two_halves_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.007942 | 0.004079 | 0.5137 | phase1_representative_case |

### `imgproc`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.251962 | 0.019667 | 0.0781 | phase1_representative_case |
| ACCUMULATE_PRODUCT | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.312421 | 0.021521 | 0.0689 | phase1_representative_case |
| ACCUMULATE_SQUARE | P1 新增 | u8c1_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.278896 | 0.019696 | 0.0706 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | P1 新增 | alpha0_1_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.310567 | 0.021758 | 0.0701 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | P1 新增 | mean11_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.435633 | 0.229371 | 0.5265 | phase1_representative_case |
| APPLY_COLOR_MAP | P1 新增 | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.731842 | 0.238996 | 0.3266 | phase1_representative_case |
| BILATERAL_FILTER | P1 新增 | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 23.485558 | 0.967396 | 0.0412 | phase1_representative_case |
| BLEND_LINEAR | P1 新增 | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.559571 | 0.220608 | 0.3942 | phase1_representative_case |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.493362 | 0.298371 | 0.1998 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.276658 | 0.046483 | 0.1680 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.280458 | 0.047787 | 0.1704 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.703979 | 0.134154 | 0.1906 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.179629 | 0.106925 | 0.5953 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.317142 | 0.314371 | 0.9913 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.415517 | 0.414746 | 0.9981 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.691658 | 0.140233 | 0.2028 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 0.699588 | 0.133304 | 0.1905 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 0.846738 | 0.176679 | 0.2087 |  |
| BUILD_PYRAMID | P1 新增 | levels3_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 3.013962 | 0.019646 | 0.0065 | phase1_representative_case |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 29.630275 | 28.240812 | 0.9531 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 4.472446 | 4.323150 | 0.9666 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 4.850563 | 4.463533 | 0.9202 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 13.589783 | 12.698400 | 0.9344 |  |
| CONVERT_MAPS | P1 新增 | f32_pair_to_fixed | public_header_baseline | CV_32F | 2 | continuous | 480x640 | 12.453646 | 0.061192 | 0.0049 | phase1_representative_case |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.071758 | 0.044913 | 0.6259 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.068217 | 0.006950 | 0.1019 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.060867 | 0.007258 | 0.1192 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.067333 | 0.020233 | 0.3005 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.068542 | 0.026354 | 0.3845 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.081533 | 0.087242 | 1.0700 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.088825 | 0.103304 | 1.1630 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.073013 | 0.022496 | 0.3081 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.070671 | 0.027454 | 0.3885 |  |
| CREATE_HANNING_WINDOW | P1 新增 | 64x64_f32 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.040100 | 0.001146 | 0.0286 | phase1_representative_case |
| CVTCOLOR | 既有 | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.114308 | 0.020842 | 0.1823 |  |
| CVTCOLOR | 既有 | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.079304 | 0.048188 | 0.6076 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.221246 | 0.219350 | 0.9914 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.030687 | 0.030600 | 0.9971 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.041642 | 0.033725 | 0.8099 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.092363 | 0.091917 | 0.9952 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.035204 | 0.035146 | 0.9983 |  |
| CVTCOLOR | 既有 | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.129158 | 0.061262 | 0.4743 |  |
| CVTCOLOR | 既有 | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.090621 | 0.063750 | 0.7035 |  |
| CVTCOLOR | 既有 | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.088862 | 0.017208 | 0.1937 |  |
| CVTCOLOR | 既有 | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.118317 | 0.081008 | 0.6847 |  |
| CVTCOLOR | 既有 | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.131346 | 0.066771 | 0.5084 |  |
| CVTCOLOR | 既有 | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.099538 | 0.041537 | 0.4173 |  |
| CVTCOLOR | 既有 | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 0.226783 | 0.076121 | 0.3357 |  |
| CVTCOLOR | 既有 | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.136458 | 0.074592 | 0.5466 |  |
| CVTCOLOR | 既有 | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.116396 | 0.061313 | 0.5268 |  |
| CVTCOLOR | 既有 | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.134308 | 0.071971 | 0.5359 |  |
| CVT_COLOR_TWO_PLANE | P1 新增 | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.413038 | 0.065625 | 0.1589 | phase1_representative_case |
| DEMOSAICING | P1 新增 | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 28.250308 | 0.039046 | 0.0014 | phase1_representative_case |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.096337 | 0.147663 | 0.1347 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.273308 | 0.022867 | 0.0837 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.275738 | 0.024967 | 0.0905 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.574292 | 0.069992 | 0.1219 |  |
| DILATE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.493950 | 0.068396 | 0.1385 |  |
| DILATE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.615371 | 0.089762 | 0.1459 |  |
| EQUALIZE_HIST | P1 新增 | u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.200163 | 0.099142 | 0.4953 | phase1_representative_case |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.127700 | 0.147454 | 0.1308 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.276129 | 0.022808 | 0.0826 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.284438 | 0.023367 | 0.0822 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.582358 | 0.069812 | 0.1199 |  |
| ERODE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.501504 | 0.068029 | 0.1356 |  |
| ERODE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.604279 | 0.087871 | 0.1454 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 2.398654 | 0.670025 | 0.2793 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.444183 | 0.094121 | 0.2119 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.409321 | 0.103696 | 0.2533 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.108188 | 0.303383 | 0.2738 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.428062 | 0.079212 | 0.1850 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.466525 | 0.211875 | 0.4542 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.434063 | 0.270725 | 0.6237 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.563092 | 0.287300 | 0.5102 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.522867 | 0.304908 | 0.5831 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.551675 | 0.402579 | 0.7297 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.577429 | 0.226562 | 0.1436 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.353617 | 0.031983 | 0.0904 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.352271 | 0.033013 | 0.0937 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.788583 | 0.098450 | 0.1248 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 0.305554 | 0.115671 | 0.3786 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 0.390604 | 0.338717 | 0.8672 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.367879 | 0.443646 | 1.2060 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 0.510938 | 0.108621 | 0.2126 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 0.511321 | 0.361867 | 0.7077 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 0.421571 | 0.137742 | 0.3267 |  |
| GET_AFFINE_TRANSFORM | P1 新增 | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000135 | 0.000258 | 1.9186 | phase1_representative_case;micro_iterations=10000 |
| GET_DERIV_KERNELS | P1 新增 | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000204 | 0.000089 | 0.4383 | phase1_representative_case;micro_iterations=10000 |
| GET_GABOR_KERNEL | P1 新增 | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.003022 | 0.000989 | 0.3271 | phase1_representative_case;micro_iterations=10000 |
| GET_GAUSSIAN_KERNEL | P1 新增 | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000240 | 0.000932 | 3.8778 | phase1_representative_case;micro_iterations=10000 |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000294 | 0.000691 | 2.3457 | phase1_representative_case;micro_iterations=10000 |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 48.213408 | 0.985242 | 0.0204 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 7.081371 | 0.160642 | 0.0227 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 6.995679 | 0.148792 | 0.0213 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 21.428517 | 0.464596 | 0.0217 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | P1 新增 | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000077 | 0.000075 | 0.9722 | phase1_representative_case;micro_iterations=10000 |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000004 | 0.000005 | 1.1173 | phase1_representative_case;micro_iterations=10000 |
| GET_STRUCTURING_ELEMENT | P1 新增 | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000421 | 0.000078 | 0.1842 | phase1_representative_case;micro_iterations=10000 |
| INTEGRAL | P1 新增 | u8c1_to_s32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.503288 | 0.041496 | 0.0276 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | P1 新增 | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000135 | 0.000062 | 0.4635 | phase1_representative_case;micro_iterations=10000 |
| LAPLACIAN | P1 新增 | ksize3_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 9.276821 | 0.118987 | 0.0128 | phase1_representative_case |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.200783 | 0.194513 | 0.9688 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.092050 | 0.028579 | 0.3105 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.085500 | 0.029825 | 0.3488 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.128763 | 0.089154 | 0.6924 |  |
| LUT | 既有 | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.117392 | 0.085583 | 0.7290 |  |
| LUT | 既有 | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.140104 | 0.114133 | 0.8146 |  |
| MEDIAN_BLUR | P1 新增 | ksize5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 51.672300 | 0.248317 | 0.0048 | phase1_representative_case |
| PYR_DOWN | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 7.735775 | 0.058492 | 0.0076 | phase1_representative_case |
| PYR_UP | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 20.233700 | 0.144613 | 0.0071 | phase1_representative_case |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 100.476200 | 4.868079 | 0.0485 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 14.834537 | 0.719092 | 0.0485 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 14.908117 | 0.737054 | 0.0494 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 44.223546 | 2.247162 | 0.0508 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 105.409700 | 5.525533 | 0.0524 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 15.499408 | 0.762304 | 0.0492 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 15.568858 | 0.756117 | 0.0486 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 46.988662 | 2.245287 | 0.0478 | no qualified SIMD fast path |
| RESIZE | 既有 | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.200263 | 0.093537 | 0.4671 |  |
| RESIZE | 既有 | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.449021 | 0.279092 | 0.6216 |  |
| RESIZE | 既有 | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.144504 | 0.065062 | 0.4502 |  |
| RESIZE | 既有 | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.141554 | 0.064562 | 0.4561 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.130438 | 0.087246 | 0.6689 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.018733 | 0.012279 | 0.6555 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.019667 | 0.015450 | 0.7856 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.055025 | 0.036212 | 0.6581 |  |
| RESIZE | 既有 | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.087242 | 0.099317 | 1.1384 |  |
| RESIZE | 既有 | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.082588 | 0.102763 | 1.2443 |  |
| SCHARR | P1 新增 | dx1_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 9.029246 | 0.109117 | 0.0121 | phase1_representative_case |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.587254 | 0.658525 | 0.4149 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.373196 | 0.094512 | 0.2533 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.362717 | 0.103537 | 0.2854 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.807071 | 0.287192 | 0.3558 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.356162 | 0.086508 | 0.2429 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.444517 | 0.239683 | 0.5392 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.422421 | 0.311529 | 0.7375 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.462579 | 0.285175 | 0.6165 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.449604 | 0.302071 | 0.6719 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.416754 | 0.382587 | 0.9180 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.394767 | 0.897413 | 2.2733 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.116871 | 0.120838 | 1.0339 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.115696 | 0.134438 | 1.1620 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.211175 | 0.359450 | 1.7021 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.181462 | 0.335050 | 1.8464 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.235071 | 0.441925 | 1.8800 |  |
| SPATIAL_GRADIENT | P1 新增 | ksize3_u8_to_s16 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.228750 | 0.035933 | 0.1571 | phase1_representative_case |
| SQR_BOX_FILTER | P1 新增 | 3x3_u8_to_f32 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 8.807829 | 0.194487 | 0.0221 | phase1_representative_case |
| STACK_BLUR | P1 新增 | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.664133 | 0.133063 | 0.0800 | phase1_representative_case |
| THRESHOLD | 既有 | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.386412 | 0.068483 | 0.1772 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 1.232258 | 0.035950 | 0.0292 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.147067 | 0.004833 | 0.0329 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.162892 | 0.005417 | 0.0333 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.436050 | 0.014154 | 0.0325 |  |
| THRESHOLD_WITH_MASK | P1 新增 | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.182900 | 0.190954 | 1.0440 | phase1_representative_case |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 23.388938 | 1.979058 | 0.0846 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 479x641 | 3.426446 | 0.319479 | 0.0932 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 480x640 | 3.573788 | 0.310921 | 0.0870 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 10.379383 | 0.879300 | 0.0847 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 3.045150 | 0.531892 | 0.1747 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 7.503913 | 0.730312 | 0.0973 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 9.872117 | 0.796779 | 0.0807 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c3 | headers_baseline | CV_8U | 3 | continuous | 480x640 | 9.878000 | 0.813888 | 0.0824 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c4 | headers_baseline | CV_8U | 4 | continuous | 480x640 | 12.911888 | 0.518721 | 0.0402 |  |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 72.383375 | 6.530242 | 0.0902 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 10.558842 | 0.981829 | 0.0930 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 10.657783 | 1.017833 | 0.0955 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 31.992367 | 2.923429 | 0.0914 | no qualified SIMD fast path |

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
