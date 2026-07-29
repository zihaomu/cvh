# cvh vs OpenCV Benchmark Report (full)

生成时间（UTC）：`2026-07-24 16:16:32Z`

## 当前项目状态

- `cvh`（cv-header-only）是独立的纯 header-only 库，不依赖项目内 `.cpp` 扩展层。
- Mode B 只比较当前 `cvh::headers_fast` 与同机编译的 upstream OpenCV；`cvh::headers_fast` 表示最快 header-only 构建配置。
- `cvh::headers_fast` 完整继承 `cvh::headers`。算子没有专用 fast-path 时继续执行继承的 header 实现并参与 benchmark，不因缺少 SIMD 特化而跳过。
- Core/Imgproc 第一阶段完成后，名称级可调用覆盖为 `107/220`：Core `57/97`，Imgproc `50/123`。
- Core 的 `add/subtract/multiply/divide/transpose/GEMM` 已迁入 ODR-safe headers；本报告通过公共 API 测量，不链接 legacy core 对象。
- OpenCV Universal Intrinsics 是默认 SIMD 方言，kernel 直接使用 OpenCV UI；项目已移除 xsimd 性能路径。
- Core F32 `patchNaNs/exp/log/pow` 已接入 UI；`pow` 分离整数指数与通用指数，特殊值 block 保留 scalar fallback。
- Core `countNonZero/hasNonZero` 已接入 UI；计数使用分段 widen 归约，存在性检测按块 early-exit。
- Core `findNonZero` 已接入稀疏感知 UI；全零 block 直接跳过，连续稠密 block 自适应切回 typed lane 枚举，并保持 row-major 坐标顺序。
- Core `sum/mean/meanStdDev` 已接入 C1-C4 channel-aware UI；`sum/mean` 共享 widen sum/count，`meanStdDev` 使用中心化 block statistics 与 Chan merge。
- Core `minMaxIdx/minMaxLoc/reduceArgMin/reduceArgMax` 已接入 UI；极值与索引在同一遍扫描中更新，并保留 first/last tie 语义。
- Core `norm/normalize` 已接入 UI；`norm` 覆盖 U8/F32 单/双输入的 L1/L2/Inf，`normalize` 复用 norm/minmax 归约并向量化 F32 apply-scale。
- P-ACC-2 至 P-ACC-7 已完成 Apple ARM 收尾，覆盖 Core 归约、布局、通道与 GEMM，以及 Imgproc 滤波、几何、非线性、形态学、累积和强度变换；真实 x86 SSE/AVX 运行验证仍是外部 gate。
- ARM 当前关注 NEON，本次实测平台为 Apple ARM；x86 目标是 SSE/AVX 系列，RVV 因 scalable vector 设计问题暂缓。
- Imgproc legacy `.cpp` fast-path 已迁入 ODR-safe detail headers；resize/cvtColor、共享 filter、几何采样、morphology、threshold、LUT/histogram 和 accumulate family 均从公共 header API 进入。
- 第一阶段新增的 `79` 个操作族已全部进入 Mode B，本报告包含 `111` 个 P1 性能 case。
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
| 专用 kernel | `cvtColor`、特定 `resize`、core 逐元素、统计/非零归约与 F32 数学 UI kernel | 实际命中时记录为 `dispatch_path=opencv_ui` |
| Header fast-path | 行并行 filter、LUT、border、Sobel、Canny、morphology | 记录为 `dispatch_path=header_fastpath` |
| 通用实现 | `cvh::headers` 中的 header baseline | 无专用 fast-path 时自动继承，记录为 `headers_baseline` 或 `public_header_scalar` |
| 对照实现 | upstream OpenCV `core` / `imgproc` | 相同输入、尺寸、border 和线程配置 |

## 运行配置

- Profile：`full`
- CVH 实现：`cvh_headers_fast`
- 采样：`warmup=2, iters=100, repeats=3`
- 线程数：`1`
- OpenMP：`dynamic=false, proc_bind=close`
- 主机：`Darwin arm64`
- CPU：`Apple M5`
- 编译器：`Apple clang version 21.0.0 (clang-2100.0.123.102)`
- 构建类型：`Release`
- CVH commit：`ab40a0865afbfc1e07070b411fca54379cd0bf70` + dirty
- OpenCV：`4.14.0`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8` + dirty
- 原始数据：`2026-07-24-opencv-upstream-performance.csv`；元数据：`2026-07-24-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`340`；有效：`339`；不支持：`1`。
- `OpenCV/CVH` 几何平均：`0.5563`；中位数：`0.7136`。
- CVH 更快：`58` 个；OpenCV 更快或相当：`281` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 172 | 0.6659 | 0.7827 | 29 | 143 |
| imgproc | 167 | 0.4623 | 0.4550 | 29 | 138 |

## 算子级概览

### `core_mat`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | opencv_ui | 1 | 0.7602 | OpenCV `1.32x` |
| ADD | 既有 | opencv_ui | 16 | 0.7594 | OpenCV `1.32x` |
| BITWISE_AND | P1 新增 | opencv_ui | 1 | 0.7030 | OpenCV `1.42x` |
| BITWISE_NOT | P1 新增 | opencv_ui | 1 | 0.4704 | OpenCV `2.13x` |
| BITWISE_OR | P1 新增 | opencv_ui | 1 | 0.7092 | OpenCV `1.41x` |
| BITWISE_XOR | P1 新增 | opencv_ui | 1 | 0.7068 | OpenCV `1.41x` |
| BORDER_INTERPOLATE | P1 新增 | public_header_baseline | 1 | 0.9706 | OpenCV `1.03x` |
| BROADCAST | P1 新增 | scalar | 1 | 0.7817 | OpenCV `1.28x` |
| CHECK_RANGE | P1 新增 | public_header_baseline | 1 | 0.6064 | OpenCV `1.65x` |
| CONVERT_FP16 | P1 新增 | opencv_ui | 1 | 1.2198 | CVH `1.22x` |
| CONVERT_SCALE_ABS | P1 新增 | opencv_ui | 1 | 0.8177 | OpenCV `1.22x` |
| COPY_TO | P1 新增 | opencv_ui | 1 | 0.9911 | OpenCV `1.01x` |
| COUNT_NON_ZERO | P1 新增 | opencv_ui | 1 | 0.9951 | OpenCV `1.00x` |
| DIVIDE | 既有 | opencv_ui, scalar | 16 | 0.4977 | OpenCV `2.01x` |
| EXP | P1 新增 | opencv_ui | 1 | 0.4781 | OpenCV `2.09x` |
| EXTRACT_CHANNEL | P1 新增 | opencv_ui | 1 | 2.7621 | CVH `2.76x` |
| FIND_NON_ZERO | P1 新增 | opencv_ui | 3 | 2.6159 | CVH `2.62x` |
| FLIP | P1 新增 | opencv_ui | 1 | 0.9978 | OpenCV `1.00x` |
| FLIP_ND | P1 新增 | opencv_ui | 1 | 7.7684 | CVH `7.77x` |
| GEMM | 既有 | opencv_ui | 6 | 0.0842 | OpenCV `11.87x` |
| HAS_NON_ZERO | P1 新增 | opencv_ui | 1 | 0.9520 | OpenCV `1.05x` |
| HCONCAT | P1 新增 | scalar | 1 | 1.3762 | CVH `1.38x` |
| INSERT_CHANNEL | P1 新增 | opencv_ui | 1 | 1.7068 | CVH `1.71x` |
| IN_RANGE | P1 新增 | opencv_ui | 1 | 0.6176 | OpenCV `1.62x` |
| LOG | P1 新增 | opencv_ui | 1 | 0.5760 | OpenCV `1.74x` |
| MAT_CLONE | 既有 | headers_baseline | 4 | 0.9657 | OpenCV `1.04x` |
| MAT_CONVERTTO | 既有 | headers_baseline | 4 | 1.0045 | CVH `1.00x` |
| MAT_COPYTO | 既有 | headers_baseline | 4 | 0.9859 | OpenCV `1.01x` |
| MAT_CREATE | 既有 | headers_baseline | 4 | 0.0728 | OpenCV `13.73x` |
| MAT_RESHAPE | 既有 | headers_baseline | 4 | 0.3408 | OpenCV `2.93x` |
| MAT_SETTO | 既有 | headers_baseline | 4 | 0.9412 | OpenCV `1.06x` |
| MAX | P1 新增 | opencv_ui | 1 | 0.6333 | OpenCV `1.58x` |
| MEAN | P1 新增 | opencv_ui | 1 | 1.9977 | CVH `2.00x` |
| MEAN_STD_DEV | P1 新增 | opencv_ui | 1 | 0.3214 | OpenCV `3.11x` |
| MIN | P1 新增 | opencv_ui | 1 | 0.6438 | OpenCV `1.55x` |
| MIN_MAX_IDX | P1 新增 | opencv_ui | 1 | 0.7136 | OpenCV `1.40x` |
| MIN_MAX_LOC | P1 新增 | opencv_ui | 1 | 0.7054 | OpenCV `1.42x` |
| MIX_CHANNELS | P1 新增 | opencv_ui | 1 | 3.6253 | CVH `3.63x` |
| MULTIPLY | 既有 | opencv_ui | 16 | 0.7813 | OpenCV `1.28x` |
| NORM | P1 新增 | opencv_ui | 6 | 0.2616 | OpenCV `3.82x` |
| NORMALIZE | P1 新增 | opencv_ui | 4 | 0.4045 | OpenCV `2.47x` |
| PATCH_NANS | P1 新增 | opencv_ui | 1 | 0.9526 | OpenCV `1.05x` |
| POW | P1 新增 | opencv_ui | 1 | 0.5857 | OpenCV `1.71x` |
| REDUCE | P1 新增 | opencv_ui | 10 | 0.4663 | OpenCV `2.14x` |
| REDUCE_ARG_MAX | P1 新增 | opencv_ui | 1 | 0.9653 | OpenCV `1.04x` |
| REDUCE_ARG_MIN | P1 新增 | opencv_ui | 1 | 0.9537 | OpenCV `1.05x` |
| REPEAT | P1 新增 | scalar | 1 | 1.0403 | CVH `1.04x` |
| ROTATE | P1 新增 | opencv_ui | 1 | 0.7453 | OpenCV `1.34x` |
| SCALE_ADD | P1 新增 | scalar | 1 | 0.8233 | OpenCV `1.21x` |
| SQRT | P1 新增 | scalar | 1 | 0.9928 | OpenCV `1.01x` |
| SUBTRACT | 既有 | opencv_ui | 16 | 0.7811 | OpenCV `1.28x` |
| SUM | P1 新增 | opencv_ui | 1 | 1.9993 | CVH `2.00x` |
| SWAP | P1 新增 | public_header_baseline | 1 | 1.0000 | OpenCV `1.00x` |
| TRANSPOSE | 既有 | opencv_ui, scalar | 16 | 1.1254 | CVH `1.13x` |
| VCONCAT | P1 新增 | scalar | 1 | 1.0383 | CVH `1.04x` |

### `imgproc`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | opencv_ui | 1 | 0.3117 | OpenCV `3.21x` |
| ACCUMULATE_PRODUCT | P1 新增 | opencv_ui | 1 | 0.2598 | OpenCV `3.85x` |
| ACCUMULATE_SQUARE | P1 新增 | opencv_ui | 1 | 0.3104 | OpenCV `3.22x` |
| ACCUMULATE_WEIGHTED | P1 新增 | opencv_ui | 1 | 0.3438 | OpenCV `2.91x` |
| ADAPTIVE_THRESHOLD | P1 新增 | opencv_ui | 1 | 0.7748 | OpenCV `1.29x` |
| APPLY_COLOR_MAP | P1 新增 | public_header_baseline | 1 | 0.3482 | OpenCV `2.87x` |
| BILATERAL_FILTER | P1 新增 | public_header_baseline | 1 | 0.0522 | OpenCV `19.15x` |
| BLEND_LINEAR | P1 新增 | public_header_baseline | 1 | 0.4260 | OpenCV `2.35x` |
| BOX_FILTER | 既有 | box3x3, header_fastpath | 10 | 0.2869 | OpenCV `3.49x` |
| BUILD_PYRAMID | P1 新增 | public_header_baseline | 1 | 0.0756 | OpenCV `13.23x` |
| CANNY | 既有 | header_fastpath | 4 | 0.9343 | OpenCV `1.07x` |
| CONVERT_MAPS | P1 新增 | opencv_ui | 1 | 0.1595 | OpenCV `6.27x` |
| COPY_MAKE_BORDER | 既有 | header_fastpath | 9 | 0.3840 | OpenCV `2.60x` |
| CREATE_HANNING_WINDOW | P1 新增 | opencv_ui | 1 | 1.8557 | CVH `1.86x` |
| CVTCOLOR | 既有 | header_fastpath, opencv_ui | 17 | 0.5595 | OpenCV `1.79x` |
| CVT_COLOR_TWO_PLANE | P1 新增 | public_header_baseline | 1 | 0.2017 | OpenCV `4.96x` |
| DEMOSAICING | P1 新增 | public_header_baseline | 1 | 0.0964 | OpenCV `10.37x` |
| DILATE | 既有 | header_fastpath | 6 | 0.2384 | OpenCV `4.19x` |
| EQUALIZE_HIST | P1 新增 | opencv_ui | 1 | 1.0446 | CVH `1.04x` |
| ERODE | 既有 | header_fastpath | 6 | 0.2333 | OpenCV `4.29x` |
| FILTER2D | 既有 | header_fastpath | 10 | 0.4177 | OpenCV `2.39x` |
| GAUSSIAN | 既有 | gauss_separable, header_fastpath | 10 | 0.2779 | OpenCV `3.60x` |
| GET_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 1.9316 | CVH `1.93x` |
| GET_DERIV_KERNELS | P1 新增 | public_header_baseline | 1 | 0.6667 | OpenCV `1.50x` |
| GET_GABOR_KERNEL | P1 新增 | public_header_baseline | 1 | 0.9652 | OpenCV `1.04x` |
| GET_GAUSSIAN_KERNEL | P1 新增 | public_header_baseline | 1 | 3.8745 | CVH `3.87x` |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.5103 | CVH `2.51x` |
| GET_RECT_SUB_PIX | P1 新增 | public_header_scalar | 4 | 11.7642 | CVH `11.76x` |
| GET_ROTATION_MATRIX_2D | P1 新增 | public_header_baseline | 1 | 0.9331 | OpenCV `1.07x` |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | public_header_baseline | 1 | 1.1010 | CVH `1.10x` |
| GET_STRUCTURING_ELEMENT | P1 新增 | public_header_baseline | 1 | 0.7549 | OpenCV `1.32x` |
| INTEGRAL | P1 新增 | opencv_ui | 1 | 0.5577 | OpenCV `1.79x` |
| INVERT_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 1.3611 | CVH `1.36x` |
| LAPLACIAN | P1 新增 | opencv_ui | 1 | 0.4192 | OpenCV `2.39x` |
| LUT | 既有 | header_fastpath | 6 | 0.8608 | OpenCV `1.16x` |
| MEDIAN_BLUR | P1 新增 | public_header_baseline | 1 | 0.0302 | OpenCV `33.13x` |
| PYR_DOWN | P1 新增 | public_header_baseline | 1 | 0.1234 | OpenCV `8.10x` |
| PYR_UP | P1 新增 | public_header_baseline | 1 | 0.1416 | OpenCV `7.06x` |
| REMAP | P1 新增 | public_header_scalar | 8 | 0.2761 | OpenCV `3.62x` |
| RESIZE | 既有 | header_fastpath, headers_baseline, opencv_ui | 10 | 0.6657 | OpenCV `1.50x` |
| SCHARR | P1 新增 | opencv_ui | 1 | 0.3202 | OpenCV `3.12x` |
| SEP_FILTER2D | 既有 | header_fastpath | 10 | 0.5240 | OpenCV `1.91x` |
| SOBEL | 既有 | header_fastpath | 6 | 1.5768 | CVH `1.58x` |
| SPATIAL_GRADIENT | P1 新增 | opencv_ui | 1 | 0.4362 | OpenCV `2.29x` |
| SQR_BOX_FILTER | P1 新增 | opencv_ui | 1 | 0.8076 | OpenCV `1.24x` |
| STACK_BLUR | P1 新增 | public_header_baseline | 1 | 0.0984 | OpenCV `10.17x` |
| THRESHOLD | 既有 | header_fastpath, headers_baseline | 5 | 0.9754 | OpenCV `1.03x` |
| THRESHOLD_WITH_MASK | P1 新增 | public_header_baseline | 1 | 0.9336 | OpenCV `1.07x` |
| WARP_AFFINE | 既有 | headers_baseline | 9 | 0.1575 | OpenCV `6.35x` |
| WARP_PERSPECTIVE | P1 新增 | public_header_scalar | 4 | 0.3444 | OpenCV `2.90x` |

## 详细结果

### `core_mat`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.035855 | 0.027258 | 0.7602 | phase1_representative_case |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.272887 | 0.204326 | 0.7488 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043362 | 0.032026 | 0.7386 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042847 | 0.032082 | 0.7488 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.122948 | 0.097324 | 0.7916 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.743440 | 0.609550 | 0.8199 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.116797 | 0.090731 | 0.7768 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.113989 | 0.091602 | 0.8036 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.333234 | 0.269315 | 0.8082 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.061712 | 0.049827 | 0.8074 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009415 | 0.004893 | 0.5196 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009412 | 0.007022 | 0.7461 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.029498 | 0.023790 | 0.8065 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.191113 | 0.149488 | 0.7822 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.029015 | 0.022123 | 0.7625 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027311 | 0.021423 | 0.7844 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.086253 | 0.065886 | 0.7639 | correctness=upstream_pass |
| BITWISE_AND | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.038489 | 0.027056 | 0.7030 | phase1_representative_case |
| BITWISE_NOT | P1 新增 | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.037828 | 0.017795 | 0.4704 | phase1_representative_case |
| BITWISE_OR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.038811 | 0.027527 | 0.7092 | phase1_representative_case |
| BITWISE_XOR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036356 | 0.025697 | 0.7068 | phase1_representative_case |
| BORDER_INTERPOLATE | P1 新增 | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.005391 | 0.005233 | 0.9706 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| BROADCAST | P1 新增 | row_to_image_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004528 | 0.003540 | 0.7817 | phase1_representative_case |
| CHECK_RANGE | P1 新增 | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.153048 | 0.092803 | 0.6064 | phase1_representative_case |
| CONVERT_FP16 | P1 新增 | f32c1_to_fp16 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.016317 | 0.019903 | 1.2198 | phase1_representative_case |
| CONVERT_SCALE_ABS | P1 新增 | f32c3_to_u8c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.082386 | 0.067366 | 0.8177 | phase1_representative_case |
| COPY_TO | P1 新增 | masked_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.022429 | 0.022230 | 0.9911 | phase1_representative_case |
| COUNT_NON_ZERO | P1 新增 | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.008906 | 0.008863 | 0.9951 | phase1_representative_case |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.253682 | 0.214904 | 0.8471 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043392 | 0.031793 | 0.7327 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042808 | 0.032579 | 0.7610 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.114409 | 0.090865 | 0.7942 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.752416 | 0.662448 | 0.8804 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.114558 | 0.090244 | 0.7878 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.112182 | 0.090765 | 0.8091 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.331697 | 0.276908 | 0.8348 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 1080x1920 | 1.767860 | 0.454995 | 0.2574 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 479x641 | 0.264792 | 0.067341 | 0.2543 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 480x640 | 0.264437 | 0.067019 | 0.2534 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 720x1280 | 0.785392 | 0.201271 | 0.2563 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 3.723175 | 1.364743 | 0.3666 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.569285 | 0.204583 | 0.3594 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.531567 | 0.200971 | 0.3781 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 1.590010 | 0.606184 | 0.3812 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | P1 新增 | bounded_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.157918 | 0.075501 | 0.4781 | phase1_representative_case |
| EXTRACT_CHANNEL | P1 新增 | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.015267 | 0.042170 | 2.7621 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | all_zero_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.010797 | 0.073244 | 6.7840 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | random_dense_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.505757 | 0.191388 | 0.3784 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | sparse_tail_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.010545 | 0.073528 | 6.9725 | phase1_representative_case |
| FLIP | P1 新增 | horizontal_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.016407 | 0.016371 | 0.9978 | phase1_representative_case |
| FLIP_ND | P1 新增 | axis1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.016413 | 0.127506 | 7.7684 | phase1_representative_case |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.083161 | 0.003635 | 0.0437 | correctness=upstream_pass;iters=8 |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.213709 | 0.022875 | 0.1070 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 1.321375 | 0.178916 | 0.1354 | correctness=upstream_pass;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.083505 | 0.003620 | 0.0433 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=8 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.222708 | 0.022959 | 0.1031 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 1.348000 | 0.170208 | 0.1263 | correctness=upstream_pass;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | P1 新增 | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.000009 | 0.000008 | 0.9520 | phase1_representative_case |
| HCONCAT | P1 新增 | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.005030 | 0.006922 | 1.3762 | phase1_representative_case |
| INSERT_CHANNEL | P1 新增 | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.025145 | 0.042919 | 1.7068 | phase1_representative_case |
| IN_RANGE | P1 新增 | scalar_bounds_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.197542 | 0.121997 | 0.6176 | phase1_representative_case |
| LOG | P1 新增 | positive_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.218912 | 0.126099 | 0.5760 | phase1_representative_case |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024691 | 0.024186 | 0.9795 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003723 | 0.003653 | 0.9812 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.007856 | 0.006988 | 0.8895 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.011004 | 0.011193 | 1.0172 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.077985 | 0.078788 | 1.0103 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.012328 | 0.013190 | 1.0700 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.016406 | 0.015330 | 0.9344 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.034127 | 0.034392 | 1.0078 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024600 | 0.023632 | 0.9607 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003344 | 0.003353 | 1.0027 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.006522 | 0.006555 | 1.0050 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.010776 | 0.010518 | 0.9760 |  |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000014 | 0.000001 | 0.0807 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000015 | 0.000001 | 0.0769 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000044 | 0.000003 | 0.0593 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000016 | 0.000001 | 0.0765 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000041 | 0.000015 | 0.3730 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000043 | 0.000015 | 0.3540 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000059 | 0.000018 | 0.3011 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000045 | 0.000015 | 0.3391 | micro_iters_x1000 |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.025261 | 0.023828 | 0.9433 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004193 | 0.003791 | 0.9043 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.006868 | 0.006599 | 0.9608 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.011373 | 0.010888 | 0.9574 |  |
| MAX | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.038686 | 0.024502 | 0.6333 | phase1_representative_case |
| MEAN | P1 新增 | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.097292 | 0.194363 | 1.9977 | phase1_representative_case |
| MEAN_STD_DEV | P1 新增 | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.412755 | 0.132653 | 0.3214 | phase1_representative_case |
| MIN | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.038610 | 0.024859 | 0.6438 | phase1_representative_case |
| MIN_MAX_IDX | P1 新增 | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.044768 | 0.031948 | 0.7136 | phase1_representative_case |
| MIN_MAX_LOC | P1 新增 | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.044910 | 0.031681 | 0.7054 | phase1_representative_case |
| MIX_CHANNELS | P1 新增 | reverse_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.035953 | 0.130340 | 3.6253 | phase1_representative_case |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.254305 | 0.202716 | 0.7971 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.042947 | 0.031569 | 0.7351 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042870 | 0.031743 | 0.7405 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.114309 | 0.091126 | 0.7972 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.750265 | 0.628616 | 0.8379 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.114312 | 0.089530 | 0.7832 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.112412 | 0.090487 | 0.8050 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.331674 | 0.271584 | 0.8188 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.061337 | 0.049830 | 0.8124 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009545 | 0.007048 | 0.7384 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009745 | 0.006938 | 0.7120 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.029548 | 0.023509 | 0.7956 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.192291 | 0.149746 | 0.7787 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.027744 | 0.022075 | 0.7957 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027532 | 0.022175 | 0.8054 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.087487 | 0.066496 | 0.7601 | correctness=upstream_pass |
| NORM | P1 新增 | inf_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.092333 | 0.019926 | 0.2158 | phase1_representative_case |
| NORM | P1 新增 | inf_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.087587 | 0.010065 | 0.1149 | phase1_representative_case |
| NORM | P1 新增 | l1_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.094951 | 0.027542 | 0.2901 | phase1_representative_case |
| NORM | P1 新增 | l1_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.087580 | 0.023049 | 0.2632 | phase1_representative_case |
| NORM | P1 新增 | l2_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.095771 | 0.035426 | 0.3699 | phase1_representative_case |
| NORM | P1 新增 | l2_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.078451 | 0.035944 | 0.4582 | phase1_representative_case |
| NORMALIZE | P1 新增 | inf_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.129340 | 0.031286 | 0.2419 | phase1_representative_case |
| NORMALIZE | P1 新增 | l1_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.129282 | 0.043424 | 0.3359 | phase1_representative_case |
| NORMALIZE | P1 新增 | l2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.113349 | 0.054038 | 0.4767 | phase1_representative_case |
| NORMALIZE | P1 新增 | minmax_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.080571 | 0.055711 | 0.6915 | phase1_representative_case |
| PATCH_NANS | P1 新增 | one_nan_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.019336 | 0.018419 | 0.9526 | phase1_representative_case |
| POW | P1 新增 | power_1_75_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.504910 | 0.295720 | 0.5857 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.065409 | 0.014117 | 0.2158 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.037597 | 0.017174 | 0.4568 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.037389 | 0.017073 | 0.4566 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.064597 | 0.017389 | 0.2692 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.065760 | 0.013670 | 0.2079 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.051094 | 0.009952 | 0.1948 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.127485 | 0.148436 | 1.1643 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.127204 | 0.149376 | 1.1743 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.066918 | 0.242868 | 3.6293 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.050653 | 0.010103 | 0.1994 | phase1_representative_case |
| REDUCE_ARG_MAX | P1 新增 | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.160054 | 0.154495 | 0.9653 | phase1_representative_case |
| REDUCE_ARG_MIN | P1 新增 | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.150324 | 0.143366 | 0.9537 | phase1_representative_case |
| REPEAT | P1 新增 | two_by_two_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004966 | 0.005166 | 1.0403 | phase1_representative_case |
| ROTATE | P1 新增 | clockwise90_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.142425 | 0.106149 | 0.7453 | phase1_representative_case |
| SCALE_ADD | P1 新增 | f32c3 | scalar | CV_32F | 3 | continuous | 480x640 | 0.120117 | 0.098893 | 0.8233 | phase1_representative_case |
| SQRT | P1 新增 | positive_f32c1 | scalar | CV_32F | 1 | continuous | 480x640 | 0.041017 | 0.040722 | 0.9928 | phase1_representative_case |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.253928 | 0.202143 | 0.7961 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.043093 | 0.032009 | 0.7428 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042761 | 0.032034 | 0.7491 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.117052 | 0.091225 | 0.7794 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.746063 | 0.640395 | 0.8584 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.113588 | 0.088500 | 0.7791 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.112483 | 0.090240 | 0.8023 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.331299 | 0.269668 | 0.8140 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.061473 | 0.049698 | 0.8084 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009405 | 0.007076 | 0.7524 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009466 | 0.007030 | 0.7426 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.031851 | 0.023778 | 0.7465 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.191262 | 0.148185 | 0.7748 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.027582 | 0.021929 | 0.7950 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027342 | 0.022062 | 0.8069 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.087332 | 0.066391 | 0.7602 | correctness=upstream_pass |
| SUM | P1 新增 | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.097240 | 0.194414 | 1.9993 | phase1_representative_case |
| SWAP | P1 新增 | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000005 | 0.000005 | 1.0000 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.587035 | 0.539773 | 0.9195 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.034822 | 0.034215 | 0.9826 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.076709 | 0.076496 | 0.9972 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.287098 | 0.285175 | 0.9933 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 1080x1920 | 0.658357 | 1.404071 | 2.1327 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 479x641 | 0.123402 | 0.127931 | 1.0367 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 480x640 | 0.122564 | 0.162768 | 1.3280 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 720x1280 | 0.281140 | 0.555900 | 1.9773 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.094948 | 0.118547 | 1.2485 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.010243 | 0.009869 | 0.9635 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.006753 | 0.006774 | 1.0031 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.053249 | 0.026802 | 0.5033 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 0.392958 | 0.755913 | 1.9236 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.112919 | 0.088036 | 0.7796 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.117037 | 0.089878 | 0.7679 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 0.216266 | 0.394157 | 1.8226 | correctness=upstream_pass |
| VCONCAT | P1 新增 | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.003675 | 0.003816 | 1.0383 | phase1_representative_case |

### `imgproc`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.059493 | 0.018544 | 0.3117 | phase1_representative_case |
| ACCUMULATE_PRODUCT | P1 新增 | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.077666 | 0.020174 | 0.2598 | phase1_representative_case |
| ACCUMULATE_SQUARE | P1 新增 | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.058417 | 0.018130 | 0.3104 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | P1 新增 | alpha0_1_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.058878 | 0.020243 | 0.3438 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | P1 新增 | mean11_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.287030 | 0.222398 | 0.7748 | phase1_representative_case |
| APPLY_COLOR_MAP | P1 新增 | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.658571 | 0.229320 | 0.3482 | phase1_representative_case |
| BILATERAL_FILTER | P1 新增 | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 17.891886 | 0.934158 | 0.0522 | phase1_representative_case |
| BLEND_LINEAR | P1 新增 | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.519843 | 0.221467 | 0.4260 | phase1_representative_case |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.485114 | 0.285010 | 0.1919 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.283218 | 0.046357 | 0.1637 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.289234 | 0.046120 | 0.1595 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.712530 | 0.129226 | 0.1814 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.182288 | 0.105086 | 0.5765 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.319472 | 0.301066 | 0.9424 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.387389 | 0.404495 | 1.0442 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.674183 | 0.128320 | 0.1903 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 0.656598 | 0.128486 | 0.1957 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 0.864891 | 0.170445 | 0.1971 |  |
| BUILD_PYRAMID | P1 新增 | levels3_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.261025 | 0.019733 | 0.0756 | phase1_representative_case |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 28.800815 | 27.760340 | 0.9639 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 4.248430 | 3.864982 | 0.9097 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 4.269565 | 3.868716 | 0.9061 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 12.731111 | 12.208492 | 0.9589 |  |
| CONVERT_MAPS | P1 新增 | f32_pair_to_fixed | opencv_ui | CV_32F | 2 | continuous | 480x640 | 0.394950 | 0.063009 | 0.1595 | phase1_representative_case |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.069052 | 0.044762 | 0.6482 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.059997 | 0.006978 | 0.1163 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.058788 | 0.007010 | 0.1192 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.064694 | 0.020168 | 0.3117 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.067879 | 0.026485 | 0.3902 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.076955 | 0.083882 | 1.0900 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.083008 | 0.099765 | 1.2019 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.063941 | 0.020277 | 0.3171 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.066192 | 0.026455 | 0.3997 |  |
| CREATE_HANNING_WINDOW | P1 新增 | 64x64_f32 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.000644 | 0.001195 | 1.8557 | phase1_representative_case |
| CVTCOLOR | 既有 | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.109206 | 0.020036 | 0.1835 |  |
| CVTCOLOR | 既有 | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.079633 | 0.045100 | 0.5663 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.205732 | 0.204214 | 0.9926 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.030540 | 0.030397 | 0.9953 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.033737 | 0.033740 | 1.0001 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.091396 | 0.091121 | 0.9970 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.035261 | 0.035195 | 0.9981 |  |
| CVTCOLOR | 既有 | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.126837 | 0.061641 | 0.4860 |  |
| CVTCOLOR | 既有 | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.091346 | 0.065964 | 0.7221 |  |
| CVTCOLOR | 既有 | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.089565 | 0.016780 | 0.1874 |  |
| CVTCOLOR | 既有 | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.121196 | 0.085977 | 0.7094 |  |
| CVTCOLOR | 既有 | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.126651 | 0.063431 | 0.5008 |  |
| CVTCOLOR | 既有 | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.094420 | 0.041695 | 0.4416 |  |
| CVTCOLOR | 既有 | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 0.233233 | 0.073156 | 0.3137 |  |
| CVTCOLOR | 既有 | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.140262 | 0.073824 | 0.5263 |  |
| CVTCOLOR | 既有 | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.117641 | 0.065138 | 0.5537 |  |
| CVTCOLOR | 既有 | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.135929 | 0.072715 | 0.5349 |  |
| CVT_COLOR_TWO_PLANE | P1 新增 | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.326700 | 0.065890 | 0.2017 | phase1_representative_case |
| DEMOSAICING | P1 新增 | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.449826 | 0.043370 | 0.0964 | phase1_representative_case |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.179570 | 0.147540 | 0.8216 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.124685 | 0.021525 | 0.1726 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.124994 | 0.021036 | 0.1683 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.144196 | 0.063646 | 0.4414 |  |
| DILATE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.530661 | 0.065615 | 0.1236 |  |
| DILATE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.636955 | 0.089800 | 0.1410 |  |
| EQUALIZE_HIST | P1 新增 | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.082852 | 0.086550 | 1.0446 | phase1_representative_case |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.180160 | 0.142140 | 0.7890 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.124900 | 0.020734 | 0.1660 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.127605 | 0.020812 | 0.1631 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.151042 | 0.065085 | 0.4309 |  |
| ERODE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.523515 | 0.065685 | 0.1255 |  |
| ERODE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.642063 | 0.089703 | 0.1397 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.797298 | 0.634288 | 0.3529 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.341287 | 0.105048 | 0.3078 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.322773 | 0.096120 | 0.2978 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.827983 | 0.288991 | 0.3490 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.374379 | 0.074282 | 0.1984 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.440218 | 0.200931 | 0.4564 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.435117 | 0.268514 | 0.6171 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.509580 | 0.298012 | 0.5848 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.524527 | 0.326416 | 0.6223 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.543192 | 0.382840 | 0.7048 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.369823 | 0.235075 | 0.1716 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.346238 | 0.033333 | 0.0963 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.342030 | 0.032172 | 0.0941 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.703577 | 0.099251 | 0.1411 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 0.436786 | 0.119837 | 0.2744 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 0.432962 | 0.336403 | 0.7770 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.389323 | 0.443909 | 1.1402 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 0.506507 | 0.109213 | 0.2156 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 0.495577 | 0.359588 | 0.7256 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 0.444000 | 0.146122 | 0.3291 |  |
| GET_AFFINE_TRANSFORM | P1 新增 | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000140 | 0.000270 | 1.9316 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_DERIV_KERNELS | P1 新增 | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000140 | 0.000093 | 0.6667 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GABOR_KERNEL | P1 新增 | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.001043 | 0.001007 | 0.9652 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GAUSSIAN_KERNEL | P1 新增 | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000266 | 0.001030 | 3.8745 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000282 | 0.000709 | 2.5103 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 0.076875 | 0.978862 | 12.7332 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 0.017125 | 0.144877 | 8.4600 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 0.010809 | 0.144940 | 13.4095 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 0.032674 | 0.433246 | 13.2596 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | P1 新增 | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000081 | 0.000075 | 0.9331 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000004 | 0.000005 | 1.1010 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_STRUCTURING_ELEMENT | P1 新增 | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000107 | 0.000081 | 0.7549 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| INTEGRAL | P1 新增 | u8c1_to_s32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.078041 | 0.043520 | 0.5577 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | P1 新增 | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000045 | 0.000061 | 1.3611 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| LAPLACIAN | P1 新增 | ksize3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.392297 | 0.164468 | 0.4192 | phase1_representative_case |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.116675 | 0.193182 | 1.6557 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.065136 | 0.028682 | 0.4403 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.065173 | 0.028619 | 0.4391 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.083930 | 0.085824 | 1.0226 |  |
| LUT | 既有 | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.081345 | 0.080524 | 0.9899 |  |
| LUT | 既有 | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.091145 | 0.114397 | 1.2551 |  |
| MEDIAN_BLUR | P1 新增 | ksize5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 8.339531 | 0.251741 | 0.0302 | phase1_representative_case |
| PYR_DOWN | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.495427 | 0.061135 | 0.1234 | phase1_representative_case |
| PYR_UP | P1 新增 | u8c3 | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 1.187725 | 0.168134 | 0.1416 | phase1_representative_case |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 16.906364 | 4.650673 | 0.2751 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 2.549038 | 0.710612 | 0.2788 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 2.518281 | 0.689061 | 0.2736 | no qualified SIMD fast path |
| REMAP | P1 新增 | fixed_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 7.396841 | 2.060846 | 0.2786 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 17.405460 | 4.871064 | 0.2799 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 2.571131 | 0.707037 | 0.2750 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 2.584781 | 0.704912 | 0.2727 | no qualified SIMD fast path |
| REMAP | P1 新增 | float_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 7.696427 | 2.117215 | 0.2751 | no qualified SIMD fast path |
| RESIZE | 既有 | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.204069 | 0.092855 | 0.4550 |  |
| RESIZE | 既有 | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.434336 | 0.267655 | 0.6162 |  |
| RESIZE | 既有 | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.142617 | 0.069398 | 0.4866 |  |
| RESIZE | 既有 | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.146430 | 0.068699 | 0.4692 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.122040 | 0.081335 | 0.6665 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.018474 | 0.012349 | 0.6685 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.020348 | 0.013320 | 0.6546 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.057000 | 0.037601 | 0.6597 |  |
| RESIZE | 既有 | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.086744 | 0.099641 | 1.1487 |  |
| RESIZE | 既有 | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.085720 | 0.103585 | 1.2084 |  |
| SCHARR | P1 新增 | dx1_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.388145 | 0.124288 | 0.3202 | phase1_representative_case |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.983865 | 0.663190 | 0.6741 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.267242 | 0.098897 | 0.3701 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.277635 | 0.098626 | 0.3552 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.510041 | 0.289688 | 0.5680 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.321534 | 0.082818 | 0.2576 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.439789 | 0.217937 | 0.4955 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.429456 | 0.281175 | 0.6547 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.467965 | 0.286097 | 0.6114 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.480582 | 0.323272 | 0.6727 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.424433 | 0.383286 | 0.9031 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.396272 | 0.855694 | 2.1594 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.108397 | 0.121370 | 1.1197 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.110284 | 0.125046 | 1.1339 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.213697 | 0.345413 | 1.6164 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.184477 | 0.334792 | 1.8148 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.238489 | 0.455743 | 1.9110 |  |
| SPATIAL_GRADIENT | P1 新增 | ksize3_u8_to_s16 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.106058 | 0.046267 | 0.4362 | phase1_representative_case |
| SQR_BOX_FILTER | P1 新增 | 3x3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.268983 | 0.217228 | 0.8076 | phase1_representative_case |
| STACK_BLUR | P1 新增 | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 1.245217 | 0.122481 | 0.0984 | phase1_representative_case |
| THRESHOLD | 既有 | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.074447 | 0.068977 | 0.9265 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.033439 | 0.033568 | 1.0038 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004769 | 0.004844 | 1.0158 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.005334 | 0.005071 | 0.9508 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.014570 | 0.014320 | 0.9828 |  |
| THRESHOLD_WITH_MASK | P1 新增 | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.176443 | 0.164734 | 0.9336 | phase1_representative_case |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 12.195618 | 1.949046 | 0.1598 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 479x641 | 1.824500 | 0.321812 | 0.1764 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 480x640 | 1.754134 | 0.289563 | 0.1651 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 5.255021 | 0.867743 | 0.1651 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 2.829979 | 0.495016 | 0.1749 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 7.229236 | 0.653560 | 0.0904 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 9.422207 | 0.710360 | 0.0754 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c3 | headers_baseline | CV_8U | 3 | continuous | 480x640 | 2.177542 | 0.733157 | 0.3367 |  |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c4 | headers_baseline | CV_8U | 4 | continuous | 480x640 | 2.502986 | 0.484169 | 0.1934 |  |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 19.119465 | 6.445113 | 0.3371 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 2.753244 | 0.967045 | 0.3512 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 2.742277 | 0.952276 | 0.3473 | no qualified SIMD fast path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 8.351536 | 2.859247 | 0.3424 | no qualified SIMD fast path |

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
