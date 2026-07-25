# cvh vs OpenCV Benchmark Report (full)

生成时间（UTC）：`2026-07-25 06:25:37Z`

## 当前项目状态

- `opencv-header-only` 当前公共定位是纯 header-only，不依赖项目内 `.cpp` 扩展层。
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
- P-ACC-2 至 P-ACC-8 已完成 Apple ARM 收尾，覆盖 Core 归约、布局、通道与 GEMM，以及 Imgproc 滤波、几何、非线性、形态学、累积和强度变换；真实 x86 SSE/AVX 运行验证仍是外部 gate。
- P-ACC-8 新增 pyramid ring workspace、非线性滤波专用算法、几何 fixed-coordinate block、S16 derivative UI、sqrBoxFilter wide sliding sum 与 F32 C1 reduction fast-path。
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
| 专用 kernel | `cvtColor`、特定 `resize`、core 逐元素、统计/非零归约、F32 数学、pyramid 与 derivative UI kernel | 实际命中时记录为 `dispatch_path=opencv_ui` |
| Header fast-path | 行并行 filter、LUT、border、Sobel、Canny、morphology、sliding sum 与 nonlinear 专用 kernel | 记录实际 `header_fastpath` / `sliding_*` / `precomputed_lut` 路径 |
| 几何采样 | 共享定点坐标 block、U8 bilinear sampler 与 interior/border 分流 | 记录为 `dispatch_path=fixed_coordinate_block` |
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
- 原始数据：`2026-07-25-opencv-upstream-performance.csv`；元数据：`2026-07-25-opencv-upstream-performance.csv.meta.json`

## 汇总

- 总 case：`344`；有效：`343`；不支持：`1`。
- `OpenCV/CVH` 几何平均：`0.4550`；中位数：`0.6419`。
- CVH 更快：`44` 个；OpenCV 更快或相当：`299` 个。

| Suite | Cases | 几何平均 OpenCV/CVH | 中位数 | CVH 更快 | OpenCV 更快/相当 |
| --- | --- | --- | --- | --- | --- |
| core_mat | 176 | 0.6135 | 0.7645 | 25 | 151 |
| imgproc | 167 | 0.3320 | 0.3836 | 19 | 148 |

## P-ACC-8 未收敛差距

以下倍率是本次 case 的组内几何平均，只用于定位后续工作，不等同于 API 支持状态。P-ACC-8 已通过相对自身旧路径的接受 gate，但部分算子仍明显落后 upstream。

| 范围 | 本报告 | 主要原因 | 后续边界 |
| --- | --- | --- | --- |
| `GEMM` | OpenCV `~12.68x` | 默认 upstream 可进入 Accelerate/LAPACK；这不是 OpenCV UI 内建 kernel 的纯 SIMD 对比 | 保留现有 header-only micro-kernel，不为追赶外部 BLAS 引入链接依赖 |
| filter / derivative | OpenCV `~8.74x` | CVH 仍有通用 filter 调度、border materialization 和中间行处理；upstream 的类型/核尺寸专用化更深 | 下一批优先做共享 row/column engine 与 U8-to-S16/F32 fused kernel |
| nonlinear | OpenCV `~6.47x` | 已消除重复窗口扫描，但 bilateral 权重累计、median lane network 和大尺寸 cache 行为仍有差距 | 保留已接受算法，后续按绝对耗时继续拆像素内核与内存访问 |
| pyramid | OpenCV `~4.59x` | ring workspace 和 UI 已落地，C3 interleave、边界行与上下采样写回仍未达到 upstream 专用 kernel | 继续复用当前 ring 基础设施，不回退到整图 temporary |
| geometry | OpenCV `~3.54x` | 坐标 block 已共享，但插值、border mask 和多通道 gather/store 仍包含较多标量工作 | 后续只扩 U8 C1/C3/C4 interior SIMD，不复制三套公共内核 |
| reduction | OpenCV `~2.27x` | 本轮 fast-path 主要覆盖 F32 C1；Mode B 仍包含多通道、双输入和高精度合同路径 | 按 variant 拆 gate，不能用降低精度换取汇总倍率 |

## 算子级概览

### `core_mat`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | opencv_ui | 1 | 0.7409 | OpenCV `1.35x` |
| ADD | 既有 | opencv_ui | 16 | 0.7789 | OpenCV `1.28x` |
| BITWISE_AND | P1 新增 | opencv_ui | 1 | 0.6831 | OpenCV `1.46x` |
| BITWISE_NOT | P1 新增 | opencv_ui | 1 | 0.4731 | OpenCV `2.11x` |
| BITWISE_OR | P1 新增 | opencv_ui | 1 | 0.7117 | OpenCV `1.41x` |
| BITWISE_XOR | P1 新增 | opencv_ui | 1 | 0.7090 | OpenCV `1.41x` |
| BORDER_INTERPOLATE | P1 新增 | public_header_baseline | 1 | 0.9695 | OpenCV `1.03x` |
| BROADCAST | P1 新增 | scalar | 1 | 0.7665 | OpenCV `1.30x` |
| CHECK_RANGE | P1 新增 | public_header_baseline | 1 | 0.6419 | OpenCV `1.56x` |
| CONVERT_FP16 | P1 新增 | opencv_ui | 1 | 1.2341 | CVH `1.23x` |
| CONVERT_SCALE_ABS | P1 新增 | opencv_ui | 1 | 0.8199 | OpenCV `1.22x` |
| COPY_TO | P1 新增 | opencv_ui | 1 | 1.0225 | CVH `1.02x` |
| COUNT_NON_ZERO | P1 新增 | opencv_ui | 1 | 0.9980 | OpenCV `1.00x` |
| DIVIDE | 既有 | opencv_ui, scalar | 16 | 0.5011 | OpenCV `2.00x` |
| EXP | P1 新增 | opencv_ui | 1 | 0.4783 | OpenCV `2.09x` |
| EXTRACT_CHANNEL | P1 新增 | opencv_ui | 1 | 2.7864 | CVH `2.79x` |
| FIND_NON_ZERO | P1 新增 | opencv_ui | 3 | 2.6454 | CVH `2.65x` |
| FLIP | P1 新增 | opencv_ui | 1 | 0.9937 | OpenCV `1.01x` |
| FLIP_ND | P1 新增 | opencv_ui | 1 | 7.7984 | CVH `7.80x` |
| GEMM | 既有 | opencv_ui | 10 | 0.0789 | OpenCV `12.68x` |
| HAS_NON_ZERO | P1 新增 | opencv_ui | 1 | 1.0447 | CVH `1.04x` |
| HCONCAT | P1 新增 | scalar | 1 | 1.3372 | CVH `1.34x` |
| INSERT_CHANNEL | P1 新增 | opencv_ui | 1 | 1.7263 | CVH `1.73x` |
| IN_RANGE | P1 新增 | opencv_ui | 1 | 0.6150 | OpenCV `1.63x` |
| LOG | P1 新增 | opencv_ui | 1 | 0.5792 | OpenCV `1.73x` |
| MAT_CLONE | 既有 | headers_baseline | 4 | 0.9848 | OpenCV `1.02x` |
| MAT_CONVERTTO | 既有 | headers_baseline | 4 | 0.9404 | OpenCV `1.06x` |
| MAT_COPYTO | 既有 | headers_baseline | 4 | 0.9752 | OpenCV `1.03x` |
| MAT_CREATE | 既有 | headers_baseline | 4 | 0.0729 | OpenCV `13.71x` |
| MAT_RESHAPE | 既有 | headers_baseline | 4 | 0.3503 | OpenCV `2.85x` |
| MAT_SETTO | 既有 | headers_baseline | 4 | 0.9506 | OpenCV `1.05x` |
| MAX | P1 新增 | opencv_ui | 1 | 0.7201 | OpenCV `1.39x` |
| MEAN | P1 新增 | opencv_ui | 1 | 2.0851 | CVH `2.09x` |
| MEAN_STD_DEV | P1 新增 | opencv_ui | 1 | 0.3208 | OpenCV `3.12x` |
| MIN | P1 新增 | opencv_ui | 1 | 0.6799 | OpenCV `1.47x` |
| MIN_MAX_IDX | P1 新增 | opencv_ui | 1 | 0.7117 | OpenCV `1.41x` |
| MIN_MAX_LOC | P1 新增 | opencv_ui | 1 | 0.7132 | OpenCV `1.40x` |
| MIX_CHANNELS | P1 新增 | opencv_ui | 1 | 3.6687 | CVH `3.67x` |
| MULTIPLY | 既有 | opencv_ui | 16 | 0.7847 | OpenCV `1.27x` |
| NORM | P1 新增 | opencv_ui | 6 | 0.3150 | OpenCV `3.17x` |
| NORMALIZE | P1 新增 | opencv_ui | 4 | 0.4611 | OpenCV `2.17x` |
| PATCH_NANS | P1 新增 | opencv_ui | 1 | 0.8118 | OpenCV `1.23x` |
| POW | P1 新增 | opencv_ui | 1 | 0.6400 | OpenCV `1.56x` |
| REDUCE | P1 新增 | opencv_ui | 10 | 0.5559 | OpenCV `1.80x` |
| REDUCE_ARG_MAX | P1 新增 | opencv_ui | 1 | 1.0355 | CVH `1.04x` |
| REDUCE_ARG_MIN | P1 新增 | opencv_ui | 1 | 0.9241 | OpenCV `1.08x` |
| REPEAT | P1 新增 | scalar | 1 | 1.0207 | CVH `1.02x` |
| ROTATE | P1 新增 | opencv_ui | 1 | 0.3937 | OpenCV `2.54x` |
| SCALE_ADD | P1 新增 | scalar | 1 | 0.8115 | OpenCV `1.23x` |
| SQRT | P1 新增 | scalar | 1 | 0.9762 | OpenCV `1.02x` |
| SUBTRACT | 既有 | opencv_ui | 16 | 0.7444 | OpenCV `1.34x` |
| SUM | P1 新增 | opencv_ui | 1 | 2.0877 | CVH `2.09x` |
| SWAP | P1 新增 | public_header_baseline | 1 | 1.0000 | OpenCV `1.00x` |
| TRANSPOSE | 既有 | opencv_ui, scalar | 16 | 0.6668 | OpenCV `1.50x` |
| VCONCAT | P1 新增 | scalar | 1 | 1.0442 | CVH `1.04x` |

### `imgproc`

| Op | 阶段 | CVH dispatch | Cases | 几何平均 OpenCV/CVH | 领先方 |
| --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | opencv_ui | 1 | 0.4563 | OpenCV `2.19x` |
| ACCUMULATE_PRODUCT | P1 新增 | opencv_ui | 1 | 0.3385 | OpenCV `2.95x` |
| ACCUMULATE_SQUARE | P1 新增 | opencv_ui | 1 | 0.2992 | OpenCV `3.34x` |
| ACCUMULATE_WEIGHTED | P1 新增 | opencv_ui | 1 | 0.3507 | OpenCV `2.85x` |
| ADAPTIVE_THRESHOLD | P1 新增 | opencv_ui | 1 | 0.6378 | OpenCV `1.57x` |
| APPLY_COLOR_MAP | P1 新增 | public_header_baseline | 1 | 0.3199 | OpenCV `3.13x` |
| BILATERAL_FILTER | P1 新增 | public_header_baseline | 1 | 0.0847 | OpenCV `11.80x` |
| BLEND_LINEAR | P1 新增 | public_header_baseline | 1 | 0.3902 | OpenCV `2.56x` |
| BOX_FILTER | 既有 | box3x3, header_fastpath | 10 | 0.1972 | OpenCV `5.07x` |
| BUILD_PYRAMID | P1 新增 | opencv_ui | 1 | 0.2347 | OpenCV `4.26x` |
| CANNY | 既有 | header_fastpath | 4 | 0.5204 | OpenCV `1.92x` |
| CONVERT_MAPS | P1 新增 | opencv_ui | 1 | 0.5866 | OpenCV `1.70x` |
| COPY_MAKE_BORDER | 既有 | header_fastpath | 9 | 0.9602 | OpenCV `1.04x` |
| CREATE_HANNING_WINDOW | P1 新增 | opencv_ui | 1 | 1.9091 | CVH `1.91x` |
| CVTCOLOR | 既有 | header_fastpath, opencv_ui | 17 | 0.3017 | OpenCV `3.31x` |
| CVT_COLOR_TWO_PLANE | P1 新增 | public_header_baseline | 1 | 0.2083 | OpenCV `4.80x` |
| DEMOSAICING | P1 新增 | public_header_baseline | 1 | 0.0852 | OpenCV `11.73x` |
| DILATE | 既有 | header_fastpath | 6 | 0.2356 | OpenCV `4.24x` |
| EQUALIZE_HIST | P1 新增 | opencv_ui | 1 | 1.0545 | CVH `1.05x` |
| ERODE | 既有 | header_fastpath | 6 | 0.2342 | OpenCV `4.27x` |
| FILTER2D | 既有 | header_fastpath | 10 | 0.0743 | OpenCV `13.46x` |
| GAUSSIAN | 既有 | gauss_separable, header_fastpath | 10 | 0.0794 | OpenCV `12.60x` |
| GET_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.0172 | CVH `2.02x` |
| GET_DERIV_KERNELS | P1 新增 | public_header_baseline | 1 | 0.6685 | OpenCV `1.50x` |
| GET_GABOR_KERNEL | P1 新增 | public_header_baseline | 1 | 0.9903 | OpenCV `1.01x` |
| GET_GAUSSIAN_KERNEL | P1 新增 | public_header_baseline | 1 | 4.0158 | CVH `4.02x` |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 2.3116 | CVH `2.31x` |
| GET_RECT_SUB_PIX | P1 新增 | public_header_scalar | 4 | 11.5659 | CVH `11.57x` |
| GET_ROTATION_MATRIX_2D | P1 新增 | public_header_baseline | 1 | 1.0423 | CVH `1.04x` |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | public_header_baseline | 1 | 1.0917 | CVH `1.09x` |
| GET_STRUCTURING_ELEMENT | P1 新增 | public_header_baseline | 1 | 0.7527 | OpenCV `1.33x` |
| INTEGRAL | P1 新增 | opencv_ui | 1 | 0.5580 | OpenCV `1.79x` |
| INVERT_AFFINE_TRANSFORM | P1 新增 | public_header_baseline | 1 | 1.4907 | CVH `1.49x` |
| LAPLACIAN | P1 新增 | opencv_ui | 1 | 0.0653 | OpenCV `15.31x` |
| LUT | 既有 | header_fastpath | 6 | 0.7749 | OpenCV `1.29x` |
| MEDIAN_BLUR | P1 新增 | opencv_ui | 1 | 0.1864 | OpenCV `5.37x` |
| PYR_DOWN | P1 新增 | opencv_ui | 1 | 0.1828 | OpenCV `5.47x` |
| PYR_UP | P1 新增 | opencv_ui | 1 | 0.2411 | OpenCV `4.15x` |
| REMAP | P1 新增 | fixed_coordinate_block | 8 | 0.3598 | OpenCV `2.78x` |
| RESIZE | 既有 | header_fastpath, headers_baseline, opencv_ui | 10 | 0.5009 | OpenCV `2.00x` |
| SCHARR | P1 新增 | opencv_ui | 1 | 0.0646 | OpenCV `15.49x` |
| SEP_FILTER2D | 既有 | header_fastpath | 10 | 0.1583 | OpenCV `6.32x` |
| SOBEL | 既有 | header_fastpath | 6 | 0.4577 | OpenCV `2.18x` |
| SPATIAL_GRADIENT | P1 新增 | opencv_ui | 1 | 0.1727 | OpenCV `5.79x` |
| SQR_BOX_FILTER | P1 新增 | opencv_ui | 1 | 0.4072 | OpenCV `2.46x` |
| STACK_BLUR | P1 新增 | public_header_baseline | 1 | 0.2334 | OpenCV `4.29x` |
| THRESHOLD | 既有 | header_fastpath, headers_baseline | 5 | 0.9730 | OpenCV `1.03x` |
| THRESHOLD_WITH_MASK | P1 新增 | public_header_baseline | 1 | 0.9668 | OpenCV `1.03x` |
| WARP_AFFINE | 既有 | fixed_coordinate_block, headers_baseline | 9 | 0.1702 | OpenCV `5.88x` |
| WARP_PERSPECTIVE | P1 新增 | fixed_coordinate_block | 4 | 0.4562 | OpenCV `2.19x` |

## 详细结果

### `core_mat`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABSDIFF | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.033536 | 0.024846 | 0.7409 | phase1_representative_case |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.255903 | 0.202317 | 0.7906 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.041010 | 0.030344 | 0.7399 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.040127 | 0.029881 | 0.7447 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.115734 | 0.091580 | 0.7913 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.753865 | 0.619772 | 0.8221 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.114374 | 0.089479 | 0.7823 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.113286 | 0.091337 | 0.8062 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.334471 | 0.272102 | 0.8135 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.062172 | 0.050074 | 0.8054 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009504 | 0.007203 | 0.7579 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009419 | 0.007076 | 0.7513 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.029575 | 0.022111 | 0.7476 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.191403 | 0.150957 | 0.7887 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.031297 | 0.023798 | 0.7604 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027372 | 0.022021 | 0.8045 | correctness=upstream_pass |
| ADD | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.087225 | 0.066570 | 0.7632 | correctness=upstream_pass |
| BITWISE_AND | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036532 | 0.024955 | 0.6831 | phase1_representative_case |
| BITWISE_NOT | P1 新增 | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.035753 | 0.016915 | 0.4731 | phase1_representative_case |
| BITWISE_OR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036501 | 0.025978 | 0.7117 | phase1_representative_case |
| BITWISE_XOR | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.036403 | 0.025808 | 0.7090 | phase1_representative_case |
| BORDER_INTERPOLATE | P1 新增 | reflect101_batch4096 | public_header_baseline | S32 | 1 | continuous | micro_batch | 0.006027 | 0.005843 | 0.9695 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| BROADCAST | P1 新增 | row_to_image_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004676 | 0.003584 | 0.7665 | phase1_representative_case |
| CHECK_RANGE | P1 新增 | quiet_f32c1 | public_header_baseline | CV_32F | 1 | continuous | 480x640 | 0.171622 | 0.110168 | 0.6419 | phase1_representative_case |
| CONVERT_FP16 | P1 新增 | f32c1_to_fp16 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.017795 | 0.021961 | 1.2341 | phase1_representative_case |
| CONVERT_SCALE_ABS | P1 新增 | f32c3_to_u8c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.090825 | 0.074464 | 0.8199 | phase1_representative_case |
| COPY_TO | P1 新增 | masked_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.025337 | 0.025907 | 1.0225 | phase1_representative_case |
| COUNT_NON_ZERO | P1 新增 | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009948 | 0.009928 | 0.9980 | phase1_representative_case |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.257370 | 0.218940 | 0.8507 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.040630 | 0.029594 | 0.7284 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.039676 | 0.029964 | 0.7552 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.115197 | 0.091902 | 0.7978 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.783931 | 0.669738 | 0.8543 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.122748 | 0.095175 | 0.7754 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.113503 | 0.092541 | 0.8153 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.336502 | 0.293826 | 0.8732 | correctness=upstream_pass |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 1080x1920 | 1.782365 | 0.456668 | 0.2562 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 479x641 | 0.275403 | 0.072075 | 0.2617 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 480x640 | 0.264563 | 0.067218 | 0.2541 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 1 | continuous | 720x1280 | 0.791548 | 0.202222 | 0.2555 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 3.601081 | 1.369092 | 0.3802 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.578723 | 0.216555 | 0.3742 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.533668 | 0.202821 | 0.3801 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| DIVIDE | 既有 | mat_mat_continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 1.599616 | 0.609453 | 0.3810 | correctness=upstream_pass;u8_divide_abs_tolerance=1 |
| EXP | P1 新增 | bounded_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.178495 | 0.085366 | 0.4783 | phase1_representative_case |
| EXTRACT_CHANNEL | P1 新增 | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.017762 | 0.049492 | 2.7864 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | all_zero_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.011795 | 0.081983 | 6.9509 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | random_dense_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.558476 | 0.213415 | 0.3821 | phase1_representative_case |
| FIND_NON_ZERO | P1 新增 | sparse_tail_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.011785 | 0.082142 | 6.9700 | phase1_representative_case |
| FLIP | P1 新增 | horizontal_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.019229 | 0.019107 | 0.9937 | phase1_representative_case |
| FLIP_ND | P1 新增 | axis1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.019085 | 0.148833 | 7.7984 | phase1_representative_case |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.091630 | 0.003266 | 0.0356 | correctness=upstream_pass;shape=square_128;component=public_end_to_end;iters=100 |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.850262 | 0.020437 | 0.0240 | correctness=upstream_pass;shape=square_256;component=public_end_to_end;iters=100 |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 256x32x256 | 0.071660 | 0.005169 | 0.0721 | correctness=upstream_pass;shape=wide_m256_k32_n256;component=public_end_to_end;iters=100 |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 32x512x64 | 0.055151 | 0.108692 | 1.9708 | correctness=upstream_pass;shape=skinny_m32_k512_n64;component=public_end_to_end;iters=100 |
| GEMM | 既有 | fp32_nn_end_to_end | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 7.132458 | 0.178250 | 0.0250 | correctness=upstream_pass;shape=square_512;component=public_end_to_end;iters=1 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 128x128x128 | 0.090607 | 0.003336 | 0.0368 | correctness=upstream_pass;shape=square_128;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=100 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x256x256 | 0.843568 | 0.020292 | 0.0241 | correctness=upstream_pass;shape=square_256;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=100 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 256x32x256 | 0.070815 | 0.005171 | 0.0730 | correctness=upstream_pass;shape=wide_m256_k32_n256;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=100 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 32x512x64 | 0.054913 | 0.110432 | 2.0110 | correctness=upstream_pass;shape=skinny_m32_k512_n64;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=100 |
| GEMM | 既有 | fp32_nn_pack_once | opencv_ui | CV_32F | 1 | continuous | 512x512x512 | 7.133334 | 0.167541 | 0.0235 | correctness=upstream_pass;shape=square_512;component=public_pack_once;opencv_reuses_B_without_public_pack_handle;iters=1 |
| HAS_NON_ZERO | P1 新增 | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.000009 | 0.000010 | 1.0447 | phase1_representative_case |
| HCONCAT | P1 新增 | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.005498 | 0.007352 | 1.3372 | phase1_representative_case |
| INSERT_CHANNEL | P1 新增 | channel1_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.028996 | 0.050056 | 1.7263 | phase1_representative_case |
| IN_RANGE | P1 新增 | scalar_bounds_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.205399 | 0.126323 | 0.6150 | phase1_representative_case |
| LOG | P1 新增 | positive_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.244715 | 0.141738 | 0.5792 | phase1_representative_case |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024326 | 0.024536 | 1.0086 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003845 | 0.003677 | 0.9563 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.007981 | 0.007860 | 0.9849 |  |
| MAT_CLONE | 既有 | full_copy | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.011202 | 0.011092 | 0.9902 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.092411 | 0.078904 | 0.8538 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.011898 | 0.013397 | 1.1260 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.018084 | 0.014230 | 0.7869 |  |
| MAT_CONVERTTO | 既有 | CV_8U_to_CV_32F | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.033266 | 0.034394 | 1.0339 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.024961 | 0.023955 | 0.9597 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.003507 | 0.003405 | 0.9710 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.006577 | 0.006552 | 0.9961 |  |
| MAT_COPYTO | 既有 | continuous_reuse | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.010984 | 0.010702 | 0.9744 |  |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000015 | 0.000001 | 0.0772 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000015 | 0.000001 | 0.0757 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000040 | 0.000003 | 0.0641 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_CREATE | 既有 | reuse_same_shape | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000017 | 0.000001 | 0.0755 | cvh_headers_fast_inherits_cvh_headers;micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.000043 | 0.000015 | 0.3579 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.000044 | 0.000015 | 0.3539 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.000054 | 0.000018 | 0.3290 | micro_iters_x1000 |
| MAT_RESHAPE | 既有 | to_column_view | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.000043 | 0.000016 | 0.3615 | micro_iters_x1000 |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.025412 | 0.024580 | 0.9672 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.004189 | 0.003797 | 0.9064 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.006689 | 0.006603 | 0.9872 |  |
| MAT_SETTO | 既有 | scalar_all | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.011547 | 0.010893 | 0.9433 |  |
| MAX | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.038064 | 0.027411 | 0.7201 | phase1_representative_case |
| MEAN | P1 新增 | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.109377 | 0.228062 | 2.0851 | phase1_representative_case |
| MEAN_STD_DEV | P1 新增 | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.464330 | 0.148964 | 0.3208 | phase1_representative_case |
| MIN | P1 新增 | mat_mat_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.038120 | 0.025917 | 0.6799 | phase1_representative_case |
| MIN_MAX_IDX | P1 新增 | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.050098 | 0.035655 | 0.7117 | phase1_representative_case |
| MIN_MAX_LOC | P1 新增 | f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.050020 | 0.035675 | 0.7132 | phase1_representative_case |
| MIX_CHANNELS | P1 新增 | reverse_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.041897 | 0.153709 | 3.6687 | phase1_representative_case |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.256245 | 0.205856 | 0.8034 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.040647 | 0.029892 | 0.7354 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.040089 | 0.029668 | 0.7401 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.115438 | 0.092404 | 0.8005 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.798334 | 0.646531 | 0.8098 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.114323 | 0.092484 | 0.8090 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.113875 | 0.092078 | 0.8086 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.333975 | 0.278340 | 0.8334 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.061549 | 0.049947 | 0.8115 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009650 | 0.007060 | 0.7316 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009432 | 0.007022 | 0.7445 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.027615 | 0.022116 | 0.8009 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.192330 | 0.150457 | 0.7823 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.029970 | 0.023770 | 0.7931 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.027745 | 0.022070 | 0.7955 | correctness=upstream_pass |
| MULTIPLY | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.088169 | 0.067526 | 0.7659 | correctness=upstream_pass |
| NORM | P1 新增 | inf_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.103993 | 0.022462 | 0.2160 | phase1_representative_case |
| NORM | P1 新增 | inf_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.041405 | 0.011524 | 0.2783 | phase1_representative_case |
| NORM | P1 新增 | l1_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.104688 | 0.030564 | 0.2920 | phase1_representative_case |
| NORM | P1 新增 | l1_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.082906 | 0.025845 | 0.3117 | phase1_representative_case |
| NORM | P1 新增 | l2_diff_zero_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.103869 | 0.039429 | 0.3796 | phase1_representative_case |
| NORM | P1 新增 | l2_single_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.082772 | 0.038963 | 0.4707 | phase1_representative_case |
| NORMALIZE | P1 新增 | inf_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.079783 | 0.031399 | 0.3936 | phase1_representative_case |
| NORMALIZE | P1 新增 | l1_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.119679 | 0.046860 | 0.3916 | phase1_representative_case |
| NORMALIZE | P1 新增 | l2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.139371 | 0.059363 | 0.4259 | phase1_representative_case |
| NORMALIZE | P1 新增 | minmax_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.091171 | 0.062793 | 0.6887 | phase1_representative_case |
| PATCH_NANS | P1 新增 | one_nan_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.021711 | 0.017625 | 0.8118 | phase1_representative_case |
| POW | P1 新增 | power_1_75_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.550021 | 0.352039 | 0.6400 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.072344 | 0.015597 | 0.2156 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042041 | 0.019180 | 0.4562 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.042040 | 0.019078 | 0.4538 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.073085 | 0.019130 | 0.2617 | phase1_representative_case |
| REDUCE | P1 新增 | axis0_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.072226 | 0.015272 | 0.2115 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_avg_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.056160 | 0.011343 | 0.2020 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_max_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.059502 | 0.174768 | 2.9372 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_min_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.059340 | 0.171781 | 2.8948 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_sum2_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.075060 | 0.247768 | 3.3009 | phase1_representative_case |
| REDUCE | P1 新增 | axis1_sum_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.055786 | 0.011219 | 0.2011 | phase1_representative_case |
| REDUCE_ARG_MAX | P1 新增 | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.158706 | 0.164337 | 1.0355 | phase1_representative_case |
| REDUCE_ARG_MIN | P1 新增 | axis0_f32c1 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.157840 | 0.145860 | 0.9241 | phase1_representative_case |
| REPEAT | P1 新增 | two_by_two_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.005300 | 0.005410 | 1.0207 | phase1_representative_case |
| ROTATE | P1 新增 | clockwise90_u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.294584 | 0.115992 | 0.3937 | phase1_representative_case |
| SCALE_ADD | P1 新增 | f32c3 | scalar | CV_32F | 3 | continuous | 480x640 | 0.132662 | 0.107659 | 0.8115 | phase1_representative_case |
| SQRT | P1 新增 | positive_f32c1 | scalar | CV_32F | 1 | continuous | 480x640 | 0.046105 | 0.045006 | 0.9762 | phase1_representative_case |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.266643 | 0.202192 | 0.7583 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.042352 | 0.030162 | 0.7122 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.041670 | 0.030145 | 0.7234 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.119517 | 0.091079 | 0.7621 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 1080x1920 | 0.786665 | 0.625549 | 0.7952 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 479x641 | 0.119835 | 0.090318 | 0.7537 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.117037 | 0.091169 | 0.7790 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_32F | 3 | continuous | 720x1280 | 0.347900 | 0.272643 | 0.7837 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.064861 | 0.049793 | 0.7677 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.009580 | 0.006848 | 0.7149 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.009556 | 0.005066 | 0.5302 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.029782 | 0.022043 | 0.7401 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.186467 | 0.148975 | 0.7989 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.029639 | 0.023173 | 0.7819 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.028595 | 0.022077 | 0.7721 | correctness=upstream_pass |
| SUBTRACT | 既有 | mat_mat_continuous | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.084693 | 0.066533 | 0.7856 | correctness=upstream_pass |
| SUM | P1 新增 | f32c3 | opencv_ui | CV_32F | 3 | continuous | 480x640 | 0.109352 | 0.228300 | 2.0877 | phase1_representative_case |
| SWAP | P1 新增 | mat_headers | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000005 | 0.000005 | 1.0000 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 1080x1920 | 0.566341 | 0.545820 | 0.9638 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 479x641 | 0.032810 | 0.032305 | 0.9846 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.072220 | 0.070327 | 0.9738 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_32F | 1 | continuous | 720x1280 | 0.288403 | 0.287582 | 0.9972 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 1080x1920 | 2.257950 | 1.450359 | 0.6423 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 479x641 | 0.288372 | 0.119217 | 0.4134 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 480x640 | 0.269253 | 0.152452 | 0.5662 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_32F | 3 | continuous | 720x1280 | 0.849584 | 0.584169 | 0.6876 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.094996 | 0.119190 | 1.2547 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.011032 | 0.010514 | 0.9531 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.006815 | 0.006941 | 1.0185 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.052387 | 0.027092 | 0.5171 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 1080x1920 | 1.657366 | 0.707542 | 0.4269 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 479x641 | 0.256208 | 0.085106 | 0.3322 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 480x640 | 0.238358 | 0.083997 | 0.3524 | correctness=upstream_pass |
| TRANSPOSE | 既有 | continuous | scalar | CV_8U | 3 | continuous | 720x1280 | 0.726661 | 0.369934 | 0.5091 | correctness=upstream_pass |
| VCONCAT | P1 新增 | two_halves_u8c1 | scalar | CV_8U | 1 | continuous | 480x640 | 0.004068 | 0.004248 | 1.0442 | phase1_representative_case |

### `imgproc`

| Op | 阶段 | Variant | CVH dispatch | Depth | Ch | Layout | Shape | CVH ms | OpenCV ms | OpenCV/CVH | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCUMULATE | P1 新增 | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.049252 | 0.022472 | 0.4563 | phase1_representative_case |
| ACCUMULATE_PRODUCT | P1 新增 | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.073173 | 0.024767 | 0.3385 | phase1_representative_case |
| ACCUMULATE_SQUARE | P1 新增 | u8c1_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.070740 | 0.021165 | 0.2992 | phase1_representative_case |
| ACCUMULATE_WEIGHTED | P1 新增 | alpha0_1_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.071008 | 0.024899 | 0.3507 | phase1_representative_case |
| ADAPTIVE_THRESHOLD | P1 新增 | mean11_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.382894 | 0.244210 | 0.6378 | phase1_representative_case |
| APPLY_COLOR_MAP | P1 新增 | jet_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.785277 | 0.251238 | 0.3199 | phase1_representative_case |
| BILATERAL_FILTER | P1 新增 | d5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 11.894075 | 1.007546 | 0.0847 | phase1_representative_case |
| BLEND_LINEAR | P1 新增 | u8c3_f32_weights | public_header_baseline | CV_8U | 3 | continuous | 480x640 | 0.613942 | 0.239558 | 0.3902 | phase1_representative_case |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 2.178228 | 0.264933 | 0.1216 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.380457 | 0.050431 | 0.1326 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.327640 | 0.041407 | 0.1264 |  |
| BOX_FILTER | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.949099 | 0.117015 | 0.1233 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c1 | box3x3 | CV_32F | 1 | continuous | 480x640 | 0.422320 | 0.106080 | 0.2512 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c3 | box3x3 | CV_32F | 3 | continuous | 480x640 | 0.607143 | 0.313995 | 0.5172 |  |
| BOX_FILTER | 既有 | 3x3_replicate_f32c4 | box3x3 | CV_32F | 4 | continuous | 480x640 | 0.644673 | 0.418859 | 0.6497 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3 | box3x3 | CV_8U | 3 | continuous | 480x640 | 0.846514 | 0.133620 | 0.1578 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c3_roi | box3x3 | CV_8U | 3 | roi | 479x641 | 0.858475 | 0.134295 | 0.1564 |  |
| BOX_FILTER | 既有 | 3x3_replicate_u8c4 | box3x3 | CV_8U | 4 | continuous | 480x640 | 1.021625 | 0.173231 | 0.1696 |  |
| BUILD_PYRAMID | P1 新增 | levels3_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.090905 | 0.021339 | 0.2347 | phase1_representative_case |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 52.699581 | 27.904995 | 0.5295 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 8.422524 | 4.166630 | 0.4947 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 7.654987 | 4.053726 | 0.5296 |  |
| CANNY | 既有 | aperture3_l1 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 23.331050 | 12.336463 | 0.5288 |  |
| CONVERT_MAPS | P1 新增 | f32_pair_to_fixed | opencv_ui | CV_32F | 2 | continuous | 480x640 | 0.115505 | 0.067753 | 0.5866 | phase1_representative_case |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.044235 | 0.042713 | 0.9656 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.008385 | 0.007656 | 0.9130 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.006959 | 0.006359 | 0.9138 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.019010 | 0.018229 | 0.9589 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 0.028423 | 0.026467 | 0.9312 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.080486 | 0.087052 | 1.0816 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 0.114858 | 0.115297 | 1.0038 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.022355 | 0.021258 | 0.9509 |  |
| COPY_MAKE_BORDER | 既有 | 2px_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.029745 | 0.027800 | 0.9346 |  |
| CREATE_HANNING_WINDOW | P1 新增 | 64x64_f32 | opencv_ui | CV_32F | 1 | continuous | 480x640 | 0.000702 | 0.001340 | 1.9091 | phase1_representative_case |
| CVTCOLOR | 既有 | BGR2BGRA_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.285733 | 0.020901 | 0.0731 |  |
| CVTCOLOR | 既有 | BGR2GRAY_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.093677 | 0.047886 | 0.5112 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 1080x1920 | 0.207883 | 0.205693 | 0.9895 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 479x641 | 0.035347 | 0.035413 | 1.0019 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.030390 | 0.030411 | 1.0007 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8 | opencv_ui | CV_8U | 3 | continuous | 720x1280 | 0.091425 | 0.091856 | 1.0047 |  |
| CVTCOLOR | 既有 | BGR2GRAY_u8_roi | opencv_ui | CV_8U | 3 | roi | 479x641 | 0.035412 | 0.035172 | 0.9932 |  |
| CVTCOLOR | 既有 | BGR2I420_u8 | header_fastpath | CV_8U | 3 | yuv420_i420 | 480x640 | 0.396696 | 0.061367 | 0.1547 |  |
| CVTCOLOR | 既有 | BGR2RGB_f32 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 0.168445 | 0.071859 | 0.4266 |  |
| CVTCOLOR | 既有 | BGR2RGB_u8 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.135666 | 0.016789 | 0.1238 |  |
| CVTCOLOR | 既有 | BGR2YUV_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.330710 | 0.084200 | 0.2546 |  |
| CVTCOLOR | 既有 | BGR2YUY2_u8 | header_fastpath | CV_8U | 3 | yuv422_yuy2 | 480x640 | 0.353767 | 0.063696 | 0.1801 |  |
| CVTCOLOR | 既有 | BGRA2GRAY_u8 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.185259 | 0.044016 | 0.2376 |  |
| CVTCOLOR | 既有 | I420_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_i420 | 480x640 | 0.924885 | 0.073322 | 0.0793 |  |
| CVTCOLOR | 既有 | NV12_TO_BGR_u8 | header_fastpath | CV_8U | 1 | yuv420_nv12 | 480x640 | 0.434705 | 0.073889 | 0.1700 |  |
| CVTCOLOR | 既有 | YUV2BGR_u8 | header_fastpath | CV_8U | 3 | yuv444_interleaved | 480x640 | 0.336830 | 0.061818 | 0.1835 |  |
| CVTCOLOR | 既有 | YUY2_TO_BGR_u8 | header_fastpath | CV_8U | 2 | yuv422_yuy2 | 480x640 | 0.432366 | 0.075553 | 0.1747 |  |
| CVT_COLOR_TWO_PLANE | P1 新增 | nv12_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.367768 | 0.076595 | 0.2083 | phase1_representative_case |
| DEMOSAICING | P1 新增 | bayer_bg_to_bgr | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.523265 | 0.044593 | 0.0852 | phase1_representative_case |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.189365 | 0.133581 | 0.7054 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.037294 | 0.023059 | 0.6183 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.034606 | 0.021135 | 0.6107 |  |
| DILATE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.090920 | 0.061989 | 0.6818 |  |
| DILATE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 2.261290 | 0.068141 | 0.0301 |  |
| DILATE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.971070 | 0.092783 | 0.0312 |  |
| EQUALIZE_HIST | P1 新增 | u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.095071 | 0.100251 | 1.0545 | phase1_representative_case |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.187782 | 0.132648 | 0.7064 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.037238 | 0.022010 | 0.5911 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.034710 | 0.021056 | 0.6066 |  |
| ERODE | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.088701 | 0.062844 | 0.7085 |  |
| ERODE | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 2.353219 | 0.070666 | 0.0300 |  |
| ERODE | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.974203 | 0.091132 | 0.0306 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 9.956713 | 0.550914 | 0.0553 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 1.770172 | 0.104985 | 0.0593 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 1.509298 | 0.091017 | 0.0603 |  |
| FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 4.478987 | 0.250567 | 0.0559 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 1.901887 | 0.075275 | 0.0396 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 2.271804 | 0.204690 | 0.0901 |  |
| FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 2.329406 | 0.256964 | 0.1103 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 2.986658 | 0.276330 | 0.0925 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 3.133596 | 0.288336 | 0.0920 |  |
| FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 2.658541 | 0.367760 | 0.1383 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 6.767304 | 0.206670 | 0.0305 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 1.113953 | 0.035533 | 0.0319 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.992187 | 0.029935 | 0.0302 |  |
| GAUSSIAN | 既有 | 5x5_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 2.966122 | 0.087409 | 0.0295 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c1 | gauss_separable | CV_32F | 1 | continuous | 480x640 | 1.771280 | 0.120510 | 0.0680 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c3 | gauss_separable | CV_32F | 3 | continuous | 480x640 | 1.461101 | 0.337184 | 0.2308 |  |
| GAUSSIAN | 既有 | 5x5_replicate_f32c4 | gauss_separable | CV_32F | 4 | continuous | 480x640 | 0.970598 | 0.465028 | 0.4791 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3 | gauss_separable | CV_8U | 3 | continuous | 480x640 | 1.599587 | 0.102992 | 0.0644 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c3_roi | gauss_separable | CV_8U | 3 | roi | 479x641 | 1.736238 | 0.361882 | 0.2084 |  |
| GAUSSIAN | 既有 | 5x5_replicate_u8c4 | gauss_separable | CV_8U | 4 | continuous | 480x640 | 1.215871 | 0.138097 | 0.1136 |  |
| GET_AFFINE_TRANSFORM | P1 新增 | three_points | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000145 | 0.000292 | 2.0172 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_DERIV_KERNELS | P1 新增 | dx1_ksize5_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000148 | 0.000099 | 0.6685 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GABOR_KERNEL | P1 新增 | 15x15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.001112 | 0.001101 | 0.9903 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_GAUSSIAN_KERNEL | P1 新增 | ksize15_f32 | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000265 | 0.001063 | 4.0158 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_PERSPECTIVE_TRANSFORM | P1 新增 | four_points_lu | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000320 | 0.000739 | 2.3116 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 1080x1920 | 0.087937 | 1.087791 | 12.3701 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 479x641 | 0.018981 | 0.160895 | 8.4765 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 480x640 | 0.011111 | 0.144730 | 13.0261 | no qualified SIMD fast path |
| GET_RECT_SUB_PIX | P1 新增 | full_frame_u8c3 | public_header_scalar | CV_8U | 3 | continuous | 720x1280 | 0.033298 | 0.436239 | 13.1011 | no qualified SIMD fast path |
| GET_ROTATION_MATRIX_2D | P1 新增 | point_angle_scale | public_header_baseline | CV_32F | 1 | continuous | micro_batch | 0.000079 | 0.000082 | 1.0423 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_ROTATION_MATRIX_2D_ | P1 新增 | matx23d | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000005 | 0.000005 | 1.0917 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| GET_STRUCTURING_ELEMENT | P1 新增 | ellipse7x7 | public_header_baseline | CV_8U | 1 | continuous | micro_batch | 0.000116 | 0.000088 | 0.7527 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| INTEGRAL | P1 新增 | u8c1_to_s32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.084745 | 0.047291 | 0.5580 | phase1_representative_case |
| INVERT_AFFINE_TRANSFORM | P1 新增 | f64_2x3 | public_header_baseline | CV_64F | 1 | continuous | micro_batch | 0.000045 | 0.000067 | 1.4907 | phase1_representative_case;micro_warmup=2;micro_iterations=100;micro_repeats=3 |
| LAPLACIAN | P1 新增 | ksize3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 2.026166 | 0.132305 | 0.0653 | phase1_representative_case |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 0.223794 | 0.174978 | 0.7819 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.041144 | 0.031555 | 0.7669 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.033852 | 0.026163 | 0.7729 |  |
| LUT | 既有 | invert_u8 | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.098952 | 0.077031 | 0.7785 |  |
| LUT | 既有 | invert_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.115813 | 0.089798 | 0.7754 |  |
| LUT | 既有 | invert_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.154108 | 0.119266 | 0.7739 |  |
| MEDIAN_BLUR | P1 新增 | ksize5_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 1.523952 | 0.284006 | 0.1864 | phase1_representative_case |
| PYR_DOWN | P1 新增 | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.375958 | 0.068715 | 0.1828 | phase1_representative_case |
| PYR_UP | P1 新增 | u8c3 | opencv_ui | CV_8U | 3 | continuous | 480x640 | 0.701069 | 0.169061 | 0.2411 | phase1_representative_case |
| REMAP | P1 新增 | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 12.952285 | 5.060862 | 0.3907 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 1.930964 | 0.771628 | 0.3996 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 1.864397 | 0.691780 | 0.3710 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | fixed_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 5.624973 | 2.104468 | 0.3741 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 14.941002 | 5.130040 | 0.3434 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 2.195107 | 0.760317 | 0.3464 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.133985 | 0.712235 | 0.3338 | Shared fixed coordinate block and U8 bilinear sampling path |
| REMAP | P1 新增 | float_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 6.534969 | 2.133977 | 0.3265 | Shared fixed coordinate block and U8 bilinear sampling path |
| RESIZE | 既有 | linear_0.75_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 0.196016 | 0.093376 | 0.4764 |  |
| RESIZE | 既有 | linear_0.75_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 0.453685 | 0.290178 | 0.6396 |  |
| RESIZE | 既有 | linear_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.387833 | 0.065366 | 0.1685 |  |
| RESIZE | 既有 | linear_0.75_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 0.384184 | 0.064966 | 0.1691 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 1080x1920 | 0.122755 | 0.081538 | 0.6642 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 479x641 | 0.021546 | 0.013970 | 0.6484 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.018220 | 0.012063 | 0.6621 |  |
| RESIZE | 既有 | linear_half_u8c1 | opencv_ui | CV_8U | 1 | continuous | 720x1280 | 0.054154 | 0.037518 | 0.6928 |  |
| RESIZE | 既有 | nearest_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.136731 | 0.103163 | 0.7545 |  |
| RESIZE | 既有 | nearest_exact_0.75_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.132751 | 0.101927 | 0.7678 |  |
| SCHARR | P1 新增 | dx1_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 1.985321 | 0.128180 | 0.0646 | phase1_representative_case |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 4.175134 | 0.576690 | 0.1381 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.733770 | 0.104439 | 0.1423 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.626640 | 0.094486 | 0.1508 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 1.817813 | 0.262945 | 0.1446 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c1 | header_fastpath | CV_32F | 1 | continuous | 480x640 | 1.075681 | 0.085975 | 0.0799 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c3 | header_fastpath | CV_32F | 3 | continuous | 480x640 | 1.524168 | 0.227607 | 0.1493 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_f32c4 | header_fastpath | CV_32F | 4 | continuous | 480x640 | 1.386747 | 0.298119 | 0.2150 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 1.600749 | 0.286654 | 0.1791 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c3_roi | header_fastpath | CV_8U | 3 | roi | 479x641 | 1.576934 | 0.287332 | 0.1822 |  |
| SEP_FILTER2D | 既有 | 3x3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 1.341194 | 0.369933 | 0.2758 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 1080x1920 | 1.686118 | 0.787681 | 0.4672 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 479x641 | 0.290261 | 0.133484 | 0.4599 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 480x640 | 0.263790 | 0.116470 | 0.4415 |  |
| SOBEL | 既有 | dx1_ksize3_replicate | header_fastpath | CV_8U | 1 | continuous | 720x1280 | 0.814018 | 0.312244 | 0.3836 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c3 | header_fastpath | CV_8U | 3 | continuous | 480x640 | 0.685045 | 0.370308 | 0.5406 |  |
| SOBEL | 既有 | dx1_ksize3_replicate_u8c4 | header_fastpath | CV_8U | 4 | continuous | 480x640 | 0.988461 | 0.462175 | 0.4676 |  |
| SPATIAL_GRADIENT | P1 新增 | ksize3_u8_to_s16 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.220222 | 0.038027 | 0.1727 | phase1_representative_case |
| SQR_BOX_FILTER | P1 新增 | 3x3_u8_to_f32 | opencv_ui | CV_8U | 1 | continuous | 480x640 | 0.502311 | 0.204530 | 0.4072 | phase1_representative_case |
| STACK_BLUR | P1 新增 | 5x5_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.604163 | 0.140982 | 0.2334 | phase1_representative_case |
| THRESHOLD | 既有 | binary_f32c3_roi | header_fastpath | CV_32F | 3 | roi | 479x641 | 0.076535 | 0.069077 | 0.9026 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 1080x1920 | 0.033354 | 0.033263 | 0.9973 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 479x641 | 0.005622 | 0.005725 | 1.0183 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 480x640 | 0.004774 | 0.004539 | 0.9509 |  |
| THRESHOLD | 既有 | binary_u8 | headers_baseline | CV_8U | 1 | continuous | 720x1280 | 0.014492 | 0.014505 | 1.0009 |  |
| THRESHOLD_WITH_MASK | P1 新增 | binary_masked_u8c1 | public_header_baseline | CV_8U | 1 | continuous | 480x640 | 0.191124 | 0.184785 | 0.9668 | phase1_representative_case |
| WARP_AFFINE | 既有 | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 1080x1920 | 11.721412 | 1.957577 | 0.1670 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | 既有 | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 479x641 | 1.759004 | 0.313391 | 0.1782 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | 既有 | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 480x640 | 1.655646 | 0.289979 | 0.1751 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | 既有 | linear_inverse_replicate | fixed_coordinate_block | CV_8U | 1 | continuous | 720x1280 | 5.020819 | 0.870428 | 0.1734 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c1 | headers_baseline | CV_32F | 1 | continuous | 480x640 | 2.910061 | 0.530339 | 0.1822 | F32 path remains the public header baseline |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c3 | headers_baseline | CV_32F | 3 | continuous | 480x640 | 7.606045 | 0.758765 | 0.0998 | F32 path remains the public header baseline |
| WARP_AFFINE | 既有 | linear_inverse_replicate_f32c4 | headers_baseline | CV_32F | 4 | continuous | 480x640 | 10.125158 | 0.848578 | 0.0838 | F32 path remains the public header baseline |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.168270 | 0.816060 | 0.3764 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_AFFINE | 既有 | linear_inverse_replicate_u8c4 | fixed_coordinate_block | CV_8U | 4 | continuous | 480x640 | 2.330568 | 0.538627 | 0.2311 | U8 linear path uses fixed coordinate blocks and shared sampling |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 1080x1920 | 14.744065 | 6.750619 | 0.4579 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 479x641 | 2.264334 | 1.070728 | 0.4729 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 480x640 | 2.120560 | 0.959225 | 0.4523 | Shared fixed coordinate block and U8 bilinear sampling path |
| WARP_PERSPECTIVE | P1 新增 | projective_linear_u8c3 | fixed_coordinate_block | CV_8U | 3 | continuous | 720x1280 | 6.508190 | 2.877213 | 0.4421 | Shared fixed coordinate block and U8 bilinear sampling path |

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
