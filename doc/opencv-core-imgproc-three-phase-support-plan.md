# OpenCV Core / Imgproc 三阶段支持清单

更新时间：2026-08-06

## 1. 文档范围

本清单依据
[opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md)
中的 220 个 upstream CPU C++ 操作族进行分期。

P2-P0 的 17 个操作族已完成并通过最终阶段 gate；当前共有 124 个可调用
操作族，其中 `demosaicing` 作为可调用 preview 保留但不计入 v0.1 支持承诺。
因此 v0.1 承诺 123 个操作族，没有仅声明未实现的 API，另有 96 个操作族
未支持。其余第二阶段操作族只保留为需求驱动的候选。本清单只回答：

- 每个阶段支持哪些 `core` 算子；
- 每个阶段支持哪些 `imgproc` 算子；
- 为什么把这些算子放在该阶段。

本清单不描述实现方法、测试步骤、SIMD 策略或交付流程。P2-P0 的当前
可调用范围以 API coverage 为准，参数子集以 Core/Imgproc 模块 README
为准，正确性以单测和 OpenCV contract 差分为准。`220 / 220` 只作为
upstream 操作族盘点上限，不再作为发布承诺。

## 2. 三阶段总览

| 阶段 | 定位 | Core | Imgproc | 操作族 | 累计覆盖/状态 |
|---|---|---:|---:|---:|---|
| 第一阶段 | 高频基础算子与通用图像流水线 | 43 | 35 | 78 | 已完成；106 / 220，48.2% |
| 第二阶段 P0 | 随机数据、坐标、区域、轮廓、形状、直方图和模板匹配 | 4 | 13 | 17 | 已完成；123 / 220，55.9% |
| 第二阶段 backlog | 低频或高依赖数值/特征/形状候选 | 31 | 35 | 66 | 不计入承诺覆盖；含可调用 preview `demosaicing` |
| 第三阶段 backlog | 高复杂度算法与长尾接口 | 5 | 26 | 31 | 不计入承诺覆盖 |

## 3. 第一阶段：高频基础算子与通用图像流水线

第一阶段优先补齐调用频率高、复用范围广，并且会被第二、三阶段反复依赖的
基础能力。

第一阶段已经完成；当前可调用范围及限制以
[API coverage](opencv-core-imgproc-api-coverage.md) 为准。完成的拆分与逐项
验收记录由 Git 历史归档。

### 3.1 Core 支持列表

<!-- P1_CORE_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第一阶段的原因 |
|---|---|---:|---|
| 逐元素与逻辑运算 | `absdiff`<br>`bitwise_and`<br>`bitwise_not`<br>`bitwise_or`<br>`bitwise_xor`<br>`inRange`<br>`min`<br>`max` | 8 | 掩码、阈值、形态学、合成和后续区域算法都会直接使用。 |
| 转换、数学与数据校验 | `scaleAdd`<br>`convertScaleAbs`<br>`convertFp16`<br>`sqrt`<br>`pow`<br>`exp`<br>`log`<br>`checkRange`<br>`patchNaNs` | 9 | 是归一化、数值预处理、浮点数据清理和后续统计计算的基础。 |
| 归约与统计 | `norm`<br>`sum`<br>`mean`<br>`meanStdDev`<br>`countNonZero`<br>`hasNonZero`<br>`findNonZero`<br>`minMaxIdx`<br>`minMaxLoc`<br>`reduce`<br>`reduceArgMax`<br>`reduceArgMin`<br>`normalize` | 13 | 使用频率高，并为特征评分、直方图、距离度量和数值算法提供公共能力。 |
| 布局、复制与通道操作 | `borderInterpolate`<br>`copyTo`<br>`extractChannel`<br>`insertChannel`<br>`mixChannels`<br>`flip`<br>`flipND`<br>`rotate`<br>`repeat`<br>`hconcat`<br>`vconcat`<br>`broadcast`<br>`swap` | 13 | 解决数据搬运和布局转换，能够被滤波、金字塔、几何变换及测试数据构造复用。 |
| **合计** |  | **43** |  |
<!-- P1_CORE_API_LIST_END -->

### 3.2 Imgproc 支持列表

<!-- P1_IMGPROC_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第一阶段的原因 |
|---|---|---:|---|
| 核生成、滤波与强度处理 | `getStructuringElement`<br>`getGaussianKernel`<br>`getDerivKernels`<br>`getGaborKernel`<br>`createHanningWindow`<br>`integral`<br>`Scharr`<br>`Laplacian`<br>`spatialGradient`<br>`sqrBoxFilter`<br>`medianBlur`<br>`bilateralFilter`<br>`stackBlur`<br>`adaptiveThreshold`<br>`thresholdWithMask`<br>`equalizeHist`<br>`applyColorMap` | 17 | 与现有滤波、Sobel、threshold 路径关联紧密，也是最常见的预处理能力。 |
| 累积、金字塔与颜色输入 | `accumulate`<br>`accumulateProduct`<br>`accumulateSquare`<br>`accumulateWeighted`<br>`blendLinear`<br>`pyrDown`<br>`pyrUp`<br>`buildPyramid`<br>`cvtColorTwoPlane` | 9 | 视频统计、图像融合、多尺度处理和 YUV 输入场景使用频率较高。 |
| 几何变换基础 | `remap`<br>`convertMaps`<br>`warpPerspective`<br>`getAffineTransform`<br>`getPerspectiveTransform`<br>`getRotationMatrix2D`<br>`getRotationMatrix2D_`<br>`invertAffineTransform`<br>`getRectSubPix` | 9 | remap 和变换矩阵是透视、极坐标、配准及后续几何算法的共同基础。 |
| **合计** |  | **35** |  |
<!-- P1_IMGPROC_API_LIST_END -->

## 4. 第二阶段 P0：精选高价值操作族

P2-P0 建立在第一阶段的 Mat、归约、滤波、布局和几何能力之上，只选择能够
形成完整实际流水线且依赖边界可控的操作族。17 个操作族已从 public umbrella
header 可调用并计入 available coverage，并已进入长期单测、header
contract 和 OpenCV 差分门禁。

### 4.1 P2-P0 Core：4 个

<!-- P2_P0_CORE_API_LIST_START -->
| 类别 | 算子 | 数量 | 选择原因 |
|---|---|---:|---|
| 随机 Mat | `randu`<br>`randn` | 2 | 服务数据初始化、噪声模拟、测试和 benchmark 输入。 |
| 点坐标变换 | `transform`<br>`perspectiveTransform` | 2 | 与现有仿射、透视和点坐标处理直接衔接。 |
| **合计** |  | **4** |  |
<!-- P2_P0_CORE_API_LIST_END -->

### 4.2 P2-P0 Imgproc：13 个

<!-- P2_P0_IMGPROC_API_LIST_START -->
| 类别 | 算子 | 数量 | 选择原因 |
|---|---|---:|---|
| 区域分析 | `connectedComponents`<br>`connectedComponentsWithStats` | 2 | 分割 mask、OCR、缺陷检测和目标区域过滤；共享标记内核。 |
| 轮廓入口 | `findContours` | 1 | 解锁后续基础轮廓和形状处理。 |
| 基础形状 | `boundingRect`<br>`contourArea`<br>`arcLength`<br>`approxPolyDP`<br>`convexHull`<br>`isContourConvex`<br>`moments` | 7 | 共用点集、轮廓遍历和几何基础设施。 |
| 直方图 | `calcHist`<br>`compareHist` | 2 | 用于质量分析、颜色统计和简单检索。 |
| 模板匹配 | `matchTemplate` | 1 | 用于工业视觉、固定 UI/图标检测和小目标定位。 |
| **合计** |  | **13** |  |
<!-- P2_P0_IMGPROC_API_LIST_END -->

### 4.3 第二阶段候选 backlog：66 个

以下操作族不属于 P2-P0。只有真实流水线需求、共享基础设施和维护成本得到确认
后，才从 backlog 中建立新的 P2 批次。

Core 候选 31 个：

- 线性代数与统计：`setIdentity`、`trace`、`determinant`、`completeSymm`、
  `invert`、`solve`、`mulTransposed`、`SVDecomp`、`SVBackSubst`、
  `calcCovarMatrix`、`PCACompute`、`PCAProject`、`PCABackProject`、
  `Mahalanobis`、`PSNR`、`batchDistance`；
- 坐标、频域、随机状态与排序：`cartToPolar`、`polarToCart`、`phase`、
  `magnitude`、`dft`、`idft`、`dct`、`idct`、`mulSpectrums`、
  `getOptimalDFTSize`、`randShuffle`、`setRNGSeed`、`theRNG`、`sort`、
  `sortIdx`。

Imgproc 候选 35 个：

- 低频颜色输入：`demosaicing`。现有 U8 bilinear preview、正确性测试和
  benchmark 证据保留，但 v0.1 RC 中 OpenCV 快 `11.70x`，只有出现真实
  Bayer 流水线需求并建立性能验收后才重新进入支持面；
- 直方图、频域、区域与极坐标：`calcBackProject`、`createCLAHE`、
  `phaseCorrelate`、`phaseCorrelateIterative`、`divSpectrums`、
  `distanceTransform`、`floodFill`、`linearPolar`、`logPolar`、`warpPolar`；
- 角点与特征：`cornerEigenValsAndVecs`、`cornerHarris`、
  `cornerMinEigenVal`、`cornerSubPix`、`preCornerDetect`、
  `goodFeaturesToTrack`；
- 高阶形状：`HuMoments`、`approxPolyN`、`boxPoints`、`convexityDefects`、
  `findContoursLinkRuns`、`fitEllipse`、`fitEllipseAMS`、`fitEllipseDirect`、
  `fitLine`、`getClosestEllipsePoints`、`intersectConvexConvex`、
  `matchShapes`、`minAreaRect`、`minEnclosingCircle`、
  `minEnclosingConvexPolygon`、`minEnclosingTriangle`、`pointPolygonTest`、
  `rotatedRectangleIntersection`。

## 5. 第三阶段：高复杂度算法与长尾接口

第三阶段处理实现复杂度高、数值稳定性要求高、需要对象模型，或者相对低频的
算法。这些算子会复用前两阶段已经建立的矩阵、频域、轮廓、区域和几何能力。

### 5.1 Core 支持列表

<!-- P3_CORE_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第三阶段的原因 |
|---|---|---:|---|
| 高复杂度数值算法 | `eigen`<br>`eigenNonSymmetric`<br>`solveCubic`<br>`solvePoly`<br>`kmeans` | 5 | 对收敛性、数值稳定性和边界输入要求较高；kmeans 还会被 GrabCut 依赖。 |
| **合计** |  | **5** |  |
<!-- P3_CORE_API_LIST_END -->

### 5.2 Imgproc 支持列表

<!-- P3_IMGPROC_API_LIST_START -->
| 类别 | 算子 | 数量 | 放在第三阶段的原因 |
|---|---|---:|---|
| 绘制与文本 | `arrowedLine`<br>`circle`<br>`clipLine`<br>`drawContours`<br>`drawMarker`<br>`ellipse`<br>`ellipse2Poly`<br>`fillConvexPoly`<br>`fillPoly`<br>`getFontScaleFromHeight`<br>`getTextSize`<br>`line`<br>`polylines`<br>`putText`<br>`rectangle` | 15 | 对核心预处理性能影响较小，但需要共享光栅化、裁剪、填充和字体能力。 |
| Hough 与检测器 | `HoughCircles`<br>`HoughLines`<br>`HoughLinesP`<br>`HoughLinesPointSet`<br>`createGeneralizedHoughBallard`<br>`createGeneralizedHoughGuil`<br>`createLineSegmentDetector` | 7 | 依赖 Canny、排序、几何类型和累加器，并涉及较复杂的检测器对象接口。 |
| 分割与专用算法 | `EMD`<br>`grabCut`<br>`pyrMeanShiftFiltering`<br>`watershed` | 4 | 算法复杂、工作区大、依赖链长；GrabCut 依赖 kmeans，均适合在基础能力稳定后支持。 |
| **合计** |  | **26** |  |
<!-- P3_IMGPROC_API_LIST_END -->

## 6. 分期原则

| 原则 | 对应阶段 |
|---|---|
| 高频、通用、可被大量其他算子复用 | 第一阶段 |
| 高频、能形成完整用户流水线、依赖边界可控 | 第二阶段 P0 |
| 需要真实需求证明价值的数值、频域、特征和高阶形状 API | 第二阶段 backlog |
| 高复杂度、长依赖链、对象型接口或低频长尾能力 | 第三阶段 backlog |

类成员 API、C API、`UMat`、CUDA、OpenCL、OpenGL、DirectX 等不在这份
三阶段算子清单中，其范围以
[opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md)
为准。
