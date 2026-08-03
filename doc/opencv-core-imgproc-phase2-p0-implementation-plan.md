# OpenCV Core / Imgproc 第二阶段 P0 落地计划

更新时间：2026-08-03

状态：已完成；P2-P0-S0 至 S7 全部门禁通过

## 1. 目标

第二阶段不再以原清单中的 82 个操作族全部落地为目标。P2-P0 只承诺对实际
图像预处理、分割后处理、工业视觉和测试数据构造价值较高的 17 个操作族：

- Core：4 个；
- Imgproc：13 个；
- 完成后当前可调用覆盖从 `107 / 220` 提升到 `124 / 220`，即 56.4%；
- 其余第二阶段操作族保留为需求驱动的候选 backlog。

本阶段保持纯 header-only、CPU-only 和单一 `cvh::headers` 产品 target。
正确性先于 SIMD；没有 benchmark 证据的实现不宣称 fast path。

## 2. P2-P0 操作族

### 2.1 Core：4 个

| 类别 | 操作族 | 价值 |
| --- | --- | --- |
| 随机 Mat | `randu`、`randn` | 数据初始化、噪声模拟、测试与 benchmark 输入生成。 |
| 点坐标变换 | `transform`、`perspectiveTransform` | 与现有仿射、透视和点坐标处理直接衔接。 |

### 2.2 Imgproc：13 个

| 类别 | 操作族 | 价值 |
| --- | --- | --- |
| 区域分析 | `connectedComponents`、`connectedComponentsWithStats` | 分割 mask、OCR、缺陷检测和目标区域过滤；共享标记内核。 |
| 轮廓入口 | `findContours` | 解锁基础轮廓和形状处理。 |
| 基础形状 | `boundingRect`、`contourArea`、`arcLength`、`approxPolyDP`、`convexHull`、`isContourConvex`、`moments` | 共用点集、轮廓遍历和几何基础设施。 |
| 直方图 | `calcHist`、`compareHist` | 质量分析、颜色统计和简单检索。 |
| 模板匹配 | `matchTemplate` | 工业视觉、固定 UI/图标检测和小目标定位。 |

## 3. 首批公开合同

### 3.1 `randu` 与 `randn`

公开入口：

```cpp
cvh::randu(dst, low, high);
cvh::randn(dst, mean, stddev);
```

首批支持：

- 已分配、非空的 Mat；
- `CV_8U`、`CV_8S`、`CV_16U`、`CV_16S`、`CV_32S`、`CV_32F`、`CV_64F`；
- C1/C2/C3/C4，二维连续 Mat、ROI 和 N-D Mat；
- `Scalar` 为各通道提供上下界、均值和标准差；
- uniform 整数上界不包含，浮点范围与 pinned OpenCV 行为对齐；
- normal 输出按目标深度执行 OpenCV 对齐的舍入与饱和。

随机状态使用 C++17 ODR-safe 的 inline thread-local engine。P2-P0 只计入
`randu`、`randn` 两个操作族；`setRNGSeed`、`theRNG`、`randShuffle` 和公开
`RNG` 类型仍属于 backlog。测试使用进程首次调用的固定初始状态验证确定性，
不得依赖执行顺序跨测试共享随机序列。

### 3.2 `transform` 与 `perspectiveTransform`

首批支持：

- `transform`：F32/F64，C1-C4，二维连续和 ROI 输入；
- 变换矩阵为 F32/F64，列数等于输入通道数或输入通道数加一；
- `perspectiveTransform`：F32/F64 C2/C3 点集，3x3 或 4x4 矩阵；
- 支持 `N x 1`、`1 x N` 点 Mat；
- 目标 Mat 允许复用，不允许静默接受不兼容通道或矩阵形状；
- 透视除法、零 `w`、NaN 和 Inf 与 pinned OpenCV 行为差分验证。

### 3.3 Connected components

首批支持：

- `CV_8UC1` 二值输入；
- 4 连通和 8 连通；
- `CV_32S` label 输出；
- 背景固定为 label 0，前景标签按稳定的行优先顺序生成；
- stats 使用 `CV_32S`，包含 left/top/width/height/area；
- centroids 使用 `CV_64F`；
- 两个公开 API 共享同一个 two-pass union-find 内核；
- 覆盖空前景、全前景、单像素、对角连接、ROI 和大量小区域。

`CV_16U` label 和 OpenCV 的多种 CCL algorithm selector 不进入 P2-P0。

### 3.4 Contours and basic shapes

P2-P0 引入完成本组 API 所需的最小公开类型：

- `Rect_<T>`、`Rect`；
- `Moments`；
- contour retrieval 和 approximation 枚举。

首批 `findContours` 支持：

- `CV_8UC1` 二值输入；
- 不修改输入；
- `RETR_EXTERNAL`、`RETR_LIST`；
- `CHAIN_APPROX_NONE`、`CHAIN_APPROX_SIMPLE`；
- `std::vector<std::vector<Point>>` 输出；
- 可选 offset；
- hierarchy 输出、`RETR_CCOMP` 和 `RETR_TREE` 留在 backlog。

基础形状首批支持：

- `boundingRect`：整数或浮点点集；
- `contourArea`：有向和无向面积；
- `arcLength`：open/closed；
- `approxPolyDP`：open/closed；
- `convexHull`：返回点，支持 clockwise；索引输出留在 backlog；
- `isContourConvex`；
- `moments`：点集和 `CV_8UC1` 二值图像。

所有点集 API 必须定义空集、退化线段、重复点、自相交轮廓、整数溢出和
浮点非有限值的行为。

### 3.5 Histogram

`calcHist` 首批支持：

- 单个 U8/F32 二维 Mat；
- C1/C3/C4 中选择一个通道；
- 一维 dense histogram；
- uniform bins；
- 可选 `CV_8UC1` mask；
- F32 histogram 输出；
- accumulate=false/true。

`compareHist` 首批支持 dense F32 单通道 histogram，并覆盖 correlation、
chi-square、intersection 和 Bhattacharyya 四种方法。多图输入、多维 histogram、
非 uniform ranges 和 SparseMat 不进入 P2-P0。

### 3.6 `matchTemplate`

首批支持：

- U8/F32 C1；
- `TM_SQDIFF`、`TM_SQDIFF_NORMED`、`TM_CCORR`、`TM_CCORR_NORMED`；
- F32 单通道结果；
- template 不得大于 image；
- 首版使用直接空间滑窗实现，不为此引入 DFT；
- mask、C3/C4、`TM_CCOEFF` 和 `TM_CCOEFF_NORMED` 留在 backlog。

## 4. 落地顺序与实时状态

| Step | 内容 | 新增操作族 | 状态 |
| --- | --- | ---: | --- |
| P2-P0-S0 | 公共类型、枚举、测试 backend 和支持矩阵前置 | 0 | 已完成；独立/aggregate header compile、ODR 和 install consumer 通过 |
| P2-P0-S1 | `randu`、`randn` | 2 | 已完成；U8/S8/U16/S16/S32/F32/F64、C1-C4、ROI/N-D 和零方差覆盖通过 |
| P2-P0-S2 | `transform`、`perspectiveTransform` | 2 | 已完成；F32/F64、ROI/alias、零 `w`、NaN/Inf 与 pinned OpenCV 差分通过 |
| P2-P0-S3 | connected components 共享内核与两个 API | 2 | 已完成；4/8 连通、labels/stats/centroids 与 pinned OpenCV SAUF/WU 差分通过 |
| P2-P0-S4 | `findContours` 与基础形状 API | 8 | 已完成；retrieval/chain、offset、顺序/tie、退化形状和 moments 差分通过 |
| P2-P0-S5 | `calcHist`、`compareHist` | 2 | 已完成；U8/F32、mask/accumulate 和四种比较方法差分通过 |
| P2-P0-S6 | `matchTemplate` | 1 | 已完成；U8/F32、ROI、归一化零分母和四种方法差分通过 |
| P2-P0-S7 | 全量差分、benchmark、文档和 coverage 收口 | 0 | 已完成；Phase 2 单测 9/9、upstream 差分组 5/5、完整 CI 405/405 和 ASan/UBSan 通过；26 条 benchmark case 已并入 canonical Core/Imgproc suite 且均为 `OK` |
| **合计** |  | **17** |  |

每完成一个 Step，必须在本表中更新状态和实际验证结果。未通过验收的 API 不得
只保留声明，也不得提前加入 README 的 supported 表。

## 5. 代码组织

建议新增：

```text
include/cvh/core/random.h
include/cvh/core/transform.h
include/cvh/core/detail/random_impl.hpp
include/cvh/core/detail/transform_impl.hpp
include/cvh/imgproc/connected_components.h
include/cvh/imgproc/contours.h
include/cvh/imgproc/shape.h
include/cvh/imgproc/histogram.h
include/cvh/imgproc/template_match.h
include/cvh/imgproc/detail/*_impl.hpp
```

公开头加入对应模块 umbrella header；实现必须 inline、模板化或使用 C++17
inline state。不得新增项目 `.cpp` 产品文件、编译型 backend、外部运行时依赖或
新的 public CMake target。

## 6. 正确性和差分测试

每个操作族至少覆盖：

- 独立 public header 编译；
- aggregate header 编译；
- 多翻译单元 ODR；
- 支持类型、通道、flag 和 layout；
- ROI/non-contiguous 输入；
- 空输入、最小输入、退化输入和非法参数；
- 固定 OpenCV revision 的隔离差分；
- unsupported 组合显式报错；
- sanitizer 和 x86/macOS/Windows 编译验证。

整数结果应 byte-exact。浮点结果必须按算法固定绝对/相对误差，不得为了通过
随机或大规模 case 临时放宽容差。轮廓顺序、label 顺序和 tie 规则必须单独固定。

## 7. Benchmark 范围

P2-P0 建立性能基线，但不要求首版全部拥有 SIMD fast path：

- random：U8/F32，连续/ROI，不同元素规模；
- transform：点数、通道数和连续/ROI；
- connected components：稀疏、稠密、棋盘格和大量小区域；
- contours：简单外轮廓、密集边缘和多区域；
- shapes：点数和退化比例；
- histogram：U8/F32、bin 数和 mask；
- template matching：image/template 尺寸组合和四种方法。

Mode A 负责 cvh 内部回归；Mode B 只比较 `cvh_ui` 产品入口与 upstream
OpenCV。没有稳定收益时保留 scalar header implementation。

## 8. P2-P0 完成条件

- [x] 17 个操作族均可从 public umbrella header 调用。
- [x] 当前支持矩阵在 Core/Imgproc README 和 API coverage 中一致。
- [x] Core coverage 为 61/97，Imgproc coverage 为 63/123。
- [x] 总 callable family coverage 为 124/220，且 declared-only 为 0。
- [x] 所有支持组合通过固定 OpenCV revision 差分。
- [x] header compile、ODR、install consumer 和完整 CI 通过。
- [x] 新增 benchmark 记录 dispatch、输入矩阵和 raw CSV。
- [x] 文档一致性检查和 `git diff --check` 通过。

只有全部条件通过后，三阶段总清单才能把 P2-P0 标记为完成。

## 9. 明确不属于 P2-P0

- OpenCV 完整 RNG/RNG_MT19937 对象模型和跨实现相同随机字节序列；
- DFT/DCT、SVD/PCA、通用矩阵求解和频域 phase correlation；
- CLAHE 对象模型；
- 完整 contour hierarchy、ellipse fitting 和复杂凸多边形算法；
- corner/feature detection；
- GPU、OpenCL、IPP 或第三方数值运行时；
- 为提高覆盖率数字而增加未被真实流水线使用的长尾 API。
