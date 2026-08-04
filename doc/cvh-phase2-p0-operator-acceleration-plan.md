# cvh Phase 2-P0 算子加速实施计划（第二阶段）

Updated: 2026-08-04

Status: A0-A6 已完成；A7 本机收口已完成，等待 Linux x86-64 运行证据。

## 1. 目标

这里的“第二阶段”指 Phase 2-P0 算子完成 API 与正确性收口之后的性能阶段，
不是新一轮 API 扩张。本阶段只优化已经支持的 17 个操作族：

```text
Core
  randu, randn, transform, perspectiveTransform

Imgproc
  connectedComponents, connectedComponentsWithStats, findContours,
  boundingRect, contourArea, arcLength, approxPolyDP, convexHull,
  isContourConvex, moments, calcHist, compareHist, matchTemplate
```

本阶段要实现三个结果：

1. OpenCV upstream 继续作为输出、异常和边界语义的硬正确性合同；
2. 优先消除扫描、重复计算和逐元素高层访问造成的结构性开销；
3. 将有效 fast path 接入 header-only 公共目标、长期正确性测试和 canonical
   benchmark，不留下阶段性 target 或一次性诊断脚本。

性能对比用于量化差距，不替代正确性。“对齐 upstream”在本计划中首先表示
支持范围内的结果和行为保持一致；不通过降低精度、改变标签/轮廓顺序或缩小
公开支持面换取 benchmark 数字。

## 2. 范围边界

### 2.1 本阶段包含

- 优化现有公开 API 的 scalar/header fast path；
- 在收益明确时增加 OpenCV Universal Intrinsics（UI）kernel；
- 补充 fast path 的 forced-path、ROI、尾部和退化输入测试；
- 扩充现有 canonical benchmark 的输入形态；
- 使用同机、同线程、同参数的 upstream OpenCV 做专项比较；
- 按本文状态表实时记录实施和验收结果。

### 2.2 本阶段不包含

- 新增操作族、类型、通道或 flag；
- 引入 OpenCV 二进制运行时依赖；
- 新增公开 dispatch、RNG seed 或性能调参 API；
- 恢复已删除的阶段性 benchmark target；
- 为追求速度改变随机数统计范围、模板匹配归一化、连通域 canonical label、
  轮廓发现顺序或形状算子的数值合同；
- 没有跨平台证据时直接增加 NEON/AVX2 专用实现；
- 发布打包、版本号、变更日志等 release 工作。

## 3. 起始基线

### 3.1 代码与 upstream

- CVH 起始提交：`8360e586d8c004954a2cfd0b22ce1a1476cf9af9`；
- upstream OpenCV 参考提交：
  `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`；
- 公共实现：`cvh::headers`，`CVH_ENABLE_OPTIMIZATION=ON`；
- 对比模式：forced `OpenCVUIOnly`，单线程；
- 专项矩阵：17 个操作族、26 条同参数 case，当前全部记录为
  `public_header_scalar`。

已有 quick 结果只用于排优先级，不作为最终验收数字。正式优化前必须用
`stable` profile 冻结带 CVH/OpenCV revision、编译器、CPU、线程数、参数、
dispatch path、CSV 和 metadata 的基线。

2026-08-04 已完成 stable 基线：26/26 case 有效，采样为
`warmup=2, iters=20, repeats=5`。CVH working tree 的 dirty 内容仅为本计划文档，
OpenCV working tree 的 dirty 内容仅为 `.gitignore` 的本地 build 目录规则；两者
都不改变被测 kernel，metadata 保留了真实 dirty 标记。基线文件见
[benchmark/opencv_compare/results/README.md](../benchmark/opencv_compare/results/README.md)。

### 3.2 当前热点排序

排序同时看绝对耗时、upstream 差距和代码中的可消除开销，不按倍率单独排序。

| 优先级 | 操作族 | 当前主要成本 | 本阶段策略 |
| --- | --- | --- | --- |
| P0 | `matchTemplate` | 每个输出点重复读取模板，同时计算未使用的 correlation、平方和与平方差 | 方法专用 kernel、模板预计算、窗口平方和、UI 点积 |
| P0 | `connectedComponents*` | 逐像素 `Mat::at`、频繁 union-find 查找、`unordered_map` 重标号、stats 多次访问 | 行指针扫描、紧凑 label 表、融合重标号/统计 |
| P0 | `findContours` | 全图复制、外部背景 BFS、逐点容器访问与分配 | 连续 workspace、扫描分流、复用容量、保持发现顺序 |
| P1 | `transform` / `perspectiveTransform` | 内层循环反复读取/转换小矩阵系数 | 系数预打包、通道特化、F32 UI kernel |
| P1 | `calcHist` / `compareHist` | 每像素/每 bin 高层访问，通用分支留在热循环 | 类型/通道/方法专用循环、局部累计、UI reduction |
| P1 | `randu` / `randn` | 逐标量构造标准库 distribution，逐元素 channel 取模 | 分布器移出热循环、按通道批处理、连续/ROI 专用遍历 |
| P2 | shape primitives | 多数绝对耗时已经很小，个别 vector/stack 仍可减少分配 | 只做有稳定绝对收益的结构性清理 |

## 4. 统一实现规则

### 4.1 先优化算法和数据访问，再增加 SIMD

每个操作族按以下顺序推进：

1. 将 `Mat::at`、重复类型判断、重复系数转换和不必要分配移出热循环；
2. 建立 typed row-pointer scalar kernel，作为所有平台的可读 fallback；
3. 对稳定且占主要耗时的连续段增加 UI kernel；
4. 用明确 predicate 选择 fast path，不满足条件时回到 scalar；
5. 只有 UI 无法表达且真实机器数据证明必要时，才单独评审专用 ISA。

UI kernel 遵循
[opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md)，
不得绕过 `CVH_ENABLE_OPTIMIZATION`、内部 runtime permission、尾部处理或 ROI
步长。公共 API 和 CMake target 保持不变。

### 4.2 Dispatch 可观察性

新增 fast path 后，专项 benchmark 必须把实际路径写入 `dispatch_path`：

```text
public_header_scalar
header_fastpath
opencv_ui
```

名称表达实际执行路径，不能仅因为编译时打开 UI 就标记为 `opencv_ui`。每个
fast path 至少有一个命中 case 和一个明确回退 case。

### 4.3 数值与顺序不变量

- `transform` 和 `perspectiveTransform` 保持 F32/F64 计算、alias 和零分母语义；
- `randu`/`randn` 保持各深度饱和、上下界、零标准差和 ROI 写入合同；不要求
  与 upstream 生成相同随机序列，但统计范围和退化行为必须一致；
- connected components 保持 4/8 connectivity、背景标签、first-seen canonical
  label、stats 和 centroid 布局；
- `findContours` 保持 `RETR_LIST`/`RETR_EXTERNAL`、
  `CHAIN_APPROX_NONE`/`CHAIN_APPROX_SIMPLE`、offset 和输出顺序；
- histogram 保持 mask、range、右开区间、accumulate 和空分母语义；
- template matching 保持四个已支持 method、归一化 clamp、alias 和 F32 输出；
- shape primitives 保持点顺序、clockwise/oriented、闭合曲线和非有限值错误。

## 5. 分批实施

### A0：冻结可复现基线

目标：让后续每个优化提交都有同一套可比较证据。

工作项：

- 运行 `stable + PHASE2_P0`，确认 26/26 case 为 `OK`；
- 审核 metadata 中的 CVH/OpenCV revision、dirty 状态、Release、单线程和 CPU；
- 将审核后的 date-named Markdown、CSV、metadata 一起保存；
- 给 26 条 case 增加 `baseline_id` 或等价的可追踪标识；
- 在 canonical Core/Imgproc benchmark 中确认相同操作族可做 CVH 内部回归。

验收：基线可以在相同机器上用一条命令复现，所有 row 都有实际 dispatch
标签，不能用 quick 单次采样作为收益结论。

### A1：`matchTemplate` 主热点

第一步先去掉无条件三统计量计算：

- U8/F32 分开 typed kernel，行首一次取得 image/template/result 指针；
- `TM_CCORR*` 只计算 correlation，`TM_SQDIFF*` 由相关项与平方和组合，避免
  每个像素同时维护三套 accumulator；
- 模板平方和只计算一次；
- normalized method 使用内部 summed-square table O(1) 取得窗口平方和；
- 对常量、零能量和接近归一化边界的输入保留现有 clamp 规则。

第二步增加 UI 点积内核：

- 优先 U8C1 和 F32C1 连续行；
- 每行 SIMD 主循环加标量 tail，ROI 通过 step 正确推进；
- 小模板/小输出保留低启动成本 scalar 路径；
- 在现有 DFT/FFT 基础设施缺失的情况下，本阶段不引入一次性的频域后端。

验收目标：

- 四个 method 全部通过 unit + upstream contract；
- benchmark 中 fast path 与 fallback 均被覆盖；
- 代表 case 相对冻结 CVH 基线至少 `5x`，目标 `10x`；
- 不接受任一 method 为换取 aggregate 收益出现超过 `8%` 的稳定回退。

### A2：连通域与轮廓扫描

`connectedComponents*`：

- 改成源/标签行指针，缓存 north/current row；
- union-find 使用连续 parent 表和一次压缩，避免热循环重复 `find`；
- 用 root-indexed `std::vector<int>` 替代 `unordered_map` canonicalization；
- 在 final relabel pass 中同步累计 area、边界和坐标和，
  `connectedComponentsWithStats` 不再额外多次访问 stats Mat；
- 稀疏、稠密、checkerboard、单像素、全零、全前景分别建 case。

`findContours`：

- 用行复制和连续索引构造带边框 workspace；
- 将 exterior-background、raster discovery 和 contour tracing 的临时容量复用；
- 只在证明确保 hole/external 判定和 OpenCV 输出顺序不变后融合扫描；
- UI 只用于二值化/连续背景块跳过等无顺序副作用的前处理，不向量化
  stateful contour tracing。

验收目标：

- 4/8 connectivity 的 labels、stats、centroids 对齐 upstream；
- 两种 retrieval mode、两种 approximation method、holes 和 offset 对齐；
- `connectedComponents` 代表 case 至少 `3x`，with-stats 至少 `4x`；
- `findContours` 代表 case 至少 `2x`；
- 目标未达成时保留正确性改进，但不能声称该批性能完成。

### A3：小矩阵点变换

- 在 API 校验后一次性将矩阵系数装入固定大小本地数组；
- 对 C1-C4 和 affine/non-affine 使用编译期通道特化，移除热循环的
  `matrix.depth()`、`matrix.at` 与通道动态循环；
- F32 常用 C3->C4 transform、C2/C3 perspective 增加 UI block；
- F64 先保留优化后的 scalar，除非 benchmark 证明 UI 有稳定收益；
- in-place alias 仍先 clone，后续只有在单独证明安全时才改变策略。

验收目标：两个代表 case 相对冻结 CVH 基线至少 `4x`；奇数点数、ROI、
F32/F64、alias、接近零 `w` 都有测试。

2026-08-04 实测决策：系数预打包、源通道特化和连续点 span 已使两个代表
case 分别提升 `53.2x` 和 `62.4x`，`transform` 略快于 upstream，
`perspectiveTransform` 与 upstream 基本持平。继续增加 F32 UI block 不再是主要
瓶颈，还会引入 float 累计/除法与现有 double 计算合同的差异，因此 A3 保留
结构化 header fast path，不增加收益不足的 UI 分支。

### A4：Histogram

`calcHist`：

- U8C1、U8C3/C4 selected-channel 和 F32 分开循环；
- U8 256-bin/full-range 建立最短路径，避免每像素浮点 range 计算；
- mask、非默认 range、accumulate 保留独立 fallback；
- 使用小型局部 bin 累计并分块合并，避免每像素反复访问 `Mat`；
- SIMD 没有安全高效 scatter 时不强行使用 UI。

`compareHist`：

- 四种 method 分开循环，删除每 bin method 分支；
- correlation/intersect 等连续 reduction 在精度合同允许时使用 UI，最终仍以
  double accumulator 合并；
- 零和、常量 histogram 的结果保持 upstream 语义。

验收目标：`calcHist` U8C1 代表 case至少 `4x`，`compareHist` 256-bin case
至少 `2x`；mask/range/accumulate 不出现超过 `8%` 的稳定回退。

2026-08-04 stable 实测：`calcHist` 从冻结基线 `1.182075 ms` 降至
`0.023681 ms`，提升 `49.9x`，upstream 为 `0.015296 ms`；`compareHist`
四种方法从 `0.001314-0.001430 ms` 降至 `0.000084-0.000152 ms`，提升
`9.0x-15.6x`，已处于 upstream 的 `0.98x-1.49x` 范围。U8C4 非连续 ROI、
mask、非默认 range、非连续 histogram 累加和 compare 输入已通过专项测试；
Imgproc 193/193、upstream contract 24/24、header/ODR 和 scalar-only 已通过，
A4 验收完成。

### A5：Random fill

- distribution 对象按 depth/channel 在填充前构造，不在逐元素函数中构造；
- 内部 RNG 使用固定初始状态的 `xorshift64*` 64-bit 引擎；未新增公开
  seed/state API，输出序列仍不属于公开合同；
- 连续 Mat 按 channel 周期展开，避免每元素 `% channels`；
- 2D ROI 使用同一 typed row kernel；
- 常量区间和 `stddev == 0` 直接 fill/saturate；
- 标准库 distribution 仍可能造成跨标准库序列差异；本阶段的合同是范围、统计
  分布和零标准差行为，不承诺逐值随机序列；
- UI 仅用于后处理转换/饱和且不改变统计分布时采用。

验收目标：U8C3/F32C3 continuous 与 U8C1 ROI 的代表 case至少 `2x`；新增
range、均值/方差容差、饱和边界、零标准差和 ROI guard 测试。

2026-08-04 stable 实测已跨过性能线：U8C3/F32C3 `randu` 分别提升
`2.39x` / `2.15x`，U8C1 ROI 提升 `2.97x`；U8C3/F32C3 `randn` 分别
提升 `3.16x` / `3.28x`。范围、均值/方差、饱和、全零/混合零标准差和 ROI
guard 专项测试，以及 Core 213/213、contract 24/24、header/ODR 和
scalar-only 均已通过，A5 验收完成。

### A6：Shape primitives 收尾

只在 A1-A5 完成后处理：

- `approxPolyDP` 复用 stack/output 容量并减少重复模运算；
- `convexHull` 减少复制和临时分配；
- `arcLength`/`contourArea`/`moments` 仅在较大点集证明有绝对收益时增加 UI；
- 已经领先或接近 upstream 的 `boundingRect`、`isContourConvex` 等微算子不做
  为倍率而增加复杂度的专用实现。

验收：目标 case 至少改善 `20%` 且绝对节省可稳定测量；否则保持当前实现，
在状态表中记录“测量后不优化”。

2026-08-04 测量结论：`boundingRect`、`contourArea`、`isContourConvex` 和
`moments` 已与 upstream 持平或领先；`arcLength` 的绝对差距约 `0.0003 ms`。
`approxPolyDP` 与 `convexHull` 的两轮候选分别尝试预分配/减少模运算，以及
复用输出容量/跳过已排序输入；前者无稳定收益，后者最好约 `10%`、绝对节省
约 `0.002 ms`，均未达到 `20%` 保留门槛。候选改动已撤回，七个 shape
primitive 统一记为“测量后不优化”，不为微小收益增加长期实现复杂度。

### A7：全矩阵与跨平台收口

2026-08-04 本机收口结果：

- `scripts/ci_headers_all.sh` 通过：Core 213/213、Imgproc 193/193、
  header self-containment、ODR、install-tree consumer 和 20/20 clean UI CTest
  全部通过；
- 新建 optimization-off/scalar-only build 通过：Core 200/200，另有 13 个
  UI-only case 按预期 skip，Imgproc 193/193；
- Phase 2 upstream contract 24/24 通过；`PHASE2_P0` 最终 stable 快照
  26/26 有效，A1-A5 的冻结基线收益均保留；
- OpenCV upstream full profile 产生 370 行，369 行 `OK`，唯一
  `UNSUPPORTED` 是 upstream 没有单调用 BGR-to-NV12 encoder，与本阶段
  kernel 无关。

Canonical quick/full 都以 baseline-first 和 current-first 正逆序复跑。
quick 的 Core 失败交集为空，Imgproc 交集仅有未改动的 `blur` 和
`medianBlur`微秒级 case；full 交集为 2 个 Core 和 3 个 Imgproc
case，均属于本轮未改动的 `Mat::create`、`reduce`、pyramid、Gaussian
blur 和 resize 路径。失败集合随执行顺序大幅变化，且 A2.G 已对重复
Gaussian case 完成独立复验，因此判定为长矩阵温度/执行顺序噪声，
未发现 A1-A6 目标 kernel 的未解释回退。原始 canonical 报告保留在
本地 ignored build 目录，不引入一次性产品资产。

当前主机为 `Darwin arm64` Apple M5，本机没有 Docker、Podman、Colima
或 Lima 可用的 Linux x86-64 runtime。该运行证据不能用 compile-only
替代，因此 A7 仍保持“进行中”，但本机范围已无其他未完成项。

## 6. 正确性门禁

每个实施批次都按下面顺序执行，任一步失败就不能进入性能验收：

1. 操作族单元测试；
2. Core/Imgproc 全量测试；
3. header self-containment、ODR smoke 和 install consumer；
4. optimization-enabled 与 `CVH_ENABLE_OPTIMIZATION=OFF`；
5. forced scalar 与 forced UI 的同输入差分；
6. `test/opencv_contract` Phase 2 upstream 差分；
7. macOS ARM64 实跑；涉及 UI 后补 Linux x86-64 实跑，不能只做 compile-only。

必须补齐的专项输入：

| 类别 | 输入 |
| --- | --- |
| Layout | continuous、奇数宽度、非连续 ROI、不同 step |
| Boundary | 空/最小合法尺寸、全零、全前景、单点、模板等于图像 |
| Alias | dst 与 src/image/template 共用 data 的已支持路径 |
| Numeric | F32/F64、零分母、接近归一化 clamp、饱和上下界、非有限点 |
| Ordering | canonical labels、contour 顺序、hull clockwise、closed/open curve |
| Fallback | 不命中 fast-path predicate 的合法输入仍走 scalar 且结果一致 |

允许的容差必须来自已有 upstream contract 或以单独评审说明浮点累计顺序；
不得在性能提交中顺带放宽容差。

## 7. Benchmark 与性能门禁

### 7.1 两类 benchmark 的职责

| 模式 | 作用 | 门禁 |
| --- | --- | --- |
| CVH internal baseline/current | 防止优化目标和非目标 case 回退 | Mode A，quick 默认最大 slowdown `8%`，full `15%` |
| CVH UI vs upstream OpenCV | 量化 upstream 差距和 dispatch 归因 | Mode B，log-only；不因 upstream 更快直接失败 |

专项 upstream 命令：

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
./benchmark/opencv_compare/run_compare.sh \
  --profile stable --ops PHASE2_P0 --threads 1
```

开发循环可以使用 `quick`，批次验收必须使用 `stable`。阶段收口时再跑一次
全量 canonical Core/Imgproc benchmark 和完整 upstream comparison，确认没有
把回退转移到 Phase 2-P0 之外。

### 7.2 每批性能判定

每个优化批次必须同时满足：

- 目标 case 达到该批列出的最低改善；
- quick 与 stable 结论方向一致；
- 非目标 Phase 2-P0 case 没有超过 `8%` 的稳定回退；
- full canonical matrix 没有超过现有 `15%` 门限的未解释回退；
- report 记录 `dispatch_path`，不能把未命中的 UI build 误报成 UI kernel；
- 比较使用相同输入、参数、线程数、build type 和 upstream revision；
- 收益来自端到端公开 API，不只统计内部 helper。

upstream gap 和 CVH baseline speedup 分开报告：前者回答“还差多少”，后者回答
“本批是否真的变快”。跨操作族 geometric mean 只用于趋势，不作为唯一完成条件。

## 8. 实时状态表

状态只使用 `待执行`、`进行中`、`完成`、`测量后不优化`。代码、测试、benchmark
或门禁发生变化的提交必须同步更新本表；不提前勾选尚未通过 stable 验收的项目。

| ID | 工作项 | 状态 | 正确性 | 性能证据 |
| --- | --- | --- | --- | --- |
| A0.1 | 17 操作族、26 条 upstream case 接入 | 完成 | 26/26 quick case valid | focused runner 已可复现 |
| A0.2 | 冻结审核后的 stable 基线 | 完成 | 26/26 case valid | stable Markdown/CSV/metadata 已归档 |
| A1 | `matchTemplate` | 完成 | Imgproc 193/193；upstream contract 24/24；scalar/UI/ROI/tail/fallback 通过 | stable `184x-211x`；upstream 仅快 `1.38x-1.70x` |
| A2.1 | `connectedComponents*` | 完成 | Imgproc 193/193；upstream contract 24/24；4/8-connectivity、checkerboard、全零/全前景通过 | stable `26.6x` / `43.8x`；upstream 仅快 `2.17x` / `1.34x` |
| A2.2 | `findContours` | 完成 | Imgproc 193/193；upstream contract 24/24；两种 retrieval/approximation、hole、offset 通过 | stable `9.2x`；upstream 仅快 `1.34x` |
| A2.G | A2 full canonical 非目标回退判定 | 完成 | 326/326 case 可执行；唯一重复失败项 isolated checksum 一致 | 正/逆序失败项不稳定；Gaussian F32C3 isolated current `1.07-1.14 ms`、baseline `1.27-1.29 ms`，判定无 kernel 回退 |
| A3 | `transform` / `perspectiveTransform` | 完成 | Core 213/213；upstream contract 24/24；header+ODR+scalar-only；C1-C4、odd-count、ROI、F32/F64、alias、近零 w 通过 | stable `53.2x` / `62.4x`；upstream 比值 `1.04x` / `1.00x` |
| A4.1 | `calcHist` | 完成 | Imgproc 193/193；contract 24/24；header+ODR+scalar-only；mask/range/ROI/accumulate 通过 | stable `49.9x`；upstream 快 `1.55x` |
| A4.2 | `compareHist` | 完成 | 四方法、连续/非连续输入；同上全量门禁通过 | stable `9.0x-15.6x`；upstream 比值 `0.98x-1.49x` |
| A5 | `randu` / `randn` | 完成 | Core 213/213；contract 24/24；header+ODR+scalar-only；统计/range/ROI 通过 | stable `2.15x-3.28x` |
| A6 | shape primitives | 测量后不优化 | Imgproc 193/193；contract 24/24；候选均已撤回 | 5 项持平/领先；`approxPolyDP`/`convexHull` 候选 `<20%` |
| A7 | 全矩阵收口与跨平台复验 | 进行中 | 本机全部通过；Linux x86-64 待运行 | 最终 stable 26/26；full 369/370 有效；无目标 kernel 回退 |

每次更新在表后追加一条简短记录：

```text
YYYY-MM-DD / ID / commit-or-working-tree / correctness / benchmark / decision
```

当前记录：

```text
2026-08-04 / A0.1 / 8360e58 / 26 focused cases valid /
quick prioritization complete / begin stable baseline and A1
2026-08-04 / A0.2 / 8360e58+docs / stable 26/26 valid /
baseline report+CSV+metadata frozen / dirty scope audited as non-kernel files
2026-08-04 / A1 / working tree / Imgproc 193/193, contract 24/24,
header+ODR+scalar-only passed / stable 184x-211x, dispatch=opencv_ui /
accept A1 and start A2.1
2026-08-04 / A2.1 / working tree / Imgproc 193/193, contract 24/24,
header+ODR+scalar-only passed / stable connectedComponents 26.6x,
with-stats 43.8x; upstream gap 2.17x/1.34x / accept A2.1
2026-08-04 / A2.2 / working tree / Imgproc 193/193, contract 24/24,
header+ODR+scalar-only passed / stable findContours 9.2x; upstream gap 1.34x /
accept A2 target kernel and begin full canonical adjudication
2026-08-04 / A2.G / working tree / full canonical 326/326 executable /
15% gate: baseline-first 4 failures, current-first 2 failures; only common failure is
untouched GAUSSIANBLUR 3x3_F32C3 and its current/baseline kernel instructions match /
isolated current 1.07-1.14ms vs baseline 1.27-1.29ms with identical checksum /
adjudicate as monolithic binary/order noise, accept A2 and start A3
2026-08-04 / A3 / working tree / Core 213/213, contract 24/24,
header+ODR+scalar-only and C1-C4/odd/ROI/F32/F64/alias/near-zero-w passed /
stable transform 53.2x, perspectiveTransform 62.4x; upstream ratio 1.04x/1.00x /
accept A3 without a numerically weaker UI branch; full matrix rerun remains A7
2026-08-04 / A4 / working tree / Imgproc 193/193, contract 24/24,
header+ODR+scalar-only and mask/range/non-contiguous ROI/accumulate passed /
stable calcHist 49.9x, compareHist 9.0x-15.6x; upstream ratio 0.65x and
0.67x-1.02x / accept typed-row/local-accumulator fast paths and start A5
2026-08-04 / A5 / working tree / Core 213/213, contract 24/24,
header+ODR+scalar-only and range/statistics/saturation/zero-stddev/ROI passed /
stable randu 2.15x-2.97x, randn 3.16x-3.28x; upstream remains 1.37x-4.39x
faster / accept internal xorshift64* and hoisted-distribution typed kernels;
random sequence remains outside the public contract; start A6
2026-08-04 / A6 / working tree / existing Imgproc 193/193 and shape upstream
contract retained / stable: 5 primitives at parity or ahead; approxPolyDP and
convexHull candidate best gains below 20% with <=0.002ms absolute saving /
revert candidate edits, record all shape primitives as measured-no-optimize,
start A7
2026-08-04 / A7 / working tree / UI Core 213/213, Imgproc 193/193,
scalar-only Core 200/200 + 13 expected UI skips, Imgproc 193/193, contract 24/24,
headers+ODR+install consumers passed / final stable 26/26; upstream full 369 OK
+ 1 expected unsupported; canonical quick/full order intersections contain only
untouched kernels and order/thermal noise / accept all local gates; Linux x86-64
runtime evidence remains pending
```

## 9. 提交和回退边界

建议每个操作族保持独立提交序列：

```text
benchmark/test coverage
    -> scalar structural optimization
    -> UI kernel and dispatch
    -> stable evidence and plan update
```

- 一个提交不跨多个热点批次，便于二分和回退；
- benchmark 数据与实现提交分离时，必须准确记录被测 commit；
- 共享 helper 只有在至少两个当前操作族真正复用时才抽取；
- 某个 fast path 正确性或跨平台收益不稳定时，只回退该 dispatch 分支，保留
  scalar 结构优化和测试；
- 不保留未接入长期门禁的专项 target、脚本或阶段名称兼容层。

## 10. 完成定义

本清单按实时事实更新。勾选表示当前批次已有可复核证据；第二阶段只有在所有
条目勾选，并由 A7 重新执行最终门禁后才结束。

算子批次：

- [x] A1 `matchTemplate` 达到性能目标，正确性、dispatch 和 stable 证据完成；
- [x] A2 `connectedComponents*` / `findContours` 达到性能目标，正确性和
  canonical 噪声判定完成；
- [x] A3 `transform` / `perspectiveTransform` 达到性能目标，明确记录不增加
  数值收益不足的 UI 分支；
- [x] A4 `calcHist` / `compareHist` 达到性能目标，正确性与 stable 证据完成；
- [x] A5 `randu` / `randn` 达到性能目标，统计正确性与 stable 证据完成；
- [x] A6 每个 shape primitive 已形成“测量后不优化”的数据结论，候选改动撤回；
- [ ] A7 完成全矩阵、跨平台和最终证据收口（本机收口已完成，
  仅缺 Linux x86-64 运行证据）。

全阶段门禁：

- [x] 17 个操作族公开支持范围截至 A7 没有缩小；
- [x] A1-A7 本机最终门禁已通过：Core 213/213、Imgproc 193/193、
  header/ODR、install consumer 和 upstream contract 24/24；
- [x] 当前 fast path 的 optimization-enabled、scalar-only、forced UI
  适用路径已验证；
- [x] A0-A7 stable `PHASE2_P0` 快照均为 26/26 有效，dispatch 标签反映
  实际路径；
- [x] macOS ARM64 已有新增 UI kernel 和 scalar fallback 运行证据；
- [x] A7 重新执行单元、Core/Imgproc、header/ODR、install consumer 和
  upstream contract 最终门禁；
- [x] canonical quick/full 完成正逆序最终复跑，失败交集已按未改动
  kernel 和执行顺序/温度噪声完成裁决，无 A1-A6 目标 kernel 未解释回退；
- [ ] Linux x86-64 对新增 UI kernel 和 fallback 有运行证据；
- [x] 最终 date-named Markdown、CSV、metadata 指向同一 CVH/upstream
  revision；
- [x] 本文、coverage 和长期 benchmark 更新为当前最终事实，且不遗留一次性
  诊断资产。

完成后，长期事实分别回到 API coverage、测试、canonical benchmark 和 dated
performance report；本文保留未完成项，全部完成后由 Git history 归档。
