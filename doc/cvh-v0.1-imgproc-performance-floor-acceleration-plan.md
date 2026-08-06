# cvh v0.1 Imgproc 性能底线加速计划（第三轮性能收口）

更新时间：2026-08-04
状态：B4 测量后不优化；B5/B6 完成；B7 进行中（全量矩阵与跨平台收口）

## 1. 目的与阶段定位

本文定义 v0.1 发布前最后一轮 Imgproc 性能收口。

这是继 [Phase 2-P0 算子加速](cvh-phase2-p0-operator-acceleration-plan.md)
之后的第三轮**性能优化阶段**，不是 API 支持路线中的 “Phase 3”，不新增
算子、类型、枚举或边界语义。

本轮只处理已经纳入 v0.1 支持面的旧 Imgproc 热点：

- filter / derivative：`GaussianBlur`、`filter2D`、`sepFilter2D`、
  `boxFilter`、`Sobel`、`Scharr`、`Laplacian`；
- morphology：`erode`、`dilate`；
- edge：`Canny`；
- pyramid：`pyrDown`、`pyrUp`、`buildPyramid`；
- geometry：`warpAffine`、`warpPerspective`、`remap`；
- nonlinear：`medianBlur`、`bilateralFilter`、`stackBlur`。

Phase 2-P0 新增算子已经进入正确性与性能回归阶段，本轮不重新设计其
实现，只保留防回归门禁。

## 2. 完成目标

1. **结果对齐**：OpenCV upstream 合同继续是硬门禁，不以性能为理由
   放宽容差、边界规则、舍入行为或异常合同。
2. **用户可见延迟**：优先处理 1080p 绝对损失大，或稳定慢于 OpenCV
   `3.0x` 以上的路径。
3. **Imgproc 整体观感**：同一 full profile 下，Imgproc 相对性能几何
   平均从 `0.3701` 提升到不低于 `0.50`。
4. **全量观感**：同一 full profile 下，全套相对性能几何平均从
   `0.4808` 提升到不低于 `0.55`。
5. **热点底线**：目标 case 中 CVH 耗时不低于 `1 ms` 的 case，原则上
   不得稳定慢于 OpenCV `3.0x`；本文列出的更严格 family 目标除外。
6. **架构可持续性**：优先采用共享算法重构和 OpenCV Universal
   Intrinsics（UI），不以 direct NEON 作为默认方案。

本文的“相对性能”沿用 OpenCV compare 报告口径：

```text
relative performance = OpenCV latency / CVH latency
1.0 = 持平；小于 1.0 = CVH 更慢
```

## 3. 范围边界

### 3.1 本轮包含

- 消除全图临时缓冲、重复转换、重复边界判断和不必要的浮点中间结果；
- 建立可复用的 row buffer、ring buffer、通道展开和 interior / border
  双路径；
- 将适合的热点迁移到 OpenCV UI，并保留 scalar fallback；
- 修正 benchmark 中硬编码或含混的 dispatch 标记；
- 增补通道数、尾部、ROI、非连续输入和边界模式的差分测试；
- 在 Apple ARM64 与 Linux x86_64 上验证 UI / scalar 分派和结果一致性；
- 更新 benchmark 报告、本文状态表和完成定义。

### 3.2 本轮不包含

- 新增公开 API、数据类型、插值模式、边界模式或枚举支持；
- 修改既有数值、异常、ROI 或 inplace 合同；
- 通过放宽 tolerance、减少 benchmark case 或关闭 checksum 获得“提速”；
- 引入 OpenCV 运行时依赖；
- 用多线程掩盖单线程 kernel 差距；
- 在 UI 尚未验证到达上限前引入 direct NEON / AVX2；
- GEMM 后端优化；GEMM 是独立问题，不计入本轮 Imgproc 排期；
- 发布打包、版本号、制品和发布说明等 release 流程工作。

## 4. 冻结基线

基线报告：
[2026-08-04 OpenCV upstream performance report](../benchmark/opencv_compare/results/2026-08-04-opencv-upstream-performance.en.md)。

| 项目 | 基线 |
| --- | --- |
| CVH revision | `8360e586d8c004954a2cfd0b22ce1a1476cf9af9 + dirty` |
| OpenCV revision | `d48bf69f65444a13f8a34b8982b083c1b78fa0e8 + dirty` |
| 平台 | Apple M5 / Darwin arm64 |
| 构建 | Release，单线程 |
| CVH dispatch | forced `OpenCVUIOnly` |
| 结果 | 370 rows，369 OK，1 expected unsupported |

> 当前报告只用于确定优先级，不是最终发布证据。最终报告必须来自可复现
> 的 release-candidate revision，并记录干净工作树或完整补丁身份。

### 4.1 整体基线

| 范围 | 相对性能几何平均 | 等价描述 |
| --- | ---: | ---: |
| 全套 | `0.4808` | CVH 约慢 `2.08x` |
| Core | `0.6273` | CVH 约慢 `1.59x` |
| Imgproc | `0.3701` | CVH 约慢 `2.70x` |
| 全套（排除 GEMM） | `0.4981` | CVH 约慢 `2.01x` |
| Core（排除 GEMM） | `0.6855` | CVH 约慢 `1.46x` |
| Phase 2-P0 full | `0.7458` | 已进入回归维护 |
| Phase 2-P0 stable focused | `0.7722` | 已进入回归维护 |

### 4.2 主要热点

| family | OpenCV 约快 | 主要问题 |
| --- | ---: | --- |
| filter / derivative | `8.32x` | 全图临时量、通用浮点卷积、C3/C4 scalar |
| nonlinear | `5.56x` | scalar 邻域访问、LUT/排序网络策略未分流 |
| pyramid | `4.82x` | `pyrUp` horizontal scalar、vertical pack 标量化 |
| geometry | `3.33x` | 坐标与采样逐像素、interior 未分流 |
| reduction | `2.29x` | 非本轮主线，保留回归观察 |

代表性用户可见 case：

| 算子 / case | CVH | OpenCV | 差距 |
| --- | ---: | ---: | ---: |
| `GaussianBlur` U8C1 5x5 1080p | `11.055 ms` | `0.343 ms` | `32.27x` |
| `filter2D` U8C1 480p | `2.035 ms` | `0.121 ms` | `16.75x` |
| `filter2D` F32C1 480p | `3.702 ms` | `0.143 ms` | `25.87x` |
| `erode` / `dilate` U8 C3/C4 | — | — | 部分 case `31–36x` |
| `warpAffine` F32C4 1080p | `18.217 ms` | `1.962 ms` | 绝对损失 `16.255 ms` |
| `bilateralFilter` 1080p | `18.614 ms` | `2.751 ms` | `6.77x` |
| `Canny` 1080p | `67.168 ms` | `34.522 ms` | 绝对损失 `32.645 ms` |
| `Scharr` | `2.961 ms` | `0.185 ms` | `15.98x` |
| `Laplacian` | `2.937 ms` | `0.194 ms` | `15.14x` |

若所有目标 family 达到本文的 `1.5–3.0x` 分档底线，按现有 case 权重
静态估算，全套相对性能几何平均约为 `0.5785`，Imgproc 约为 `0.5343`。
这只是排期模型，不是实测承诺；最终只接受同环境、同 case 的 benchmark。

## 5. 当前 SIMD / UI 覆盖审计

本轮热点中目前没有直接使用 project-owned NEON intrinsic 的 kernel。
Apple ARM64 上的 SIMD 主要来自 OpenCV UI 编译到 NEON；x86_64 上由 UI
选择相应 ISA。

| family | 当前 UI 覆盖 | 当前主要空白 |
| --- | --- | --- |
| `GaussianBlur` | U8/F32 C1 separable path | C3/C4 scalar；全图 float 临时量 |
| `filter2D` | U8/F32 C1 `filter2d_c1` | C3/C4 scalar double accumulation |
| `sepFilter2D` | U8/F32 C1 `separable_c1` | C3/C4 scalar；全图 float 临时量 |
| `boxFilter` | 无有效 UI 主路径 | row sums 与 vertical accumulation 均待重构 |
| `Sobel` | 公开 fast path 仍为 scalar | 已有 `spatial_gradient_u8_c1` UI 能力未复用 |
| `Scharr` / `Laplacian` | eligible C1 经通用 `filter2d_c1` | 通用 float convolution 成本过高 |
| `erode` / `dilate` | 仅 C1 | C3/C4 完全 scalar |
| `Canny` | 无有效 UI 主路径 | 两次 Sobel、全图 magnitude、scalar direction / NMS |
| `medianBlur` | U8 k3/k5 sorting network | k5 register pressure；histogram 分流未测量 |
| `bilateralFilter` / `stackBlur` | 无 | 邻域、LUT、通道循环均为 scalar |
| `pyrDown` | horizontal 部分类型；vertical 多数类型 | U8 vertical 向量结果逐 lane pack |
| `pyrUp` | vertical | horizontal scalar |
| geometry family | 无有效 UI sampler | 坐标生成、interior sampler、F32 path 逐像素 |

审计结论：

1. benchmark 中的 `opencv_ui` 标签不能直接解释为“整个算子已 SIMD 化”；
2. 单通道 filter 有 UI，不代表 C3/C4、边界、转换和临时量已经解决；
3. 本轮先解决算法结构和共享 UI kernel，再用数据判断是否需要 direct ISA；
4. direct ISA 仍须遵循
   [UI kernel migration checklist](opencv-ui-kernel-migration-checklist.md)
   的 fallback、平台和测试要求。

## 6. 实施原则

### 6.1 正确性先于性能

- integer 输出保持既有 bit-exact 要求；
- float 输出保持已冻结的 upstream tolerance，不新增特例；
- border、anchor、ROI、submatrix、stride、inplace 和异常行为不得改变；
- 每批先通过 targeted differential，再进入 full differential；
- 发现结果差异时回退该批，不允许用 benchmark 参数掩盖。

### 6.2 先重构数据流，再扩大 SIMD

```text
减少工作量和临时量
  -> 建立 row ring 与 interior path
  -> 复用 OpenCV UI
  -> 验证 dispatch 与跨架构
  -> 证明 UI 瓶颈后再讨论 direct ISA
```

公共策略：

- separable filter 使用有界 row ring，避免整幅 float intermediate；
- U8 固定 kernel 优先评估 fixed-point，并逐 case 验证 upstream 舍入；
- C3/C4 优先将像素行视为连续 typed span，按 channel stride 生成邻域；
- interior 不重复执行 border interpolation，边界保留独立 scalar path；
- workspace 由调用级复用，禁止在每行或每像素内分配；
- SIMD tail 必须有 scalar fallback，覆盖 odd width、短行和非对齐地址。

### 6.3 upstream 参考与许可证

算法重构可参考冻结 revision 下 OpenCV 的 `smooth.dispatch.cpp`、filter
engine、derivative、morphology、Canny、pyramid 和 imgwarp 实现。

每次移植或等价改写都要在提交说明或代码注释中记录参考 revision、关键
行为和许可证来源；不得复制未确认许可的第三方实现。

## 7. Benchmark 可观测性合同

现有 compare benchmark 中部分算子将 dispatch label 硬编码为
`opencv_ui` 或 `header_fastpath`，混淆了算法选择和实际 SIMD kernel。
B0 必须先拆分：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `algorithm_path` | 算子级算法 | `gauss_separable`、`morph_rect3x3` |
| `dispatch_path` | kernel 实际分派 | `scalar`、`opencv_ui` |
| `isa_observed` | 能可靠获得时记录；否则为 `unknown` | `neon`、`avx2`、`unknown` |

规则：

- 不根据编译平台猜测 `dispatch_path`；
- `isa_observed` 不能可靠探测时明确写 `unknown`，不作为 correctness gate；
- 自定义算法名不能覆盖实际 scalar / UI dispatch；
- forced scalar 与 forced UI 报告必须证明走了不同 kernel；
- schema 调整须同步 runner、parser、文档和历史兼容测试。

## 8. B0–B7 实施计划

### B0：测量与 dispatch 可信化

任务：

- 清理 Gaussian、box、filter、sepFilter、Sobel、Canny、morphology 的
  硬编码 dispatch label；
- 引入 `algorithm_path` / `dispatch_path`，可选记录 `isa_observed`；
- 补齐 U8/F32、C1/C3/C4、480p/1080p、odd width、tail、ROI case；
- 记录 allocation / workspace 证据，识别全图临时量；
- 冻结 `stable` 与 `full` profile 的 case 列表和命令。

验收：

- 每个结果行能解释真实算法与 kernel 分派；
- forced scalar / UI 都能执行并通过 checksum；
- 基线可在同机连续复跑，波动满足现有稳定性规则。

### B1：Morphology C3/C4 快速止血

任务：

- 为 rect kernel 的 C3/C4 horizontal / vertical path 建立 UI 实现；
- 将 channel stride 纳入邻域 offset，向量处理连续 byte span；
- 分离 interior 与 border，复用行缓冲；
- 保持任意支持 kernel、anchor 和 border 的 scalar fallback。

验收：

- U8 C3/C4 代表 case 从 `31–36x` 降到不高于 OpenCV `2.0x`；
- C1 stable case 不回退超过 `5%`；
- scalar/UI、C1/C3/C4、odd width、ROI 和支持的 border 差分通过。

### B2：共享 Filter / Derivative Engine

任务：

- `GaussianBlur`、`sepFilter2D` 建立 typed row ring，移除全图 float
  intermediate；
- U8 常用 Gaussian kernel 评估 fixed-point separable path；
- 将 C3/C4 卷积迁移到 UI-friendly channel-flattened row kernel；
- `filter2D` 建立 interior UI accumulator 与独立 border path；
- `boxFilter` 使用 rolling sum / column accumulator；
- `Sobel` 复用并扩展 `spatial_gradient_u8_c1`，避免重复取样；
- `Scharr` / `Laplacian` 使用专用小 kernel，避开通用 float convolution。

验收：

| 目标 | 批次底线 |
| --- | ---: |
| `GaussianBlur` U8C1 代表 case | 相对当前 CVH 至少 `10x`，且不慢于 OpenCV `3.0x` |
| `GaussianBlur` C3/C4 代表 case | 相对当前 CVH 至少 `4x` |
| `filter2D` / `sepFilter2D` | 不慢于 OpenCV `3.0x` |
| `boxFilter` | 不慢于 OpenCV `3.0x` |
| `Sobel` / `Scharr` / `Laplacian` | 不慢于 OpenCV `3.0x` |

family 内不得只优化单一展示 case。fixed-point 路径必须证明舍入、饱和
和 border 与 upstream 对齐。

### B3：Canny 数据流融合

任务：

- 使用共享 `spatialGradient` UI kernel 一次生成 dx/dy；
- 将 magnitude 与 direction 分类改为整数友好的流式计算；
- 使用三行 ring 完成 NMS / threshold，避免完整 magnitude 与 NMS 副本；
- hysteresis 保持既有连接规则和输出确定性；
- 复用 workspace，减少大图 allocation。

验收：

- 1080p 代表 case 不慢于 OpenCV `1.5x`；
- 相对基线至少减少 `20 ms` 绝对延迟；
- L1/L2 gradient、threshold 边界、弱边连接、ROI 和窄图差分通过。

### B4：Pyramid 水平 / 垂直闭环

任务：

- 为 `pyrUp` horizontal phase 增加 UI kernel；
- 将 U8 vertical 的 vector-to-array-to-scalar pack 改为向量 narrowing /
  saturate；
- `pyrDown` / `pyrUp` 统一 typed row workspace；
- `buildPyramid` 复用层间 workspace；
- 保持奇数尺寸、边界扩展和目标尺寸规则。

验收：

- `pyrDown`、`pyrUp`、`buildPyramid` 均不慢于 OpenCV `2.5x`；
- U8/F32、C1/C3/C4、奇数宽高和最小合法尺寸差分通过；
- forced scalar 与 UI 均可独立运行。

### B5：Geometry interior sampler

任务：

- `warpAffine` 先行，按 scanline 增量生成 fixed coordinates；
- 将完全落在源图内的坐标块与 border block 分开；
- 建立 U8/F32 typed linear sampler，移除逐像素通用分支；
- 将 sampler 下沉给 `remap`，再评估 `warpPerspective`；
- 避免为整幅图预生成不必要的坐标 map。

验收：

| 目标 | 批次底线 |
| --- | ---: |
| `warpAffine` | 不慢于 OpenCV `3.0x` |
| `warpAffine` F32 | 相对当前 CVH 至少 `3x` |
| `remap` / `warpPerspective` | 不慢于 OpenCV `2.5x`，或证明已非 v0.1 主瓶颈 |

正确性覆盖 identity、translation、rotation、scale、negative coordinates、
nearest/linear、支持的 border、C1/C3/C4、U8/F32、ROI 和极小图。

### B6：Nonlinear 策略分流

任务：

- `bilateralFilter` 使用 padded row / neighbor offset，专门化 C1/C3；
- 合并或复用 spatial / color LUT，移除内层重复计算；
- 只在收益明确时使用 UI gather / table，不引入近似 exp；
- `medianBlur` 对 sorting network 与 histogram 做尺寸、kernel、channel 分流；
- 检查 k5 UI register pressure，避免短图和多通道反向回退；
- `stackBlur` 建立 rolling sum 和 channel-specialized kernel。

验收：

- `bilateralFilter`、`medianBlur`、`stackBlur` 的目标 family 均不慢于
  OpenCV `3.0x`；
- k3/k5 不出现稳定反向回退；
- 不使用近似算法、降采样或改变参数语义换取性能。

### B7：全量矩阵与跨平台收口

任务：

- 顺序执行单测、full tests、header compile、ODR smoke、install smoke；
- 执行 optimization-off、forced scalar、forced UI；
- 执行 upstream differential full matrix；
- 执行 canonical quick/full 与 OpenCV compare stable/full；
- 在 Apple ARM64 和 Linux x86_64 留存结果；
- 生成同一 release-candidate revision 的最终性能报告；
- 更新本文状态、支持矩阵和 benchmark 文档。

最终验收：

- Imgproc 相对性能几何平均 `>= 0.50`；
- 全套相对性能几何平均 `>= 0.55`；
- 所有 family 达到批次底线，或有明确、经数据批准的例外；
- Phase 2-P0 stable focused 不低于基线 `0.7722` 的 `95%`；
- 无新增 correctness、dispatch、header、ODR、install 或跨平台回归。

## 9. 正确性与回归门禁

每批至少按以下顺序验证：

```text
targeted unit / differential
  -> forced scalar
  -> forced OpenCV UI
  -> full unit tests
  -> OpenCV upstream differential
  -> header compile / ODR / install smoke
  -> quick benchmark
  -> stable benchmark
  -> full benchmark
```

硬门禁：

- `CVH_ENABLE_OPENCV_COMPARE` 路径继续保留；
- expected unsupported 只能是已有明确合同，不能因优化新增；
- checksum / output compare 必须在计时前后保持有效；
- correctness 失败会阻止该批性能结论进入“完成”；
- direct ISA 若后续获准，必须同时具备 runtime dispatch、scalar fallback、
  forced-mode 测试和至少两种架构证据。

## 10. 性能判定规则

### 10.1 ratio 与绝对延迟

- ratio 暴露小图和高倍率差距，absolute delta 决定用户可见优先级；
- `CVH < 1 ms` 的微小 case 不单独阻止发布，但不得系统性回退；
- `CVH >= 1 ms` 的目标 case 原则上执行 `3.0x` 底线；
- `Canny`、morphology 等按本文更严格目标执行。

### 10.2 稳定性与回退预算

- quick 用于开发反馈，stable 用于批次验收，full 用于阶段完成；
- 单 case 异常必须复跑，不能用几何平均掩盖严重绝对延迟；
- 批次目标 case 不得稳定回退；
- 同 family 非目标 stable case 默认不超过 `5%`；
- full profile 非目标 case 超过 `10%` 必须解释，超过 `15%` 阻止合入；
- Phase 2-P0 与 Core canonical benchmark 继续执行既有 gate。

## 11. 实时状态表

状态只使用：`待执行`、`进行中`、`完成`、`测量后不优化`。

| ID | 批次 | 当前状态 | 当前证据 / 下一步 |
| --- | --- | --- | --- |
| B0 | 测量与 dispatch 可信化 | 完成 | 75-row stable 三轮 case/dispatch 一致；整体几何平均跨轮差约 `1.6%`；后续采用三轮中位数 |
| B1 | Morphology C3/C4 | 完成 | C3/C4 提升 `33.6–34.5x`、speedup `0.6770–0.6876`；C1 改善 `16.4–34.6%`；UI/scalar 194/194、contract 24/24 通过 |
| B2 | Shared filter / derivative | 完成 | 最终 75-row 三轮中位数几何平均 `0.6062`；全部子项达标；UI/scalar imgproc 194/194、upstream contract 29/29 通过 |
| B3 | Canny 数据流融合 | 完成 | 三轮中位数 1080p `26.51 ms` 对 OpenCV `27.41 ms`；相对 B0 减少 `42.12 ms`；UI/scalar 194/194、contract 30/30 通过 |
| B4 | Pyramid 闭环 | 测量后不优化 | pyrUp 达标；pyrDown/buildPyramid 保留明确微延迟例外：stable CVH `0.084/0.020 ms`，绝对损失仅 `0.068/0.015 ms`；UI/scalar 194/194、contract 30/30 通过 |
| B5 | Geometry interior sampler | 完成 | 三轮 full：warpAffine 最差 `0.5064`、remap 最差 `0.9846`、warpPerspective 最差 `0.4359`；UI/scalar 194/194、contract 30/30、header/ODR/checksum 通过 |
| B6 | Nonlinear 策略分流 | 完成 | 三轮中位数 bilateral `0.5135`、median `0.6022`、stackBlur `1.3103`；UI/scalar 194/194、contract 30/30、header/ODR/checksum 通过 |
| B7 | 全量矩阵与跨平台收口 | 进行中 | UI/scalar/upstream 与 canonical 本机矩阵完成；Linux x86_64 本机无容器、VM、交叉编译或远端 CLI，必须由现有 `ci-x86-correctness` 在 RC 推送后补证；继续建立干净 RC 并生成 compare/focused 报告 |

### 11.1 实时更新规则

每完成一个可验证步骤，立即更新：

1. 状态表中的状态、证据和下一步；
2. 下方执行记录；
3. 实测报告链接、revision 和命令；
4. 未达标 case、原因与回退决定；
5. 完成定义中的对应 checkbox。

状态变更：

- `待执行 -> 进行中`：已经有代码或测试变更；
- `进行中 -> 完成`：正确性与本批性能门槛均通过；
- `进行中 -> 测量后不优化`：数据证明收益不足或已非 v0.1 主瓶颈；
- 只有实现完成但没有 stable benchmark，不得标记“完成”。

### 11.2 执行记录

| 日期 | 批次 | revision | 结果 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-08-04 | PLAN | working tree | 完成基线、热点和 SIMD/UI 审计；冻结 B0–B7 | 启动 B0 |
| 2026-08-04 | B0 | `6175707` + working tree | B0 已启动；核对 benchmark schema、真实 dispatch 来源与 stable/full 入口 | 实现可观测字段并补齐测试 |
| 2026-08-04 | B0 | `6175707` + working tree | 新增 `algorithm_path` / `dispatch_path` / `isa_observed`；`IMGPROC_FLOOR` UI+scalar quick 各 20 rows 通过；旧 CSV 可继续渲染；dispatch smoke 通过 | 执行 UI stable 多次复跑与 scalar correctness 复核 |
| 2026-08-04 | B0 | `6175707` + working tree | UI stable 三轮均为 75 rows，case/dispatch 完全一致，几何平均 `0.1794 / 0.1822 / 0.1798`；UI 与 optimization-off scalar smoke 通过 | 关闭 B0，启动 B1 morphology C3/C4 |
| 2026-08-04 | B1 | `6175707` + working tree | C3/C4 rect3x3 改为 channel-stride UI byte-span；vertical constant border 使用 per-channel 重复行；新增 forced scalar/UI odd-width ROI 测试 | 执行 targeted、full differential 与 stable benchmark |
| 2026-08-04 | B1 | `6175707` + working tree | `build-ci-headers-ui` targeted morphology 14/14 通过；覆盖 C1/C3/C4、奇数尺寸 ROI、replicate/constant、forced scalar/UI 与 upstream regression | 执行 optimization-off targeted 与 UI/scalar full imgproc |
| 2026-08-04 | B1 | `6175707` + working tree | `build-phase2-accel-scalar` optimization-off targeted morphology 14/14 通过；scalar fallback 与 upstream regression 未受影响 | 执行 UI/scalar full imgproc 与 OpenCV differential |
| 2026-08-04 | B1 | `6175707` + working tree | `build-ci-headers-ui/cvh_test_imgproc` 全量 194/194 通过 | 执行 optimization-off full imgproc 与 OpenCV differential |
| 2026-08-04 | B1 | `6175707` + working tree | `build-phase2-accel-scalar/cvh_test_imgproc` 全量 194/194 通过 | 执行 OpenCV differential 与 stable benchmark |
| 2026-08-04 | B1 | `6175707` + working tree | OpenCV upstream contract smoke 全量 24/24 通过 | 执行 B1 UI stable 三轮并与 B0 中位数对比 |
| 2026-08-04 | B1 | `6175707` + working tree | UI stable run 1 完成 75 rows；C3/C4 erode/dilate speedup `0.6617–0.6854`，单轮已越过 `0.50` 底线，实际 dispatch 均为 `opencv_ui` | 执行 stable run 2/3，按三轮中位数验收并检查 C1 回退 |
| 2026-08-04 | B1 | `6175707` + working tree | UI stable run 2 完成 75 rows；C3/C4 erode/dilate speedup `0.6752–0.6840`，与 run 1 结论一致 | 执行 stable run 3，汇总三轮中位数与 B0 回退 |
| 2026-08-04 | B1 | `6175707` + working tree | UI stable run 3 完成；三轮中位数显示 C3/C4 CVH time 改善 `27.5–28.1x`，OpenCV 仅快 `1.46–1.49x`；C1 的 720p/1080p 改善，但 480p 回退 `8.75–9.70%`，未过 5% gate | 去掉 replicate 路径不需要的 per-call constant-border row 分配并复测 C1 |
| 2026-08-04 | B1 | `6175707` + working tree | constant-border row 改为仅 `BORDER_CONSTANT` 构造；refined stable run 1 的 C1 480p erode/dilate 均约 `0.0352 ms`，相对 B0 单轮方向已消除回退；C3/C4 speedup 仍为 `0.6751–0.6880` | 执行 refined stable run 2/3 与 correctness 复核 |
| 2026-08-04 | B1 | `6175707` + working tree | refined stable run 2 的 C1 480p 为 `0.0366 / 0.0370 ms`，C3/C4 speedup `0.6733–0.6969`，与 run 1 一致 | 执行 refined stable run 3，汇总最终三轮中位数 |
| 2026-08-04 | B1 | `6175707` + working tree | refined stable 三轮中位数：C3/C4 CVH time 改善 `33.6–34.5x`，speedup `0.6770–0.6876`；C1 全尺寸改善 `16.4–34.6%`，无回退 | 重编译 UI/scalar tests，复核最终代码正确性后关闭 B1 |
| 2026-08-04 | B1 | `6175707` + working tree | refined 最终代码在 `build-ci-headers-ui` 重编译完成，full imgproc 194/194 通过 | 重编译并运行 optimization-off full imgproc |
| 2026-08-04 | B1 | `6175707` + working tree | 最终 optimization-off full imgproc 194/194、OpenCV contract 24/24 通过；性能与正确性 gate 全部满足 | 关闭 B1，启动 B2 shared filter / derivative |
| 2026-08-04 | B2 | `6175707` + working tree | B2 已启动 | 复核共享过滤基础设施与 B0 case 级热点，选择首个可独立验收 kernel |
| 2026-08-04 | B2 | `6175707` + working tree | 复核确认 Gaussian/sepFilter C1 UI 仍分配全图 float intermediate，C3/C4 回落 scalar；5x5 sigma=0 benchmark 可使用 `[1,4,6,4,1]` bit-exact fixed-point 两阶段卷积 | 先实现 U8 C1/C3/C4 typed five-row ring、interior UI 与 scalar border，并做 upstream bit-exact 验证 |
| 2026-08-04 | B2 | `6175707` + working tree | Gaussian 5x5 U8 使用五行 `uint16` ring、interior UI、scalar border 与单次 `(sum+128)>>8`；本地 targeted 6/6、OpenCV C1/C3/C4 × 4 border 逐字节 differential 通过 | stable 测量并按 B2 门槛优化 |
| 2026-08-04 | B2 | `6175707` + working tree | Gaussian stable run 1：U8C1 从 B0 `0.974/3.046/7.070 ms` 降至 `0.090/0.224/0.480 ms`，提升 `10.8–14.7x`；C3 `7.2x`、C4 `3.7x`；480p C1 距 3x gate 约 6%，C4 距 4x gate 约 8% | 缓存 UI mode 并展开 horizontal block，复测三轮中位数 |
| 2026-08-04 | B2 | `6175707` + working tree | Gaussian 独立参考改为 upstream 5x5 bit-exact 核；新增实际调用 sibling OpenCV 的 C1/C3/C4 × constant/replicate/reflect/reflect101 逐字节合同，并覆盖 forced scalar/UI；当前 differential 通过 | 重编译新增 forced-mode 合同并完成 refined 三轮 stable |
| 2026-08-04 | B2 | `6175707` + working tree | Gaussian refined stable run 1：U8C1 `0.084/0.217/0.465 ms`，speedup `0.332/0.389/0.420`；C3/C4 `0.226/0.295 ms`，speedup `0.391/0.400`；C4 已越过 4x 相对提升 | 去掉 replicate 无用 zero-row allocation，执行 refined run 2/3 |
| 2026-08-04 | B2 | `6175707` + working tree | boxFilter U8 3x3 新增三行 `uint16` ring、horizontal UI、exact `/9` vector normalization；stable run 1 为 `0.81–1.04x` OpenCV，C1/C3/C4 全部达标 | 扩展 F32 flattened UI 并补 upstream 合同 |
| 2026-08-04 | B2 | `6175707` + working tree | boxFilter F32 3x3 新增三行 float ring 与 flattened UI；U8/F32 × C1/C3/C4 × 4 border × forced scalar/UI upstream differential 全部通过；stable 为 `0.92–1.48x` OpenCV | boxFilter 子项关闭，转入 filter2D/sepFilter2D |
| 2026-08-04 | B2 | `6175707` + working tree | sepFilter2D 常用 binomial3 新增 U8 typed intermediate 与 F32 flattened UI；stable U8 为 `1.96–2.15x` OpenCV、F32 为 `0.89–0.99x`，全部达标；upstream U8 最大误差 1 LSB、F32 `2e-5` 内，forced scalar/UI 全矩阵通过 | 关闭 sepFilter2D 子项，处理 filter2D |
| 2026-08-04 | B2 | `6175707` + working tree | filter2D 常用 cross3 新增 direct U8/F32 channel-flattened UI；stable U8 为 `2.85–3.08x` OpenCV、F32 `1.12–1.18x`，全部达标；forced scalar/UI upstream 合同通过 | 同步真实 algorithm path，转入 derivative 专用 kernel |
| 2026-08-04 | B2 | `6175707` + working tree | U8-to-F32 3x3 derivative 新增 channel-flattened direct UI kernel；Sobel C1 为 OpenCV 的 `0.38–0.43x`、Scharr `0.40x`、Laplacian `0.45x`，均通过“不慢于 `3x`”门槛；Sobel C3/C4 因 direct path 对自身基线回退而保留原路径；本地 derivative 10/10 与既有 upstream differential 通过 | 补齐真实 algorithm path，执行 B2 最终三轮 stable 和 full correctness |
| 2026-08-04 | B2 | `6175707` + working tree | 三轮初验发现 Gaussian F32 仍使用整幅 intermediate，未满足 B2 结构任务；补充 F32 5x5 五行 ring 与 channel-flattened UI，actual OpenCV 的 C1/C3/C4 × 4 border × forced scalar/UI differential 通过；stable probe 为 OpenCV 的 `0.99–1.06x`，旧路径 `0.07–0.46x` 的缺口已消除 | 重新执行包含最终代码的三轮 stable，并完成 full correctness |
| 2026-08-04 | B2 | `6175707` + working tree | 最终代码三轮 stable 中位数：Gaussian U8 提升 `8.0–24.3x`、F32 提升 `4.1–23.8x`；box `3.7–13.2x`；filter2D `16.7–72.9x`；sepFilter2D `7.5–24.0x`；Scharr/Laplacian `9.3/10.3x`；75-row 几何平均达 `0.6062`；所有 B2 case 均通过各自速度底线 | 执行 UI/scalar full imgproc 与全量 upstream contract，正确性通过后关闭 B2 |
| 2026-08-04 | B2 | `6175707` + working tree | 最终 UI full imgproc 194/194、optimization-off scalar full imgproc 194/194、actual OpenCV contract 29/29 通过；B2 性能与正确性硬门禁全部关闭 | 启动 B3 Canny 数据流融合 |
| 2026-08-04 | B3 | `6175707` + working tree | B3 已启动 | 审计 gradient、magnitude、NMS、hysteresis 数据流与 allocation，建立可独立验收的首个改动 |
| 2026-08-04 | B3 | `6175707` + working tree | aperture3 将两次独立 Sobel 改为一次共享 `spatial_gradient_u8_c1` UI；Canny targeted 6/6 通过；1080p stable probe 从 B0 中位数 `68.62 ms` 降至 `49.48 ms`，减少 `19.14 ms`，但仍未满足 `20 ms` 与 `1.5x` 双门槛 | 融合 magnitude/NMS 状态并移除逐像素方向除法 |
| 2026-08-04 | B3 | `6175707` + working tree | NMS 改为直接生成 padded weak/strong state map，hysteresis 使用无边界分支的线性邻居 offset；移除整幅 NMS copy 与方向除法；targeted 6/6、actual OpenCV aperture3/5 L1/L2 bit-exact 合同通过；1080p probe `28.72 ms`，已越过双门槛 | 将 full-frame magnitude 改为三行 ring 后复测 |
| 2026-08-04 | B3 | `6175707` + working tree | magnitude/NMS 改为三行 ring，移除完整 magnitude workspace；targeted 6/6 通过；stable probe 480p/720p/1080p 为 `3.85/11.67/26.51 ms`，对应 OpenCV 的 `1.03/1.03/1.04x`，1080p 相对 B0 减少 `42.11 ms` | 重编译 actual OpenCV contract，执行最终三轮 stable 与 UI/scalar full correctness |
| 2026-08-04 | B3 | `6175707` + working tree | 最终三轮中位数 480p/720p/1080p 为 `3.87/11.68/26.51 ms`，对应 OpenCV 的 `1.03/1.03/1.03x`；1080p 相对 B0 减少 `42.12 ms`；UI/scalar full imgproc 194/194、actual OpenCV contract 30/30 通过 | 关闭 B3，启动 B4 pyramid 闭环 |
| 2026-08-04 | B4 | `6175707` + working tree | B4 已启动 | 审计现有 pyramid UI 覆盖、逐 lane pack 和 layer workspace，选择首个改动 |
| 2026-08-04 | B4 | `6175707` + working tree | U8 vertical 将 int32 vector 落栈逐 lane saturate 改为 UI narrowing/rshift pack；pyrUp stable 从约 `0.167 ms` 降至 `0.136 ms`，但三项尚未全部达标 | 专用化 pyrUp horizontal interior，并复核 pyrDown vertical 组织 |
| 2026-08-04 | B4 | `6175707` + working tree | pyrUp interior 按 expanded parity 化简：偶数位三 tap、奇数位两 tap，移除五 tap index/branch 循环；再采用 upstream 同构的 u16 vertical pack；stable pyrUp 降至约 `0.052–0.067 ms`，full `0.232 ms` 对 OpenCV `0.179 ms`，已通过 `2.5x` gate | 继续处理 pyrDown/buildPyramid |
| 2026-08-04 | B4 | `6175707` + working tree | `buildPyramid` level 0 改为与 upstream 一致的 Mat header alias，去掉不必要的全图 clone，并增加 alias 合同；index/TLS cache 与 full-blur-then-decimate 探索均未产生稳定收益，已撤回；pyrDown stable/full 仍约为 OpenCV `0.19x`，buildPyramid 约 `0.30x`，但三者 CVH stable 均 `<0.1 ms`、full 均 `<0.4 ms` | 完成 correctness 与三轮数据；若仍未过 ratio，按微延迟规则记录明确例外而不引入复杂 direct ISA |
| 2026-08-04 | B4 | `6175707` + working tree | 最终三轮中位数：pyrDown `0.0837 ms` 对 `0.0160 ms`，buildPyramid `0.0204 ms` 对 `0.0057 ms`，绝对差分别仅 `0.0676/0.0147 ms`；pyrUp `0.0590 ms` 对 `0.0394 ms`，相对 B0 提升 `3.72x` 并达标；full profile 三者 CVH 也均 `<0.4 ms`；UI/scalar 194/194、actual OpenCV contract 30/30 通过 | 根据 10.1 的 `<1 ms` 规则，将未过 ratio 的两项记录为明确微延迟例外，关闭 B4 优化，启动 B5 |
| 2026-08-04 | B5 | `6175707` + working tree | B5 已启动 | 审计 stable/full geometry case 与现有 sampler 分支 |
| 2026-08-04 | B5 | `6175707` + working tree | 新增保持 double 权重语义的 F32 typed interior bilinear sampler，边界继续走通用慢路径；WarpAffine targeted 8/8、header compile、ODR smoke 通过；full probe 中 F32C3 `8.019 -> 1.663 ms`（`4.82x`）、F32C4 `10.557 -> 1.746 ms`（`6.05x`），C1 `3.155 -> 1.393 ms`（`2.27x`） | 保留首轮 fast path；将 channel dispatch 移出逐像素循环并实现 scanline 坐标增量，继续压低 C1 |
| 2026-08-04 | B5 | `6175707` + working tree | F32 translation 将坐标 floor 降为每行一次，并把 leading border / interior / trailing border 分段；full probe F32C1/C3/C4 为 `0.897/1.212/1.348 ms`，相对 B4 分别提升 `3.52x/6.62x/7.83x`，对 OpenCV 为 `1.94x/1.85x/1.90x` | F32 性能子目标完成；复用分段 scanline 到 U8 fixed sampler，并补齐 actual upstream differential |
| 2026-08-04 | B5 | `6175707` + working tree | U8 translation 将 fixed coordinate/fraction 降为每行一次并直接展开连续 interior，1080p C1 `12.232 -> 0.440 ms`；U8C3/C4 为 `0.213/0.317 ms`；actual OpenCV contract 新增 U8/F32 translation，U8 bit-exact、F32 `1e-5` 通过 | U8/F32 warpAffine 性能与 translation 正确性子目标完成；优化 remap 连续 coordinate block |
| 2026-08-04 | B5 | `6175707` + working tree | shared fixed sampler 识别连续 coordinate/fraction block；fixed remap 改为 64-pixel block，1080p float/fixed remap 为 `4.073/2.320 ms` 对 OpenCV `5.229/4.875 ms`；warpPerspective 保持 `0.437–0.469` ratio，符合 `2.5x` 底线 | 修正 Geometry benchmark algorithm/dispatch 可观测性；执行 B5 完整 correctness 与三轮 full 验收 |
| 2026-08-04 | B5 | `6175707` + working tree | 三轮 full 中位数：warpAffine 最差 F32C1 `0.902 ms` 对 `0.457 ms`（`0.5064`），remap 最差 `0.9846`，warpPerspective 最差 `0.4359`；algorithm path 为 scanline/block，dispatch 如实为 scalar；UI/scalar Imgproc 194/194、actual upstream 30/30、header compile、ODR、UI/scalar benchmark checksum 全通过 | B5 完成，启动 B6 nonlinear 策略分流 |
| 2026-08-04 | B6 | `6175707` + working tree | B6 已启动 | 从 B5 final full 三轮数据审计 bilateral、median、stack 的最新热点和现有实现 |
| 2026-08-04 | B6 | `6175707` + working tree | bilateral 将二维 border index/分支移出内层，建立调用级 padded source 与相对 neighbor offset，并专门化 U8 C1/C3；targeted unit 通过，full probe `10.546 -> 1.682 ms`（`6.27x`），对 OpenCV ratio `0.5138` | 执行 C3 actual upstream contract；通过后关闭 bilateral 子项，处理 median k5 |
| 2026-08-04 | B6 | `6175707` + working tree | bilateral C3 actual upstream contract 通过；median 测得 histogram scalar `8.177 ms`、UI sorting network `1.441 ms`，因此保留 UI 策略；将每行 scalar `nth_element` tail 改为重叠 UI tail + scalar selection network，降至 `0.415 ms`，ratio `0.6120` | bilateral/median 子项完成；进入 stackBlur |
| 2026-08-04 | B6 | `6175707` + working tree | stackBlur 单独收窄 accumulator 无收益（`0.513 -> 0.520 ms`）；进一步为 U8 5x5 建 typed horizontal/vertical direct span，保留其他 kernel rolling fallback，并新增 5x5 naive bit-exact case；降至 `0.090 ms` 对 OpenCV `0.119 ms`（CVH `1.32x`） | 性能子目标全部完成；修正 nonlinear algorithm/dispatch 可观测性并执行完整验收 |
| 2026-08-04 | B6 | `6175707` + working tree | 三轮 full 中位数 bilateral `1.686 ms` 对 `0.866 ms`（`0.5135`）、median `0.415 ms` 对 `0.250 ms`（`0.6022`）、stackBlur `0.091 ms` 对 `0.119 ms`（`1.3103`）；UI/scalar Imgproc 194/194、actual upstream 30/30、header compile、ODR、UI/scalar benchmark checksum 全通过 | B6 完成，启动 B7 全量矩阵与跨平台收口 |
| 2026-08-04 | B7 | `6175707` + working tree | B7 已启动 | 执行全测试、optimization-off、install smoke、canonical quick/full、OpenCV compare stable/full 与 Phase 2-P0 回归；审计 Linux x86_64 可用执行环境 |
| 2026-08-04 | B7 | `6175707` + working tree | 首次正式 `ci_headers_all.sh` 在 installed-header 12 项 smoke 的 `cvh_resize_dispatch_smoke` 失败（其余 11/12 通过）；定位为 B2 Gaussian 5x5 algorithm path 已变为 `gauss5x5_fixedpoint`，smoke 仍断言旧 `gauss_separable`；同步断言后单独 smoke 通过 | 从头复跑正式 UI header-only gate，确认安装/CTest/报告全链路 |
| 2026-08-04 | B7 | `6175707` + working tree | 第二次正式 gate：installed-header 12/12、CTest 20/20、Core 213/213、Imgproc 194/194 全通过；报告清单仍期望 Imgproc 193，因 B1 新增 morphology 用例后未同步而退出 1；arm64/x86_64 expectation 均更新为 194 | 第三次从头复跑正式 gate，取得完整退出码 0 证据 |
| 2026-08-04 | B7 | `6175707` + working tree | 第三次正式 `CVH_CI_PARALLEL=4 ./scripts/ci_headers_all.sh` 完整退出 0：installed-header contract 12/12、header compile/ODR、CTest 20/20、Core 213/213、Imgproc 194/194 全通过；Apple ARM64 UI-on 正式门禁关闭 | 执行干净 optimization-off 全量构建与测试 |
| 2026-08-04 | B7 | `6175707` + working tree | 全新 `build-b7-optimization-off` 明确配置 `CVH_ENABLE_OPTIMIZATION=OFF`；构建完成，CTest 18/18、Core 213（200 pass、13 个 SIMD-only 按设计 skip）、Imgproc 194/194，通过 optimization-disabled smoke、header compile 与 ODR | 重建并执行 upstream differential full matrix |
| 2026-08-04 | B7 | `6175707` + working tree | 全新 `build-b7-opencv-contract` 链接 upstream OpenCV 4.14.0，重建 backend 与 smoke；full matrix 30/30 通过 | 执行 canonical Core/Imgproc quick 与 full benchmark |
| 2026-08-04 | B7 | `6175707` + working tree | canonical quick 完成：Core 416、Imgproc 67、cvtColor 64、resize 36 行全部 `OK`；canonical full 完成：Core 1007/1007 `OK`，Imgproc 326 `OK` + 1 个既有 `INTER_CUBIC_OUTSIDE_ACCEPTED_CONTRACT` expected unsupported；checksum 均有效 | 执行 baseline/candidate stable 回归并审计异常 |
| 2026-08-04 | B7 | `6175707` + working tree | canonical internal runner 单向 quick 报 Core 13/416 超 8%；stable baseline-first 报 Core 186/716、Imgproc 49/204，但反向后分别降到 69、25，几何平均 Core 从 `0.9705` 反转为 `1.0096`、Imgproc 从 `1.6177` 变为 `1.7001` | 用同二进制跨轮差验证是否为执行顺序/机器状态偏差 |
| 2026-08-04 | B7 | `6175707` + working tree | 同一个 candidate 二进制的两份 stable 结果自身相差 Core `1.3050x`、Imgproc `1.0913x`，且 Imgproc 仍有 5 个 >8% 假告警（stackBlur 同二进制波动约 `2.1x`）；证明双进程 runner 当前不满足逐 case 发布判定的稳定性前提。本轮不据此回退代码，保留原始 forward/reverse/balanced 结果，最终采用同进程 OpenCV compare 三轮中位数及 focused gate | 审计并执行 Linux x86_64 环境；随后生成干净 RC compare 报告 |
| 2026-08-04 | B7 | `6175707` + working tree | 当前宿主为 Darwin arm64；Docker、Podman、Colima、Lima、nerdctl、Linux x86_64 cross compiler、QEMU、Multipass/Tart 与 `gh` 均不可用。仓库已有 Ubuntu `ci-x86-correctness.yml` 和 `scripts/ci_x86_correctness.sh`，但未推送的 working tree 无法取得同 revision Linux 证据 | 不擅自 push；先建立干净本地 RC、完成最终报告，Linux x86_64 明确保留为远端 CI 待补项 |
| 2026-08-04 | B7 | `9cad23c` | 建立首个干净代码 RC；Phase 2-P0 stable focused 三轮均 26/26 `OK`，几何平均 `0.7696 / 0.7657 / 0.7723`，中位数 `0.7696 >= 0.7336`；无筛选 stable 311/311 `OK`，整体/Core/Imgproc 为 `0.7411 / 0.6464 / 0.8490` | 运行 full 并生成最终报告 |
| 2026-08-04 | B7 | `9cad23c` | 首次 full 369 `OK` + 1 expected unsupported，整体/Core/Imgproc 为 `0.7321 / 0.6427 / 0.8323`，性能达标；但 runner 在写入 untracked CSV 后才读取 git 状态，把原本干净的 RC 错标为 dirty，报告不能作为最终发布证据 | 在输出创建前冻结 source identity，提交新 RC 后完整重跑 full；不手改元数据 |

### 11.3 B0 当前证据

已验证命令：

```bash
cmake --build build-opencv-compare \
  --target cvh_benchmark_opencv_compare_ui -j
cmake --build build-ci-headers-ui \
  --target cvh_resize_dispatch_smoke -j
ctest --test-dir build-ci-headers-ui \
  -R '^cvh_resize_dispatch_smoke$' --output-on-failure
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
CVH_COMPARE_BUILD_DIR=build-opencv-compare \
benchmark/opencv_compare/run_compare.sh \
  --profile quick --impls ui,scalar --ops IMGPROC_FLOOR \
  --warmup 0 --iters 1 --repeats 1 --threads 1
```

Quick 观测结果：

| 算子 | `cvh_ui` | `cvh_scalar` | 算法 |
| --- | --- | --- | --- |
| `GaussianBlur` | `opencv_ui` | `scalar` | `gauss_separable` |
| `boxFilter` | `scalar` | `scalar` | `box3x3` |
| `filter2D` | `opencv_ui` | `scalar` | `generic_filter2d` |
| `sepFilter2D` | `opencv_ui` | `scalar` | `separable_filter2d` |
| `Sobel` | `scalar` | `scalar` | `derivative_convolve` |
| `Canny` | `scalar` | `scalar` | `canny_fullframe_nms` |
| `erode` / `dilate` C1 | `opencv_ui` | `scalar` | `morph_rect3x3` |

`isa_observed` 当前均为 `unknown`：UI 层未暴露可靠的底层 ISA 观测，本轮
不根据 Apple ARM64 主机反推 `neon`。

源代码 workspace 审计：

| family | 当前主要 workspace / allocation |
| --- | --- |
| Gaussian / separable | 整幅 `rows * cols * channels` float intermediate |
| box | 全行和 vertical accumulator；无 UI 主路径 |
| filter2D | 全宽 x-offset 与全高 y-index 表 |
| morphology | 整幅 U8 horizontal intermediate |
| Canny | 完整 magnitude、NMS copy、edge map 和 hysteresis stack |

冻结的 focused 命令为 `--ops IMGPROC_FLOOR`；批次验收使用 `stable`，阶段
完成使用 `full`。生成物写入 build 目录，只有经复核的 date-named snapshot
进入 `benchmark/opencv_compare/results/`。

Stable 三轮中，case-level ratio spread 中位数为 `5.7%`；7 个 case 超过
`15%`，其中 2 个超过 `25%`。这些 case 集合与 dispatch 均一致，整体几何
平均跨轮差约 `1.6%`。因此 B1–B6 的 before/after 判定固定使用三轮中位数；
单轮异常不能单独证明回退或收益。

### 11.4 B1 当前证据

UI targeted correctness 已通过：

```bash
cmake --build build-ci-headers-ui --target cvh_test_imgproc -j4
build-ci-headers-ui/cvh_test_imgproc \
  --gtest_filter='MorphologyTest.*:MorphologyDerivativesUpstreamTest.*'
```

UI 与 optimization-off 构建结果均为 14/14 通过。新增用例在 C3/C4、
odd-width ROI、`BORDER_REPLICATE` 和 `BORDER_CONSTANT` 下逐项比较
forced scalar 与 forced UI，并校验实际 dispatch tag；full imgproc、OpenCV
differential 和 stable benchmark 尚未完成，因此 B1 保持“进行中”。

UI 与 optimization-off 构建的 full imgproc 结果均为 194/194 通过；
OpenCV upstream contract smoke 结果为 24/24 通过。B1 正确性验收已完成，
stable run 1 已完成 75 rows：C3/C4 erode/dilate speedup 为
`0.6617–0.6854`，实际 dispatch 均为 `opencv_ui`；仍需 run 2/3 后用
三轮中位数确认门槛与 C1 回退。

Stable run 2 同样完成 75 rows，C3/C4 speedup 为 `0.6752–0.6840`；
两轮均已越过 `0.50` 底线。Run 3 后的三轮中位数如下：

| case | B0 CVH | B1 CVH | CVH 提升 | B1 speedup |
| --- | ---: | ---: | ---: | ---: |
| erode U8C3 480p | `3.1112 ms` | `0.1116 ms` | `27.89x` | `0.6728` |
| dilate U8C3 480p | `3.0783 ms` | `0.1118 ms` | `27.54x` | `0.6773` |
| erode U8C4 480p | `4.0768 ms` | `0.1449 ms` | `28.13x` | `0.6800` |
| dilate U8C4 480p | `4.0710 ms` | `0.1448 ms` | `28.12x` | `0.6836` |

C3/C4 已通过 `speedup >= 0.50` 门槛；但 480p C1 erode/dilate 相对 B0
分别回退 `9.70% / 8.75%`，尚未通过 `5%` 非目标回退 gate。初步定位为
replicate benchmark 也无条件构造 constant-border row；先消除此 per-call 开销
再复测，B1 暂不关闭。

优化为仅在 `BORDER_CONSTANT` 下构造 border row 后，refined stable run 1
的 C1 480p erode/dilate 均约 `0.0352 ms`，已低于 B0 三轮中位数
`0.0426 / 0.0430 ms`；C3/C4 speedup 仍为 `0.6751–0.6880`。继续完成
refined run 2/3，避免用单轮结果关闭 gate。

Refined stable 三轮最终中位数：C3/C4 CVH time 相对 B0 改善
`33.6–34.5x`，speedup 为 `0.6770–0.6876`；C1 的 480p/720p/1080p
CVH time 全部改善 `16.4–34.6%`，没有超过 5% 的回退。性能 gate 已通过，
最终代码在 UI 与 optimization-off 构建中均重编译完成，full imgproc 各
194/194 通过，OpenCV contract 24/24 通过。B1 已关闭。

## 12. 提交与回退边界

建议按 B0 benchmark、B1 morphology、B2 shared filter、B3 Canny、
B4 pyramid、B5 geometry、B6 nonlinear、B7 report/docs 分开提交。
B2 和 B6 可继续按算子拆分。

每个 kernel 提交应具备：

- 可独立回退；
- 对应 targeted tests；
- forced scalar / UI 证据；
- before / after stable 数据；
- 不混入无关格式化与重构。

## 13. 完成定义

- [x] B0 提供可信的 `algorithm_path` / `dispatch_path`，不再用硬编码 label
      冒充实际 SIMD；
- [x] B1 morphology C3/C4 达到 `2.0x` 底线且 C1 无显著回退；
- [x] B2 filter / derivative family 达到各自门槛；
- [x] B3 Canny 1080p 达到 `1.5x` 底线并减少至少 `20 ms`；
- [x] B4 pyrUp 达到 `2.5x` 底线；pyrDown/buildPyramid 按 `<1 ms`
      规则记录了绝对损失 `<0.07 ms` 的明确微延迟例外；
- [x] B5 geometry 达到门槛或留有经测量批准的例外；
- [x] B6 nonlinear 达到门槛或留有经测量批准的例外；
- [x] full tests、header compile、ODR、install、optimization-off 全部通过；
- [ ] forced scalar 与 forced UI 都通过 correctness 和 checksum；
- [x] OpenCV upstream differential full matrix 无新增失败；
- [ ] Apple ARM64 与 Linux x86_64 均有可追溯证据；
- [ ] Imgproc 相对性能几何平均达到 `>= 0.50`；
- [ ] 全套相对性能几何平均达到 `>= 0.55`；
- [ ] Phase 2-P0 与 canonical Core/Imgproc benchmark 无超预算回退；
- [ ] 最终报告来自同一 release-candidate revision，环境与命令完整；
- [ ] 本文状态、执行记录、支持矩阵和 benchmark 文档已同步。

## 14. 阶段关闭后的长期保留项

阶段关闭后保留 correctness 与 upstream differential、forced scalar/UI
测试、canonical benchmark、stable/full OpenCV compare profile、最终报告、
dispatch 可观测性、UI migration checklist 和跨平台 fallback。

阶段名称、一次性调查脚本或未接入长期 gate 的专项 benchmark，不因本文
自动获得长期保留资格；关闭时按
[v0.1 release closure plan](cvh-v0.1-release-closure-plan.md)
的代码与文档收口规则重新审计。
