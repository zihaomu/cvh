# cvh v0.1 Imgproc 性能底线加速计划（第三轮性能收口）

更新时间：2026-08-04
状态：方案已冻结，B0 待执行

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
| B0 | 测量与 dispatch 可信化 | 待执行 | 修复硬编码 label，冻结 stable/full profile |
| B1 | Morphology C3/C4 | 待执行 | 基线已识别 `31–36x` case |
| B2 | Shared filter / derivative | 待执行 | Gaussian、filter2D、Scharr、Laplacian 为最高倍率热点 |
| B3 | Canny 数据流融合 | 待执行 | 1080p 绝对损失约 `32.645 ms` |
| B4 | Pyramid 闭环 | 待执行 | `pyrUp` horizontal 与 U8 vertical pack 是明确空白 |
| B5 | Geometry interior sampler | 待执行 | `warpAffine` F32C4 绝对损失约 `16.255 ms` |
| B6 | Nonlinear 策略分流 | 待执行 | bilateral、median、stack 按实测逐项进入 |
| B7 | 全量矩阵与跨平台收口 | 待执行 | 等待 B1–B6 完成 |

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

- [ ] B0 提供可信的 `algorithm_path` / `dispatch_path`，不再用硬编码 label
      冒充实际 SIMD；
- [ ] B1 morphology C3/C4 达到 `2.0x` 底线且 C1 无显著回退；
- [ ] B2 filter / derivative family 达到各自门槛；
- [ ] B3 Canny 1080p 达到 `1.5x` 底线并减少至少 `20 ms`；
- [ ] B4 pyramid family 达到 `2.5x` 底线；
- [ ] B5 geometry 达到门槛或留有经测量批准的例外；
- [ ] B6 nonlinear 达到门槛或留有经测量批准的例外；
- [ ] full tests、header compile、ODR、install、optimization-off 全部通过；
- [ ] forced scalar 与 forced UI 都通过 correctness 和 checksum；
- [ ] OpenCV upstream differential full matrix 无新增失败；
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
