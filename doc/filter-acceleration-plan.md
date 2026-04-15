# Filter Acceleration Plan

- 文档版本：v0.1
- 更新时间：2026-04-14
- 目标：在保持 OpenCV 风格接口的前提下，实现
  - 3x3 常用滤波高性能路径（按 kernel 类型专项优化）
  - 通用 `filter2D` 路径（兼容任意用户 kernel）

## 1. 问题定义

当前 `imgproc` 滤波链路以 fallback 为主，重点缺口是：

1. 缺少统一通用滤波引擎（任意 kernel 无统一入口）
2. 缺少 3x3 常用场景的专项加速路径
3. 缺少可持续性能回归门禁（基准、dispatch 命中可观测）

## 2. 目标与非目标

### 2.1 目标

1. 新增统一 `filter2D` 引擎，支持任意 kernel 类型（优先 float kernel）
2. 针对 3x3 常用 kernel 提供专项快速实现（高频路径优先）
3. 建立稳定 benchmark + regression gate，避免“优化后不可验证”

### 2.2 非目标

1. 不做 GPU/OpenCL/CUDA 路径
2. 不做 FFT 大核卷积优化
3. 不做自动算子融合（fusion）

## 3. 实现策略（推荐方案）

采用“双轨统一分发”：

1. 3x3 专项路径：识别常用 kernel 类型并进入专门内核
2. 通用路径：未命中专项时统一进入 generic `filter2D`

分发优先级：

1. `3x3-specialized`
2. `separable`（可分离 kernel）
3. `generic`

调用关系（ASCII）：

```text
filter2D API
  |
  +--> validate/normalize args
  |
  +--> classify kernel
         |
         +--> 3x3-specialized ----> fast kernel
         |
         +--> separable ----------> 1D x + 1D y
         |
         +--> generic ------------> reference/high-perf generic
```

## 4. 3x3 专项范围

首批专项 kernel：

1. `box3x3`
2. `gaussian3x3`
3. `sobel3x3_x`
4. `sobel3x3_y`
5. `laplacian3x3`
6. `generic3x3`（任意 3x3，做手工展开）

首批类型与通道：

1. depth：`CV_8U`、`CV_16S`、`CV_32F`
2. channels：`C1`、`C3`、`C4`

## 5. 通用滤波（任意 kernel）要求

1. 支持任意 `kx * ky` kernel（奇数优先，偶数按兼容规则处理）
2. 支持 `anchor/delta/borderType`
3. 支持非连续内存（ROI/submat）
4. 对不支持组合显式 fallback 或抛错，不允许 silent wrong result

## 6. 阶段计划

### P0：基线与门禁

交付：

1. 新增 `imgproc` filter benchmark（C1/C3/C4 + 常见分辨率 + ROI）
2. benchmark 输出增加 `dispatch_path` 字段
3. regression 脚本接入 CI（slowdown 阈值）

DoD：

1. 性能结果可复现
2. PR 能判断是否退化

### P1：通用 `filter2D` 引擎

交付：

1. 新增 `filter2D` API 与 backend 分发
2. 通用 kernel 计算路径（先 correctness 优先）
3. 边界与异常路径完备（参数校验、fallback 规则）

DoD：

1. 对齐核心行为（输入合法/非法、边界类型）
2. ROI/submat 路径正确

### P2：3x3 专项内核

交付：

1. 实现首批 3x3 kernel 专项快路径
2. `generic3x3` 展开路径
3. 按 `depth+channel` 做最小矩阵覆盖

DoD：

1. 3x3 高频场景性能显著优于 generic
2. 结果与通用路径一致（允许浮点容差）

### P3：并行与 SIMD

交付：

1. 行块并行（OpenMP）
2. C1/C4 优先 SIMD；C3 做 interleaved 专项策略
3. 统一 dispatch 命中标记

DoD：

1. 多核机器吞吐有稳定收益
2. `xsimd-only` / fallback 边界可解释

### P4：收口与文档

交付：

1. 支持矩阵文档（type/channel/layout）
2. 已知限制与非目标明确化
3. benchmark 基线固化

DoD：

1. 新人可按文档复现结果并继续迭代
2. 性能/正确性/边界策略三者一致

## 7. 验证矩阵

shape：

1. `720x1280`
2. `1080x1920`
3. `8x256x384`（扩展维）

layout：

1. continuous
2. ROI/non-continuous

组合：

1. depth：`CV_8U`/`CV_16S`/`CV_32F`
2. channel：`1/3/4`
3. kernel：专项 3x3 + generic 3x3 + generic 5x5/7x7

## 8. 性能目标（首版）

相对当前 fallback 目标：

1. 3x3 `C1`：`>= 2.5x`
2. 3x3 `C3`：`>= 1.8x`
3. 3x3 `C4`：`>= 2.2x`
4. 通用 kernel：不低于当前 fallback（至少持平）

## 9. 错误与回退策略

1. kernel 为空/尺寸非法/anchor 越界：抛 `StsBadArg`
2. 不支持的 `ddepth/type`：显式 fallback 或抛 `StsNotImplemented`
3. `xsimd-only` 且无 SIMD 覆盖：抛 `StsNotImplemented`
4. 不允许 silent fallback 导致行为不可观测

## 10. 风险与缓解

1. 风险：3x3 kernel 识别误判  
   缓解：严格模式匹配 + epsilon 策略 + 单测覆盖
2. 风险：C3 interleaved 性能不稳  
   缓解：专项内核 + 单独 benchmark 分桶
3. 风险：优化后可维护性下降  
   缓解：统一 dispatch 层 + 文档化 + 回归门禁

## 11. 里程碑（估算）

1. P0：human ~1 天 / CC ~20-40 分钟
2. P1：human ~2-3 天 / CC ~1-2 小时
3. P2：human ~3-5 天 / CC ~2-4 小时
4. P3：human ~2-3 天 / CC ~1-2 小时
5. P4：human ~1 天 / CC ~20-40 分钟

## 12. 当前执行顺序（建议）

1. 先完成 P0（先把可见性立住）
2. 再做 P1（先通用，再专项）
3. 再做 P2/P3（拿核心性能）
4. 最后 P4 收口
