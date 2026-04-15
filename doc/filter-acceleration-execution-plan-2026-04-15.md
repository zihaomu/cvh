# Filter Acceleration Execution Plan (2026-04-15)

- 状态基线：`boxFilter/GaussianBlur` backend 仍是 fallback 透传，`filter2D` 尚未实现。
- 目标周期：7~10 天形成可回归、可解释、可扩展的 filter 加速闭环。
- 执行原则：先可观测，再优化；先高频，再泛化；任何加速都必须附带正确性回归与 benchmark 证据。

## 1. 本轮范围

### 1.1 In Scope

1. `boxFilter`（优先 `CV_8U`, `C1/C3/C4`）
2. `GaussianBlur`（优先 odd `ksize`，`3x3/5x5`）
3. 内部通用滤波引擎（先内部入口，不急于开放 API）
4. imgproc filter benchmark + regression gate

### 1.2 Out of Scope

1. GPU/OpenCL/CUDA
2. FFT 大核优化
3. 任意类型全覆盖（首轮不追 `CV_64F`）

## 2. 分阶段执行

## P0：可观测性与基线（Day 1）

交付：

1. 新增 `benchmark/imgproc_filter_benchmark.cpp`
2. 新增 `scripts/check_imgproc_filter_benchmark_regression.py`
3. CSV 输出字段固定：
   - `op,kernel,depth,channels,shape,layout,border,dispatch_path,ms_per_iter`
4. 产出 baseline：`benchmark/baseline_filter_quick.csv`

DoD：

1. 能稳定复现 quick/full 两档结果
2. regression 脚本可在 slowdown 超阈值时返回非 0

## P1：backend 结构收口（Day 2）

交付：

1. 将 `boxFilter_backend_impl`、`gaussianBlur_backend_impl` 从“直接 fallback”改为“分类分发”
2. 增加内部 `dispatch_path` 标识（至少：`fallback`、`box3x3`、`gauss3x3`、`gauss_separable`）
3. 边界策略统一（`borderType/anchor/ROI/in-place`）

DoD：

1. 行为与当前合同测试一致（`cvh_test_imgproc` 全绿）
2. benchmark 能看见不同路径命中率

## P2：3x3 专项快路径（Day 3~5）

交付：

1. `box3x3` 快路径（`CV_8U`, `C1/C3/C4`）
2. `gaussian3x3` 快路径（可先固定系数，再扩 sigma 变体）
3. OpenMP 行并行（按工作量阈值开启）

DoD：

1. 对比 fallback 达成最小收益：
   - 3x3 C1: `>=2.0x`
   - 3x3 C3: `>=1.5x`
   - 3x3 C4: `>=1.8x`
2. 非命中场景不慢于 fallback（允许 5% 波动）

## P3：通用路径与分离卷积（Day 6~7）

交付：

1. `GaussianBlur` 统一走 separable 内核（X/Y 两次 1D）
2. 补 internal generic filter kernel（支持 `5x5/7x7`）
3. 不支持组合显式 fallback（禁止 silent wrong result）

DoD：

1. `5x5/7x7` 正确性与参考路径一致（浮点容差可配置）
2. ROI/non-continuous 全部通过合同测试

## P4：CI 门禁与收口（Day 8~10）

交付：

1. CI 新增 imgproc-filter benchmark job（quick）
2. 设定首版 slowdown 阈值（全局 + 分桶）
3. 文档同步：支持矩阵、已知限制、命中路径说明

DoD：

1. PR 可自动判断 filter 是否性能退化
2. 新人可按文档复现实验结果

## 3. 验证矩阵（首版）

shape：

1. `720x1280`
2. `1080x1920`
3. `roi: 513x769`（non-continuous）

depth/channel：

1. `CV_8U`: `C1/C3/C4`
2. `CV_16S/CV_32F`: 先保 correctness，后续再做专项优化

kernel：

1. `box3x3/5x5`
2. `gaussian3x3/5x5/7x7`

border：

1. `BORDER_REPLICATE`
2. `BORDER_REFLECT_101`
3. `BORDER_CONSTANT`

## 4. 风险与应对

1. C3 interleaved 吞吐不稳定
   - 应对：C3 单独分桶，阈值独立，不与 C1/C4 混评
2. 多线程收益不稳定
   - 应对：并行阈值与最小行块限制，避免小图反向退化
3. 专项路径维护成本高
   - 应对：只保留高频 `3x3/5x5` 专项，其余回归到 generic

## 5. 本周落地清单（按顺序）

1. 建立 `imgproc_filter_benchmark` + regression script（先有数据）
2. 完成 backend 分发重构（先可观测）
3. 上 `box3x3`，再上 `gaussian3x3`（先拿最大收益）
4. 完成 ROI/non-contiguous 回归与阈值固化（再开门禁）

## 6. 验收命令（建议）

```bash
cmake -S . -B build-full-check -DCVH_USE_OPENMP=ON
cmake --build build-full-check -j --target cvh_test_imgproc cvh_benchmark_core_ops

./build-full-check/cvh_test_imgproc

# 待 P0 完成后启用
# ./build-full-check/cvh_benchmark_imgproc_filter --profile quick --output benchmark/current_filter_quick.csv
# python3 scripts/check_imgproc_filter_benchmark_regression.py \
#   --baseline benchmark/baseline_filter_quick.csv \
#   --current benchmark/current_filter_quick.csv \
#   --max-slowdown 0.08
```
