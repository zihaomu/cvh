# `benchmark` 目录规划

## 目录职责

提供性能基准与回归检查，防止优化退化和跨版本性能漂移。

## 规划目标

- 建立可重复的 benchmark 入口（固定数据、固定线程、固定输出格式）。
- 优先覆盖 `core` 热点算子，再覆盖 `imgproc`。
- 可选对比 OpenCV 或仓库历史版本。

## 阶段计划

### P1：基准框架

- 统一 CLI 参数：输入规模、线程数、warmup、repeat。
- 输出统一为 `csv/json`，便于 CI 归档。

### P2：Core 基线

- 覆盖 `add/mul/convertTo/transpose/gemm` 等高频路径。
- 分离标量路径与 SIMD 路径数据。

### P3：Imgproc 基线

- 覆盖 `cvtColor/resize/GaussianBlur`。
- 建立不同分辨率下的吞吐与延迟对比。

## 完成定义（DoD）

- 关键算子有可追踪性能曲线。
- PR 可基于 benchmark 报告判断是否有明显性能回退。

## 当前可用工具

- 可执行程序：`cvh_benchmark_core_ops`
  - 源码：`benchmark/core_ops_benchmark.cpp`
  - 覆盖：`core` 二元基础算子（含多 `depth` / `channel` / `shape`）
  - 输出：标准 CSV（stdout）或 `--output <file>`

- 回归检查脚本：`scripts/check_core_benchmark_regression.py`
  - 对比 baseline/current CSV
  - 超过阈值时返回非 0（可用于 CI gate）

## 使用示例

1. 运行 quick 基准并导出结果：

```bash
./build-full-test/cvh_benchmark_core_ops --profile quick --warmup 2 --iters 20 --repeats 5 --output benchmark/current_quick.csv
```

2. 生成一次基线（例如当前主分支）：

```bash
cp benchmark/current_quick.csv benchmark/baseline_quick.csv
```

3. 对比回归（默认允许最多 8% 变慢）：

```bash
python3 scripts/check_core_benchmark_regression.py \
  --baseline benchmark/baseline_quick.csv \
  --current benchmark/current_quick.csv \
  --max-slowdown 0.08
```

## 建议流程

- 日常提交：使用 `quick` profile 做回归门禁。
- 周期性评估：使用 `full` profile 做深度扫描（更慢，但覆盖更多组合）。
