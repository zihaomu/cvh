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
  - 覆盖：`core` 二元基础算子与 `compare(Mat,Mat)`（含多 `depth` / `channel` / `shape`）
  - 输出：标准 CSV（stdout）或 `--output <file>`

- 回归检查脚本：`scripts/check_core_benchmark_regression.py`
  - 对比 baseline/current CSV
  - 支持全局阈值 `--max-slowdown`
  - 支持分桶阈值 `--max-slowdown-by-op-depth`（按 `op+depth` 或单维覆盖）
  - 超过阈值时返回非 0（可用于 CI gate）

- 可执行程序：`cvh_benchmark_imgproc_filter`
  - 源码：`benchmark/imgproc_filter_benchmark.cpp`
  - 覆盖：`boxFilter/GaussianBlur`（`CV_8U`，`C1/C3/C4`，continuous/ROI）
  - 输出字段：`op,kernel,depth,channels,shape,layout,border,dispatch_path,ms_per_iter`
  - `dispatch_path` 典型值：`fallback / box3x3 / box_generic / gauss3x3 / gauss_separable`
  - 支持 `--dispatch auto|optimized-only|fallback-only`（用于 A/B 对照）

- 回归检查脚本：`scripts/check_imgproc_filter_benchmark_regression.py`
  - 对比 baseline/current CSV
  - 支持全局阈值 `--max-slowdown`
  - 支持分桶阈值 `--max-slowdown-by-op-kernel`（按 `op+kernel` 或单维覆盖）
  - 超过阈值时返回非 0（可用于 CI gate）

- 速度对照脚本：`scripts/report_imgproc_filter_speedup.py`
  - 对比 baseline/candidate CSV（按相同 case）
  - 输出 `geomean/median/min/max speedup`
  - 可用 `--fail-below-speedup` 设最低加速门槛

## 使用示例

0. （可选）启用 OpenMP 构建（对 `compare` 等已并行化 kernel 提升明显）：

```bash
cmake -S . -B build-full-test -DCVH_USE_OPENMP=ON
cmake --build build-full-test -j --target cvh_benchmark_core_ops
```

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

4. 对热点分桶设置阈值（例如对 `CV_16F` 和 compare 单独放宽）：

```bash
python3 scripts/check_core_benchmark_regression.py \
  --baseline benchmark/baseline_quick.csv \
  --current benchmark/current_quick.csv \
  --max-slowdown 0.08 \
  --max-slowdown-by-op-depth CV_16F=0.20 \
  --max-slowdown-by-op-depth CMP_GT:CV_16F=0.25
```

5. 运行 imgproc filter quick 基准并导出结果：

```bash
./build-full-test/cvh_benchmark_imgproc_filter --profile quick --output benchmark/current_filter_quick.csv
```

6. 生成一次 filter 基线（例如当前主分支）：

```bash
cp benchmark/current_filter_quick.csv benchmark/baseline_filter_quick.csv
```

7. 对比 filter 回归（默认允许最多 8% 变慢）：

```bash
python3 scripts/check_imgproc_filter_benchmark_regression.py \
  --baseline benchmark/baseline_filter_quick.csv \
  --current benchmark/current_filter_quick.csv \
  --max-slowdown 0.08
```

8. 量化 filter 加速（A/B 对照：forced fallback vs optimized）：

```bash
./build-full-test/cvh_benchmark_imgproc_filter --profile quick --dispatch fallback-only --output benchmark/filter_quick_fallback.csv
./build-full-test/cvh_benchmark_imgproc_filter --profile quick --dispatch optimized-only --output benchmark/filter_quick_optimized.csv

python3 scripts/report_imgproc_filter_speedup.py \
  --baseline benchmark/filter_quick_fallback.csv \
  --candidate benchmark/filter_quick_optimized.csv \
  --fail-below-speedup 1.0
```

## 建议流程

- 日常提交：使用 `quick` profile 做回归门禁。
- 周期性评估：使用 `full` profile 做深度扫描（更慢，但覆盖更多组合）。
