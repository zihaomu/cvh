# cvh GEMM 第四阶段加速计划：常驻线程运行时与分层并行调度

> 状态：主要实现已落地，性能发布门禁部分通过（2026-07-28）
>
> 计划日期：2026-07-28
>
> 主优化范围：AArch64 NEON FP32 GEMM
>
> 运行时范围：为 GEMM 建立可复用的 header-only 并行能力；不要求其他算子在本阶段迁移
>
> 产品边界：严格 header-only；不新增必需的 `.cpp/.c/.S` 编译单元；不链接 Accelerate、BLAS、LAPACK、OpenMP 或第三方线程库
>
> 兼容边界：保留阶段三 `6×16/6×8`、B v2、scalar、OpenCV UI、旧 NEON `8×12`、AVX2 和全部 fallback
>
> 前置文档：[第三阶段 BLISlab/KleidiAI NEON GEMM 计划](./neon-gemm-phase3-blislab-kleidiai-acceleration-plan.md)

## 0. 结论摘要

第三阶段已经把 GEMM 从 micro-tile 任务提升为 `(ic,jc)` 输出宏块任务，也建立了真实 `MC/NC/KC`、Direct-A、B v2 和 NEON `6×16/6×8` 内核。但是，它只解决了“一个任务计算什么”，没有完整解决以下问题：

1. 当前 `StdThread` 后端在每次 `parallel_for_` 中创建和回收线程，线程生命周期成本没有被纳入 GEMM 设计。
2. cache block `MC/NC` 同时被当作调度 block，导致 cache 参数和负载均衡互相牵制。
3. 文档定义了分级 worker 策略，实现却只有一个 `8M FMA` 布尔门槛。
4. batch、packing、transA 转换和单矩阵计算各自做局部决策，没有统一的算子级并行计划。
5. benchmark 记录的是 chunk 数，不是实际 worker 数，也没有独立测量 pool wake、scheduler、barrier 和负载不均衡。

第四阶段不以“让所有 GEMM 无条件使用 8 线程”为目标，而是以“让合适的规模获得可证明的并行正收益”为目标：

| Shape | 第四阶段初始目标 |
| --- | --- |
| `128³` cold call | 保持串行或不超过阶段三串行 3% |
| `128³` warm pool | 尝试 2 workers；只有稳定快于串行至少 10% 才进入 Auto |
| `256³` | 默认评估 4 workers，不盲目使用 8 workers |
| `512³` | 使用最多 8 workers，关闭阶段三 `<=0.45 ms` 目标 |
| 多个 small/medium batch | 优先按 batch 并行，消除 32K..8M 的并行空档 |
| one-shot packing | 只在 pack 工作足够大时并行；broadcast B 只 pack 一次 |

第四阶段的核心顺序是：

```text
常驻线程运行时
  -> 实际 worker/调度成本遥测
  -> cache block 与调度分区解耦
  -> shape/batch/packing 分层并行
  -> 1/2/4/8 workers 离线调优
  -> Auto 白名单
```

不能先降低 `8M` 门槛。若线程仍按调用创建，降低门槛只会把阶段二出现过的小矩阵倒退重新引入。

### 0.1 落地结果

第四阶段已完成以下代码交付：

| 模块 | 落地内容 |
| --- | --- |
| `parallel_runtime.h` | function-local static 常驻线程池、caller 参与、按调用 worker budget、并发 caller 串行化、nested guard、异常传播、warm 状态和真实参与 worker 遥测 |
| `gemm_parallel.hpp` | Serial/M/MN/Batch 计划、按工作量限制 worker、MR 对齐均衡分区、task min/max FMA 遥测 |
| `gemm_blocked.hpp` | 调度分区与 `MC/NC/KC` cache blocking 解耦；executor 继续在每个调度分区内部执行 cache blocking |
| `gemm_impl.hpp` | operator 级 batch/inner 二选一、broadcast B pack once、large-B panel 并行 pack、transA slice pack 与 compute 融合 |
| benchmark | requested/chosen/participating workers、axis、task、pack、broadcast、pool warm、task min/max FMA；独立 2/4/8-worker persistent 与 spawn/join 基准 |
| test | persistent reuse、1000 次 tiny-job、worker budget、nested、concurrent caller、M/MN 均衡、batch 单层并行、broadcast pack once 和 transA tile pack |

最终 Auto 策略为：

| Case | axis | chosen workers | tasks | 说明 |
| --- | --- | ---: | ---: | --- |
| `128³` one-shot | Serial | 1 | 1 | pack included 时 2T 未稳定通过 cold/end-to-end 门禁 |
| `128³` pack-once | M | 2 | 4 | warm pool 可获得收益 |
| `256³` | M | 4 | 8 | 避免阶段三直接使用全部 8 workers |
| `512³` | M | 8 | 16 | 2× oversubscription，MR 对齐动态领取 |
| `8×128³` broadcast B | Batch | 4 | 8 | 外层 batch 并行，内部 GEMM 强制串行；B 只 pack 一次 |
| M tiles 不足且 N 足够大 | MN | 由 work cap 决定 | `M parts × N parts` | N 按 `NC` 宏块边界划分 |

`128/256/512` 的 M 分区最大任务均为平均任务的 `1.125×`，低于 `1.15×` 门槛；具体 min/max FMA 已直接输出到 benchmark CSV，而不再由 chunk 数推测。

### 0.2 本机性能验收

环境为 Apple arm64、AppleClang、Release、NEON native v2。以下数据采用同一轮内的 median；微秒级结果会受到温升和 P/E core 调度影响，因此保留 raw CSV 重测要求。

| 项目 | 串行/旧方式 | 第四阶段 | 结论 |
| --- | ---: | ---: | --- |
| warm no-op，2 workers | spawn/join `0.021913 ms` | persistent `0.001220 ms` | 固定成本降低 `94.4%`，通过 `80%` 门槛 |
| warm no-op，4 workers | spawn/join `0.031667 ms` | persistent `0.014312 ms` | 降低 `54.8%`，尚未达到 `80%` |
| warm no-op，8 workers | spawn/join `0.049664 ms` | persistent `0.026094 ms` | 降低 `47.5%`，尚未达到 `80%` |
| `128³` pack-once | 1T `0.032833 ms` | warm 2T `0.025589 ms` | 快 `22.1%`，通过；one-shot Auto 仍保持串行 |
| `256³` one-shot | 阶段三 4T `0.113324 ms` | 4T `0.108330 ms` | 有提升但未达到 `0.105 ms` hard gate |
| `256³` pack-once | 阶段三 4T `0.113324 ms` | 4T `0.101105 ms` | 通过 `0.105 ms` 门槛 |
| `512³` one-shot | 阶段三 8T `0.495750 ms` | 8T `0.454750 ms` | 快 `8.3%`，但比 `0.45 ms` 门槛高约 `1.1%` |
| `512³` pack-once | 阶段三 8T `0.495750 ms` | 8T `0.424167 ms` | 通过 `0.45 ms` 门槛 |
| `8×128³` broadcast B | 1T `0.294033 ms` | 4 workers `0.126453 ms` | throughput 提升 `2.33×`，通过 |

最终 OpenCV compare smoke 使用项目的 OpenCV/Accelerate build，并通过 upstream correctness：

| Shape | 路径 | cvh | OpenCV/Accelerate | 当前差距 |
| --- | --- | ---: | ---: | ---: |
| `128³` | one-shot | `0.045408 ms` | `0.004867 ms` | cvh 慢 `9.33×` |
| `128³` | pack-once | `0.030400 ms` | `0.003958 ms` | cvh 慢 `7.68×` |
| `256³` | one-shot | `0.102933 ms` | `0.021175 ms` | cvh 慢 `4.86×` |
| `256³` | pack-once | `0.096425 ms` | `0.020625 ms` | cvh 慢 `4.68×` |
| `512³` | one-shot | `0.587125 ms` | `0.160959 ms` | cvh 慢 `3.65×` |
| `512³` | pack-once | `0.498625 ms` | `0.169291 ms` | cvh 慢 `2.95×` |

该 smoke 的 `512³` 每 repeat 仅 1 iteration，且在整套测试之后执行，只用于验证当前差距趋势，不替代前述同轮性能门禁数据。

结论是：runtime、分层调度、batch 和 packing 结构已经落地，`128³ pack-once`、`256³ pack-once`、`512³ pack-once` 与 batched GEMM 已获得净收益；`256³/512³ one-shot` 和 4/8-worker 固定调度成本还没有全部关闭 hard gate。因此本文档状态不是“全部性能门禁完成”，后续调优不得删去这些未通过项。

### 0.3 验证矩阵与遗留项

已通过：

- ARM64 Release：GEMM fixture/public API、native dispatch、parallel runtime、phase4 planner/batch/transA。
- ASan+UBSan、TSan、UI-off、multi-TU ODR、header compile 和 install-tree consumer。
- x86_64/AVX2 相关目标交叉编译；同时修复了 CMake 在 x86 交叉构建中错误注入 `CV_NEON=1` 的问题。
- 五轮连续 benchmark 和 1000 次 tiny-job 回归；曾评估的无锁 job publication 因压力测试卡住而撤回，发布版本保留已通过 sanitizer 的互斥发布协议。

尚未关闭：

- 真实 x86 运行验证：当前 Apple 主机未安装 Rosetta，本轮只能做 x86_64 编译门禁。
- Linux AArch64 Clang/GCC CI。
- 4/8-worker wake/publication/barrier 进一步降本。
- `256³` 与 `512³` one-shot hard gate 的冷机稳定复测和后续优化。

## 1. 阶段三并行策略审计

### 1.1 当前线程生命周期

`include/cvh/core/detail/parallel_runtime.h` 的 `run_stdthread` 每次调用都会：

```text
allocate vector<thread>
spawn worker_count - 1 threads
caller also runs work
join every worker
destroy thread vector
```

这个模型对毫秒级工作可以获益，但不适合 `128³` 这类约 35 微秒的 GEMM。阶段三通过禁止 `128³` 并行规避了问题，没有消除问题。

第四阶段必须提供常驻 worker，并把单次并行调用的固定成本降低到：

- 发布 job；
- 唤醒所需 worker；
- 原子领取 task；
- 完成 barrier。

### 1.2 文档策略与实现不一致

阶段三文档规定：

| 工作量 | 计划策略 |
| --- | --- |
| `<8M` FMA | 1 worker |
| `8M..32M` FMA | 2–4 workers |
| `>=32M` FMA | 最多使用配置 worker |

实际 `should_parallelize_macro_tasks` 只有：

```text
if M*N*K >= 8M:
    parallel_for_(all macro tasks)
```

因此 `256³≈16.8M` 在配置 8 线程时会使用最多 8 workers，而不是计划中的 2–4 workers。阶段三稳定数据中 `256³` 的 4T 为 0.113324 ms，8T 为 0.116199 ms，已经说明“更多线程”不必然更快。

### 1.3 固定 `MC=72,NC=128` 的负载形态

阶段三任务数：

```text
task_count = ceil(M / 72) * ceil(N / 128)
```

方阵任务分布：

| Shape | task_count | M 方向高度 | 问题 |
| --- | ---: | --- | --- |
| `128³` | 2 | 72 / 56 | 最多只有 2 workers |
| `256³` | 8 | 72 / 72 / 72 / 40 | 尾任务约为 full task 的 56% |
| `512³` | 32 | 7×72 / 8 | 尾任务仅为 full task 的 11%，单任务相差 9 倍 |

动态领取可以降低空转时间，但最后一个 wave 仍由最大 task 决定。`512³` 的 8-row tail 说明 `MC` 既承担 cache block 又承担线程分区是不合理的。

### 1.4 batch 并行空档

当前 `for_each_gemm_batch` 只在单矩阵工作量不超过 `32K` 时考虑 batch 外层并行；单矩阵内部则需要达到 `8M` 才进入宏块并行。

因此：

```text
32K < per_matrix_work < 8M
```

这个区间中的 batched GEMM 可能出现：

- batch 外层串行；
- 单矩阵内部串行；
- 全部 batch 逐个执行。

一批 `128³` 就处在这个空档中。第四阶段必须让 operator 先选择“batch 并行”还是“单矩阵并行”，而不是让两层互不知情地分别判断。

### 1.5 packing 位于串行关键路径

阶段三 one-shot 顺序是：

```text
allocate/reuse workspace
pack B v2 serially
parallel compute
```

TN/TT 还会先串行生成 row-major A workspace。对 `512³`，pack B 占比有限；对 FP16、INT8、small-M、transA 和 broadcast batch，packing 可能成为 Amdahl 串行部分。

第四阶段不能无条件并行 packing，因为额外 barrier 也有成本；必须按 pack bytes、panel 数和 pool 状态单独设门槛。

### 1.6 当前遥测不足

阶段三 benchmark 的 `effective_chunks` 来源于 `last_parallel_chunks()`：

- 它表示 dispatch chunk 数；
- 不等于实际 worker 数；
- 不表示每个 worker 是否真正执行过 task。

例如 `256³,4T` 可以显示 8 chunks，但实际最多只有 4 workers。第四阶段必须区分：

```text
requested_threads
chosen_workers
dispatch_tasks
dispatch_chunks
participating_workers
task_min_fma
task_max_fma
pool_state=cold|warm
```

## 2. 第四阶段目标与非目标

### 2.1 功能目标

1. 建立 C++17 header-only 常驻线程池。
2. caller 参与计算，不为 caller 单独保留空转角色。
3. GEMM 显式生成一次算子级 `ParallelPlan`。
4. cache blocking 与线程分区解耦。
5. 支持 Serial、Batch、M、MN 四种调度轴。
6. 防止 batch 外层和单矩阵内层嵌套并行。
7. one-shot B packing、transA packing 和 compute 共享同一个 worker 预算。
8. 对 broadcast B 只 pack 一次并跨 batch 只读共享。
9. cold call 与 warm steady-state 分开决策和测量。
10. 真实记录 worker、task、scheduler 和 barrier 元数据。

### 2.2 性能目标

以阶段三本机数据为起点：

| Shape | 阶段三参考 | 第四阶段接纳目标 | Stretch |
| --- | ---: | ---: | ---: |
| `128³` 1T/serial | 0.035–0.038 ms | cold 不回退；warm 2T 至少快 10% | `<=0.025 ms` |
| `256³` 4T | 0.113324 ms | `<=0.105 ms` | `<=0.090 ms` |
| `512³` 8T | 0.495750 ms | `<=0.450 ms` | `<=0.400 ms` |
| batch `8×128³` | 当前可能全串行 | warm throughput 至少提升 2× | 随 worker 接近线性扩展 |

目标必须在同一机器、同一编译器、相同数据、相同输出分配合同下重新基线。上表不是脱离重测结果的绝对承诺。

### 2.3 非目标

- 不在本阶段重写 NEON `6×16/6×8` 算术内核。
- 不在首版引入 K-split 和跨 worker reduction。
- 不让所有 shape 无条件并行。
- 不把 OpenMP、GCD、TBB 或 Accelerate 变成依赖。
- 不修改 AVX2 默认门槛；线程运行时通过后再单独评估 AVX2 复用。
- 不用线程数掩盖单线程 micro-kernel 仍落后 Accelerate 的事实。

## 3. 总体架构

第四阶段将 GEMM 分为四层：

```text
Public GEMM operator
  |
  +-- GemmDispatchPlan
  |     dtype/layout/kernel/MC/NC/KC/packing format
  |
  +-- GemmParallelPlan
  |     axis/workers/tasks/cold-warm/pack policy/nesting policy
  |
  +-- PersistentParallelRuntime
  |     publish job/wake workers/claim task/barrier/exception
  |
  +-- GEMM task executor
        owns disjoint C region
        completes all pc blocks
        calls existing 6x16/6x8 micro-kernel
```

`GemmDispatchPlan` 决定“使用什么内核和 cache 参数”；`GemmParallelPlan` 决定“由多少 worker、沿哪个维度处理哪些输出”。两者不能继续隐式绑定。

## 4. Header-only 常驻线程运行时

### 4.1 基本设计

在 `include/cvh/core/detail/parallel_runtime.h` 中新增内部 persistent runtime：

```text
PersistentWorkers
  worker threads: N-1
  caller thread: participates
  active job: pointer to stack-owned JobState
  next_task: atomic<int>
  remaining_workers/tasks: atomic<int>
  epoch: monotonically increasing
  wake: condition_variable
  completion: condition_variable or atomic + cv
```

约束：

- 所有定义保持 `inline`；
- function-local static/inline storage 在多 TU 下只有一个逻辑 runtime；
- 不为每个 task 分配 `std::function`；
- 不为每次调用创建 `vector<thread>`；
- job state 在调用栈上，调用返回前等待全部 worker 退出该 job；
- worker 数只增长或按安全协议重建，普通 `setNumThreads` 不反复销毁线程；
- 进程退出时设置 stop flag 并 join；
- worker callback 异常由 runtime 捕获并在 caller 重抛。

### 4.2 低开销 job 接口

避免为 20–100 微秒任务引入堆分配和重型 type erasure。建议使用：

```cpp
struct ParallelJob
{
    void* context;
    void (*run_task)(void*, int);
    std::atomic<int> next_task;
    int task_count;
    int worker_budget;
};
```

模板 `parallel_for_persistent` 在栈上保存具体 lambda context，通过函数指针 trampoline 调用。

### 4.3 cold 与 warm 语义

必须明确三种状态：

| 状态 | 行为 |
| --- | --- |
| `Disabled` | `setNumThreads(1)` 或 Serial backend，永远串行 |
| `Cold` | pool 尚未建立；小 GEMM不得为了启动 pool 而稳定回退 |
| `Warm` | workers 已存在，可使用低门槛策略 |

初始建议：

- 显式 `setNumThreads(n>1)` 可以建立或唤醒 pool，启动成本发生在设置阶段；
- 默认配置下，首次 large parallel operator 可以 lazy-create pool；
- 首次 small GEMM 若 pool cold，保持串行；
- benchmark 必须分别输出 cold-call 和 warm-call。

不得在 benchmark warmup 中悄悄隐藏 pool 创建，却把结果标为 cold。

### 4.4 worker 唤醒

`128³` 只希望唤醒 1 个额外 worker，而不是唤醒全部硬件线程。runtime 需要支持 worker budget：

```text
chosen_workers=2
caller + 1 persistent worker
```

首版使用 condition variable；只有数据证明 wake latency 仍占比过高时，才评估：

- bounded spin then sleep；
- per-worker epoch；
- 平台特定 wait primitive。

任何自旋策略必须同时评估能耗和空闲 CPU 占用。

### 4.5 嵌套和并发调用

定义 thread-local parallel depth：

```text
if already inside pool callback:
    inner parallel_for runs serially
```

多个用户线程同时调用 GEMM 时，首版采用安全策略：

- 一个调用获得 gang execution；
- runtime busy 时，其他调用串行执行自己的 GEMM，或进入有界等待；
- 禁止两个调用各自唤醒完整 worker 集导致 oversubscription；
- 后续只有在并发 server benchmark 证明需要时才增加多 job queue。

## 5. GEMM 分层并行计划

### 5.1 新增 `GemmParallelPlan`

建议内部结构：

```cpp
enum class GemmParallelAxis
{
    Serial,
    Batch,
    M,
    MN
};

struct GemmParallelPlan
{
    GemmParallelAxis axis;
    int requested_threads;
    int chosen_workers;
    int task_count;
    int batch_tasks;
    int m_partitions;
    int n_partitions;
    bool pool_warm;
    bool parallel_pack_b;
    bool fuse_pack_a;
    std::uint64_t total_fma;
    std::uint64_t min_task_fma;
    std::uint64_t max_task_fma;
};
```

它不进入 public API，不进入 `GemmPackedB` ABI。

### 5.2 worker 预算

worker 数至少受以下因素约束：

```text
chosen_workers <= requested_threads
chosen_workers <= available output partitions or batch count
chosen_workers <= workers allowed by total work
chosen_workers <= architecture profile cap
```

初始 shape policy：

| 条件 | warm pool 初始值 |
| --- | --- |
| `work < 1M` | 1 worker |
| `1M <= work < 8M` | 最多 2 workers，必须有至少 2 个均衡输出任务 |
| `8M <= work < 32M` | 最多 4 workers |
| `work >= 32M` | 最多 8 workers或用户配置值，取较小者 |

这些数值是搜索起点，不是直接写死的最终值。N4-0/N4-3 必须扫描 break-even。

### 5.3 并行轴选择

按以下顺序选择：

1. 若已在并行 callback 中：Serial。
2. 若 `batch_count >= 2` 且单矩阵不足以内并行高效：Batch。
3. 若 M 方向 micro-tile 数足以填满 chosen workers：M。
4. 若 M 不足、N 足够宽：MN。
5. 否则：Serial。

首版不使用 K-split，因为 K-split需要：

- 每 worker 私有 partial C；
- reduction；
- 额外内存和 barrier；
- 更复杂的 NaN/Inf/FMA 顺序合同。

### 5.4 cache block 与调度分区解耦

调度器按 `MR=6` 的 micro-tile 数平衡 M：

```text
m_tiles = ceil(M / MR)
partition m_tiles as evenly as possible across workers
each partition owns a contiguous row range
```

示例：

| Shape | M micro-tiles | workers | 近似分配 |
| --- | ---: | ---: | --- |
| `128³` | 22 | 2 | 11 / 11 tiles，即约 66 / 62 valid rows |
| `256³` | 43 | 4 | 11 / 11 / 11 / 10 tiles |
| `512³` | 86 | 8 | 11 / 11 / 11 / 11 / 11 / 11 / 10 / 10 tiles |

每个调度分区内部仍可按 `MC=72` 或重新调优的 cache block 迭代。调度边界不再要求等于 MC 边界。

### 5.5 M 优先、MN 补充

方阵初始策略：

```text
if balanced M partitions >= chosen_workers:
    each worker owns M range and iterates all jc
else:
    split N by NR-aligned groups to fill remaining workers
```

优点：

- square 256/512 可以优先使用 M 方向；
- 同一 worker 的 A row range可跨多个 N block 复用；
- 减少 task 数和 scheduler 原子操作；
- C 按完整行范围分离，降低 false sharing 风险。

MN 分区的 N 边界必须按 16-column/64-byte 对齐；尾块只由一个 task 拥有。

### 5.6 task ownership

每个 task 仍满足：

- 拥有互不重叠的 C rectangle；
- 在 task 内完成全部 `pc`；
- packed B 只读共享；
- Direct A 只读共享；
- 不需要 C reduction；
- 不在 K loop 中调度；
- overwrite/accumulate 仍以 `pc==0` 为边界。

## 6. 三个方阵的初始策略

### 6.1 `128³`

阶段三：

```text
2 macro tasks
work≈2.1M FMA
pool cold/current std::thread spawn: serial
```

第四阶段：

- cold pool：1 worker；
- warm pool：候选 2 workers；
- M micro-tiles 平衡为 11/11；
- caller + 1 persistent worker；
- 不并行 pack B；
- 只有 median 至少快 10%、p90 不回退时进入 Auto。

不得把 `threads=8` 记录成“使用了 8 线程”；应记录：

```text
requested_threads=8
chosen_workers=2
participating_workers=2
```

### 6.2 `256³`

阶段三：

```text
8 macro tasks
4T 与 8T 接近，4T 稳定数据略快
```

第四阶段：

- 默认候选 4 workers；
- M 方向 43 个 micro-tiles 平衡成 4 组；
- 每个 worker 迭代完整 N；
- 比较 3/4/6/8 worker，但 Auto 初始上限为 4；
- pack-once 与 one-shot 分别验收；
- 若 parallel pack B 需要第二次 barrier 且收益不足，保持串行 pack。

### 6.3 `512³`

阶段三：

```text
32 macro tasks
28 个 72-row tasks + 4 个 8-row tasks
8T=0.495750 ms
```

第四阶段：

- 8 workers；
- M micro-tile 平衡分区，消除 72/8 的极端尾块；
- M-only 与 MN 两种调度做 A/B；
- 扫描 cache `MC=48/60/66/72/84/96`，但调度分区不随 MC 改变；
- 扫描 `NC=64/128/256`；
- 目标先关闭 `<=0.45 ms`，再挑战 `<=0.40 ms`；
- 记录 P/E core 混合造成的方差，必要时将 Apple profile worker cap 与通用 ARM profile 分开。

## 7. Packing 与 batch

### 7.1 B v2 并行 packing

B v2 的 `(jc,pc)` macro panel 物理区间互不重叠，可以并行 packing。候选 task：

```text
pack_task = one or more (jc,pc) panels
```

门槛必须基于：

- `K*N*sizeof(float)`；
- panel 数；
- 是否 FP16/INT8 转换；
- pool 是否 warm；
- 后续 compute 能否复用 packed B；
- 额外 barrier 成本。

128³ 初始保持串行 pack。256/512、FP16、INT8 再评估。

### 7.2 broadcast B

one-shot batched GEMM 若 B 在 batch 维广播：

- 在 batch 调度前 pack B 一次；
- 所有 batch task 共享只读 packed B；
- 不允许每个 batch callback 重复 pack 相同 B；
- workspace 生命周期必须覆盖全部 batch task。

非广播 B 可以：

- 每 batch 私有 pack；
- 或 batch worker 使用自己的 thread-local workspace；
- 不允许多个 worker 写同一个 thread-local/共享 buffer。

### 7.3 transA

TN/TT 不再无条件先生成完整 A workspace再计算。M 分区任务可以：

```text
pack assigned A row range
compute assigned C row range
```

即每个 worker：

- 拥有 A workspace slice；
- pack 与 compute 融合；
- 不需要全局 A-pack barrier；
- 不重复 pack 同一 row range。

只有 benchmark 证明完整并行 transpose 更快时才保留全量方案。

### 7.4 batch 与单矩阵互斥

算子层一次性选择：

| 情况 | 选择 |
| --- | --- |
| batch 多、单矩阵小/中 | batch parallel，inner serial |
| batch 少、单矩阵大 | outer serial，inner M/MN parallel |
| batch 与单矩阵都大 | 首版仍选一层；根据 total work 选择，不做 nested |

消除阶段三 32K..8M 的空档。

## 8. Apple Silicon 与通用 AArch64

Apple Silicon 可能包含性能核和能效核，`std::thread::hardware_concurrency()` 只提供总逻辑线程数，不能保证 worker 都运行在相同类型核心。

第四阶段分两个 profile：

| Profile | 策略 |
| --- | --- |
| Generic AArch64 | 只依赖 requested threads 和实测 break-even |
| Apple arm64 | 可选读取系统性能核数量，建立 conservative worker cap |

约束：

- Apple 检测必须通过 public system API；
- 不新增链接库；
- 检测失败时回退 Generic；
- 不在首版引入强 affinity；
- QoS/affinity 只有在方差和吞吐数据证明必要时评估。

## 9. Dispatch 与 API

### 9.1 保持 public API

以下 API 不变：

```cpp
cvh::setNumThreads(n);
cvh::setParallelBackend(...);
cvh::gemm(...);
cvh::gemm_pack_b(...);
```

可以新增只读诊断 API，但不得要求用户调用新 API 才能获得正确结果。

### 9.2 Auto 规则

Auto 的接纳单位是：

```text
shape class + layout + dtype + pool state + worker budget
```

而不只是 `M*N*K`。

候选只有满足以下条件才进入 Auto：

- 与阶段三 serial/parallel baseline 配对比较稳定更快；
- cold 与 warm 行为分别达标；
- chosen workers 不超过实测最优上限；
- packing included/excluded 两种合同均无隐藏回退；
- batch 和单矩阵没有 nested oversubscription；
- correctness、TSAN/ASan/UBSan、ODR、install-tree 通过。

## 10. 代码落点

| 文件 | 第四阶段职责 |
| --- | --- |
| `include/cvh/core/detail/parallel_runtime.h` | 常驻 worker、job epoch、worker budget、nested guard、真实参与度 |
| `include/cvh/core/parallel.h` | 保持 API；可增加内部 warm-state 查询和诊断 |
| `include/cvh/core/detail/gemm_parallel.hpp` | 新增 `GemmParallelPlan`、axis/worker/task partition policy |
| `include/cvh/core/detail/gemm_blocked.hpp` | 接受 row/column task slice；cache block 与 scheduler partition 解耦 |
| `include/cvh/core/detail/gemm_pack.hpp` | panel-range pack、parallel-safe offset、A slice pack |
| `include/cvh/core/detail/gemm_impl.hpp` | operator 级 batch/pack/compute 计划，禁止 nested parallel |
| `include/cvh/core/detail/gemm_dispatch.hpp` | 保持 kernel dispatch；提供 cache traits，不直接决定 worker |
| `benchmark/core_mat_header_benchmark.cpp` | cold/warm、pool/scheduler/barrier、worker/task 真实元数据 |
| `benchmark/opencv_compare_header_benchmark.cpp` | 继续输出 cvh/Accelerate 趋势，不混淆 requested/chosen workers |
| `test/core/runtime/parallel_for_test.cpp` | pool 生命周期、异常、重配、nested、并发 caller |
| `test/core/internal/gemm_native_dispatch_test.cpp` | 1/2/4/8 worker correctness、batch/M/MN axis、tail/race |

`gemm_parallel.hpp` 必须独立于 NEON 类型，使 scalar reference 和未来 AVX2 可以复用调度模型。

## 11. Benchmark 设计

### 11.1 必须新增的指标

每行至少记录：

```text
requested_threads
chosen_workers
participating_workers
parallel_axis
pool_state
task_count
dispatch_chunks
min_task_fma
max_task_fma
task_imbalance_ratio
pack_b_ms
pack_a_ms
pool_dispatch_ms
compute_ms
barrier_ms
public_end_to_end_ms
```

`participating_workers` 是本次 job 真正执行过 task 的 unique worker 数，不是 chunk 数。

### 11.2 cold/warm 分离

至少提供：

| 模式 | 定义 |
| --- | --- |
| `cold_process_first_call` | pool 未创建、workspace 未扩容 |
| `warm_pool_cold_workspace` | pool 已存在，workspace 首次扩容 |
| `warm_steady_state` | pool 和 workspace 都已准备 |
| `pack_once` | B 预打包，输出分配合同明确 |
| `kernel/macro only` | 不包含 packing 和输出分配 |

不能只用 benchmark warmup 后的 steady-state 结果决定 cold public API 的 Auto。

### 11.3 shape 矩阵

方阵：

```text
64³, 96³, 128³, 160³, 192³, 256³, 384³, 512³, 768³, 1024³
```

非方阵：

```text
32×512×64
64×1024×64
256×32×256
64×256×1024
1024×256×64
```

batch：

```text
B = 2,4,8,16
per matrix = 32³,64³,128³,256³
broadcast B / non-broadcast B
```

布局/类型：

```text
NN, TN, TT
FP32, FP16 packed conversion, INT8 dequant
one-shot, pack-once
```

线程：

```text
requested = 1,2,3,4,6,8,hardware_default
```

### 11.4 稳定性

- 至少 9 repeats，最终报告使用 median/p90；
- coefficient of variation 超过 5% 重新测量；
- 随机化 1/2/4/8 worker 测试顺序，降低温度和频率偏差；
- 同一 shape 的 candidate 与 baseline 交替测量；
- 记录编译器、commit、CPU、系统电源状态；
- 128³ 使用足够 iterations，不能用单次微秒计时下结论；
- 512³ 控制温升并报告完整 repeat。

## 12. 正确性与并发门禁

### 12.1 GEMM correctness

- M/N/K 在 MR/NR/KC/MC/NC 前后边界；
- balanced M partition 的每个边界；
- MN 的 N split 边界；
- `pc>0` accumulate；
- NN/TN/TT；
- one-shot/pack-once；
- broadcast/non-broadcast batch；
- FP16/INT8；
- NaN/Inf/subnormal/`0/-0`；
- forced Serial 与 Auto worker plan 对照。

### 12.2 线程 runtime

- task 恰好执行一次；
- caller 参与；
- chosen worker cap 生效；
- pool cold/warm 状态转换；
- `setNumThreads(1→8→2→1)`；
- backend Serial 强制串行；
- worker exception 回传；
- strict/non-strict fallback；
- nested parallel 自动串行；
- 两个外部 caller 同时调用不死锁、不 oversubscribe；
- shutdown/静态析构无挂起；
- 多 TU 只有一个逻辑 runtime；
- fork 后行为若不能安全支持，必须记录限制并安全回退。

### 12.3 工具

- ASan + UBSan；
- TSAN：pool/job state、workspace、C ownership；
- Release 全量 CTest；
- UI on/off；
- generic x86 compile；
- AppleClang；
- Linux AArch64 Clang/GCC；
- ODR/install-tree/header-only consumer。

## 13. 工作包

### N4-0：重建并行基线与遥测

交付：

- 固定阶段三 commit 的 1/2/3/4/6/8T raw CSV；
- 增加 requested/chosen/participating workers；
- 增加 task min/max/imbalance；
- 增加 cold/warm 标记；
- 增加 pool dispatch、barrier 和 scheduler microbenchmark；
- 记录现有 `std::thread` spawn/join 成本。

退出条件：

- 能解释 128/256/512 每个 worker 实际做了什么；
- 不再用 chunks 代替 workers；
- 阶段三 scheduler `<5%` 条件有真实数据。

### N4-1：常驻线程运行时

交付：

- persistent worker singleton；
- allocation-free job publication；
- caller participation；
- worker budget；
- cold/warm state；
- nested guard；
- exception/shutdown/reconfigure；
- 原有 `parallel_for_` 合同保持。

退出条件：

- no-op job 的 warm dispatch 明显低于当前 spawn/join；
- runtime 单元测试通过；
- TSAN/ASan/UBSan 通过；
- 多 TU/installed consumer 通过；
- Serial backend 无行为变化。

### N4-2：GEMM balanced M scheduler

交付：

- `GemmParallelPlan`；
- M micro-tile 平衡分区；
- cache MC 与 schedule partition 解耦；
- M-only executor；
- 128→2、256→4、512→8 worker 候选；
- 真实 worker 元数据。

退出条件：

- task 不重叠、不遗漏；
- task max/mean FMA 不超过 1.15，纯 tail case 有说明；
- 256/512 不低于阶段三性能；
- 128 cold 不回退。

### N4-3：worker policy 与 MN fallback

交付：

- cold/warm 两套 threshold；
- 1/2/4/8 worker 离线搜索；
- M 不足时的 NR-aligned MN 分区；
- Apple/Generic profile；
- Auto 白名单。

退出条件：

- 128 warm 2T 达到至少 10% 收益，否则继续串行；
- 256 默认 worker 数由数据决定；
- 512 达到 `<=0.45 ms`；
- small-K/GEMV/skinny 无超过 3% 回退。

### N4-4：batch 与 packing

交付：

- operator 级 outer/inner parallel 决策；
- 修复 32K..8M batch 空档；
- broadcast B pack once；
- B panel-range parallel pack；
- transA slice pack + compute 融合；
- FP16/INT8 pack policy。

退出条件：

- batch 无 nested oversubscription；
- `8×128³` throughput 至少提升 2×；
- one-shot packing included 仍有净收益；
- pack-once 路径不重打 B；
- broadcast B 只 pack 一次。

### N4-5：参数调优与发布

交付：

- MC/NC/KC 与 worker 数联合扫描；
- M-only/MN A/B；
- cold/warm/one-shot/pack-once 最终表；
- cvh vs 阶段三、OpenCV CPU-only、OpenCV/Accelerate 趋势；
- 全平台和 sanitizer 报告；
- 更新第三/第四阶段状态。

退出条件：

- 全部 accepted path 达到第 14 节门槛；
- 未达标路径保持阶段三策略；
- 项目仍为严格 header-only。

## 14. 性能接纳门槛

### 14.1 runtime

- warm pool no-op dispatch 相比当前 spawn/join 至少降低 80%；
- job publication 不做堆分配；
- chosen workers 不超过 requested workers；
- participating workers 与 chosen workers 不一致时必须可解释；
- nested/concurrent caller 不产生 oversubscription。

### 14.2 GEMM

| Case | Hard gate |
| --- | --- |
| `128³` cold | 相对阶段三串行不回退超过 3% |
| `128³` warm 2T | 至少快 10%，否则 Auto 保持 1T |
| `256³` | 相对阶段三最佳结果至少快 8%，或达到 `<=0.105 ms` |
| `512³` | `<=0.45 ms` |
| `8×128³` batch | throughput 至少提升 2× |
| skinny/small-K/GEMV | 不回退超过 3% |
| scheduler+barrier | 256/512 accepted path 中低于 5% |
| task imbalance | max/mean FMA `<=1.15`，尾块例外需量化 |

### 14.3 正确性

任何以下问题直接拒绝：

- C 写竞争；
- packed workspace 跨 worker 写冲突；
- batch broadcast 错误；
- nested deadlock；
- exception 丢失；
- sanitizer/TSAN/ODR/install-tree 回归；
- cold-call 性能被 warmup 结果掩盖；
- benchmark 把 chunks 写成 worker 数。

## 15. 风险与止损

| 风险 | 处理 |
| --- | --- |
| header-only persistent pool 增加复杂度 | runtime 独立文件、严格单测、保持 Serial fallback |
| 静态析构/进程退出挂起 | 明确 stop/epoch/join 协议；析构专项测试 |
| 多个用户线程并发调用 | 首版单 gang + busy fallback，禁止 oversubscription |
| 128 wake/barrier 仍太贵 | Auto 保持串行；不为“用了线程”牺牲性能 |
| condition variable latency高 | 先测量；再评估 bounded spin，不直接自旋 |
| Apple P/E core 方差 | conservative worker cap，分离 Apple/Generic profile |
| parallel packing 多一次 barrier | 只对白名单大 pack 开启 |
| M-only 降低 B/A cache locality | 与 MN 配对测试，按 shape 选择 |
| 调度分区破坏 MC cache policy | executor 内部继续 cache blocking，不让 scheduler 替代 MC |
| TN per-worker workspace 过大 | capacity reuse、worker cap、全量 pack fallback |
| 编译时间/代码体积增长 | scheduler 与 runtime 独立 header，避免大量 shape 模板实例 |

满足以下任一条件时停止候选的 Auto 接入：

- warm pool 后仍没有至少 5% 稳定收益；
- cold public API 回退超过 3%；
- 必须依赖外部线程库；
- TSAN 无法关闭数据竞争；
- runtime 生命周期无法满足 header-only 多 TU；
- 只有单一 Apple 型号受益且不能通过 profile 隔离；
- 并行后能耗/CPU 占用显著增加但延迟收益不足。

## 16. 完成定义

第四阶段当前验收记录如下；未勾选项是发布门禁，不得因主体代码已落地而隐藏：

- [x] 常驻 header-only worker runtime 落地。
- [x] `parallel_for_` 不再为每个 accepted GEMM 创建/回收线程。
- [x] cold/warm pool 语义和 benchmark 明确。
- [x] requested/chosen/participating workers 分开记录。
- [x] task min/max FMA 与真实负载不均衡进入 benchmark 元数据。
- [x] cache MC 与 scheduler partition 解耦。
- [x] M、MN、Batch 三种并行轴落地并禁止 nested oversubscription。
- [x] `128³` pack-once warm 2T 达标；one-shot 以数据决定 Auto 保持串行。
- [x] `256³` 完成 1/2/3/4/6/8 配置搜索，Auto 选择 4 workers。
- [ ] `512³` one-shot 达到 `<=0.45 ms`；当前 `0.454750 ms`，pack-once 已达到 `0.424167 ms`。
- [x] 32K..8M batched GEMM 并行空档关闭。
- [x] broadcast B one-shot 只 pack 一次。
- [x] TN/TT 改为每个 worker 的 A slice pack + compute，不再做全量 A transpose barrier。
- [x] scheduler+barrier 已可分别测量。
- [ ] accepted 4/8-worker large shape 的固定 scheduler+barrier 成本稳定低于 5%。
- [x] TSAN、ASan、UBSan、ODR、install-tree、UI-off 通过。
- [ ] AppleClang 和 Linux AArch64 Clang/GCC 全部通过；AppleClang 已通过，Linux CI 待执行。
- [x] scalar、UI、旧 NEON fallback 无回归，AVX2 通过 x86_64 编译门禁。
- [ ] AVX2 真实 x86 运行门禁；当前主机无 Rosetta。
- [x] 项目仍是严格 header-only，无新增外部链接依赖。

## 17. 推荐实施顺序

```text
N4-0 telemetry + stage3 baseline
  -> N4-1 persistent runtime
  -> N4-2 balanced M scheduler
  -> N4-3 worker policy + MN fallback
  -> N4-4 batch + packing
  -> N4-5 tuning + release
```

最重要的约束：

1. 先降低线程固定成本，再降低 GEMM 并行门槛。
2. 先记录真实 worker，再讨论“8T 是否生效”。
3. 先把 M 分区做均衡，再调 MC/NC。
4. 先选择 outer batch 或 inner GEMM，只允许一层并行。
5. 只有 end-to-end、cold/warm 和 correctness 同时通过，才进入 Auto。
