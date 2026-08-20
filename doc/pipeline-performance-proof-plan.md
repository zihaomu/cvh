# cvh Pipeline 性能优势证明计划

状态：E0–E4 已完成；E5 尚未开始（L3 家族门禁未通过，且需要真实 ARM Linux 设备）。

更新时间：2026-08-18

基线提交：`448f545 feat: complete pipeline P1 model input fusion`

上位合同：[pipeline-module-design.md](pipeline-module-design.md)。P1 的实现和正确性证据
由 [pipeline-p1-implementation-plan.md](pipeline-p1-implementation-plan.md) 维护。
本文只回答一个问题：**在数值语义、线程数、输入输出和内存复用方式公平的前提下，
cvh Pipeline 是否比逐算子方案和 OpenCV 等价预处理链更快、更省中间内存？**

## 0. 当前事实和证据缺口

P1 已经证明：

- canonical Recipe 可以编译为 1 execution group、0 完整中间图、0-byte workspace；
- prepared run 为 0 heap allocation；
- packed F32 scalar fusion 相对 cvh staged 路径已有 1.20x–1.61x 的代表性收益；
- 窄 Nearest ARM NEON predicate 相对 scalar fused 已有约 3.06x 收益；
- packed U8/S8 scalar direct-store 相对 cvh staged 路径已有约 1.34x/1.33x 收益；
- NV12/NV21 已有真实 scalar 成本和正确性证据。

这些结果证明了“融合比 cvh 自己的 staged 路径更好”，但**尚不能证明 cvh Pipeline
整体快于 OpenCV 的完整预处理链**。现有报告没有把 OpenCV 的颜色转换、resize、
normalize、layout/pack、letterbox 和 quantize 全部纳入同一端到端计时，也没有覆盖
流式输入对 cache/memory traffic 的影响。

因此，当前对外只能陈述已记录的内部收益，不能提前使用“Pipeline 比 OpenCV 快”或
“边缘设备更快”的宽泛话术。

## 1. 要证明的四层结论

性能结论分层关闭，不能用低层证据替代高层结论。

| 层级 | 要证明的结论 | 合法证据 |
| --- | --- | --- |
| L1 结构优势 | 少物化完整中间图、少显式临时内存、prepared run 零分配 | `PipelineInfo`、allocator hook、descriptor byte accounting |
| L2 内部融合优势 | cvh fused 比语义等价的 cvh staged 链更快 | 同一 binary、同一输入、paired checksum/tolerance、稳定多轮采样 |
| L3 OpenCV 竞争优势 | cvh fused 比最快的等价 OpenCV 完整链更快 | 独立 OpenCV backend、完整端到端链、公平预分配和线程合同 |
| L4 边缘设备优势 | 优势能跨真实 ARM Linux 边缘设备复现 | 至少两个目标设备类别、固定 governor/affinity、dated report |

L3 没有关闭时，只能说“相对 cvh staged 更快”。L4 没有关闭时，只能绑定具体开发机
型号，不能外推到“边缘设备”。

## 2. 冻结的实验假设

### H1：完整中间图消除

canonical model-input Recipe 应保持：

~~~text
execution_groups = 1
full_frame_intermediates = 0
workspace_bytes = 0
allocations_per_prepared_run = 0
~~~

OpenCV baseline 允许预创建并复用所有 Mat；比较的是每帧必须写入的逻辑中间结果和
实际运行成本，不通过故意反复分配放大 OpenCV 成本。

### H2：融合收益来自完整链，而不是单算子宣传

主指标比较：

~~~text
color? -> resize/letterbox -> normalize -> quantize? -> layout/store
~~~

不得拿 cvh 完整链与 OpenCV 单独 `resize()` 比较，也不得用某个 cvh 单算子较慢来否定
整条融合链。报告同时保留逐 stage 时间，用于解释收益来源，但最终结论只看等价的
端到端输出。

### H3：streaming 模式应放大低访存优势

同一输入反复运行的 hot-cache 模式可能掩盖中间图成本，因此每个主要 case 同时运行：

- `hot`：重复使用同一输入、输出和临时 buffer；
- `streaming`：轮换输入和输出 ring，ring 总输入字节至少为 `max(64 MiB, 2 x LLC)`；
  无法可靠检测 LLC 时固定使用至少 64 MiB。

两种模式必须独立报告，不能只选择对 cvh 更有利的一种。

### H4：NEON 收益只绑定真实 predicate

只有 `candidate_route`、`actual_route` 和 `observed_isa` 同时为 NEON，才进入 NEON
结果。其他尺寸、Linear、NHWC、YUV 或量化路径不能继承窄 predicate 的数字。

## 3. 对比实现

每个能够语义对齐的 case 至少运行以下实现：

| 实现名 | 含义 |
| --- | --- |
| `cvh_staged` | 明确阻止 canonical fusion、但保持相同数值顺序的 cvh staged reference |
| `cvh_fused_scalar` | 同一 Recipe，forced scalar direct-store |
| `cvh_fused_auto` | 同一 Recipe，正常 dispatch，并记录 actual/observed route |
| `opencv_explicit` | 预分配的 OpenCV core/imgproc 完整链，包含最终 layout/store |
| `opencv_best_valid` | 所有数值等价 OpenCV 变体中的最快者；可能是 explicit 或适用时的 `blobFromImage` |

OpenCV 的候选必须先通过数值合同，才有资格进入 `opencv_best_valid`。例如
`blobFromImage` 无法表达逐通道 stddev 或当前 letterbox/quantize 合同时，不得把它
伪装为等价 baseline；反过来，能精确表达某个 case 时必须纳入，避免只挑较慢的
OpenCV 写法。

NV12/NV21、letterbox 和 quantize 若找不到数值等价的 OpenCV 组合，只关闭 L1/L2，
不得生成 L3 speedup。报告应明确写 `not_comparable` 及原因，而不是使用不等价输出。

## 4. 公平性合同

### 4.1 构建和线程

- cvh 和 OpenCV 使用同一机器、同一编译器家族、Release、相同目标架构；
- 记录双方实际编译选项、源码 revision 和 OpenCV build information；
- 主结果固定单线程：`cvh::setNumThreads(1)`、`cv::setNumThreads(1)`；
- 多线程结果只能作为独立 secondary profile，不能混入单线程几何平均值；
- 双方 Plan、Mat、resize tables、ring buffers 和临时存储都在计时前创建；
- 主计时只覆盖 per-frame run，不包含 Pipeline `prepare()`；prepare latency 单独报告；
- output checksum/误差验证在计时区外完成，但每次测量后必须消费输出，防止优化删除。

### 4.2 输入、输出和操作顺序

- 输入字节由同一确定性生成器产生；
- 输入 format、ColorSpec、row/plane stride 完全一致；
- 输出 shape、dtype、channel order 和 NCHW/NHWC 布局完全一致；
- OpenCV baseline 必须计入最终 planar pack 或 NHWC store；
- letterbox 必须计入 resize、padding 和最终 transform 对应的全部像素工作；
- quantize 必须计入 normalize 后的 round、zero point 和 saturate；
- 所有实现使用相同 caller-owned 输出复用策略。

### 4.3 运行顺序和系统噪声

- 每个 case 至少运行 3 个独立进程 session；
- 每个 session：warmup 20 frames、每个 sample 10 frames、至少 15 samples；
- 使用固定 Latin-square 顺序轮换实现，避免总让某个实现处于更冷或更热的位置；
- 记录 CPU、OS、温度/降频状态、线程 affinity 和 governor；
- Linux ARM 主报告固定 performance governor 并绑定同一性能核；
- sample 变异系数超过 3% 时标记不稳定并重跑，不从噪声结果得出结论。

## 5. Case 矩阵

### 5.1 Packed F32 主矩阵

| ID | 输入 | 输出 | 几何 | 布局 | 目的 |
| --- | --- | --- | --- | --- | --- |
| PF1 | 1280x720 BGR8 | 640x640 RGB F32 | Linear resize | NCHW | 常见检测输入 |
| PF2 | 1920x1080 BGR8 | 640x640 RGB F32 | Linear resize | NCHW | 高分辨率相机输入 |
| PF3 | 640x480 RGB8 | 224x224 RGB F32 | Linear resize | NCHW | 分类模型输入 |
| PF4 | 1280x720 BGR8 | 640x640 RGB F32 | Linear resize | NHWC | 移动/边缘 runtime 布局 |
| PF5 | 1280x720 BGR8 | 640x640 RGB F32 | Nearest resize | NCHW | 当前窄 NEON predicate |
| PF6 | 1280x720 BGR8 | 640x640 RGB F32 | Nearest letterbox | NCHW | 几何 metadata + NEON predicate |

PF1–PF6 是“Pipeline 相对 OpenCV 完整链”第一阶段的主要宣称集。normalize 参数固定为
公开模型常见的三通道 mean/stddev，并写入 case manifest，不在看过结果后调整。

### 5.2 Quantize 主矩阵

| ID | 输入 | 输出 | 几何 | 布局 | 目的 |
| --- | --- | --- | --- | --- | --- |
| PQ1 | 1280x720 BGR8 | 640x640 RGB U8 | Nearest resize | NCHW | U8 per-tensor direct-store |
| PQ2 | 1280x720 BGR8 | 640x640 RGB S8 | Nearest resize | NCHW | S8 per-tensor direct-store |
| PQ3 | 640x480 BGR8 | 224x224 RGB S8 | Linear resize | NHWC | 小输出和弱收益边界 |

PQ3 必须保留，即使它没有性能优势；它用于阻止只发布大图有利 case。只有 OpenCV
round/saturate 合同通过差分验证的 case 才进入 L3。

### 5.3 YUV 主矩阵

| ID | 输入 | 输出 | ColorSpec | 目的 |
| --- | --- | --- | --- | --- |
| PY1 | 1280x720 NV12 | 640x640 RGB F32 NCHW Linear | BT.709/Limited/Left | 双 plane F32 |
| PY2 | 1280x720 NV21 | 640x640 RGB F32 NCHW Linear | BT.709/Limited/Left | NV21 顺序 |
| PY3 | 1280x720 NV12 | 640x640 RGB S8 NCHW Linear | BT.709/Limited/Left | YUV + quantize |
| PY4 | 1280x720 NV21 | 640x640 RGB S8 NCHW Linear | BT.709/Limited/Left | NV21 + quantize |

另设一个经过验证可与 `cvtColorTwoPlane` 数值对齐的 OpenCV-common ColorSpec case；
具体 matrix/range/chroma 必须由独立差分结果确定，不能按 API 名字猜测。PY1–PY4 即使
无法与 OpenCV 对齐，仍保留 L1/L2 和 absolute latency 证据。

### 5.4 边界和反例矩阵

secondary profile 覆盖：

- 1x1、奇偶尺寸、非整数缩放、upscale；
- 641/1279 等 SIMD tail；
- padded/unaligned input 和 ROI；
- NHWC、Linear、非 2x 比例等 scalar fallback；
- hot/streaming 两种 cache 模式。

这些 case 不参与第一版主几何平均值，但任何超过 10% 的稳定性能回退必须单独列出，
不能从报告中删除。

## 6. 正确性门禁

性能采样之前必须按以下顺序通过：

1. `cvh_fused_scalar` 与 `cvh_staged` 使用 P1 冻结合同，逐 byte 或逐元素一致；
2. `cvh_fused_auto` 与 forced scalar 一致，并验证实际 route/ISA；
3. OpenCV 候选与独立 reference 比较，而不是让 cvh 或 OpenCV 互相充当唯一 oracle；
4. 只有通过数值门禁的 OpenCV 候选进入 speedup 计算。

冻结误差原则：

- Nearest、layout 和 U8/S8 无损路径要求精确一致；
- Linear 若双方 U8 插值存在合法的 1-LSB 实现差异，F32 normalize tolerance 按
  `1 / abs(stddev[channel]) + floating_epsilon` 从输入误差传播计算，不使用事后常数；
- quantize 只有 rounding、NaN/Inf、zero point 和 saturate 合同一致时才比较；
- checksum 不一致时必须同时输出 max absolute error、max relative error、不同元素数和
  首个差异位置；不得仅关闭 checksum 继续计时。

## 7. 指标和报告

### 7.1 延迟

每个实现至少记录：

- p50、p90、p95、min、max ns/frame；
- FPS 和 MPix/s；
- 相对 `cvh_staged` 与 `opencv_best_valid` 的 paired speedup；
- 3 个独立 session 的几何平均值和 95% bootstrap confidence interval；
- prepare latency，单独于 run latency。

### 7.2 内存和执行属性

- semantic stage 和 execution group 数量；
- full-frame intermediate 数量；
- caller-owned output、workspace 和显式临时 Mat 字节数；
- prepared run allocation count；
- 基于 descriptor 的逻辑读写字节估算。

逻辑读写估算不得宣传为“物理内存只读一次”。Linux 目标设备可额外用硬件计数器记录
cycles、instructions、cache miss；只有具备 memory-controller 计数器时才报告实际
DRAM bytes。

### 7.3 路径可观察性

CSV 必须记录：

- algorithm path；
- candidate route；
- actual dispatch path；
- observed ISA；
- fallback reason；
- thread count；
- cache mode。

## 8. 接受门槛和对外话术

### 8.1 单 predicate 优势

某个精确 predicate 只有同时满足以下条件，才能公开说“快于 OpenCV”：

- 输出通过数值门禁；
- 3 个独立 session 的 median speedup 至少 1.20x；
- paired 95% confidence interval 下界至少 1.10x；
- hot 和 streaming 至少一种通过，话术必须写明模式；
- 另一 cache 模式不得稳定慢于 OpenCV 超过 10%；
- 没有通过额外线程、隐藏分配或省略 layout/store 获得优势。

### 8.2 Packed F32 家族优势

只有 PF1–PF6 满足：

- 对 `opencv_best_valid` 的几何平均 speedup 至少 1.20x；
- 没有主 case 稳定低于 0.95x；
- PF5/PF6 的 observed ISA 符合窄 NEON predicate；

才能使用“cvh Pipeline 在该设备的常见 packed 模型输入矩阵上整体快于 OpenCV
完整链”。否则只发布通过门槛的精确 case，不做家族级结论。

### 8.3 边缘设备优势

“边缘设备性能优势”要求 L4：至少两个 ARM Linux 设备类别重复完整 stable 测量，并且
Packed F32 家族门槛在两个设备上都成立。Apple M 系列结果可以作为 ARM 开发机证据，
但不能单独关闭边缘设备话术。

### 8.4 失败处理

- 不降低 tolerance、case 数、采样数或 speedup 门槛；
- 不删除慢 case；
- 区分算法问题、dispatch 未命中、cache 模式和 baseline 实现问题；
- 优化候选失败时保留 dated report，并从产品路径回退；
- 如果结果显示只具有内存属性而没有时间优势，就只发布 L1，不包装成 L3/L4。

## 9. Header-only 和 OpenCV 边界

新增 OpenCV Pipeline benchmark 必须使用独立 compare target，例如：

~~~text
cvh_benchmark_pipeline_compare
  pipeline_compare_header_backend.cpp
  pipeline_compare_opencv_backend.cpp
~~~

它只在 `CVH_ENABLE_OPENCV_COMPARE=ON` 时构建。必须保持：

- `cvh_benchmark_pipeline_header` 只链接 `cvh::headers`；
- `cvh::headers` 和安装包不增加 OpenCV include/link requirement；
- OpenCV 类型不出现在 `include/cvh/pipeline/` 或 `include/cvh/recipes/`；
- 默认 configure 不调用 `find_package(OpenCV)`；
- public-header、ODR、install consumer 和 binary dependency 门禁继续通过。

OpenCV 在这里是实验对手，不是产品运行时依赖。

## 10. 实施批次

| 批次 | 工作 | 状态 | 完成条件/当前证据 |
| --- | --- | --- | --- |
| E0 | 冻结本文、基线、case 和话术门槛 | 已完成 | 2026-08-18：本文进入文档索引；L1–L4、PF/PQ/PY 矩阵和 1.20x/CI 门槛已冻结 |
| E1 | 建立 machine-readable case manifest 和 OpenCV Pipeline compare target | 已完成 | 2026-08-18：新增 PF1–PF6 manifest、独立 OpenCV backend 和 `cvh_benchmark_pipeline_compare`；复用 Release compare cache 编译通过，quick/hot smoke 完成 |
| E2 | 关闭正确性和公平性合同 | 已完成 | 2026-08-18：PF1–PF6 hot/streaming 全部通过独立 oracle；CTest smoke、header-only/install 门禁和产品/compare 二进制依赖检查通过 |
| E3 | Apple M5 hot/streaming stable 探索 | 已完成 | 2026-08-18：3 个正式 session、CSV/metadata/aggregate/report 已生成；L1 通过，L3 PF1–PF6 家族门禁失败，CV 超标行原样保留 |
| E4 | 针对失败 hot path 优化并回归 | 已完成 | `linear-gather-normalize-neon-v1` 失败并回退；`normalize-u8-lut-v1` 通过全部本地门禁并保留；PF5 hot 关闭单 predicate L3，PF1–PF6 家族仍失败 |
| E5 | 两类 ARM Linux 边缘设备复测 | 未开始 | 当前 M5 的 L3 家族门禁已失败，不提前投入 L4 复测；开始 E5 仍需两类真实 ARM Linux 设备 |

E1–E5 开始后，本文成为实时状态所有者；每批开始、完成、回退或阻塞时立即更新状态、
命令和证据。

### 10.1 实时执行记录

2026-08-18，E1：

- case manifest：`benchmark/pipeline_model_input_cases.csv`，第一阶段冻结 PF1–PF6；
- compare target：`cvh_benchmark_pipeline_compare`，仅在
  `CVH_ENABLE_OPENCV_COMPARE=ON` 时构建；OpenCV include 保留在独立 backend translation
  unit，`cvh::headers` 和默认 configure 边界不变；
- 构建配置：复用 `build-core-mat-neon-compare`，其身份为 Release、optimization ON、
  benchmark ON、OpenCV compare ON；
- 构建命令：

  ~~~sh
  cmake -S . -B build-core-mat-neon-compare
  cmake --build build-core-mat-neon-compare \
    --target cvh_benchmark_pipeline_compare --parallel 2
  ~~~

- E1 smoke 命令（仅证明基础设施和正确性路径可运行，不作为性能结论）：

  ~~~sh
  ./build-core-mat-neon-compare/cvh_benchmark_pipeline_compare \
    --profile quick --cache-mode hot --warmup 1 --iters 1 --repeats 1 \
    --ring-mib 4 --session 1 \
    --output build-core-mat-neon-compare/pipeline-e1-smoke.csv
  ~~~

- smoke 结果：PF1/PF4/PF5 的四个实现均通过校验；CSV schema 包含 dispatch/ISA、
  中间图、workspace、显式临时内存、误差和逐 sample 字段，并生成同名 metadata。
  该单 sample 结果 `CV=0` 没有统计意义，严禁用于对外性能话术。

2026-08-18，E2：

- OpenCV 完整链按合法可交换顺序优化：NCHW 使用
  `resize -> split U8 -> per-plane convert/normalize`，NHWC 使用
  `resize -> convert -> transform(channel reorder + normalize)`；避免把输入分辨率上的
  冗余 BGR/RGB swap 计入 baseline；
- PF1–PF6 使用逐通道 stddev，OpenCV DNN `blobFromImage` 的单一 scalefactor 不能表达该
  合同，因此不属于 valid candidate；本轮经过重排和预分配的 `opencv_explicit` 就是
  `opencv_best_valid`，CSV 的 `speedup_vs_opencv` 以它为分母；
- cvh dispatch mode 在每个 sample 计时前设置，不再计入 per-frame latency；
- `--profile stable --cache-mode both --warmup 1 --iters 1 --repeats 1`
  正确性运行覆盖 PF1–PF6；staged、fused scalar、fused auto 与 OpenCV 全部通过冻结 oracle；
- 注册并通过可选 CTest：`cvh_benchmark_pipeline_compare_quick_correctness`；
- CSV 将 OpenCV 不可观测的 execution group/allocation 写为 `not_reported`，不再误写为 0；
  cvh 行记录真实 `allocations_per_run`，OpenCV 行记录显式完整中间图和临时字节；
- `./scripts/check_public_headers.sh` 与 `./scripts/check_header_only_contract.sh` 通过；
- `otool -L` 证明 `cvh_benchmark_pipeline_header` 只依赖系统 C++/System 库；OpenCV 仅存在于
  可选 compare executable，产品 benchmark 和安装目标未增加 OpenCV 依赖。

2026-08-18，E3 采样调整：

- 初始 session 1 及其 `rerun-1` 均原样保留；`iters=10` 时 PF2/PF3/PF4/PF5 的部分
  streaming sample 出现 `CV > 3%`，因此按门禁判为不稳定，不能进入结论；
- 小图 PF3 的 OpenCV sample 在 `iters=10` 时计时窗口仅约 0.35 ms，容易被调度噪声主导。
  后续三个正式 session 将 `iters` 从 10 提高到 50，其他冻结条件不变。这是增加每个
  sample 的工作量，不减少 warmup、repeat、case 或门槛；三次正式 session 使用完全相同
  的参数，旧文件不覆盖。
- 首次 `iters=50` 诊断进一步发现，streaming output ring 的 `vector` 零初始化在 macOS
  上可能使用 demand-zero page；第一个 sample 因 caller-owned output 首次写入而出现
  明显离群值。benchmark 已改为在计时前用非零值显式预触碰全部输出页。该修正不触碰
  产品实现，也不在计时区做额外工作；修正前结果继续保留但标为无效。
- 最终审计补充了 timed sample 后的 output consumption：每个 sample 在计时区外 checksum
  其最后输出帧，明确阻止 dead-store elimination。此前汇编/实测未显示 store 被删除，
  但旧 aggregate 仍保留为诊断，最终候选结论只采用带 `consumed` 名称的新 session。
- E4 候选 `linear-gather-normalize-neon-v1` 已回退：PF1 hot probe 的
  `cvh_fused_auto=1.253 ms` 慢于 `cvh_fused_scalar=1.049 ms`，PF4 分别为
  `1.241 ms` 与 `1.027 ms`。候选虽然 byte-exact 且报告真实 NEON，但逐 lane gather 的
  成本超过向量 normalize 收益；因此未保留其 predicate 或产品代码，冻结门槛不变。

2026-08-18，E3 正式结果：

- 有效输入为 `session-1-pretouch`、`session-2-pretouch` 和
  `session-3-pretouch`；均使用 Release、单线程、warmup 20、50 frames/sample、15 samples、
  hot + 64 MiB streaming ring；早期 `iters=10` 和首次未预触碰结果只作为失败诊断保留；
- 设备：Apple M5 10-core MacBook Air (`Mac17,3`)、16 GiB、macOS 26.4.1；cvh 基线
  `448f545` dirty working tree，OpenCV `d48bf69`、4.14.0-pre；
- [基线 aggregate CSV](../benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-aggregate-v2.csv)
  与 [基线报告](../benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-report-v2.md)
  记录：PF1–PF6 对 OpenCV 家族几何均值 hot `0.500x`、streaming `0.495x`；
  fused 相对 staged 家族几何均值 hot `2.300x`；
- L1 结构门禁通过；L3 家族门禁明确失败。PF5 呈现约 2.5–2.7x 的 NEON Nearest
  探索性信号，但所需 session 有 CV 超标，不能发布正式 predicate 话术；PF6 hot 接近
  parity、streaming 略慢；PF1–PF4 Linear 明显慢于优化后的 OpenCV 完整链；
- macOS 无 governor 接口，本轮未固定 CPU affinity，也未捕获可审核的温度/降频数据；
  这些限制与 CSV 中的稳定性失败一并公开，不能选择性忽略。

正式 session 的命令模板（`N=1,2,3`，baseline 与 LUT candidate 仅输出文件名和
`CVH_BENCHMARK_CVH_COMMIT` 标签不同）：

~~~sh
CVH_BENCHMARK_CVH_COMMIT=448f545a35522584336a29a7538b2a87ffdd0e57-dirty \
CVH_BENCHMARK_OPENCV_COMMIT=d48bf69f65444a13f8a34b8982b083c1b78fa0e8 \
CVH_BENCHMARK_OPENCV_SOURCE=../opencv \
CVH_BENCHMARK_OPENCV_BUILD_DIR=../opencv/build-slim \
CVH_BENCHMARK_CPU_MODEL='Apple M5 (10 cores, MacBook Air Mac17,3)' \
./build-core-mat-neon-compare/cvh_benchmark_pipeline_compare \
  --profile stable --cache-mode both --warmup 20 --iters 50 --repeats 15 \
  --threads 1 --session N --ring-mib 64 \
  --output benchmark/results/opencv/pipeline/stable/<dated-session-N>.csv
~~~

2026-08-18，E4 保留候选：

- `normalize-u8-lut-v1` 利用 resize/letterbox 后仍为 U8 的冻结语义，在栈上生成每通道
  256-entry F32 normalize 表；它消除重复除法但保持同一表达式、byte-exact 输出、零堆
  分配、零 workspace 和零完整中间图；
- targeted probe：PF1 scalar fused 约 `1.049 -> 0.889 ms`，PF4
  `1.027 -> 0.890 ms`，PF5 scalar `0.376 -> 0.346 ms`；
- 3-session [最终候选 aggregate CSV](../benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-aggregate-lut-consumed-v2.csv)
  与 [最终候选报告](../benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-report-lut-consumed-v2.md)
  记录：PF1 hot 比值 `0.256x -> 0.295x`，PF4 hot `0.324x -> 0.373x`；
  fused 相对 staged 家族几何均值 hot `2.300x -> 2.489x`；
- `cvh_benchmark_pipeline_compare_quick_correctness`、`cvh_test_pipeline`、
  `cvh_pipeline_zero_allocation_smoke`、Pipeline header compile 和 ODR smoke 全部通过；
- optimization OFF 的 Release Pipeline/零分配测试通过；ASan+UBSan Pipeline/零分配测试
  通过；x86_64 cross 配置编译通过。当前主机未安装/启用 x86_64 运行支持，cross CTest
  以 macOS error 86 `BAD_COMMAND` 未运行，因此只记录 compile gate，不伪报 runtime pass；
- 候选整体改善 L2，但不能据此宣称 packed F32 家族追平 OpenCV。下一轮若继续 E4，
  应优先研究可复用的 separable/resample table 与真正连续 load 的 Linear kernel，而
  不是扩应用层适配器。
- PF5 hot 是唯一关闭 L3 单 predicate 门禁的结果：Apple M5、1280x720 BGR8 到
  640x640 RGB F32/NCHW、Nearest、hot cache 下，对 `opencv_best_valid` 的三 session
  几何平均为 `2.693x`，95% CI 下界 `2.678x`，3/3 session 的 cvh/OpenCV 行均满足
  `CV <= 3%`，actual/observed route 为 NEON。只允许绑定上述设备、shape、插值、布局和
  cache mode 的话术；PF5 streaming、PF1–PF6 家族和“边缘设备更快”仍未关闭。
- 最终仓库门禁：`scripts/check_docs.sh`、`git diff --check`、
  `scripts/check_header_only_contract.sh`、benchmark JSON/CSV schema 和 report script 语法检查
  通过；核心默认 configure 仍不查找或链接 OpenCV。
- compare runner 和 aggregate reporter 均拒绝覆盖已有 CSV/metadata/report；现有输出上的
  overwrite smoke 已按预期失败，后续修正必须使用新文件名。

## 11. 构建配置和产物

复用或建立具有稳定身份的配置：

~~~text
build-pipeline-proof-release
build-pipeline-proof-opencv
build-pipeline-proof-opt-off
build-pipeline-proof-sanitize
build-pipeline-proof-x86-cross
~~~

建议结果位置：

~~~text
benchmark/results/internal/pipeline/proof/<profile>/
benchmark/results/opencv/pipeline/<profile>/
~~~

每份 dated 结果必须同时包含 CSV、metadata 和 Markdown report，记录 cvh/OpenCV commit、
dirty 状态、compiler、CPU、OS、build type、线程、cache mode、采样参数和完整命令。
历史报告不可覆盖。

## 12. 第一执行顺序

第一轮只做以下工作，不进入 OpenCV/V4L2/ROS 2/推理 runtime 应用集成：

1. 将 PF1–PF6 写入 machine-readable manifest；
2. 新建独立 OpenCV Pipeline compare backend；
3. 实现 `cvh_staged/cvh_fused_scalar/cvh_fused_auto/opencv_explicit`；
4. 先关闭 PF1、PF4、PF5 三个 anchor 的正确性；
5. 加入 hot/streaming、paired order 和稳定统计；
6. 在 Apple M5 生成第一份探索报告；
7. 根据结果决定优化方向，而不是提前扩 Pipeline API。

第一份实验的目标不是得到好看的数字，而是判断优势究竟来自 fusion、NEON、layout
direct-store 还是 cache/memory traffic，并找出 cvh 仍落后的准确 predicate。
