# Pipeline P1 Scalar Fused Baseline — 2026-08-10

这份报告冻结 P1.3 首个 packed-f32 scalar fused baseline。它比较同一工作树、同一输入
上的 prepared `staged_p0` 和 `fused_p1`，不包含 OpenCV，也不代表 ARM NEON 最终门槛。

## 测量合同

- 源码：`3857147+working-tree-p1.3`；Release，Clang 21.0.0；Apple M5 / arm64。
- 单线程；warmup 3，iters 3，repeats 7；表中使用每帧中位数。
- 两条路径都包含 prepared `PipelinePlan::run()` 的运行期校验；Plan、Workspace 和输出
  在计时前创建并复用。
- 每个 case 计时前逐 byte 比较输出；CSV checksum 相同才计入结果。
- `staged_p0` 使用等价的 resize → channel swap → normalize → layout 顺序作为独立
  oracle；通道交换与逐通道 resize 可交换，数值结果精确一致。

## 结果

| Case | Staged ms | Fused ms | Fused 加速 | groups | intermediates | workspace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280×720 BGR → 640×640 RGB F32 NCHW Linear | 1.529 | 1.018 | 1.50× | 1 | 0 | 0 B |
| 1920×1080 BGR → 640×640 RGB F32 NCHW Linear | 1.536 | 1.031 | 1.49× | 1 | 0 | 0 B |
| 640×480 RGB → 224×224 RGB F32 NCHW Linear | 0.191 | 0.141 | 1.35× | 1 | 0 | 0 B |
| 1280×720 BGR → 640×640 RGB F32 NHWC Linear | 1.241 | 1.012 | 1.23× | 1 | 0 | 0 B |
| 1280×720 BGR → 640×640 RGB F32 NCHW Nearest | 0.950 | 0.360 | 2.64× | 1 | 0 | 0 B |

staged 640×640 行使用 4 个 execution group、3 张完整中间图和 7,372,800-byte
workspace；224×224 行使用 903,168-byte workspace。fused 行均直接写最终 tensor，
因此是 1 group、0 完整中间图和 0-byte workspace。所有行的 candidate、actual 和
observed route 都是 scalar。

## 结论

首个 scalar fusion 在本机五个冻结 case 上均快于 staged，最小中位数收益为 1.23×，
同时消除了完整中间图。这足以建立 P1.3 baseline，但暂不把 1.23×写成跨平台硬门槛；
P1.5 需要在目标 ARM 设备上重新冻结 NEON 与 scalar 的稳定门槛。

原始证据：[CSV](2026-08-10-p1-scalar-fused-baseline.csv)；
[metadata](2026-08-10-p1-scalar-fused-baseline.meta.json)。
