# Pipeline P1.6 YUV/Letterbox Candidate — Hot Path Restored

这份报告评估 P1.6 的两条新路径：NV12/NV21 scalar direct-store，以及 packed
letterbox Nearest 的 ARM NEON predicate。它同时复测 P1.5 普通 packed Nearest，防止
letterbox 共用逻辑拖慢已有热路径。

## 测量合同

- 源码：`3857147+working-tree-p1.6-hotpath-restored`；Release，Clang 21.0.0；
  Apple M5 / arm64；
- 单线程；warmup 3，iters 3，repeats 7；使用每帧中位数；
- forced scalar 与 Auto 使用同一 Plan/输入/输出合同；staged/scalar/Auto 行 checksum
  必须一致；只有 executor 实际进入 NEON 后才记录 `observed_isa=neon`；
- letterbox case 为 1280×720 BGR8 → 640×640 RGB F32 NCHW、Nearest、pad=114，
  content 为 640×360，上下各 pad 140；NV12 case 为 BT.709/Limited/Left、Linear。

## 结果

| Case | Staged | Scalar fused | Auto/NEON | NEON / scalar | Auto / staged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 普通 packed Nearest | 0.952 ms | 0.368 ms | 0.119 ms | 3.09× | 7.98× |
| packed letterbox Nearest | 1.200 ms | 0.597 ms | 0.251 ms | 2.38× | 4.79× |
| NV12 Linear scalar | — | 10.281 ms | — | — | — |

普通 Nearest 三行 checksum 为 `2170428856288077002`，scalar fused 已恢复到 P1.5
约 0.36 ms 的量级。letterbox 三行 checksum 为 `11849062763261262192`，Auto 行的
candidate、dispatch 和 observed ISA 均为 NEON；NV12 checksum 为
`9366306608337106819`，route 明确为 scalar。

三条 fused 路径均为 1 execution group、0 完整中间图和 0-byte workspace。NV12 当前
报告原始 scalar 成本，不虚构 staged speedup；后续 YUV NEON 必须拥有独立 predicate、
checksum 和收益数据。

## 结论

接受 packed letterbox 的窄 NEON predicate：U8 BGR/RGB、Nearest、NCHW、输入宽度为
content 宽度 2 倍、content 至少 256 pixels。普通 resize 无分支热路径保留；Linear、
NHWC、其他比例及 YUV 继续使用 scalar。首版共用 loop 的性能回退记录在前一份失败
报告中，没有留在最终路径。

原始证据：[CSV](2026-08-10-p1-yuv-letterbox-hotpath-restored.csv)；
[metadata](2026-08-10-p1-yuv-letterbox-hotpath-restored.meta.json)。
