# Pipeline P1.7 U8/S8 Quantize Scalar Evidence

这份报告关闭 P1.7 per-tensor U8/S8 scalar direct-store 性能门禁，并继续复测已有
F32、letterbox 和 NV12 路径，避免为量化收益引入旧路径回退。

## 测量合同

- 源码：`3857147+working-tree-p1.7-large-s8-evidence`；Release，Clang 21.0.0；
  Apple M5 / arm64；
- 单线程；warmup 3，iters 3，repeats 7；每帧中位数；
- staged/scalar/Auto 使用相同输入和量化参数：`scale=0.025`，U8 zero point=128，
  S8 zero point=0；checksum 必须一致；
- P1.7 当前没有 NEON quantize predicate，candidate/actual/observed route 均为 scalar。

## 接受结果

| Case | Staged | Fused scalar | Speedup | Checksum |
| --- | ---: | ---: | ---: | ---: |
| packed U8 Nearest/NCHW 1280×720→640×640 | 1.944 ms | 1.459 ms | 1.33× | 18335021459575583639 |
| packed S8 Nearest/NCHW 1280×720→640×640 | 1.941 ms | 1.456 ms | 1.33× | 2609738794107954583 |

两条 fused Plan 都是 1 execution group、0 完整中间图、0-byte workspace。较小的 S8
Linear/NHWC case 为 staged `0.262 ms`、fused `0.256 ms`，仍无回退但不作为主要收益
声明。NV12 S8 Linear scalar 为 `11.955 ms`，checksum
`13423987305786661862`；由于没有可表达 multi-plane 的 staged 路径，只报告原始成本。

已有 F32 packed Nearest scalar 为 `0.362 ms`、NEON `0.119 ms`，letterbox scalar
`0.605 ms`、NEON `0.277 ms`；checksum 保持既有值，说明 quantize 接入没有再次破坏
P1.5/P1.6 热路径。

## 结论

接受 per-tensor U8/S8 scalar direct-store predicate。它保留 ordered normalize →
quantize → layout 语义，并相对 staged 去掉 4 个完整中间图。当前不增加 NEON 或
per-channel 量化；未来候选必须另建数值和性能证据。

原始证据：[CSV](2026-08-10-p1-quantize-large-s8-evidence.csv)；
[metadata](2026-08-10-p1-quantize-large-s8-evidence.meta.json)。
