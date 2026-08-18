# Pipeline P1.7 Quantize Hot Path Restored — Intermediate

这份中间快照验证拆分无 padding 热路径后，首版 S8 回退已消失。它随后被包含大尺寸
S8 Nearest case 的 expanded stable 报告取代，但作为修复过程证据保留。

- Release / Apple M5 / arm64 / 单线程；warmup 3，iters 3，repeats 7；
- packed U8 Nearest/NCHW：staged `1.950 ms`，fused `1.465 ms`，`1.33×`；
- packed S8 Linear/NHWC：staged `0.262 ms`，fused `0.253 ms`，`1.035×`；
- checksum 分别为 `18335021459575583639` 和 `3045878766183144906`，各自三行一致；
- 两条 quantize route 都是 scalar，1 group、0 intermediate、0-byte workspace。

小尺寸 S8 收益较窄，因此最终接受证据增加了与 U8 对称的 1280×720→640×640
Nearest/NCHW case；本报告不用于外推普遍收益。

原始证据：[CSV](2026-08-10-p1-quantize-hotpath-restored.csv)；
[metadata](2026-08-10-p1-quantize-hotpath-restored.meta.json)。
