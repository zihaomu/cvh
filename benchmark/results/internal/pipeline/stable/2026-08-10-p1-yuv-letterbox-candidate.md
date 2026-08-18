# Pipeline P1.6 YUV/Letterbox First Candidate — Rejected

这份快照保留 P1.6 第一版共用 resize/letterbox scalar loop 的失败证据。它的 YUV、
letterbox checksum 和 NEON route 正确，但给普通 packed Nearest 热路径引入了不可接受的
性能回退，因此该实现没有留在产品路径。

## 测量合同

- 源码：`3857147+working-tree-p1.6`；Release，Clang 21.0.0；Apple M5 / arm64；
- 单线程；warmup 3，iters 3，repeats 7；使用每帧中位数；
- Plan、Workspace 和输出在计时前创建并复用；同 case 的 staged/scalar/Auto checksum
  必须一致，NEON 只在实际执行后记录 `observed_isa=neon`。

## 回退原因

普通 `1280×720 → 640×640` packed Nearest 的 scalar fused 中位数为 `0.688 ms`；
P1.5 同合同热路径约为 `0.356 ms`。根因是第一版 executor 为 resize 和 letterbox 共用
逐像素 padding 分支，即使普通 resize 没有 padding 也进入了更宽的控制流。

letterbox 自身在本快照中为 staged `1.264 ms`、scalar fused `0.627 ms`、NEON
`0.252 ms`，checksum 均为 `11849062763261262192`；NV12 Linear scalar 为
`10.126 ms`。这些行证明候选功能可运行，但不能抵消对已有 Supported 热路径的回退。

处理决定：保留本报告，拆回普通 resize 的无 padding 分支热路径，并使用新的 dated
快照重新测量；不覆盖本 CSV/metadata。

原始证据：[CSV](2026-08-10-p1-yuv-letterbox-candidate.csv)；
[metadata](2026-08-10-p1-yuv-letterbox-candidate.meta.json)。
