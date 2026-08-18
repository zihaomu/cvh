# Pipeline P1 NEON Candidate — 2026-08-10

这份报告评估 P1.5 第一条 ARM NEON predicate：packed BGR/RGB U8、Nearest resize、
输入宽度精确为输出宽度的 2 倍、NCHW F32 输出，且输出至少 256 pixels。其他合法链
继续使用 scalar fused 或 staged，不从平台名称推断 NEON。

## 测量合同

- 源码：`3857147+working-tree-p1.5`；Release，Clang 21.0.0；Apple M5 / arm64。
- 单线程；warmup 3，iters 3，repeats 7；使用每帧中位数。
- 同一二进制分别运行 forced scalar 与 Auto；CSV 记录 candidate、actual 和 observed
  route。只有 executor 实际进入 NEON 后 `observed_isa=neon`。
- staged、fused scalar、fused Auto 三行 checksum 必须相同；Plan、Workspace 和输出在
  计时前创建并复用。

## 接受的 predicate

| Case | Staged | Fused scalar | Fused Auto/NEON | NEON / scalar | NEON / staged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1280×720 BGR → 640×640 RGB F32 NCHW Nearest | 0.950 ms | 0.356 ms | 0.120 ms | 2.97× | 7.94× |

Auto 行记录 `candidate_route=neon`、`dispatch_path=neon`、`observed_isa=neon`；三行
checksum 都是 `2170428856288077002`。NEON 行仍保持 1 execution group、0 完整
中间图和 0-byte workspace。

Linear、非 2× 横向比例、NHWC、小图和非 ARM/优化关闭构建不进入这条 predicate。
本次同一 stable run 中这些 Linear 行的 Auto/observed route 均为 scalar，说明候选没有
扩大到未测量矩阵。

## 结论

该窄 NEON 候选在目标 ARM 开发机上相对 scalar fused 有稳定 2.97× 中位数收益，数值
和内存合同不变，因此接受进入产品路径。后续扩展必须建立新的 predicate 和证据，
不能把这里的收益外推到 Linear、NHWC 或任意 resize 比例。

原始证据：[CSV](2026-08-10-p1-neon-candidate.csv)；
[metadata](2026-08-10-p1-neon-candidate.meta.json)。
