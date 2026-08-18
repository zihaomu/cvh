# Pipeline P1.7 Quantize First Scalar Candidate — Rejected

这份快照保留 P1.7 第一版 quantized direct-store loop 的失败证据。U8 已有收益，但
S8 Linear/NHWC 比 staged 更慢，因此共用 resize/letterbox 分支的版本没有留在产品
路径。

## 测量合同

- 源码：`3857147+working-tree-p1.7`；Release，Clang 21.0.0；Apple M5 / arm64；
- 单线程；warmup 3，iters 3，repeats 7；使用中位数；Plan/Workspace/输出复用；
- staged/scalar/Auto checksum 必须逐 case 一致；P1.7 quantize candidate 只选择 scalar。

## 回退原因

| Case | Staged | Fused scalar | Speedup |
| --- | ---: | ---: | ---: |
| packed U8 Nearest/NCHW 1280×720→640×640 | 1.947 ms | 1.629 ms | 1.20× |
| packed S8 Linear/NHWC 640×480→224×224 | 0.262 ms | 0.272 ms | 0.965× |

S8 checksum `3045878766183144906` 一致，说明问题是性能而非正确性。根因是第一版
quantized executor 即使执行普通 resize，也保留逐像素 letterbox/padding 分支。

处理决定：保留本快照，拆出无 padding 的 quantized resize 热路径，并以新的 dated
报告复测；不降低门槛或覆盖原始文件。

原始证据：[CSV](2026-08-10-p1-quantize-scalar.csv)；
[metadata](2026-08-10-p1-quantize-scalar.meta.json)。
