# 测试状态台账

- 更新时间：2026-07-25
- 当前状态：`cvh_test_core` 和 `cvh_test_imgproc` 均无失败、无 skip。

## Core upstream 边界

`test/upstream/opencv/core/channel_manifest.json` 中的 Mat-only 兼容子集分为：

- `PASS`：已在 `test/core/upstream/mat_channel_upstream_test.cpp` 落地并执行。
- `OUT_OF_SCOPE`：OpenCV `OutputArray` 重载不属于当前公开 API，不注册成
  GTest，也不以永久 `GTEST_SKIP` 冒充测试。

当前两项 `OUT_OF_SCOPE` 为：

- `Core_Mat.reinterpret_OutputArray_8UC3_8SC3`
- `Core_Mat.reinterpret_OutputArray_8UC4_32FC1`

只有当项目决定把 `OutputArray` 纳入公开 API 时，才重新评估这两项；在此之前它们
不是实现故障，也不计入可执行测试数。

## 维护规则

1. 失败测试必须保留为可执行失败，不能改成无期限 skip。
2. 产品边界外的 upstream case 记录为 `OUT_OF_SCOPE`，并写明重新评估条件。
3. manifest 中不得记录本机绝对路径。
4. case 状态、consumer 文件路径和本台账必须在同一变更中更新。
