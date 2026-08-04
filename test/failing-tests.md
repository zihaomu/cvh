# 测试状态台账

- 更新时间：2026-08-04
- 当前状态：默认 UI-enabled header-only 配置下，
  `cvh_test_core` 的 213 个测试和 `cvh_test_imgproc` 的 193 个测试均无失败、
  无 skip。native GEMM ISA 测试由 `cvh_test_gemm_isa` 使用
  `cvh::headers` 独立执行，不重复注册到默认 Core 基线。
- scalar-only 配置仅保留为本地诊断能力，不属于托管 CI 门禁。
- 2026-08-04 Phase 2-P0 收口复跑中，scalar-only Core 200/200
  通过，13 个 UI-only case 按预期 skip，Imgproc 193/193 通过；
  Phase 2 upstream contract 24/24 通过。
- `cvh_test_highgui` 在 `CVH_HIGHGUI_HEADLESS=1` 下验证 API 合同；macOS
  AppKit Runtime 实际窗口 smoke 也已通过，当前无已知 HighGUI 失败。

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
