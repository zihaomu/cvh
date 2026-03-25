# 测试未通过台账（接口未对齐）

- 更新时间：2026-03-25
- 作用：记录当前 `test/` 下未通过、未运行或显式挂起（skip/pending）的测试及原因。

## 1. 构建状态（2026-03-25）

已解锁此前阻塞 `cvh_test_core` 的编译问题：

- `Error::Code` 增加了 `StsOutOfMem` / `StsBadType` 兼容别名，`cvh_test_core` 可完成编译与执行。
- 当前 `cvh_test_core` 运行结果：`PASSED=28`，`SKIPPED=16`（均为显式 pending）。

## 2. Upstream Channel 迁移用例（已纳入台账但当前未通过/未完成）

来源：`test/upstream/opencv/core/channel_manifest.json`  
当前状态：`PASS_NOW = 0`，`PENDING_CHANNEL = 17`

### 2.1 merge/split 相关（6）

| 用例 | 当前状态 | 原因 | 解锁条件 |
|---|---|---|---|
| `Core_Merge.shape_operations` | `PENDING_CHANNEL` | `merge/split` API 未暴露 | 实现 `cvh::merge` + `cvh::split` |
| `Core_Split.shape_operations` | `PENDING_CHANNEL` | 同上 | 同上 |
| `Core_Merge.hang_12171` | `PENDING_CHANNEL` | `merge` 缺失 | `cvh::merge` + ROI/stride 安全语义 |
| `Core_Split.hang_12171` | `PENDING_CHANNEL` | `split` 缺失 | `cvh::split` + ROI/stride 安全语义 |
| `Core_Split.crash_12171` | `PENDING_CHANNEL` | `split` 缺失 | `cvh::split` + ROI/stride 安全语义 |
| `Core_Merge.bug_13544` | `PENDING_CHANNEL` | `merge` 缺失 | `cvh::merge` 通道拼接语义对齐 |

### 2.2 reinterpret / OutputArray 相关（4）

| 用例 | 当前状态 | 原因 | 解锁条件 |
|---|---|---|---|
| `Core_Mat.reinterpret_Mat_8UC3_8SC3` | `PENDING_CHANNEL` | `Mat::reinterpret` 未实现 | 实现 `cvh::Mat::reinterpret` |
| `Core_Mat.reinterpret_Mat_8UC4_32FC1` | `PENDING_CHANNEL` | 同上 | 同上 |
| `Core_Mat.reinterpret_OutputArray_8UC3_8SC3` | `PENDING_CHANNEL` | OutputArray 兼容层缺失 | 补齐 `cvh` OutputArray 兼容接口 |
| `Core_Mat.reinterpret_OutputArray_8UC4_32FC1` | `PENDING_CHANNEL` | 同上 | 同上 |

### 2.3 MatExpr / compare 相关（4）

| 用例 | 当前状态 | 原因 | 解锁条件 |
|---|---|---|---|
| `Core_MatExpr.issue_16655` | `PENDING_CHANNEL` | 多通道 `MatExpr` compare/type 传播不完整 | 对齐 `MatExpr` 通道保持 compare 语义 |
| `Compare.empty` | `PENDING_CHANNEL` | `cvh::compare` 未实现 | 实现 `cvh::compare` 及空输入行为 |
| `Compare.regression_8999` | `PENDING_CHANNEL` | 同上 | 对齐 broadcasting/异常语义 |
| `Compare.regression_16F_do_not_crash` | `PENDING_CHANNEL` | 同上 | 对齐 16F 类型错误路径语义 |

### 2.4 scalar subtract / operations 相关（3）

| 用例 | 当前状态 | 原因 | 解锁条件 |
|---|---|---|---|
| `Subtract.scalarc1_matc3` | `PENDING_CHANNEL` | 缺少 `cv::subtract(Scalar, Mat, dst)` 等价重载 | 增加 scalar + multichannel subtract 接口语义 |
| `Subtract.scalarc4_matc4` | `PENDING_CHANNEL` | 同上 | 同上 |
| `Core_Array.expressions` | `PENDING_CHANNEL` | 依赖更完整的 channel 运算族（`mixChannels`/MatExpr/bitwise/compare） | 补齐上述运算接口并完成行为对齐 |

## 3. 已接线但受阻说明

- `test/core/mat_upstream_channel_port_test.cpp` 已新增 17 个上游对齐入口测试（对应上述 case）。
- `cvh_test_core` 已可运行；当前 channel 相关 pending 用例均通过 `GTEST_SKIP` 显式暴露。
- 后续每解锁一个 API 能力，应把对应 case 从 `PENDING_CHANNEL` 提升为 `PASS_NOW` 并更新 manifest。

## 4. 维护规则

1. 每次修复一个 pending 能力后，先把对应测试改为可运行，再更新 `channel_manifest.json` 状态为 `PASS_NOW`。
2. 本文档与 `channel_manifest.json` 需同步更新，避免“文档说已修，台账仍 pending”。
