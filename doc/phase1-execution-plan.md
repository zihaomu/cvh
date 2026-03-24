# Phase 1 执行计划（Core MVP）

- 时间范围：2026-03-10 ~ 2026-03-24（10 个工作日）
- 对应里程碑：`doc/header-only-opencv-plan.md` 的 Phase 1
- 目标：冻结 `Mat` 合同并推动实现与测试对齐，形成可稳定迭代的 `core` MVP 基线

## 1. Phase 1 目标与退出条件

### 目标

- 冻结 `Mat` 的 v1 行为合同，并作为后续实现/测试唯一语义基线。
- 将 `Mat` 关键 API（`create/clone/copyTo/convertTo/reshape`）对齐合同语义。
- 形成可重复执行的 `core` 合同回归测试入口。

### 退出条件（全部满足）

- `doc/mat-contract-v1.md` 冻结并纳入主规划索引。
- `Mat` 关键 API 的实现行为与合同一致（至少覆盖连续多通道语义）。
- `test/core` 覆盖合同清单中的成功路径与失败路径。
- `./scripts/ci_smoke.sh` 与 `./scripts/ci_core_basic.sh` 稳定通过。

## 2. 工作分解（WBS）

### P1-01（D1）：Phase 1 任务拆解与门禁固化

- 任务：
  - 建立 Phase 1 执行条目、顺序、门禁与验收命令。
  - 明确后续任务编号（P1-02 ~ P1-05）的输入输出关系。
- 交付物：
  - 本文档 `doc/phase1-execution-plan.md`。
- 验收：
  - 能基于本文档直接推进 P1-02 ~ P1-05。

### P1-02（D1-D2）：Mat 合同冻结

- 任务：
  - 固化 `Mat` v1 语义边界（type/channel/shape/ownership/continuous/error）。
  - 明确 v1 非目标，防止范围漂移。
- 交付物：
  - `doc/mat-contract-v1.md`。
- 验收：
  - 合同覆盖关键 API 行为与失败语义，并在规划文档中可追踪。

### P1-03（D3-D5）：实现差异清理（合同对齐）

- 任务：
  - 对齐 `include/cvh/core/mat.h`、`include/cvh/core/mat.inl.h`、`src/core/mat.cpp` 与合同差异。
  - 清理与合同冲突的隐式行为（优先错误路径与边界路径）。
- 交付物：
  - `Mat` 关键 API 对齐补丁（以合同为准）。
- 验收：
  - 合同条目对应实现均可被测试验证，不存在 silent wrong result。

### P1-04（D6-D7）：Header-only 迁移首批闭环

- 任务：
  - 将合同范围内的 `Mat` 稳定实现优先迁至 `include/cvh/core/*.inl.h` / `detail/*.h`。
  - 缩减 `src/core/mat.cpp` 对默认能力的承载范围。
- 交付物：
  - 首批迁移闭环记录（迁移前后能力和限制说明）。
- 验收：
  - 默认使用路径对 `src/core` 依赖进一步下降，行为与测试不回退。

### P1-05（D8-D10）：合同测试闭环

- 任务：
  - 按合同第 7 节补齐测试：`clone/copyTo/reshape/convertTo` 成功与失败路径。
  - 增加边界输入与错误路径的明确行为测试。
- 交付物：
  - `test/core` 新增或更新的合同测试用例。
- 验收：
  - 合同测试可稳定复现，CI 可运行且结果一致。

### P1-06（D11）：OpenCV 高频元信息接口对齐（连续内存语义）

- 任务：
  - 增加 `Mat` 的高频元信息接口：`depth/channels/elemSize/elemSize1/isContinuous/step/step1`。
  - 增加连续内存语义下的接口对齐测试与生命周期安全回归。
- 交付物：
  - `Mat` 元信息接口实现补丁。
  - `test/core/mat_opencv_compat_test.cpp`（并接入 `cvh_test_core`）。
- 验收：
  - `./build-*/cvh_test_core '--gtest_filter=MatOpenCVCompat_TEST.*'` 通过。
  - `cvh_test_core` 全量通过，不回归现有 `Mat` 合同测试。

### P1-07（D12）：Type 宏对齐与连续多通道首批支持

- 任务：
  - 对齐必要的 OpenCV 风格 type 宏（`CV_MAKETYPE/CV_MAT_DEPTH/CV_MAT_CN` 与 `CV_*C(n)`）。
  - 让 `Mat` 在连续内存语义下支持多通道：`create/setTo/copyTo/convertTo`。
  - 增加首批多通道合同测试，验证 RED->GREEN 闭环。
- 交付物：
  - `include/cvh/core/define.h` 宏补丁与 `src/core/mat*.cpp` 多通道实现补丁。
  - `test/core/mat_channel_contract_test.cpp`（并接入 `cvh_test_core`）。
- 验收：
  - `./build-*/cvh_test_core '--gtest_filter=MatChannelContract_TEST.*'` 通过。
  - `cvh_test_core` 全量与 smoke/test 目标通过。

### P1-08（D13）：2D submat 与非连续步长首批闭环

- 任务：
  - 增加 `Mat` 的 2D view API：`rowRange/colRange/operator()(Range, Range)`。
  - 在非连续步长场景打通 `clone/copyTo/convertTo/setTo` 的正确性路径。
  - 为非连续 `Mat` 的 `reshape` 增加明确失败语义，避免 silent wrong result。
- 交付物：
  - `include/cvh/core/mat.h` 与 `src/core/mat.cpp`/`src/core/mat_convert.cpp` 的 stride-aware 补丁。
  - `test/core/mat_submat_test.cpp`（并接入 `cvh_test_core`）。
- 验收：
  - `./build-*/cvh_test_core '--gtest_filter=MatSubmat_TEST.*'` 通过。
  - `cvh_test_core` 全量与 smoke/test 目标通过。

## 3. 执行状态

| 任务 | 状态 | 更新时间 | 备注 |
|---|---|---|---|
| P1-01 | 已完成 | 2026-03-10 | 已建立 Phase 1 执行文档，固化任务与门禁 |
| P1-02 | 已完成 | 2026-03-10 | 已新增 `doc/mat-contract-v1.md`，并完成索引与 core 文档联动 |
| P1-03 | 已完成 | 2026-03-10 | 已对齐 `create/reshape/copyTo/convertTo/setTo` 前置条件与错误语义，并通过 `ci_smoke + ci_core_basic` |
| P1-04 | 已完成 | 2026-03-10 | 已完成首批 header-only 迁移闭环（`MatSize` 与 `total(MatShape,...)` 迁移至 `mat.inl.h`，include-only smoke 直接调用通过） |
| P1-05 | 已完成 | 2026-03-10 | 已新增 `cvh_test_core` 聚合测试二进制（支持 `gtest_filter`）并完成 Mat 合同测试闭环，且修复 `setTo` 在 16bit odd shape 的尾元素漏写问题 |
| P1-06 | 已完成 | 2026-03-10 | 已补齐 `depth/channels/elemSize/elemSize1/isContinuous/step/step1` 连续语义，并新增 `MatOpenCVCompat_TEST` 对齐回归 |
| P1-07 | 已完成 | 2026-03-10 | 已对齐必要 OpenCV type 宏并落地连续多通道首批支持（`create/setTo/copyTo/convertTo`），新增 `MatChannelContract_TEST` 回归 |
| P1-08 | 已完成 | 2026-03-10 | 已落地 2D submat（`rowRange/colRange`）与非连续步长首批语义，新增 `MatSubmat_TEST` 并通过全量回归 |

## 4. 风险与应对

- 风险：合同冻结后仍出现“实现先变更、文档后补”的倒挂。  
  应对：P1 期间 `Mat` 行为变更必须先改合同，再改代码和测试。

- 风险：`src` 与 `include` 双份实现短期分叉。  
  应对：P1-04 每次迁移都写明删除条件并同步台账。

- 风险：测试覆盖偏正常路径，边界/错误路径缺失。  
  应对：P1-05 以合同第 7 节为强制清单逐项验收。

## 5. 建议验收命令基线

```bash
./scripts/ci_smoke.sh
CVH_WARNING_BUDGET=0 ./scripts/ci_core_basic.sh
```
