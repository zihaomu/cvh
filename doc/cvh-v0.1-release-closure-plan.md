# cvh v0.1 代码与文档收口计划

更新时间：2026-08-03

状态：执行中；CL1–CL4 已完成，CL5 完整回归进行中

## 1. 目标

Phase 2-P0 是 v0.1 前最后一批算子支持。本轮只处理当前仓库的代码与文档
收口，不处理发布时才需要的打包、tag、跨平台发布矩阵、源码归档、checksum、
release note 或 release-candidate 性能报告。

本轮目标固定为：

- Phase 2-P0 的 17 个操作族、公开 API 和正确性设施全部保留；
- 删除历史兼容层和已经无引用的诊断工具；
- 把 Phase 2-P0 专项 benchmark 合并到长期维护的 canonical benchmark；
- 把已完成的阶段性文档交给 Git 历史归档；
- 把 Core/Imgproc 文档中的阶段术语改为 v0.1 稳定支持矩阵；
- 收口后保持 API coverage、数值合同和完整测试结果不变。

## 2. 明确不在本轮范围

以下工作不进入本计划：

- `CHANGELOG.md`、`CONTRIBUTING.md`、`SECURITY.md` 等发布文件；
- direct include、安装包和 FetchContent 用户接入示例；
- Linux、macOS、Windows hosted release matrix；
- v0.1 tag、release note、源码 archive 和 checksum；
- 基于最终 release commit 的新性能报告；
- 新增 Core/Imgproc 操作族；
- 为现有 scalar 实现增加新的 SIMD fast path；
- 改变已经接受的 Phase 2-P0 类型、通道、flag 或数值范围。

如果执行收口时发现上述范围内的问题，只记录为后续工作，不扩大本轮修改。

## 3. 当前稳定基线

- Core coverage：`61 / 97`；
- Imgproc coverage：`63 / 123`；
- 总 callable family coverage：`124 / 220`；
- declared-only：0；
- Core tests：213；
- Imgproc tests：192；
- Phase 2 upstream 差分组：5；
- Phase 2 专项 benchmark：26 条数据行。

收口不得降低以上覆盖、测试和差分基线。删除 benchmark 工具不等于删除算子
性能 case；需要长期保留的 case 必须先合入 canonical suite。

## 4. 必须保留

以下内容不是试验代码，本轮不得删除或弱化：

- 17 个操作族的 public header 和 header-only 实现；
- `Rect`、`Moments` 和相关公开枚举；
- 新增 Core/Imgproc 单元测试；
- 每个新增 public header 的独立编译 smoke；
- aggregate header compile 和多翻译单元 ODR smoke；
- install consumer 和 header-only contract；
- `test/opencv_contract` 中的 Phase 2 upstream 差分；
- `CVH_ENABLE_OPENCV_COMPARE` 和 OpenCV compare runner；
- `cvh_benchmark_core_mat_header`；
- `cvh_benchmark_imgproc_header`；
- 当前 OpenCV compare benchmark 基础设施；
- unsupported OpenCV enum 常量及其显式报错行为。

`RETR_TREE`、`TM_CCOEFF` 等常量属于 OpenCV 枚举兼容面。它们可以被当前实现
显式拒绝，但不能因为对应算法尚未支持就从枚举定义中删除。

## 5. 可以直接删除

以下内容已确认没有产品调用、必要 CI 或长期 benchmark gate 引用。

| 内容 | 同步修改 | 依据 |
| --- | --- | --- |
| `include/cvh/core/simd/simd.h` | public-header manifest 或文档中如有引用则同时移除 | 明确标记 deprecated，仓库内零引用。 |
| `cvh_benchmark_imgproc_coverage` | `CMakeLists.txt`、`benchmark/readme.md` | 文档明确标为非产品 gate，只有 CMake 和自身引用。 |
| `benchmark/imgproc_coverage_benchmark.cpp` | 同上 | 只实现被删除的 coverage 诊断 target。 |
| `cvh_benchmark_imgproc_filter` | `CMakeLists.txt`、`benchmark/readme.md` | 独立诊断 target，不进入当前 Mode A 回归框架。 |
| `benchmark/imgproc_filter_benchmark.cpp` | 同上 | 只实现被删除的 filter 诊断 target。 |
| `scripts/check_core_benchmark_regression.py` | 无 | 零引用、使用旧 CSV schema，已由 `benchmark/common/benchmark_report.py` 取代。 |
| `scripts/check_imgproc_filter_benchmark_regression.py` | 无 | 零引用，只服务旧 filter benchmark。 |
| `scripts/report_imgproc_filter_speedup.py` | 无 | 零引用，只服务旧 filter benchmark。 |
| `CVH_OPENCV_BENCH_DIR` | `CMakeLists.txt` 和 `benchmark/opencv_compare/run_compare.sh` | CMake 只声明、不消费，runner 传参没有效果。 |
| `doc/documentation-current-state-cleanup-plan.md` | `doc/README.md`、`scripts/check_docs.py` | 已完成的执行记录且包含旧测试数量，Git 历史已经可以归档。 |

预计直接删除约 2,500 行。执行前必须再次使用仓库引用搜索确认目标仍满足上述
条件；如果执行期间出现新消费者，则停止删除并重新分类。

## 6. 合并后删除

### 6.1 Phase 2-P0 专项 benchmark

`benchmark/phase2_p0_header_benchmark.cpp` 原未接入 benchmark CI 或
`benchmark/gate_policy.json`，不适合作为长期孤立 target 保留。

合并要求：

- random、transform 和 perspective transform case 合入
  `cvh_benchmark_core_mat_header`；
- connected components、contours、shape、histogram 和 template matching
  case 合入 `cvh_benchmark_imgproc_header`；
- 保留原有输入类型、通道、continuous/ROI、shape、dispatch、checksum、状态和
  scalar baseline 说明；
- quick profile 能覆盖每个操作族至少一个代表 case；
- 需要更大输入的 case 可以只进入 stable/full profile；
- 合并前后对相同输入的 checksum 必须一致；
- 合并后的 CSV 继续使用 canonical schema。

合并验证通过后已删除：

- `cvh_benchmark_phase2_p0_header` CMake target；
- `benchmark/phase2_p0_header_benchmark.cpp`；
- `benchmark/readme.md` 中的 Phase 2 专项 target 和运行命令；
- Phase 2 落地计划中只为该专项 target 保留的状态描述。

`cvh_benchmark_cvtcolor_bgr2gray_header` 和
`cvh_benchmark_resize_bilinear_header` 当前仍被 quick benchmark CI 和 gate
policy 使用，不属于本轮直接删除范围。

执行结果（2026-08-03）：

- 7 条 Core case 已并入 `cvh_benchmark_core_mat_header`；
- 19 条 Imgproc case 已并入 `cvh_benchmark_imgproc_header`；
- quick profile 下 26/26 条的输入类型、通道、layout、shape、elements、
  implementation、dispatch、checksum、status 和 note 与合并前 CSV 一致；
- full profile 以 `warmup=0/iters=1/repeats=1` 完成 smoke，7 条 Core 和
  19 条 Imgproc 搬迁 case 均为 `OK`；
- 专项 CMake target、source、README target 和运行命令已删除，非构建
  目录中无残余引用。

### 6.2 阶段性实施文档

以下文档描述已经完成的实施过程，不应长期作为 v0.1 当前事实 owner：

- `doc/opencv-core-imgproc-phase1-implementation-plan.md`；
- `doc/opencv-core-imgproc-phase2-p0-implementation-plan.md`。

两者当前尚未进入 Git 历史，不能直接丢弃。处理顺序必须是：

1. 先让算子实现、测试和对应验收记录进入一个可追溯提交；
2. 确认 API coverage、三阶段支持清单和模块 README 已接管当前事实；
3. 在后续独立 cleanup 变更中删除两份实施计划；
4. 同步删除 `doc/README.md` 中的索引项和其他文档链接；
5. 运行文档一致性检查。

执行结果（2026-08-03）：

- 两份当前实施与验收记录已由提交 `33045f1` 纳入 Git 历史；
- API coverage、三阶段支持清单和模块 README 已接管当前支持事实；
- 两份阶段实施文档和 `doc/README.md` 索引已删除；
- 三阶段支持清单已改为直接引用 coverage、模块 README、单测和
  OpenCV contract，不再依赖阶段实施计划解释当前范围；
- `check_docs.sh` 通过，当前维护文档由 33 份收口为 31 份。

## 7. 文档术语收口

### 7.1 模块 README

已将以下标题：

- `include/cvh/core/readme.md` 中的 `Phase 2 P0 Support Matrix`；
- `include/cvh/imgproc/readme.md` 中的 `Phase 2 P0 Support Matrix`；

统一改为 `v0.1 Support Matrix`。表格内容继续描述已接受的具体类型、通道、
layout、flag 和 unsupported 边界，不缩减技术细节。

### 7.2 当前事实同步

必须同步检查：

- `test/failing-tests.md` 的 Core/Imgproc 测试数量；
- `doc/opencv-core-imgproc-api-coverage.md` 的 61/63/124 统计；
- `doc/opencv-core-imgproc-three-phase-support-plan.md` 的 P2-P0 完成状态；
- `doc/README.md` 的当前文档索引和事实 owner；
- `benchmark/readme.md` 的 canonical/diagnostic target 列表；
- `scripts/check_docs.py` 对已删除 cleanup record 的特殊豁免。

三阶段支持清单可以继续保留 post-v0.1 backlog，但不得再依赖已删除实施计划解释
当前支持范围。

执行结果（2026-08-03）：

- Core/Imgproc 模块 README 标题已统一为 `v0.1 Support Matrix`；
- `test/failing-tests.md` 已同步为 Core 213、Imgproc 192，与 arm64/x86_64
  gate expectation 一致；
- API coverage 按条目重新计数为 Core 61 available / 36 missing、Imgproc 63
  available / 60 missing、总 available 124，declared-only 0；
- coverage 的 cvh baseline 已更新为包含算子与验收记录的 `33045f1`；
- `doc/README.md`、三阶段支持清单和 `benchmark/readme.md` 已只保留
  当前事实 owner 和 canonical benchmark 入口；
- `check_docs.sh` 和 `git diff --check` 通过。

## 8. 执行步骤和实时状态

| Step | 内容 | 状态 | 完成条件 |
| --- | --- | --- | --- |
| CL0 | 审计引用、分类删除/合并/保留内容 | 已完成 | 已形成第 4 至第 7 节清单；尚未删除文件。 |
| CL1 | 删除 deprecated shim、旧诊断 benchmark、无引用脚本和无效 CMake 变量 | 已完成 | 目标文件、target、脚本、CMake 变量及残余引用已删除；文档检查、default/test/benchmark 配置和 `git diff --check` 通过。 |
| CL2 | 把 26 条 Phase 2 benchmark case 合入 canonical Core/Imgproc suite | 已完成 | Core 7 条、Imgproc 19 条已合并；quick 26/26 checksum 及关键字段一致，full smoke 26/26 `OK`，专项 target/source 已删除。 |
| CL3 | 归档并删除已完成的阶段性文档 | 已完成 | 两份验收记录已进入 `33045f1` 的 Git 历史后删除；coverage/README/三阶段清单接管当前事实，文档检查通过。 |
| CL4 | 修改 v0.1 support matrix 名称并同步测试、coverage、索引和 benchmark 文档 | 已完成 | README 标题已收口；测试台账/gate expectation 为 213/192，coverage 按条目复核为 61/63/124 且 declared-only 为 0，文档检查通过。 |
| CL5 | 完整代码与文档回归 | 进行中 | 开始执行第 9 节的静态、header contract、完整 CI、OpenCV 差分和 benchmark 门禁。 |

每完成一个 Step，必须更新本表状态和实际验证结果。不得在 benchmark 尚未合并时
提前删除专项 case，也不得在实施记录尚未进入 Git 历史时删除未跟踪文档。

## 9. 验证门禁

### 9.1 静态和文档检查

```bash
./scripts/check_docs.sh
./scripts/check_public_headers.sh
./scripts/check_header_only_contract.sh
git diff --check
```

额外确认：

- 已删除文件没有残余路径、target 或命令引用；
- 当前文档不存在旧的 209/187 测试数量；
- coverage 仍为 Core 61、Imgproc 63、总计 124；
- public umbrella header 仍包含全部 17 个操作族；
- 安装树中没有 deprecated `core/simd/simd.h`。

### 9.2 编译与正确性

```bash
./scripts/ci_headers_all.sh
```

要求：

- header compile、aggregate compile、ODR 和 install consumer 通过；
- CTest 全部通过；
- Core 213/213；
- Imgproc 192/192；
- 无新增 skip。

### 9.3 OpenCV 差分

在启用 `CVH_ENABLE_OPENCV_COMPARE` 的现有 build 中运行完整 contract smoke，至少
确认 Phase 2 差分组 5/5 和现有差分组全部通过。

本轮删除不得修改 upstream tolerance、label/contour 顺序、tie rule 或 unsupported
参数行为。

### 9.4 Benchmark

- `CVH_BUILD_BENCHMARKS=ON` 配置成功；
- canonical Core/Imgproc target 均可编译；
- quick profile 产生有效 CSV；
- 合入的 Phase 2 case 每个操作族至少一条 `OK`；
- 同一输入合并前后的 checksum 一致；
- 已删除 target 不再出现在 CMake、README、CI 或 gate policy 中。

## 10. 完成条件

- [ ] Phase 2-P0 17 个操作族及全部正确性设施保持不变。
- [x] `core/simd/simd.h` 已删除且无引用。
- [x] 两个旧 Imgproc 诊断 benchmark target/source 已删除。
- [x] 三个无引用 benchmark 脚本已删除。
- [x] `CVH_OPENCV_BENCH_DIR` 声明和无效传参已删除。
- [x] documentation cleanup 执行记录及检查特例已删除。
- [x] Phase 2 benchmark case 已进入 canonical Core/Imgproc suite。
- [x] Phase 2 专项 benchmark target/source 已删除。
- [x] Phase 1/Phase 2 实施记录已进入 Git 历史并从当前文档树移除。
- [x] Core/Imgproc README 使用 `v0.1 Support Matrix`。
- [x] 测试台账更新为 Core 213、Imgproc 192。
- [x] coverage 保持 61/63/124，declared-only 保持 0。
- [ ] 文档、header contract、完整 CI、upstream 差分和 benchmark quick 通过。
- [ ] `git diff --check` 通过，删除清单之外没有意外改动。

全部条件满足后，本轮代码与文档收口完成。后续发布打包和平台验证由独立计划
处理，不回填到本计划。

## 11. 建议变更边界

为了便于审查，建议拆成以下变更：

1. Phase 2-P0 算子实现、测试、差分和阶段验收记录；
2. deprecated/unused 代码和诊断工具删除；
3. Phase 2 benchmark 合入 canonical suites；
4. 阶段性文档归档删除和 v0.1 术语同步；
5. 最终回归结果和状态表更新。

删除、benchmark 搬迁和事实文档更新应保持可独立审查，不混入新的算子功能或
优化实现。
