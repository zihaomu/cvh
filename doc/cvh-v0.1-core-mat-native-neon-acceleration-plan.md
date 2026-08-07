# cvh v0.1 Core/Mat Native-NEON 加速计划

更新时间：2026-08-06
状态：N0/N1 已完成（clean 报告待 N6）；N2 实现/正确性/floor 完成（7 个短 ROI stable case 待隔离宿主关闭）；N3 已完成并回退；N4 已完成；N5 已完成；N6 进行中

## 1. 目的与阶段定位

本文定义 v0.1 在 U8C3 Resize 定点 NEON 收口之后的 Core/Mat 性能专项，目标是
从当前 OpenCV Universal Intrinsics（UI）路径中筛选少数值得维护原生 AArch64
NEON kernel 的高价值算子。

本阶段的核心判断不是“`opencv_ui` 是否等于 NEON”。在 Apple/AArch64 上，
当前 vendored OpenCV UI 的 `cv::vx_*` 通常已经编译为 128-bit NEON 指令。
因此，只有满足以下至少一项时才允许新增 native-NEON 路径：

1. 可以改变算法或数据流，例如融合多个 pass、消除临时 `Mat`；
2. 可以采用 UI 通用实现难以表达的多累加器、分块合并或特殊 layout；
3. 可以共享一个 kernel 覆盖多个高频公开算子；
4. 三轮 stable benchmark 证明相对当前 UI 有足够收益。

本阶段不以改变 telemetry 标签为目标，也不把 UI 当成 scalar。所有未达到保留
门槛的原生 NEON candidate 必须删除，继续保留现有 UI/scalar 路径。

执行本计划前，应先让
[Resize U8C3 定点 NEON 阶段](cvh-v0.1-resize-u8c3-fixed-point-neon-acceleration-plan.md)
完成提交和 clean performance report，或在独立分支/稳定 build identity 上冻结
本阶段基线，避免两个专项共用 dirty revision 导致性能归因不清。

## 2. 当前证据

主基线使用：

- [2026-08-06 v0.1 NEON hot-kernel product-auto report](../benchmark/opencv_compare/results/2026-08-06-v0.1-neon-hot-opencv-upstream-performance.en.md)；
- [raw CSV](../benchmark/opencv_compare/results/2026-08-06-v0.1-neon-hot-opencv-upstream-performance.csv)；
- [metadata](../benchmark/opencv_compare/results/2026-08-06-v0.1-neon-hot-opencv-upstream-performance.meta.json)。

该报告为 Apple M5、Release、单线程、product `Auto` 的 clean snapshot。后续
Resize 修改没有改变 Core/Mat 产品源码，因此可以作为本计划 N0 排序依据；N1
仍需在新的 clean revision 上重新跑三轮 focused baseline。

相对性能定义保持不变：

```text
relative performance = OpenCV latency / CVH latency
1.0 = 持平；小于 1.0 = CVH 更慢
```

### 2.1 候选差距

| family / case | 当前 relative | 等价差距 | 最大代表性绝对损失 | 判断 |
| --- | ---: | ---: | ---: | --- |
| `reduce` F32 SUM/AVG | `0.19–0.27` | OpenCV 约快 `3.8–5.1x` | 约 `0.050 ms` | P0，共享 reduction kernel |
| `norm` F32C1 | 几何平均 `0.2955` | OpenCV 约快 `3.38x` | 约 `0.070 ms` | P0，与 normalize 共享 |
| `normalize` F32C1 | 几何平均 `0.4686` | OpenCV 约快 `2.13x` | 约 `0.071 ms` | P0，主要受 reduction 支配 |
| `meanStdDev` F32C3 | `0.3169` | OpenCV 约快 `3.16x` | 约 `0.275 ms` | P0，共享稳定统计设施 |
| `rotate` 90° U8C3 | `0.4266` | OpenCV 约快 `2.34x` | 约 `0.134 ms` | P1，融合 transpose+flip |
| `inRange` U8C3 | `0.6187` | OpenCV 约快 `1.62x` | 约 `0.067 ms` | P1，packed compare/mask |
| `convertScaleAbs` F32C3 | `0.6011` | OpenCV 约快 `1.66x` | 约 `0.040 ms` | 仅诊断 candidate |
| `add/subtract/multiply` | 几何平均约 `0.78` | OpenCV 约快 `1.28x` | 1080p 约 `0.10–0.12 ms` | 高频但 UI 已是 NEON，条件保留 |
| `divide` F32 | 几何平均 `0.7610` | OpenCV 约快 `1.31x` | 1080p 约 `0.161 ms` | 不做 native rewrite |
| `exp/log/pow` F32 | `0.47–0.58` | OpenCV 约快 `1.7–2.1x` | `pow` 约 `0.205 ms` | 无硬件指令，本阶段排除 |

### 2.2 已经不值得替换的 UI 路径

- `copyTo`、`flip`、`countNonZero`、`hasNonZero` 已接近 OpenCV；
- `extractChannel`、`insertChannel`、`mixChannels`、`sum`、`mean` 当前快于
  OpenCV；
- `transpose` 表现随类型和尺寸变化，F32 基本持平，部分 U8 case 已快于
  OpenCV，不适合整体替换；
- `findNonZero` 的 all-zero/sparse case 约快 `7x`，只有 dense case 慢；dense
  瓶颈主要是坐标 emission、`vector` 扩容和写出，不是 UI scan 本身；
- bitwise family 已经是连续 NEON load/op/store，虽然个别 relative 较低，但
  绝对损失小且受内存带宽限制。

## 3. 范围与优先级

### 3.1 P0：共享 reduction/statistics NEON 后端

首个产品范围严格限定为：

- AArch64/ARM64；
- `CV_32F`；
- 无 mask 的 continuous 和逐 row contiguous 输入；
- `reduce`：C1，axis 0/1，`REDUCE_SUM`、`REDUCE_AVG`、`REDUCE_SUM2`；
- `norm`：C1，single/diff，`NORM_INF`、`NORM_L1`、`NORM_L2`；
- `normalize`：C1，`NORM_INF/L1/L2`，F32 输出；
- `meanStdDev`：C1/C3，无 mask。

以下内容在首版继续使用 UI/scalar：

- mask、非连续 inner span、ND 特殊 layout；
- U8/F64 和其他深度；
- `REDUCE_MIN/MAX`：当前 axis-1 case 已明显快于 OpenCV；
- `reduceArgMin/Max`、`minMaxIdx/Loc`；
- 超短行和无法形成完整 NEON block 的输入。

### 3.2 P1：融合 rotate90

- `ROTATE_90_CLOCKWISE`、`ROTATE_90_COUNTERCLOCKWISE`；
- `CV_8UC1/C3/C4`；
- continuous 与 ROI source；
- 使用 tiled native-NEON kernel，一次从 source 读取并直接写入最终方向；
- 不创建完整 transposed 临时 `Mat`，不再执行第二次 full-image `flip`；
- `ROTATE_180` 保留现有 `flip` 实现；
- 小 tile、边缘和不支持的 element size 使用 scalar/UI fallback。

当前报告虽将 90° rotate 记录为 `opencv_ui`，但实际 C3 路径先执行 scalar
3-byte transpose，再由 UI flip 覆盖最终 tag。本阶段必须同时修正 stage-level
telemetry，避免把最后一个子阶段当成整个算子的真实实现。

### 3.3 P1：`inRange` U8 packed

- scalar lower/upper bounds；
- `CV_8UC1/C3/C4`；
- continuous、ROI、unaligned row start 和 scalar tail；
- C3/C4 使用 deinterleaved load，分别完成上下界比较，再 AND 成单通道 U8
  mask；
- Mat lower/upper、其他深度和短行继续走 UI/scalar。

### 3.4 P2：高频 elementwise 条件 candidate

只建立共享原型，不预设一定进入产品：

- `add`、`subtract`、`multiply`；
- mat-mat、U8/F32、C1/C3；
- continuous flatten fast path 与逐 row ROI；
- 4-vector unroll、独立 load/store stream、显式 prefetch 只根据 profile 决定；
- mask、scalar operand、alias 和特殊类型继续走现有路径。

由于 UI 已经映射到 `vaddq/vsubq/vmulq` 等 NEON 指令，native candidate 只有
相对 UI 达到统一 retention gate 才能保留。`divide` 不进入该原型：严格除法仍
会使用与 UI 等价的 `vdivq_f32`，reciprocal estimate 会改变数值合同。

### 3.5 诊断后才允许纳入

- `convertScaleAbs` F32C3 -> U8C3；
- dense `findNonZero` 的 reserve、two-pass count 或 block compaction；
- `transpose` 的单个已证实慢尺寸/type；
- U8 reduction 与 masked statistics。

这些项目必须先有独立成本分解，不能因为当前 relative 较低就直接增加 native
产品路径。

## 4. 明确不包含

- `exp`、`log`、非整数 `pow` 的 native-NEON 数学库；
- 用 reciprocal estimate 替换精确 F32 division；
- 重写已经接近或快于 OpenCV 的 copy/channel/flip/mean/sum 路径；
- 修改公开 API、类型、枚举或异常合同；
- OpenCV/KleidiCV/Accelerate 运行时依赖；
- 多线程掩盖单线程 kernel 差距；
- 为 benchmark 尺寸、seed 或对齐方式硬编码分支；
- 放宽 upstream tolerance、删除 special-value case 或关闭 checksum；
- 为每个算子复制一套不能复用的 dispatch/control 代码。

## 5. Reduction 数值与 kernel 设计

### 5.1 数值合同

当前 reduction UI 为了稳定性大量使用 F32 -> F64、`long double` 和 Chan/Welford
式合并。native candidate 不得简单改为整幅图 F32 累加。

必须保持：

- `NaN`、`Inf`、`-0`、空 mask 和零范数行为；
- `NORM_INF/L1/L2` 的 single/diff 语义；
- `REDUCE_AVG` 的除法时机和输出 dtype；
- `meanStdDev` 在大 offset + 小 variance 输入上的稳定性；
- 现有 scalar/UI/internal tests 与 OpenCV differential tolerance，不得放宽。

### 5.2 两个允许比较的累加 candidate

Candidate A：F32 block accumulation + periodic F64 merge。

```text
4–8 个 float32x4 accumulator
  -> 每固定 block 做 pairwise horizontal reduction
  -> 转为 double/long-double block result
  -> 使用稳定公式合并 block
```

Candidate B：direct F64 NEON accumulation。

```text
连续加载 float32x4
  -> low/high 转 float64x2
  -> 4-way unroll 的 F64 accumulator
  -> block 末尾一次归约/merge
```

选择规则：

- Candidate A 必须先证明所有 adversarial numeric cases 满足现有合同；
- 若 A 的误差无法稳定关闭，保留 B，不得用放宽 tolerance 换性能；
- 若 B 相对 UI 未达到 retention gate，则 reduction native candidate 回退；
- `meanStdDev` 可以使用两 pass 或 block Chan merge，但不得使用不稳定的
  `E[x^2] - E[x]^2` 直接公式处理完整输入。

### 5.3 共享设施

建议新增内部 header：

```text
include/cvh/core/detail/reduction_neon.hpp
```

共享组件至少包括：

- continuous/row-span eligibility；
- F32 single/diff load adapter；
- INF/L1/L2/SUM/SUM2 block accumulator；
- block result 的 F64/stable merge；
- scalar tail 和 special-value observation；
- 真实 dispatch/route 写入。

不得让 `norm`、`normalize`、`reduce` 和 `meanStdDev` 各自复制一套 NEON loop。

## 6. Rotate 与 packed kernel 设计

### 6.1 Fused rotate90

建议新增：

```text
include/cvh/core/detail/rotate_neon.hpp
```

主体策略：

1. 按 8x8 或经 benchmark 选定的 tile 遍历 source；
2. C1 使用 NEON zip/trn 完成 tile transpose；
3. C3/C4 使用 deinterleaved load 或 byte-table shuffle，按目标方向直接存储；
4. clockwise/counterclockwise 通过目标基址和 store 顺序复用同一 primitive；
5. 边界 tile 显式 scalar，不做越界 load/store；
6. alias 时继续使用受控 source clone，但禁止再分配完整 transpose 中间图。

### 6.2 `inRange`

建议新增或扩展共享 packed compare header：

```text
include/cvh/core/detail/inrange_neon.hpp
```

每个 block：

- load C1 或 deinterleave C3/C4；
- 对每个通道计算 `lower <= value && value <= upper`；
- 跨通道 AND；
- 写出连续 U8 mask；
- scalar tail 使用与 public path 相同的 inclusive boundary。

## 7. Dispatch 与 telemetry

通用选择顺序保持：

```text
eligible native NEON
  -> existing OpenCV UI
  -> scalar
```

只有真正执行至少一个 NEON 主体 block 才能记录 `DispatchTag::NEON`。短行、全
tail 或 unsupported parameter 必须记录真实 fallback。

建议 route：

```text
reduce_f32:layout=continuous;load=neon;row_group=4;accumulate=f32_block;merge=f64;tail=scalar
norm_f32:mode=l2_diff;load=neon;accumulate=f32_block;merge=f64;tail=scalar
mean_stddev_f32c3:pass=two;load=neon_deinterleave;merge=chan_f64;tail=scalar
rotate90_u8c3:tile=8x8;load=neon_deinterleave;shuffle=transpose;store=direct;tail=scalar
inrange_u8c3:load=neon_deinterleave;compare=inclusive;reduce=channel_and;store=mask;tail=scalar
arithm_f32c3:op=add;layout=flat;unroll=4;load=neon;store=neon;tail=scalar
```

`OpenCVUIOnly` 和 `ScalarOnly` 绝不进入 direct NEON；`NeonOnly` 只强制已满足
完整 eligibility 的 kernel，不得绕过类型、layout、数值或 workload 检查。

## 8. Correctness 与安全门禁

### 8.1 Reduction/statistics

- continuous、ROI、non-contiguous step、unaligned row start、短行和 tail；
- C1/C3、多行、奇数宽高和最小合法尺寸；
- zero、constant、gradient、random、large offset + small variance；
- `NaN`、`+Inf/-Inf`、`-0`、subnormal、极大/极小有限 F32；
- single/diff norm、zero norm、in-place normalize；
- scalar/UI/NEON 三方对照和多个 seed 的 upstream differential；
- 多线程配置不得改变结果，首版 kernel 本身仍以单线程基准衡量。

### 8.2 Rotate/inRange/elementwise

- C1/C3/C4、odd width/height、tile `N-1/N/N+1`；
- continuous、ROI、non-contiguous、unaligned、alias/in-place；
- clockwise/counterclockwise 和现有 rotate180 fallback；
- inRange inclusive edge：`0/1/127/128/254/255`；
- U8 saturating add/sub/multiply，F32 NaN/Inf/signed-zero；
- ASan/UBSan guard，证明 tile 与 tail 无 overread/overwrite。

### 8.3 全局合同

- `Auto`、`NeonOnly`、`OpenCVUIOnly`、`ScalarOnly` 全覆盖；
- optimization-on/off；
- header independent compile、ODR、install consumer；
- Apple ARM64 full Core tests 与 OpenCV contract full；
- x86_64 compile；Linux x86_64 runtime 如本机不可用，明确保留外部门禁；
- public API coverage、异常和 unsupported 类型不变化。

## 9. Benchmark 设计与性能门禁

### 9.1 Canonical focused matrix

扩展现有 `benchmark/opencv_compare` runner，增加筛选集合：

```text
CORE_MAT_NEON
```

该集合复用 canonical binary，不创建阶段性 benchmark target。至少包含：

- `reduce` axis 0/1 SUM/AVG/SUM2，480x640、720x1280、1080x1920；
- norm single/diff INF/L1/L2；
- normalize INF/L1/L2；
- meanStdDev C1/C3；
- rotate90 CW/CCW U8C1/C3/C4；
- inRange U8C1/C3/C4；
- add/subtract/multiply U8/F32 C1/C3；
- continuous 与代表性 odd ROI；
- auto/ui/scalar 三种 benchmark mode，保持 checksum 开启。

每个 product case 都必须报告：

```text
algorithm_path -> dispatch_path -> isa_observed -> kernel_route
```

### 9.2 统一 retention gate

任何新增 direct-NEON 路径必须同时满足：

| 项目 | 门槛 |
| --- | ---: |
| 相对当前 UI 三轮逐 case 中位数提升 | `>= 1.25x` |
| candidate 稳定性 | 三轮 CVH latency 极差 `<= 5%` |
| 同 family 非目标回退 | `<= 5%` |
| full Core/full compare 非目标 aggregate 回退 | `<= 1%` |

没有达到 `1.25x` 的 elementwise native candidate 应删除，而不是因为调用频率高
就降低保留标准。

### 9.3 Family performance floor

| family | 完成 floor |
| --- | ---: |
| `reduce` SUM/AVG/SUM2 目标 case | `relative >= 0.50` |
| `norm` / `normalize` | `relative >= 0.60` |
| `meanStdDev` | `relative >= 0.60` |
| fused rotate90 | `relative >= 0.70` |
| `inRange` | `relative >= 0.80` |
| retained add/subtract/multiply | `relative >= 0.90` |

如果某 family 达到 retention、但没有达到完成 floor，只能标记“保留的局部改进”，
不能把该 family 的性能收口标记完成。

## 10. N0–N6 实施批次

| 批次 | 内容 | 状态 | 主要产物 |
| --- | --- | --- | --- |
| N0 | UI/native 审计与候选排序 | 完成 | 本文、clean report 差距表、明确排除项 |
| N1 | baseline、focused matrix 与 telemetry | 功能完成；clean 待 N6 | 三轮 clean UI/Auto baseline、`CORE_MAT_NEON` |
| N2 | shared reduction/statistics NEON | 实现/正确性/floor 完成；7 个短 ROI stable case 待隔离宿主 | reduce/norm/normalize/meanStdDev kernel 与 tests |
| N3 | fused rotate90 NEON | 完成（candidate 回退） | direct tile kernel、stage telemetry、tests |
| N4 | inRange packed NEON | 完成 | C1/C3/C4 compare-mask kernel、tests |
| N5 | elementwise/convertScaleAbs candidate | 完成 | 全部 native candidate 回退；保留诊断 matrix |
| N6 | 全量正确性、跨平台、stable/full 收口 | 进行中 | clean report、dispatch 文档、最终状态 |

### N1：基线与可观测性

1. 从 clean revision 冻结 Auto/UI/Scalar 三轮 stable baseline；
2. 将上述目标 case 接入 canonical `CORE_MAT_NEON` filter；
3. 修正 rotate 的 stage telemetry，区分 transpose、flip 和 fused route；
4. 建立 current UI -> candidate NEON 的逐 case 对照脚本/表；
5. 确认 checksum、threads=1、OpenCV revision/config 和 build fingerprint 一致。

### N2：Reduction/statistics

1. 先实现独立可单测 block accumulator；
2. 对 Candidate A/B 做 correctness 和 micro-cost 比较；
3. 接入 `reduce` SUM/AVG/SUM2；
4. 复用到 norm/normalize；
5. 最后接入 meanStdDev，单独关闭 stable numeric gate；
6. 任一算子未达到 retention 时，只回退对应 selector，不删除可复用且已证明
   有收益的底层 primitive。

### N3：Fused rotate90

1. 冻结现有 transpose+flip 输出与 alias 合同；
2. 实现 C1 tile，再扩展 C3/C4；
3. 同一 kernel 复用 CW/CCW；
4. ROI/odd/tail/ASan 关闭后接入 selector；
5. public-call benchmark 必须包含 dst allocation，且不得保留完整临时 transpose。

### N4：inRange

1. scalar bounds C1；
2. deinterleaved C3/C4；
3. inclusive boundary、ROI、tail 和 forced-mode tests；
4. stable retention 未达到 `1.25x` 时完整回退 native candidate。

### N5：条件 candidate

1. 共享 add/subtract/multiply streaming primitive；
2. 先测 F32C1/C3 与 U8C1/C3 mat-mat；
3. 分离 allocation、kernel 和内存带宽 floor，确认差距是否真的来自 UI loop；
4. `convertScaleAbs` 只做诊断原型；
5. 不达 gate 的代码和一次性测试必须删除，失败数字记录在本文。

### N6：全量收口

1. targeted -> module full -> OpenCV contract full；
2. sanitizer、optimization-off、header/ODR/install；
3. x86_64 cross compile，尽可能取得 Linux runtime；
4. 三轮 stable + full product-auto；
5. 排除目标 case 后核对 non-target aggregate；
6. 从 clean revision 归档 date-named Markdown/CSV/metadata；
7. 更新 results index、`cpu-optimization.md`、本文和 successor/backlog。

## 11. Canonical 命令骨架

### 11.1 Focused correctness

```bash
build-core-mat-neon-tests/cvh_test_core \
  --gtest_filter='*Reduction*:*Norm*:*Statistics*:*Rotate*:*InRange*:*Dispatch*'

build-core-mat-neon-tests/cvh_test_opencv_contract_smoke \
  --gtest_filter='OpenCVContractSmoke_TEST.core*'
```

开始 N1 时必须先用 `--gtest_list_tests` 核对实际 suite/filter，禁止把空 filter
当成通过。

### 11.2 Focused performance

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
CVH_COMPARE_BUILD_DIR=build-core-mat-neon-compare \
CVH_COMPARE_THREADS=1 \
benchmark/opencv_compare/run_compare.sh \
  --profile stable \
  --impls auto,ui,scalar \
  --ops CORE_MAT_NEON
```

连续运行三次，以逐 case median 的中位数关闭 gate。

### 11.3 全局门禁

```bash
ctest --test-dir build-core-mat-neon-tests --output-on-failure
./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
./scripts/check_docs.sh
python3 scripts/sync_opencv_intrin.py --check
git diff --check
```

遵循仓库 build-cache 规则：稳定的 tests/compare/sanitizer/x86 配置应持续增量
复用；只有 release、CI 配置验证或 cache 可信度不足时执行 clean build。

## 12. 提交与回退边界

建议提交顺序：

1. `bench(core): add core-mat neon focused observability`；
2. `perf(core): add native-neon reduction backend`；
3. `perf(core): add fused neon rotate90`；
4. `perf(core): add packed neon inRange`；
5. `perf(core): retain accepted elementwise neon kernels`；
6. `docs(bench): publish core-mat native-neon results`。

每个提交必须包含对应 correctness、forced dispatch 和 focused benchmark
证据，并可独立回退。以下情况立即停止或回退 candidate：

- upstream tolerance 被迫放宽；
- NaN/Inf/large-offset-small-variance 合同失败；
- UIOnly/ScalarOnly 误入 direct NEON；
- native candidate 相对 UI 提升低于 `1.25x`；
- 为 benchmark 尺寸或 seed 增加特判；
- 非目标 family 出现稳定超过门槛的回退；
- 非 ARM 编译失败或 header-only/ODR 合同被破坏。

## 13. 实时更新规则

执行期间本文是该专项唯一状态 owner：

- 开始批次前将“待执行”改为“进行中”；
- 每完成 correctness、dispatch、benchmark 或安全 gate，立即写入 revision、
  build identity、命令摘要、结果与未决问题；
- failed candidate 记录数字和回退原因，不把无收益 native code 留在产品路径；
- ignored probe 只用于成本定位，不能代替 clean canonical report；
- date-named report 提交后不可覆盖；
- N6 完成后把当前产品事实同步到 `cpu-optimization.md`，历史实验留在本文。

### 13.1 执行记录

#### 2026-08-06：N1 启动

- 当前分支：`main`；工作区保留尚未提交的 Resize U8C3 专项修改，不覆盖、不清理；
- N1 使用独立且可持续复用的 Core/Mat tests/compare build identity，避免与
  Resize benchmark cache 混用；
- clean performance report 暂不关闭：最终数据必须来自可追踪 revision；在此
  之前允许使用同一 dirty source snapshot 的 focused 数据定位热点，但必须明确
  标为 diagnostic；
- 下一步：审计现有 compare case/filter、dispatch route 和可复用 build cache，
  随后接入 `CORE_MAT_NEON` focused matrix。

#### 2026-08-06：N1 focused matrix 接入

- canonical `cvh_benchmark_opencv_compare` 和 `run_compare.sh` 已支持
  `--ops CORE_MAT_NEON`，没有增加一次性 benchmark target；
- matrix 的 Core/Mat 主体当前覆盖每个 shape 的 26 个 case：reduce axis 0/1 的
  SUM/AVG/SUM2、norm single/diff INF/L1/L2、normalize INF/L1/L2、
  meanStdDev F32C1/C3、rotate90 U8C1/C3/C4 CW/CCW、inRange scalar bounds
  U8C1/C3/C4；另固定加入 479x641 odd ROI；N5 又增加每个 continuous shape
  12 个 add/subtract/multiply U8/F32 C1/C3 case 和 1 个 convertScaleAbs F32C3
  diagnostic case，stable matrix 最终为每个 impl 143 rows；
- 新建并保留 `build-core-mat-neon-compare`（Release、optimization-on、OpenCV
  compare-on）作为本阶段稳定 compare cache；
- Auto/UI/Scalar quick smoke 均完成：每个 impl 52 rows，共 156 rows，三种
  dispatch mode 未串路；诊断文件位于 build 目录，不作为归档报告；
- smoke 暴露的 telemetry 缺口：46/52 Auto case 只能显示笼统
  `opencv_ui` route；rotate counterclockwise 因最后一个 vertical flip 覆盖 tag，
  被错误显示为 scalar。下一步先修正 composite/stage route，再冻结 baseline。

#### 2026-08-06：N1 stage telemetry 完成

- reduction、norm single/diff、normalize、meanStdDev、inRange 已报告具体
  load/accumulate/merge/channel-reduce/tail route；
- rotate90 现在保存 transpose 与 flip 两个阶段的 dispatch，并生成 composite
  tag；不再由最后一个 flip 覆盖整个 public call 的观测结果；
- 第二轮 Auto/UI/Scalar quick smoke 均为 52 rows；Auto/UI 可区分 10 条真实
  route，Scalar 可区分 8 条 fallback route；UI/Scalar forced mode 未串路；
- 已确认 C3 CCW 当前确实为 `transpose=scalar;flip=scalar;temporary=full`，这是
  产品事实而非 telemetry 误报；C1/C4 CCW 则可观察到 UI transpose + scalar
  vertical flip；
- 下一步：在同一 compare cache 上完成三轮 stable diagnostic baseline，随后进入
  N2。clean baseline 仍等待可追踪 revision 后补跑。

#### 2026-08-06：N1 diagnostic baseline 完成，N2 启动

- 三轮 stable Auto/UI/Scalar diagnostic 均为每个 impl 104 rows；输入、OpenCV
  配置、threads=1、warmup=2、iters=20、repeats=5 保持一致；
- Auto 与 UI 当前基本重合；family 几何平均 relative：REDUCE `0.3617`（受
  OpenCV 异常慢的 SUM2 axis1 拉高；SUM/AVG 目标 case 约 `0.20–0.32`）、NORM
  `0.3106`、NORMALIZE `0.3971`、MEAN_STDDEV `0.4236`、ROTATE90 `0.6013`、
  IN_RANGE `0.5734`；
- 1080p continuous 代表值：SUM/AVG `0.214–0.250`，norm `0.233–0.388`，
  meanStdDev F32C3 `0.304`，rotate90 U8C3 `0.490–0.493`，inRange U8C3
  `0.624`；
- 部分长耗时 case 的三轮极差超过 5%，因此这些数据只承担候选排序；最终
  retention 必须在 candidate 完成后重新测量；
- N2 采用 Candidate B：F32 load、F64 NEON 多累加器、scalar tail；meanStdDev
  使用稳定 two-pass centered M2，禁止整图 F32 sum 或 raw-moment variance。

#### 2026-08-06：N2 Candidate A/B 筛选与 correctness

- 首版 direct-F64 结果：reduce axis1 和 meanStdDev F32C3 有收益，但 norm
  因逐 vector F32→F64 转换整体慢于 UI；该 norm candidate 已被替换，没有按原样
  留在 selector；
- 当前 norm 使用每 256 values 的四路 F32 accumulator，并在 block 末尾转 F64
  合并；NaN mask 延迟到整次 kernel 归约，极大有限值/溢出风险自动重跑 F64
  fallback；single INF 使用无转换的四路 max kernel；
- reduce axis1 使用 F64 多累加器；axis0 改为 row-major source stream + 小型 F64
  accumulator buffer，消除了首版 column-major source stride；
- meanStdDev F32C3 保留 two-pass centered F64 kernel，约比 UI 快 `2.4–2.8x`；
  F32C1 只有 `0.85–0.96x`，已从 native selector 删除并回到 UI；
- targeted `*Reduction*`：32/32 通过；新增 forced Neon/UI/Scalar、C1/C3 ROI、
  odd tail、wide finite difference、NaN/Inf、large offset + small variance 测试；
- 单轮 stable probe 的典型 retention：reduce `1.6–3.7x`、norm 多数 case
  `1.25–1.8x`、meanStdDev C3 `2.4–2.8x`；最终保留仍以三轮 candidate
  median 决定；
- N3 fused rotate90 的 8x8 direct-store prototype 已开始接入，但在 N2 三轮 gate
  关闭前不标记 N3 完成。

#### 2026-08-06：N3 fused rotate90 candidate 完整回退

- 完成 U8C1/C3/C4、CW/CCW 共用的 8x8 fused direct-store 原型与 focused
  correctness；原型消除了完整 transpose 临时图，但小块内的 gather/store 成本
  没有被 Apple M5 上的收益覆盖；
- 单轮 stable diagnostic 相对现有 UI/composite 路径：C1 仅 `0.06–0.13x`，C4
  `0.34–0.49x`，C3 `0.97–1.14x`，全部未达到 `1.25x` retention gate；
- 已删除 prototype header、selector 和 prototype-only test，不在产品中保留无收益
  native 路径；现有 transpose+flip 行为保持不变；
- 保留 N1 增加的 stage-level/composite telemetry，使 CW/CCW 的 transpose、flip、
  temporary 路线可长期观测；N3 按“完成实验并回退”关闭，rotate performance floor
  不标记达成。

#### 2026-08-06：N4 inRange packed NEON 实现完成

- 新增 scalar-bound U8C1/C3/C4 direct-NEON 路径：C1 使用连续 16-byte
  load/compare/store，C3/C4 使用 `vld3q`/`vld4q` deinterleave、逐通道 inclusive
  compare 与 channel AND；ROI 按 step 逐 row 处理，尾部沿用 scalar 合同；
- lower/upper 继续通过 canonical scalar-bound prepare 逻辑转换，保留 fractional、
  越界 bound、inclusive edge 和空区间语义；Mat-bound 与非 U8 输入继续 UI/scalar；
- forced Neon/UI/Scalar、U8 scalar bounds、Mat bounds、ROI/tail 与 alias focused tests
  3/3 通过；只有实际执行 vector block 才记录 NEON route；
- 单轮 stable diagnostic 相对 UI retention：C1 `1.67–1.93x`、C3
  `10.75–11.28x`、C4 `8.93–9.23x`；相对 OpenCV 分别为 `1.00–1.28`、
  `6.68–6.81`、`4.35–4.60`，family 几何平均 relative `3.2132`；
- N4 已明显超过 `1.25x` retention 与 `relative >= 0.80` floor。最终保留结论仍在
  N6 用三轮 stable median、sanitizer 与 full regression 关闭。

#### 2026-08-06：N5 elementwise candidate 筛选完成

- `CORE_MAT_NEON` 已补齐 add/subtract/multiply U8/F32 C1/C3 continuous case；
  benchmark 现在记录 native/UI/scalar 的真实 dispatch、ISA 与 kernel route；
- 首版共享原型覆盖 U8/F32 C1/C3、add/subtract/multiply、continuous flatten、逐
  row ROI、4-vector unroll、alias 与 scalar tail；focused correctness 5/5 通过；
- 三轮 stable diagnostic 的 36 个逐 case median 中有 18 个未通过 retention 或
  stability gate：F32C3 几何平均 `1.2129x`，U8 虽有局部收益但存在 `<1.25x`
  case 和 480p 抖动；F32C1 add 的 1080p median 为 `1.2490x`，同样严格回退；
- 第二组最终三轮数据确认 F32C1 multiply 1080p median retention 只有
  `1.2482x`，按严格 `>=1.25x` gate 回退；最终只保留 F32C1 subtract，三种
  canonical shape 的 median retention 约 `1.31–1.37x`，相对 OpenCV约
  `1.00x`；
- 该轮产品 selector 曾收缩到 F32C1 subtract；U8、F32C3、add 与 multiply 的 native 实现和对应
  candidate-only 预期已删除，继续使用现有 UI/scalar；保留 F32C1 ROI/tail、
  NeonOnly/UIOnly/ScalarOnly 与 alias correctness；
- convertScaleAbs F32C3 direct-NEON diagnostic correctness 通过，但 480p/720p/
  1080p 相对 UI 仅 `1.0011x/1.0006x/1.0005x`，相对 OpenCV仍约 `0.60`；确认
  UI 已生成等价 FMA/round/pack 指令链，差距不来自 UI wrapper。原型 header、
  selector 和一次性 forced test 已完整删除；canonical diagnostic benchmark 保留。
- 最终 canonical stable 三轮中，F32C1 subtract 1080p median retention 为
  `1.2485x`，低于统一 `>=1.25x` gate；按与 add/multiply 相同标准，subtract
  native header、selector 和专用 forced test 也已删除。N5 最终没有保留任何
  arithmetic/convertScaleAbs native 路径，canonical diagnostic rows 继续长期保留。

#### 2026-08-06：N2 第二轮 reduction/norm 数据流优化

- 首轮最终源码三轮 stable 暴露：原生路径 retention 大多通过，但 reduce axis0
  SUM/AVG relative 仅 `0.318–0.417`，norm family 几何平均 relative `0.4552`，
  normalize `0.5235`；L2 diff 480p retention `1.2394x`，因此 N2 未按“局部变快”
  提前关闭；
- reduce axis0 改为每 32 rows 的 F32 vector accumulator，再周期性并入 F64
  accumulator；block 出现 NaN/Inf/overflow 时自动从头重跑 direct-F64 fallback。
  首版 SUM2 曾在 block 和 merge 重复平方，focused test 立即失败并在 benchmark 前
  修正；新增跨 block accuracy 与 extreme fallback test；
- 单轮 stable diagnostic 显示 axis0 相对 UI提升到 `2.49–2.80x`，相对 OpenCV
  提升到约 `0.50–0.72`（AVG odd ROI 单轮为 `0.484`，留待三轮 median 判断），
  已显著跨过旧实现的带宽/转换瓶颈；
- norm INF single/diff 分离为专用 max kernel；L1/L2 block 从 256 扩至 1024 values，
  移除每 vector 的重复 NaN/overflow bookkeeping，改为 block-end finite check，异常
  自动重跑 F64；新增 4099-value multi-chunk numeric contract test；
- norm 单轮 stable family 几何平均 relative 提升到 `0.6639`，所有 case retention
  高于 `1.61x`；normalize 复用该 reduction 后为 `0.5997`，随后增加 F32C1
  4-stream native F64-scale apply pass；continuous case relative 提升到
  `0.625–0.835`，ROI 单轮存在调度噪声，最终结论仍以新的三轮 stable 为准；
- 第二轮修改后的 norm/reduce/normalize focused tests 当前 16/16 通过；N2 仍为
  “进行中”，待 multi-chunk 新测试、full OpenCV contract 和三轮 stable 关闭。

#### 2026-08-06：N2 axis-0 block 收口与最终稳定性测试启动

- axis-0 F32 block-merge 先从 32 rows 扩至 128 rows；一次 register-tiled
  accumulator 实验使 relative 下降到约 `0.25–0.35`，已完整回退，没有留在产品
  路径；
- 128-row row-major 版本单轮可达到约 `0.52–0.76`，但三轮数据中 odd ROI SUM
  median 为 `0.4933`，没有用单次通过掩盖边界波动；最终把通用 block 扩至
  256 rows，以减少 F32 block 到 F64 merge 的固定成本；
- 256-row 版本新增 259-row 跨 block accuracy 与 `1e38` extreme fallback case；
  reduce/norm 及相关 forced-mode focused tests 共 11/11 通过；
- 同一 Release、optimization-on、OpenCV `build-slim`、threads=1 cache 的单轮
  stable probe 中，axis-0 全部 case 达到 floor：最弱的 479x641 odd ROI SUM 为
  `0.5039`，1080p SUM/AVG 为 `0.7449/0.7436`，SUM2 为 `0.7734`；
- 当前进入最终高分辨率三轮 Auto/UI 测量；使用更长 warmup/iteration/repeat 窗口
  降低 sub-0.05 ms case 的计时分辨率影响。完成前 N2 仍保持“进行中”。
- 首组三轮高分辨率数据发现明显的非 kernel 调度干扰：部分相邻 case 跨轮跳变
  `2–3x`，最大极差超过 `200%`，连纯 streaming subtract 也达到 `146%`；因此
  该组三轮不用于关闭稳定性 gate；
- canonical compare 在 macOS 上现显式设置 main thread 为 user-initiated QoS，
  防止从 Codex/CI 等后台宿主继承 utility/background scheduling class；同一进程内
  的 CVH 与 OpenCV 使用完全相同的 QoS。下一步先验证该设施确实降低跨轮方差，
  再重跑最终三轮，不选择性采用受干扰数据中的最好一轮。
- QoS 验证的后两轮多数 case 已收敛，但每个新进程开头的 480p short cases 仍有
  约 `30–66%` 的首轮爬升差异；canonical compare 因此增加固定 process CPU
  precondition；500 ms 验证仍呈现整批逐轮加速，最终固定为 3 秒。该阶段在任何 case 前执行且不计入 latency，Auto/UI 都执行，
  用于统一 macOS 频率/QoS 启动状态，而不是扩大单个算子的 warmup 取最好值。
- 3 秒 precondition 的三轮中等窗口验证后，79 个 native case 有 77 个达到
  `<=5%` 极差；仅约 0.05 ms 的 480p L1/L2 diff norm 为 `7.00%/6.60%`。
  所有 case retention 已达到 `>=1.25x`；family 几何平均 relative 为 reduce
  `1.0489`、norm `0.6635`、normalize `0.6976`、meanStdDev `0.7875`、inRange
  `3.2035`、subtract `1.0058`。现以更长采样窗口重跑最终三轮关闭两个短 case。
- `warmup=5/iters=100/repeats=10` 的超长三轮没有作为最终数据：第 1/2 轮高度
  重合，但第 3 轮整批约慢 `7%`，共 49 个 case 超过 5% 极差，呈现持续负载后的
  热/系统漂移；route 79/79、retention 79/79 和 family floor 仍通过。后续回到
  canonical stable 量级，并在三轮之间执行非性能门禁、避免连续长压测；不从该组
  数据中选择最好两轮关闭 gate。
- subtract 回退后的首组 canonical 三轮也不用于关闭稳定性 gate：第 1/2 轮几乎
  重合，但第 3 轮紧跟 optimization-off 全量重编译，76/76 native case 整批出现
  `12%–73%` 的延迟抬升；route 仍为 76/76 NEON、三轮逐 case median retention
  仍全部 `>=1.25x`，各 family floor 仍通过。这证明把编译任务当作“间隔”会引入
  热状态偏差；最终三轮将使用明确的低负载冷却间隔重新完整采集，不能只挑第 1/2
  轮拼成通过结果。
- 60 秒低负载间隔的第二组完整三轮确认 76/76 retained case 路线、retention 和
  family floor 均通过，但仍有 16 个亚毫秒 case 因孤立调度尖峰超过 `5%`；把
  repeats 从 5 增到 10 后仍有 12 个，说明继续整批重跑不能可靠关闭门禁。
- canonical runner 因此新增长期可复用的 `CORE_MAT_NEON_RETAINED` 子过滤器：只
  保留最终产品中的 reduce/norm/normalize/meanStdDev F32C3/inRange 76 rows，原
  143-row `CORE_MAT_NEON` candidate/fallback 诊断矩阵不删不改。该子集用于提高
  每个 case 的采样时长，避免为了短 case 把失败 candidate 和 rotate 一起长时间
  压测并引入新的热漂移。
- retained-only `warmup=5/iters=100/repeats=5` 首组三轮把 reduce、normalize、
  meanStdDev、inRange 的 candidate 极差压到 `4.01%/3.98%/2.82%/2.67%`，但
  norm 的 8 个短 ROI/480p case 仍受桌面调度尖峰影响；更重要的是高分辨率数据
  确认 reduce SUM2 axis0 relative 只有 `0.426–0.466`，没有达到逐 target case
  `>=0.50` floor。N2 因此继续优化 axis0 accumulator 数据流，不采用先前短采样
  的临界通过结论。
- axis0 现按 4-row group 同时流式读取四条相邻源行，每个 vector accumulator
  每四行只 load/store 一次；仍保留 256-row F32 block、F64 merge 与异常时整次
  F64 fallback。focused reduction/norm/statistics 14/14 通过；单轮同参数
  high-resolution probe 相对上一版本的 axis0 SUM/AVG 提升 `1.65–1.85x`，SUM2
  提升 `2.69–2.99x`。最终 relative：continuous SUM/AVG/SUM2 为
  `1.01–1.32`，odd ROI 为 `0.90–1.25`，12/12 axis0 case 均越过 `0.50`
  floor，下一步在最终源码上重新执行完整三轮。
- 4-row 最终源码的 retained-only 高分辨率三轮为 76/76 NEON、76/76 retention
  `>=1.25x`，最低 retention `1.5949x`。family 几何平均 retention/relative：
  reduce `3.8228/1.3776`、norm `2.1666/0.6611`、normalize
  `1.8079/0.6899`、meanStdDev `2.6461/0.7883`、inRange
  `5.7288/3.4945`；reduce 最弱 target case relative `0.5469`，所有完成 floor
  均通过。
- strict candidate latency 极差仍有 7/76 未关闭：6 个 odd ROI norm 为
  `6.14%–11.85%`，1 个 odd ROI normalize L1 为 `5.32%`；其余 family 最大极差
  为 reduce `2.83%`、meanStdDev `0.64%`、inRange `4.06%`。这些短 case 的三轮
  latency 分别约 `0.018–0.052 ms` 和 `0.066–0.069 ms`，retention/floor 均有充足
  余量；但按本文 `<=5%` 原定义，N2 stable gate 仍明确保留为“需隔离宿主关闭”，
  不在当前高负载桌面环境伪记通过。

#### 2026-08-06：N6 Apple ARM64 全量正确性门禁通过

- 复用 `build-core-mat-neon-tests`：Release、optimization-on、OpenCV compare-on，
  使用 OpenCV `build-slim`；完整增量构建成功；
- `ctest --test-dir build-core-mat-neon-tests --output-on-failure` 为 21/21 通过；
  覆盖 full Core、full Imgproc、OpenCV contract、GEMM ISA、公开头独立编译、
  Core/Imgproc ODR、aggregate/include-only 与 dispatch smoke；
- 本阶段 reduction/statistics、inRange、subtract 的 forced Auto/NeonOnly/UIOnly/
  ScalarOnly，以及 ROI/tail/alias/special-value tests 均包含在上述全量通过中；
- 下一步复用稳定 optimization-off、sanitizer 与 x86_64 build identity，继续关闭
  跨配置门禁。
- F32C1 subtract 按最终 stable retention 回退后，再次复用同一 Release build
  完成全量增量构建，CTest 仍为 21/21 通过；因此当前产品态（N5 不保留任何
  arithmetic native selector）已经重新覆盖 full Core、Imgproc 与 OpenCV contract，
  不是沿用回退前的测试结论。
- axis0 4-row 最终优化和 retained 子过滤器接入后，又一次完成 Release 全量增量
  构建，CTest 21/21 通过；最终性能源码已覆盖 full Core、full Imgproc、OpenCV
  contract、GEMM ISA 与全部 smoke/ODR 测试。

#### 2026-08-06：N6 optimization-off、sanitizer 与 x86_64 compile 通过

- 复用 `build-b7-optimization-off`（Release、optimization-off）：完整构建成功，
  CTest 18/18 通过；native header 在禁用优化时没有暴露 NEON symbol；
- 复用 `build-phase2-sanitize`（Debug、optimization-on、
  `-fsanitize=address,undefined`）：完整构建成功，CTest 20/20 通过；ROI、tail、
  alias 与 special-value 路径未触发 ASan/UBSan；
- 复用 `build-v01-neon-hot-x86-cross`（Release、`x86_64`、optimization-on）：
  全量编译和链接成功，产物确认为 Mach-O x86_64；三个新增 native header 的
  非 ARM guard、header compile 和 ODR 均通过编译；
- 本机未安装 Rosetta，尝试运行 x86_64 CTest 时所有进程均由系统以 error `-86`
  拒绝启动；因此没有把“未执行”记作 runtime 通过，Linux x86_64 runtime 继续
  明确保留为外部 CI 门禁。
- subtract native 最终回退后，optimization-off build 已再次完成全量增量编译，
  CTest 18/18 通过；禁用优化的最终头文件集合与 selector 状态已复验。
- 同一最终产品态随后在 ASan/UBSan build 完成全量增量编译，CTest 20/20 通过；
  回退没有遗留悬空 selector，保留的 reduction/statistics 与 inRange ROI/tail 路径
  未触发 sanitizer。
- x86_64 cross build 也在最终产品态重新完成全量编译和链接；`cvh_test_core` 与
  `cvh_test_imgproc` 均确认为 Mach-O x86_64。Linux x86_64 runtime 的外部门禁
  约束保持不变。
- axis0 4-row 最终优化后，optimization-off 又完成一次全量增量编译，CTest
  18/18 通过；ASan/UBSan 随后也在同一最终源码完成全量增量编译，CTest
  20/20 通过；x86_64 cross build 也完成全量编译链接，Core/Imgproc 测试产物均
  确认为 Mach-O x86_64。header gate 继续刷新。

#### 2026-08-06：N6 header/install gate 首轮计数清单修正

- `check_header_only_contract.sh` 通过：public boundary/dependency/macro checks 与
  临时安装树 consumer/compile/ODR CTest 12/12 全绿；
- `ci_headers_all.sh` 首轮的代码门禁全部通过：installed-header 12/12、仓库
  CTest 20/20、Core 219/219、Imgproc 208/208；最终仅因长期清单仍预期 Core
  213 而退出 1；
- 本阶段新增的 6 个 Core tests 已把 ARM64/x86_64 `ui-on` 期望同步为 219；
  Imgproc 208 保留此前 Resize 专项的用户修改。下一步从头复跑正式 gate。
- 正式 gate 复跑通过：installed-header 12/12、仓库 CTest 20/20、Core
  219/219、Imgproc 208/208，XML report 与当时的期望清单一致；
- 随后的 OpenCV intrinsic sync check 确认所有 vendored 文件与 upstream HEAD
  一致，但旧脚本会把本机绝对 source path 和无关工作区 `-dirty` 状态写入
  `UPSTREAM.md`。生成规则已改为稳定的官方 repository URL 与 HEAD describe，
  不复制、不改写 intrinsic 源文件；待重新执行 sync check。
- 修正规则后 `sync_opencv_intrin.py --check`、`check_docs.sh` 与
  `git diff --check` 均通过；`cpu-optimization.md` 已同步 reduce、norm/normalize、
  meanStdDev、inRange 的 eligibility、native route 与 fallback inventory；subtract
  在最终 strict gate 回退后已从 inventory 删除。
- subtract 专用 test 删除后，长期 ARM64/x86_64 `ui-on` Core 期望已由 219 调整为
  最终的 218；重新从头执行 `check_header_only_contract.sh` 为 12/12，通过完整
  `ci_headers_all.sh` 为 installed-header 12/12、仓库 CTest 20/20、Core
  218/218、Imgproc 208/208。当前 header/install 结论不再引用回退前的 219 清单。
- axis0 4-row 与 retained filter 的最终源码又从头执行 header 门禁：独立
  `check_header_only_contract.sh` 12/12，通过正式 `ci_headers_all.sh` 的
  installed-header 12/12、仓库 CTest 20/20、Core 218/218、Imgproc 208/208；
  因此上述计数和安装合同也是最终源码结果。

#### 2026-08-06：N6 full product-auto dirty diagnostic 通过 aggregate gate

- 复用 `build-core-mat-neon-compare`，以 full、Auto、threads=1、
  `warmup=1/iters=10/repeats=3` 运行 canonical binary；输出位于 build 目录，
  executable 报告 370 rows，其中 369 个 `OK` case 与最近 clean full baseline
  一一匹配，无 missing/new key；
- 排除本阶段 `REDUCE/NORM/NORMALIZE/MEAN_STD_DEV/IN_RANGE/SUBTRACT`，并排除
  工作区中独立的 `RESIZE` 专项后，共 321 个 non-target common case；CVH latency
  几何平均 current/baseline 为 `0.9680`，即改善 `3.20%`，通过 aggregate 回退
  `<=1%` gate；
- 同 family fallback 也未出现 aggregate 回退：arithmetic UI/scalar fallback
  `0.9869`、reduce MIN/MAX fallback `0.9934`、normalize MINMAX fallback
  `0.9907`；
- 该结果来自未提交工作区，只承担 N6 regression 诊断；date-named clean
  Markdown/CSV/metadata 仍须在用户授权提交后的可追踪 revision 上生成。
- subtract 回退后已再次运行 full Auto diagnostic：370 rows 中 369 个 `OK` case
  与 2026-08-06 clean baseline 全部匹配；排除
  `REDUCE/NORM/NORMALIZE/MEAN_STD_DEV/IN_RANGE/SUBTRACT/RESIZE` 后仍为 321 个
  non-target common case，current/baseline CVH latency 几何平均 `0.9746`，即改善
  `2.54%`，aggregate gate 继续通过；subtract 的 16 个 full case 已全部回到
  `opencv_ui` route。
- 最终同-family fallback 几何平均 current/baseline：ADD/SUBTRACT/MULTIPLY
  `1.0112`、reduce MIN/MAX `1.0026`、normalize MINMAX `0.9914`，均在 `<=5%`
  回退限制内。该轮仍是 dirty diagnostic，不替代 clean archive。
- axis0 4-row 最终源码在 60 秒低负载冷却后再次完成 full Auto diagnostic：369
  个 `OK` case 与 baseline 全部匹配；321 个 non-target common case 的
  current/baseline 几何平均为 `0.9624`，即改善 `3.76%`。同-family fallback：
  ADD/SUBTRACT/MULTIPLY `0.9976`、reduce MIN/MAX `1.0083`、normalize MINMAX
  `1.0091`，均通过 `<=5%`；subtract 16/16 仍为 `opencv_ui` route。紧跟 clean
  header 编译的一次热态 full run曾使 arithmetic 达到 `1.0779`，已完整记录但
  未用于关闭门禁；冷却后整轮复测而非抽取单 case。

## 14. 完成定义

- [x] 已解释 ARM64 UI 与 native NEON 的关系，不以 dispatch 标签作为优化目标；
- [x] 已按差距、绝对损失、调用频率和实现成本完成候选排序；
- [x] `CORE_MAT_NEON` canonical focused matrix 与 stage telemetry 完成；
- [x] reduce/norm/normalize/meanStdDev 共享 native backend 通过数值合同；
- [x] reduction/statistics 达到各自 performance floor；
- [x] fused rotate90 U8C1/C3/C4 已完成候选验证；未达到 retention/floor，产品
  native 路径与 candidate-only test 已完整回退；
- [x] inRange U8C1/C3/C4 达到 `relative >= 0.80`；
- [x] elementwise candidate 仅在相对 UI `>=1.25x` 时保留；
- [x] 所有保留路径覆盖 Auto/NeonOnly/UIOnly/ScalarOnly；
- [x] ROI、non-contiguous、unaligned、odd/tail、alias、special values 通过；
- [x] optimization-on/off、ASan/UBSan、header、ODR、install consumer 通过；
- [x] Apple ARM64 full Core 与 OpenCV contract full 通过；
- [x] x86_64 compile 通过，Linux x86_64 runtime 完成或明确保留外部门禁；
- [x] full Core/full compare 非目标 aggregate 回退不超过 `1%`；
- [ ] 76 个 retained native case 的三轮 CVH latency 极差全部 `<=5%`；当前仅
  69/76，剩余 7 个亚毫秒 odd-ROI case 待隔离宿主复测；
- [ ] 新 clean product-auto Markdown/CSV/metadata 已归档；
- [x] CPU dispatch inventory 和本文状态已同步；
- [ ] clean report 生成后同步 results index。
