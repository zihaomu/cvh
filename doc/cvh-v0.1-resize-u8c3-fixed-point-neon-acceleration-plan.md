# cvh v0.1 Resize U8C3 定点 NEON 加速计划

更新时间：2026-08-06
状态：R0–R5 完成；R6 本地实现与 dirty candidate 完成，clean report 待提交后重跑

## 1. 目的与阶段定位

本文定义 v0.1 发布前 `resize` 的专项性能收口，产品路线固定为：

```text
C3 扁平字节流
  + 8-bit 定点双线性插值
  + 连续 NEON load/store
```

本阶段承接
[v0.1 高频 NEON 热点加速计划](cvh-v0.1-neon-hot-kernel-acceleration-plan.md)
中尚未关闭的 Resize H2 performance floor。旧 H2 候选要求 direct NEON 与
现有 float/std::round scalar 逐字节一致，因此 fixed-point candidate 因
half-way tie 修正成本过高而回退。后续调查确认：

1. 当前 upstream OpenCV 在 Apple ARM64 上实际命中 KleidiCV HAL，而不是
   OpenCV 通用 resize kernel；
2. 当前 cvh 已快于关闭 KleidiCV/Carotene 后的 OpenCV 通用实现；
3. 主要差距来自 cvh 的 `vld3 + per-channel table + U8/F32 conversion +
   float FMA + vst3` 数据流；
4. 独立诊断原型使用扁平 C3、8-bit 定点和连续 NEON load/store 后，目标 case
   端到端达到 `0.057471 ms`，OpenCV/KleidiCV 为 `0.056633 ms`；
5. 现有 OpenCV differential 对 U8 linear resize 的合同本来就是最大误差
   `1`。诊断原型在目标 case 上与 KleidiCV 逐字节一致。

因此，本阶段不再微调现有浮点 gather kernel，而是建立新的定点数值参考和
扁平化 direct-NEON kernel。OpenCV upstream 继续是结果合同；性能优化不引入
OpenCV/KleidiCV 二进制依赖，也不复制第三方实现。

## 2. 范围边界

### 2.1 必做范围

- 输入与输出：`CV_8UC3`；
- 插值：`INTER_LINEAR`；
- 首个 product predicate：宽高均为精确 `0.75x` 下采样；
- layout：continuous、ROI、非连续 step 和非对齐 row start；
- ARM64：`Auto` / `NeonOnly` 命中新 direct-NEON kernel；
- 所有平台：建立同一套 fixed-point scalar reference，用于边界、tail、
  fallback 和 direct-kernel 精确对照；
- 保留当前 `0.5x` U8C3 专用 2x2 kernel，不用新路径覆盖已经更快的实现；
- 保留现有浮点通用路径作为未纳入 predicate 的 fallback；
- 更新 dispatch telemetry、专项单测、OpenCV differential、canonical
  benchmark 和本文实时状态。

### 2.2 验证后才允许扩展

- 将 predicate 从精确 `0.75x` 扩展到宽高均下采样，且每一维比例位于
  `[1/3, 1)`；
- 为相邻 block 成对处理 32 个输出字节；
- 根据实测增加 prefetch 或 workload threshold；
- 让其他 U8 通道数复用 fixed-point coordinate builder。

这些项目不得与首个 0.75x product kernel 同时扩大。只有正确性、跨平台和
性能门禁关闭后，才逐项启用。

### 2.3 明确不包含

- `INTER_NEAREST`、`INTER_NEAREST_EXACT` 或其他插值模式；
- U8C1、U8C2、U8C4、F32 和其他深度；
- upscale、单行/单列退化图像的专用 NEON；
- 修改公开 API、枚举、CMake product target 或 header-only 合同；
- 引入 OpenCV、KleidiCV 或其他运行时依赖；
- 使用多线程掩盖单线程 kernel 差距；
- 为固定 benchmark 尺寸硬编码坐标表、宽度或图像内容；
- 放宽既有 OpenCV contract、关闭 checksum 或删除困难 case；
- 将 ignored 诊断程序加入产品 benchmark/CI。

## 3. 冻结证据与性能口径

正式基线报告：

- [v0.1 NEON hot-kernel report](../benchmark/opencv_compare/results/2026-08-06-v0.1-neon-hot-opencv-upstream-performance.en.md)；
- [raw CSV](../benchmark/opencv_compare/results/2026-08-06-v0.1-neon-hot-opencv-upstream-performance.csv)。

相对性能继续使用：

```text
relative performance = OpenCV latency / CVH latency
1.0 = 持平；小于 1.0 = CVH 更慢
```

### 3.1 Canonical product-auto 基线

| case | CVH | OpenCV | relative | 等价差距 |
| --- | ---: | ---: | ---: | ---: |
| U8C3 0.75x continuous | `0.169096 ms` | `0.060208 ms` | `0.3561` | OpenCV `2.81x` |
| U8C3 0.75x ROI | `0.156067 ms` | `0.055342 ms` | `0.3546` | OpenCV `2.82x` |

### 3.2 2026-08-06 单线程诊断分解

下表来自 ignored build 目录中的本机诊断程序，只用于确定实现方向，不能代替
clean revision 的 canonical 报告：

| 路径 / 阶段 | 耗时 | 结论 |
| --- | ---: | --- |
| 当前 CVH public Auto | `0.159746 ms` | 与正式报告差距一致 |
| 当前 CVH mapping/allocation | `0.000640 ms` | 约占 `0.4%`，不是主瓶颈 |
| 当前 CVH vector load/store floor | `0.022125 ms` | 内存基本下限较低 |
| 当前 CVH vector gather/store | `0.037445 ms` | table gather 不是主瓶颈 |
| 当前 CVH float math without table | `0.142513 ms` | U8/F32 转换与浮点插值占主导 |
| 当前 CVH full vector blocks | `0.151076 ms` | 标量尾部不是主导 |
| 定点扁平化原型 core | `0.056848 ms` | 已接近 OpenCV HAL |
| 定点扁平化原型 mapping | `0.000731 ms` | 端到端建表成本可接受 |
| 定点扁平化原型 end-to-end | `0.057471 ms` | 比 OpenCV 慢约 `1.5%` |
| OpenCV/KleidiCV | `0.056633 ms` | 当前 upstream 实际路径 |
| OpenCV 无 KleidiCV/Carotene | `0.250171 ms` | 当前 CVH 比通用实现快约 `1.57x` |

正确性诊断：

- 当前 CVH 对 OpenCV：`518400` 个输出 byte 中 `141627` 个相差 `1`
  （`27.32%`），相差超过 `1` 的数量为 `0`；
- OpenCV public resize 与 direct KleidiCV：目标 case `0` 个 byte 不一致；
- 定点扁平化原型与 direct KleidiCV：目标 case `0` 个 byte 不一致。

R1 必须在 clean revision 上重新冻结目标 benchmark 与 checksum。以上原型数字
只是 feasibility proof，不能直接写入发布性能声明。

## 4. 根因与选定架构

### 4.1 当前内核的成本

当前通用 U8C3 NEON 每个 block 处理 8 个输出像素，执行：

```text
每个 source row 使用 vld3q_u8 反交织 C3
  -> 三个通道分别执行四次 table gather
  -> 12 组 U8 扩展为 24 组 F32x4
  -> 浮点 horizontal/vertical FMA
  -> F32 round/narrow 为 U8
  -> vst3_u8 重新交织
```

mapping、dispatch 和 scalar tail 都不是主要损失。继续增加 float block 宽度、
只减少 table 次数或只优化尾部，无法关闭约 `0.10 ms` 的浮点转换与计算成本。

### 4.2 新内核的数据流

新内核把 C3 图像视为连续 byte stream，不拆分通道：

```text
连续加载 top/bottom 各 32 bytes
  -> TBL2 取得 a/b/c/d 四组相邻 byte
  -> 8-bit wy 完成 vertical integer lerp
  -> 8-bit wx 完成 horizontal integer lerp
  -> 连续存储 16 output bytes
```

一个 16-byte 输出向量可以跨越像素和通道边界；每个 lane 的右邻点索引恒为
`left_index + 3`。这消除了 C3 反交织、三通道重复控制流和所有 U8/F32 转换。

### 4.3 数值合同

R1 冻结以下内部定点语义：

- 使用 64-bit 整数计算 half-pixel aligned source coordinate，避免尺寸乘法
  溢出和浮点平台差异；
- source index 来自坐标整数部分；
- x/y fraction 取对齐坐标的小数高 8 bit；
- 先做 vertical lerp，再做 horizontal lerp；
- 每一级使用等价于 `vraddhn` 的 round-and-narrow；
- 左、右、上、下边界均通过显式 clamp/replicate 处理，不依赖越界 table lane；
- fixed scalar reference 与 direct NEON 对相同 predicate 必须逐字节一致；
- 对冻结的 ARM upstream/KleidiCV 目标 case 争取逐字节一致；
- 对完整 OpenCV upstream differential 保留现有最大误差 `1`，不得再放宽。

该迁移可能使目标 predicate 下的 CVH 输出相对旧 float scalar 有最大 `1` 的
变化。这不是降低 upstream 正确性：旧实现本身已有同等级差异，新定点结果在
目标 case 上更接近当前 upstream。R1 必须把变化范围、tie case 和跨 dispatch
一致性写成测试，不能只凭 benchmark 接受。

## 5. 内部数据结构与 kernel 设计

### 5.1 调用级映射

建议使用只读调用级结构，不暴露到公开 API：

```cpp
struct FixedResizeBlockU8C3
{
    int source_byte_base;
    std::array<unsigned char, 16> left_index;
    std::array<unsigned short, 16> x_fraction;
};

struct FixedResizeRow
{
    int source_y;
    unsigned short y_fraction;
};
```

规则：

- block map 和 row map 每次 public call 只建立一次；
- hot row loop 内不得分配；
- map 对所有 worker 只读，继续复用现有 `parallel_for_index_if`；
- `source_byte_base` 必须保证本 block 的每个 table index 有效；
- 禁止为了 vector load 在合法 source row 外 overread；
- map builder 使用 `size_t`/64-bit 中间量并检查极端尺寸。

### 5.2 16-byte NEON primitive

每个 block：

1. 从 top/bottom row 各连续加载两个 `uint8x16_t`；
2. 加载 16 个 `left_index`，用 `left + 3` 得到 right index；
3. 用 `vqtbl2q_u8` 取得 `a/b/c/d`；
4. `vshll + vsubl + vmulq + vraddhn` 完成 vertical left/right；
5. 用相同 primitive 和 per-lane x fraction 完成 horizontal；
6. `vst1q_u8` 连续写出 16 bytes。

首版以清晰的 16-byte primitive 为准。32-byte pair、预取、常量复用只有在
profile 证明有额外稳定收益时才加入，不能先增加寄存器压力和边界复杂度。

### 5.3 边界与尾部

- source row 不足安全连续窗口时使用 fixed scalar；
- 正常主体使用 16-byte block；
- 剩余 `8–15` bytes 可评估 8-byte half-vector；
- 剩余不足 8 bytes 使用 fixed scalar；
- 右边界优先回退到显式、安全的 fixed scalar，不做越界加载；
- scalar tail 必须和 vector 主体使用相同的两级定点舍入；
- ROI 使用 `step(0)` 定位每行，禁止把 `isContinuous()` 当成命中条件。

### 5.4 Dispatch 与 telemetry

精确 0.5x 专用 kernel 保持优先。首版选择顺序：

```text
CV_8UC3 + INTER_LINEAR
  -> exact 0.5x: existing ratio_half NEON
  -> exact 0.75x + direct NEON allowed + workload threshold:
       flat fixed-point NEON
  -> fixed scalar reference / existing generic fallback
```

建议新 route：

```text
resize_linear_u8c3:
map=fixed_q16_q8;
layout=flat_c3;
load=neon_contiguous;
gather=tbl2;
interpolate=fixed8_vertical_horizontal;
store=neon_contiguous;
tail=fixed_scalar
```

只有实际进入 NEON 主循环才记录 `DispatchTag::NEON`。小图、窄图或全行落入
scalar 时必须记录真实 fallback；`OpenCVUIOnly`/`ScalarOnly` 不得执行 direct
NEON。

## 6. 正确性门禁

### 6.1 Fixed-point reference

- 建立独立 scalar reference，不从 NEON intrinsic 反推结果；
- 用穷举/构造输入覆盖 `0/1/127/128/254/255`、棋盘格、渐变、常量和随机值；
- 专门覆盖两个插值阶段的 half-way tie、正负像素差和饱和边界；
- scalar 与 NEON 对所有 eligible case 要求 byte-exact；
- 记录相对旧 float path 的 diff count、max diff 和具体 tie 分类。

### 6.2 尺寸与内存布局

- destination byte width：`1–33` 附近逐值、`15/16/17`、`31/32/33`；
- odd width/height、非 4 倍整除尺寸和右边界不足一个 source window；
- continuous、ROI、non-contiguous step、unaligned row start；
- 最小合法 source、窄图、小 workload 和大图；
- guard bytes、ASan/UBSan，证明无前后 overread/overwrite；
- 空输入、无效 size、单行/单列等继续满足公开异常/fallback 合同。

### 6.3 Dispatch 与 upstream

- `Auto`、`NeonOnly`、`OpenCVUIOnly`、`ScalarOnly` 全覆盖；
- ARM64 eligible case 断言新 route，非 ARM 编译时 direct NEON 不可达；
- OpenCV differential 使用多个 seed、多个尺寸、continuous/ROI；
- 完整 differential 的最大 U8 误差不得超过既有 `1`；
- 冻结的 480x640 -> 360x480 ARM/KleidiCV case 要求 byte-exact；
- upstream 使用无 KleidiCV 的构建时允许实现路径不同，但仍受最大误差 `1`
  约束；
- 现有 resize、header compile、ODR、install consumer 和 pipeline tests 不得
  新增失败。

## 7. 性能门禁

所有性能结论必须在同一机器、Release、单线程、相同 OpenCV revision/config、
相同 warmup/iters/repeats 下产生。每个关闭结论至少使用三轮 stable run 的逐
case 中位数。

### 7.1 必达目标

| 项目 | 门槛 |
| --- | ---: |
| 0.75x continuous candidate speedup | 相对当前 CVH `>= 2.20x` |
| 0.75x ROI candidate speedup | 相对当前 CVH `>= 2.20x` |
| 0.75x continuous relative | `>= 0.90` |
| 0.75x ROI relative | `>= 0.90` |
| 480x640 -> 360x480 CVH latency | `<= 0.065 ms` |
| mapping/allocation 占端到端 | `<= 5%` |
| 同 family 非目标 case 回退 | 不超过 `5%` |
| full Imgproc/full compare 非目标几何平均回退 | 不超过 `1%` |

Stretch goal 是 continuous 与 ROI 均达到 `relative >= 0.95`。诊断原型已达到
约 `0.985`，因此本阶段不再沿用旧 H2 的 `0.50` 作为完成线。

### 7.2 防止错误归因

- performance gate 测量 public call end-to-end，包含建表和 dst create；
- benchmark 双方保持 `threads=1`；
- checksum/结果验证必须开启；
- 不使用 `cv::setUseOptimized(false)` 声称关闭 KleidiCV HAL；需要通用 OpenCV
  对照时必须使用配置上关闭 KleidiCV/Carotene 的独立构建；
- no-HAL 数据仅解释实现层级，不替代产品对当前 upstream 构建的主 gate；
- ignored microbenchmark 只能定位成本，不能成为发布报告。

## 8. R0–R6 实施批次

| 批次 | 内容 | 状态 | 主要产物 |
| --- | --- | --- | --- |
| R0 | dispatch/HAL、耗时分解和 feasibility prototype | 完成 | 根因、no-HAL 对照、0.057 ms 原型证据 |
| R1 | 冻结基线与定点数值合同 | 完成 | clean baseline、scalar reference tests、tie/diff inventory |
| R2 | 调用级 fixed map 与 scalar product path | 完成 | Q16/Q8 map、fixed scalar、fallback tests |
| R3 | 16-byte flat-C3 direct NEON 主循环 | 完成 | TBL2 + integer lerp kernel、exact 0.75x dispatch |
| R4 | ROI、边界、tail 与安全性闭环 | 完成 | odd/tail/ROI tests、ASan/UBSan |
| R5 | dispatch、全量正确性与跨平台 | 完成；Linux runtime 为外部门禁 | forced modes、optimization-off、x86 compile/runtime |
| R6 | stable/full 性能与文档收口 | 进行中 | 三轮 stable、clean report、状态与 dispatch 文档 |

### R0：调查结论

已完成：

- benchmark 双方均为单线程；
- OpenCV public resize、关闭 optimized flag 和 direct KleidiCV 耗时相同，确认
  HAL 路径；
- 独立关闭 KleidiCV/Carotene 后测得 OpenCV generic 明显慢于当前 cvh；
- mapping、gather、float arithmetic、tail 已分别测量；
- ignored 诊断原型端到端接近 OpenCV，并在目标 case 上 byte-exact；
- 本轮调查未修改产品源码，开始 R1 前工作树应保持 clean。

### R1：冻结基线与数值合同

#### R1 实时记录（2026-08-06）

- 已开始 R1；当前分支为 `main`。工作树中的 `AGENTS.md`、本文、文档索引和旧
  H2 successor 链接均为本阶段尚未提交的预期修改，未发现需要覆盖的无关用户
  源码改动；
- 按仓库 build-cache 规则，先审计并优先复用现有
  `build-v01-neon-hot-tests`、`build-v01-neon-hot-compare`、
  `build-v01-neon-hot` / sanitizer 与 x86 cross cache；R1 暂不创建新的 build
  目录；
- 下一步冻结现有 Resize targeted 测试、三轮 stable baseline、fixed coordinate
  与两级 rounding reference；在 R1 correctness 关闭前不修改 public selector。
- 复用 `build-v01-neon-hot-tests` 执行实际 GTest filter，Resize targeted
  13/13 通过；suite 包含 `ResizeTest` 9、`ResizeDispatchInternalTest` 2、
  `ResizeUpstreamTest` 2，不存在空 filter；
- 已在同一 `build-v01-neon-hot-compare` cache 上完成三轮
  `stable + V01_NEON_HOT + auto/ui/scalar`，每轮每种 mode 均为 70 行且通过
  checksum。Auto 的 480x640 -> 360x480 三轮中位数为 CVH
  `0.157062 ms`、OpenCV `0.056510 ms`、relative `0.360738`；ROI 为 CVH
  `0.156290 ms`、OpenCV `0.055283 ms`、relative `0.353724`；Scalar
  continuous/ROI 中位数分别为 `0.332717/0.332287 ms`；
- 三轮原始 CSV/metadata 保存在 ignored 的
  `build-v01-neon-hot-compare/results/r1-baseline-run{1,2,3}.csv{,.meta.json}`；
  这些数据用于 candidate 对照，不覆盖已归档的 dated report。
- 已新增独立、全平台可编译的 fixed scalar reference：64-bit overflow-safe
  half-pixel Q16 coordinate、fraction 高 8 bit、vertical 后 horizontal 的两级
  round-and-narrow；首个 shape predicate 同时覆盖 480x640 continuous 与
  479x641 odd ROI 的 floor-3/4 尺寸；
- internal tests 新增 3 项，覆盖 `4->3` 坐标常量、正负 delta、half-way tie、
  continuous/ROI 以及对 legacy float 最大误差 `1`；Resize focused 当前
  14/14 通过；
- OpenCV contract 新增 exact validator；480x640 -> 360x480、U8C3、固定 seed
  的 scalar reference 与当前 ARM OpenCV/KleidiCV byte-exact。相关 contract
  focused 3/3 通过；
- 首次增量编译发现 internal test 未显式包含已有 `resize_test_utils.hpp`，编译
  失败后只补齐测试依赖；重新编译两个 target 成功，未改变产品实现。R1 的
  baseline、数值、tie、legacy diff 与 upstream exact 证据齐备，R1 关闭。

1. 从 clean revision 运行现有 Resize targeted tests 和三轮 stable baseline；
2. 将 fixed coordinate、fraction、两级 rounding 写成可单测 scalar primitive；
3. 对旧 float、fixed scalar、OpenCV/KleidiCV 建立 diff inventory；
4. 用 adversarial tie inputs 确认差异不超过 `1`；
5. 更新 internal test：eligible fixed scalar 与 NEON 要求 exact，OpenCV 保持
   tolerance `1`，冻结 target case 对当前 upstream exact；
6. R1 correctness 未关闭前，不允许把诊断 NEON 接入 public selector。

### R2：Fixed map 与 scalar path

#### R2 实时记录（2026-08-06）

- 已开始 R2；复用 R1 的 fixed scalar reference 和
  `build-v01-neon-hot-tests` 增量 cache；
- 本批只把 exact floor-0.75x U8C3 `INTER_LINEAR` 接入全平台 scalar product
  path，不改变 public selector 的 direct-NEON 分支；0.5x、upscale、其他比例/
  类型继续保持现状。
- fixed scalar product path 已接入 `try_resize_fastpath_u8`，并记录
  `map=fixed_q16_q8;layout=flat_c3;interpolate=fixed8_scalar` route；只有
  U8C3、`INTER_LINEAR`、宽高均为 floor-0.75x 的 case 命中，其余路径未改；
- focused build 与 upstream exact contract 通过。R2/R3 的短暂中间状态中，
  ScalarOnly 已使用 fixed reference，而 Auto 仍使用旧 float NEON；既有
  byte-exact dispatch test 因此报告两个最大误差 `1` 的 mismatch。该结果与 R1
  diff inventory 一致，不放宽测试；R3 直接用相同 fixed map/rounding 替换目标
  Auto kernel 后再恢复全绿。R2 product scalar 工作完成，R2 关闭。

1. 实现 overflow-safe half-pixel aligned integer coordinate builder；
2. 建立 16-byte block map 与 y-row map；
3. 实现 scalar 两级 fixed lerp，覆盖主体、边界和 tail；
4. 对 exact 0.75x product predicate 使用 fixed scalar reference；
5. 保留现有 0.5x、upscale 和其他比例路径；
6. 运行 optimization-off，证明非 ARM header-only fallback 正常。

### R3：Flat-C3 direct NEON

#### R3 实时记录（2026-08-06）

- 已开始 R3；目标是让 exact floor-0.75x Auto/NeonOnly 与 R2 fixed scalar
  byte-exact，旧 0.5x 专用 NEON 和非目标 generic float NEON 均保留；
- 首版只实现安全的 16-output-byte block：top/bottom 各连续加载 32 bytes、
  TBL2 gather、vertical/horizontal integer lerp 和连续 store；不足 block、窄行
  与右边界使用同一 fixed scalar tail，不做 overread。
- 已在 product header 中完成 16-byte flat-C3 kernel：调用级 map 记录安全的
  32-byte source base、16 个 left index 与 x fraction；row loop 使用 top/bottom
  TBL2 gather、两级 `vraddhn` 等价 integer lerp 和 `vst1q_u8`，剩余 output
  bytes 逐 byte 调用同一 fixed scalar primitive；
- selector 顺序为既有 exact 0.5x -> 新 exact floor-0.75x fixed NEON -> 既有
  generic float NEON。focused Resize 14/14、upstream contract focused 3/3
  通过，eligible Auto 与 ScalarOnly byte-exact，当前 upstream target exact；
- 单轮 `stable + V01_NEON_HOT + auto` retention probe 为：continuous
  `0.064694 ms` 对 OpenCV `0.060227 ms`，relative `0.930957`，相对 R1
  baseline 提升 `2.4278x`；ROI `0.059675 ms` 对 OpenCV `0.055250 ms`，
  relative `0.925848`，提升 `2.6190x`；全部超过 `2.20x/0.90/0.065 ms`
  gate；
- probe 中 0.5x relative 为 `2.28–2.48`，1.5x 为 `0.973–0.987`，route
  均保持旧实现，未发现 selector 误命中。原始证据为 ignored
  `build-v01-neon-hot-compare/results/r3-retention-probe.csv{,.meta.json}`；R3
  correctness、dispatch 与 retention 条件齐备，R3 关闭。

1. 将 ignored prototype 的算法重新按 cvh 风格实现，不直接复制第三方源码；
2. 每次处理 16 个连续 output bytes；
3. 使用两组连续 source load、TBL2 gather、vertical/horizontal fixed lerp；
4. 接入 exact 0.75x、workload threshold 和真实 telemetry；
5. NEON 对 fixed scalar 必须 byte-exact；
6. 首轮 focused benchmark 若未达到 `2.20x` candidate retention，R3 回退，
   不进入 R4 泛化。

### R4：布局、边界和安全

#### R4 实时记录（2026-08-06）

- 已开始 R4；复用 optimization-on test cache 和现有
  `build-phase2-sanitize`，不创建新 build 目录；
- 已补齐 continuous、unaligned odd ROI、16-byte 主体加 scalar tail、窄行全
  scalar fallback、map index 安全和极端 coordinate arithmetic；覆盖 destination
  row byte width `24/36/48`，分别验证全 scalar、vector+tail 和整 vector 路径；
- fixed scalar 与 direct NEON 在上述 eligible product case 全部逐字节一致，focused
  Resize 当前 16/16 通过；没有为了 tail 增加不安全的 partial load；
- 复用 `build-phase2-sanitize`，确认编译/链接启用
  `-fsanitize=address,undefined -fno-omit-frame-pointer`。在
  `halt_on_error=1` 下 Resize targeted 16/16、Imgproc full 208/208 通过，未报告
  ASan/UBSan 错误；R4 安全性与布局门禁关闭。

1. 补齐 8-byte/标量 tail 或证明全 fixed-scalar tail 更快；
2. 覆盖 ROI/non-contiguous/unaligned row start；
3. 覆盖窄图、极小图、右边界和无法安全加载 32 bytes 的 fallback；
4. ASan/UBSan halt-on-error 运行 Resize targeted 和 Imgproc full；
5. 只有 exact 0.75x 全部关闭后，才评估 `[1/3, 1)` 一般下采样扩展。

### R5：Dispatch 与跨平台

#### R5 实时记录（2026-08-06）

- 已开始 R5；将先把 exact floor-0.75x product test 扩展到全部四种 forced
  dispatch mode，再复用现有 optimization-off、header/ODR、install 与 x86
  cross cache；若本机没有 Linux x86_64 runtime，将按计划保留为外部 gate。
- exact floor-0.75x product matrix 已扩展到 `Auto`、`NeonOnly`、
  `OpenCVUIOnly`、`ScalarOnly`：ARM64 的可向量化 case 中前两者命中
  flat-C3 NEON，后两者明确保持 fixed scalar；窄行在全部模式真实回退 scalar。
  四种模式与 fixed reference 逐字节一致，focused Resize 16/16 通过。
- 首次复用 `build-b7-optimization-off` 做全构建时，independent header smoke
  暴露新 fixed header 使用了未由自身依赖提供的 `CV_DbgAssert`；optimization-on
  的包含顺序此前遮蔽了问题。该断言对 map builder 已限定为 8-bit 的 fraction
  不构成运行时合同，已移除；保留此次失败记录并重跑同一完整配置。
- 修正后 `build-b7-optimization-off` 全量增量构建成功，CTest 18/18 通过，
  包含 independent headers、Imgproc ODR、include-only、pipeline、Resize
  dispatch、optimization-disabled smoke 以及完整 Core/Imgproc 单测；
- optimization-on Apple ARM64 Imgproc full 208/208 通过；链接当前
  `../opencv/build-slim` 的 OpenCV contract full 32/32 通过，新增 fixed target
  exact case 与既有 tolerance-1 Resize differential 均未放宽；
- `scripts/check_header_only_contract.sh` 通过：public boundary 检查通过，临时
  clean build 的 header/ODR/smoke CTest 12/12，通过安装包导出与外部
  `cvh::headers`/`cvh::highgui` consumer 编译运行。
- 复用 `build-v01-neon-hot-x86-cross` 成功交叉编译 independent header、
  Imgproc header/ODR、include-only、Resize dispatch smoke 与完整 Imgproc
  test binary；产物经 `file` 确认为 x86_64 Mach-O，证明非 ARM 编译不会实例化
  direct NEON；本机未安装 Rosetta、Docker/Podman 或 QEMU，无法执行 Linux
  x86_64 runtime，按计划保留为外部 gate，不将 compile 结果冒充 runtime。
- 除明确的外部 Linux runtime 外，R5 本机可执行项均已关闭；开始 R6 三轮
  stable、full product-auto 与文档收口。

1. 验证 ARM64 Auto/NeonOnly 命中，UIOnly/ScalarOnly 不进入 direct NEON；
2. optimization-on/off、header independent compile、ODR 和 install smoke；
3. OpenCV upstream differential full matrix；
4. Apple ARM64 full unit tests；
5. Linux x86_64 compile/runtime，证明新 NEON header 不改变 UI/scalar fallback；
6. 若本机无法执行 Linux runtime，明确保留为外部 gate，不伪造完成。

### R6：性能与收口

#### R6 实时记录（2026-08-06）

- 已开始 R6；复用 `build-v01-neon-hot-compare`，不创建新的 compare cache。
  当前源码尚未提交，stable/full 数据先作为 dirty-worktree candidate evidence；
  只有用户后续明确授权提交并从 clean revision 重跑后，才会归档并勾选
  clean product-auto dated report。
- candidate stable run 1/3 完成（auto/ui/scalar 各 70 行、checksum 通过）。目标
  Auto continuous 为 `0.059902 ms` 对 OpenCV `0.055779 ms`，relative
  `0.931172`；odd ROI 为 `0.059692 ms` 对 `0.055263 ms`，relative
  `0.925799`。运行中 telemetry 确认 fixed map + flat-C3 + TBL2 route；原始证据
  位于 ignored `build-v01-neon-hot-compare/results/r6-candidate-run1.csv{,.meta.json}`。
- candidate stable run 2/3 完成。目标 Auto continuous 为 `0.059856 ms` 对
  `0.055881 ms`，relative `0.933591`；odd ROI 为 `0.059629 ms` 对
  `0.055275 ms`，relative `0.926979`。原始证据位于 ignored
  `build-v01-neon-hot-compare/results/r6-candidate-run2.csv{,.meta.json}`。
- candidate stable run 3/3 完成；三轮 Auto 目标逐 case 中位数为：continuous
  CVH `0.059875 ms`、OpenCV `0.055796 ms`、relative `0.931872`，相对 R1
  baseline 提升 `2.6232x`；odd ROI CVH `0.059679 ms`、OpenCV
  `0.055275 ms`、relative `0.926412`，提升 `2.6188x`。三轮 CVH 极差分别仅
  `0.000046/0.000063 ms`；`2.20x`、`0.90` 和 `0.065 ms` 三项必达 gate
  均稳定通过。第三轮证据位于 ignored
  `build-v01-neon-hot-compare/results/r6-candidate-run3.csv{,.meta.json}`。
- dirty-worktree full product-auto 370 行完成，checksum/metadata 生成成功。相对
  2026-08-06 NEON-hot clean snapshot，排除两项目标 Resize 后 368 个 case 的
  CVH latency 几何平均改善 `3.22%`，其中 185 个 Imgproc 非目标 case 改善
  `1.12%`；以 normalized OpenCV/CVH relative 复核时分别改善 `0.69%` 和
  `0.10%`，均未出现超过 `1%` 的 aggregate 回退。full 目标 continuous 为
  `0.060154 ms`、relative `0.926855`、相对旧 full 提升 `2.8111x`；ROI 为
  `0.059679 ms`、relative `0.925922`、提升 `2.6151x`。ignored 证据为
  `build-v01-neon-hot-compare/results/r6-full-dirty.{csv,md}` 及 metadata；因
  `repo_git_dirty=true`，不将其复制成 date-named 发布快照。
- 三轮 stable 的 Resize 非目标逐 case 中位数对比中，最大回退为 F32C1
  `2.92%`，其余回退均小于 `0.4%` 或改善，满足 `5%` family gate；R0 已量得
  同架构 mapping `0.000731 ms`，相对当前 continuous 端到端中位数约
  `1.22%`，满足 `5%` mapping/allocation gate。
- `doc/cpu-optimization.md` 已同步 exact floor-0.75x 的 flat-C3 fixed route；旧
  H2 文档已追加 successor 实测和“不再需要 floor 例外”的状态，同时明确 clean
  report 前不把历史阶段勾选为最终完成。
- `scripts/check_docs.sh` 与 `git diff --check` 通过。`sync_opencv_intrin.py
  --check` 仅因生成的 `UPSTREAM.md` 期望写入本机绝对路径且 sibling OpenCV
  checkout 有既存 `.gitignore` dirty marker 而报告 metadata 不一致；11 个
  whitelist upstream 文件已逐字节核对全部匹配，未改写 immutable vendor
  provenance。该问题不由 Resize 变更引入，留作独立工具/metadata 收口。
- 正式 header CI 的机器可读 gate 原先固定 Imgproc GTest `203` 项；本轮新增
  5 项 Resize fixed-point/internal 测试后已同步 arm64/x86_64 期望为 `208`，
  防止只看普通 GTest 通过而遗漏 test-count drift。下一步运行完整
  `scripts/ci_headers_all.sh` 验证报告校验器。
- 完整 `scripts/ci_headers_all.sh` 已通过：安装包 contract 12/12、正式 CTest
  20/20、Core GTest `213/213`、Imgproc GTest `208/208`，machine-readable
  report checker 接受 arm64 `ui-on` 的新计数；header CI 门禁关闭。
- 最终自审按计划补强两处但不增加 test case 数：internal product test 逐宽执行
  `source_cols=2..65`，实际穿过前几个 16-byte block/tail/全 scalar 布局并要求
  Auto 对 fixed scalar byte-exact；OpenCV contract 在同一现有 test 内增加 4 组
  seed/尺寸以及 odd non-contiguous ROI，全部保持既有 tolerance `1`，冻结的
  480x640 case 仍另外要求 exact。补强后将重跑 focused、contract 和 sanitizer。
- 补强后的普通 Resize focused 16/16、OpenCV Resize contract 2/2、ASan/UBSan
  Resize focused 16/16 均通过；最终再次执行 Imgproc full 208/208、OpenCV
  contract full 32/32 和 `git diff --check`，全部通过。产品源码在三轮 stable/full
  测速后未再变化，仅测试覆盖继续增强，因此 candidate 性能证据仍对应当前
  product implementation。
- R6 当前只剩两项不能在本轮自行宣称完成的发布证据：仓库规则禁止在用户未
  明确要求时创建 commit，因此尚不能从 clean revision 生成 date-named report
  和更新 results index；Linux x86_64 runtime 仍需有对应执行环境的外部 gate。

1. 从 clean revision 连续运行三轮 `stable + V01_NEON_HOT`；
2. 逐 case 计算 current-to-candidate speedup、relative 和波动；
3. 运行 full product-auto，确认非目标 aggregate 回退不超过 `1%`；
4. 保存新的 date-named Markdown/CSV/metadata；
5. 更新 results index、CPU dispatch inventory、旧 H2 successor 链接和本文状态；
6. 只有正确性、性能、跨平台和 clean evidence 齐备后才关闭本阶段。

## 9. Canonical 命令

### 9.1 Targeted build/test

```bash
cmake -S . -B build-v01-resize-fixed-neon \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON \
  -DCVH_ENABLE_OPENCV_COMPARE=ON \
  -DCVH_ENABLE_OPTIMIZATION=ON \
  -DOpenCV_DIR=../opencv/build-slim

cmake --build build-v01-resize-fixed-neon --parallel 2

build-v01-resize-fixed-neon/cvh_test_imgproc \
  --gtest_filter='Resize*:ResizeDispatchInternalTest*'

ctest --test-dir build-v01-resize-fixed-neon --output-on-failure
```

实际 suite/filter 在 R1 用 `--gtest_list_tests` 核对；不得把空 filter 当成通过。

### 9.2 Focused stable benchmark

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
CVH_COMPARE_BUILD_DIR=build-v01-resize-fixed-compare \
CVH_COMPARE_THREADS=1 \
benchmark/opencv_compare/run_compare.sh \
  --profile stable \
  --impls auto,ui,scalar \
  --ops V01_NEON_HOT
```

连续运行三次，以逐 case median 的中位数关闭 gate。

### 9.3 Header-only 与 optimization-off

```bash
cmake -S . -B build-v01-resize-fixed-scalar \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_ENABLE_OPTIMIZATION=OFF
cmake --build build-v01-resize-fixed-scalar --parallel 2
ctest --test-dir build-v01-resize-fixed-scalar --output-on-failure

./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
./scripts/check_docs.sh
python3 scripts/sync_opencv_intrin.py --check
git diff --check
```

## 10. 提交和回退边界

建议提交顺序：

1. `test(imgproc): freeze fixed-point resize contract`；
2. `perf(imgproc): add fixed-point u8c3 resize reference`；
3. `perf(imgproc): add flat-c3 neon resize kernel`；
4. `test(imgproc): close resize layout and platform matrix`；
5. `docs(bench): publish fixed-point resize results`。

每个提交必须独立通过对应正确性 gate，且可以独立回退。以下情况立即停止或
回退 candidate：

- upstream 最大误差超过 `1`；
- scalar/NEON 对 eligible case 不再 byte-exact；
- 出现越界读写、未定义行为或非 ARM 编译失败；
- current-to-candidate stable 提升低于 `2.20x`；
- continuous 或 ROI relative 低于 `0.90`；
- 为命中 benchmark 而增加尺寸/seed 特判；
- 非目标 family 出现超过门槛的稳定回退。

## 11. 实时更新规则

执行期间本文是该专项的唯一状态 owner：

- 开始批次前，将状态从“待执行”改为“进行中”；
- 每完成 correctness、dispatch、benchmark 或安全 gate，立即追加 revision、
  命令、结果与未决问题；
- 失败 candidate 记录原始结果和回退原因，不把废弃实现留在产品路径；
- ignored probe 只能记录诊断，不代替 clean canonical evidence；
- dated benchmark 报告提交后不可覆盖，更正必须使用新文件名；
- R6 完成后同步旧 H2 文档，但历史实验记录不删除。

## 12. 完成定义

- [x] 确认 OpenCV 目标 case 命中 KleidiCV HAL，且比较为单线程；
- [x] 拆分 mapping、gather、float arithmetic、tail 成本；
- [x] 诊断原型证明 flat-C3 fixed-point NEON 可达到 upstream 附近；
- [x] 定点 scalar reference 的坐标、舍入、边界和 tie 合同冻结；
- [x] exact 0.75x fixed scalar 与 direct NEON 对已覆盖 eligible case byte-exact；
- [x] OpenCV full differential 最大误差不超过既有 `1`；
- [x] 冻结 ARM/KleidiCV 目标 case与 upstream byte-exact；
- [x] continuous、ROI、non-contiguous、unaligned、odd/tail/窄图全部通过；
- [x] Auto/NeonOnly/UIOnly/ScalarOnly dispatch 与 route 可信；
- [x] optimization-on/off、ASan/UBSan、header、ODR、install consumer 通过；
- [x] Apple ARM64 full tests 通过；
- [ ] Linux x86_64 compile 已通过；runtime 因本机无执行环境保留为外部门禁；
- [x] continuous 和 ROI current-to-candidate speedup 均 `>= 2.20x`；
- [x] continuous 和 ROI relative 均 `>= 0.90`；
- [x] 480x640 -> 360x480 public-call latency `<= 0.065 ms`；
- [x] full Imgproc/full compare 非目标回退不超过 `1%`；
- [ ] 新 clean product-auto Markdown/CSV/metadata 已归档；
- [ ] CPU dispatch inventory、旧 H2 和本文状态已同步；results index 等待 clean
      date-named report。
