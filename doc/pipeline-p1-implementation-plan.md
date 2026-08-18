# cvh Pipeline P1 模型输入融合实施计划

状态：P1 已完成；P1 v1 已通过 Supported 审计。

更新时间：2026-08-10

基线提交：`3857147 feat: implement pipeline P0 runtime`

上位合同：[pipeline-module-design.md](pipeline-module-design.md)。上位文档负责产品定位、
公开 API、顺序语义和长期边界；本文只负责 P1 的实施批次、支持矩阵、验证门禁和
实时证据。两者冲突时以上位合同为准。

## 0. 实时状态

P1 的状态所有者是本表。批次开始、完成、回退或阻塞时必须先更新状态和证据；代码
存在、能够编译或单次 benchmark 变快，都不能单独作为完成证据。

| 批次 | 范围 | 状态 | 完成证据 |
| --- | --- | --- | --- |
| P1.0 | 冻结范围、基线、数值合同和实施门禁 | 已完成 | 2026-08-10：P1 计划建立，P0 基线固定为 `3857147` |
| P1.1 | 拆分内部 planner/executor，建立 execution-group IR | 已完成 | 2026-08-10：新增 `detail/ir.hpp`、`planner.hpp`、`scalar_stage_executor.hpp`；workspace、执行和 `explain()` 改由显式 execution group 驱动；Release 25/25、优化关闭 4/4、ASan/UBSan 4/4、header-only 15/15 通过 |
| P1.2 | borrowed Image/Tensor view 的 prepared 执行合同 | 已完成 | 2026-08-10：Mat/view 差分覆盖 BGR/RGB、NCHW/NHWC、padding、unaligned U8、tail 和并发；view allocator hook 零分配；Release 25/25、优化关闭 4/4、ASan/UBSan 4/4、header-only 15/15 通过 |
| P1.3 | packed BGR/RGB → F32 tensor scalar fused path | 已完成 | 2026-08-10：1 group/0 intermediate direct-store executor 通过固定矩阵、64 条 fused/staged 随机链、Release 25/25、优化关闭 4/4、ASan/UBSan 4/4、header-only 15/15；stable baseline 五个 case 为 1.23×–2.64× staged |
| P1.4 | `modelInput` Recipe v1、融合证明和 fallback 可观察性 | 已完成 | 2026-08-10：严格 packed-f32 v1、普通 Pipeline 等价、id/version/fingerprint 和明确失败通过；Release 25/25、优化关闭 4/4、ASan/UBSan 4/4、header-only/install 15/15 |
| P1.5 | ARM NEON 候选、dispatch 和目标设备门禁 | 已完成 | 2026-08-10：Nearest 横向 2× NCHW 窄 predicate 通过 forced scalar/NEON/Auto、tail、ROI、zero-allocation、sanitizer 和 x86_64 compile；Apple M5 stable 为 NEON 相对 scalar fused 2.97× |
| P1.6 | NV12/NV21 与 letterbox 融合扩展 | 已完成 | 2026-08-10：YUV scalar/Recipe/192 组数值矩阵、letterbox transform/packed/YUV fused/Recipe/zero-allocation 通过；letterbox NEON 为 scalar 的 2.38×；Release 25/25、opt-off 4/4、ASan/UBSan 4/4、header-only 15/15、x86_64 compile 和文档门禁通过 |
| P1.7 | UINT8/INT8 quantize 融合扩展 | 已完成 | 2026-08-10：per-tensor U8/S8 数值、packed/YUV Recipe、borrowed/零分配已通过；首版 S8 回退已移除，stable 大图 U8/S8 均为 staged 的 1.33×；Release、opt-off、sanitizer、header/install、x86_64 和文档门禁通过 |
| P1.8 | Supported 审计、安装消费和稳定性能报告 | 已完成 | 2026-08-10：Supported 矩阵、12 个 Recipe id、测试 inventory 和产品依赖完成审计；最终 ARM stable report、Release 25/25、opt-off 4/4、sanitizer 4/4、x86_64 compile、install 15/15 与文档门禁通过 |

当前结论：本文 2.2/2.3 和上位设计 16.1 限定的 packed/YUV F32/U8/S8、
letterbox/transform、Recipe 和窄 NEON predicate 已进入 Supported。quantize NEON、
per-channel、动态 shape 和 adapter 不在缺少需求与证据时扩面，继续保持 Proposed。

## 1. P1 要解决的问题

P0 已经证明以下语义骨架可行：

- Builder 严格记录用户书写顺序；
- 合法链能够使用 staged scalar executor 正确执行；
- `prepare()` 可以预生成 resize 表和 workspace；
- prepared `run()` 可以做到零堆分配；
- `PipelineInfo`、`explain()` 和硬要求已经具备公开形态。

P0 还没有形成 cvh 的性能差异化：

- 每个语义 stage 都是独立 execution group；
- 多 stage 链仍会物化完整中间图；
- `PipelineRoute` 固定为 Scalar；
- `ImageView/TensorView` 只有描述 helper，不能参与 `run()`；
- 没有 `modelInput` Recipe；
- 没有 Pipeline 专属可复现 benchmark。

P1 的交付结果是：常见模型输入链可以在保持 P0 顺序和数值语义的前提下，编译为
无完整中间图的确定执行计划，并对 scalar fused、ARM NEON、实际 fallback、workspace
和性能给出可检查的证据。

## 2. 首个产品闭环

### 2.1 Recipe v1 的规范链

第一个可进入 Supported 审计的链固定为：

~~~text
Packed BGR8/RGB8 Image
  -> optional Color(RGB/BGR)
  -> Resize(Linear/Nearest)
  -> Normalize(F32)
  -> Layout/Store(NCHW/NHWC, batch=1)
  -> F32 Tensor
~~~

对应 fluent API：

~~~cpp
const cvh::PipelinePlan plan =
    cvh::pipe(input_desc, output_desc)
        .color(cvh::Color::RGB)
        .resize(640, 640, cvh::Interpolation::Linear)
        .normalize(mean, stddev)
        .layout(cvh::Layout::NCHW)
        .requireNoFullFrameIntermediate()
        .requireSingleExecutionGroup()
        .prepare();
~~~

如果输入已经是目标颜色，可以没有 `color` stage。其他顺序仍然由通用 Pipeline
接受或拒绝，但只有上述模式进入 Recipe v1 的融合 predicate。

### 2.2 第一波支持矩阵

| 维度 | P1 packed-f32 v1 |
| --- | --- |
| 输入内存 | `cvh::Mat` 或单 plane borrowed image view |
| 输入 dtype | U8 |
| 输入颜色 | BGR、RGB |
| 输入布局 | packed HWC，允许 row padding |
| 输入尺寸 | prepare 时固定，正整数 |
| 几何 | Nearest、Linear resize |
| normalize | 1 个或 3 个 mean/stddev，stddev 非零 |
| 输出 dtype | F32 |
| 输出布局 | NCHW、NHWC |
| batch | 1 |
| 输出内存 | 连续 `cvh::Mat` 或连续 borrowed tensor view |
| 运行模式 | one-shot、prepared |
| backend | scalar fused 必选；通过 predicate 的 ARM NEON |

P1 第一波不承诺 padded/strided tensor 输出。它可以在后续 predicate 中加入，但不能
让调用者误以为任意 tensor stride 已经支持。

### 2.3 P1 扩展矩阵

packed-f32 v1 关闭后再依次扩展：

| 扩展 | 输入/输出 | 必须新增的合同 |
| --- | --- | --- |
| NV12/NV21 | 双 plane U8 → RGB/BGR/F32 tensor | matrix、range、chroma location、plane stride/容量 |
| letterbox | packed/YUV → 固定模型尺寸 | scale、rounding、pad 分配、pad value、坐标变换元数据 |
| quantize | F32/归一化值 → U8/S8 tensor | scale、zero point、per-tensor/per-channel、round/saturate |

每个扩展必须有自己的 fusion predicate、reference、错误矩阵和 benchmark 行；不能把
“model input”作为一个宽泛条件直接选择同一个内核。

## 3. 明确不进入 P1 的内容

- 摄像头打开、配置、采集、buffer dequeue/enqueue；
- OpenCV、V4L2、ROS 2、推理 runtime 或厂商 SDK 产品依赖；
- DMA-BUF、GPU surface、NPU 私有内存；
- 动态 shape、batch 大于 1、任意 tensor stride；
- crop/rectify/remap、模型特定 decode、NMS；
- 多输入、多输出和任意 DAG；
- x86 专用 Pipeline SIMD；
- 为任意 operation 排列生成专用融合内核。

合法但不匹配 P1 fusion predicate 的链继续使用 P0 staged fallback。P1 不能以性能
优化为理由缩小 P0 已经支持的合法语义面。

## 4. 数值合同

### 4.1 顺序不可重排

P1 识别的是连续的有序 stage 模式，不是可交换操作集合。例如：

~~~text
Resize -> Normalize
Normalize -> Resize
~~~

两者仍是不同语义。只有第一条进入 packed-f32 v1；第二条保持 staged fallback。

### 4.2 packed resize 后的 U8 舍入

P0 的公开顺序是先得到 U8 resize 结果，再执行 F32 normalize。P1 fused executor
即使不物化完整 U8 图，也必须逐输出元素保留等价边界：

~~~text
sampled_f32
  -> P0-compatible saturate/round to U8
  -> channel mapping
  -> (u8 - mean[c]) / stddev[c]
  -> final tensor store
~~~

禁止为了少一条转换而把它改成“浮点 resize 后直接 normalize”。如果未来需要模型
生态常见的浮点 resize 语义，必须使用不同 operation 或 Recipe 版本。

### 4.3 resize

- 坐标映射、边界、Nearest/Linear 行为复用 P0 和 `cvh::resize` 的冻结合同；
- resize 坐标表在 `prepare()` 生成并由不可变 Plan 持有；
- `run()` 不重新计算整套表，不分配容器；
- U8 resize 结果必须精确匹配 P0 reference；
- odd size、1 像素维度、upscale/downscale 和尾部都进入测试矩阵。

### 4.4 color 与 normalize

- BGR/RGB 通道含义来自 descriptor/helper，不从指针或模型名猜测；
- `color` 位于 resize 前时，fused kernel 可以在取样后选择通道，但结果必须等价于
  先完成 color stage 再 resize；
- normalize 公式保持 `(x - mean[c]) / stddev[c]`；
- 参数顺序对应 color stage 的逻辑输出通道；
- F32 最终输出的初始差分门槛为绝对误差 `<= 1e-5`，U8 边界值必须精确；
- NaN/Inf 和零 stddev 继续沿用 P0 的准备期错误合同。

### 4.5 NV12/NV21

P1.6 的第一条 YUV 数值合同冻结如下：

- 只接受偶数宽高、U8、双 plane NV12/NV21；Y plane 为 `width × height`，交错
  UV/VU plane 的逻辑尺寸为 `width × height/2`，两者分别携带 row stride 和容量；
- `Limited` 使用 `Y' = (Y - 16) × 255 / 219`、
  `Cb'/Cr' = (sample - 128) × 255 / 224`；`Full` 使用原始 Y 和
  `sample - 128`；超出名义 limited range 的输入不预裁剪，在矩阵计算后统一
  saturate/round 到 U8；
- BT.601 使用 `(Rcr, Gcb, Gcr, Bcb) = (1.402, -0.344136, -0.714136,
  1.772)`；BT.709 使用 `(1.5748, -0.187324, -0.468124, 1.8556)`；
  BT.2020 使用 `(1.4746, -0.164553, -0.571353, 1.8814)`；
- `Left` 的 chroma 横坐标为 `x / 2`，`Center` 为 `(x - 0.5) / 2`；纵坐标统一为
  `(y - 0.5) / 2`。坐标先 clamp 到 UV 边界，再做双线性 chroma 重建；
- 每个被 resize 采样的源像素先完成 YUV → RGB 和 U8 saturate/round，再执行 P0
  Nearest/Linear resize 的 U8 舍入边界，最后 channel mapping、normalize 和 tensor
  store；不得改成连续浮点域的一次复合插值；
- NV12/NV21 只改变 UV/VU 存储顺序，不改变颜色语义；`ColorSpec` 是 descriptor、
  view 校验和 Recipe fingerprint 的组成部分，不按尺寸或平台猜测。

当前 scalar predicate 精确匹配
`color(RGB/BGR) → resize/letterbox(Nearest/Linear) → normalize(F32) → layout(NCHW/NHWC)`，
编译为 1 execution group、0 完整中间图、0 workspace。其他 YUV 链在
`prepare()` 明确拒绝，避免进入不支持 multi-plane 的 staged executor。

### 4.6 letterbox

P1.6 冻结合同如下：

- `scale = min(target_width / source_width, target_height / source_height)`；
- 正数 resized 尺寸使用 half-up：
  `floor(source_dimension × scale + 0.5)`，并 clamp 到 `[1, target_dimension]`；
- `left = horizontal_padding / 2`、`top = vertical_padding / 2` 使用整数向下取整，
  奇数余量固定给 right/bottom；
- 单 pad value 广播到所有逻辑输出通道，三值 pad 按 letterbox stage 输入图像的逻辑
  channel 顺序解释；U8 使用 cvh 的 saturate/round，F32 保留给定值；
- content 区域复用 P0 Nearest/Linear 坐标表与 U8 舍入合同；canonical model-input
  chain 直接把 content/pad normalize 后写 tensor，不物化完整 padded 中间图；
- `PipelineTransform` 记录原尺寸、目标尺寸、nominal `scale`、实际 `scale_x/scale_y`、
  resized 尺寸和四边 padding。坐标映射采用连续图像边界坐标及实际 x/y scale，因此
  `(0,0)` 与 `(source_width,source_height)` 精确映射到 content 外框并可往返；
- `isPadding()` 将 content 视为左/上闭、右/下开区间。

独立 operation 测试冻结 `.5` rounding、奇数 padding、单/三通道 pad、正反边界映射
和 padding 判断；invalid pad count、NaN 和同一 Plan 多次 letterbox 在 `prepare()` 失败。

### 4.7 quantize

P1.7 第一条合同只覆盖 per-tensor U8/S8，公式冻结为：

~~~text
q = saturate(round(real_value / scale[c]) + zero_point[c])
~~~

- v1 只有一个正且有限的 `scale` 和一个整数 `zero_point`；U8 zero point 必须在
  `[0,255]`，S8 必须在 `[-128,127]`；per-channel 保持 Proposed；
- `round` 使用 C++ `std::round` 的 half-away-from-zero，发生在加 zero point 之前；
- 加 zero point 后，U8 saturate 到 `[0,255]`，S8 saturate 到 `[-128,127]`；
- `real_value=NaN` 映射到 zero point，`+Inf/-Inf` 分别映射到目标最大/最小值；有限但
  超出整数转换范围的值在转换前按符号 saturate，禁止依赖未定义的浮点转整数行为；
- quantize operation 接受 F32 Image，输出保持 shape/color/layout 语义但 dtype 改为
  U8/S8；canonical model-input 顺序是
  `... → normalize(F32) → quantize(U8/S8) → layout/store`；
- scalar staged reference、direct-store executor 和 Recipe 共用同一 helper/预计算合同；
  测试侧另建独立 double reference，覆盖正负 `.5` ties、zero point 和两端 saturation。

P1.7 暂不承诺 per-channel 或任意预量化 F32 tensor 输入。它们只有在明确模型需求和
性能证据存在时才新增独立 predicate/Recipe 版本。

## 5. Execution Group 和 planner 设计

### 5.1 内部表示

P1 不再把“一个 stage”直接等同于“一个 execution group”。Plan 内部增加紧凑的
执行组描述：

~~~text
PipelineExecutionGroup
  semantic_begin / semantic_end
  execution_class
  input/output descriptor
  immutable prepared parameters
  scalar entry
  optional ISA entries + predicates
  workspace slice
~~~

`PipelinePlanImpl` 继续保存完整 ordered stages，用于语义、错误位置和 `explain()`；
执行时遍历 prepared execution groups，不在逐帧路径重新做 fusion matching。

### 5.2 第一条 fusion rule

首条规则只识别：

~~~text
[Color]? -> Resize -> Normalize -> Layout(NCHW/NHWC)
~~~

并同时检查：

- 输入/输出满足 packed-f32 v1 矩阵；
- 中间 dtype 和通道数可由 P0 traits 推导；
- resize interpolation 是 Nearest 或 Linear；
- normalize 参数个数合法；
- 没有未知别名、动态 shape 或不支持 stride；
- scalar fused entry 对该 predicate 完整；
- 请求的硬要求能够被真实计划满足。

规则不匹配时生成 P0 staged groups。规则匹配但某个 ISA predicate 不满足时，仍可
选择 scalar fused group；这不是完整中间图 fallback。

### 5.3 可验证的内存属性

packed-f32 v1 成功融合后必须满足：

| 属性 | 要求 |
| --- | --- |
| `semantic_stage_count` | 与用户链一致 |
| `execution_group_count` | 1 |
| `full_frame_intermediates` | 0 |
| `allocations_per_run` | 0 |
| final tensor | 直接写调用者输出 |
| workspace | 只允许行/tile scratch 或其他已声明临时区 |

“无完整中间图”不等于零 workspace，也不等于每个源 byte 只发生一次 CPU load。
Linear resize 可以重复采样，NEON 候选也可以使用固定大小行/tile scratch；这些都要
在 `info()` 中诚实报告。

### 5.4 代码布局目标

P1.1 优先把当前大型实现拆到 `include/cvh/pipeline/detail/`，建议职责如下：

~~~text
detail/
  descriptor.hpp
  ir.hpp
  planner.hpp
  fusion_rules.hpp
  workspace_planner.hpp
  scalar_stage_executor.hpp
  scalar_model_input_fused.hpp
  neon_model_input_fused.hpp
  run_validation.hpp
~~~

这次拆分必须是行为保持型重构：P0 单元、随机链、ODR、安装消费和零分配测试先保持
全绿，再引入第一条 fusion rule。公开 forwarding header 和 header-only/ODR 合同
不得改变。

## 6. Borrowed view 执行合同

P1.2 增加与 `cvh::Mat` 同构的 prepared 入口：

~~~cpp
cvh::PipelineStatus PipelinePlan::tryRun(
    cvh::ConstImageView input,
    cvh::TensorView output,
    cvh::PipelineWorkspaceView workspace,
    cvh::PipelineRunInfo* run_info = nullptr) const;
~~~

需要补充 `bgr(...)` 等与 descriptor 一致的 helper。P1 packed 输入校验至少包括：

- data 非空；
- width/height、dtype、color、plane count 与 Plan 一致；
- `row_stride >= width * channels * element_size`；
- `size_bytes` 覆盖最后一行可访问范围；
- 输出 tensor descriptor 完全匹配，连续容量足够；
- input、output、workspace 可验证范围互不重叠；
- workspace 来自同一个 Plan 且满足大小、对齐和内部 header 数量；
- 所有失败通过 `tryRun()` 返回固定类别，不在失败路径执行部分写入。

prepared view 路径不能创建临时 `cvh::Mat` 所有者或在运行期分配维度数组。可以在
setup 阶段创建无所有权内部 view/header，但其生命周期必须由 Plan/Workspace 明确
持有。

### 6.1 P1.2 已冻结的错误分类

| 条件 | `PipelineStatusCode` |
| --- | --- |
| view 自身 descriptor/plane count 非法、地址或 stride 不满足 dtype 对齐 | `InvalidDescriptor` |
| dtype、颜色、pixel format、tensor layout 与 Plan 不同 | `TypeMismatch` |
| image 宽高、tensor rank/shape 与 Plan 不同 | `ShapeMismatch` |
| data 为空、row stride 不足、可访问容量少于 1 byte | `BufferTooSmall` |
| Workspace 不属于同一个 Plan 或 endpoint header 缺失 | `WorkspaceMismatch` |
| input/output/workspace 的声明 buffer 范围重叠 | `AliasingNotSupported` |
| Plan 不是单 plane Image 输入到连续 Tensor 输出 | `InvalidDescriptor` 或明确的 `Unsupported` |

所有 view 校验先于 endpoint header 重绑定和 executor 调用；因此上述失败不会产生部分
输出。非拥有 Mat header 在 `PipelineWorkspace` 构造时建立，逐次 `tryRun()` 只更新 data
和 image row step，不分配 owner、shape 数组或 heap block。同一个 Plan 可以并发使用，
但每个并发调用必须使用独立 Workspace、输出和 `PipelineRunInfo`。

## 7. Scalar fused executor

P1.3 的 scalar fused path 是所有 ISA 的独立 oracle，也是非 ARM 的产品 fallback。

建议执行顺序：

1. 遍历目标 y/x；
2. 从 Plan 的坐标/权重表取得 source 索引；
3. 按 P0 规则完成 Nearest/Linear U8 采样和舍入；
4. 按逻辑 color stage 映射通道；
5. 执行逐通道 normalize；
6. 直接计算 NCHW/NHWC 目标 offset 并写出。

executor 不读取 Builder，不做字符串查找，不创建 vector，不调用可能分配输出的公开
基础算子。为避免测试 oracle 与产品实现共享错误，差分 reference 继续由 P0 staged
路径或独立基础算子组合产生。

## 8. ARM NEON 和 dispatch

### 8.1 候选策略

NEON 不要求一个内核覆盖全部矩阵。P1.5 为常见 predicate 评估以下候选：

- output-pixel register fusion；
- 预计算 x/y 表的 fixed-point/tiled fusion；
- 允许有限 row/tile scratch、但不生成完整 resize 图的组合内核；
- Nearest 与 Linear 分开实现；
- NCHW 与 NHWC final store 分开评估。

每个候选先用 Release microprobe 确认热点和 ISA，再进入产品代码。没有稳定收益、
破坏数值合同或扩大代码体积但价值不足的候选必须记录结果并从产品路径移除。

### 8.2 dispatch 合同

- 编译期非 ARM 平台必须保留 scalar fused/staged fallback；
- runtime capability 复用 cvh 现有 CPU dispatch 设施；
- forced scalar、forced NEON、Auto 和优化关闭都必须可测试；
- `candidate_route` 记录 prepare 后的首选候选；
- `actual_route` 记录本次真实 executor；
- `observed_isa` 只能由执行器实际写入，不能由平台或候选推断；
- 指针、stride 或尺寸导致 ISA predicate 失败时，`used_fallback` 和原因必须可查询；
- Plan 保持不可变，实际运行信息只能写调用者提供的 `PipelineRunInfo`。

如果 scalar fused 已满足单执行组，而 NEON predicate 不满足，允许回退到 scalar
fused；如果连 scalar fusion predicate 都不满足，普通 Pipeline 才回退到 staged。
Recipe 是否允许后者由其硬合同决定。

## 9. `modelInput` Recipe v1

### 9.1 公开形态

建议新增：

~~~text
include/cvh/recipes/model_input.h
~~~

最小公开配置：

~~~cpp
cvh::ModelInputRecipe spec;
spec.input = input_desc;
spec.output = output_desc;
spec.color = cvh::Color::RGB;
spec.interpolation = cvh::Interpolation::Linear;
spec.mean = mean;
spec.stddev = stddev;

const cvh::PipelinePlan plan =
    cvh::recipes::modelInput(spec)
        .requireNoFullFrameIntermediate()
        .requireSingleExecutionGroup()
        .prepare();
~~~

Recipe 只生成普通 ordered IR 和硬要求，不复制 planner/executor。普通 Pipeline 的同一
链和 Recipe 必须得到相同语义结果。

### 9.2 版本和 fingerprint

Recipe info 至少记录：

- 稳定 recipe id，例如 `cvh.model_input.packed_f32`；
- recipe contract version；
- input/output descriptor；
- color、interpolation、normalize 参数；
- 后续 YUV 的完整 `ColorSpec`；
- quantize 配置；
- execution-group、workspace 和 backend predicate 版本。

fingerprint 用于日志和部署审计，不用来决定语义。它必须在 `prepare()` 生成，逐帧
运行不做字符串拼接或哈希容器操作。

### 9.3 Recipe 失败原则

Recipe predicate 不满足时 `prepare()` 明确失败，不静默变成多组 staged 计划。需要
通用正确 fallback 的用户使用 `cvh::pipe(...)`；需要确定内存/性能属性的用户使用
Recipe。这是两种产品承诺，不是两套执行引擎。

## 10. 分批实施和完成条件

### P1.1：execution-group IR 和行为保持重构

工作：

- 拆分 internal headers；
- stages 与 execution groups 分离；
- 让 P0 staged planner 显式生成一 stage 一 group；
- executor 改为运行 prepared groups；
- `explain()` 同时展示语义 stage 和执行组。

完成条件：

- P0 结果、异常、ROI 和随机链无变化；
- prepared run 继续零分配；
- public-header、ODR、install consumer、优化开/关全绿；
- 无融合时 `execution_group_count` 与 P0 一致。

### P1.2：borrowed view runtime

工作：

- 实现 packed const image 和连续 tensor output 入口；
- 增加 `bgr/rgb/nchw/nhwc` 容量和 stride 校验；
- 建立 buffer-too-small、stride、descriptor、alias 测试。

完成条件：

- Mat 与 view 路径对相同数据结果一致；
- 非连续输入 row stride、unaligned 地址和尾部通过；
- prepared view run 由 allocator hook 证明零分配；
- 错误路径不改写输出 canary。

### P1.3：packed scalar fusion

工作：

- 加入第一条 fusion rule；
- 实现 Nearest/Linear × NCHW/NHWC scalar fused；
- 将 P0 staged 作为差分 oracle；
- 更新 info/explain/hard requirements。

完成条件：

- 支持矩阵全部满足一 execution group、零完整中间图；
- U8 边界精确、F32 在冻结 tolerance 内；
- 64 条随机链扩展为包含“可融合/不可融合”计划差分；
- 不匹配规则的合法链仍可靠 staged fallback。

### P1.4：Recipe v1 和可观察性

工作：

- 实现 `ModelInputRecipe` 和 `recipes::modelInput`；
- 增加 recipe id/version/fingerprint；
- 完成 candidate/actual/observed/fallback 信息；
- 添加普通 Pipeline 与 Recipe 等价测试。

完成条件：

- Recipe 支持矩阵外稳定失败；
- Recipe 内不允许多组 staged 静默成功；
- 日志信息与 forced dispatch 实际执行一致；
- 安装后的外部消费者能够构建一次性和 prepared 示例。

### P1.5：ARM NEON

工作：

- 建立 scalar fused、NEON candidate、P0 staged 的同机 benchmark；
- 实现通过 microprobe 的窄 predicate；
- 补齐 forced route、tails、unaligned、ROI、优化关闭和非 ARM 编译。

完成条件：

- 正确性差分和 sanitizer 通过；
- observed ISA 证明实际执行 NEON；
- 目标 ARM 设备稳定多轮性能达到冻结门槛；
- 失败候选有数据并从产品路径移除。

### P1.6：NV12/NV21 和 letterbox

工作：

- 开放双 plane borrowed view run；
- 建立 matrix/range/chroma reference；
- 实现 letterbox 和 `PipelineTransform`；
- 分别建立 scalar fused 和有价值的 NEON predicate。

完成条件：

- plane stride、奇偶尺寸、UV 边界和 ColorSpec 矩阵通过；
- transform 能正确往返映射边界点和 padding 区域；
- 无摄像头/SDK/OpenCV 产品依赖；
- Recipe 的 YUV predicate 和 fingerprint 完整。

### P1.7：U8/S8 quantize

工作：

- 新增 operation traits 和 output descriptor 推导；
- scalar reference、fused store 和可选 NEON；
- per-tensor 优先，per-channel 只有需求和性能证据后加入。

完成条件：

- rounding、saturate、zero point 边界精确；
- U8/S8 tensor 容量、布局和 alias 测试通过；
- Recipe 明确区分 F32、U8、S8 输出合同。

### P1.8：Supported 审计

工作：

- 完整 Release、ASan/UBSan、优化开/关和非目标平台门禁；
- header、ODR、安装消费；
- 目标 ARM 稳定性能报告；
- 审计 README、总设计、Recipe 表和实际代码。

完成条件：

- 每个宣布 Supported 的 predicate 都有正确性、内存、ISA 和性能证据；
- 未关闭的 NV12/letterbox/quantize 子能力继续明确标记 Proposed；
- 总设计和本计划状态同步；
- 不降低已有 tolerance、测试数量或性能门槛。

## 11. 测试矩阵

### 11.1 正确性维度

packed-f32 v1 至少组合：

- BGR/RGB 输入和可选通道交换；
- Nearest/Linear；
- NCHW/NHWC；
- 单值/三通道 normalize；
- 1x1、极小图、奇偶尺寸、upscale、downscale；
- 224、320、640 等模型输出尺寸；
- 1280x720、1920x1080 等相机输入尺寸；
- continuous、ROI、row padding、unaligned、SIMD tail；
- Mat/view、one-shot/prepared、scalar fused/NEON/staged oracle；
- 同一 Plan 多线程共享且使用独立 workspace/output/run info。

### 11.2 计划属性

每个 case 除结果外还检查：

- semantic stages 和 execution groups；
- full-frame intermediate 数量；
- workspace byte/alignment；
- allocations per run；
- candidate/actual/observed route；
- fallback 是否发生及原因；
- 硬要求成功或准确失败的 stage。

### 11.3 内存安全

- allocator hook 覆盖普通、aligned、array 和 nothrow new；
- 输入、输出、workspace 前后 canary；
- 容量少 1 byte、stride 少 1 byte、错误 plane count；
- input/output、input/workspace、output/workspace 重叠；
- 错误 Plan 的 workspace；
- ASan/UBSan；必要时增加 guard-page 诊断程序，但不创建一次性产品 target。

### 11.4 平台和编译

- `CVH_ENABLE_OPTIMIZATION=ON/OFF`；
- ARM Auto/forced scalar/forced NEON；
- 非 ARM 构建不包含不可编译的 NEON 代码；
- public header 独立编译、multi-TU ODR、安装后消费；
- header-only 产品 target 不获得 OpenCV 二进制依赖。

## 12. Benchmark 计划

### 12.1 新目标

复用现有 benchmark common、CSV、metadata 和报告生成器，新增长期维护的
`cvh_benchmark_pipeline_header`。不要使用临时可执行文件关闭性能门禁。

基准至少区分：

~~~text
algorithm_path -> dispatch_path -> observed_isa
staged_p0      -> scalar        -> scalar
fused_p1       -> scalar        -> scalar
fused_p1       -> auto          -> scalar/neon
opencv_chain   -> upstream      -> reported backend
~~~

OpenCV 只存在于 compare benchmark，正常 Pipeline target 保持独立 header-only。

### 12.2 首批 case

- 1280x720 BGR8 → 640x640 RGB F32 NCHW Linear；
- 1920x1080 BGR8 → 640x640 RGB F32 NCHW Linear；
- 640x480 RGB8 → 224x224 RGB F32 NCHW Linear；
- 同尺寸 NHWC；
- 对应 Nearest 行；
- P1.6 后增加 NV12/NV21 与 letterbox；
- P1.7 后增加 U8/S8 output。

报告同时包含：

- ns/frame、FPS、相对 staged/OpenCV 比值；
- input/output shape、dtype、layout、interpolation；
- execution groups、完整中间图数量、workspace；
- candidate/actual/observed route；
- 线程数、CPU、编译器、build type、源码 revision；
- 估算内存流量，并明确它不是物理 load 次数。

### 12.3 性能门槛冻结

P1.3 首个正确 scalar fused 完成后，先生成同机稳定 baseline，再冻结数值门槛。不得
在没有测量时凭经验写“快 2x”，也不得因候选失败降低门槛。

Recipe 进入 Supported 前必须同时满足：

- 比 P0 staged 有稳定、可复现的端到端收益；
- ARM NEON predicate 在目标设备达到记录的最低收益；
- 没有通过扩大 tolerance、减少 case、关闭 checksum 获得收益；
- workspace 和完整中间图属性满足 Recipe 声明。

dated 报告、CSV 和 metadata 同步新增且不可覆盖。失败候选记录在本计划的决策日志，
然后从产品路径移除。

## 13. 建议复用的构建配置

开始各批次前先检查现有 cache；只有配置身份不兼容时才建立：

~~~text
build-pipeline-p1-release
build-pipeline-p1-opt-off
build-pipeline-p1-sanitize
build-pipeline-p1-opencv-compare
build-pipeline-p1-baseline
build-pipeline-p1-candidate
~~~

baseline/candidate 需要同时保留时使用独立稳定目录，不删除兼容 cache 反复 clean。
默认迭代只构建 Pipeline unit、header/ODR、zero-allocation 和 benchmark 目标，关闭批次
时再运行完整 header-only gate。

## 14. 决策和失败日志

| 日期 | 批次 | 候选/决策 | 结果 | 证据 |
| --- | --- | --- | --- | --- |
| 2026-08-10 | P1.0 | packed-f32 v1 先于 NV12/letterbox/quantize | 接受 | 控制首个 fusion predicate 和测试矩阵规模 |
| 2026-08-10 | P1.0 | P0 staged 继续作为任意合法链 fallback | 接受 | 保持“用户定义顺序，cvh 保证语义” |
| 2026-08-10 | P1.0 | 不宣传“每个 byte 只读一次” | 接受 | 使用 execution group/intermediate/direct-store 可验证属性 |
| 2026-08-10 | P1.1 | execution group 成为 workspace、执行和说明的内部事实来源 | 接受 | 空链显式编译为一个 copy group；P0 每个语义 stage 暂保持一个 staged group，新增说明测试覆盖 group span |
| 2026-08-10 | P1.2 | borrowed endpoint header 由 Workspace 在 setup 阶段预建 | 接受 | `tryRun(view)` 只重绑定 data/row step；allocator hook、并发独立 Workspace 和 sanitizer 通过 |
| 2026-08-10 | P1.3 | 仅融合 optional color → resize → normalize → layout 的 packed-f32 窄链 | 接受 | 不匹配顺序仍 staged；fused 与独立 oracle checksum 一致，stable case 均有收益且不需完整中间图 |
| 2026-08-10 | P1.4 | Recipe 始终强制 1 group / 0 intermediate，不提供 staged 静默 fallback | 接受 | 支持面外 `tryPrepare` 返回稳定错误；普通 Pipeline 继续承担通用 fallback |
| 2026-08-10 | P1.5 | 第一条 NEON predicate 仅覆盖 Nearest、横向 2×、NCHW、至少 256 pixels | 接受 | forced scalar/NEON checksum 一致；observed ISA 为 NEON；相对 scalar fused 2.97×，未测矩阵保持 scalar |
| 2026-08-10 | P1.6 | YUV 先关闭 scalar 数值与双 plane 运行合同，再选择 NEON predicate | 接受 | NV12/NV21 共用一条显式 ColorSpec reference；不让 Mat 或 staged executor 假装表达 multi-plane 输入 |
| 2026-08-10 | P1.6 | letterbox 使用正数 half-up，奇数 padding 余量给 right/bottom，坐标映射使用实际 x/y scale | 接受 | 规则不依赖 Python/banker rounding；连续边界能精确映射到整数 content 外框并往返 |
| 2026-08-10 | P1.6 | resize/letterbox 共用逐像素 padding 分支 | 回退 | 首次 stable 中普通 packed Nearest scalar 从约 0.356 ms 退到 0.688 ms；保留失败报告并拆回 resize 无分支热路径 |
| 2026-08-10 | P1.6 | packed letterbox Nearest、横向 content 2×、NCHW、至少 256 content pixels 的 NEON predicate | 接受 | 热路径复原后 letterbox NEON 0.251 ms、scalar 0.597 ms、staged 1.200 ms；observed ISA=NEON 且 checksum 一致 |
| 2026-08-10 | P1.7 | quantize v1 只开放 per-tensor U8/S8，NEON/per-channel 由后续证据决定 | 接受 | 先关闭可审计的 half-away-from-zero、zero point 和 saturate 合同，避免把不同量化语义塞入同一 predicate |
| 2026-08-10 | P1.7 | quantized resize/letterbox 共用逐像素 padding 分支 | 回退 | 首轮 S8 Linear/NHWC fused 仅为 staged 的 0.965×；保留失败报告并拆出无 padding 热路径 |
| 2026-08-10 | P1.7 | per-tensor U8/S8 scalar direct-store predicate | 接受 | 1280×720→640×640 Nearest/NCHW 的 U8 与 S8 均为 staged 的 1.33×，checksum 一致；不提前增加 NEON/per-channel |

### 14.1 P1.1 验证证据

- Release：`build-core-mat-neon-tests` 完整构建并通过 25/25 CTest；Pipeline
  定向门禁 4/4。
- 优化关闭：`build-b7-optimization-off` 的 Pipeline unit、zero-allocation、公共头编译
  和 ODR 门禁 4/4。
- Sanitizer：`build-phase2-sanitize` 在 ASan/UBSan 下通过相同门禁 4/4。
- 独立 header-only 安装消费：`./scripts/check_header_only_contract.sh` 通过 15/15。
- `./scripts/check_public_headers.sh`、`./scripts/check_docs.sh` 和
  `git diff --check` 全部通过。

### 14.2 P1.2 验证证据

- Release：Mat/view 差分、typed error matrix、canary、alias、并发和 allocator hook
  通过；完整 `build-core-mat-neon-tests` 通过 25/25。
- 优化关闭：`build-b7-optimization-off` 的 Pipeline 定向门禁通过 4/4。
- Sanitizer：`build-phase2-sanitize` 在 ASan/UBSan 下通过相同门禁 4/4。
- 独立 header-only 安装消费通过 15/15；public-header、ODR、文档合同和
  `git diff --check` 全部通过。

### 14.3 P1.3 验证和性能证据

- Nearest/Linear × NCHW/NHWC × optional color × 单值/三通道 normalize 固定矩阵
  与 staged oracle 一致；64 条确定性随机链强制同时包含 fused 和 staged 计划。
- canonical plan 的 `semantic_stage_count` 保持用户链数量，而
  `execution_group_count=1`、`full_frame_intermediates=0`、`workspace_bytes=0`；硬要求
  真实通过，不匹配 predicate 的合法链保持 staged。
- Release 完整 CTest 25/25；优化关闭与 ASan/UBSan 定向门禁各 4/4；独立
  header-only 安装消费 15/15；public-header、ODR、文档和 diff 门禁通过。
- 新增长期维护的 `cvh_benchmark_pipeline_header` 及 CI quick 行；
  [2026-08-10 stable baseline](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-scalar-fused-baseline.md)
  在 Apple M5 单线程五个冻结 case 上记录 1.23×–2.64× staged 中位数收益，成对
  checksum 一致。该数字是 scalar baseline，不提前充当 P1.5 NEON 跨平台门槛。

### 14.4 P1.4 验证证据

- `cvh.model_input.packed_f32` contract v1 生成与普通 canonical Pipeline 相同的 IR 和
  数值结果，且总是强制 1 execution group、0 完整中间图。
- fingerprint 对相同 spec 稳定，mean 等语义字段改变时变化；普通 Pipeline 的 recipe
  id/version/fingerprint 保持空值。
- Gray、非 F32 output 和非法 normalize count 通过 `tryPrepare` 明确失败，不生成
  staged Recipe Plan。
- Release 完整 CTest 25/25；优化关闭和 ASan/UBSan 定向门禁各 4/4；独立安装消费
  直接包含 `cvh/recipes/model_input.h` 并 prepare，header-only 15/15；public-header、
  ODR、文档和 diff 门禁通过。

### 14.5 P1.5 验证和性能证据

- ARM predicate：packed BGR/RGB U8、Nearest、输入宽度精确为输出 2×、NCHW F32、
  至少 256 output pixels；其余形状、Linear、NHWC、优化关闭和非 ARM 保持 scalar。
- forced scalar、forced NEON 与 Auto 结果一致；11-pixel 输出宽度覆盖 8-lane NEON 和
  3-pixel scalar tail，输入使用非连续 ROI 和偏移首地址。`candidate_route`、
  `actual_route`、`observed_isa`、fallback reason 与真实选择一致。
- NEON 候选已加入 allocator hook；Release 25/25、优化关闭 4/4、ASan/UBSan 4/4、
  header-only 15/15 通过。x86_64 交叉构建四个 Pipeline 目标成功；当前 ARM 主机没有
  Rosetta，x86 二进制运行返回系统 `-86`，因此这里只认定 compile gate，不宣称运行。
- [2026-08-10 NEON candidate report](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-neon-candidate.md)
  在 Apple M5 单线程记录 NEON 0.120 ms、scalar fused 0.356 ms，即 2.97×；相对
  staged 为 7.94×，三行 checksum 一致且 observed ISA 明确为 NEON。

### 14.6 P1.6 验证和性能证据

- 已新增 NV12/NV21 双 plane borrowed view 执行路径和 mutable/const helper；Y、UV/VU
  的空指针、逐 plane stride/容量、plane 间重叠、plane/output alias、descriptor/
  `ColorSpec` 不匹配均在任何输出写入前返回 typed status；multi-plane Mat run 明确返回
  `Unsupported`。
- YUV canonical chain 编译为单个 `YuvModelInputFused` group，scalar executor 直接写
  F32 NCHW/NHWC tensor，`semantic_stage_count=4`、`execution_group_count=1`、
  `full_frame_intermediates=0`、`workspace_bytes=0`，`explain()` 明确显示
  `yuv420-model-input`。
- 定向测试覆盖 NV12/NV21 × BT.601/709/2020 × Limited/Full × Center/Left ×
  Nearest/Linear × RGB/BGR × NCHW/NHWC，共 192 组，与测试侧独立 reference 逐元素
  一致；另有 limited/full neutral black/gray/white 锚点和奇数尺寸/非法链准备期失败。
- `cvh.model_input.yuv420_f32` contract v1 已建立；普通 canonical Pipeline 与 Recipe
  逐元素一致，ColorSpec range 改变会改变 fingerprint，且 Recipe 继续强制 1 group / 0
  intermediate。
- YUV canonical view run 已加入全局 allocator hook，warmup 后 measured run 为 0 heap
  allocation。
- `.letterbox()` 已支持单值/三通道 pad、Nearest/Linear staged operation 和 packed/YUV
  canonical direct-store fusion；packed 4 组 interpolation/layout 与 staged reference 一致，
  YUV letterbox Recipe 与独立 YUV+content/pad reference 一致，均保持 1 group / 0
  intermediate / 0 workspace。
- `PipelineTransform` 的 nominal/actual scale、resized size、四边 padding、连续边界正反
  映射和 padding 区域已由 half-up 与奇数 padding case 冻结；letterbox pad 改变会改变
  Recipe fingerprint，并使用独立 Recipe id。
- 首版 scalar 共用 loop 使普通 packed Nearest 从 P1.5 约 0.356 ms 回退到 0.688 ms，
  已从产品路径移除并保留[失败报告](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-yuv-letterbox-candidate.md)。
  拆回 resize 无分支热路径后为 0.368 ms，P1.5 性能量级恢复。
- [hot-path-restored stable 报告](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-yuv-letterbox-hotpath-restored.md)
  记录 letterbox staged/scalar/NEON 为 1.200/0.597/0.251 ms，即 NEON 相对 scalar
  2.38×、相对 staged 4.79×；三行 checksum 一致且 observed ISA=NEON。NV12 BT.709/
  Limited/Left Linear scalar 为 10.281 ms、checksum 稳定，未虚构 staged speedup。
- Release 完整构建与 CTest 25/25、优化关闭 4/4、ASan/UBSan 4/4、独立
  header-only 安装消费 15/15 已通过；allocator
  hook 当前直接测量 YUV+letterbox canonical view run。letterbox NEON predicate 和
  stable benchmark 已完成；x86_64 非 ARM 交叉编译四个 Pipeline 目标通过；
  public-header、文档和 `git diff --check` 门禁通过。P1.6 完成。

### 14.7 P1.7 验证和性能证据

- 新增 `quantize(target_type, scale, zero_point)` ordered operation；planner 只接受 F32
  Image → U8/S8 Image，scale 必须正且有限，zero point 必须在目标 dtype 范围内。
- scalar staged helper 和 packed/YUV direct-store executor 共用产品 quantize helper；
  测试侧使用独立 double reference，正负 `.5` ties、NaN、±Inf、有限溢出和 U8/S8
  saturation 锚点逐值一致。
- packed `resize/letterbox × Nearest/Linear × NCHW/NHWC × U8/S8` 共 16 组 canonical
  direct-store 与重排但语义等价的 staged reference 字节一致；fused Plan 为 1 group / 0
  intermediate / 0 workspace。
- `cvh.model_input.yuv420_s8` Recipe 与独立 YUV F32 reference 再量化的结果逐字节一致；
  quantize scale 改变会改变 fingerprint，Recipe id 区分 input format、dtype 和
  letterbox 变体。
- U8 NCHW、S8 NHWC borrowed tensor 的精确容量成功，少 1 byte 在写入前返回
  `BufferTooSmall` 且 canary 不变，input/output alias 返回 `AliasingNotSupported`。
- allocator hook 当前测量 YUV + letterbox + normalize + S8 quantize + NCHW direct-store，
  warmup 后 measured run 为 0 heap allocation。
- 首版共用 padding loop 的[失败报告](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-quantize-scalar.md)
  记录 U8 1.20×，但 S8 Linear/NHWC 只有 0.965×，因此该 loop 已移除。拆出无 padding
  热路径后，小 S8 case 恢复到 1.025×–1.049×。
- [U8/S8 stable 报告](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-quantize-large-s8-evidence.md)
  记录 1280×720→640×640 Nearest/NCHW：U8 staged/fused 为 1.944/1.459 ms，S8 为
  1.941/1.456 ms，两者均 1.33×，成对 checksum 一致。YUV S8 Linear scalar 为
  11.955 ms，只报告原始成本，不虚构 staged speedup。
- Release 完整构建与 CTest 25/25、opt-off 4/4、ASan/UBSan 4/4、header-only 安装
  消费 15/15、x86_64 四目标编译、zero-allocation 和 stable benchmark 已通过；
  public-header、文档和 `git diff --check` 门禁通过。P1.7 完成。

### 14.8 P1.8 Supported 审计证据

- 总设计 16.1 节已按实际 fusion predicate 冻结 packed/YUV、resize/letterbox、
  F32/U8/S8、NCHW/NHWC、scalar/NEON 和明确 Proposed 边界；Recipe 表与代码中的 12 个
  v1 id 一致，fingerprint 覆盖 input format、ColorSpec、geometry、normalize 和
  quantize。
- `cvh_test_pipeline` 在目标 ARM 运行 12 suites / 34 tests 全通过；机器可读
  `test/ci/header_gate_expectations.json` 已包含 Pipeline unit、header、ODR 和
  zero-allocation targets，无需降低或删除已有 inventory。
- [最终 P1 Supported stable audit](../benchmark/results/internal/pipeline/stable/2026-08-10-p1-supported-audit.md)
  在 Apple M5 / arm64、Release、单线程、warmup 3 / iters 3 / repeats 7 下复测全部
  代表 predicate：packed Linear scalar 为 staged 的 1.20×–1.61×；packed Nearest
  NEON 为 scalar 的 3.06×；letterbox NEON 为 scalar 的 2.36×；U8/S8 scalar 为
  staged 的 1.34×/1.33×；NV12/NV21 F32/S8 均记录真实 scalar 成本、独立 checksum、
  1 group / 0 intermediate / 0 workspace。
- 最终 Release 完整构建与 CTest 25/25、优化关闭 4/4、ASan/UBSan 4/4、x86_64
  非目标四目标编译、header-only 安装消费 15/15、public-header 和产品依赖门禁通过；
  `otool -L cvh_benchmark_pipeline_header` 只包含 libc++ 与 libSystem，不含 OpenCV、
  摄像头或推理 runtime。
- README、总设计、Recipe 表、benchmark index 和实际代码已同步；最终文档检查与
  `git diff --check` 通过。P1.8 与整体 P1 完成。

后续 NEON 或融合候选即使最终回退，也必须在这里记录输入、route、测量方法、结果和
回退原因。只有最终接受的实现留在产品路径。

## 15. 文档和状态所有权

| 信息 | 所有者 |
| --- | --- |
| Pipeline 产品定位、公开语义和长期边界 | `pipeline-module-design.md` |
| P1 批次、进行状态、候选和完成证据 | 本文 |
| 当前公开支持面 | 顶层 `README.md` 和对应 public header |
| 测试 inventory | `test/ci/header_gate_expectations.json` 和测试 manifest |
| 性能数字 | dated benchmark report + CSV + metadata |
| 历史完成记录 | Git history |

上位设计只保留 P1 总状态和本文链接，不复制本计划的逐批日志。本文不重新定义
`pipe()` 顺序语义；新增或改变公开合同必须先同步上位设计。

## 16. P1 总完成定义

P1 整体只有在以下条件全部满足后才能标为完成：

- packed-f32 v1 的 Mat/view、scalar fused、Recipe 和至少一个 ARM NEON predicate
  完成 Supported 审计；
- `requireNoFullFrameIntermediate()` 和 `requireSingleExecutionGroup()` 对 Recipe
  真实成立；
- prepared run 零分配、Plan 并发和 workspace/alias 合同通过；
- info/explain/run_info 与实际计划和 observed ISA 一致；
- NV12/NV21、letterbox、U8/S8 各自至少有一个完成 Supported 审计的 predicate；
  如果数据证明某项不应留在 P1，必须先同步修改上位阶段合同并记录迁移决定，不能
  仅以保持 Proposed 为由把 P1 标成完成；
- Release、优化关闭、sanitizer、非 ARM、public-header、ODR、安装消费全部通过；
- 目标 ARM 设备存在可复现 dated 性能报告；
- 没有摄像头、OpenCV 或推理 runtime 产品二进制依赖；
- 总设计、README、本文和机器可读测试 inventory 同步。

P1 完成不等于 Pipeline 的全部路线完成。OpenCV/V4L2/ROS 2 adapter、动态 shape、
机器人后处理和多输出仍分别属于 P2、P3、P4。
