# cvh Pipeline 模块设计与支持合同

状态：Supported；承诺范围限于本文 P0 ordered Pipeline 和 P1 v1 支持矩阵，后续路线
仍为 Proposed。

本文冻结 Pipeline 的产品边界、首选 API、执行语义、优化合同和分阶段落地方案。
示例用于指导实现；只有对应 header、测试、安装消费和性能门禁全部完成后，相关
能力才进入 Supported 状态。

## 0. 实施状态

更新时间：2026-08-10

总状态：**P0 和 P1 已完成，P1 v1 已通过 Supported 审计**。P1 的逐批状态和证据由
[pipeline-p1-implementation-plan.md](pipeline-p1-implementation-plan.md) 维护。

| 批次 | 范围 | 状态 | 完成证据 |
| --- | --- | --- | --- |
| P0.0 | 冻结首批 API 子集、建立实时台账、复核工程接入点 | 已完成 | 2026-08-07：确认独立 public-header/ODR/unit 接入点；冻结下述 P0 子集 |
| P0.1 | descriptor、borrowed view、status/info、workspace 基础类型 | 已完成 | 2026-08-07：8 个 public header 独立编译与 ODR smoke 通过 |
| P0.2 | fluent builder、ordered IR、类型/shape 推导、`prepare/explain` | 已完成 | 2026-08-07：ordered stage、顺序错误定位、输出约束和硬要求单测通过 |
| P0.3 | scalar reference、staged workspace、一次性/prepared `run` | 已完成 | 2026-08-07：数值/ROI/prepared-one-shot/并发测试通过；allocator hook 实测 prepared `tryRun()` 为 0 次堆分配 |
| P0.4 | public header、单元/随机链、ODR、install consumer 门禁 | 已完成 | 2026-08-07：64 条确定性随机合法链、public-header、ODR、install consumer、ASan+UBSan、优化开/关和完整 25 项 CTest 通过 |
| P1 | 模型输入融合、ARM NEON、`modelInput` Recipe | 已完成 | 2026-08-10：packed/YUV scalar direct-store、letterbox、per-tensor U8/S8、Recipe、borrowed/零分配、窄 NEON 和完整发布门禁通过；最终 stable audit 覆盖 NV12/NV21 与所有代表 predicate |

当前实现接受单输入、单输出的 `cvh::Mat`，prepared 路径也接受 packed 或双 plane
NV12/NV21 `ConstImageView` 到连续 `TensorView`；packed/YUV F32 scalar fusion、
letterbox、per-tensor U8/S8 quantize、Recipe v1 和窄 ARM NEON predicates 已形成
reference 闭环并进入本文件 16.1 节限定的 Supported 支持面。per-channel quantize、
动态 shape、非连续 tensor、adapter 和后处理等未关闭能力继续明确保持 Proposed。

作为历史验收基线，P0 冻结子集为：

- 可执行输入：二维 `CV_8U/CV_32F` Gray、BGR、RGB `cvh::Mat`，三通道 Mat 默认
  按 BGR 解释；
- 可执行输出：二维 packed Image，或连续 NCHW/NHWC Tensor `cvh::Mat`；
- ordered operations：`color/resize/normalize/layout`；
- 运行方式：一次性 `run()` 与 descriptor `prepare()` + 外部
  `PipelineWorkspace`；
- 执行策略：可信 scalar staged reference，每个未融合 stage 保持独立执行组；
- 强制融合策略按真实计划检查，P0 不满足时必须失败；
- `ImageView/TensorView/NV12` 先冻结描述和 helper，执行 overload 在完成容量、stride
  和多 plane 门禁前保持不可调用。

实施规则：每个批次开始、完成、回退或阻塞时先更新本表；完成状态必须附带实际
构建/测试命令，不能仅以代码存在作为完成证据。

最近验证命令（2026-08-07）：

~~~bash
cmake -S . -B build-core-mat-neon-tests
cmake --build build-core-mat-neon-tests \
  --target cvh_test_pipeline cvh_pipeline_zero_allocation_smoke \
           cvh_pipeline_headers_compile_smoke \
           cvh_pipeline_header_odr_smoke --parallel 2
ctest --test-dir build-core-mat-neon-tests --output-on-failure \
  -R '^cvh_(pipeline_headers_compile_smoke|pipeline_header_odr_smoke|pipeline_zero_allocation_smoke|test_pipeline)$'
./scripts/check_header_only_contract.sh
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build-phase2-sanitize --output-on-failure \
  -R '^cvh_(pipeline_headers_compile_smoke|pipeline_header_odr_smoke|pipeline_zero_allocation_smoke|test_pipeline)$'
cmake --build build-b7-optimization-off \
  --target cvh_test_pipeline cvh_pipeline_zero_allocation_smoke \
           cvh_pipeline_headers_compile_smoke \
           cvh_pipeline_header_odr_smoke --parallel 2
ctest --test-dir build-b7-optimization-off --output-on-failure \
  -R '^cvh_(pipeline_headers_compile_smoke|pipeline_header_odr_smoke|pipeline_zero_allocation_smoke|test_pipeline)$'
cmake --build build-core-mat-neon-tests --parallel 2
ctest --test-dir build-core-mat-neon-tests --output-on-failure
~~~

Sanitizer 记录：首次使用 `ASAN_OPTIONS=detect_leaks=1` 的运行在 macOS 上被 ASan
运行时以“不支持 leak detection”拒绝，测试主体未执行，因此不计为产品失败或通过；
随后使用该平台支持的 `detect_leaks=0` 重新验证，4 项 Pipeline ASan+UBSan 测试
全部通过。

## 1. 对外话术

最简版本：

> **你定义顺序，cvh 保证语义；cvh 自动融合能够安全融合的步骤；需要确定性能时，
> 使用 Recipe 或强制融合策略。**

这句话对应三个不可混淆的概念：

| 概念 | 含义 |
| --- | --- |
| Pipeline | 用户按链式调用定义的、有顺序的处理语义 |
| Execution Group | `prepare()` 后实际由一次循环、一个 tile 调度或一个专用内核完成的一组步骤 |
| Recipe | cvh 官方维护并冻结输入条件、数值结果、内存属性和性能门槛的常用 Pipeline |

**Pipeline 不等于一个 kernel。** 一个 Pipeline 可以编译成一个融合执行组，也可以
编译成多个执行组。无论采用哪条路径，结果都必须符合用户写下的顺序。

## 2. 产品定位

Pipeline 是面向边缘 AI 和机器人感知的数据准备层。它位于相机/中间件与模型、
跟踪、定位等消费者之间：

~~~text
camera / ROS 2 / file / SDK buffer
                  |
             cvh Pipeline
          /         |         \
 model tensor   tracking image   mask / regions
~~~

`core` 和 `imgproc` 继续提供独立的 OpenCV-style 基础算子；Pipeline 负责：

- 用一条有序链表达真实部署中的预处理和后处理；
- 在不改变语义的前提下消除完整中间图、合并循环和复用采样；
- 为实时循环提供准备一次、无逐帧堆分配的执行计划；
- 让调用者能够检查是否真的融合、用了多少 workspace、实际走了什么 ISA；
- 用 Recipe 给高频固定组合提供可测试、可承诺的性能产品。

这也是 cvh 与 OpenCV 做差异化的核心：基础算子保持熟悉和可迁移，部署层则提供
有序组合、prepare/run、外部 workspace、融合证明和面向边缘设备的 Recipe。差异
不建立在重新发明 `Mat` 或摄像头 API 上。

它不是另一套推理框架，也不负责设备采集、模型执行、SLAM 或运动控制。

## 3. 设计原则

### 3.1 用户顺序就是语义

下面两条链不是同一个 Pipeline：

~~~cpp
cvh::pipe(input, output)
    .resize(640, 640)
    .normalize(mean, stddev)
    .run();

cvh::pipe(input, output)
    .normalize(mean, stddev)
    .resize(640, 640)
    .run();
~~~

Builder 按调用顺序记录操作。优化器可以融合步骤，但不得进行会改变公开结果的
重排。只有经过数值合同证明安全的参数折叠或等价变换才允许发生。

### 3.2 任意合法组合都正确，不承诺任意组合都融合

“链可以自由组合”不等于“每一种排列都要有专用融合内核”。

- 合法且已识别的高频组合：走融合路径；
- 合法但未识别的组合：使用通用执行组和 workspace 中间态；
- 非法组合：在 `prepare()`，或一次性 `run()` 的准备阶段，给出明确错误；
- 要求确定内存/性能属性：使用 Recipe 或 `require...()` 策略，不接受静默退化。

因此实现不需要为 N 个操作手写 N! 个排列。正确性由通用路径兜底，性能由有限的
融合规则和 Recipe 收敛。

### 3.3 简单入口优先，底层视图按需出现

首屏 API 只展示 `cvh::pipe(input, output)` 和链式操作。`ImageView`、
`TensorView`、descriptor、stride、plane 等概念保留在高级接口中，只有接入相机
buffer、推理运行时或非连续内存时才需要理解。

### 3.4 显式执行

链式调用只构建语义，`run()` 才执行。析构函数绝不隐式运行：

~~~cpp
cvh::pipe(input, output)
    .color(cvh::Color::RGB)
    .resize(640, 640)
    .run();
~~~

这样错误位置、耗时边界和对象生命周期都是可见的。

## 4. 模块和命名

用户包含：

~~~cpp
#include <cvh/pipeline/pipeline.h>
~~~

继续链接现有 header-only target：

~~~cmake
find_package(cvh CONFIG REQUIRED)
target_link_libraries(app PRIVATE cvh::headers)
~~~

公开名称位于 `cvh` 命名空间：

- `cvh::pipe(...)`：创建 fluent builder；
- `cvh::PipelineBuilder`：记录有序语义；
- `cvh::PipelinePlan`：不可变的已准备计划；
- `cvh::PipelineWorkspace`：调用者持有的运行临时空间；
- `cvh::PipelineInfo`：计划的静态信息；
- `cvh::PipelineRunInfo`：某次执行实际观察到的路径；
- `cvh::PipelineStatus`：`tryPrepare/tryRun` 的显式状态。

入口命名为 `pipe` 而不是 `pipeline`，避免函数名与未来模块命名产生冲突，也让
高频链更短。文件目录可以是 `include/cvh/pipeline/`，但不建立
`cvh::pipeline` C++ 命名空间。

## 5. 三层使用方式

### 5.1 一次性调用：最容易理解

已有 `cvh::Mat` 的用户可以直接写：

~~~cpp
cvh::pipe(input, output)
    .color(cvh::Color::RGB)
    .resize(640, 640)
    .normalize({0.485f, 0.456f, 0.406f},
               {0.229f, 0.224f, 0.225f})
    .layout(cvh::Layout::NCHW)
    .run();
~~~

模型运行时提供裸指针时，用辅助函数描述输出，不要求用户先声明
`TensorView` 类型：

~~~cpp
auto output = cvh::nchw<float>(
    tensor_data, tensor_bytes, 1, 3, 640, 640);

cvh::pipe(input, output)
    .color(cvh::Color::RGB)
    .resize(640, 640)
    .normalize(mean, stddev)
    .layout(cvh::Layout::NCHW)
    .run();
~~~

一次性 `run()` 可以完成校验、建计划和临时空间分配。它适合上手、工具程序和
低频调用，不提供实时循环的零分配保证。

### 5.2 Prepared Plan：实时路径

机器人和边缘设备应在初始化阶段准备计划：

~~~cpp
const auto input_desc = cvh::imageDesc(
    width, height, cvh::PixelFormat::NV12, color_spec);
const auto output_desc =
    cvh::tensorDesc<float>({1, 3, 640, 640}, cvh::Layout::NCHW);

const cvh::PipelinePlan plan =
    cvh::pipe(input_desc, output_desc)
        .color(cvh::Color::RGB)
        .resize(640, 640, cvh::Interpolation::Linear)
        .normalize(mean, stddev)
        .layout(cvh::Layout::NCHW)
        .preferFusion()
        .prepare();

cvh::PipelineWorkspace workspace(plan);

while (camera.read(frame)) {
    auto input = cvh::nv12(
        frame.yData(), frame.yStride(), frame.yBytes(),
        frame.uvData(), frame.uvStride(), frame.uvBytes(),
        width, height, color_spec);

    cvh::PipelineRunInfo run_info;
    plan.run(input, output, workspace.view(), &run_info);
    model.invoke();
}
~~~

`prepare()` 可以做分配和较重的分析；成功后 `run()` 必须满足：

- 不修改计划；
- 不做堆分配；
- 不重新编译或重新选择整条计划；
- 输入、输出和 workspace 与 descriptor 相符时直接执行；
- 同一个 plan 可被多线程共享，每个并发调用使用独立 workspace 和
  `PipelineRunInfo`。

### 5.3 Recipe：确定的产品路径

高频模型输入使用官方 Recipe：

~~~cpp
cvh::ModelInputRecipe spec;
spec.input = input_desc;
spec.output = output_desc;
spec.color = cvh::Color::RGB;
spec.interpolation = cvh::Interpolation::Linear;
spec.mean = mean;
spec.stddev = stddev;
spec.normalize_count = 3;

const auto plan = cvh::recipes::modelInput(spec)
    .requireNoFullFrameIntermediate()
    .requireSingleExecutionGroup()
    .prepare();
~~~

Recipe 不是另一套执行引擎。它生成同一种有序 IR，但只接受经过产品验证的形状、
格式和操作集合，并附带版本化保证。Recipe 的价值是“可承诺”，不是“换一种写法”。

每个 Recipe 必须发布并可查询：Recipe id/version、支持的输入 predicate、数值合同、
最大 workspace、完整中间态数量、可用 backend，以及有测量证据的性能门槛。输入
不满足 predicate 时应在 `prepare()` 明确失败；需要通用 fallback 的用户改用普通
Pipeline，不能让 Recipe 在未告知用户时失去保证。

F32 Recipe v1 的稳定 id 是 `cvh.model_input.packed_f32` 和
`cvh.model_input.yuv420_f32`；量化输出将后缀替换为 `packed_u8`、`packed_s8`、
`yuv420_u8` 或 `yuv420_s8`。letterbox 变体使用对应的 `_letterbox` id，contract
version 均为 1。
`PipelineInfo::recipe_id`、`recipe_contract_version` 和 `recipe_fingerprint` 可查询。
fingerprint 覆盖 input/output descriptor、完整 YUV `ColorSpec`、目标颜色、几何模式、
插值、letterbox pad、normalize 参数，以及 U8/S8 输出的 quantize scale/zero point，
只用于部署审计，不替代语义或 capability 判断。

Recipe v1 的 id 表与实际代码一致：

| 输入 | 几何 | F32 | U8 | S8 |
| --- | --- | --- | --- | --- |
| packed BGR8/RGB8 | resize | `cvh.model_input.packed_f32` | `cvh.model_input.packed_u8` | `cvh.model_input.packed_s8` |
| packed BGR8/RGB8 | letterbox | `cvh.model_input.packed_f32_letterbox` | `cvh.model_input.packed_u8_letterbox` | `cvh.model_input.packed_s8_letterbox` |
| NV12/NV21 | resize | `cvh.model_input.yuv420_f32` | `cvh.model_input.yuv420_u8` | `cvh.model_input.yuv420_s8` |
| NV12/NV21 | letterbox | `cvh.model_input.yuv420_f32_letterbox` | `cvh.model_input.yuv420_u8_letterbox` | `cvh.model_input.yuv420_s8_letterbox` |

这些 id 共享 contract version 1，但 fingerprint 仍区分 NV12/NV21、完整 ColorSpec、
shape、layout、插值、pad、normalize 和 quantize 参数。

## 6. 操作语义

### 6.1 `normalize`，不使用 `mean().std()`

模型预处理中的 mean 和 stddev 通常是常量参数，不是对当前图像做统计。因此主 API
使用：

~~~cpp
.normalize(mean, stddev)
~~~

其逐通道语义冻结为：

~~~text
y = (x - mean[c]) / stddev[c]
~~~

`mean()` 和 `std()` 容易被理解为全局归约，会引入完全不同的输入依赖和输出类型，
所以不用于表达常量归一化。高级用户需要拆分公式时可使用：

~~~cpp
.sub(mean)
.mul(inv_stddev)
~~~

但 `normalize` 是模型输入 Recipe 的标准拼写。参数是 scalar 还是逐通道数组、通道
对应顺序、输入缩放范围和舍入规则都必须进入数值合同。

### 6.2 `quantize` 数值合同

P1 v1 只支持 per-tensor F32 Image 到 U8/S8 Image 的量化：

~~~text
q = saturate(round(real / scale) + zero_point)
~~~

`scale` 必须为正且有限，`zero_point` 必须在目标 dtype 范围内。有限值使用
half-away-from-zero 舍入；NaN 映射到 `zero_point`，正负无穷分别饱和到目标类型的
最大值和最小值，有限大数也必须在转成整数前安全饱和。布局转换发生在量化之后，
因此 canonical 链为
`color? -> resize/letterbox -> normalize -> quantize -> layout/store`。

per-channel 量化没有复用该合同，保持 Proposed；需要时必须增加显式参数、独立
predicate 和 Recipe 版本，不能让实现按通道数猜测。

### 6.3 类型状态限制“随机组合”

每个操作声明允许的输入状态和产生的输出状态。首批建议：

| 操作类别 | 示例 | 合法输入 | 输出状态 |
| --- | --- | --- | --- |
| 颜色 | `color` | Image | Image |
| 几何 | `crop/resize/letterbox/rectify` | Image 或受支持 Mask | 保持输入的逻辑种类 |
| 点运算 | `convert/normalize/quantize/threshold/inRange/equal` | Image 或受支持 Tensor/Mask | Image、Tensor 或 Mask |
| 布局 | `layout` | Image/HWC Tensor | Tensor |
| 邻域 | `morphology` | 单通道 Image/Mask | Mask |
| 全局/区域 | `argmax/connectedComponents` | 受支持 Tensor/Mask | Mask/Regions |

例如第一版不支持在 NCHW tensor 上直接调用图像 resize：

~~~cpp
cvh::pipe(input, output)
    .layout(cvh::Layout::NCHW)
    .resize(640, 640)  // prepare 失败：resize 不接受 NCHW Tensor
    .prepare();
~~~

错误至少包含 stage index、操作名、实际状态和期望状态：

~~~text
pipeline stage 1 "resize": expected Image, got Tensor<F32,NCHW>
~~~

未来若增加 tensor resize，它应作为明确的新能力进入支持矩阵，而不是让当前实现
猜测用户意图。

### 6.4 输出 descriptor 是最终约束

Builder 推导每一步的 dtype、shape、layout 和颜色状态。最终状态必须与输出
descriptor 一致；不允许因为输出 buffer 恰好够大而默默改变布局或类型。

动态输入只能在 descriptor 明确允许的维度范围内变化。超出范围需要重新
`prepare()`，不能在实时 `run()` 中隐式重建计划。

## 7. 内部编译模型

Fluent API 记录一个有序、不可变的中间表示：

~~~text
Input
  -> Color(RGB)
  -> Resize(640x640, Linear)
  -> Normalize(mean, stddev)
  -> Layout(NCHW)
  -> Output
~~~

`prepare()` 依次完成：

1. 输入/输出 descriptor 和 buffer 条件校验；
2. 逐 stage 的类型、shape、颜色和坐标变换推导；
3. 在数值合同允许时折叠常量和等价点运算；
4. 按依赖、halo、边界模式和数据类型划分 Execution Group；
5. 为每组选择 scalar、Universal Intrinsics 或专用 ISA 候选；
6. 生成坐标表、量化参数和 workspace 布局；
7. 检查调用者指定的融合/内存要求；
8. 生成不可变 `PipelinePlan` 和可解释信息。

Builder 内部使用紧凑的 operation node/parameter variant 记录链，不为每一种链生成
新的模板类型。已准备 plan 保存确定的执行组和入口，逐帧运行不再解释 builder。
这可以控制 header-only 库的编译时间和代码体积。

### 7.1 执行组类别

- **Pointwise Group**：颜色映射、线性变换、normalize、quantize、layout/store；
- **Geometry Group**：crop、resize、letterbox、remap/rectify，通常按 tile 或行执行；
- **Neighborhood Group**：滤波、形态学等需要 halo 的操作；
- **Global Group**：argmax、直方图、连通域等需要归约或跨 tile 状态的操作。

融合规则围绕这些类别编写，不围绕完整操作排列编写。例如一个 geometry producer
可以把采样值直接交给 pointwise consumer 和最终 store；遇到需要完整全局状态的
操作时则形成边界。

### 7.2 通用兜底路径

合法链没有专用融合规则时，编译器生成多个执行组。组间中间态来自预先规划的
workspace，prepared `run()` 仍然不做堆分配。

因此系统同时具备：

- 一条简单、可信的顺序 reference 路径；
- 一套可组合的通用执行组；
- 少量高价值 fused kernel；
- 对高频固定组合有硬门槛的 Recipe。

### 7.3 新操作的接入合同

每个新 operation 只需要提供本操作的 traits，而不是实现与所有已有操作的排列：

- `validate`：参数及输入状态是否合法；
- `infer`：输出 dtype、shape、layout、颜色和坐标变换；
- `reference`：可信的顺序 scalar 实现；
- `memory`：halo、临时空间、in-place 和别名属性；
- `fusion tags`：允许与哪些 producer/consumer 类别融合；
- `backend candidates`：可选的 UI/ISA 专用执行器及其 predicate。

融合规则是少量的 producer-consumer 模式。没有匹配规则时自动落到 reference/staged
路径，所以增加 operation 不会要求同步补齐所有组合内核。

## 8. 融合策略和可观察性

### 8.1 默认策略

`preferFusion()` 是默认策略：

- 尽可能减少中间态和内存流量；
- 无法融合时允许正确的 staged fallback；
- 不因追求融合改变操作顺序或数值合同。

显式写出它主要用于代码自说明：

~~~cpp
auto plan = cvh::pipe(input_desc, output_desc)
    .color(cvh::Color::RGB)
    .resize(640, 640)
    .normalize(mean, stddev)
    .preferFusion()
    .prepare();
~~~

### 8.2 强制策略

两个首批硬约束：

~~~cpp
.requireNoFullFrameIntermediate()
.requireSingleExecutionGroup()
~~~

- `requireNoFullFrameIntermediate()`：允许多个 tile/row 执行组，但禁止物化完整帧；
- `requireSingleExecutionGroup()`：整条链必须编译为一个执行组，是更强的要求。

不满足要求时 `prepare()` 失败，并说明第一个阻断融合的 stage。不能静默回退后仍
返回成功。

`requireFused()` 不作为首选公开名，因为“fused”容易被误解成一个函数、一次
调度或一次物理 load。若以后提供，只能作为
`requireSingleExecutionGroup()` 的清晰别名。

### 8.3 `explain()`

`explain()` 是 setup/debug API，可以分配字符串，不允许在实时路径中调用：

~~~cpp
std::cout << plan.explain();
~~~

示例输出：

~~~text
semantic stages: 4
  [0] color NV12 -> RGB8
  [1] resize 1920x1080 -> 640x640 linear
  [2] normalize RGB8 -> F32
  [3] layout HWC -> NCHW

execution groups: 1
  [0] fused tiled: color + resize + normalize + layout/store

full-frame intermediates: 0
workspace: 49152 bytes, alignment 64
candidate route: neon
requirements: no-full-frame-intermediate [satisfied]
~~~

`PipelineInfo` 至少公开：

~~~cpp
struct PipelineInfo {
    int semantic_stage_count;
    int execution_group_count;
    int full_frame_intermediates;
    std::size_t workspace_bytes;
    std::size_t workspace_alignment;
    PipelineExecutionClass execution_class;
    PipelineRoute candidate_route;
};
~~~

候选 route 不是实际执行证据。每次 `run()` 的真实 dispatch、observed ISA、
线程数和 fallback 原因写入调用者提供的 `PipelineRunInfo`，不得保存在全局
“last route”或可变 plan 状态中。

## 9. “一次访存”的准确合同

对通用 Pipeline 的宣传使用：

> cvh 自动融合安全步骤，减少完整中间态和重复内存流量。

不宣传“每个输入 byte 物理上只读取一次”。双线性 resize、remap、halo、cache
miss 和多输出采样都可能重复读取源数据。

可测试的属性是：

- **single execution group**：整条链由一个执行组完成；
- **no full-frame intermediate**：不物化 BGR、resized、float planar 等完整中间图；
- **direct final store**：最终元素直接写入目标布局；
- **zero heap allocation per run**：所有临时空间来自外部 workspace；
- **shared upstream work**：多输出时复用可共享的解码、坐标和采样。

只有通过静态计划检查、运行时观测和目标设备基准的 Recipe，才能公开声称上述具体
属性。即使 single execution group 成立，也不等于每个源像素只执行一次 CPU load。

## 10. 数据和内存接口

### 10.1 简单类型

主入口直接支持 `cvh::Mat`，并提供常用 borrowed-view helper：

~~~cpp
auto image = cvh::rgb(data, bytes, width, height, row_stride);
auto yuv = cvh::nv12(
    y_data, y_stride, y_bytes,
    uv_data, uv_stride, uv_bytes,
    width, height, color_spec);
auto tensor = cvh::nchw<float>(
    data, bytes, batch, channels, height, width);
~~~

用户不需要写出 `ImageView` 或 `TensorView` 类型名，但 helper 返回的仍是严格的
无所有权视图。

一次性 API 接收 `cvh::Mat&` 输出时，可以按 cvh/OpenCV 风格创建或调整目标存储；
接收 `nchw/nhwc/image` 等 borrowed view 时只能写入调用者已提供的容量。prepared
`run()` 对两者都不允许重新分配，输出必须与 prepared descriptor 完全匹配。

### 10.2 高级视图

底层必须能表达：

- dtype、shape、layout 和逐维 stride；
- packed、planar 和 multi-plane 图像；
- 非连续行、ROI 和带 padding 的 tensor；
- 每个 plane 的地址、row stride 和可访问 byte 数；
- YUV 的 matrix、range 和 chroma location；
- 输出容量、对齐和可写范围。

Tensor 不是第二套用户必须学习的容器，而是连接推理运行时所必需的内存合同。
Pipeline 不拥有输入、输出和 workspace；同步 `run()` 返回前，它们必须保持有效。

### 10.3 YUV 不能只写格式名

NV12/NV21 还必须携带：

~~~cpp
cvh::ColorSpec color_spec{
    cvh::ColorMatrix::BT709,
    cvh::ColorRange::Limited,
    cvh::ChromaLocation::Left
};
~~~

Pipeline 不根据分辨率、设备名或平台猜测颜色合同。Adapter 可以设置明确默认值，
但 `PipelineInfo` 和 Recipe fingerprint 必须记录最终值。

## 11. 摄像头和外部依赖边界

把 Pipeline 集成进摄像头应用，不意味着 cvh 要依赖摄像头库。职责分界是：

~~~text
camera driver / V4L2 / ROS 2 / vendor SDK
              |
       obtains and owns frame
              |
      wraps CPU-accessible memory
              |
          cvh::pipe(...)
~~~

Pipeline core：

- 不打开、关闭或配置摄像头；
- 不负责 dequeue/enqueue buffer；
- 不包含 V4L2、ROS 2、OpenCV 或厂商 SDK header；
- 只接收指针、尺寸、格式、plane、stride、容量和颜色规格；
- 第一阶段只接受 CPU 可访问的已映射内存。

DMA-BUF fd、GPU surface、NPU 私有内存和厂商 opaque handle 的 import/sync 是后续
platform adapter 能力，不放进 MVP core。可选集成建议位于：

~~~text
integration/opencv/
integration/ros2/
integration/v4l2/
integration/vendor/
~~~

这些 adapter 或示例不得成为 `cvh::headers` 的传递依赖。

## 12. 模型和机器人使用示例

### 12.1 模型输入

~~~cpp
auto plan = cvh::pipe(camera_desc, model_input_desc)
    .color(cvh::Color::RGB)
    .letterbox(640, 640, 114)
    .normalize(mean, stddev)
    .layout(cvh::Layout::NCHW)
    .requireNoFullFrameIntermediate()
    .prepare();
~~~

`plan.transform()` 返回确定性的 `PipelineTransform`：原/目标尺寸、nominal scale、
实际 `scale_x/scale_y`、resized 尺寸和四边 padding。`sourceToTarget()`、
`targetToSource()` 使用连续图像边界坐标，`isPadding()` 判断目标点是否落在 content
外；用户不需要重复推导检测框或关键点的 letterbox 变换。

### 12.2 模型输出后处理

通用图像/张量后处理可以继续用 fluent API：

~~~cpp
cvh::pipe(logits, mask)
    .argmax(cvh::Axis::Channels)
    .resize(frame_width, frame_height,
            cvh::Interpolation::Nearest)
    .equal(class_id)
    .morphology(cvh::Morphology::Open, kernel)
    .run();
~~~

YOLO、DETR 等模型特有 decode、anchor/grid 解释和 NMS 不进入首批通用 core，
应作为 `recipes/model/...` 或独立扩展。通用 Pipeline 保持模型无关。

### 12.3 经典机器人感知

~~~cpp
cvh::pipe(camera_frame, regions)
    .color(cvh::Color::Gray)
    .rectify(rectify_map)
    .inRange(low, high)
    .morphology(cvh::Morphology::Close, kernel)
    .connectedComponents()
    .run();
~~~

这条链覆盖机器人常见的灰度初始化、畸变校正、二值化、形态学和区域提取。
Neighborhood/Global stage 可能形成多个执行组；用户需要硬性能属性时应选择对应
Recipe 并检查 `PipelineInfo`。

### 12.4 多输出

多输出是高级能力，语义是共享前缀后建立分支：

~~~cpp
auto plan = cvh::pipe(camera_desc)
    .rectify(rectify_map)
    .output("tracking", tracking_desc, [] (auto branch) {
        return branch
            .color(cvh::Color::Gray)
            .resize(960, 540);
    })
    .output("detector", detector_desc, [&] (auto branch) {
        return branch
            .color(cvh::Color::RGB)
            .letterbox(640, 640, 114)
            .normalize(mean, stddev)
            .layout(cvh::Layout::NCHW);
    })
    .prepare();
~~~

前缀和每个 branch 内部仍严格按书写顺序执行；不同输出分支之间没有人为先后语义，
编译器可以交错 tile 以复用输入工作。

字符串查找只发生在初始化：

~~~cpp
const int tracking_slot = plan.outputSlot("tracking");
const int detector_slot = plan.outputSlot("detector");

cvh::PipelineOutputBindings outputs(plan);
outputs.bind(tracking_slot, tracking_image);
outputs.bind(detector_slot, detector_tensor);

plan.run(input, outputs, workspace.view(), &run_info);
~~~

实时循环中使用整数 slot，不做字符串查找或容器扩容。多输出不进入第一阶段 MVP，
但有序 IR 和 workspace 设计必须预留这一能力。

## 13. 运行时合同

### 13.1 生命周期和所有权

- Builder 拥有操作描述和小型标量参数，不拥有输入/输出 buffer；
- 大型 LUT、rectify map 等资源使用显式 owned/borrowed handle；borrowed 资源必须
  至少存活到 plan 的最后一次 `run()` 返回；
- Plan 是准备完成后的不可变对象；
- Workspace 由调用者持有，不可被两个并发 `run()` 共用；
- 输入、输出和 workspace 在同步 `run()` 返回前保持有效；
- Plan 可以跨线程共享，前提是每个调用拥有独立 workspace、输出和 run info。

### 13.2 分配

| 阶段 | 是否允许堆分配 |
| --- | --- |
| Builder 链式调用 | 允许 |
| `prepare()` | 允许 |
| `plan.explain()` | 允许 |
| 一次性 `run()` | 允许 |
| prepared `plan.run()` | 不允许 |

零分配门禁要通过 allocator hook 或测试计数器验证，不能只靠代码审查。

### 13.3 地址重叠

MVP 默认要求输入、输出和 workspace 的可访问范围互不重叠。只有某个 operation 或
Recipe 明确声明并测试了 in-place 安全时才允许别名。可验证的重叠在运行前报错；
无法完整验证的外部内存由调用者合同约束。

### 13.4 错误处理

常规 `prepare()` 和 `run()` 遵循 cvh 现有 `cvh::Exception` 错误风格。边缘系统
需要显式状态时提供不抛异常的同构入口：

~~~cpp
cvh::PipelinePlan plan;
cvh::PipelineStatus status = builder.tryPrepare(plan);

status = plan.tryRun(
    input, output, workspace.view(), &run_info);
~~~

两套入口必须共享校验和执行实现，不能形成不同语义。错误至少区分：

- descriptor/shape/layout 不匹配；
- buffer 太小或 stride 非法；
- operation 状态转换不合法；
- backend/ISA 不可用；
- workspace 太小或未对齐；
- 强制融合/内存要求不满足。

## 14. 数值和正确性合同

融合前后必须满足同一公开结果合同。不能为了减少访存而随意改变：

- resize 的坐标映射、边界和 rounding；
- YUV matrix、range、chroma sampling；
- saturate cast 和量化舍入；
- normalize 的通道顺序和运算精度；
- NaN/Inf、空输入和非法参数行为；
- ROI、step、非连续内存和尾部处理。

首选目标是与 cvh 对应基础算子一致，并在支持面内通过 sibling OpenCV differential
validation。若 Recipe 有意采用模型生态中的另一套明确语义，必须用不同名字和
独立合同，不能伪装成 OpenCV 等价操作。

优化器的安全定义不是“数学上看起来相同”，而是“在冻结的 dtype、舍入、溢出和
tolerance 合同下通过差分验证”。

## 15. 当前代码布局

~~~text
include/cvh/pipeline/
  pipeline.h              # public umbrella
  builder.h               # fluent builder
  plan.h                  # immutable prepared plan
  types.h                 # descriptor、ColorSpec 和 letterbox transform
  views.h                 # packed/YUV image 与连续 tensor borrowed views
  workspace.h             # workspace owner/view
  operations.h            # operation descriptors
  info.h                   # PipelineInfo / PipelineRunInfo
  detail/
    ir.hpp
    planner.hpp
    fusion_rules.hpp
    quantize.hpp
    scalar_stage_executor.hpp
    scalar_model_input_fused.hpp
    scalar_quantized_model_input_fused.hpp
    scalar_yuv_model_input_fused.hpp
    scalar_yuv_quantized_model_input_fused.hpp
    neon_model_input_fused.hpp

include/cvh/recipes/
  model_input.h
~~~

Pipeline 继续是 header-only，不新增二进制 runtime 依赖。专用 ISA 路径必须保留
可靠 scalar fallback，并满足 ODR、安装消费和非目标平台编译门禁。

## 16. MVP 和阶段计划

### 16.1 P1 v1 Supported 支持面

P1 关闭后公开承诺一个窄而完整的模型输入闭环：

~~~text
CPU-accessible camera/image buffer
  -> optional color
  -> resize/letterbox
  -> normalize
  -> optional per-tensor quantize
  -> layout/store
  -> model tensor
~~~

| 维度 | Supported v1 合同 |
| --- | --- |
| packed 输入 | U8 BGR8/RGB8；`cvh::Mat` 或单 plane borrowed view；允许 row padding、ROI 和 unaligned data |
| YUV 输入 | 偶数宽高 U8 NV12/NV21 双 plane borrowed view；BT.601/709/2020、Limited/Full、Center/Left 均由显式 ColorSpec 决定 |
| 几何 | 固定正尺寸 Nearest/Linear resize，或本文件冻结 rounding/padding/transform 的单次 letterbox |
| 数值 | 1 或 3 通道 finite mean/stddev，stddev 非零；可选 per-tensor U8/S8 quantize |
| 输出 | batch=1、3 通道、连续 F32/U8/S8 NCHW/NHWC `cvh::Mat` 或 borrowed tensor view |
| Recipe 顺序 | `color? -> resize/letterbox -> normalize -> quantize? -> layout/store` |
| 内存属性 | Recipe 为 1 execution group、0 完整中间图、0-byte workspace；prepared run 0 heap allocation |
| backend | 全支持矩阵有 scalar direct-store；仅下述窄 predicate 可进入 ARM NEON，其他组合可靠回退 scalar |

ARM NEON 的 Supported predicate 只覆盖 packed U8 输入、F32 NCHW 输出、Nearest、输入
宽度为 resize/letterbox content 宽度的 2 倍、content 宽度至少 8 且至少 256 pixels。
candidate、actual route 和 observed ISA 必须同时可查询，不能因为运行在 ARM 上就宣称
使用了 NEON。

普通 ordered Pipeline 仍支持已冻结的 packed Gray/BGR/RGB U8/F32 scalar staged
组合；合法但不满足上述 model-input fusion predicate 的链保持顺序语义并使用 staged
fallback。双 plane YUV 只在上述 canonical fused 链/Recipe 中承诺，不提供虚构的
multi-plane `cvh::Mat` staged 表达。

per-channel quantize、batch>1、非连续 tensor 输出、动态 shape、任意 DAG、设备内存
import、JIT、模型推理、多输出和模型特定 decoder 继续是 Proposed 或后续阶段内容。

### 16.2 落地顺序

#### P0：语义骨架

- fluent builder、ordered IR 和明确的 `run/prepare`；
- Image/Tensor descriptor 与 borrowed view helper；
- 类型/shape 推导、错误诊断；
- 顺序 scalar reference executor；
- staged workspace planner；
- `explain()` 和随机合法链差分测试。

#### P1：模型输入融合

逐批实施、predicate、数值合同和验收证据见
[Pipeline P1 实施计划](pipeline-p1-implementation-plan.md)。

- color + resize/letterbox + normalize/quantize + layout/store；
- scalar fused path 与 ARM NEON；
- `modelInput` Recipe；
- zero-allocation、no-full-frame-intermediate 和实际 ISA 观测门禁。

#### P2：适配与部署

- OpenCV、V4L2、ROS 2 和至少一个推理运行时示例；
- 零拷贝生命周期、adapter 错误和设备能力边界；
- rectify/remap 等需要相机标定参数的几何能力。

#### P3：机器人后处理

- threshold、argmax、morphology、connected components；
- tracking frame 和 segmentation mask Recipe；
- neighborhood/global execution group 和 workspace 复用。

#### P4：多输出和平台扩展

- 共享前缀、多输出 slot/bindings；
- x86 SIMD 和更多 ARM 特化；
- 经需求验证后的 DMA-BUF/设备内存 adapter。

每个阶段都必须保持前一阶段的公开语义；不能以未来融合为理由推迟当前正确性兜底。

## 17. 测试和性能门禁

### 17.1 正确性

- 为每个 operation 建立基础算子或独立公式 reference；
- 生成符合类型状态机的随机合法链，比较顺序 reference 与优化 plan；
- 生成非法链，检查 `prepare()` 的 stage 定位和错误类别；
- 覆盖空输入、奇偶尺寸、极小图、尾部、unaligned、ROI 和非连续 stride；
- 覆盖 NV12/NV21 的 matrix/range/chroma 组合；
- 覆盖优化 on/off、forced dispatch、scalar/NEON 和 fallback；
- 检查一次性路径与 prepared 路径结果一致；
- 检查同一 plan 的并发执行不共享可变状态；
- 对支持面进行 sibling OpenCV differential validation；
- 运行 header、ODR、install consumer、ASan/UBSan 和非目标平台编译门禁。

随机测试解决“用户组合顺序不可枚举”的问题：生成器只产生语法合法的链，顺序
executor 给出基准结果，任何融合计划都必须与它一致。

### 17.2 内存合同

- allocator hook 证明 prepared `run()` 为零堆分配；
- canary/guard page 检查输入、输出和 workspace 越界；
- `PipelineInfo` 声明的 workspace 大小和对齐必须足够；
- Recipe 逐项验证完整中间态数量和最终 store；
- 低 workspace、错误对齐和重叠 buffer 必须稳定失败。

### 17.3 性能

性能结论使用同机、同输入、同线程数、Release、多轮稳定采样：

- 分别报告端到端 Pipeline 与逐算子基线；
- 报告实际 dispatch 和 observed ISA，不能从平台推断；
- 同时报告耗时、workspace、完整中间态数量和估算内存流量；
- 在目标 ARM 边缘设备上验证，不只在桌面开发机上验证；
- Recipe 的性能门槛绑定明确输入 predicate，不能外推到任意链；
- benchmark 可以链接 sibling OpenCV 作参考，产品 target 不得因此获得 OpenCV
  二进制依赖。

## 18. 进入 Supported 的完成条件

一个 Pipeline 能力只有满足以下条件才能从 Proposed 进入 Supported：

- 公开 header、安装包和 `cvh::headers` 消费路径完整；
- fluent API、顺序语义、错误和数值合同冻结；
- 一次性和 prepared 示例可以编译运行；
- 通用 scalar fallback 覆盖全部已声明合法组合；
- 随机链、融合差分、ROI/stride/tail 和 sanitizer 测试通过；
- prepared `run()` 的零分配、并发和 workspace 合同通过；
- `info/explain/run_info` 与实际执行一致；
- Recipe 在目标设备上达到记录的正确性、内存和性能门槛；
- 文档不使用“每个 byte 只读一次”等无法证明的表述；
- 不引入 OpenCV、摄像头 SDK 或推理 runtime 的产品二进制依赖。

## 19. 最终决策摘要

1. 对外主 API 是 `cvh::pipe(input, output).op(...).run()`。
2. 链的书写顺序定义语义；优化器只做已证明安全的融合和等价变换。
3. 合法但未融合的链使用 workspace staged fallback，保证正确。
4. 非法状态转换在准备阶段明确失败，不猜测用户意图。
5. 实时用户 `prepare()` 一次，之后用外部 workspace 零分配运行。
6. Tensor/View 是高级内存合同，不占据首屏体验。
7. Pipeline 不依赖或控制摄像头，只消费宿主提供的 CPU 可访问 buffer。
8. 通用 Pipeline 尽力优化；Recipe 和强制策略提供确定的内存/性能属性。
9. “一次访存”改写为可验证的 execution group、完整中间态、直接 store 和逐帧
   分配合同。
10. 首个产品闭环只做模型输入初始化，先建立可信 reference，再做 ARM 融合和
    机器人后处理、多输出扩展。
