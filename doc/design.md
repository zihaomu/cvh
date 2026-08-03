# cvh（cv-header-only）设计文档

## 当前定位

`cvh`（cv-header-only）是一个独立品牌的纯 header-only C++ 计算机视觉库。
项目以 OpenCV 作为 API 风格和行为兼容参照，在不引入必需库构建步骤的
前提下，提供常用 `Mat`、`core`、`imgproc`、`imgcodecs` 能力和可选
`highgui` 显示子集，并为 AI
vision preprocessing/postprocessing 的热点路径提供可验证的 header-only
加速。项目不是 OpenCV 的发行版，也不以完整替代 OpenCV 为目标。

公开产品面包含两个 CMake INTERFACE target：

```cmake
cvh::headers
cvh::highgui
```

- `cvh::headers`：唯一计算入口，默认启用所有已接纳优化并保留 scalar fallback。
- `cvh::highgui`：可选窗口与事件循环入口，依赖平台系统 GUI 库，但不生成 cvh 二进制。

P7.1 已删除旧编译型 HighGUI `.cpp/.mm` 后端。HighGUI 随后以
C++17 inline header 重新实现；scalar、OpenCV UI、direct NEON 和 direct
AVX2 也均为 header 内联实现。

## 核心目标

- 提供接近 OpenCV 风格的基础数据结构和常用算子。
- 保证用户仅包含 headers 或链接 interface target 即可使用。
- 保持公开依赖、宏开关、安装导出和文档叙事一致。
- 先建立 correctness contract，再引入 benchmark-gated SIMD fast-path。
- 用 benchmark 决定 fast-path 是否进入 `cvh::headers` 默认 dispatch。

## 非目标

- 不追求完整复现 OpenCV 全模块。
- 不承诺所有算子、所有类型、所有 flag 与 OpenCV 完全一致。
- 不承诺完整复现 OpenCV HighGUI；当前只提供五个基础窗口 API。
- 不引入项目 `.cpp` 实现。
- 不把 xsimd 作为默认或推荐性能路线。
- 不承诺所有路径都快于 OpenCV；只承诺 benchmark-gated 的热点优化。

## Header-only Contract

公开 API 必须满足：

- `include/` 内可独立完成编译，仓库不存在产品实现 `src/`。
- `cvh::headers` 只传播 headers、compile features 和唯一优化策略宏。
- `cvh::highgui` 只传播 inline headers 和平台系统 GUI 链接依赖，不生成 cvh 库文件。
- 每个标记为 Supported 的算子必须有 header-only correctness test。
- 没有 header-only 实现或链接不过的 API 必须标记为 WIP 或移出公开入口。

CPU 优化的唯一公开策略宏是 `CVH_ENABLE_OPTIMIZATION`，默认值为 `1`；
设为 `0` 时只编译 scalar 路径。OpenCV UI、NEON 和 AVX2 的编译能力均由
内部检测结果表示，消费者不配置逐 ISA 宏。

## Public Targets

### `cvh::headers`

默认入口，适合所有需要稳定 header-only 行为的用户。

要求：

- 默认启用 OpenCV Universal Intrinsics。
- 自动编译当前工具链支持的已接纳 NEON/AVX2 kernel，并在运行时安全选择。
- 不默认启用 xsimd。
- 不要求 OpenCV 库或其它二进制依赖。
- 保留 scalar fallback 和标准 C++ 实现作为 correctness 基线。

### `cvh::highgui`

可选显示入口，继承 `cvh::headers`。

要求：

- 公开 `namedWindow`、`imshow`、`waitKey`、`destroyWindow` 和
  `destroyAllWindows`；
- macOS 使用纯 C++ Objective-C Runtime 调用 AppKit，Windows 使用 Win32，
  Linux desktop 使用 X11；
- 不进入默认聚合头，避免计算型用户被迫引入 GUI 依赖；
- 只链接系统 GUI framework/library，不生成 cvh 编译产物；
- 使用 C++17 inline 状态保证多翻译单元 ODR 安全。

当前 accepted fast-path：

- `cvtColor`：`CV_8UC3 BGR2GRAY/RGB2GRAY`
- `resize`：exact 2x `CV_8UC1 INTER_LINEAR`

## Module Responsibilities

### `core`

负责 `Mat`、基础类型、错误处理、类型/channel 宏、ROI、copy/clone/convert 等基础能力。

`core` 的首要职责是支撑 header-only correctness，而不是承接所有 AI kernel 或历史算子。

### `imgproc`

负责 OpenCV-style 图像处理算子，例如：

- `resize`
- `cvtColor`
- `threshold`
- `LUT`
- `copyMakeBorder`
- `filter2D`
- `sepFilter2D`
- `boxFilter` / `blur`
- `GaussianBlur`
- `Sobel`
- `Canny`
- `erode` / `dilate` / `morphologyEx`
- `warpAffine`

每个算子必须明确支持的 depth、channel、flag、border 和错误行为。

### `imgcodecs`

提供最小读写闭环：

- `imread`
- `imwrite`

当前读写能力基于 vendored stb，目标是满足“读图 -> 处理 -> 写图”的端到端 header-only 使用链路。

### `highgui`

提供最小显示闭环：

- `namedWindow`
- `imshow`
- `waitKey`
- `destroyWindow`
- `destroyAllWindows`

当前 `imshow` 输入限制为二维 `CV_8U` C1/C3/C4。窗口和事件调用应由应用
UI 线程驱动；Linux 无 X11、无桌面会话或其它不支持的平台会给出明确错误。

## SIMD Strategy

项目采用三条规则：

- scalar fallback 是所有公开算子的 correctness 基线。
- OpenCV Universal Intrinsics 是默认内部 SIMD dialect。业务 SIMD kernel 可以直接使用 `cv::v_*`、`cv::VTraits`、`CV_SIMD`、`CV_SIMD_WIDTH` 和 `vx_*`，但这些类型不构成 `cvh` 用户公开 API。
- direct platform intrinsics 只能在 benchmark 证明 OpenCV Universal Intrinsics 不足时进入候选。

xsimd 不再作为图像 kernel 的主性能路线。P5.3 已移除 public adapter surface、legacy `.cpp` xsimd kernel、内部 `XSimdOnly` dispatch、测试入口和 vendor 目录；默认 header-only target、安装导出和 header-only CI 都不能依赖它。

`cvh::detail::simd` 二次 facade 不再作为未来路线。P6 开始，已接受的 OpenCV UI fast path 会迁移到 direct OpenCV UI 写法；scalar fallback 保持为显式 `*_scalar_impl` 或 benchmark helper，而不是伪装成 SIMD backend。

当前 SIMD 平台范围只处理 ARM NEON 和 x86 AVX 系列。RVV 支持放入后续 TODO；SSE header/宏只作为 x86 OpenCV UI/AVX 编译链路的基础条件，不作为当前独立优化路线。

从 OpenCV 迁移新的 SIMD kernel 时，使用
`doc/opencv-ui-kernel-migration-checklist.md` 作为评审 checklist；迁移应尽量保留
OpenCV UI 原始表达，只替换 OpenCV runtime/module 依赖。

## Documentation Rules

公开文档必须保持一致：

- 品牌短名统一写作 `cvh`，全称统一写作 `cv-header-only`。
- `OpenCV-style`、`OpenCV-compatible` 或“与 OpenCV API 对齐”只描述 API
  风格、兼容目标或对照基线，不作为项目名称。
- 模块描述应明确 `core`、`imgproc`、`imgcodecs`、`highgui` 是有边界的兼容子集，
  避免暗示全部 OpenCV API 均已实现。
- 第一屏定位必须是 pure header-only。
- 计算用法统一使用 `cvh::headers`，窗口用法显式使用
  `cvh::highgui`。
- 产品实现不得新增 `.cpp`、运行时函数指针 backend 注册表或编译型
  backend；HighGUI 的 inline 窗口状态表不属于 backend dispatch。
- 算子支持状态必须区分 Supported、Supported + fast path、WIP、Out of scope。
- 性能描述必须绑定 benchmark，不写泛化的“整体快于 OpenCV”。

## Completion Criteria

一个公开算子进入 Supported 状态，至少需要：

- header-only 实现存在。
- header-only target 可编译。
- correctness test 覆盖正常路径和关键边界。
- 文档明确输入约束和未支持范围。
- README 支持矩阵能追溯到 `scripts/ci_headers_all.sh` 中的 header-only test/gate。

一个 CPU fast-path 进入默认 dispatch，至少需要：

- scalar fallback 已稳定。
- fast-path 正确性与 scalar 对齐。
- benchmark 证明收益存在。
- 不满足 fast-path 条件时能回退到 scalar fallback。

## 当前结论

项目价值来自“独立、真实可用的纯 header-only 计算机视觉库”和低学习成本的
OpenCV-style API，而不是依附旧仓库名称，也不是 header 和 `.cpp` 扩展的
混合叙事。后续工作应优先收口公开面、补齐 header-only contract，再用 direct
OpenCV UI 迁移和 benchmark gate 扩展内部 SIMD 能力。
