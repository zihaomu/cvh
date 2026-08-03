# CPU 加速配置与 Dispatch 重构方案

状态：已实施
日期：2026-08-02

## 实施反馈

| 里程碑 | 状态 | 实时反馈 |
| --- | --- | --- |
| M1：统一配置入口 | 已完成 | 已新增唯一公开开关 `CVH_ENABLE_OPTIMIZATION`，UI 判断全部迁移到 `CVH_DETAIL_HAVE_OPENCV_UI`；已删除四个旧公开宏的定义和 CMake 传播，并停止手工定义 `CV_NEON`。AppleClang/AArch64 下完成 4 个核心 smoke/unit target 构建与运行，4/4 通过。 |
| M2：收敛能力检测 | 已完成 | 已将底层头重命名为 `isa_intrinsics.hpp`，编译结果统一为 `CVH_DETAIL_HAVE_NEON_KERNEL` / `CVH_DETAIL_HAVE_AVX2_KERNEL`；运行时接口统一为 `neon_runtime_available()` / `avx2_fma_runtime_available()`，删除全部 `_AUTO_ENABLED` 状态并保留 AVX2/FMA 的 CPUID、OSXSAVE、XCR0 检查及函数级 target attribute。优化开启的 core/GEMM 测试 2/2 通过，优化关闭的 scalar smoke 2/2 通过。 |
| M3：统一 Dispatch | 已完成 | 已将 dispatch tag 收敛为 `neon`、`avx2`、`opencv_ui`、`scalar`；`gemm_native.hpp`、namespace、packed metadata、专项测试文件及 target 均迁移到 `isa` 命名。新增统一 `opencv_ui_allowed()` 策略，保证 UI 只在 `Auto` / `OpenCVUIOnly` 中可选，forced ISA 不会误落入 UI。Core、Imgproc、GEMM ISA 三组测试 3/3 通过。 |
| M4：收敛 Target 和 CI | 已完成 | 已删除 `cvh::headers_fast` 及其安装导出、smoke 和消费契约，所有计算型测试/benchmark 改用 `cvh::headers`。Compare target/实现名改为 `cvh_benchmark_opencv_compare_ui` / `cvh_ui`，启动时强制 `OpenCVUIOnly`，拒绝 `neon`/`avx2` tag，并对明确要求 UI 的 resize/cvtColor case 校验真实 tag。已删除三个历史构建选项的 CMake 拒绝兼容层及对应契约测试，CMake 仅呈现当前有效配置。全量 CTest 20/20、安装消费契约 12/12、hosted CI 脚本（Core 209、Imgproc 187）、优化关闭 smoke 3/3 均通过；真实 OpenCV 4.14 full compare 生成 344 行，0 个专用 ISA tag，9 个 UI-required case 全部为 `opencv_ui`。 |

本节随实施进度实时更新；最终验证结果记录在各里程碑反馈中。

## 1. 决策摘要

cvh 不再把 OpenCV Universal Intrinsics、NEON 和 AVX2 暴露为多套可任意
组合的产品模式。配置层只表达“是否启用已接纳优化”，ISA 能力由编译器和
目标架构自动检测，具体实现由运行时 dispatch 选择。

目标公开配置只保留：

```cpp
CVH_ENABLE_OPTIMIZATION
```

默认值为 `1`。关闭后只保留 scalar 路径。

以下宏不进入最终公开配置：

```cpp
CVH_ENABLE_OPENCV_INTRIN
CVH_ENABLE_DIRECT_INTRINSICS
CVH_ENABLE_DIRECT_NEON
CVH_ENABLE_DIRECT_AVX2
```

`cvh::headers` 是唯一计算入口，默认包含所有已经通过 correctness 和
benchmark 门槛的优化。`cvh::headers_fast` 已在 M4 删除。

## 2. 目标

- 用户只决定是否启用优化，不直接配置具体实现。
- NEON、AVX2 和 OpenCV UI 的可用性由编译条件自动得出。
- 编译能力、运行时硬件能力和实际 dispatch 结果相互独立。
- scalar 始终作为 correctness fallback 存在。
- UI-only CI 通过运行时 dispatch 明确选择 UI，而不是通过 CMake target
  名称间接推断。
- 所有配置在多翻译单元中保持一致，避免 header-only ODR 风险。

## 3. OpenCV 的分层方式

OpenCV 将 CPU 优化分为四层：

| 层级 | OpenCV 机制 | 职责 |
| --- | --- | --- |
| 用户策略 | `CV_ENABLE_INTRINSICS`、`CV_DISABLE_OPTIMIZATION` | 控制优化总策略 |
| 构建配置 | `CPU_BASELINE`、`CPU_DISPATCH` | 指定最低 ISA 和额外 dispatch ISA |
| 编译结果 | `CV_CPU_COMPILE_*`、`CV_SIMD*`、`CV_NEON`、`CV_AVX2` | 描述编译器最终生成的能力 |
| 运行时选择 | `checkHardwareSupport()`、`CV_CPU_DISPATCH()` | 根据当前 CPU 选择实现 |

OpenCV 已不推荐使用独立的 `ENABLE_AVX2`、`ENABLE_NEON` 等用户开关，而是
通过 baseline、dispatch 和能力检测生成内部宏：

- [OpenCV CPU optimization build options](https://github.com/opencv/opencv/wiki/CPU-optimizations-build-options)
- [OpenCV configuration options](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)
- [OpenCVCompilerOptimizations.cmake](https://github.com/opencv/opencv/blob/4.x/cmake/OpenCVCompilerOptimizations.cmake)
- [cv_cpu_dispatch.h](https://github.com/opencv/opencv/blob/4.x/modules/core/include/opencv2/core/cv_cpu_dispatch.h)

cvh 是 header-only，不能直接复制 OpenCV 通过多个 `.cpp` 生成多份 ISA
实现的机制，但应沿用其职责分层。

## 4. 当前问题

### 4.1 用户配置与实现选择混层

`CVH_ENABLE_OPENCV_INTRIN` 和 `CVH_ENABLE_DIRECT_INTRINSICS` 都在选择优化
实现；`CVH_ENABLE_DIRECT_NEON` 和 `CVH_ENABLE_DIRECT_AVX2` 又允许用户
干预具体 ISA。四个宏共同决定同一条 dispatch 链，组合状态过多。

### 4.2 ISA 检测重复

vendored OpenCV `cvdef.h` 已根据编译器预定义宏产生 `CV_NEON`、`CV_AVX2`
和 `CV_SIMD*`。CMake 不应再为 `cvh::headers_fast` 手工设置 `CV_NEON=1`。

cvh 自己的 NEON/AVX2 kernel 只需要内部编译结果，不需要公开的逐 ISA
开关。

### 4.3 编译能力与运行时选择混层

当前 `*_COMPILED`、`*_AUTO_ENABLED` 和 `DispatchMode` 同时参与选择。
其中：

- `COMPILED` 应只表示当前编译单元是否存在该实现；
- CPU feature check 应只表示当前机器能否安全执行；
- `DispatchMode` 应只决定本次调用允许选择哪些实现。

三者不应互相替代。

### 4.4 UI-only benchmark 没有被严格保证

当前 OpenCV compare target 链接 `cvh::headers_fast`，而 GEMM 在 `Auto` 模式
下会优先尝试直接 NEON/AVX2 kernel。因此名为 UI-only 的 CI 可能测到非 UI
实现。

记录 `dispatch_path` 只能观察结果，不能代替执行路径约束。

## 5. 目标配置模型

### 5.1 唯一公开优化宏

```cpp
#ifndef CVH_ENABLE_OPTIMIZATION
#define CVH_ENABLE_OPTIMIZATION 1
#endif
```

语义：

| 值 | 行为 |
| --- | --- |
| `1` | 编译并自动选择所有已接纳的可用优化，保留 scalar fallback |
| `0` | 不编译 OpenCV UI 和 cvh 专用 ISA kernel，只使用 scalar |

该宏必须作为整个程序的一致配置，不能在不同 translation unit 中使用不同值。

### 5.2 内部编译能力

内部能力宏由架构、编译器和总开关推导，不允许消费者直接配置：

```cpp
CVH_DETAIL_HAVE_OPENCV_UI
CVH_DETAIL_HAVE_NEON_KERNEL
CVH_DETAIL_HAVE_AVX2_KERNEL
```

含义：

- `CVH_DETAIL_HAVE_OPENCV_UI`：Universal Intrinsics 头和有效 SIMD/scalar
  facade 可用于当前实现；
- `CVH_DETAIL_HAVE_NEON_KERNEL`：当前编译单元可以生成 cvh NEON kernel；
- `CVH_DETAIL_HAVE_AVX2_KERNEL`：当前编译单元可以生成 cvh AVX2/FMA
  kernel。

这些宏是检测结果，不是产品模式。底层应优先复用 `CV_NEON`、`CV_AVX2`、
`CV_SIMD*` 及编译器预定义宏，避免重复维护架构判断。

### 5.3 运行时能力

运行时接口负责判断编译出的实现能否在当前 CPU 安全执行：

```cpp
bool neon_runtime_available();
bool avx2_fma_runtime_available();
```

AArch64 baseline 可以直接确认 NEON。x86 AVX2/FMA 必须保留 CPUID、OSXSAVE
和 XCR0 检查；不能仅因为代码成功编译就执行。

### 5.4 Dispatch 模型

```text
Auto
  → 已接纳的 cvh 专用 ISA kernel
  → OpenCV Universal Intrinsics
  → scalar

OpenCVUIOnly
  → OpenCV Universal Intrinsics
  → scalar fallback

ScalarOnly
  → scalar

NeonOnly / Avx2Only
  → 仅用于内部 correctness、benchmark 和 forced-path 测试
```

`OpenCVUIOnly`、`NeonOnly` 和 `Avx2Only` 是测试/诊断控制，不是构建模式，
也不应改变公开 target。

## 6. CMake Target 模型

最终公开 target：

```cmake
cvh::headers
cvh::highgui
```

职责：

- `cvh::headers`：默认优化计算入口，所有实现仍为 header-only；
- `cvh::highgui`：继承 `cvh::headers`，只增加系统 GUI 链接依赖。

`cvh::headers_fast` 的实施结果：

1. 停止传播任何独立优化宏；
2. 在迁移过程中退化为与 `cvh::headers` 等价的兼容 target；
3. 更新消费者、测试、benchmark 和文档；
4. 在 `0.1` 正式发布前完成删除，避免形成第二套产品模式。

未经正式接纳的专用 kernel 应由内部测试 target 覆盖，不能通过公共 target
暴露为长期配置。

## 7. UI-only CI 与 Benchmark

OpenCV compare 应链接 `cvh::headers`，并在执行 benchmark 前显式设置：

```cpp
cvh::cpu::set_dispatch_mode(
    cvh::cpu::DispatchMode::OpenCVUIOnly);
```

每个要求 UI 的 case 必须验证：

```text
dispatch_path == opencv_ui
```

不满足时 benchmark 直接失败，不能把 direct ISA 或 scalar 结果记录为 UI
对比数据。

Hosted CI 继续只运行 UI-enabled correctness 和 UI-only OpenCV compare；
scalar、NEON-only 和 AVX2-only 用于本地或架构专项 correctness，不生成公开
性能报告。

## 8. 实施步骤

### M1：统一配置入口

- 新增 `CVH_ENABLE_OPTIMIZATION=1`。
- 将所有 `CVH_ENABLE_OPENCV_INTRIN` 判断迁移到内部能力判断。
- 删除 `CVH_ENABLE_DIRECT_INTRINSICS`、`CVH_ENABLE_DIRECT_NEON` 和
  `CVH_ENABLE_DIRECT_AVX2`。
- 删除 CMake 手工传播的 `CV_NEON=1`。

### M2：收敛能力检测

- 建立统一的内部 CPU capability header。
- 区分 compile-time capability 与 runtime capability。
- 删除 `_AUTO_ENABLED` 一类重复状态。
- 保留 x86 AVX2/FMA 安全检测和函数级 target attribute。

### M3：统一 Dispatch

- 所有算子遵循 `专用 ISA → OpenCV UI → scalar` 的统一顺序。
- forced dispatch 只在测试和 benchmark 边界使用。
- dispatch tag 使用 `neon`、`avx2`、`opencv_ui`、`scalar`，不再使用
  `native` 表述。

### M4：收敛 Target 和 CI

- OpenCV compare 改为链接 `cvh::headers`。
- UI-only compare 强制 `OpenCVUIOnly` 并校验 dispatch tag。
- 将 `cvh::headers_fast` 迁移为临时兼容别名后删除。
- 更新 README、设计文档、安装契约和 CI expectation。

## 9. 验收门槛

- 公开配置中只有 `CVH_ENABLE_OPTIMIZATION` 一个 CPU 优化宏。
- 有效代码不存在 `CVH_ENABLE_OPENCV_INTRIN` 和 `CVH_ENABLE_DIRECT_*`。
- CMake 不手工定义 `CV_NEON`、`CV_AVX2` 或其它 OpenCV 能力结果。
- `cvh::headers` 在 AArch64、generic x86 和 AVX2-capable x86 上均可构建。
- UI-only benchmark 的所有可向量化 case 均报告 `opencv_ui`。
- forced NEON/AVX2 correctness 测试不会进入 hosted UI-only compare 报告。
- `CVH_ENABLE_OPTIMIZATION=0` 的本地 scalar smoke 通过。
- 安装消费、多翻译单元 ODR、ASan/UBSan 和 `git diff --check` 全部通过。
