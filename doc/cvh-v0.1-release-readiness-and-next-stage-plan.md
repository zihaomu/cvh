# cvh 0.1 首发收口与下一阶段实施计划

更新时间：2026-07-29

基线版本：`0.1.0`

基线提交：`6fe6748`（`chore: rebrand project as cvh`）

实施状态：P7.1 已于 2026-07-29 落地，等待提交。

## 1. 文档目的

`cvh` 已经完成独立品牌切换，并形成了可工作的纯 header-only
`core`、`imgproc`、`imgcodecs` 子集。下一阶段不应立即追求更多算子数量，
也不应继续无边界扩大 GEMM 实现，而应先把现有能力收敛为一个可安装、
可验证、可发布、可被陌生用户采用的 `0.1.x` 产品。

本计划回答以下问题：

1. `cvh 0.1` 发布前还缺少什么；
2. 哪些工作必须先于 API 扩张和新一轮性能优化；
3. 后续高价值 API 与性能工作如何排序；
4. 每一阶段通过什么客观门槛验收。

## 2. 当前基线

### 2.1 已经具备的能力

- 品牌、CMake 包、命名空间和公开 target 已统一为 `cvh`。
- 公开产品面为：

  ```cmake
  cvh::headers
  cvh::highgui
  ```

- 已有可调用的 `Mat`、Core 基础算子、Imgproc 高频算子和 Imgcodecs
  最小读写闭环。
- `core` 和 `imgproc` 名称级可调用覆盖为 `107 / 220`，没有
  declaration-only API。
- 默认 UI-enabled 配置下，Core 和 Imgproc 没有已知失败或 skip。
- 已建立：
  - 公共头自包含检查；
  - 多翻译单元 ODR smoke；
  - 安装后 CMake consumer 验证；
  - x86 AVX2/FMA correctness；
  - ASan/UBSan；
  - 内部回归与 OpenCV upstream 对比 benchmark。
- GEMM 已具备 scalar/OpenCV UI、direct NEON、direct AVX2 三类执行路径，
  并支持 packed-B 和线程运行时。

### 2.2 当前主要缺口

| 维度 | 当前状态 | 主要问题 |
|---|---|---|
| 产品叙事 | 品牌已统一 | README、设计文档和实现状态仍有少量漂移，特别是 GEMM fast-path 描述 |
| 用户上手 | 有 include/CMake target 示例 | 缺少安装全过程、FetchContent 和真实处理流水线示例 |
| 发布工程 | 有 `VERSION.txt` 和 Apache-2.0 | 缺少 changelog、贡献指南、安全策略、Release 资产验证 |
| CI | Ubuntu/x86 correctness 较完整 | 缺少持续的 macOS ARM64/NEON 和 Windows/MSVC 门禁 |
| 产品边界 | P7.1 已收口 | 编译型 backend、`src/` 和历史 build mode 已移除；HighGUI 以可选 header-only target 恢复 |
| 性能 | 有 benchmark 框架 | 公开报告早于最新 GEMM 整理，缺少分架构稳定基线 |
| API 覆盖 | 高频基础能力已形成 | 仍有 40 个 Core 和 73 个 Imgproc 操作族未支持 |
| Header-only 成本 | 无二进制库依赖 | 尚未监控编译时间、聚合头膨胀和消费者二进制体积 |

## 3. 下一阶段总目标

下一阶段命名为：

> **P7：cvh 0.1 Release Readiness**

核心目标不是扩充算子数量，而是完成以下产品闭环：

```text
公开能力与文档一致
    → 默认入口只包含正式产品能力
    → 新用户可按文档完成接入
    → 三个平台持续验证
    → 发布资产可被独立工程消费
    → 性能报告与当前实现一致
    → 再进入高价值 API 和性能扩张
```

## 4. P7.0：产品事实与文档收口

### 4.1 统一事实来源

建立以下事实来源关系：

| 内容 | 唯一事实来源 | 下游文档 |
|---|---|---|
| 版本 | `VERSION.txt` | README、Release、CMake package |
| 支持算子 | 测试注册 + API coverage | README Operator Status |
| fast-path | dispatch 测试 + benchmark | README Performance、设计文档 |
| 公共 target | CMake install/export contract | README Usage |
| 已知失败/skip | GTest XML + expectations | `test/failing-tests.md` |
| 性能数字 | dated benchmark report + metadata | README 最新性能入口 |

README 不应手工维护无法追溯到测试或 benchmark 的性能承诺。

### 4.2 修正当前漂移

- 将 GEMM 状态改为真实的三类实现：
  - scalar/OpenCV UI；
  - direct NEON；
  - direct AVX2。
- 说明 `cvh::headers` 的 `ISA → OpenCV UI → scalar` GEMM dispatch 顺序。
- 更新设计文档中的 accepted fast-path，不再只列早期的
  `cvtColor` 和 `resize` 两项。
- 重新生成品牌迁移和 GEMM 整理后的最新性能报告。
- 清理文档和脚本中的本机绝对路径，使用仓库相对路径或显式参数。
- 检查历史阶段文档，将“计划”“已落地”“已废弃”状态明确区分。

### 4.3 验收门槛

- 全仓不存在旧项目品牌和旧 CMake 包名。
- README 的每个 Supported/fast-path 声明均能追溯到测试或 benchmark。
- 最新公开性能报告基于当前 `main` 提交。
- 文档示例不包含开发者机器绝对路径。
- 文档链接检查无失效的仓库内链接。

## 5. P7.1：纯 header-only 产品边界收口

### 5.1 已落地的边界决策

P7.1 删除旧编译型 HighGUI backend，并将必要显示能力重建为可选
header-only 模块：

- 从 `cvh/cvh.h` 移除 HighGUI；
- 删除旧 `src/highgui` Cocoa、Win32、X11 和 framebuffer 实现；
- 删除整个产品实现 `src/`；
- 删除 `CVH_BUILD_NATIVE_BACKEND`、`CVH_BUILD_FULL_BACKEND` 和
  `CVH_USE_OPENMP` 构建选项；
- 删除 `cvh::native`、`cvh::full`、`cvh::full_backend` 等历史 target；
- 删除 `CVH_LITE/CVH_NATIVE/CVH_FULL` 构建模式宏；
- 删除旧 HighGUI 示例、编译型测试和对应 CI 入口；
- 新增 `cvh::highgui` INTERFACE target；
- 以 inline AppKit Runtime、Win32 和 X11 backend 恢复
  `namedWindow`、`imshow`、`waitKey`、`destroyWindow` 和
  `destroyAllWindows`；
- 将测试和 benchmark 改为显式 opt-in，默认配置只生成 interface target；
- 将原有最小 pipeline smoke 改为不带 `lite` 历史模式的正式 smoke。

这里删除的是旧编译型 `.cpp/.mm` backend。HighGUI 平台调用和 GEMM
direct NEON/AVX2 都保留为 header 内联路径；HighGUI 的 AppKit/Win32/X11
属于系统链接依赖，不是 cvh 编译产物。

### 5.2 公共头边界

正式入口冻结为：

- `include/cvh/cvh.h`
- `include/cvh/core/*.h`
- `include/cvh/imgproc/imgproc.h`
- `include/cvh/imgcodecs/imgcodecs.h`
- `include/cvh/highgui/highgui.h`（可选，不进入聚合头）

`detail/**`、`simd/**`、`*.inl.h` 和实现型 `*.hpp` 会随 header-only
安装包提供，但不属于源码兼容承诺。平台 intrinsic、workspace 和 forced
dispatch 继续限制在 internal/detail 或测试边界。

正式 CPU 优化配置只保留 `CVH_ENABLE_OPTIMIZATION`，默认值为 `1`。关闭时
只编译 scalar 路径；OpenCV UI、NEON 和 AVX2 均由内部编译能力与运行时
检测决定，不再暴露逐实现宏。

### 5.3 验收门槛

- [x] 默认产品构建不编译任何项目 `.cpp`。
- [x] 安装包只导出两个 INTERFACE target：`cvh::headers` 和
  `cvh::highgui`。
- [x] 聚合头只包含 Core、Imgproc 和 Imgcodecs。
- [x] `src/` 产品实现树已删除。
- [x] 安装契约验证 header-only HighGUI 消费，并拒绝编译型源文件重新进入安装包。
- [x] 公共头独立编译、安装消费和多翻译单元 ODR 测试全部通过。

## 6. P7.2：用户接入与示例闭环

### 6.1 必须补充的接入方式

README 至少提供三种可复制用法：

#### 直接 include

说明所需 include roots、C++ 标准和 scalar/UI 配置方式。

#### CMake install

```bash
cmake -S . -B build \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=OFF
cmake --install build --prefix /path/to/cvh-install
```

```cmake
find_package(cvh CONFIG REQUIRED)
target_link_libraries(app PRIVATE cvh::headers)
```

#### FetchContent

给出固定 tag/commit 的最小 CMake 示例，不推荐跟随浮动 `main`。

### 6.2 正式示例

新增以下独立示例，每个示例都必须使用正式 header-only target：

| 示例 | 覆盖能力 | 输出 |
|---|---|---|
| `image_pipeline` | `imread → resize → cvtColor → normalize → imwrite` | 输出图片和尺寸信息 |
| `mat_core` | `Mat`、ROI、clone、convert、逐元素运算、reduce | 控制台校验 |
| `gemm` | FP32 GEMM、packed-B、多次复用 | 结果摘要与 dispatch 信息 |
| `geometry` | `warpAffine` 或 `warpPerspective` | 输出图片 |

示例要求：

- 不依赖 OpenCV 库；
- 不依赖 `src/`；
- 使用公开 API，不 include `detail/**`；
- 纳入 CI 编译和最小运行 smoke；
- 输入缺失时可生成确定性测试数据。

### 6.3 发布配套文件

新增：

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- 可选 `CODE_OF_CONDUCT.md`
- Release checklist

`CHANGELOG.md` 至少记录：

- `cvh` 品牌与 CMake 包迁移；
- `0.1.0` 稳定模块；
- 明确的 breaking changes；
- 已知限制；
- 支持的平台和编译器。

### 6.4 验收门槛

- 新用户可仅按 README 在全新目录完成 direct include、install package 和
  FetchContent 三种接入。
- 四个正式示例在 CI 中编译，其中无文件依赖的示例可运行。
- Release 源码包解压后可独立配置、安装和被外部 consumer 使用。
- README 不再以 Out-of-scope HighGUI 作为唯一示例。

## 7. P7.3：跨平台 CI 与质量门禁

### 7.1 最小持续验证矩阵

| 平台 | 编译器/ISA | 必须验证 |
|---|---|---|
| Linux x86-64 | GCC、Clang、AVX2/FMA | headers、tests、install consumer、ASan/UBSan |
| macOS ARM64 | AppleClang、NEON | headers、tests、GEMM direct-ISA dispatch、install consumer |
| Windows x64 | MSVC | headers、tests、CMake install consumer、AVX2 编译路径 |

附加配置：

- C++17：最低标准；
- C++20：兼容 smoke；
- `cvh::headers`；
- `CVH_ENABLE_OPTIMIZATION=OFF` scalar compile/runtime smoke。

### 7.2 NEON 专项

macOS ARM64 或 Linux AArch64 门禁至少覆盖：

- direct NEON feature detection；
- GEMM NN/NT、尾块、packed-B、FP16 weight；
- scalar/UI/NEON 数值对齐；
- 多线程和串行阈值；
- AddressSanitizer 可行配置下的边界验证。

### 7.3 Header-only 特有指标

建立非功能回归指标：

| 指标 | 建议测量对象 |
|---|---|
| 聚合头编译时间 | 单 TU `#include <cvh/cvh.h>` |
| 模块头编译时间 | core/imgproc/imgcodecs 独立 TU |
| 预处理后体积 | 聚合头和模块头 |
| 最小 consumer 二进制大小 | scalar 与 fast target |
| 警告数量 | GCC/Clang/MSVC 严格警告配置 |

初期可 log-only；形成稳定基线后再设置宽松退化阈值。

### 7.4 验收门槛

- Linux、macOS ARM64、Windows 三个平台的 required CI 均为绿色。
- NEON 和 AVX2 不再只有单侧持续验证。
- 安装消费测试在三平台运行。
- sanitizer 没有新增 suppressions 才能通过。
- 所有正式示例在三平台至少完成编译。

## 8. P7.4：性能基线重建

### 8.1 基线原则

性能报告必须记录：

- `cvh` commit；
- OpenCV commit 和加速配置；
- CPU、OS、编译器、build type；
- dispatch path；
- requested/chosen threads；
- warmup、iterations、repeats；
- median、p90 和吞吐；
- cold/steady-state 区分；
- 是否包含 allocation、packing 和 output create。

### 8.2 GEMM 基线

GEMM 至少覆盖：

| 类别 | 代表 shape |
|---|---|
| 小方阵 | `32³`、`64³`、`128³` |
| 中大方阵 | `256³`、`512³` |
| Skinny | `32×512×64` |
| Wide | `256×32×256` |
| Tail | M/N/K 非 micro-kernel 整数倍 |
| Batched | 多 batch 与 broadcast-B |
| Packed-B | one-shot 与 pack-once |
| Weight type | FP32 与 FP16 weight |

每类同时记录：

- scalar；
- OpenCV UI；
- direct NEON 或 AVX2；
- Auto；
- 1T 与项目默认线程数；
- OpenCV CPU-only；
- 平台默认 OpenCV/Accelerate 仅作为外部上限参考。

### 8.3 非 GEMM 优先热点

下一轮性能优先级：

1. 通用 `resize`，而不是只覆盖 exact 2x；
2. RGB/BGR/GRAY 与 YUV420/422/444 热路径；
3. `add/subtract/multiply/divide/compare`；
4. `normalize`、reduce 与 channel routing；
5. `remap/warpAffine/warpPerspective`；
6. 常用滤波和形态学的非 3x3/尾块路径。

### 8.4 性能合入门槛

- 新 fast-path 必须有 forced-path correctness 测试。
- Auto dispatch 必须证明选择正确路径。
- 主 anchor shape 不允许超过既定退化阈值。
- 不能仅通过改变 benchmark 采样、线程数或工作范围得到“提速”。
- 对小任务必须证明 dispatch/线程启动成本没有吞噬收益。
- 没有稳定收益的专项实现应删除，而不是永久留在 dispatch 树中。

## 9. P8：高价值 API 扩张

P8 只在 P7 发布门槛完成后启动。目标不是追求 `220 / 220`，而是补齐
真实计算机视觉流水线中的高价值缺口。

### 9.1 推荐 Core 顺序

#### 第一批：小型线性代数与变换

- `setIdentity`
- `trace`
- `determinant`
- `invert`
- `solve`
- `transform`
- `perspectiveTransform`

这些能力可复用现有 GEMM、归约和几何基础，用户价值高于直接进入 PCA/SVD
全家族。

#### 第二批：坐标、排序与距离

- `magnitude`
- `phase`
- `cartToPolar`
- `polarToCart`
- `sort`
- `sortIdx`
- `PSNR`
- `batchDistance`

#### 暂缓

- 完整 PCA/SVD/eigen 家族；
- DFT/DCT 全类型和全 flag 矩阵；
- `kmeans`；
- OpenCV `InputArray/OutputArray` 对象模型。

### 9.2 推荐 Imgproc 顺序

#### 第一批：区域与轮廓

- `connectedComponents`
- `connectedComponentsWithStats`
- `findContours`
- `boundingRect`
- `contourArea`
- `moments`
- `minAreaRect`

#### 第二批：直方图、匹配与角点

- `calcHist`
- `calcBackProject`
- `compareHist`
- `matchTemplate`
- `cornerHarris`
- `cornerMinEigenVal`
- `goodFeaturesToTrack`

#### 暂缓

- 文字与完整绘制系统；
- Generalized Hough/LineSegmentDetector 对象接口；
- `grabCut`、`watershed`、Mean Shift；
- 为覆盖率数字而实现的低频长尾 API。

### 9.3 AI 预处理扩展

OpenCV-style 兼容 API 之外，可以增加少量明确的 `cvh` 扩展：

- HWC → CHW；
- CHW → HWC；
- tensor packing；
- fused resize + color + normalize；
- FP32/FP16 输出布局转换。

扩展 API 必须满足：

- 与 OpenCV 兼容 API 明确分区；
- 不伪装成 OpenCV 已有接口；
- 有 scalar 正确性基线；
- 有 batch、ROI、channel、tail 测试；
- 融合版本必须证明优于分步调用。

## 10. 分阶段交付顺序

| 顺序 | 阶段 | 主要交付 | 是否阻塞后续 |
|---:|---|---|---|
| 1 | P7.0 | 文档、状态与 benchmark 事实统一 | 是 |
| 2 | P7.1 | 编译型 backend 边界收口、公共头冻结 | 已完成 |
| 3 | P7.2 | 安装、FetchContent、示例、Release 文件 | 是 |
| 4 | P7.3 | Linux/macOS ARM64/Windows CI | 是 |
| 5 | P7.4 | 当前实现的分架构性能基线 | 是 |
| 6 | `0.1.0` Release | tag、源码包、校验和、发布说明 | 是 |
| 7 | P8 | 高价值 API 与 AI 预处理扩展 | 否 |
| 8 | P9 | benchmark 驱动的下一轮 SIMD 优化 | 否 |

建议每一阶段独立提交或独立 PR，避免重新形成无法归因的大 patch。

## 11. `cvh 0.1.0` 发布完成定义

只有同时满足以下条件，才发布 `0.1.0`：

### 品牌与文档

- [ ] 项目名、包名、namespace、target 和仓库链接统一为 `cvh`。
- [ ] README、设计文档、API coverage 和 benchmark 无状态冲突。
- [ ] 支持范围、非目标和已知限制明确。

### 产品边界

- [x] 默认聚合头只包含正式 Supported 模块。
- [x] 安装包不导出或依赖旧 `.cpp` backend。
- [x] detail/internal API 不进入兼容承诺。

### 用户接入

- [ ] direct include 示例通过。
- [ ] CMake install consumer 通过。
- [ ] FetchContent consumer 通过。
- [ ] 至少四个正式 header-only 示例可编译。

### 正确性与平台

- [ ] Linux GCC/Clang required CI 通过。
- [ ] macOS ARM64/NEON required CI 通过。
- [ ] Windows MSVC required CI 通过。
- [ ] ASan/UBSan 通过。
- [ ] 安装后的公共头和 target 在三平台可消费。

### 性能

- [ ] 最新 dated performance report 对应 release commit。
- [ ] GEMM scalar/UI/NEON/AVX2 dispatch 有 forced-path tests。
- [ ] 关键 Core/Imgproc anchor 没有未解释的明显退化。
- [ ] 报告包含硬件、线程、dispatch 和采样元数据。

### 发布资产

- [ ] `VERSION.txt`、tag 和 CMake package version 一致。
- [ ] `CHANGELOG.md` 和 release notes 完成。
- [ ] Release 源码包通过独立 consumer 验证。
- [ ] 生成并发布校验和。

## 12. 风险与控制

### 风险 1：继续追求覆盖率导致范围失控

控制方式：

- 以真实用户流水线决定 API；
- 每批 API 必须有共同依赖和共同测试设计；
- 不以 `220 / 220` 作为 `0.1` 发布条件。

### 风险 2：GEMM 再次形成超大 patch

控制方式：

- scalar/UI、NEON、AVX2 三类实现保持固定；
- 新策略必须替换旧策略，不能只叠加；
- pack、kernel、scheduler 分开 benchmark；
- 无收益路径及时删除。

### 风险 3：跨平台结果不可比较

控制方式：

- correctness gate 与 performance report 分离；
- benchmark 必须固定机器类别和元数据；
- OpenCV/Accelerate 只作为平台上限，不作为所有机器的统一硬门槛。

### 风险 4：header-only 便利性被编译成本抵消

控制方式：

- 持续测量聚合头编译时间和预处理体积；
- 降低不必要的聚合 include；
- 大型表、模板和平台内核按模块隔离；
- detail 实现避免在无关 consumer TU 中被实例化。

## 13. 最终建议

下一阶段的正确顺序是：

```text
事实与文档收口
    → 移除默认产品中的编译型 backend 模糊边界
    → 补齐安装、示例和 Release 工程
    → 建立 ARM64/Windows 持续验证
    → 重建当前性能基线
    → 发布 cvh 0.1.0
    → 选择性进入高价值 API 和下一轮性能优化
```

`cvh` 当前最需要的不是更多代码，而是把已经拥有的能力变成可信、清晰、
可重复使用的产品。完成 P7 后，P8 的 API 扩张和后续 SIMD 优化才能建立在
稳定边界上，不再反复产生大规模整理成本。
