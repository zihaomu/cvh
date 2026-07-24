# Core / Imgproc 测试重构方案

## 1. 结论

本轮测试重构采用以下原则：

1. 测试文件按稳定的公开 API、数据结构或运行时职责命名，不再使用
   `phase1`、版本号、里程碑等阶段性名称。
2. `contract` 是默认测试语义，不再作为大多数文件名的固定后缀；只有
   `regression`、`integration`、`upstream`、`internal` 等确实有不同职责的
   类型才在路径或文件名中显式表达。
3. `test/core` 和 `test/imgproc` 按功能域分目录，文件名不再重复携带
   `core_` 或 `imgproc_` 前缀。
4. 目录中的每个 `*_test.cpp` 必须被唯一构建目标收录；CMake 在配置期检查
   “存在但未注册”和“重复注册”，不再依赖人工发现。
5. 先修复当前测试链路中的假覆盖、重复覆盖和失效 fixture，再做批量改名与
   拆分。文件移动本身不得改变测试行为。
6. 当前可运行用例必须建立迁移台账；历史文件中的未接线用例必须逐项判定为
   迁入、被现有用例替代或删除，不能继续静默留在目录中。

`imgproc_phase1_geometric_sampling_contract_test.cpp` 不会简单改成另一个仍然
宽泛的文件名。它应按职责拆到：

- `geometry/convert_maps_test.cpp`
- `geometry/remap_test.cpp`
- `geometry/warp_perspective_test.cpp`
- `geometry/rect_sub_pix_test.cpp`

跨 API 的 map 表示一致性 case 归属 `remap_test.cpp`，`convert_maps_test.cpp`
只验证 map 转换本身的输入、输出和错误合同。

## 2. 当前基线

盘点日期：2026-07-24。

### 2.1 规模与接线状态

| 模块 | 目录内 `.cpp` | 已接入目标 | 目录内 `TEST` | 当前目标实际执行 | 代码行数 |
|---|---:|---:|---:|---:|---:|
| `core` | 22 | 19 | 196 | 176 | 6,404 |
| `imgproc` | 18 | 18 | 173 | 173 | 9,049 |

在当前 Release/header-only 构建中：

- `cvh_test_core_lite`：176 个用例，174 个通过，2 个按设计跳过。
- `cvh_test_imgproc`：173 个用例，173 个通过。
- 两个模块级 CTest 均通过。

这只能证明当前被接线的两个二进制可运行，不能证明目录中所有测试都有效。

### 2.2 已确认的问题

#### 2.2.1 三个 core 测试源没有进入任何活动目标

- `test/core/kernel_op_test.cpp`：7 个用例。
- `test/core/mat_test.cpp`：12 个用例。
- `test/core/system_test.cpp`：1 个用例。

其中 `kernel_op_test.cpp` 和 `mat_test.cpp` 不是“暂时漏加一个 CMake 行”这么
简单。独立编译审计已经暴露：

- `test/utils/mat_load.h` 使用未限定的 `Mat`，在当前命名空间下无法编译。
- `test/utils/mat_load.cpp` 没有接入测试目标。
- `mat_test.cpp` 引用了当前不存在的 `quantize_int8_per_row`。
- `system_test.cpp` 的唯一用例没有任何活动断言，只保留了注释代码。

因此不能把这三个文件直接批量加入 target。必须先逐 case 审核和修复。

#### 2.2.2 core fixture 当前没有活动消费者

`test/core/test_data/data` 中有 122 个被版本控制的 `.npy` 文件，但读取这些
fixture 的只有上述未接线的 `kernel_op_test.cpp` 和 `mat_test.cpp`。当前通过的
176 个 core 用例不会读取这些数据。

这意味着 fixture 的存在目前不构成有效覆盖。重构后必须满足：

- 有活动测试消费的 fixture 才保留。
- 每个 fixture 都能追溯到生成器、随机种子、dtype/shape、oracle 和消费用例。
- 无消费者、重复或无法复现的 fixture 删除，并在迁移台账中记录原因。

#### 2.2.3 CTest 重复执行同一个 core 二进制

`cvh_test_core` 是 `cvh_test_core_lite` 的 target alias，但 CTest 同时注册了：

- `cvh_test_core_lite`
- `cvh_test_core`

两者执行同一个二进制。这会让全量 CTest 重复跑同一套 core 用例，并让报告看似
多了一层覆盖。最终只保留一个规范名称 `cvh_test_core` 和一个 CTest 入口。

#### 2.2.4 文件名和职责被历史阶段污染

以下文件按旧的 Phase 1 实施批次分组，而不是按长期维护职责分组：

- `imgproc_phase1_kernels_contract_test.cpp`
- `imgproc_phase1_intensity_contract_test.cpp`
- `imgproc_phase1_pyramid_color_contract_test.cpp`
- `imgproc_phase1_geometry_matrix_contract_test.cpp`
- `imgproc_phase1_geometric_sampling_contract_test.cpp`

例如 `pyramid_color` 同时包含 accumulate、blend、pyramid、two-plane color 和
demosaicing；`kernels` 同时包含 kernel 生成、integral、导数和 squared box
filter。实现阶段结束后，这些名称已经不能帮助定位维护责任。

#### 2.2.5 大文件和混合职责降低可维护性

- `imgproc_cvtcolor_contract_test.cpp`：3,288 行，混合 RGB/GRAY、YUV420、
  YUV422、YUV444、连续/非连续布局和错误路径。
- `array_ops_contract_test.cpp`：1,116 行，同时测试公开 API 和私有 UI dispatch。
- `imgproc_morph_gradient_contract_test.cpp`：同时包含 morphology、Sobel 和
  upstream port。
- `imgproc_filter_contract_test.cpp`：同时包含 box filter、Gaussian blur、
  fast-path 和 upstream port。
- `mat_contract_test.cpp`：同时包含生命周期、转换、ROI、transpose 公共合同和
  transpose 私有 kernel。

#### 2.2.6 黑盒合同与白盒实现测试混在一起

当前多个 core 文件直接 include `cvh/core/detail/*`：

- `array_ops_contract_test.cpp`
- `binary_op_contract_test.cpp`
- `math_ops_contract_test.cpp`
- `mat_contract_test.cpp`

公开 API 合同和私有 dispatch/kernel 测试需要分开。否则 detail 重构会造成大量
看似 API 回归、实为白盒耦合的失败。

#### 2.2.7 “独立头可包含”用例的名字强于实际验证

`imgproc_header_layout_test.cpp` 在同一个翻译单元内按固定顺序 include 多个头，
然后执行 `SUCCEED()`。前面的头可能为后面的头补齐依赖，因此它不能证明每个
header 可以被独立 include。

真正的 header self-containment 测试必须为每个公共头建立独立翻译单元。

#### 2.2.8 数据和 upstream 清单包含本机绝对路径

`test/imgproc/data/manifest.json` 和 upstream manifest 中保留了
`/Volumes/...` 本机路径。它们不能作为可移植来源信息。清单应记录仓库 URL、
tag/commit、相对路径、SHA-256 和生成器版本，不记录开发机绝对路径。

## 3. “测试有效”的定义

一个测试只有同时满足以下条件，才计入有效覆盖：

1. **可发现**：测试源在模块 manifest 中，且配置期完整性检查通过。
2. **可构建**：从新的构建目录可以编译和链接，不能依赖旧 build 目录里的
   残留对象。
3. **可执行**：由规范 CTest 入口实际运行；目录中只有源码或 fixture 不计入
   覆盖。
4. **有观察点**：每个用例至少有一个可失败的行为断言。只打印、只调用 API、
   空 `SUCCEED()` 或全部注释不构成行为测试。纯编译测试是例外，但必须放在
   compile smoke 层，并通过独立翻译单元表达。
5. **oracle 可信**：期望值来自固定数学合同、独立 reference、版本固定的
   upstream 差分或有来源的 golden data，不能复制生产实现后与自己比较。
6. **可复现**：随机输入有固定种子；fixture 有 hash 和生成方式；测试不依赖
   本机绝对路径、执行顺序、网络和残留输出文件。
7. **失败可定位**：文件、suite 和 case 名能指向一个稳定功能域。

单纯“当前为绿色”不是完成定义。

## 4. 命名与职责规则

### 4.1 文件命名

默认格式：

```text
<stable_feature>_test.cpp
```

只在职责确实不同的情况下使用额外语义：

```text
<feature>_regression_test.cpp
<feature>_integration_test.cpp
<feature>_upstream_test.cpp
<feature>_internal_test.cpp
```

规则：

- 不使用 `phase1`、`phase2`、日期、版本号或任务编号。
- 不在 `test/imgproc` 文件名中重复 `imgproc_`。
- 不在 `test/core` 文件名中重复 `core_`。
- 不把 `contract` 作为所有文件的固定后缀；普通 `_test.cpp` 默认就是公共行为
  合同。
- 文件尽量对应一个公开头或一个强内聚 API family。
- 600 行作为拆分复查线，不作为机械硬限制。reference helper 和大参数表优先
  移入 `support/`，但断言意图必须留在测试文件中。
- 禁止 `core_ops_test.cpp`、`kernel_op_test.cpp`、`mat_test.cpp` 这类无法说明
  边界的名字。

### 4.2 Suite 和 case 命名

Suite 使用稳定功能名：

```cpp
TEST(RemapTest, map_representations_produce_same_linear_result)
TEST(RemapUpstreamTest, regression_XXXXX)
TEST(ArrayDispatchInternalTest, vector_tail_matches_scalar)
```

规则：

- Suite 不再带 `_TEST`。
- Suite 不重复 `Imgproc` 或 `Core` 前缀。
- case 继续使用清晰的 lower snake case，优先表达输入条件和可观察结果。
- upstream issue/用例编号必须保留，便于回查。
- 禁止 `Op_Test`、`Op_Test2`、`log_test`、`test_mat_brodcast` 等含糊或拼写错误的
  名称。

### 4.3 测试类型

| 类型 | 位置 | 允许依赖 | 主要目的 |
|---|---|---|---|
| 公共合同 | 功能域目录 | 公共 `cvh/*` 头 | API、数值、边界、异常、alias/ROI |
| internal | `internal/` | `cvh/*/detail/*` | dispatch、SIMD tail、私有 kernel |
| upstream | `upstream/` | 公共头 + 测试 support | 固定 upstream case 的兼容回归 |
| integration | `integration/` | 多模块公共头、fixture | 跨 API pipeline |
| compile smoke | `compile/` 或 `test/smoke` | 单个公共头 | header self-containment、ODR |

## 5. 目标目录结构

目录只表达稳定职责，不要求为每个目录单独创建测试二进制。

```text
test/
  CMakeLists.txt
  cmake/
    CvhTest.cmake
  core/
    CMakeLists.txt
    README.md
    mat/
      lifecycle_test.cpp
      conversion_test.cpp
      layout_test.cpp
      channel_test.cpp
      roi_test.cpp
      expression_test.cpp
      scalar_ops_test.cpp
      display_test.cpp
      opencv_compat_test.cpp
    operations/
      arithmetic_test.cpp
      array_test.cpp
      array_layout_test.cpp
      math_test.cpp
      reduction_test.cpp
      transpose_test.cpp
      gemm_test.cpp
      inference_test.cpp
    types/
      geometry_test.cpp
      scalar_test.cpp
    runtime/
      parallel_for_test.cpp
      error_test.cpp
    internal/
      arithmetic_dispatch_test.cpp
      array_dispatch_test.cpp
      math_dispatch_test.cpp
      transpose_kernel_test.cpp
    upstream/
      mat_channel_upstream_test.cpp
    data/
      manifest.json
      generators/
      npy/
  imgproc/
    CMakeLists.txt
    README.md
    arithmetic/
      accumulate_test.cpp
      blend_linear_test.cpp
    color/
      cvtcolor_rgb_gray_test.cpp
      cvtcolor_yuv420_test.cpp
      cvtcolor_yuv422_test.cpp
      cvtcolor_yuv444_test.cpp
      cvtcolor_two_plane_test.cpp
      demosaicing_test.cpp
      colormap_test.cpp
    filtering/
      kernels_test.cpp
      box_filter_test.cpp
      gaussian_blur_test.cpp
      filter2d_test.cpp
      sep_filter2d_test.cpp
      median_blur_test.cpp
      bilateral_filter_test.cpp
      stack_blur_test.cpp
      derivatives_test.cpp
      sqr_box_filter_test.cpp
      copy_make_border_test.cpp
    geometry/
      resize_test.cpp
      transform_matrix_test.cpp
      convert_maps_test.cpp
      remap_test.cpp
      warp_affine_test.cpp
      warp_perspective_test.cpp
      rect_sub_pix_test.cpp
      pyramid_test.cpp
    intensity/
      threshold_test.cpp
      adaptive_threshold_test.cpp
      equalize_hist_test.cpp
      lut_test.cpp
    morphology/
      morphology_test.cpp
    feature/
      canny_test.cpp
    statistics/
      integral_test.cpp
    internal/
    upstream/
    integration/
      basic_pipeline_test.cpp
    support/
    data/
      manifest.json
      opencv_extra/
```

如果后续确认某个目录只有一个长期文件，可以合并一级目录；不能为了追求目录
形式而制造空目录。

## 6. Core 文件迁移建议

| 当前文件 | 目标职责 |
|---|---|
| `array_ops_contract_test.cpp` | 公开 case 迁到 `operations/array_test.cpp`；UI/detail case 迁到 `internal/array_dispatch_test.cpp` |
| `binary_op_contract_test.cpp` | 公开算术迁到 `operations/arithmetic_test.cpp`；强制 scalar/dispatch case 迁到 `internal/arithmetic_dispatch_test.cpp` |
| `core_ops_test.cpp` | `convertTo` 迁到 `mat/conversion_test.cpp`；`copyTo` 迁到 `mat/lifecycle_test.cpp` 或 `mat/roi_test.cpp` |
| `gemm_pack_contract_test.cpp` | 合并到 `operations/gemm_test.cpp` 的 pack/reuse 分组 |
| `geometry_types_contract_test.cpp` | `types/geometry_test.cpp` |
| `layout_ops_contract_test.cpp` | `operations/array_layout_test.cpp` |
| `mat_channel_contract_test.cpp` | `mat/channel_test.cpp` |
| `mat_layout_semantics_test.cpp` | `mat/layout_test.cpp` |
| `mat_contract_test.cpp` | 按 lifecycle/conversion/transpose 拆分；私有 blocked transpose 迁到 `internal/transpose_kernel_test.cpp` |
| `mat_expr_scalar_compare_test.cpp` | `mat/expression_test.cpp` |
| `mat_opencv_compat_test.cpp` | `mat/opencv_compat_test.cpp` |
| `mat_scalar_ops_test.cpp` | `mat/scalar_ops_test.cpp` |
| `mat_shape_display_test.cpp` | `mat/display_test.cpp` |
| `mat_submat_test.cpp` | `mat/roi_test.cpp` |
| `mat_upstream_channel_port_test.cpp` | `upstream/mat_channel_upstream_test.cpp` |
| `math_ops_contract_test.cpp` | 公共 math 迁到 `operations/math_test.cpp`；UI/detail case 迁到 `internal/math_dispatch_test.cpp` |
| `parallel_for_runtime_test.cpp` | `runtime/parallel_for_test.cpp` |
| `reduction_ops_contract_test.cpp` | `operations/reduction_test.cpp` |
| `scalar_contract_test.cpp` | `types/scalar_test.cpp` |

### 6.1 未接线 legacy 文件的处理

`kernel_op_test.cpp` 和 `mat_test.cpp` 不直接改名保留，而是建立 case disposition：

| 旧内容 | 处理方向 |
|---|---|
| generated binary cases | 与当前 arithmetic/array 测试比较，独有边界迁入 `operations/arithmetic_test.cpp` |
| transpose fixture cases | 独有排列迁入 `operations/transpose_test.cpp` |
| GEMM generated/FP16/INT8 cases | 修复 oracle 后迁入 `operations/gemm_test.cpp` |
| softmax/SiLU/RMSNorm/RoPE | 作为非 OpenCV 扩展明确放入 `operations/inference_test.cpp` |
| `loadNpy` | 测试的是 test support，不是 Mat；若保留，迁到共享 support 的独立测试 |
| 打印、空调用、重复用例 | 删除，并记录替代用例或无观察点原因 |
| `system_test.cpp` 空用例 | 删除；如需系统层测试，新增验证异常 code/message 的 `runtime/error_test.cpp` |

`softmax/SiLU/RMSNorm/RoPE` 是否仍属于项目公开产品范围，需要与
`include/cvh/core/readme.md` 的非主线说明保持一致；测试目录不能用含糊的
`kernel_op` 掩盖这个边界。

## 7. Imgproc 文件迁移建议

| 当前文件 | 目标职责 |
|---|---|
| `imgproc_header_layout_test.cpp` | 改为真正的逐头独立 TU compile smoke，不继续作为普通 GTest |
| `imgproc_resize_contract_test.cpp` | `geometry/resize_test.cpp`；明确 port 的 case 可迁到 `upstream/resize_upstream_test.cpp` |
| `imgproc_cvtcolor_contract_test.cpp` | 按 RGB/GRAY、YUV420、YUV422、YUV444 拆到 `color/`，共享独立 reference 放 `support/` |
| `imgproc_threshold_contract_test.cpp` | `intensity/threshold_test.cpp`，并合并 `thresholdWithMask` |
| `imgproc_pipeline_regression_test.cpp` | `integration/basic_pipeline_test.cpp` |
| `imgproc_lut_contract_test.cpp` | `intensity/lut_test.cpp` |
| `imgproc_filter_contract_test.cpp` | 拆为 `filtering/box_filter_test.cpp`、`filtering/gaussian_blur_test.cpp` 和对应 upstream case |
| `imgproc_phase1_kernels_contract_test.cpp` | 拆为 `filtering/kernels_test.cpp`、`statistics/integral_test.cpp`、`filtering/derivatives_test.cpp`、`filtering/sqr_box_filter_test.cpp` |
| `imgproc_phase1_intensity_contract_test.cpp` | 拆为 median/bilateral/stack/adaptive threshold/equalize/colormap 对应文件；`thresholdWithMask` 合入 threshold |
| `imgproc_phase1_pyramid_color_contract_test.cpp` | 拆为 accumulate、blend linear、pyramid、cvtColorTwoPlane、demosaicing |
| `imgproc_phase1_geometry_matrix_contract_test.cpp` | `geometry/transform_matrix_test.cpp` |
| `imgproc_phase1_geometric_sampling_contract_test.cpp` | 拆为 convert maps、remap、warp perspective、rect sub pix |
| `imgproc_filter2d_contract_test.cpp` | `filtering/filter2d_test.cpp` |
| `imgproc_sep_filter2d_contract_test.cpp` | `filtering/sep_filter2d_test.cpp` |
| `imgproc_copy_make_border_contract_test.cpp` | `filtering/copy_make_border_test.cpp`，upstream case 明确归档 |
| `imgproc_morph_gradient_contract_test.cpp` | 拆为 `morphology/morphology_test.cpp`、`filtering/derivatives_test.cpp` 和对应 upstream case |
| `imgproc_canny_contract_test.cpp` | `feature/canny_test.cpp` 和 `upstream/canny_upstream_test.cpp` |
| `imgproc_warp_affine_contract_test.cpp` | `geometry/warp_affine_test.cpp` |

### 7.1 cvtColor 的拆分边界

`cvtColor` 仍是一个公共入口，但单文件 3,288 行已经不适合作为维护单元。拆分按
像素格式 family，而不是按实现函数或历史提交：

- `cvtcolor_rgb_gray_test.cpp`：RGB/BGR/BGRA/RGBA/GRAY 和 `CV_32F`。
- `cvtcolor_yuv420_test.cpp`：NV12/NV21/I420/YV12。
- `cvtcolor_yuv422_test.cpp`：NV16/NV61/YUY2/UYVY。
- `cvtcolor_yuv444_test.cpp`：NV24/NV42/I444/YV24 和基础 packed YUV。

每个文件都覆盖：

- 正常值。
- ROI/step 非连续输入。
- 单行/单列或最小合法尺寸。
- 非法 channel/layout。
- 支持的 depth。

reference 转换 helper 只保留一份，但不得调用被测生产 helper。

## 8. CMake 与测试发现机制

### 8.1 CMake 分层

顶层 `CMakeLists.txt` 只保留：

```cmake
if(CVH_BUILD_TESTS)
    add_subdirectory(test)
endif()
```

具体 source manifest 分别放在：

- `test/core/CMakeLists.txt`
- `test/imgproc/CMakeLists.txt`

共享 target 规则放在 `test/cmake/CvhTest.cmake`，统一设置：

- `gtest_main`
- `cvh::headers`
- C++17
- repository/test include path
- fixture root
- CTest label

### 8.2 显式 source manifest + 完整性审计

继续显式列出 source，方便 code review 看见新增测试；同时用
`file(GLOB_RECURSE ... CONFIGURE_DEPENDS "*_test.cpp")` 只做完整性审计：

- discovered source 不在 manifest：配置失败。
- manifest 指向不存在的 source：配置失败。
- 同一 source 出现在多个 target：配置失败。
- `support/*.cpp` 单独列入 support source，不伪装成 test source。

glob 不直接决定构建内容，因此不会静默改变 target。

### 8.3 规范 target

最终只保留：

- `cvh_test_core`
- `cvh_test_imgproc`

CTest 也只注册同名的两个模块入口，并加 `core`、`imgproc`、`unit` 等 label。
不再同时注册 `cvh_test_core_lite` 和 `cvh_test_core` 执行同一命令。

模块内继续使用 GTest filter 做本地精确运行。当前用例规模很小，不需要把每个
GTest case 拆成一个独立进程，避免 CTest 产生数百次进程启动开销。

### 8.4 Header self-containment

每个公共 header 使用一个独立 `.cpp`：

```cpp
#include <cvh/imgproc/remap.h>
int main() { return 0; }
```

这些翻译单元作为 compile smoke 构建。禁止在一个文件中顺序 include 所有头后
声称“individually includable”。

## 9. Oracle、断言与 fixture 规则

### 9.1 Oracle 优先级

按以下顺序选择：

1. 小尺寸、手工可验证的固定结果。
2. 与生产实现不同结构的独立 scalar reference。
3. 固定 OpenCV tag/commit 生成的差分结果。
4. 有生成器和 hash 的 golden fixture。
5. 不变量和 metamorphic relation，例如 round-trip、in-place/out-of-place
   等价、separable/full kernel 等价。

同一用例可以组合多种 oracle。不能把“当前 cvh 输出”无来源地固化为 golden。

### 9.2 数值断言

- 整数和 bit-exact API 优先 `EXPECT_EQ`/逐元素精确比较。
- 浮点必须写清 absolute/relative tolerance，不能统一使用一个拍脑袋 epsilon。
- NaN、Inf、signed zero、饱和、舍入模式单独覆盖。
- 比较 helper 在失败时输出 case、shape、type、最大 absolute/relative error
  和首个错误位置。

### 9.3 Skip 规则

- 默认 core/imgproc 测试运行的目标是 0 个意外 skip。
- 当前两个 OutputArray 用例属于明确的产品非目标，不应永久作为总是 skip 的
  可执行测试；它们保留在 upstream manifest，状态改为 `OUT_OF_SCOPE`。
- 暂时性平台 skip 必须有条件、原因和对应 CI 平台；skip 数量增长视为失败。
- 不能用 skip 隐藏缺 fixture、未接线实现或编译问题。

### 9.4 Fixture manifest

每个 fixture 条目至少包含：

```json
{
  "path": "npy/gemm_nn_small_odd_a.npy",
  "sha256": "...",
  "generator": "generators/generate_gemm_cases.py",
  "seed": 20260724,
  "dtype": "float32",
  "shape": [3, 5],
  "oracle": "numpy.matmul",
  "consumers": ["GemmTest.generated_nn_small_odd"]
}
```

生成脚本统一使用 `generate_*` 命名，修正现有 `generater` 拼写。生成后必须能用
`git diff --exit-code` 证明结果稳定。

Imgproc manifest 记录：

- upstream repository URL。
- tag 和 commit SHA。
- upstream 相对路径。
- 本地 snapshot 相对路径。
- size/hash。
- consumer suite/case。

不得记录 `/Users/...`、`/Volumes/...` 等本机路径。

### 9.5 Upstream port

`test/upstream/opencv/*/case_manifest.json` 是来源台账，活动测试是执行事实。二者
通过稳定 case ID 关联，不再靠自然语言 `reason` 猜测目标文件。

建议字段：

- `id`
- `upstream_commit`
- `source_file`
- `source_lines`
- `snapshot_sha256`
- `status`
- `local_test`
- `adaptation`

状态只使用：

- `PASS`
- `PENDING`
- `OUT_OF_SCOPE`
- `REPLACED`

README 和 failing-tests 文档由 manifest 生成或只链接 manifest，不再手工复制
数量，避免当前台账与源码数量漂移。

## 10. 实施顺序

### T0：冻结迁移基线

产物：

- 导出 core/imgproc 的 `--gtest_list_tests`。
- 记录每个文件的 suite/case 数量。
- 记录当前 174 pass + 2 skip 和 173 pass。
- 建立 case disposition 表。

门槛：

- 当前规范目标从已有干净构建成功。
- 工作区已有功能修改不被测试重构覆盖或回退。

### T1：先修测试真相链

动作：

1. 移除重复 core CTest 注册。
2. 引入 source manifest 完整性检查。
3. 审核三个未接线 core 文件。
4. 修复有价值的 NPY reader 和独有用例，删除空测试与无观察点用例。
5. 建立 fixture consumer/hash 审计。

门槛：

- `test/core`、`test/imgproc` 不再存在静默未注册的 `*_test.cpp`。
- 每个 legacy case 有明确 disposition。
- 默认运行没有因缺 fixture 而跳过的用例。

### T2：纯路径与命名迁移

动作：

- 使用 `git mv` 建立功能域目录。
- 同步 CMake、README、manifest 和脚本路径。
- 本阶段不改断言、不改 tolerance、不改输入数据。

门槛：

- before/after 用例清单按 rename map 一一对应。
- pass/fail/skip 数量不变。
- clean build 通过。

### T3：拆分 core 混合职责

动作：

- 分离 public contract 与 internal dispatch/kernel。
- 拆分 `mat_contract`、array、binary、math。
- 合并两个已有 GEMM 来源并去重。
- 明确 inference 扩展算子的产品边界。

门槛：

- 公共合同目录不 include `cvh/core/detail/*`。
- internal 测试仍通过公共入口验证最终输出，私有 helper 只用于控制/观测路径。
- 删除重复 case 时记录替代 case。

### T4：拆分 imgproc 历史批次与大文件

动作：

- 首先拆五个 `phase1` 文件。
- 再拆 `cvtColor`、filter、morphology/Sobel 混合文件。
- 抽取独立 reference 和参数构造 support。

门槛：

- `test/imgproc` 下不再出现 `phase1`。
- `cvtColor` 各 family 覆盖矩阵不减少。
- upstream case ID 与新本地 case 关联正确。

### T5：数据和 upstream 治理

动作：

- `test_data/data` 收敛为 `data/npy`。
- 生成器改为稳定入口和正确命名。
- 清理无消费者 fixture。
- manifest 去除绝对路径，补 hash/oracle/consumer。
- pipeline golden 补充生成来源；无法证明来源的 hash 重新由固定 upstream 生成。

门槛：

- 删除任一保留 fixture 后，测试明确失败并指出缺失文件。
- 重新生成 fixture 后仓库无 diff。
- manifest 中不存在本机绝对路径。

### T6：文档和 CI 收口

动作：

- 重写 `test/readme.md`、`test/core/README.md`、`test/imgproc/README.md`。
- 更新 failing/upstream 台账为 manifest 驱动。
- CI 增加 registration、fixture 和 clean-build gate。

门槛：

- README 描述与 CMake target、case 数和目录一致。
- CI 不重复运行同一 core 二进制。

## 11. 每个提交的验证命令

最终规范命令：

```bash
cmake -S . -B build-test-refactor-release \
  -DCVH_BUILD_TESTS=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-test-refactor-release -j \
  --target cvh_test_core cvh_test_imgproc

ctest --test-dir build-test-refactor-release \
  --output-on-failure \
  -R '^(cvh_test_core|cvh_test_imgproc)$'

./build-test-refactor-release/cvh_test_core --gtest_list_tests
./build-test-refactor-release/cvh_test_imgproc --gtest_list_tests
```

过渡期在 target 尚未改名时使用 `cvh_test_core_lite`，但最终文档和 CI 只暴露
`cvh_test_core`。

另外必须执行：

```bash
git diff --check
rg -n 'phase[0-9]+|imgproc_.*_contract_test|_TEST' test/core test/imgproc
ctest --test-dir build-test-refactor-release -N
```

`rg` 的剩余结果必须逐项解释；upstream snapshot 或历史文档中的原始名字不要求
机械改写。

建议在最终合并前增加一个 Debug + ASan/UBSan 构建，重点覆盖 Mat
ownership/ROI、in-place、border、remap 和 filter 路径。

## 12. 完成定义

- [ ] `test/core` 和 `test/imgproc` 的文件、目录和 suite 名不含实施阶段标签。
- [ ] 每个 `*_test.cpp` 被一个且仅一个活动 target 收录。
- [ ] `cvh_test_core` 和 `cvh_test_imgproc` 从新 build 目录构建并通过。
- [ ] 不再重复注册 core CTest。
- [ ] 当前所有有效 case 都有迁移映射；删除项有替代或删除理由。
- [ ] 默认运行没有设计性永久 skip；非目标只留 manifest。
- [ ] 公共合同测试不 include `detail` 头。
- [ ] 每个测试都有行为断言，compile smoke 使用独立翻译单元。
- [ ] 所有保留 fixture 有 generator、hash、oracle 和 consumer。
- [ ] manifest 不含本机绝对路径。
- [ ] `cvtColor`、Mat、array 等大文件按稳定职责拆分。
- [ ] `test/readme.md`、模块 README、CMake 和 upstream manifest 一致。

## 13. 提交策略

为了让 review 和回退可控，至少拆成以下提交：

1. `test: enforce source registration and remove duplicate core ctest`
2. `test(core): triage orphan tests and fixtures`
3. `test: reorganize core test paths without behavior changes`
4. `test: reorganize imgproc test paths without behavior changes`
5. `test(imgproc): split phase-tagged and oversized suites`
6. `test: normalize fixtures, manifests, and documentation`

路径移动、行为修改、golden 更新不放在同一个提交中。任何一步出现输出变化，都应
先判断是测试组织错误、原有测试缺陷还是产品行为变化，不能用更新 golden 直接
消除失败。
