# OpenCV Upstream SIMD 算子加速计划

> - 状态：P-ACC-3 至 P-ACC-8 Apple ARM 已收尾，真实 x86 SSE/AVX 运行待验证
> - 日期：2026-07-25
> - 性能基线：`benchmark/opencv_compare/results/2026-07-25-opencv-upstream-performance.md`
> - OpenCV 参考树：`/Users/zmu/work/my_project/ocvh/opencv`
> - OpenCV 参考版本：`4.14.0-pre`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`

## 0. 实施状态

| 阶段 | 状态 | 当前范围 |
| --- | --- | --- |
| P-ACC-0 | Apple ARM 已完成，x86 运行待验证 | dispatch tag、scalar 强制模式、tail/ROI/多 TU 门禁、Mode A 路径标识 |
| P-ACC-1 | 第五组 Apple ARM 已完成，x86 运行待验证 | `PATCH_NANS/EXP/LOG/POW` 的 F32 UI fast-path 已进入公共路径 |
| P-ACC-2 | Apple ARM 已完成，x86 运行待验证 | reduction UI 基础设施、非零检测、统计量、极值与索引归约、`norm/normalize/reduce` |
| P-ACC-3 | Apple ARM 已完成，x86 运行待验证 | Core 内存布局、通道与 GEMM |
| P-ACC-4 | Apple ARM 已完成，x86 运行待验证 | Imgproc 滤波共享底座与导数 |
| P-ACC-5 | Apple ARM 已完成，x86 运行待验证 | 颜色、去马赛克与几何采样 |
| P-ACC-6 | Apple ARM 已完成，x86 运行待验证 | 非线性、形态学、累积与强度变换 |
| P-ACC-7 | Apple ARM 已完成，x86 运行待验证 | scalar-small 与全矩阵收尾 |
| P-ACC-8 | Apple ARM 已完成，x86 运行待验证 | 已收敛 pyramid、nonlinear、geometry、filter/derivative 与 Core reduction 热点，不扩 API 范围 |

P-ACC-1 首组实施边界：

- Mat-Mat `absdiff/min/max` 先覆盖 UI 可直接表达且没有浮点 NaN 语义差异的整数深度。
- 无 mask bitwise 按原始字节向量化，因此覆盖所有现有 Mat depth/channel。
- masked bitwise、FP16 以及浮点 `min/max/absdiff` 先保留 scalar fallback。
- 先以当前 Mode B 已测量的 `CV_8UC3` 路径作为性能 gate，再扩展 scalar overload。

### 0.1 已落地基础设施

- 新增 `DispatchTag::OpenCVUI`，公共 API 可报告实际命中的 `scalar` 或 `opencv_ui`。
- `cvh_benchmark_core_mat_header` 新增 `--dispatch auto|scalar`，支持同一二进制内的
  scalar/UI Mode A 对比。
- 新增 ODR-safe `core/detail/arithm_ui.hpp`，直接使用 vendored OpenCV UI 的
  `vx_load/vx_store/VTraits/v_*`，没有恢复项目自定义 SIMD adapter。
- correctness 覆盖整数 depth、有符号极值、C3、非连续 ROI、非向量宽度、短行、in-place
  和 masked fallback。
- Apple ARM Release 全量构建通过，CTest `16/16` 通过，含 core/imgproc 多 TU smoke。
- Apple Clang 的 x86_64 交叉编译已分别用 `-msse2 -mno-avx` 和 `-mavx2` 成功实例化
  core benchmark；当前机器没有 Rosetta，因此这不能替代真实 x86 运行验证。

### 0.2 P-ACC-1 首组结果

已进入 UI fast-path：

| API 范围 | UI 覆盖 | 保留 scalar 的范围 |
| --- | --- | --- |
| Mat-Mat `absdiff/min/max` | `CV_8U/8S/16U/16S/32S/32U` | `CV_16F/32F/64F`、Mat-Scalar |
| Mat-Mat `bitwise_and/or/xor` | 无 mask，所有 depth/channel，按原始字节处理 | masked、Mat-Scalar |
| `bitwise_not` | 无 mask，所有 depth/channel，按原始字节处理 | masked |

Mode A 使用同一 Release 二进制、`quick` case 矩阵、`warmup=2`、`iters=20`、
`repeats=5`；下表是 VGA `CV_8UC3` 中位数：

| 算子 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | Checksum |
| --- | ---: | ---: | ---: | --- |
| `ABSDIFF` | 0.313471 | 0.030377 | 10.32x | 一致 |
| `BITWISE_AND` | 0.293790 | 0.032888 | 8.93x | 一致 |
| `MIN` | 0.216521 | 0.035881 | 6.03x | 一致 |
| `MAX` | 0.209865 | 0.035313 | 5.94x | 一致 |

Mode B 已用 `full` profile 重跑全部 `321` 个 case，并更新
`benchmark/opencv_compare/results/2026-07-24-opencv-upstream-performance.md`：

| 算子 | 实施前 OpenCV 领先 | 当前 OpenCV 领先 | 当前 dispatch |
| --- | ---: | ---: | --- |
| `ABSDIFF` | 12.94x | 1.23x | `opencv_ui` |
| `BITWISE_AND` | 13.07x | 1.33x | `opencv_ui` |
| `BITWISE_NOT` | 19.78x | 2.10x | `opencv_ui` |
| `BITWISE_OR` | 13.02x | 1.32x | `opencv_ui` |
| `BITWISE_XOR` | 12.86x | 1.32x | `opencv_ui` |
| `MIN` | 9.20x | 1.55x | `opencv_ui` |
| `MAX` | 9.19x | 1.63x | `opencv_ui` |

剩余门禁：在真实 x86 SSE/AVX 环境运行 correctness 和 Mode A；完成前不能把跨平台
P-ACC-0 或 P-ACC-1 首组标记为全部完成。

### 0.3 P-ACC-1 第二组结果

当前实施前 Mode B 基线：

| 算子 | OpenCV 领先 | 当前 dispatch |
| --- | ---: | --- |
| `ADD` | 5.27x | `headers_baseline` |
| `SUBTRACT` | 5.20x | `headers_baseline` |
| `MULTIPLY` | 5.46x | `headers_baseline` |
| `DIVIDE` | 2.91x | `headers_baseline` |

本组边界：

- Mat-Mat `add/subtract/multiply` 使用 upstream 同类 `v_add/v_sub/v_mul`，覆盖
  `CV_8U/8S/16U/16S/32S/32U/32F`，`CV_SIMD_64F` 可用时覆盖 `CV_64F`。
- Mat-Mat `divide` 先覆盖 `CV_32F`，`CV_SIMD_64F` 可用时覆盖 `CV_64F`。
- 整数 `divide` 保留 scalar，避免改变项目现有的 `std::round` 舍入和除零语义。
- `CV_16F`、Mat-Scalar、Scalar-Mat 保留 scalar；后续必须先建立 scalar broadcast
  UI helper，再单独扩展。
- Mode A 增加四个公共入口 case；Mode B 必须记录每个 depth/channel 实际命中的
  `opencv_ui` 或 `scalar`，不能以算子名推断。

已完成：

- 新增 Mat-Mat `add/subtract/multiply/divide` UI 分流，并复用
  `core/detail/arithm_ui.hpp` 的统一 row/tail helper。
- `add/subtract/multiply` 覆盖整数与浮点 UI；`divide` 只对浮点启用 UI，整数继续走 scalar。
- correctness 覆盖六种整数深度的饱和结果、F32 除零/NaN/Inf、C3 非连续 ROI、非对齐 tail、
  in-place、短行、整数除法与 FP16 fallback。
- gate 发现并修复了 scalar tail 直接窄化的问题；tail 现在与向量段及公共 scalar 实现一致，
  统一执行 `saturate_cast`。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`，checksum 全部一致：

| 类型/尺寸 | 算子 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | 实际 dispatch |
| --- | --- | ---: | ---: | ---: | --- |
| `CV_8UC1 640x480` | `ADD` | 0.127881 | 0.009542 | 13.40x | `opencv_ui` |
| `CV_8UC1 640x480` | `SUBTRACT` | 0.114767 | 0.009996 | 11.48x | `opencv_ui` |
| `CV_8UC1 640x480` | `MULTIPLY` | 0.118248 | 0.009585 | 12.34x | `opencv_ui` |
| `CV_8UC1 640x480` | `DIVIDE` | 0.266754 | 0.268477 | 0.99x | `scalar` |
| `CV_8UC3 640x480` | `ADD` | 0.214873 | 0.027179 | 7.91x | `opencv_ui` |
| `CV_8UC3 640x480` | `SUBTRACT` | 0.192100 | 0.029433 | 6.53x | `opencv_ui` |
| `CV_8UC3 640x480` | `MULTIPLY` | 0.207790 | 0.027169 | 7.65x | `opencv_ui` |
| `CV_32FC1 640x480` | `ADD` | 0.095879 | 0.039660 | 2.42x | `opencv_ui` |
| `CV_32FC1 640x480` | `SUBTRACT` | 0.093402 | 0.038529 | 2.42x | `opencv_ui` |
| `CV_32FC1 640x480` | `MULTIPLY` | 0.091698 | 0.040279 | 2.28x | `opencv_ui` |
| `CV_32FC1 640x480` | `DIVIDE` | 0.097144 | 0.040900 | 2.38x | `opencv_ui` |

Mode B 已用 `full` profile 重跑全部 `321` 个 case：

| 算子 | 实施前 OpenCV 领先 | 当前 OpenCV 领先 | 当前 dispatch |
| --- | ---: | ---: | --- |
| `ADD` | 5.27x | 1.28x | `opencv_ui` |
| `SUBTRACT` | 5.20x | 1.31x | `opencv_ui` |
| `MULTIPLY` | 5.46x | 1.34x | `opencv_ui` |
| `DIVIDE` | 2.91x | 2.00x | 浮点 `opencv_ui`，整数 `scalar` |

`DIVIDE` 的算子级汇总同时包含整数和浮点。VGA F32 的 OpenCV 领先约
`1.24x-1.36x`；保留的 U8 scalar case 仍领先约 `2.66x-3.97x`，后续只有在明确并统一整数
舍入契约后才允许进入 SIMD。

### 0.4 P-ACC-1 第三组结果

实施前 Mode B 基线：

| 算子 | OpenCV 领先 | 当前 dispatch | upstream SIMD 入口 |
| --- | ---: | --- | --- |
| `CONVERT_SCALE_ABS` | 6.61x | `public_header_baseline` | `convert_scale.simd.hpp::cvtabs_32f` |
| `CONVERT_FP16` | 4.00x | `public_header_baseline` | `convert.simd.hpp::cvt32f16f/cvt16f32f` |
| `SCALE_ADD` | 1.22x | `public_header_baseline` | `matmul.simd.hpp::scaleAdd_32f/64f` |

本组边界：

- `convertScaleAbs` 先覆盖 Mode B 的 `CV_32F -> CV_8U`，支持任意通道、连续 Mat、非连续
  ROI 和 scalar tail。其他输入 depth 保留 scalar；它们应在后续完整转换矩阵中统一处理。
- `convertFp16` 覆盖 `CV_32F -> CV_16S bit container`、`CV_16S -> CV_32F` 和
  `CV_16F -> CV_32F`。vendored UI 的 `cv::hfloat` 必须与项目 2 字节 `hfloat` 共用布局，
  并增加 `sizeof(hfloat)==2` 编译期门禁。
- `scaleAdd` 仅原型验证 upstream 的浮点 `v_muladd` 路径；只有相对 scalar 稳定提升超过
  `5%` 才允许接入公共 dispatch。
- 三项公共 API 都必须设置实际 `opencv_ui`/`scalar` dispatch tag。Mode A 增加缺失的
  `SCALE_ADD` 和 `CONVERT_FP16` case；Mode B 不再把第三组标记为 baseline。

验收 gate：

- `convertScaleAbs` 覆盖 ties-to-even、饱和、NaN/Inf、负 alpha、C3 ROI、tail 和 alias。
- `convertFp16` 覆盖正负零、subnormal、最大有限值、Inf、NaN、双向转换、ROI 和 tail。
- `scaleAdd` 覆盖 F32/F64、C3 ROI、tail、`dst==src1`、`dst==src2`；整数与短行确认 fallback。
- `SCALE_ADD` 只有 Mode A 相对 scalar 稳定提升超过 `5%` 才保留 UI 接入。
- Apple ARM 全量 CTest、Mode A/Mode B 和 SSE2/AVX2 交叉编译必须通过；真实 x86 运行仍作为
  跨平台未关闭 gate。

已完成：

- 新增 ODR-safe `core/detail/math_ui.hpp`，直接使用 vendored OpenCV UI 实现
  `CV_32F -> CV_8U` 的 `convertScaleAbs` 和 FP16 双向转换。
- 修正 vendored `cv::hfloat` 与项目 `hfloat` 的布局关系，并按 AArch64 NEON 或 x86 F16C
  启用 UI 原生 FP16 pack/unpack；编译期保证 `sizeof(hfloat) == 2`。
- 连续 Mat 合并为单行处理；FP16 UI 循环执行四路展开，ROI、非连续 Mat 和 scalar tail
  继续复用同一正确性契约。
- `scaleAdd` UI 原型在 Apple ARM VGA F32C1 上只有 scalar 的约 `0.90x`，未达到性能 gate，
  因而已移除原型接入。公共 API 保持 scalar，dispatch 也如实报告 `scalar`。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`；转换 kernel 的 checksum 一致：

| 类型/尺寸 | 算子 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | 最终 dispatch |
| --- | --- | ---: | ---: | ---: | --- |
| `CV_32FC1 640x480` | `CONVERT_SCALE_ABS` | 0.152006 | 0.025321 | 6.00x | `opencv_ui` |
| `CV_32FC1 640x480` | `CONVERT_FP16` | 0.071794 | 0.014515 | 4.95x | `opencv_ui` |
| `CV_32FC1 640x480` | `SCALE_ADD` 原型 | 0.035885 | 0.039958 | 0.90x | `scalar`，UI 被拒绝 |

Mode B 已用 `full` profile 重跑全部 `321` 个 case：

| 算子 | 实施前差距 | 当前差距 | 当前 dispatch |
| --- | ---: | ---: | --- |
| `CONVERT_SCALE_ABS` | OpenCV 领先 6.61x | OpenCV 领先 1.26x | `opencv_ui` |
| `CONVERT_FP16` | OpenCV 领先 4.00x | CVH 领先 1.21x | `opencv_ui` |
| `SCALE_ADD` | OpenCV 领先 1.22x | OpenCV 领先 1.21x | `scalar` |

Apple ARM Release 全量 CTest `16/16` 通过。Apple Clang x86_64 的 SSE2、AVX2 和
AVX2+F16C 编译实例化均通过；真实 x86 correctness 与 Mode A 仍是未关闭 gate。

### 0.5 P-ACC-1 第四组结果

实施前基线：

| 范围 | Mode A Apple ARM 基线 | Mode B 基线 | 当前 dispatch |
| --- | ---: | ---: | --- |
| `IN_RANGE`，VGA `CV_8UC3` scalar bounds | 1.927677 ms | OpenCV 领先 13.26x | `scalar` |
| Mat-Scalar/Scalar-Mat 算术 | 尚无独立 case | 尚无独立 case | `scalar` |
| masked bitwise | 尚无独立 case | 尚无独立 case | `scalar` |

本组边界：

- `IN_RANGE` 复用 upstream 的 compare、mask pack 和多通道 reduce 结构。UI 先覆盖
  `CV_8U/8S/16U/16S/32S/32U/32F` 的 Mat bounds 和 Scalar bounds；`CV_16F/64F`
  保留 scalar。
- Scalar bounds 必须保持当前契约：整数边界按原始 double 做闭区间判断，不能简单
  `saturate_cast` 后改变非整数、越界或反向区间结果。
- Mat-Scalar/Scalar-Mat 建立按 channel 周期生成 UI 常量向量的共享 row helper。
  `add/subtract/multiply` 覆盖与 Mat-Mat 相同的整数和浮点类型；浮点 `divide`、
  整数 `absdiff/min/max` 按前三组已验证范围接入。整数 `divide`、FP16 和存在 NaN
  语义差异的浮点 `absdiff/min/max` 保留 scalar。
- masked bitwise 使用单通道 pixel mask 和 `v_select` 合并计算结果与旧 `dst`。覆盖
  Mat-Mat、Mat-Scalar、Scalar-Mat 和 `bitwise_not`；新分配 `dst` 的未选中位置仍必须
  为零，预分配 `dst` 的未选中位置必须保持不变。
- Mode A 增加 Mat-Scalar/Scalar-Mat、masked Mat-Mat/Mat-Scalar 和 `IN_RANGE`
  Mat/Scalar bounds 变体。每条路径必须报告实际 `opencv_ui` 或 `scalar`。

验收 gate：

- correctness 覆盖 C1/C3/C4、连续 Mat、非连续 ROI、非向量 tail、短行、in-place、
  新分配/预分配 `dst`、非零 mask 值、整数越界/非整数 bounds、NaN/Inf。
- UI 与强制 scalar 的输出逐字节一致；浮点算术仅允许既有契约中已经接受的数值容差。
- 每个新增 fast-path 的 VGA 代表 case 相对 scalar 必须稳定提升超过 `5%`，否则撤销该
  路径并记录为性能 gate 拒绝。
- Apple ARM 全量 CTest、Mode A/Mode B 和 SSE2/AVX2 交叉编译必须通过；真实 x86 运行
  继续作为跨平台未关闭 gate。

已完成：

- `core/detail/arithm_ui.hpp` 新增按 channel 周期预计算常量向量的 broadcast row helper，
  Mat-Scalar/Scalar-Mat 不再在逐元素循环中执行 channel 取模和 double 运算。
- `IN_RANGE` 迁移 upstream 的 8 位直接 store、16 位 pack、32 位/F32 两级 pack，并对
  多通道逐元素 mask 做单通道归约。整数 Scalar bounds 使用 `ceil(lower)` /
  `floor(upper)` 与越界区间判定，保持原有 double 闭区间语义。
- masked bitwise 的首个“展开完整 byte mask”原型在 VGA 上只有 scalar 的
  `0.58x-0.70x`，已被性能 gate 拒绝。最终实现对 1/2/3/4 byte 像素使用
  `v_load_deinterleave`、单个 pixel mask 和 `v_select`，更大像素继续走 scalar。
- correctness 覆盖六种整数 broadcast depth、F32 双向 divide、C3 ROI/tail/alias、
  Mat/Scalar bounds、NaN/Inf、mask 值 `0/1/255`、新分配/预分配目标以及 raw F32 bits。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`，checksum 全部一致；下表为 VGA 代表 case：

| 类型 | 算子/变体 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | dispatch |
| --- | --- | ---: | ---: | ---: | --- |
| `CV_8UC3` | `IN_RANGE` Scalar bounds | 1.967183 | 0.177840 | 11.06x | `opencv_ui` |
| `CV_8UC3` | `IN_RANGE` Mat bounds | 1.540800 | 0.125346 | 12.29x | `opencv_ui` |
| `CV_8UC3` | `ADD` Mat-Scalar | 0.171794 | 0.029329 | 5.86x | `opencv_ui` |
| `CV_8UC3` | `SUBTRACT` Scalar-Mat | 0.151350 | 0.027569 | 5.49x | `opencv_ui` |
| `CV_8UC3` | `MULTIPLY` Mat-Scalar | 0.172327 | 0.027146 | 6.35x | `opencv_ui` |
| `CV_8UC3` | `ABSDIFF` Mat-Scalar | 0.233763 | 0.028142 | 8.31x | `opencv_ui` |
| `CV_8UC3` | `MIN` Mat-Scalar | 0.215554 | 0.027469 | 7.85x | `opencv_ui` |
| `CV_8UC3` | `MAX` Mat-Scalar | 0.248552 | 0.028602 | 8.69x | `opencv_ui` |
| `CV_8UC3` | masked Mat-Mat `BITWISE_AND` | 0.295129 | 0.027856 | 10.59x | `opencv_ui` |
| `CV_8UC3` | masked Mat-Scalar `BITWISE_XOR` | 0.302054 | 0.027160 | 11.12x | `opencv_ui` |
| `CV_32FC1` | `DIVIDE` Scalar-Mat | 0.084604 | 0.036852 | 2.30x | `opencv_ui` |
| `CV_32FC1` | masked Mat-Mat `BITWISE_AND` | 0.381275 | 0.046721 | 8.16x | `opencv_ui` |

整数 Mat-Scalar `divide`、FP16、浮点 `absdiff/min/max`、短行和 masked
`elemSize() > 4` 均继续报告 `scalar`。

Mode B 已用 `full` profile 重跑全部 `321` 个 case，结果仍为 `320 OK + 1 UNSUPPORTED`：

| 算子 | 实施前差距 | 当前差距 | 当前 dispatch |
| --- | ---: | ---: | --- |
| `IN_RANGE` | OpenCV 领先 13.26x | OpenCV 领先 1.63x | `opencv_ui` |

Apple ARM Release 全量 CTest `16/16` 通过。UI-disabled、Apple Clang x86_64 SSE2 和 AVX2
编译实例化均通过；真实 x86 correctness 与 Mode A 仍是未关闭 gate。

### 0.6 P-ACC-1 第五组结果

实施前 Mode B 基线：

| 算子 | CVH ms | OpenCV ms | OpenCV 领先 | 当前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `PATCH_NANS` | 0.042508 | 0.019933 | 2.13x | `public_header_baseline` |
| `EXP` | 0.350429 | 0.081871 | 4.28x | `public_header_baseline` |
| `LOG` | 0.482554 | 0.139317 | 3.46x | `public_header_baseline` |
| `POW`，`power=1.75` | 1.519492 | 0.324700 | 4.68x | `public_header_baseline` |

本组边界：

- `PATCH_NANS` 只覆盖公共 API 已支持的 `CV_32F`。按 IEEE-754 bit pattern 判断 NaN，
  保留正负 Inf 和所有非 NaN bit，replacement 只做一次 `double -> float` 转换。
- `EXP/LOG` 先覆盖 `CV_32F`，直接使用 vendored UI 的 `cv::v_exp/cv::v_log`；
  `CV_64F` 暂时保留现有 `std::exp/std::log`，避免在未建立双精度误差预算前改变精度。
- `POW` 的 F32 整数指数使用 exponentiation by squaring；非整数指数仅对正有限值使用
  `v_exp(v_log(x) * power)`。含零、负数、NaN 或 Inf 的 vector block 回落到
  `std::pow`，以保持当前公共 API 的符号零和特殊值语义。`CV_64F` 保留 scalar。
- 所有 UI row kernel 支持连续 Mat 合并、非连续 ROI、in-place、非向量 tail 和短行
  fallback，并报告实际 `opencv_ui` 或 `scalar` dispatch。

实施步骤：

| Step | 状态 | 验收标准 |
| --- | --- | --- |
| P-ACC-1.5.0 语义与基线审计 | 已完成 | 记录四项 Mode B 基线；定位 upstream 和 vendored UI 入口；明确特殊值规则 |
| P-ACC-1.5.1 `PATCH_NANS` | 已完成 | bit-exact 保留非 NaN；ROI/tail/short-row；代表 case 相对 scalar 提升超过 5% |
| P-ACC-1.5.2 `EXP/LOG` | 已完成 | 有限值误差、零/负数/NaN/Inf、ROI/alias；两项均通过性能 gate |
| P-ACC-1.5.3 `POW` | 已完成 | 正负整数指数、通用指数、负数/零/NaN/Inf、ROI/alias；通过性能 gate |
| P-ACC-1.5.4 全量 gate 与报告 | 已完成 | CTest、Mode A、Mode B、UI-disabled、SSE2/AVX2 编译全部通过并更新报告 |

已完成：

- `math_ui.hpp` 直接使用 vendored `cv::v_exp/cv::v_log`，没有增加项目自定义 SIMD adapter
  或外部数学库。普通 F32 block 进入 UI，超出受控范围或包含特殊值的 block 回落到
  `std::exp/std::log/std::pow`。
- `PATCH_NANS` 迁移 upstream 的 IEEE-754 exponent/mantissa bit mask 与 `v_select` 结构；
  无 NaN block 避免写回，非 NaN bit pattern 保持不变。
- `POW` 对 F32 整数指数使用 exponentiation by squaring，对正有限 F32 通用指数使用
  `v_exp(v_log(x) * power)`；F64、非有限指数和短行继续使用 scalar。
- correctness 新增强制 scalar 对照，覆盖正负零、subnormal、NaN payload、Inf、负输入、
  正负整数指数、通用指数、C3 非连续 ROI、tail、in-place 和短行 fallback。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`；下表为 VGA `CV_32FC1` 中位数：

| 算子/变体 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | dispatch |
| --- | ---: | ---: | ---: | --- |
| `PATCH_NANS`，one NaN | 0.036017 | 0.018642 | 1.93x | `opencv_ui` |
| `EXP` | 0.304302 | 0.153010 | 1.99x | `opencv_ui` |
| `LOG`，positive F32 | 0.442140 | 0.212171 | 2.08x | `opencv_ui` |
| `POW`，`power=1.75` | 1.328683 | 0.487423 | 2.73x | `opencv_ui` |
| `POW`，`power=3` | 2.197325 | 0.035754 | 61.46x | `opencv_ui` |

Mode B 已用 `full` profile 重跑全部 `321` 个 case，结果为 `320 OK + 1 UNSUPPORTED`：

| 算子 | 实施前 OpenCV 领先 | 当前 OpenCV 领先 | 当前 dispatch |
| --- | ---: | ---: | --- |
| `PATCH_NANS` | 2.13x | 1.24x | `opencv_ui` |
| `EXP` | 4.28x | 2.09x | `opencv_ui` |
| `LOG` | 3.46x | 1.72x | `opencv_ui` |
| `POW`，`power=1.75` | 4.68x | 1.63x | `opencv_ui` |

Apple ARM Release 全量 CTest `16/16` 通过。UI-disabled、Apple Clang x86_64 SSE2 和 AVX2
编译实例化均通过；真实 x86 correctness 与 Mode A 仍是未关闭 gate。

### 0.7 P-ACC-2.0/2.1 结果

实施前 Mode B 基线：

| 算子/输入分布 | CVH ms | OpenCV ms | OpenCV 领先 | 当前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `COUNT_NON_ZERO`，VGA random U8C1 | 0.286433 | 0.009550 | 29.99x | `public_header_baseline` |
| `HAS_NON_ZERO`，VGA random U8C1 | 0.280704 | 0.000013 | 22222.22x | `public_header_baseline` |

`HAS_NON_ZERO` 的 Mode B 输入首块通常已经存在非零值，因此该数字主要暴露实施前缺少
early-exit，不能代表全零完整扫描吞吐量。本次先增加了全零、首元素命中和尾元素命中的
独立 Mode A case。

| Step | 状态 | 当前验收点 |
| --- | --- | --- |
| P-ACC-2.0 | Apple ARM 已完成 | `reduce_ui.hpp`、全部 reduction 公共 dispatch tag、代表性 Mode A 行和实施前非零基线已建立 |
| P-ACC-2.1 | Apple ARM 已完成，x86 运行待验证 | `countNonZero/hasNonZero` UI、语义测试、性能 gate 与 Mode B 已通过 |

实施边界：

- 新增 ODR-safe `core/detail/reduce_ui.hpp`，直接使用 vendored OpenCV UI，没有增加自定义
  SIMD adapter。
- `countNonZero` 使用 upstream 的 compare/pack、8/16/32-bit 分段 widen 累加和 scalar tail；
  公共 `int` 返回前继续保留溢出检查。
- `hasNonZero` 按 depth 使用 2/4/8/16 个 vector 的 block，并在首个命中 block 立即返回；
  F32/F64 使用“按位 OR 后与零比较”，同时保留 NaN 为非零、`+0/-0` 为零的语义。
- UI 覆盖 `CV_8U/8S/16U/16S/32S/32U/32F`，`CV_SIMD_64F` 可用时覆盖 `CV_64F`；
  `CV_16F` 和短行明确回退 scalar。
- Mode A 现在记录本阶段全部操作族的代表性 case，包括统计、极值、`reduce/normalize`、
  非零检测、`findNonZero` 和 `reduceArgMin/Max`，并输出真实 dispatch。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`；下表为 VGA `CV_8UC1`：

| 算子/分布 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | 实际 dispatch |
| --- | ---: | ---: | ---: | --- |
| `COUNT_NON_ZERO`，random dense | 0.363702 | 0.008950 | 40.64x | `opencv_ui` |
| `HAS_NON_ZERO`，all zero | 0.290610 | 0.003558 | 81.68x | `opencv_ui` |
| `HAS_NON_ZERO`，first nonzero | 0.000010 | 0.000008 | 1.25x | `opencv_ui` |
| `HAS_NON_ZERO`，tail nonzero | 0.290290 | 0.003335 | 87.04x | `opencv_ui` |

首元素 case 的耗时已从“随图像总像素线性增长”变为常数级。全零和尾命中 case 才代表完整
扫描吞吐量，不能用首命中的纳秒级数字代替。

Mode B 已用 `full` profile 重跑全部 `321` 个 case：

| 算子 | 实施前 CVH/OpenCV ms | 当前 CVH/OpenCV ms | 当前差距 | 当前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `COUNT_NON_ZERO` | 0.286433 / 0.009550 | 0.009971 / 0.009954 | OpenCV 1.00x | `opencv_ui` |
| `HAS_NON_ZERO` | 0.280704 / 0.000013 | 0.000013 / 0.000013 | 持平 | `opencv_ui` |

正确性覆盖全部公开 depth、连续 Mat、非连续 ROI、稠密分布、短行、tail、首/中/尾命中、
`+0/-0/NaN/Inf` 和强制 scalar/auto 公共入口。Apple ARM Release CTest `16/16`、
header-only contract `5/5`、UI-disabled Core `168 passed / 14 skipped` 通过；Apple Clang
x86_64 SSE2 和 AVX2 模板实例化通过，真实 x86 correctness 与 Mode A 仍是未关闭 gate。

### 0.8 P-ACC-2.2 基线与状态

P-ACC-2.2 开始前冻结的 Apple ARM Release Mode A scalar 基线如下。测试参数为
`quick` profile、`warmup=2`、`iters=20`、`repeats=5`：

| 算子/输入 | Scalar ms | 实施前 dispatch |
| --- | ---: | --- |
| `SUM`，VGA U8C1 | 0.303869 | `scalar` |
| `MEAN`，VGA U8C1 | 0.309435 | `scalar` |
| `MEAN_STDDEV`，VGA U8C1 | 1.158800 | `scalar` |
| `SUM`，VGA U8C3 | 1.975717 | `scalar` |
| `MEAN`，VGA U8C3 | 1.969333 | `scalar` |
| `MEAN_STDDEV`，VGA U8C3 | 2.676194 | `scalar` |
| `SUM`，VGA F32C1 | 0.669840 | `scalar` |
| `MEAN`，VGA F32C1 | 0.666269 | `scalar` |
| `MEAN_STDDEV`，VGA F32C1 | 2.410698 | `scalar` |

当前 Mode B representative case 均为 VGA F32C3：

| 算子 | CVH ms | OpenCV ms | OpenCV 领先 | 实施前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `SUM` | 1.508767 | 0.230246 | 6.55x | `scalar` |
| `MEAN` | 1.493621 | 0.230671 | 6.48x | `scalar` |
| `MEAN_STD_DEV` | 1.794875 | 0.149125 | 12.04x | `scalar` |

| 子项 | 状态 | 验收点 |
| --- | --- | --- |
| C1-C4 channel-aware sum/count | Apple ARM 已完成 | 覆盖无 mask、全空/全选/稀疏 mask、ROI 与 tail |
| `sum/mean` 公共入口 | Apple ARM 已完成 | 共享同一 sum/count kernel，实际 dispatch 为 `opencv_ui` |
| `meanStdDev` stable block statistics | Apple ARM 已完成 | block 内中心化累计并使用 Chan merge，没有直接相减原始一阶/二阶矩 |
| correctness / Mode A / Mode B | 已完成 | 三项代表路径均超过 5%；Mode B 为 `320 OK + 1` 个既有 `UNSUPPORTED` |
| UI-disabled / SSE2 / AVX2 | 编译 gate 已完成 | fallback 与模板实例化通过；真实 x86 运行仍单独列为未关闭 gate |

实现边界：

- 新增按像素组织的连续/ROI row view，C1 使用 `vx_load`，C2/C3/C4 使用
  `v_load_deinterleave`；各公开整数 depth 在单 vector 内做无溢出的横向累计，跨 vector
  使用 `long double` 合并。
- mask 按 vector block 区分全空、全选和混合；全选 block 进入 UI，混合 block 使用 typed
  scalar，不重新进入逐元素 depth switch。
- `sum/mean` 共用 `SumCount`；`CV_16F` 与短行继续明确回退 scalar。
- `meanStdDev` 以 `2048` pixels 为 block，先得到 block mean，再累计中心化 `M2`，最后用
  Chan 公式合并 block。没有采用 `sumSq/count - mean²`。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`：

| 算子/输入 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | 实际 dispatch |
| --- | ---: | ---: | ---: | --- |
| `SUM`，VGA U8C1 | 0.297065 | 0.012550 | 23.67x | `opencv_ui` |
| `MEAN`，VGA U8C1 | 0.295458 | 0.012219 | 24.18x | `opencv_ui` |
| `MEAN_STDDEV`，VGA U8C1 | 1.081779 | 0.285021 | 3.80x | `opencv_ui` |
| `SUM`，VGA U8C3 | 1.031592 | 0.015408 | 66.95x | `opencv_ui` |
| `MEAN`，VGA U8C3 | 1.025096 | 0.015402 | 66.56x | `opencv_ui` |
| `MEAN_STDDEV`，VGA U8C3 | 1.213373 | 0.360913 | 3.36x | `opencv_ui` |
| `SUM`，VGA F32C1 | 0.315837 | 0.048925 | 6.46x | `opencv_ui` |
| `MEAN`，VGA F32C1 | 0.292858 | 0.048944 | 5.98x | `opencv_ui` |
| `MEAN_STDDEV`，VGA F32C1 | 1.080165 | 0.319985 | 3.38x | `opencv_ui` |

Mode B `full` profile 仍为全部 `321` 个 case，结果为 `320 OK + 1 UNSUPPORTED`：

| 算子，VGA F32C3 | 实施前 CVH/OpenCV ms | 当前 CVH/OpenCV ms | 当前差距 | 当前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `SUM` | 1.508767 / 0.230246 | 0.132496 / 0.264475 | CVH 2.00x | `opencv_ui` |
| `MEAN` | 1.493621 / 0.230671 | 0.132354 / 0.271246 | CVH 2.05x | `opencv_ui` |
| `MEAN_STD_DEV` | 1.794875 / 0.149125 | 0.577117 / 0.180942 | OpenCV 3.19x | `opencv_ui` |

正确性覆盖全部公开 depth、C1-C4、连续 Mat、非连续 ROI、tail、短行、mask
全空/全选/稀疏、常量、大均值小方差、NaN/Inf 和 32-bit 整数精确累计。header-only
contract `5/5`、CTest `16/16`、Core `184 passed / 2 skipped`、Imgproc `173 passed`、
UI-disabled Core `172 passed / 14 skipped` 通过；Apple Clang x86_64 SSE2 和 AVX2
模板实例化通过。

### 0.9 P-ACC-2.3 基线与状态

P-ACC-2.3 开始前冻结的 Apple ARM Release Mode A scalar 基线如下。测试参数为
`quick` profile、`warmup=2`、`iters=20`、`repeats=5`：

| 算子/输入 | Scalar ms | 实施前 dispatch |
| --- | ---: | --- |
| `MIN_MAX_LOC`，VGA U8C1 | 0.517519 | `scalar` |
| `MIN_MAX_IDX`，VGA U8C1 | 0.516081 | `scalar` |
| `REDUCE_ARG_MIN`，VGA U8C1 axis 1 first | 0.952583 | `scalar` |
| `REDUCE_ARG_MAX`，VGA U8C1 axis 0 last | 0.950233 | `scalar` |
| `MIN_MAX_LOC`，VGA F32C1 | 0.516346 | `scalar` |
| `MIN_MAX_IDX`，VGA F32C1 | 0.513867 | `scalar` |
| `REDUCE_ARG_MIN`，VGA F32C1 axis 1 first | 0.952235 | `scalar` |
| `REDUCE_ARG_MAX`，VGA F32C1 axis 0 last | 0.885021 | `scalar` |

当前 Mode B representative case 均为 VGA F32C1：

| 算子 | CVH ms | OpenCV ms | OpenCV 领先 | 实施前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `MIN_MAX_IDX` | 0.714979 | 0.045733 | 15.63x | `scalar` |
| `MIN_MAX_LOC` | 0.686133 | 0.045017 | 15.24x | `scalar` |
| `REDUCE_ARG_MIN`，axis 0 | 1.144158 | 0.187463 | 6.10x | `scalar` |
| `REDUCE_ARG_MAX`，axis 0 | 1.226404 | 0.186558 | 6.57x | `scalar` |

| 子项 | 状态 | 验收点 |
| --- | --- | --- |
| 值与线性索引共享 UI kernel | Apple ARM 已完成 | 同一次扫描得到极值和索引，不做完整二次扫描 |
| `minMaxIdx/minMaxLoc` 公共入口 | Apple ARM 已完成 | 跳过 NaN、首 tie、mask 与 ND/2D 坐标合同不变 |
| `reduceArgMin/Max` axis 0/1 | Apple ARM 已完成 | first/last tie、NaN 和非连续 ROI 合同不变 |
| correctness / Mode A / Mode B | 已完成 | 每条代表路径均超过 5%；Mode B 为 `320 OK + 1` 个既有 `UNSUPPORTED` |
| UI-disabled / SSE2 / AVX2 | 编译 gate 已完成 | fallback 与模板实例化通过；真实 x86 运行仍单独列为未关闭 gate |

实现边界：

- `minMaxIdx/minMaxLoc` 当前 UI 路径只处理 C1。先按既有合同找到首个 mask 选中且非 NaN
  的 seed，再对后续 block 同时比较 min/max；只有可能改写全局极值的 block 才从已加载
  vector 中恢复具体索引，没有进行完整二次扫描。
- mask 全选 block 使用 UI，混合 block 使用 typed scalar；全空 mask、全 NaN、`CV_16F`、
  短行和多通道 values-only `minMaxIdx` 明确回退 scalar。
- `reduceArg*` axis 1 在每个 vector lane 内同步维护候选值和真实列索引，再按 first/last
  规则合并 lanes；axis 0 跨行比较一组连续列，并从同一次比较 mask 更新行索引。
- NaN 继续保留既有行为：`minMax*` 跳过 NaN；`reduceArg*` 以轴首值初始化，轴首为 NaN
  时保持索引 0。`+0/-0` 按相等值处理，first/last 只由索引规则决定。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`：

| 算子/输入 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 | 实际 dispatch |
| --- | ---: | ---: | ---: | --- |
| `MIN_MAX_LOC`，VGA U8C1 | 0.478971 | 0.010944 | 43.77x | `opencv_ui` |
| `MIN_MAX_IDX`，VGA U8C1 | 0.443427 | 0.011123 | 39.87x | `opencv_ui` |
| `REDUCE_ARG_MIN`，VGA U8C1 axis 1 first | 0.947394 | 0.050763 | 18.66x | `opencv_ui` |
| `REDUCE_ARG_MAX`，VGA U8C1 axis 0 last | 0.948808 | 0.059027 | 16.07x | `opencv_ui` |
| `MIN_MAX_LOC`，VGA F32C1 | 0.620458 | 0.050160 | 12.37x | `opencv_ui` |
| `MIN_MAX_IDX`，VGA F32C1 | 0.626340 | 0.050067 | 12.51x | `opencv_ui` |
| `REDUCE_ARG_MIN`，VGA F32C1 axis 1 first | 1.005633 | 0.102062 | 9.85x | `opencv_ui` |
| `REDUCE_ARG_MAX`，VGA F32C1 axis 0 last | 0.951600 | 0.193517 | 4.92x | `opencv_ui` |

Mode B `full` profile 仍为全部 `321` 个 case，结果为 `320 OK + 1 UNSUPPORTED`：

| 算子，VGA F32C1 | 实施前 CVH/OpenCV ms | 当前 CVH/OpenCV ms | 当前差距 | 当前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `MIN_MAX_IDX` | 0.714979 / 0.045733 | 0.100829 / 0.071708 | OpenCV 1.41x | `opencv_ui` |
| `MIN_MAX_LOC` | 0.686133 / 0.045017 | 0.100550 / 0.071413 | OpenCV 1.41x | `opencv_ui` |
| `REDUCE_ARG_MIN`，axis 0 | 1.144158 / 0.187463 | 0.282750 / 0.263592 | OpenCV 1.07x | `opencv_ui` |
| `REDUCE_ARG_MAX`，axis 0 | 1.226404 / 0.186558 | 0.278087 / 0.280525 | CVH 1.01x | `opencv_ui` |

正确性覆盖全部公开 depth、axis 0/1、first/last tie、连续 Mat、非连续 ROI、tail、
短行、mask、全 NaN、NaN 首元素、Inf、signed zero 和 src/dst alias。header-only
contract `5/5`、CTest `16/16`、Core `188 passed / 2 skipped`、Imgproc `173 passed`、
UI-disabled Core `176 passed / 14 skipped` 通过；Apple Clang x86_64 SSE2 和 AVX2
模板实例化通过。

### 0.10 P-ACC-2.4 基线与状态

P-ACC-2.4 开始前冻结的 Apple ARM Release Mode A 基线如下。测试参数为 `quick`
profile、`warmup=2`、`iters=20`、`repeats=5`；Auto 与强制 scalar 都命中
`scalar`，下表记录强制 scalar 中位数：

| 算子/输入 | Scalar ms | 实施前 dispatch |
| --- | ---: | --- |
| `NORM L2`，VGA U8C1，threads=1 | 0.376458 | `scalar` |
| `NORMALIZE L2`，VGA U8C1，threads=1 | 0.889702 | `scalar` |
| `NORM L2`，VGA F32C1，threads=1 | 0.530556 | `scalar` |
| `NORMALIZE L2`，VGA F32C1，threads=1 | 1.224242 | `scalar` |

当前 Mode B representative case 均为 VGA F32C1：

| 算子 | CVH ms | OpenCV ms | OpenCV 领先 | 实施前 dispatch |
| --- | ---: | ---: | ---: | --- |
| `NORM L2` | 0.966146 | 0.075317 | 12.83x | `scalar` |
| `NORMALIZE L2` | 1.997821 | 0.107175 | 18.64x | `scalar` |

| 子项 | 状态 | 验收点 |
| --- | --- | --- |
| 单输入 L1/L2/Inf UI reduction | Apple ARM 已完成 | U8/F32、C1-C4 共用连续 selected-run；NaN/Inf、整数极值和 mask 合同不变 |
| 双输入 L1/L2/Inf UI reduction | Apple ARM 已完成 | U8 使用无符号 absdiff；F32 先 widen 到 F64 再求差 |
| `norm` 公共入口 | Apple ARM 已完成 | 只有实际处理 vector block 才报告 `opencv_ui` |
| `normalize` reduction 组合 | Apple ARM 已完成 | `NORM_MINMAX` 复用 P-ACC-2.3 extrema；其他类型复用本组 `norm` |
| `normalize` apply-scale | Apple ARM 已完成 | F32 同深度输出进入 UI；其他转换组合继续 scalar fallback |
| correctness / Mode A / Mode B | 已完成 | 每种 norm 类型独立行；Mode B 为 `328 OK + 1` 个既有 `UNSUPPORTED` |
| UI-disabled / SSE2 / AVX2 | 编译 gate 已完成 | fallback、模板实例化与 header-only 多 TU 通过；真实 x86 运行单列 |

实现边界：

- 先覆盖 benchmark 主路径 `CV_8U/CV_32F`，并复用 `PixelRows` 的连续 Mat、ROI、
  C1-C4 和 mask 行视图；其他 depth 在通过数值与性能 gate 后再接入，未接入者明确
  fallback。
- mask 以连续 selected-run 为单位进入 UI，run 的 scalar 数量包含全部 channels；
  稀疏短 run 使用 typed scalar tail，但没有任何 vector block 时整次调用回退现有 scalar。
- L1/L2 使用足够宽的中间累计，Inf 使用绝对值最大归约；浮点路径显式传播 NaN，
  保持现有公共合同。
- `normalize` 分开记录归约和 apply-scale 成本。不能仅凭组合路径变快就隐藏其中一个阶段
  的回退或回归。

落地结果：

- 单输入和双输入共用 `NormResult`、selected-run 遍历及相同 scalar tail。U8 的 L1/L2
  使用分块 dot-product，限制 `uint32` lane 累计不溢出；F32 在 F64 lane 中执行差值、
  绝对值、平方和累计。
- `normalize` 的 L1/L2/Inf 直接复用公共 `norm`；C1 `NORM_MINMAX` 复用 P-ACC-2.3
  extrema。F32 apply-scale 在 F64 lane 中执行 scale/shift 后 pack 回 F32，保持现有
  double 中间计算合同。
- benchmark 增加六条 `NORM` 变体、四条 `NORMALIZE` 变体和独立
  `NORMALIZE_APPLY_SCALE`，Mode B 同步扩成六条 norm 与四条 normalize 对照。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`，下表均为 VGA C1、threads=1，checksum 全部一致：

| 类型 | 算子/变体 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 |
| --- | --- | ---: | ---: | ---: |
| U8 | `NORM INF` 单输入 | 0.382556 | 0.008967 | 42.66x |
| U8 | `NORM L1` 单输入 | 0.445110 | 0.026235 | 16.97x |
| U8 | `NORM L2` 单输入 | 0.392912 | 0.035810 | 10.97x |
| U8 | `NORM INF` 双输入 | 0.519819 | 0.009329 | 55.72x |
| U8 | `NORM L1` 双输入 | 0.543246 | 0.027225 | 19.95x |
| U8 | `NORM L2` 双输入 | 0.530754 | 0.035631 | 14.90x |
| F32 | `NORM INF` 单输入 | 0.781310 | 0.130963 | 5.97x |
| F32 | `NORM L1` 单输入 | 0.769162 | 0.132671 | 5.80x |
| F32 | `NORM L2` 单输入 | 0.785631 | 0.122908 | 6.39x |
| F32 | `NORM INF` 双输入 | 1.007867 | 0.141185 | 7.14x |
| F32 | `NORM L1` 双输入 | 1.069387 | 0.141335 | 7.57x |
| F32 | `NORM L2` 双输入 | 1.074710 | 0.165475 | 6.49x |
| U8 | `NORMALIZE L2` | 1.069894 | 0.578785 | 1.85x |
| F32 | `NORMALIZE INF` | 1.637860 | 0.192746 | 8.50x |
| F32 | `NORMALIZE L1` | 1.691967 | 0.193027 | 8.77x |
| F32 | `NORMALIZE L2` | 1.695894 | 0.173110 | 9.80x |
| F32 | `NORMALIZE MINMAX` | 1.697846 | 0.125031 | 13.58x |
| F32 | apply-scale only | 0.901623 | 0.054963 | 16.40x |

Mode B `full` profile 扩展为全部 `329` 个 case，结果为 `328 OK + 1 UNSUPPORTED`：

| VGA F32C1 变体 | CVH ms | OpenCV ms | 当前差距 | dispatch |
| --- | ---: | ---: | ---: | --- |
| `NORM INF` 单输入 | 0.145588 | 0.014800 | OpenCV 9.84x | `opencv_ui` |
| `NORM L1` 单输入 | 0.129662 | 0.034067 | OpenCV 3.81x | `opencv_ui` |
| `NORM L2` 单输入 | 0.131183 | 0.051533 | OpenCV 2.55x | `opencv_ui` |
| `NORM INF` 双输入 | 0.136846 | 0.029746 | OpenCV 4.60x | `opencv_ui` |
| `NORM L1` 双输入 | 0.139092 | 0.039296 | OpenCV 3.54x | `opencv_ui` |
| `NORM L2` 双输入 | 0.163558 | 0.052358 | OpenCV 3.12x | `opencv_ui` |
| `NORMALIZE INF` | 0.169213 | 0.040904 | OpenCV 4.14x | `opencv_ui` |
| `NORMALIZE L1` | 0.169521 | 0.059150 | OpenCV 2.87x | `opencv_ui` |
| `NORMALIZE L2` | 0.175225 | 0.074858 | OpenCV 2.34x | `opencv_ui` |
| `NORMALIZE MINMAX` | 0.112971 | 0.078262 | OpenCV 1.44x | `opencv_ui` |

F32 `NORM INF` 虽通过相对 scalar 的 5% gate，但与 upstream 仍有明显差距；后续可单独
评估不需要 F64 widening 的单输入 max-abs 特化，不阻塞 P-ACC-2.5。

正确性覆盖 U8/F32、单/双输入、L1/L2/Inf、C1-C4、连续 Mat、非连续 ROI、tail、短行、
mask selected-run、NaN/Inf、U8 最大差值、alias、dtype 和常量 `NORM_MINMAX`。
header-only contract `5/5`、CTest `16/16`、Core `191 passed / 2 skipped`、
UI-disabled Core `179 passed / 14 skipped` 通过；Apple Clang x86_64 SSE2 和 AVX2
模板实例化通过。

### 0.11 P-ACC-2.5 基线与状态

P-ACC-2.5 开始前冻结的 Apple ARM Release Mode A scalar 基线如下。测试参数为
`quick` profile、`warmup=2`、`iters=20`、`repeats=5`，所有 Auto case 也仍报告
`scalar`：

| 算子/输入 | Scalar ms | 实施前 dispatch |
| --- | ---: | --- |
| `REDUCE SUM` axis 0，VGA U8C1，F64 输出 | 0.657383 | `scalar` |
| `REDUCE SUM` axis 0，VGA U8C3，F64 输出 | 1.941362 | `scalar` |
| `REDUCE SUM` axis 0，VGA F32C1，F64 输出 | 0.669071 | `scalar` |

当前 Mode B 的 VGA F32C1 axis 0 `SUM` 为 CVH `0.995121 ms`、OpenCV
`0.019358 ms`，OpenCV 领先 `51.41x`。

| Step | 状态 | 验收标准 |
| --- | --- | --- |
| P-ACC-2.5.0 语义、upstream 与基线审计 | 已完成 | 明确 axis 0/1 布局、五种 rtype、dtype/饱和/NaN/Inf/alias 合同并冻结 Mode A/Mode B 基线 |
| P-ACC-2.5.1 axis 1 行内归约 | Apple ARM 已完成 | C1/C3 连续行按 typed pointer 和 UI block 归约；奇数宽度与短行 fallback 正确 |
| P-ACC-2.5.2 axis 0 跨行归约 | Apple ARM 已完成 | 对连续列 block 跨 stride 累计，不在内层调用 `read_scalar` |
| P-ACC-2.5.3 公共入口与正确性 gate | 已完成 | `SUM/AVG/MAX/MIN/SUM2`、axis 0/1、ROI、dtype、饱和、NaN/Inf 和 alias 与强制 scalar 一致 |
| P-ACC-2.5.4 Mode A 性能 gate | Apple ARM 已完成 | 每条接入路径实际报告 `opencv_ui`，代表 case 相对 scalar 提升超过 5% |
| P-ACC-2.5.5 全量 gate 与 Mode B 报告 | 已完成 | CTest、header-only、UI-disabled、SSE2/AVX2 编译通过；Mode B 无新增 unsupported 并更新报告 |

实施边界：

- axis 1 以一行的交错 channel 数据为输入，C1 直接 load，C2/C3/C4 使用
  `v_load_deinterleave`；在 vector block 内归约，tail 使用 typed scalar。
- axis 0 以一组连续列/channel scalar 为 tile，首行初始化向量 accumulator，后续行通过
  `step(0)` 加载同一列块；禁止在跨行内循环调用 depth switch 形式的 `read_scalar`。
- `SUM/AVG/SUM2` 使用足够宽的中间累计后统一调用现有 `write_scalar`；
  `MAX/MIN` 保持首值初始化和现有浮点比较语义。
- 只有实际处理至少一个 vector block 才报告 `opencv_ui`；不支持的 depth、短行和
  UI-disabled 构建继续保留 scalar fallback。

落地结果：

- UI fast-path 覆盖 U8/F32、C1-C4 和五种 rtype；其他公开 depth 继续使用原 scalar
  合同，输出 depth 转换仍统一经过 `write_scalar`。
- axis 1 对 U8 使用 deinterleave 与 widen/dot 横向归约；F32 `SUM/AVG/SUM2` 在每个
  channel 内保持两组 F64 vector accumulator，整行结束后只做一次横向归约。
- axis 0 对 U8 逐级 widen 到 U32 并按最多 `65535` 行分段，避免 `SUM2` lane 溢出；
  F32 对连续列块做 4 路 F64 累计或 8 路 extrema 展开，避免逐 vector 重复扫描全部行。
- 浮点 MAX/MIN 保留首值初始化与 compare-select，axis 1 遇到 NaN 的行使用 typed scalar，
  从而保持首值为 NaN 时结果继续为 NaN、后续 NaN 不覆盖既有 extrema 的合同。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=20`、
`repeats=5`。下表为 VGA F32C1、F64 输出，checksum 全部与强制 scalar 一致：

| 变体 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 |
| --- | ---: | ---: | ---: |
| axis 0 `SUM` | 1.276204 | 0.076415 | 16.70x |
| axis 0 `AVG` | 1.273135 | 0.077315 | 16.47x |
| axis 0 `MAX` | 1.643481 | 0.043758 | 37.56x |
| axis 0 `MIN` | 1.643748 | 0.044021 | 37.34x |
| axis 0 `SUM2` | 1.271144 | 0.075756 | 16.78x |
| axis 1 `SUM` | 1.250748 | 0.059225 | 21.12x |
| axis 1 `AVG` | 1.262296 | 0.058469 | 21.59x |
| axis 1 `MAX` | 1.639117 | 0.152379 | 10.76x |
| axis 1 `MIN` | 1.629821 | 0.150358 | 10.84x |
| axis 1 `SUM2` | 1.257981 | 0.078223 | 16.08x |

VGA U8C1 的 10 条路径提升 `7.61x-30.04x`，U8C3 提升 `9.17x-73.53x`，均超过
5% 接入 gate。

Mode B `full` profile 扩展为 `338` 个 case，结果为 `337 OK + 1` 个既有
`UNSUPPORTED`：

| VGA F32C1 变体 | CVH ms | OpenCV ms | 当前差距 |
| --- | ---: | ---: | ---: |
| axis 0 `SUM` | 0.085008 | 0.017817 | OpenCV 4.77x |
| axis 0 `AVG` | 0.084988 | 0.018242 | OpenCV 4.66x |
| axis 0 `MAX` | 0.049183 | 0.022396 | OpenCV 2.20x |
| axis 0 `MIN` | 0.049258 | 0.024546 | OpenCV 2.01x |
| axis 0 `SUM2` | 0.086842 | 0.022596 | OpenCV 3.84x |
| axis 1 `SUM` | 0.076325 | 0.014104 | OpenCV 5.41x |
| axis 1 `AVG` | 0.076329 | 0.014075 | OpenCV 5.42x |
| axis 1 `MAX` | 0.169533 | 0.216537 | CVH 1.28x |
| axis 1 `MIN` | 0.169487 | 0.196483 | CVH 1.16x |
| axis 1 `SUM2` | 0.088029 | 0.317450 | CVH 3.61x |

F32 `SUM/AVG` 继续使用 F64 中间累计以保持项目既有精度；upstream 的 F32 输出路径直接
使用 F32 lane 累加。当前剩余差距不能在没有新精度合同的情况下通过降低累计精度消除。

正确性覆盖 axis 0/1、五种 rtype、U8/F32、C1/C3、连续 Mat、非连续 ROI、奇数 tail、
短行、整数饱和、NaN/Inf、src/dst alias、合法 dtype 和未覆盖 depth fallback。
header-only contract `5/5`、CTest `16/16`、Core `193 passed / 2 skipped`、
UI-disabled Core `181 passed / 14 skipped` 通过；Apple Clang x86_64 SSE2 和 AVX2
模板实例化通过，真实 x86 运行仍单列为未关闭 gate。

### 0.12 P-ACC-2.6 基线与状态

upstream `modules/core/src/count_non_zero.dispatch.cpp` 中的 `findNonZero` 仍是逐元素
scalar 坐标收集，没有可直接迁移的 SIMD kernel。本组复用 P-ACC-2.1 已验证的 UI
零比较语义，并针对输出密度建立稀疏/稠密自适应路径。

| Step | 状态 | 验收标准 |
| --- | --- | --- |
| P-ACC-2.6.0 语义、upstream 与分布基线 | 已完成 | 明确 `+0/-0/NaN/Inf`、row-major、2D C1 和两种输出合同；Mode A 同时冻结全零、尾部单点和随机稠密基线 |
| P-ACC-2.6.1 非零 block 检测与 lane 枚举 | Apple ARM 已完成 | 全零 block 直接跳过；部分命中 block 只枚举命中 lane；连续稠密 block 自适应进入 typed scalar 枚举 |
| P-ACC-2.6.2 公共入口与输出共享 | 已完成 | `vector<Point>` 与 `Mat CV_32SC2` 共用同一次源图扫描；Mat 输出使用连续内存写入，不再逐元素调用 `at` |
| P-ACC-2.6.3 正确性 gate | 已完成 | 覆盖全部公开 depth、全零/首尾/稀疏/稠密、NaN/Inf、ROI、tail、短行、一个 vector 和 vector+1 |
| P-ACC-2.6.4 Mode A/Mode B 性能 gate | 已完成 | 全零和稀疏超过 5%；U8/F32 稠密相对 scalar 分别提升 `1.28x/1.40x`；Mode B 扩展为 `340` 行且无新增 unsupported |
| P-ACC-2.6.5 全量与跨 ISA gate | 已完成 | Core、CTest、header-only、UI-disabled、SSE2/AVX2 模板实例化与日期报告通过 |

实现使用共享 `vector_has_nonzero` block 判定。全零 block 不读取 lane；部分命中 block
通过 `v_store` 后按 lane 递增枚举；一个 block 全部非零或连续四个 block 命中时，剩余
行切换为 typed scalar 枚举，避免稠密 U8 因重复 vector 判定和 lane 检查回退。整个过程
始终按 row、column 递增发射坐标。

Mode A 使用同一 Release 二进制、`quick` profile、`warmup=2`、`iters=100`、
`repeats=3`。下表为 VGA C1，checksum 与强制 scalar 一致：

| Depth / 分布 | Scalar ms | OpenCV UI ms | UI/Scalar 加速 |
| --- | ---: | ---: | ---: |
| U8，全零 | 0.808560 | 0.013819 | 58.51x |
| U8，尾部单点 | 0.807492 | 0.016281 | 49.60x |
| U8，随机稠密 | 0.910828 | 0.713834 | 1.28x |
| F32，全零 | 0.909812 | 0.038477 | 23.65x |
| F32，尾部单点 | 0.913423 | 0.038641 | 23.64x |
| F32，随机稠密 | 0.944271 | 0.672696 | 1.40x |

Mode B `full` profile 扩展为 `340` 个 case，结果为 `339 OK + 1` 个既有
`UNSUPPORTED`：

| VGA U8 分布 | CVH ms | OpenCV ms | 当前差距 |
| --- | ---: | ---: | ---: |
| 全零 | 0.017221 | 0.153183 | CVH 8.90x |
| 尾部单点 | 0.017329 | 0.145612 | CVH 8.40x |
| 随机稠密 | 0.982042 | 0.375179 | OpenCV 2.62x |

稀疏收益不能掩盖稠密输入仍落后 upstream 的事实；P-ACC-2.6 的接入 gate 是相对项目
scalar 无回退，进一步缩小稠密 Mode B 差距可在后续单独优化坐标输出存储。

Core `194 passed / 2 skipped`、UI-disabled Core `181 passed / 15 skipped`、
header-only contract `5/5`、Release CTest `17/17` 通过；Apple Clang x86_64 SSE2
与 AVX2 的 reduction 测试模板实例化通过。真实 x86 correctness 与 Mode A 仍是
未关闭 gate。

### 0.13 P-ACC-2.7 收尾状态

P-ACC-2.7 不增加新算子，只关闭 P-ACC-2 的共享语义、性能回归、报告和平台门禁。
真实 x86 运行需要 x86 主机，不能用 Apple Clang 交叉编译结果替代。

| Step | 状态 | 验收标准 |
| --- | --- | --- |
| P-ACC-2.7.0 收尾范围与证据冻结 | 已完成 | P-ACC-2.1 至 2.6 的公共入口、共享 helper、Mode A/Mode B case 和尚未关闭的平台 gate 已列入本节 |
| P-ACC-2.7.1 共享语义审计 | 已完成 | zero/NaN/Inf、mask selected-run、first/last tie、row-major 坐标、累计精度和实际 dispatch 没有发现跨算子语义泄漏 |
| P-ACC-2.7.2 correctness 与 fallback 回归 | 已完成 | Release Core/CTest、UI-disabled、短行、未覆盖 depth、连续/ROI、alias 与两种输出合同全部通过 |
| P-ACC-2.7.3 Mode A 全矩阵回归 | Apple ARM 已完成 | 整数、索引和 Mat 精确输出 checksum 一致；`meanStdDev` 按既有数值容差验收；吞吐代表 case 无稳定超过 `5%` 回退；纳秒级 early-exit 单独按复杂度验收 |
| P-ACC-2.7.4 Mode B 与报告复核 | 已完成 | `full` profile 为 `339 OK + 1` 个既有 unsupported；逐项保留 upstream 差距，没有用几何平均覆盖单项结论 |
| P-ACC-2.7.5 跨平台门禁与阶段关闭 | 已完成 | header-only、UI-disabled、SSE2/AVX2 模板实例化通过；真实 x86 运行保留为外部未关闭 gate；下一入口更新为 P-ACC-3.1 |

共享 helper 审计结论：

| 共享范围 | 审计结论 |
| --- | --- |
| `SourceRows` / `PixelRows` | 只在线性归约中合并连续 Mat；需要二维坐标的 `findNonZero` 使用真实 row/column，未错误复用展平视图 |
| `SumCount` / `StableStatistics` | mask 部分命中 block 回到逐 lane；整数累计保持宽类型；方差继续使用中心化 block 与 Chan merge |
| `ExtremaResult` | NaN 跳过、signed zero 不覆盖首 tie、mask 全空保持 not-found；ND 坐标仍由线性索引统一还原 |
| `reduceArg*` | axis 0/1 独立；first/last 使用不同 compare/merge；NaN 首元素和后续 NaN 与 scalar 合同一致 |
| `NormResult` / normalize 组合 | NaN/Inf、U8 宽差值和 mask selected-run 保留；只有实际使用 vector block 才报告 UI |
| nonzero block helper | `+0/-0` 为零、NaN/Inf 为非零；`hasNonZero` 保留块级 early-exit，`findNonZero` 保持 row-major |

Mode A 使用同一 Release 二进制、相同输入、`quick` profile、`warmup=2`、
`iters=100`、`repeats=3` 重跑 scalar/auto。共匹配 `225` 条 P-ACC-2 行：

- `223` 条实际进入 `opencv_ui`，其中 `220` 条吞吐 case 提升超过 `5%`。
- `findNonZero` 随机稠密 U8/F32 分别为 `1.067x/1.063x`，仍通过接入 gate。
- `hasNonZero(first_nonzero)` 为 `8-9 ns`，32x32 与 VGA 耗时相同，不随总像素数增长；
  该 case 按 P-ACC-2.1 的 O(1) early-exit gate 验收，不把 `1 ns` 计时差当作吞吐回退。
- `hasNonZero` 全零/尾命中最小提升 `22.39x`；其余 UI 算子族的最小提升如下。

| 算子族 | UI 行数 | 最小 UI/Scalar 加速 |
| --- | ---: | ---: |
| `COUNT_NON_ZERO` | 3 | 19.14x |
| `SUM` / `MEAN` / `MEAN_STDDEV` | 24 | 3.16x |
| `MIN_MAX_IDX` / `MIN_MAX_LOC` | 12 | 9.70x |
| `NORM` | 48 | 4.50x |
| `NORMALIZE` / apply-scale | 32 | 1.44x |
| `REDUCE` | 80 | 5.48x |
| `REDUCE_ARG_MIN/MAX` | 6 | 3.67x |

8 条 raw-bit checksum 差异全部来自 `meanStdDev`：scalar 使用逐样本 Welford，UI 使用
中心化 block 与 Chan merge，浮点累计顺序不同。benchmark 的 checksum 只是防止输出被
优化掉的 guard，数值等价由 absolute/relative tolerance 测试验收；其余 `217` 条输出
checksum 完全一致。

Mode B 日期报告保持 `340` 行、`339 OK + 1` 个既有 `UNSUPPORTED`。P-ACC-2 的
32 条有效行全部保留：CVH 在 `SUM/MEAN/COUNT_NON_ZERO/HAS_NON_ZERO`、稀疏
`findNonZero` 和部分 `REDUCE` 上领先；upstream 在 `MEAN_STD_DEV`、极值、norm/normalize
及稠密 `findNonZero` 上仍领先，继续作为 P-ACC-3 之后的优化输入。

最终门禁为 Core `194 passed / 2 skipped`、UI-disabled Core
`181 passed / 15 skipped`、Release CTest `17/17`、header-only contract `5/5`。
Apple Clang 已分别生成 SSE2 与 AVX2 的 x86_64 reduction object；这只证明模板实例化，
真实 x86 correctness 与 Mode A 仍未关闭。

### 0.14 P-ACC-3 至 P-ACC-7 推进状态

本节是后续批次的实时状态表。每个子批次必须依次关闭实现、correctness、Mode A、
Mode B 和 header-only/cross-ISA gate，未获得真实 x86 机器的项目只能标记为
“Apple ARM 已完成，x86 运行待验证”。

| Step | 状态 | 验收标准 |
| --- | --- | --- |
| P-ACC-3.1 copy/setTo/clone | Apple ARM 已完成，x86 运行待验证 | 连续与 ROI、mask、alias、reuse/recreate 正确；消除逐像素地址计算；代表 case 相对 scalar 无超过 5% 回退 |
| P-ACC-3.2 transpose/flip/rotate | Apple ARM 已完成，x86 运行待验证 | tile/连续行路径覆盖常用 depth/channel；奇偶尺寸、in-place/alias 与组合语义正确 |
| P-ACC-3.3 channel operations | Apple ARM 已完成，x86 运行待验证 | C1/C3/C4 常见映射走 deinterleave/interleave 或连续 byte kernel；泛型映射保留正确 fallback |
| P-ACC-3.4 repeat/broadcast/concat | Apple ARM 已完成 | 使用块复制和预分配；ND shape、ROI、单行/单列及 alias 正确 |
| P-ACC-3.5 GEMM | Apple ARM 已完成，x86 运行待验证 | 现有 F32 activation API 的 NN/NT、transpose、batch broadcast 与 packed-B 正确；UI kernel 相对 scalar 有稳定收益 |
| P-ACC-4.1 filter engine | Apple ARM 已完成，x86 运行待验证 | box/sqrBox/Gaussian/filter2D/sepFilter2D 共享行列缓存；border、C1/C3/C4、8U/32F 正确 |
| P-ACC-4.2 derivatives | Apple ARM 已完成，x86 运行待验证 | Scharr/Laplacian/spatialGradient 复用 filter 底座；Sobel 保持零回退 |
| P-ACC-4.3 integral | Apple ARM 已完成，x86 运行待验证 | sum/sqsum、8U/32F、边界零行零列和非对齐宽度正确 |
| P-ACC-4.4 pyramid | Apple ARM 已完成，x86 运行待验证 | pyrDown/pyrUp/buildPyramid 复用 5-tap kernel；奇偶尺寸和 border 正确 |
| P-ACC-5.1 color conversion | Apple ARM 已完成，x86 运行待验证 | 已支持 cvtColor/two-plane code 的 channel、色序、舍入和 tail 正确；现有 fast-path 无回退 |
| P-ACC-5.2 demosaicing | Apple ARM 已完成，x86 运行待验证 | 支持的 Bayer code、边界、奇偶尺寸和 C3/C4 输出正确 |
| P-ACC-5.3 convertMaps | Apple ARM 已完成，x86 运行待验证 | float/fixed map 双向转换、nninterpolation、饱和和交错布局正确 |
| P-ACC-5.4 remap/warp | Apple ARM 已完成，x86 运行待验证 | nearest/linear、border、inverse flags、ROI 和退化坐标正确；共享坐标 block |
| P-ACC-5.5 getRectSubPix | Apple ARM 已完成，x86 运行待验证 | bilinear row kernel 复用；中心/边界、U8/F32 和 patch type 正确 |
| P-ACC-6.1 nonlinear filters | Apple ARM 已完成，x86 运行待验证 | median 3x3/5x5、bilateral、stack blur 的 border、channel 和小图正确 |
| P-ACC-6.2 morphology | Apple ARM 已完成，x86 运行待验证 | erode/dilate 的 kernel/anchor/iterations/border 和 in-place 正确 |
| P-ACC-6.3 accumulate/blend | Apple ARM 已完成，x86 运行待验证 | 四个 accumulate 及 blendLinear 的 mask、channel、F32/F64 和 alpha 边界正确 |
| P-ACC-6.4 threshold | Apple ARM 已完成，x86 运行待验证 | 全部 threshold mode、mask、NaN/边界阈值正确；adaptive 复用 P-ACC-4 |
| P-ACC-6.5 histogram/LUT/color map | Apple ARM 已完成，x86 运行待验证 | equalize/LUT/applyColorMap 的查表、channel 和自定义 colormap 正确 |
| P-ACC-6.6 Hanning | Apple ARM 已完成，x86 运行待验证 | 1D/2D、F32/F64、对称性和小尺寸正确 |
| P-ACC-7.1 Core scalar-small | Apple ARM 已完成 | borderInterpolate、Mat metadata 与已接近 upstream 的 core case 完成复杂度和零回退审计 |
| P-ACC-7.2 Imgproc scalar-small | Apple ARM 已完成 | 核生成和 2x3/3x3 变换 API 完成分配、数值与零回退审计 |
| P-ACC-7.3 全矩阵收尾 | Apple ARM 已完成，x86 运行待验证 | Mode A/Mode B 日期报告、全量 CTest、UI-disabled、多 TU、header-only contract 与 SSE2/AVX2 编译 gate 已通过 |

### 0.15 P-ACC-3.1 收尾状态

masked `copyTo` 已移除逐像素 ND 坐标反解：所有 fallback 按 outer row 顺序扫描，U8/S8
C1/C3/C4 直接复用 UI `v_select` 和 deinterleave/interleave。`Mat::copyTo/clone`
原有连续 `memcpy` 保持不变；`setTo` 增加正零 `memset`、uniform scalar fill 和
倍增 pattern copy。Mode A 的 masked destination 同时改为确定性零初始化，避免不同进程
未初始化字节造成 checksum 假差异。

| Gate | 结果 |
| --- | --- |
| Correctness | Core `195 passed / 2 skipped`；新增 U8C1/S8C3/U8C4、非连续 source/mask ROI、vector tail 和 scalar/UI 一致性 |
| Mode A | `quick`、`2/100/3`；VGA U8C3 masked copy `11.06x`，32x32 U8C1 `6.67x`；全部 P-ACC-3.1 checksum 一致 |
| Memory baseline | VGA continuous clone/copyTo 为 `0.99x-1.02x`，没有算法性回退；setTo U8C1 `1.38x` |
| Mode B | quick `141/141 OK`；120x160 U8C3 masked copy CVH `0.001517 ms`、OpenCV `0.001433 ms` |
| 平台 | Apple ARM correctness/performance 已关闭；真实 x86 SSE/AVX 运行待统一阶段 gate |

### 0.16 P-ACC-3.2 收尾状态

1/2/4-byte transpose 已迁移 upstream 的 16x16、8x8、4x4 UI block；8-byte 和多通道
fallback 改为 typed tile，移除单像素 `memcpy`。horizontal/both flip 对
1/2/3/4/6/8/12-byte 像素使用 `v_reverse`，C3 使用 deinterleave/interleave；
vertical flip 使用整行 copy。2D `flipND` 复用 flip，泛型 ND 改为连续 inner block
反向复制。rotate 180 复用 flip，90 度复用 transpose 加 flip。

| Gate | 结果 |
| --- | --- |
| Correctness | Core `196 passed / 2 skipped`；覆盖 ROI、奇数宽度、vector tail、全部 rotate code、alias 和 scalar/UI 一致性 |
| Mode A | VGA U8C1 transpose `1.58x`、flip `20.71x`、rotate `6.46x`；VGA U8C3 flip `26.36x`、rotate `4.00x`；checksum 全一致 |
| Mode B | quick `141/141 OK`；VGA U8C1 transpose 与 upstream 基本持平，F32C1 慢 `4.4%`；120x160 U8C3 flip 慢 `6.6%`，flipND 快 `7.16x` |
| 已知差距 | 120x160 U8C3 rotate 仍慢 upstream `5.93x`；当前组合路径需要 transpose 中间 Mat，保留为后续 fused layout kernel 候选 |
| 平台 | Apple ARM correctness/performance 已关闭；真实 x86 SSE/AVX 运行待统一阶段 gate |

### 0.17 P-ACC-3.3 收尾状态

`mixChannels` 已移除 route 内部的逐像素 ND 坐标反解，泛型路径按 outer row 和连续 pixel
扫描。U8/S8 C3/C4 extract、insert 和完整 reorder 使用 UI
deinterleave/interleave；alias 仍先 snapshot，负 source channel、多 source/destination
和不完整映射继续走 scalar fallback。

| Gate | 结果 |
| --- | --- |
| Correctness | Core `197 passed / 2 skipped`；覆盖 C3/C4、ROI、tail、extract/insert/reorder、alias、负 channel 和 scalar/UI 一致性 |
| Mode A | VGA U8C3 extract `42.20x`、insert `26.12x`、reverse mix `53.90x`；C1 fallback `0.97x-0.99x`，checksum 全一致 |
| Mode B | quick `141/141 OK`；120x160 U8C3 extract、insert、mix 分别快 upstream `2.51x/1.69x/3.49x` |
| 平台 | Apple ARM correctness/performance 已关闭；真实 x86 SSE/AVX 运行待统一阶段 gate |

### 0.18 P-ACC-3.4 收尾状态

repeat 先复制 source row，再分别沿 row 和 row-block 倍增 `memcpy`；broadcast 以最长
trailing exact block 为单位复制，单一 source block 直接倍增，泛型 singleton 组合使用
prefix stride counter。二输入 hconcat/vconcat 增加无 `std::vector` 分配的常用 overload，
vconcat 对连续输入整块复制；所有 concat 在 destination alias 时先深拷贝对应 source。

| Gate | 结果 |
| --- | --- |
| Correctness | Core `197 passed / 2 skipped`；repeat、ND singleton broadcast、shape Mat、C3、ROI concat 和 output alias 通过 |
| Mode A | VGA U8C1 repeat/broadcast/vconcat 吞吐分别为 `87.26/120.96/131.45 GB/s`；本批为 memory algorithm，无独立 UI/scalar dispatch 倍数 |
| Mode B | quick `141/141 OK`；120x160 broadcast 快 upstream `1.24x`，vconcat 持平 `1.02x`；repeat/hconcat 分别剩余 `1.50x/2.11x` 差距 |
| 关键修正 | row-to-image broadcast 从 `0.062353 ms` 降至 `0.000216 ms`；二输入 hconcat 从 `0.002561 ms` 降至 `0.000727 ms` |
| 平台 | 纯 `memcpy`/stride 算法，无 ISA 专用实现；跨平台 correctness 由统一阶段 gate 关闭 |

### 0.19 P-ACC-3.5 与 P-ACC-3 收尾状态

现有项目 GEMM 不是 OpenCV `gemm(A,B,alpha,C,beta,flags)` 的完整兼容接口：它只接受
F32 activation，支持 F32/F16 weight、NT INT8 weight、bool transpose、batch broadcast
和项目自有 packed-B。P-ACC-3.5 没有借加速阶段扩大 API，而是为现有 F32 NN/packed-NN
增加沿 N 的四向量 UI FMA，为 F32 NT 增加沿 K 的 UI dot；F16/INT8 保持 scalar。
F64、alpha/beta/C 属于 API coverage 工作，不计入本批验收。

| Gate | 结果 |
| --- | --- |
| Correctness | Core `198 passed / 2 skipped`；NN/NT、vector tail、packed reuse、FP16 fallback、batch/transpose 既有合同通过 |
| Mode A | 128x128x128 NN end-to-end `2.77x`、pack-once `3.05x`、NT `1.86x`；NT raw checksum 因 reduction order 不同，数值 tolerance gate 通过 |
| Mode B | quick `141/141 OK`；NN end-to-end/pack-once 为 `0.084687/0.081818 ms`，仍慢 upstream `23.20x/22.41x` |
| 阶段全量 | Release CTest `17/17`；UI-disabled Core `185 passed / 15 skipped`；header-only 与多 TU smoke 通过 |
| 平台 | Apple ARM P-ACC-3 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.20 P-ACC-4.1 收尾状态

P-ACC-4.1 新增 `detail/filter_ui.hpp`，为 C1 8U/32F 的 `filter2D` 与 separable filter
提供固定 128-bit UI 内核。U8 路径按 16 像素分组做四组 float 累加和 pack 写回，边界与
非 C1 类型继续使用原有泛型路径。`GaussianBlur` 复用 separable 内核；`sqrBoxFilter`
的 32F 输出复用 convert/multiply/box 组合，64F 和 8U 保留宽累加实现，避免大核精度回退。

| 门禁 | 结果 |
| --- | --- |
| Correctness | Gaussian/filter2D/sepFilter2D/box 相关 14 项测试通过，包含 Gaussian bit-exact 与 sqrBox 大核宽累加 |
| Mode B 配置 | Apple ARM，Release，单线程，`warmup=2`、`iters=100`、`repeats=3` |
| Mode B 代表值 | VGA U8C1：Gaussian `0.341 ms`、filter2D `0.343 ms`、sepFilter2D `0.270 ms`；原实现约为 `0.350/0.420/0.363 ms` |
| sqrBox | VGA U8C1->F32 从约 `0.634 ms` 降至 `0.025 ms`；仍约为 upstream `2.56x` |
| 保留限制 | C3/C4 与 box rolling sum 继续走既有路径；没有用低收益 UI 替换它们 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.21 P-ACC-4.2 收尾状态

Scharr 与 Laplacian 的 C1 U8/F32->F32 常用路径已改为调用共享 UI filter 内核；带父 ROI
且未设置 `BORDER_ISOLATED` 的输入继续使用原卷积语义。`spatialGradient` 新增一次加载三行并
同时计算 dx/dy 的 U8C1->S16 UI 内核，Sobel 本身不改动。

| 门禁 | 结果 |
| --- | --- |
| Correctness | `ImgprocPhase1Kernels_TEST` 7/7 通过；Sobel border 与 S16 regression 通过 |
| Mode B quick | Scharr `0.591 -> 0.091 ms`，Laplacian `0.576 -> 0.085 ms`，spatialGradient `0.124 -> 0.059 ms` |
| Sobel 零回退 | 同一 quick case `0.109 -> 0.108 ms`，仍领先 upstream |
| Dispatch | 三个新增代表 case 均报告 `opencv_ui` |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.22 P-ACC-4.3 收尾状态

`integral` 不再先清零整张输出，也不再在每个像素内判断输出深度。常用 U8C1->S32 路径
按四像素生成行前缀，并用 UI 向量加上前一积分行；多通道和 64F 使用移出分支后的 scalar
fallback。当前公开 API 只提供 `sum` 输出，`sqsum` 不在本阶段偷偷扩展 API。

| 门禁 | 结果 |
| --- | --- |
| Correctness | 零首行/首列、多通道、64F 和 ROI contract 通过 |
| Mode B quick | U8C1->S32 `0.0634 -> 0.00418 ms`，约 `15.2x` 内部提速 |
| Dispatch | 代表 case 报告 `opencv_ui` |
| API 边界 | `sqsum` 仍是未支持 API，留在 API coverage 计划中 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.23 P-ACC-4.4 与 P-ACC-4 收尾状态

`pyrDown`/`pyrUp` 从每输出像素直接执行 25-tap double 卷积，改为共享的水平/垂直 5-tap
两阶段实现。边界索引按调用预计算，U8 使用整数工作缓存和 upstream 一致的定点舍入，
F32 保留 double 工作精度；`buildPyramid` 直接复用优化后的 `pyrDown`。

| 门禁 | 结果 |
| --- | --- |
| Correctness | pyramid/color 相关 contract 6/6 通过，覆盖尺寸、常量图、层级复用和非法参数 |
| Mode B quick | pyrDown `0.487 -> 0.0309 ms`，pyrUp `1.285 -> 0.0781 ms`，buildPyramid `0.192 -> 0.0166 ms` |
| 复杂度 | 每输出像素从 25 次二维采样降为两个 5-tap pass；边界插值移出像素/通道内循环 |
| P-ACC-4 结论 | Apple ARM 四个子批次完成；滤波、导数、integral、pyramid 均保留 scalar/泛型 fallback |
| 平台 | 真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.24 P-ACC-5.1 收尾状态

既有 `cvtColor` UI 路径保持不变，VGA BGR2GRAY 与 upstream 基本持平。独立双平面
NV12/NV21 路径改为按 2x2 block 处理，四个 Y 像素共享一次 U/V 偏移和色度乘法；
RGB/BGR 排列继续在最终写回处选择。

| 门禁 | 结果 |
| --- | --- |
| Correctness | `ImgprocCvtColor_TEST` 58/58 与 two-plane 4-code contract 通过 |
| BGR2GRAY 零回退 | Mode B VGA `0.0325 ms`，与 upstream `0.0326 ms` 持平，dispatch=`opencv_ui` |
| Two-plane | quick NV12->BGR `0.0273 -> 0.0202 ms`，约 `1.35x` 内部提速 |
| 覆盖 | NV12/NV21、RGB/BGR、非连续 Y/UV step 均保持原语义 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.25 P-ACC-5.2 收尾状态

基础 Bayer 双线性去马赛克不再为 B/G/R 三个输出通道分别遍历 3x3 邻域，也不再在内循环中
调用 `Mat::at`。固定位置直接计算 cross、diagonal、horizontal 和 vertical 平均，边界仍复制
相邻完整输出行/列，四种 Bayer pattern 共用同一实现。

| 门禁 | 结果 |
| --- | --- |
| Correctness | 四种 Bayer pattern、边界复制和非法 code/dstCn contract 通过 |
| Mode B quick | U8C1 BayerBG->BGR `1.680 -> 0.0288 ms`，约 `58.3x` 内部提速 |
| 剩余差距 | 约为 upstream `11.7x`；后续收益点是多像素 UI 与 C3 pack，不再是算法复杂度 |
| API 边界 | 当前公开 contract 仅 C3 基础双线性；EA/VNG/C4 没有在加速阶段扩展 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.26 P-ACC-5.3 收尾状态

`convertMaps` 只在输出真实 alias 输入时 snapshot，不再无条件 clone 两张 map。F32 pair 到
S16C2/U16 的主路径使用四坐标 UI round，行指针直接读写；非有限值、tail、nninterpolation
和其他 map 方向保留精确 fallback。

| 门禁 | 结果 |
| --- | --- |
| Correctness | float pair/interleaved/fixed 的等价性、nearest、负坐标与恢复误差 contract 通过 |
| Mode B quick | F32 pair->fixed `0.762 -> 0.0246 ms`，约 `31.0x` 内部提速 |
| Dispatch | 代表 case 报告 `opencv_ui` |
| Alias | map 与任一输出共享 storage 时才 clone；普通调用不再复制输入 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.27 P-ACC-5.4 收尾状态

新增共享 U8 bilinear 采样器：每个坐标只解析一次 x0/x1/y0/y1 和 border，C1/C3/C4 通道
复用四个像素地址。fixed map 直接读取 S16C2/U16 行，float map 直接读取 float 行并量化；
warpAffine 使用行内递增坐标，warpPerspective 复用 fixed-fraction 写回。

| 门禁 | 结果 |
| --- | --- |
| Correctness | remap/warpPerspective 7 项与 warpAffine 8 项 contract 全通过 |
| Mode B VGA | float remap `14.60 -> 2.44 ms`，fixed remap `13.78 -> 2.41 ms` |
| Mode B VGA | warpAffine `3.24 -> 1.75 ms`，warpPerspective `10.18 -> 2.71 ms` |
| 语义 | nearest/linear、constant/replicate/reflect、inverse map、ROI 和 alias 保持 |
| 剩余差距 | 仍需真正的坐标 block/gather 与平台运行验证；当前先关闭重复分派问题 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.28 P-ACC-5.5 与 P-ACC-5 收尾状态

`getRectSubPix` 只计算一次 patch 原点整数部分和固定小数权重。整数原点、同 depth 且 patch
完全位于输入内时直接逐行块复制；其他 U8 输出复用 P-ACC-5.4 的共享 bilinear 采样器，
U8->F32/F32 保留通用类型转换路径。

| 门禁 | 结果 |
| --- | --- |
| Correctness | 中心、边界、U8->U8/U8->F32/F32 和非法参数 contract 通过 |
| Mode B VGA | full-frame U8C3 `6.58 -> 0.0109 ms`，约为 upstream `13.2x` |
| P-ACC-5 结论 | color/two-plane、Bayer、convertMaps、remap/warp、sub-pixel 五个子批次完成 |
| 平台 | Apple ARM P-ACC-5 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.29 P-ACC-6.1 收尾状态

U8C1 median 改为滑动双层直方图，查找中值最多扫描 16 个 coarse bin 和对应 16 个 fine
bin；U8 bilateral 将色差 `exp` 预计算为 LUT；U8 stack blur 使用整数两阶段工作缓存并预计算
x/y 边界索引。F32、多通道 median 和通用边界仍使用原 fallback。

| 门禁 | 结果 |
| --- | --- |
| Correctness | median ROI/in-place/small image、bilateral constant/alias、stack triangular/alias 3 项 contract 通过 |
| Mode B quick | median `2.95 -> 0.560 ms`，bilateral `1.47 -> 1.04 ms`，stack `0.0948 -> 0.0750 ms` |
| 剩余差距 | median 仍需 sorting network/UI；bilateral 仍受 gather 与浮点归一化限制 |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.30 P-ACC-6.2 收尾状态

常用 U8C1、3x3、单次迭代的 erode/dilate 使用两阶段 UI min/max：水平阶段按 16 像素读取
三列并写入临时行，垂直阶段再合并三行。其他 kernel、anchor、iterations、channel 和 depth
继续使用原泛型实现，避免扩大语义风险。

| 门禁 | 结果 |
| --- | --- |
| Correctness | morphology/gradient 相关 contract `7/7` 通过，覆盖 kernel、anchor、iterations、border、ROI 与 in-place |
| Mode B quick | erode `0.303 -> 0.124 ms`，dilate `0.310 -> 0.124 ms`，均约 `2.5x` 内部提速 |
| Dispatch | 满足 U8C1/3x3/单次迭代条件时报告 `opencv_ui`，其余路径如实报告 scalar |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.31 P-ACC-6.3 收尾状态

四个 accumulate 的无 mask U8/F32 输入使用 UI load/convert 和 F32 add/mul/FMA；mask、
F64 目标及其他类型继续保留原合同路径。`blendLinear` 将 depth 分派移出像素循环，并将
每像素 reciprocal 复用于全部通道，暂不加入低收益的向量除法路径。

| 门禁 | 结果 |
| --- | --- |
| Correctness | accumulate family、weighted alpha 边界和 blendLinear 非归一化/零权重 3 项 contract 通过 |
| Mode B 配置 | Apple ARM，Release，单线程，`warmup=2`、`iters=100`、`repeats=3`，`141/141 OK` |
| accumulate | `0.0150 -> 0.00452 ms`；product `0.0231 -> 0.00559 ms`；square `0.0169 -> 0.00378 ms` |
| weighted/blend | weighted `0.0187 -> 0.00323 ms`；blendLinear `0.0334 -> 0.0291 ms` |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.32 P-ACC-6.4 收尾状态

固定 U8/F32 threshold 的五种模式直接使用 UI compare/select，连续 Mat 合并为一行，
ROI 和 vector tail 由同一 row kernel 处理。adaptiveThreshold 继续复用 P-ACC-4 的
box/Gaussian，并用 widen 后的有符号差值比较生成输出；masked threshold 因已接近 upstream
而保留现状。

| 门禁 | 结果 |
| --- | --- |
| Correctness | threshold 全模式、U8/F32、C1/C3/C4、ROI、dry-run、auto mode 与 adaptive/masked 共 `11/11` 通过 |
| Mode B quick | VGA U8 binary threshold `0.218 -> 0.00525 ms`，当前与 upstream `0.00537 ms` 基本持平 |
| Adaptive | mean11 U8C1 `0.0839 -> 0.0736 ms`；剩余差距主要来自 11x11 local mean |
| Mask 零回退 | `0.0115 ms`，与实施前持平，约为 upstream 的 `1.23x` |
| 平台 | Apple ARM 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.33 P-ACC-6.5 收尾状态

byte LUT 按 upstream 的平台结论分流：NEON 和 AVX-512 VBMI 使用 UI `v_lut`，SSE/AVX2
继续使用展开 scalar，避免高延迟 gather 回退。连续单通道 LUT 不再每次复制 256 项；
equalizeHist 的直方图和映射循环展开，并复用同一平台查表策略。当前已经领先 upstream 的
applyColorMap 不改实现，只执行正确性与性能零回退 gate。

| 门禁 | 结果 |
| --- | --- |
| Correctness | LUT 的 C1/C3/C4、F32 table、ROI、非连续 table、in-place 与 equalize/color map 共 `9/9` 通过 |
| Equalize Mode B | U8C1 `0.0127 -> 0.00528 ms`，与 upstream `0.00548 ms` 基本持平 |
| LUT Mode B | VGA invert U8 `0.0795 -> 0.0677 ms`；仍慢 upstream `2.37x`，不为 AVX2 启用负收益 gather |
| Color map 零回退 | `0.0413 ms`，继续领先 upstream 约 `1.69x` |
| 平台 | Apple ARM 已关闭；SSE/AVX2 明确保留 scalar lookup，真实 x86 运行仍为外部 gate |

### 0.34 P-ACC-6.6 与 P-ACC-6 收尾状态

Hanning window 先按 x/y 分别计算对称的一维平方根权重，再使用 UI multiply 生成外积。
三角函数和平方根调用从 O(width*height) 降为 O(width+height)，F32/F64 共用相同数学
分解，小宽度与无 SIMD 平台保留 scalar tail。

| 门禁 | 结果 |
| --- | --- |
| Correctness | F32/F64 的边界零值、中心值、对称性既有 contract 通过 |
| Mode B quick | 64x64 F32 `0.0383 -> 0.000616 ms`，约 `62.1x` 内部提速，当前领先 upstream `1.78x` |
| 复杂度 | 每个输出像素只剩一次乘法；一维权重利用镜像对称，只计算约一半 `cos/sqrt` |
| P-ACC-6 结论 | nonlinear、morphology、accumulate/blend、threshold、LUT/histogram、Hanning 六个子批次完成 |
| 平台 | Apple ARM P-ACC-6 已关闭；真实 x86 SSE/AVX correctness 与运行性能仍为外部 gate |

### 0.35 P-ACC-7.1 收尾状态

REFLECT/REFLECT_101 的远距离 border 坐标改为按周期一次取模，不再循环折返；`swap(Mat,Mat)`
直接交换 header 字段和 shape/stride storage，不再执行三次带引用计数的 Mat 赋值。reshape
已确认是 O(1) view，剩余微基准差距来自 Mat 的动态 shape header，不在低收益阶段改动布局。

| 门禁 | 结果 |
| --- | --- |
| Correctness | border 全模式、远距离正负坐标、isolated flag 与 header swap/refcount 2 项定向 contract 通过 |
| Border Mode B | reflect101 batch4096 `0.0135 -> 0.00561 ms`，当前领先 upstream `2.28x` |
| Swap Mode B | `0.000024 -> 0.000005 ms`，当前与 upstream 持平 |
| 保留现状 | `MAT_RESHAPE` 约 `0.000053 ms`，保持 O(1)；不为微秒以下 header 操作改变 Mat ABI |

### 0.36 P-ACC-7.2 收尾状态

已经领先的 affine/perspective solve、Gaussian 与 rotation 保持算法不变。结构元素改为直接按行
清零/填充；derivative 和 Gabor 把 depth 分支及 `Mat::at` 地址计算移出元素循环；
invertAffineTransform 使用直接行指针读取和写回，alias snapshot 与奇异矩阵语义不变。

| 门禁 | 结果 |
| --- | --- |
| Correctness | kernel generators 3 项与 geometry matrix 5 项，共 `8/8` 定向 contract 通过 |
| Kernel Mode B | structuring `0.000380 -> 0.000100 ms`；derivative `0.000196 -> 0.000130 ms` |
| Gabor Mode B | 15x15 F32 `0.00286 -> 0.000962 ms`，当前与 upstream `0.000942 ms` 基本持平 |
| Geometry Mode B | invertAffine `0.000120 -> 0.000039 ms`，当前领先 upstream `1.49x`；其余领先项无回退 |
| 设计边界 | 没有为几十次标量运算引入 SIMD；只消除重复分派、地址计算和初始化开销 |

### 0.37 P-ACC-7.3 与 P-ACC-3 至 P-ACC-7 收尾状态

2026-07-24 在 Apple ARM Release、单线程环境完成最终矩阵。Mode A 和 Mode B 统一使用
`warmup=2`、`iters=100`、`repeats=3`；日期性能报告已由当前完整 CSV 重新生成。

| 门禁 | 结果 |
| --- | --- |
| 默认 UI correctness | CTest `17/17`；Core `198 passed / 2 skipped`；Imgproc `173/173`；upstream contract smoke `19/19` |
| Scalar fallback | `CVH_ENABLE_OPENCV_INTRIN=0` 下 Core `185 passed / 15 skipped`、Imgproc `173/173`，header/ODR smoke `4/4` |
| Header-only contract | public header 检查及多 TU contract `5/5` |
| Mode A | Auto 与 Scalar 各 `397/397 OK`；`384` 行 checksum 位级一致；其余 `13` 行来自 `meanStdDev/exp/log/pow/GEMM` 的浮点重排，均由数值容差 contract 覆盖 |
| Mode B full | `340` 个 case：`339 OK + 1` 个既有 `UNSUPPORTED`；无新增不支持项；OpenCV/CVH 几何平均 `0.5563`，CVH 领先 `58` 个 case |
| 既有 UNSUPPORTED | `CVTCOLOR BGR2NV12_u8`；upstream OpenCV 没有单调用 BGR-to-NV12 encoder，不是本轮回退 |
| x86 编译 gate | Apple Clang x86_64 的 SSE2/AVX2 下，聚合公共 header 与 Mode A benchmark 均完成模板实例化 |
| x86 运行 gate | 尚未关闭；交叉编译不能替代真实 x86 SSE/AVX correctness 与性能运行 |

最终报告：
`benchmark/opencv_compare/results/2026-07-24-opencv-upstream-performance.md`。
P-ACC-3 至 P-ACC-7 的 Apple ARM 实现、correctness、fallback、header-only 与 benchmark
收尾已完成；后续只保留真实 x86 机器运行验证，不继续扩大本批次实现范围。

### 0.38 P-ACC-8 推进状态

P-ACC-8 不新增 API，以 2026-07-24 完整 Mode B 报告为冻结基线。倍率只用于发现候选，
最终顺序同时考虑绝对耗时、复用面、实现复杂度和 upstream 是否使用不在项目边界内的外部库。

| Step | 状态 | 当前验收点 |
| --- | --- | --- |
| P-ACC-8.0 benchmark 归因 | Apple ARM 已完成 | 输出分配、预计算和像素/矩阵内核已拆分；默认 upstream 与内建 CPU-only GEMM 对照已建立 |
| P-ACC-8.1 pyramid family | Apple ARM 已完成 | 5-row ring workspace、U8 C1/C3/C4 水平 UI、U8/F32 垂直 UI 与跨层 workspace 已进入公共路径 |
| P-ACC-8.2 nonlinear filters | Apple ARM 已完成 | median sorting network、stack sliding triangular sum 与 bilateral 预计算 LUT/坐标均通过 1.5x gate |
| P-ACC-8.3 geometry sampling | Apple ARM 已完成 | `convertMaps` UI 打包与 `remap/warpAffine/warpPerspective` 共享定点坐标 block、双线性采样器和 interior/border 分流 |
| P-ACC-8.4 filter/morphology | Apple ARM 已完成 | Scharr/Laplacian 复用 S16 UI filter kernel；sqrBoxFilter 使用 U8 square + S64 横纵滑动和；既有 filter/morphology 无 >5% 回退 |
| P-ACC-8.5 Core GEMM/reduction | Apple ARM 已完成 | GEMM 维持已验证内建 CPU 优势；F32 C1 meanStdDev 双遍 UI、单输入 NORM_INF F32 max、axis-1 extrema 向量累计均通过 gate |
| P-ACC-8.6 全矩阵收尾 | Apple ARM 已完成，x86 运行待验证 | 默认/UI-disabled、Mode A/Mode B、header/ODR 与 SSE2/AVX2 编译已关闭；真实 x86 correctness/performance 保留外部 gate |

### 0.39 P-ACC-8.0 收尾状态

归因 benchmark 统一显式设置 `cvh` 与 upstream 线程数，修复了此前 CSV 写
`threads=1`、但只限制 upstream OpenCV 线程的测量漏洞。Mode B 新增 `--ops GEMM`
focused run；metadata 记录 `ops`、OpenCV variant 及
`WITH_LAPACK/WITH_IPP/WITH_KLEIDICV/WITH_CAROTENE` 实际值。

Pyramid 在不改变算法的前提下抽出 index precompute 和 preallocated-workspace kernel helper。
Apple M5 Release、单线程、`2/100/3` 结果：

| 组件 | `pyrDown` ms | `pyrUp` ms | 结论 |
| --- | ---: | ---: | --- |
| public reuse | 0.211280 | 1.811377 | 端到端冻结基线 |
| public recreate | 0.209212 | 1.764222 | 与 reuse 无稳定差异，输出分配不是主要瓶颈 |
| index precompute | 0.002623 | 0.011317 | 仅约占端到端 `1.2%/0.6%` |
| precomputed kernel | 0.203422 | 1.798014 | 几乎覆盖全部成本，P-ACC-8.1 应直接优化 5-tap 内核 |

GEMM Mode A 同样使用单线程 `2/100/3`。UI end-to-end 相对 scalar 在 square/skinny/wide
三种形状分别提升约 `5.97x/12.17x/8.68x`；kernel-only 提升约
`10.90x/12.20x/8.10x`。B packing 只消耗约 `0.0006-0.0027 ms`，不是主要瓶颈。

Mode B focused GEMM 同时运行默认 upstream 和独立内建 CPU 构建。CPU 构建禁用 LAPACK、
IPP、KleidiCV、Carotene 与 OpenCL：

| Shape | CVH ms | 默认 OpenCV ms | 内建 CPU OpenCV ms | 归因 |
| --- | ---: | ---: | ---: | --- |
| 128x128x128 | 0.090635 | 0.003374 | 0.161422 | 默认路径约 `47.8x` 快于内建 CPU；CVH 比内建 CPU 快 `1.78x` |
| 32x512x64 | 0.055685 | 0.109595 | 0.108082 | LAPACK 开关几乎无影响；CVH 快约 `1.94x` |
| 256x32x256 | 0.065286 | 0.004887 | 0.222794 | 默认路径约 `45.6x` 快于内建 CPU；CVH 比内建 CPU 快 `3.41x` |

因此 2026-07-24 报告中的 GEMM 大差距主要来自 Apple Accelerate/LAPACK，不是 OpenCV UI
内建 kernel。P-ACC-8.5 不再以默认 OpenCV GEMM 倍数作为追赶目标；只在高优先级 Imgproc
批次完成后，评估 square case 约 `0.07 ms` 的公共 shape/output 开销。

验收：默认 UI CTest `17/17`，UI-disabled 定向 CTest `6/6`；pyramid public/direct
checksum 一致；focused GEMM 默认/CPU-only 各 `6/6 OK`，无 `UNSUPPORTED`。
公共聚合头、Core benchmark 和 Imgproc benchmark 均通过 SSE2/AVX2 交叉编译门禁。
真实 x86 运行仍属于 P-ACC-8.6。

### 0.40 P-ACC-8.1 收尾状态

Pyramid 公共路径不再为整张源图保存水平卷积结果，workspace 从
`src.rows * dst.cols * channels` 收敛为固定 `5 * dst.cols * channels`。`pyrDown`
迁移 upstream 同类 U8 C1/C3/C4 水平 5-tap UI，U8/F32 垂直 5-tap 直接使用 vendored
OpenCV UI；`pyrUp` 使用 5-row cache 和共享垂直 UI。`buildPyramid` 在全部层之间复用同一
最大 workspace。

Apple M5 Release、单线程、`2/100/3` 的 640x480 Mode A：

| Case | P-ACC-8.0 ms | P-ACC-8.1 ms | 提升 |
| --- | ---: | ---: | ---: |
| `pyrDown U8C1` public reuse | 0.211280 | 0.055619 | 3.80x |
| `pyrUp U8C1` public reuse | 1.811377 | 0.911617 | 1.99x |
| `pyrDown U8C3` public reuse | 0.495427 | 0.325964 | 1.52x |
| `buildPyramid U8C1 levels=3` | 0.261025 | 0.079461 | 3.28x |

正确性覆盖 U8/F32、C1/C3/C4、奇数 ROI、`REPLICATE/REFLECT/REFLECT_101/WRAP`、
scalar/UI 一致和 in-place alias。P-ACC-8.6 仍需重跑完整 Mode B，并在真实 x86 机器执行。

### 0.41 P-ACC-8.2 收尾状态

- `medianBlur` 的 U8 3x3/5x5 公共路径迁移 upstream sorting network，UI 同时覆盖
  C1/C3/C4 的连续通道元素，边界与 tail 使用标量参考。
- `stackBlur` 用两阶段滑动三角和替代每像素重复遍历 kernel；前缀和原型因 cache 行为回退
  已删除，没有进入公共路径。
- `bilateralFilter` 将 U8 color LUT、全部 border x/y 坐标移出像素内循环，并使用 float
  权重累计，保持 F32 路径和不支持 in-place 的合同。

Apple M5 Release、单线程、`2/100/3` 的 640x480 U8C1：

| Case | 冻结基线 ms | 当前 ms | 提升 |
| --- | ---: | ---: | ---: |
| `medianBlur 3x3` | 7.448417 | 0.314719 | 23.67x |
| `medianBlur 5x5` | 8.299958 | 1.386294 | 5.99x |
| `stackBlur 5x5` | 1.245217 | 0.527053 | 2.36x |
| `bilateralFilter d5` | 17.891886 | 10.924313 | 1.64x |

正确性新增 median scalar/UI 的 C1/C3/C4、ROI、in-place 对照，以及 stack blur 与独立朴素
三角核的 C1/C3/C4、非对称 kernel、ROI 对照。完整跨平台 gate 留在 P-ACC-8.6。

### 0.42 P-ACC-8.3 收尾状态

`convertMaps` 的 float-to-fixed 公共路径不再把 UI 结果落回临时数组后逐元素做除法，而是直接
完成有限值检查、round、fraction mask、算术 shift、坐标 interleave 和 pack。几何采样公共
底座新增固定点坐标 block 与共享 U8 双线性采样器；`remap` 的 float/fixed map、
`warpAffine` 和 `warpPerspective` 均复用该实现。采样器将常见 interior pixel 与 border
pixel 分流，并按 C1/C3/C4 特化，避免 interior 每像素重复四次 border resolve。

Apple M5 Release、单线程、`2/100/3` 的 640x480 U8C3：

| Case | 冻结/改造前 ms | 当前 ms | 提升 |
| --- | ---: | ---: | ---: |
| `convertMaps F32 pair -> fixed` | 0.394950 | 0.101220 | 3.90x |
| `remap float linear` | 2.584781 | 2.069287 | 1.25x |
| `remap fixed linear` | 2.518281 | 1.813072 | 1.39x |
| `warpPerspective linear` | 2.742277 | 2.186548 | 1.25x |
| `getRectSubPix full frame` | 0.011490 | 0.010998 | 无回退 |

Mode A 新增 `warpAffine` 公共路径行，当前为 `2.062172 ms`；由于此前 Mode A 未记录相同
输入，暂不宣称倍率。正确性覆盖 float/fixed map 等价、U8 C1/C3/C4、非连续 ROI、
`CONSTANT/REPLICATE/REFLECT/REFLECT_101`、nearest/linear、inverse map 和 alias。
几何采样属于 gather 与坐标计算受限路径，本轮未达到 `1.5x` 的三个 public case 保留现有
共享实现并如实记录，不以重复维护三份 bilinear loop 换取单 case 特化。真实 x86 运行和
完整 Mode B 留在 P-ACC-8.6。

### 0.43 P-ACC-8.4 收尾状态

`filter_ui::filter2d_c1` 的公共内核新增 S16 输出，使用 UI round + pack 写回，并把纵向 border
索引移出向量内循环。`Scharr` 与 `Laplacian` 因此不再进入逐像素 double 卷积，和
`filter2D` 共用同一 C1 kernel。`sqrBoxFilter` 的 U8 -> U8/F64 路径改为先做平方宽整数
横向滑动和，再做纵向滑动和；S64 accumulator 保留大 kernel 的精确合同。

Apple M5 Release、单线程、`2/100/3` 的 640x480：

| Case | P-ACC-8.3 ms | 当前 ms | 提升 |
| --- | ---: | ---: | ---: |
| `Scharr dx=1 U8C1 -> S16` | 9.461904 | 1.558625 | 6.07x |
| `Laplacian k3 U8C1 -> S16` | 9.336646 | 1.653769 | 5.65x |
| `sqrBoxFilter 3x3 U8C1 -> F64` | 9.119206 | 0.275333 | 33.12x |

回归门禁中，`boxFilter/GaussianBlur/filter2D/sepFilter2D/Sobel/erode/dilate/morphologyEx`
相对 P-ACC-8.3 均未出现稳定 `>5%` 回退。正确性新增 Scharr/Laplacian 的 scalar/UI、
短行与 vector tail、非连续 ROI、四种 border 对照；既有 filter、morphology 和
sqrBoxFilter wide-accumulation 合同继续通过。S16 fast-path 当前只覆盖 C1；C3/C4 自动
回退现有公共实现，不标记为缺失或跳过。

### 0.44 P-ACC-8.5 收尾状态

GEMM 按 P-ACC-8.0 归因不增加新的 packing 或 micro-kernel：CVH 已快于 upstream 内建
CPU 路径，默认 upstream 的大幅领先主要来自 Apple Accelerate/LAPACK。Core 收敛集中在：

- F32 C1 无 mask `meanStdDev` 使用稳定双遍算法；第一遍保留现有宽精度 sum，第二遍用
  F64 UI 对中心差平方并行累计，替代逐元素 `long double`。
- F32 单输入 `NORM_INF` 直接在 F32 UI 中做 abs/max；双输入 diff 继续使用 F64，避免
  `FLT_MAX - (-FLT_MAX)` 在 F32 溢出。
- axis-1 C1 `REDUCE_MAX/MIN` 在整个 row 上累计 vector extrema，最后只做一次横向归约，
  不再对每个 vector block 执行 `v_reduce_max/min`。

Apple M5 Release、单线程、`2/100/3` 的 640x480 F32C1：

| Case | P-ACC-8.4 ms | 当前 ms | 提升 |
| --- | ---: | ---: | ---: |
| `meanStdDev` | 0.324042 | 0.116872 | 2.77x |
| 单输入 `NORM_INF` | 0.086069 | 0.036604 | 2.35x |
| axis-1 `REDUCE_MAX -> F64` | 0.132684 | 0.054330 | 2.44x |
| axis-1 `REDUCE_MIN -> F64` | 0.131523 | 0.054097 | 2.43x |

square/skinny/wide 的 GEMM NN/NT 相邻复测均无 `>5%` 回退，checksum 不变。Reduction
focused contract 覆盖 scalar/UI、mask、C1-C4、ROI、tail、NaN/Inf、宽差值、alias 和
两条 axis，`8/8` 通过。完整 Core 与 Imgproc 回归、UI-disabled 和跨架构编译归入
P-ACC-8.6。

### 0.45 P-ACC-8.6 收尾状态

2026-07-25 在 Apple M5 Release、单线程环境完成 P-ACC-8 全矩阵收尾。Mode A 和
Mode B 统一使用 `warmup=2`、`iters=100`、`repeats=3`；full profile 仅作为阶段性
gate，日常优化继续使用 quick/focused profile。

| 门禁 | 结果 |
| --- | --- |
| 默认 UI correctness | CTest `16/16`；Core 与 Imgproc contract、header-only smoke 和多 TU ODR smoke 全部通过 |
| UI-disabled | Core `185 passed / 15 UI-only skipped`；Imgproc `178/178`；Core/Imgproc ODR smoke 通过 |
| Mode A Core full | `1000/1000 OK`，覆盖 `52` 个操作、`12` 种 shape；`799` 行命中 UI，`106` 行 scalar fallback |
| Mode A Imgproc full | `307 OK + 1` 个明确合同外 `INTER_CUBIC`；覆盖 `36` 个操作及 VGA/HD/FHD/奇数/4K shape |
| Mode B full | `344` 个 case：`343 OK + 1` 个既有 `CVTCOLOR BGR2NV12_u8` unsupported；无新增不支持项 |
| Mode B 汇总 | `OpenCV/CVH` 几何平均 `0.4550`；CVH 领先 `44` 个 case；详细差距与归因见 2026-07-25 日期报告 |
| benchmark contract | 修复 `ops=all` 在严格 shell 模式下展开空数组的问题；two-plane full case 将奇数 Y 尺寸向下调整为合法偶数尺寸并记录实际 shape |
| x86 编译 gate | 最新聚合公共 header、Core Mode A 和 Imgproc Mode A 均在 Apple Clang x86_64 SSE2/AVX2 下完成模板实例化 |
| x86 运行 gate | 尚未关闭；真实 x86 SSE/AVX correctness、Mode A 和 Mode B 运行仍是外部验证项 |

日期报告更新为
`benchmark/opencv_compare/results/2026-07-25-opencv-upstream-performance.md`。报告新增
P-ACC-8 未收敛差距表：GEMM 默认 upstream 优势主要来自 Accelerate/LAPACK，不作为纯 UI
追赶目标；filter/derivative、nonlinear、pyramid、geometry 和 reduction 的剩余差距按
算法结构与 variant 继续拆分，不能通过引入链接依赖、复制公共内核或降低精度关闭。

## 1. 目标

本计划覆盖 2026-07-24 Mode B 报告中的全部 `105` 个操作族：

- Core：`55` 个。
- Imgproc：`50` 个。

目标不是机械复制 OpenCV 的完整 dispatch 系统，而是从 upstream CPU 路径中提取适合纯
header-only 项目的加速知识：

1. 优先迁移使用 OpenCV Universal Intrinsics（以下简称 UI）的可移植 SIMD kernel。
2. 保持同一份 `v_*` kernel 在 ARM NEON 与 x86 SSE/AVX 上编译。
3. 复用 upstream 的分块、行缓存、定点权重、尾部处理和连续内存快路径。
4. 每个算子都建立 upstream 对应关系；upstream 没有 SIMD 的算子必须如实标记，不能为了
   “SIMD 覆盖率”制造无收益的向量代码。

## 2. 参考边界

### 2.1 可以参考

- `modules/core/src/*.simd.hpp`
- `modules/imgproc/src/*.simd.hpp`
- `.cpp` 中直接使用 `v_*`、`vx_*`、`VTraits<>` 的 CPU 实现
- upstream CPU baseline 中与 SIMD 配套的分块、行缓存、预计算和尾部循环
- `.avx2.cpp`、`.sse4_1.cpp` 中的算法组织方式，但只能转写为 UI 或作为性能分析依据

### 2.2 不进入本项目

- IPP、IPP IW
- OpenCL、OpenVX、CUDA、Vulkan
- Apple Accelerate、BLAS、LAPACK 等外部数学库
- 平台 HAL 插件、运行时动态加载和需要单独编译的 CPU dispatch translation unit
- 直接写 NEON/SSE/AVX intrinsic 作为首选实现
- xsimd 性能路径
- RVV；继续保留为后续 scalable-vector 专题

所有迁移代码必须是 ODR-safe header 实现。优先使用当前 vendored OpenCV UI，不能重新引入
`opencv_intrin_adapter`。

### 2.3 加速类型

| 标记 | 含义 | 本项目处理方式 |
| --- | --- | --- |
| `UI-direct` | upstream 有明确的 UI 向量循环 | 优先迁移 |
| `UI-composed` | API 自身主要组合其他已加速 kernel | 先完成依赖 kernel，再压缩调度和临时 Mat 开销 |
| `CPU-blocked` | upstream 核心收益来自分块、缓存、预计算，SIMD 是其中一部分 | 同时迁移算法骨架和 UI 内核 |
| `Memory` | upstream 主要依赖 `memcpy`、`memset`、连续行合并或 O(1) 元数据操作 | 不强制 SIMD，优先修复算法与分配 |
| `Scalar-small` | 小矩阵、核生成或单点函数，upstream 没有有意义的 SIMD fast-path | 保持正确性，只有实测证明必要时才优化 |

## 3. 性能判读约束

报告中的倍数是当前 case 集的算子级几何平均，不等同于“缺少 SIMD 的纯收益”。例如：

- `HAS_NON_ZERO` 的 `32258.06x` 首先来自 early-exit 语义；只增加向量归约仍可能扫描完整图像。
- `REPEAT`、`BROADCAST`、`COPY_TO` 的巨大差距主要指向重复分配、逐元素访问或缺少块复制。
- `BUILD_PYRAMID` 的主要收益来自复用 `pyrDown` 和避免多次建立中间状态。
- 当前 upstream `GEMM` 可能经 `modules/core/src/hal_internal.cpp` 进入 CBLAS/LAPACK。该路径不在
  本项目目标内，不能把报告中的 `78.01x` 全部当作 UI 可追回空间。

因此每个实现批次必须同时跑：

1. Mode A：scalar/baseline 与新 UI fast-path 的内部对比。
2. Mode B：`cvh_headers_fast` 与当前 upstream OpenCV 的外部对比。
3. 对 GEMM 增加禁用外部 BLAS/LAPACK 的 upstream CPU-only 对照，分离库加速与内建 SIMD。

## 4. 分批加速顺序

### P-ACC-0：共享门禁与内核迁移规则

先建立后续所有批次共用的验收方式，不新增大面积算子实现。

| 工作 | 目的 | 验收 |
| --- | --- | --- |
| 为每个 fast-path 记录 `scalar` / `opencv_ui` dispatch | 确认 benchmark 实际进入新内核 | Mode A 原始数据可区分两条路径 |
| 建立连续、非连续 ROI、非向量宽度和小图矩阵 | 覆盖 stride、tail 与短循环 | ARM/x86 correctness 全通过 |
| 规定 UI kernel 的 header/namespace/inline 模板 | 防止 ODR 和宏污染 | 两个 TU 同时包含并链接通过 |
| 固定单线程 benchmark 与预分配输出 | 排除线程和分配噪声 | 重复运行的中位数稳定 |

### P-ACC-1：Core 逐元素、转换与数学函数

先处理复用面最广、实现最规则的一维行 kernel。目标包括
`ADD/SUBTRACT/MULTIPLY/DIVIDE/ABSDIFF/MIN/MAX`、全部 bitwise、`IN_RANGE`、
`SCALE_ADD`、`CONVERT_SCALE_ABS`、`CONVERT_FP16`、`POW/EXP/LOG/PATCH_NANS`。

主要参考：

- `modules/core/src/arithm.simd.hpp`
- `modules/core/src/convert.simd.hpp`
- `modules/core/src/convert_scale.simd.hpp`
- `modules/core/src/matmul.simd.hpp`
- `modules/core/src/mathfuncs_core.simd.hpp`
- `modules/core/src/mathfuncs.cpp`

先后顺序：无 mask 同类型二元运算，饱和转换与 scale，mask/scalar 输入，超越函数。`SQRT` 当前只差
`1.05x`，保留为回归基线，不为追求覆盖重写。

验收：报告覆盖的类型/通道/ROI 全部正确；批次几何平均有明确提升；任何既有 case 不允许出现稳定
超过 `5%` 的回退。

### P-ACC-2：Core 归约、统计与 early-exit

处理 `COUNT_NON_ZERO/HAS_NON_ZERO/FIND_NON_ZERO`、`SUM/MEAN/MEAN_STD_DEV`、
`MIN_MAX_IDX/MIN_MAX_LOC`、`NORM/NORMALIZE`、`REDUCE/REDUCE_ARG_*`。

主要参考：

- `modules/core/src/count_non_zero.simd.hpp`
- `modules/core/src/has_non_zero.simd.hpp`
- `modules/core/src/sum.simd.hpp`
- `modules/core/src/mean.simd.hpp`
- `modules/core/src/stat.simd.hpp`
- `modules/core/src/minmax.simd.hpp`
- `modules/core/src/norm.simd.hpp`
- `modules/core/src/reduce.simd.hpp`

#### P-ACC-2 实施步骤

| Step | 算子/范围 | 实施边界 | 验收标准 |
| --- | --- | --- | --- |
| P-ACC-2.0 | reduction UI 基础设施与基线 | 新建 ODR-safe `reduce_ui.hpp`；统一连续 Mat、逐行 ROI、scalar tail、widen/分段归约和实际 dispatch tag；为本阶段全部算子补 Mode A scalar/auto case | 不改变任何公共结果；所有现有 reduction 测试通过；benchmark 能区分 `scalar/opencv_ui`；记录实施前 Mode A 和 Mode B 基线；短行、UI-disabled 和未覆盖 depth 必须明确报告 `scalar` |
| P-ACC-2.1 | `countNonZero`、`hasNonZero` | 共享“与零比较”row kernel；`countNonZero` 做向量计数和防溢出分段累加；`hasNonZero` 使用逐块 `v_check_any`/`!v_check_all(v_eq(...))` 并立即返回 | `+0/-0` 不计数，NaN/Inf 计为非零；覆盖全零、首元素、首个 vector、tail 和末元素命中；C1、非连续 ROI、非对齐宽度和各公开 depth 正确；VGA `countNonZero` 相对 scalar 至少 `1.5x`，`hasNonZero` 首元素命中不得随总像素数线性增长 |
| P-ACC-2.2 | `sum`、`mean`、`meanStdDev` | 建立 C1/C2/C3/C4 channel-aware widening 累加；`sum/mean` 共享 sum/count；`meanStdDev` 使用可合并的稳定 block statistics，禁止直接用不稳定的 `E[x^2]-E[x]^2` 替代当前 Welford 契约 | 整数输入在合法累计范围内结果精确；F32/F64 建立明确绝对/相对误差；覆盖 mask 全空/全选/稀疏、C1-C4、常量、大均值小方差、NaN/Inf、ROI 和 tail；三项代表 case 相对 scalar 均提升超过 `5%`，否则拒绝对应 UI 路径 |
| P-ACC-2.3 | `minMaxIdx`、`minMaxLoc`、`reduceArgMin/Max` | 共享“值+线性索引”归约；先处理单通道连续 row，再扩 ROI/axis；首索引和 `lastIndex` 使用独立 merge 规则，不能只归约值后重新扫描 | 固化 NaN、全 NaN、mask 全空、Inf、相同极值和 signed zero 语义；`minMaxIdx` 覆盖 ND 坐标，`minMaxLoc` 覆盖 2D 点；`reduceArg*` 覆盖 axis 0/1 和 first/last tie；结果值和索引与强制 scalar 一致 |
| P-ACC-2.4 | `norm`、`normalize` | `norm` 分拆单输入/双输入及 L1/L2/Inf；复用 widening、abs/max 和差值归约；`normalize` 组合已验证 norm/minmax 与现有 arithmetic/convert UI，单独记录 reduction 和 apply-scale 成本 | 覆盖 mask、C1-C4、零矩阵、极大/极小值、NaN/Inf、src/dst alias、dtype 转换和 `NORM_MINMAX` 常量输入；新分配 masked dst 与预分配 dst 保持既有语义；每种 norm 类型有独立 correctness 和性能行 |
| P-ACC-2.5 | `reduce` | 按 axis 1 连续行归约和 axis 0 列向/tile 归约分开实现；覆盖 `SUM/AVG/MAX/MIN/SUM2`，输出 depth 转换继续复用统一 `write_scalar`/转换契约 | axis 0/1、C1/C3、奇数尺寸、非连续 ROI、所有 rtype、合法 dtype 和 in-place alias 正确；整数饱和、浮点 NaN/Inf 和累计精度有明确测试；axis 0 的跨行 stride 不得退化成逐元素 `read_scalar` 调度 |
| P-ACC-2.6 | `findNonZero` | 复用非零 block 检测，只对命中 block 做标量 lane 枚举；保持 row-major 坐标顺序；`vector<Point>` 与 `Mat CV_32SC2` 两个输出共用一次扫描 | 覆盖全零、单点、首尾命中、稀疏、稠密、NaN/Inf、ROI 和非对齐 tail；两种输出逐项一致；全零和稀疏 case 相对 scalar 提升超过 `5%`，稠密 case 不允许稳定回退超过 `5%` |
| P-ACC-2.7 | 阶段收尾 | 全量 correctness、Mode A/Mode B、文档和跨平台门禁；检查共享 helper 没有为单个算子泄漏特殊语义 | Apple ARM Release CTest、header-only contract、UI-disabled、SSE2/AVX2 交叉编译通过；Mode B 全矩阵无新增 unsupported；既有 case 无稳定超过 `5%` 回退；更新日期性能报告和本计划状态 |

#### P-ACC-2 正确性总 gate

- 每个公共 API 都必须从同一个入口运行 `DispatchMode::ScalarOnly` 和
  `DispatchMode::Auto`，不能直接比较两个私有 helper 来代替公共合同。
- 覆盖连续 Mat、非连续 ROI、宽度小于一个 vector、恰好一个 vector、vector+1、C1/C3/C4
  和当前公开支持的 depth。没有进入 UI 的 depth 必须保留 scalar 并有 dispatch 断言。
- 非零类统一固化 `+0/-0/NaN/Inf`；极值类统一固化 NaN、相同值的首末索引和 mask 全空；
  累加类统一固化整数溢出边界、浮点累计顺序和允许误差。
- `meanStdDev` 必须保留大均值小方差场景的数值稳定性。任何仅为接近 upstream 速度而导致
  明显 cancellation 的实现不得进入公共 fast-path。
- `hasNonZero` 必须保留真正的块级 early-exit；`findNonZero` 必须保持 row-major 输出顺序；
  `reduceArg*` 的 `lastIndex` 不能在向量 merge 时退化为 first-index。

#### P-ACC-2 性能与架构总 gate

- Mode A 使用同一 Release 二进制、同一输入分别执行 `--dispatch scalar` 和
  `--dispatch auto`，记录耗时、吞吐量、checksum/数值 probe 和实际 dispatch。
- 每个接入公共入口的 UI 路径相对 scalar 至少提升 `5%`；`countNonZero` 的代表 case 目标为
  `>=1.5x`。未达到门槛的原型必须删除或保留为实验代码之外的文档结论。
- `hasNonZero` 分别测全零、首元素命中和尾元素命中；不得用首元素 case 的巨大倍数代表完整
  扫描性能。`findNonZero` 分别测 sparse/dense，不能只选择有利分布。
- Mode B 继续只比较 `cvh::headers_fast` 与 upstream OpenCV，并更新同一日期报告；
  单 case 稳定回退超过 `5%` 必须定位，不能被算子级几何平均掩盖。
- 所有实现直接使用 vendored OpenCV UI，不增加自定义 SIMD adapter，不引入 xsimd、IPP、
  BLAS 或需要链接的 `.cpp`。真实 x86 SSE/AVX 运行仍是最终跨平台 gate，Apple Clang
  交叉编译只负责模板实例化检查。

### P-ACC-3：Core 内存布局、通道与 GEMM

这批不能只看 SIMD。先消除逐元素 `Mat::at`、重复 `create` 和不必要的临时对象，再迁移 UI。

| 子批次 | 算子 | 重点 |
| --- | --- | --- |
| P-ACC-3.1 | `COPY_TO/MAT_SETTO/MAT_COPYTO/MAT_CLONE` | 连续块复制、masked `v_select`、零值 `memset` |
| P-ACC-3.2 | `TRANSPOSE/FLIP/ROTATE` | tiled transpose、`v_zip`、`v_reverse`、组合复用 |
| P-ACC-3.3 | `EXTRACT_CHANNEL/INSERT_CHANNEL/MIX_CHANNELS` | 常见 C3/C4 的 UI deinterleave/interleave |
| P-ACC-3.4 | `REPEAT/BROADCAST/HCONCAT/VCONCAT` | 行复制、倍增复制、连续块合并 |
| P-ACC-3.5 | `GEMM` | blocked packing、UI FMA 微内核；单独隔离 BLAS/LAPACK 对照 |

`FLIP_ND`、`MAT_CREATE`、`MAT_RESHAPE`、`SWAP` 主要是泛型 stride 或元数据成本，不作为 SIMD
迁移目标；在本批次只修复明显的复杂度和分配问题。

### P-ACC-4：Imgproc 滤波共享底座与导数

先收敛可以被多个高层 API 复用的 row/column filter 和边界缓存：

1. `BOX_FILTER/SQR_BOX_FILTER/GAUSSIAN/FILTER2D/SEP_FILTER2D`
2. `SCHARR/LAPLACIAN/SPATIAL_GRADIENT`
3. `INTEGRAL`
4. `PYR_DOWN/PYR_UP/BUILD_PYRAMID`

主要参考：

- `modules/imgproc/src/filter.simd.hpp`
- `modules/imgproc/src/filterengine.hpp`
- `modules/imgproc/src/box_filter.simd.hpp`
- `modules/imgproc/src/smooth.simd.hpp`
- `modules/imgproc/src/deriv.cpp`
- `modules/imgproc/src/spatialgradient.cpp`
- `modules/imgproc/src/sumpixels.simd.hpp`
- `modules/imgproc/src/pyramids.cpp`

验收必须覆盖 border 类型、奇偶尺寸、非对齐宽度、C1/C3/C4、8U/32F 和 in-place 限制。
`SOBEL` 当前 CVH 已领先，只作为共享 filter 底座的零回退门禁。

### P-ACC-5：颜色、去马赛克与几何采样

| 子批次 | 算子 | 共享策略 |
| --- | --- | --- |
| P-ACC-5.1 | `CVTCOLOR/CVT_COLOR_TWO_PLANE` | deinterleave、定点乘加、pack |
| P-ACC-5.2 | `DEMOSAICING` | Bayer 邻域向量计算、mask 与 pack |
| P-ACC-5.3 | `CONVERT_MAPS` | 坐标转换、舍入、交错存储 |
| P-ACC-5.4 | `REMAP/WARP_AFFINE/WARP_PERSPECTIVE` | 坐标块预计算、插值表、批量采样 |
| P-ACC-5.5 | `GET_RECT_SUB_PIX` | 复用 bilinear row kernel |

主要参考：

- `modules/imgproc/src/color_rgb.simd.hpp`
- `modules/imgproc/src/color_yuv.simd.hpp`
- `modules/imgproc/src/color.simd_helpers.hpp`
- `modules/imgproc/src/demosaicing.cpp`
- `modules/imgproc/src/imgwarp.cpp`
- `modules/imgproc/src/imgwarp.avx2.cpp`
- `modules/imgproc/src/imgwarp.sse4_1.cpp`
- `modules/imgproc/src/samplers.cpp`

`.avx2.cpp/.sse4_1.cpp` 只用于理解 block layout；公共实现仍必须使用 UI。几何采样应先减少坐标与权重
重复计算，再决定哪些内循环值得 UI 化。

### P-ACC-6：非线性、形态学、累积与强度变换

1. `MEDIAN_BLUR/BILATERAL_FILTER/STACK_BLUR`
2. `ERODE/DILATE`
3. 四个 `ACCUMULATE*` 与 `BLEND_LINEAR`
4. `THRESHOLD/THRESHOLD_WITH_MASK/ADAPTIVE_THRESHOLD`
5. `EQUALIZE_HIST/LUT/APPLY_COLOR_MAP`
6. `CREATE_HANNING_WINDOW`

主要参考：

- `modules/imgproc/src/median_blur.simd.hpp`
- `modules/imgproc/src/bilateral_filter.simd.hpp`
- `modules/imgproc/src/stackblur.cpp`
- `modules/imgproc/src/morph.simd.hpp`
- `modules/imgproc/src/accum.simd.hpp`
- `modules/imgproc/src/blend.cpp`
- `modules/imgproc/src/thresh.cpp`
- `modules/imgproc/src/equalize_hist.simd.hpp`
- `modules/core/src/lut.simd.hpp`
- `modules/imgproc/src/phasecorr.cpp`

`ADAPTIVE_THRESHOLD` 应优先复用 P-ACC-4 的 box/Gaussian 和本批次 threshold，不再维护独立像素循环。
`CANNY` 当前仅落后 `1.04x`，只做回归门禁。

### P-ACC-7：Scalar-small 与低收益收尾

包括 `BORDER_INTERPOLATE`、核生成 API、2x3/3x3 变换矩阵 API，以及已经接近或快于 upstream 的
操作。处理规则：

- 不为单次调用只有几十个标量运算的 API 引入复杂 SIMD。
- 先检查 benchmark 是否把分配、异常检查或测试夹具算进被测区间。
- 只接受能降低真实流水线耗时、同时保持代码体积合理的改动。

## 5. Core 完整对应表

下表的“差距”冻结为 P-ACC 开始前的 2026-07-24 算子级几何平均，用于保留优化优先级。
最新实测以同日期性能报告和第 0 节实施状态为准。
表格中同一单元格内省略目录的后续文件，与该单元格中前一个完整路径位于同一目录。

| 报告算子 | 差距 | 类型 | upstream CPU/UI 对应文件 | 批次与提取重点 |
| --- | ---: | --- | --- | --- |
| `ABSDIFF` | OpenCV 12.94x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.dispatch.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_absdiff`/饱和差 |
| `ADD` | OpenCV 5.55x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.dispatch.cpp`、`arithm.simd.hpp` | P-ACC-1，统一 binary loop |
| `BITWISE_AND` | OpenCV 13.07x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_and` |
| `BITWISE_NOT` | OpenCV 19.78x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_not` |
| `BITWISE_OR` | OpenCV 13.02x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_or` |
| `BITWISE_XOR` | OpenCV 12.86x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_xor` |
| `BORDER_INTERPOLATE` | OpenCV 1.18x | Scalar-small | `modules/core/src/copy.cpp` | P-ACC-7，分支与取模优化，无独立 SIMD |
| `BROADCAST` | OpenCV 3773.58x | Memory | `modules/core/src/matrix_transform.cpp` | P-ACC-3.4，shape flatten 与块复制 |
| `CHECK_RANGE` | OpenCV 1.52x | CPU-blocked | `modules/core/src/mathfuncs.cpp`、`minmax.dispatch.cpp`、`minmax.simd.hpp` | P-ACC-2/7，按块 early-exit；可复用 min/max |
| `CONVERT_FP16` | OpenCV 4.00x | UI-direct | `modules/core/src/convert.dispatch.cpp`、`convert.simd.hpp` | P-ACC-1，向量 FP16 pack/unpack |
| `CONVERT_SCALE_ABS` | OpenCV 7.20x | UI-direct | `modules/core/src/convert_scale.dispatch.cpp`、`convert_scale.simd.hpp` | P-ACC-1，scale/abs/saturate |
| `COPY_TO` | OpenCV 158.03x | UI-direct | `modules/core/src/copy.cpp` | P-ACC-3.1，masked `v_select`；无 mask 用块复制 |
| `COUNT_NON_ZERO` | OpenCV 29.37x | UI-direct | `modules/core/src/count_non_zero.dispatch.cpp`、`count_non_zero.simd.hpp` | P-ACC-2，compare 与向量归约 |
| `DIVIDE` | OpenCV 2.94x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.dispatch.cpp`、`arithm.simd.hpp` | P-ACC-1，scale/divide 与零值语义 |
| `EXP` | OpenCV 4.42x | UI-direct | `modules/core/src/mathfuncs.cpp`、`mathfuncs_core.dispatch.cpp`、`mathfuncs_core.simd.hpp` | P-ACC-1，UI math kernel |
| `EXTRACT_CHANNEL` | OpenCV 57.57x | CPU-blocked | `modules/core/src/channels.cpp`；参考 `split.simd.hpp` | P-ACC-3.3，常见 C3/C4 deinterleave |
| `FIND_NON_ZERO` | OpenCV 2.75x | CPU-blocked | `modules/core/src/count_non_zero.dispatch.cpp`、`count_non_zero.simd.hpp` | P-ACC-2，向量检测加标量输出压缩 |
| `FLIP` | OpenCV 208.25x | UI-direct | `modules/core/src/matrix_transform.cpp` | P-ACC-3.2，`v_reverse` 与 C3 交错处理 |
| `FLIP_ND` | OpenCV 95.11x | Memory | `modules/core/src/matrix_transform.cpp` | P-ACC-3.2，stride/block copy；upstream 泛型路径无独立 UI |
| `GEMM` | OpenCV 78.01x | CPU-blocked | `modules/core/src/matmul.dispatch.cpp`、`matmul.simd.hpp` | P-ACC-3.5，packing/block/FMA；排除 `hal_internal.cpp` 的 CBLAS |
| `HAS_NON_ZERO` | OpenCV 32258.06x | UI-direct | `modules/core/src/has_non_zero.dispatch.cpp`、`has_non_zero.simd.hpp` | P-ACC-2，块级 `v_check_any` 与 early-exit |
| `HCONCAT` | OpenCV 1.98x | Memory | `modules/core/src/matrix_operations.cpp`、`copy.cpp` | P-ACC-3.4，预分配与连续块复制 |
| `INSERT_CHANNEL` | OpenCV 57.02x | CPU-blocked | `modules/core/src/channels.cpp`；参考 `merge.simd.hpp` | P-ACC-3.3，常见 C3/C4 interleave |
| `IN_RANGE` | OpenCV 13.25x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，双边 compare 与 channel reduce |
| `LOG` | OpenCV 3.51x | UI-direct | `modules/core/src/mathfuncs.cpp`、`mathfuncs_core.dispatch.cpp`、`mathfuncs_core.simd.hpp` | P-ACC-1，UI math kernel |
| `MAT_CLONE` | OpenCV 1.05x | Memory | `modules/core/src/copy.cpp`、`matrix.cpp` | P-ACC-3.1，连续 copy；低收益门禁 |
| `MAT_CONVERTTO` | OpenCV 1.03x | UI-direct | `modules/core/src/convert.dispatch.cpp`、`convert.simd.hpp`、`convert_scale.simd.hpp` | P-ACC-1，保持现有性能 |
| `MAT_COPYTO` | CVH 1.00x | Memory | `modules/core/src/copy.cpp` | P-ACC-3.1，只作零回退门禁 |
| `MAT_CREATE` | OpenCV 13.35x | Memory | `modules/core/src/matrix.cpp` | P-ACC-3/7，复用容量与元数据，无 SIMD |
| `MAT_RESHAPE` | OpenCV 2.93x | Scalar-small | `modules/core/src/matrix.cpp` | P-ACC-7，O(1) header 更新，无 SIMD |
| `MAT_SETTO` | OpenCV 75.99x | UI-direct | `modules/core/src/copy.cpp` | P-ACC-3.1，zero/memset、scalar expand 与 masked store |
| `MAX` | OpenCV 9.19x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_max` |
| `MEAN` | OpenCV 4.45x | UI-direct | `modules/core/src/mean.dispatch.cpp`、`mean.simd.hpp` | P-ACC-2，widen 与分段累加 |
| `MEAN_STD_DEV` | OpenCV 8.42x | UI-direct | `modules/core/src/mean.dispatch.cpp`、`mean.simd.hpp`、`stat.simd.hpp` | P-ACC-2，sum/sqsum 共享遍历 |
| `MIN` | OpenCV 9.20x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.simd.hpp` | P-ACC-1，`v_min` |
| `MIN_MAX_IDX` | OpenCV 15.73x | UI-direct | `modules/core/src/minmax.dispatch.cpp`、`minmax.simd.hpp` | P-ACC-2，值与索引同步归约 |
| `MIN_MAX_LOC` | OpenCV 15.68x | UI-direct | `modules/core/src/minmax.dispatch.cpp`、`minmax.simd.hpp` | P-ACC-2，复用 minMaxIdx |
| `MIX_CHANNELS` | OpenCV 56.72x | CPU-blocked | `modules/core/src/channels.cpp`；参考 `split.simd.hpp`、`merge.simd.hpp` | P-ACC-3.3，常见映射特化，泛型保留 scalar |
| `MULTIPLY` | OpenCV 5.37x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.dispatch.cpp`、`arithm.simd.hpp` | P-ACC-1，widen/mul/pack |
| `NORM` | OpenCV 12.77x | UI-direct | `modules/core/src/norm.dispatch.cpp`、`norm.simd.hpp` | P-ACC-2，L1/L2/Inf 分支 |
| `NORMALIZE` | OpenCV 18.52x | UI-composed | `modules/core/src/norm.dispatch.cpp`、`norm.simd.hpp`、`convert_scale.simd.hpp` | P-ACC-2，norm/minmax 加 scale-convert |
| `PATCH_NANS` | OpenCV 2.30x | UI-direct | `modules/core/src/mathfuncs.cpp` | P-ACC-1，bit mask compare 与 `v_select` |
| `POW` | OpenCV 4.68x | UI-direct | `modules/core/src/mathfuncs.cpp`、`mathfuncs_core.dispatch.cpp`、`mathfuncs_core.simd.hpp` | P-ACC-1，整数指数和通用指数分流 |
| `REDUCE` | OpenCV 45.77x | UI-direct | `modules/core/src/reduce.dispatch.cpp`、`reduce.simd.hpp` | P-ACC-2，axis-aware 向量归约 |
| `REDUCE_ARG_MAX` | OpenCV 6.04x | UI-direct | `modules/core/src/minmax.dispatch.cpp`、`minmax.simd.hpp` | P-ACC-2，复用 arg min/max |
| `REDUCE_ARG_MIN` | OpenCV 6.04x | UI-direct | `modules/core/src/minmax.dispatch.cpp`、`minmax.simd.hpp` | P-ACC-2，复用 arg min/max |
| `REPEAT` | OpenCV 693.00x | Memory | `modules/core/src/copy.cpp` | P-ACC-3.4，首块复制与倍增 `memcpy` |
| `ROTATE` | OpenCV 35.00x | UI-composed | `modules/core/src/matrix_transform.cpp` | P-ACC-3.2，transpose/flip 组合 |
| `SCALE_ADD` | OpenCV 1.22x | UI-direct | `modules/core/src/matmul.dispatch.cpp`、`matmul.simd.hpp` | P-ACC-1，FMA；低收益门禁 |
| `SQRT` | OpenCV 1.05x | UI-direct | `modules/core/src/mathfuncs.cpp`、`mathfuncs_core.simd.hpp` | P-ACC-1/7，保持现状，不优先重写 |
| `SUBTRACT` | OpenCV 5.27x | UI-direct | `modules/core/src/arithm.cpp`、`arithm.dispatch.cpp`、`arithm.simd.hpp` | P-ACC-1，统一 binary loop |
| `SUM` | OpenCV 4.40x | UI-direct | `modules/core/src/sum.dispatch.cpp`、`sum.simd.hpp` | P-ACC-2，widen 与分段累加 |
| `SWAP` | OpenCV 4.85x | Scalar-small | `modules/core/src/matrix_operations.cpp`、Mat header | P-ACC-7，O(1) 元数据交换，无 SIMD |
| `TRANSPOSE` | OpenCV 1.92x | UI-direct | `modules/core/src/matrix_transform.cpp` | P-ACC-3.2，tile、`v_zip`、4x4/8x8/16x16 |
| `VCONCAT` | OpenCV 1.83x | Memory | `modules/core/src/matrix_operations.cpp`、`copy.cpp` | P-ACC-3.4，预分配与连续块复制 |

## 6. Imgproc 完整对应表

本表沿用上一节的路径缩写规则。

| 报告算子 | 差距 | 类型 | upstream CPU/UI 对应文件 | 批次与提取重点 |
| --- | ---: | --- | --- | --- |
| `ACCUMULATE` | OpenCV 12.75x | UI-direct | `modules/imgproc/src/accum.dispatch.cpp`、`accum.simd.hpp` | P-ACC-6，load/convert/add/store |
| `ACCUMULATE_PRODUCT` | OpenCV 13.88x | UI-direct | `modules/imgproc/src/accum.dispatch.cpp`、`accum.simd.hpp` | P-ACC-6，双源乘加 |
| `ACCUMULATE_SQUARE` | OpenCV 14.31x | UI-direct | `modules/imgproc/src/accum.dispatch.cpp`、`accum.simd.hpp` | P-ACC-6，平方累加 |
| `ACCUMULATE_WEIGHTED` | OpenCV 14.38x | UI-direct | `modules/imgproc/src/accum.dispatch.cpp`、`accum.simd.hpp` | P-ACC-6，alpha/beta FMA |
| `ADAPTIVE_THRESHOLD` | OpenCV 2.01x | UI-composed | `modules/imgproc/src/thresh.cpp`、`box_filter.simd.hpp`、`smooth.simd.hpp` | P-ACC-6，复用 box/Gaussian 与 threshold |
| `APPLY_COLOR_MAP` | OpenCV 3.07x | CPU-blocked | `modules/imgproc/src/colormap.cpp`；参考 `modules/core/src/lut.simd.hpp` | P-ACC-6，表查找与 C3 交错输出 |
| `BILATERAL_FILTER` | OpenCV 25.75x | UI-direct | `modules/imgproc/src/bilateral_filter.dispatch.cpp`、`bilateral_filter.simd.hpp` | P-ACC-6，邻域权重、向量 exp/accumulate |
| `BLEND_LINEAR` | OpenCV 2.52x | UI-direct | `modules/imgproc/src/blend.cpp` | P-ACC-6，双源权重乘加 |
| `BOX_FILTER` | OpenCV 3.40x | UI-direct | `modules/imgproc/src/box_filter.dispatch.cpp`、`box_filter.simd.hpp` | P-ACC-4，rolling sum 与 pack |
| `BUILD_PYRAMID` | OpenCV 154.80x | UI-composed | `modules/imgproc/src/pyramids.cpp` | P-ACC-4，复用 pyrDown 并复用层间状态 |
| `CANNY` | OpenCV 1.04x | UI-direct | `modules/imgproc/src/canny.cpp` | P-ACC-6/7，现有 fast-path 零回退 |
| `CONVERT_MAPS` | OpenCV 212.72x | UI-direct | `modules/imgproc/src/imgwarp.cpp` | P-ACC-5，坐标 round/pack/interleave |
| `COPY_MAKE_BORDER` | OpenCV 2.64x | Memory | `modules/core/src/copy.cpp` | P-ACC-4，连续中心块 copy 与边界索引表 |
| `CREATE_HANNING_WINDOW` | OpenCV 34.29x | UI-direct | `modules/imgproc/src/phasecorr.cpp` | P-ACC-6，`v_cos` 与行乘 |
| `CVTCOLOR` | OpenCV 1.79x | UI-direct | `modules/imgproc/src/color.cpp`、`color_rgb.simd.hpp`、`color_yuv.simd.hpp`、`color_hsv.simd.hpp`、`color_lab.cpp` | P-ACC-5，补齐现有 UI 覆盖 |
| `CVT_COLOR_TWO_PLANE` | OpenCV 6.25x | UI-direct | `modules/imgproc/src/color.cpp`、`color_yuv.dispatch.cpp`、`color_yuv.simd.hpp` | P-ACC-5，NV12/NV21 定点转换 |
| `DEMOSAICING` | OpenCV 576.37x | UI-direct | `modules/imgproc/src/demosaicing.cpp` | P-ACC-5，Bayer 邻域、mask、pack |
| `DILATE` | OpenCV 8.78x | UI-direct | `modules/imgproc/src/morph.dispatch.cpp`、`morph.simd.hpp` | P-ACC-6，滑窗 max |
| `EQUALIZE_HIST` | OpenCV 2.10x | UI-composed | `modules/imgproc/src/histogram.cpp`、`equalize_hist.dispatch.cpp`、`equalize_hist.simd.hpp`；`modules/core/src/lut.simd.hpp` | P-ACC-6，histogram 加 LUT |
| `ERODE` | OpenCV 8.63x | UI-direct | `modules/imgproc/src/morph.dispatch.cpp`、`morph.simd.hpp` | P-ACC-6，滑窗 min |
| `FILTER2D` | OpenCV 2.78x | UI-direct | `modules/imgproc/src/filter.dispatch.cpp`、`filter.simd.hpp`、`filterengine.hpp` | P-ACC-4，row/filter2D kernel 与 ring buffer |
| `GAUSSIAN` | OpenCV 3.61x | UI-direct | `modules/imgproc/src/smooth.dispatch.cpp`、`smooth.simd.hpp`、`filter.simd.hpp` | P-ACC-4，separable fixed/general kernel |
| `GET_AFFINE_TRANSFORM` | CVH 2.06x | Scalar-small | `modules/imgproc/src/imgwarp.cpp` | P-ACC-7，6x6 solve，无 SIMD |
| `GET_DERIV_KERNELS` | OpenCV 2.44x | Scalar-small | `modules/imgproc/src/deriv.cpp` | P-ACC-7，小型核生成，无独立 SIMD |
| `GET_GABOR_KERNEL` | OpenCV 3.04x | Scalar-small | `modules/imgproc/src/gabor.cpp` | P-ACC-7，小核 exp/cos；先检查分配 |
| `GET_GAUSSIAN_KERNEL` | CVH 4.03x | Scalar-small | `modules/imgproc/src/smooth.dispatch.cpp` | P-ACC-7，保留现状 |
| `GET_PERSPECTIVE_TRANSFORM` | CVH 2.46x | Scalar-small | `modules/imgproc/src/imgwarp.cpp`；求解器在 `modules/core/src/matmul.simd.hpp` | P-ACC-7，8x8 solve，无专用像素 SIMD |
| `GET_RECT_SUB_PIX` | OpenCV 45.85x | CPU-blocked | `modules/imgproc/src/samplers.cpp` | P-ACC-5，bilinear 权重预计算与 row kernel |
| `GET_ROTATION_MATRIX_2D` | CVH 1.14x | Scalar-small | `modules/imgproc/src/imgwarp.cpp` | P-ACC-7，2x3 标量计算 |
| `GET_ROTATION_MATRIX_2D_` | CVH 1.13x | Scalar-small | `modules/imgproc/src/imgwarp.cpp` | P-ACC-7，2x3 标量计算 |
| `GET_STRUCTURING_ELEMENT` | OpenCV 4.99x | Scalar-small | `modules/imgproc/src/morph.dispatch.cpp` | P-ACC-7，小 mask 生成；先检查分配 |
| `INTEGRAL` | OpenCV 37.22x | UI-direct | `modules/imgproc/src/sumpixels.dispatch.cpp`、`sumpixels.simd.hpp` | P-ACC-4，prefix sum 的向量展开 |
| `INVERT_AFFINE_TRANSFORM` | OpenCV 2.11x | Scalar-small | `modules/imgproc/src/imgwarp.cpp` | P-ACC-7，2x3 解析逆，无 SIMD |
| `LAPLACIAN` | OpenCV 83.77x | UI-composed | `modules/imgproc/src/deriv.cpp`、`filter.dispatch.cpp`、`filter.simd.hpp` | P-ACC-4，复用 separable/filter2D；忽略 IPP |
| `LUT` | OpenCV 1.60x | UI-direct | `modules/core/src/lut.cpp`、`lut.dispatch.cpp`、`lut.simd.hpp` | P-ACC-6，保留现有 fast-path 并补类型 |
| `MEDIAN_BLUR` | OpenCV 202.59x | UI-direct | `modules/imgproc/src/median_blur.dispatch.cpp`、`median_blur.simd.hpp` | P-ACC-6，3x3/5x5 sorting network |
| `PYR_DOWN` | OpenCV 129.08x | UI-direct | `modules/imgproc/src/pyramids.cpp` | P-ACC-4，5-tap separable、widen/pack |
| `PYR_UP` | OpenCV 140.10x | UI-direct | `modules/imgproc/src/pyramids.cpp` | P-ACC-4，零插值 5-tap 与行缓存 |
| `REMAP` | OpenCV 19.88x | CPU-blocked | `modules/imgproc/src/imgwarp.cpp`、`imgwarp.avx2.cpp`、`imgwarp.sse4_1.cpp` | P-ACC-5，map block、插值表与 UI 采样 |
| `RESIZE` | OpenCV 1.51x | CPU-blocked | `modules/imgproc/src/resize.cpp`、`resize.hpp`、`resize.avx2.cpp`、`resize.sse4_1.cpp` | P-ACC-5/7，现有 UI 路径补覆盖；平台文件只作参考 |
| `SCHARR` | OpenCV 86.13x | UI-composed | `modules/imgproc/src/deriv.cpp`、`filter.simd.hpp` | P-ACC-4，固定导数核与 separable engine |
| `SEP_FILTER2D` | OpenCV 2.04x | UI-direct | `modules/imgproc/src/filter.dispatch.cpp`、`filter.simd.hpp`、`filterengine.hpp` | P-ACC-4，row/column UI 与 ring buffer |
| `SOBEL` | CVH 1.51x | UI-composed | `modules/imgproc/src/deriv.cpp`、`filter.simd.hpp` | P-ACC-4，只作共享底座零回退 |
| `SPATIAL_GRADIENT` | OpenCV 7.63x | UI-direct | `modules/imgproc/src/spatialgradient.cpp` | P-ACC-4，三行 load/expand 与 dx/dy 同算 |
| `SQR_BOX_FILTER` | OpenCV 47.40x | UI-direct | `modules/imgproc/src/box_filter.dispatch.cpp`、`box_filter.simd.hpp` | P-ACC-4，square/widen/rolling sum |
| `STACK_BLUR` | OpenCV 12.72x | UI-direct | `modules/imgproc/src/stackblur.cpp` | P-ACC-6，rolling weighted sum |
| `THRESHOLD` | OpenCV 26.02x | UI-direct | `modules/imgproc/src/thresh.cpp` | P-ACC-6，compare/select 与特殊常量分支 |
| `THRESHOLD_WITH_MASK` | CVH 1.00x | UI-direct | `modules/imgproc/src/thresh.cpp` | P-ACC-6，masked `v_select`；只作零回退 |
| `WARP_AFFINE` | OpenCV 11.55x | CPU-blocked | `modules/imgproc/src/imgwarp.cpp`、`imgwarp.avx2.cpp`、`imgwarp.sse4_1.cpp` | P-ACC-5，坐标 block 与插值复用 |
| `WARP_PERSPECTIVE` | OpenCV 10.50x | CPU-blocked | `modules/imgproc/src/imgwarp.cpp`、`imgwarp.avx2.cpp`、`imgwarp.sse4_1.cpp` | P-ACC-5，透视除法、坐标 block 与插值复用 |

## 7. 跨算子依赖

```text
arithm/convert UI
  -> normalize
  -> accumulate/blend

copy/layout UI
  -> channel ops
  -> rotate
  -> border/filter staging

filter row/column engine
  -> box/sqrBox/Gaussian/filter2D/sepFilter2D
  -> Sobel/Scharr/Laplacian/spatialGradient
  -> adaptiveThreshold
  -> pyramid

color load/deinterleave/pack
  -> cvtColorTwoPlane
  -> demosaicing

map coordinate/weight preparation
  -> remap
  -> warpAffine/warpPerspective
  -> getRectSubPix
```

这也是批次顺序不能只按报告倍数从大到小排列的原因。例如先单独实现 `BUILD_PYRAMID`，会重复解决
`pyrDown` 的同一问题；先单独实现 `LAPLACIAN`，会绕过后续必需的 filter engine。

## 8. 每批统一验收标准

### 正确性

- 与 upstream OpenCV 对比，覆盖 benchmark 当前类型、通道和尺寸矩阵。
- 增加非连续 ROI、in-place/alias、宽度小于一个 vector、恰好一个 vector、vector+1。
- 浮点算子明确 NaN、Inf、舍入、饱和和允许误差。
- scalar fallback 与 UI fast-path 使用同一公开 API，结果一致。

### 性能

- Mode A 必须证明新 UI 路径相对项目 scalar/baseline 有收益。
- Mode B 更新同一份日期性能报告或创建新的日期报告，保留历史结果。
- 新路径在 ARM NEON 和至少一个 x86 SSE/AVX 环境验证；不能只在 Apple ARM 上接受。
- 单 case 稳定回退超过 `5%` 时必须分析，不能用批次平均掩盖。
- 算法/内存受限算子必须拆分预计算、分配和像素内核，禁止把所有收益归为 SIMD。

### Header-only

- 不增加需要链接的项目 `.cpp`。
- 所有定义满足 inline/模板/内部链接规则。
- 两个及以上 translation unit 同时包含 `cvh::headers_fast` 并链接通过。
- `cvh::headers` 仍可作为 baseline；`cvh::headers_fast` 自动回退到 baseline。
- 不重新引入 xsimd、IPP、BLAS 或其他必需外部运行库。

## 9. P-ACC-8：性能缺口收敛

### P-ACC-8.0：Benchmark 归因

| 工作 | 实现要求 | 验收标准 |
| --- | --- | --- |
| 热点 case 拆解 | 为候选算子记录 `public_reuse`、`public_recreate`、`precompute`、`kernel` 中适用的组件；不适用项必须显式说明 | CSV 能按 `allocation_mode` 和 note 区分组件，同一输入和采样配置可复现 |
| 公共路径与内核分离 | 输出预分配不能掩盖 API 内部临时 buffer；benchmark-only direct kernel 只能调用已有 detail helper，不能复制生产算法 | public 与 direct kernel checksum/数值 probe 一致；报告同时给出端到端和内核耗时 |
| GEMM pack 归因 | 保留 NN end-to-end、pack-once、NT；增加单独 pack 成本和不同 M/N/K 形状 | 能区分 B packing、输出分配和 UI micro-kernel，不用单个 128 方阵代表全部 GEMM |
| Upstream CPU-only | 从本地 OpenCV 树建立 `WITH_LAPACK=OFF`、`WITH_IPP=OFF` 的独立 build，不覆盖现有 upstream build | 同机报告明确记录 build 配置；GEMM 同时有默认 upstream 与 CPU-only upstream 数据 |
| 基线冻结 | 使用 Apple ARM Release、单线程、`warmup=2`、`iters=100`、`repeats=3` | 目标 case 全部 `OK`；无新增 `UNSUPPORTED`；文档记录拆解结论 |

### P-ACC-8.1：Pyramid family

`PYR_DOWN/PYR_UP/BUILD_PYRAMID` 当前分别落后 upstream `8.10x/7.06x/13.23x`。
P-ACC-8.0 已证明索引与输出分配不足端到端的 `2%`，因此先迁移 5-tap row/column UI
kernel，并把整图 temporary 收敛为 ring rows；不单独做低收益的 index cache。
`buildPyramid` 必须复用每层 ring workspace，不能只是循环调用未优化的公共入口。

验收：U8/F32、C1/C3/C4、奇偶尺寸、合法 dstsize、全部支持 border、ROI 和 alias 正确；
三项 public reuse 都有独立 Mode A 行，代表 case 相对当前冻结基线至少提升 `1.5x`。

### P-ACC-8.2：Nonlinear filters

按 `medianBlur -> stackBlur -> bilateralFilter` 推进。三者虽然都属于非线性滤波，但不共享
像素算法，禁止为了形式统一建立大而泛化的 filter engine。

验收：kernel size、border、U8/F32 支持范围、ROI、in-place 和短行正确；每个被接受的新路径
相对自身 public reuse 基线至少提升 `1.5x`，否则保留现状并记录拒绝原因。

### P-ACC-8.3：Geometry sampling

先收敛 `convertMaps`，再让 `remap/warpAffine/warpPerspective` 复用坐标 block、定点插值表和
border mask。不能分别维护三份 bilinear inner loop。

验收：float/fixed map、nearest/linear、C1/C3/C4、ROI、全部公开 border、inverse map 和 alias
正确；端到端与 direct kernel 分开记录，既有 `getRectSubPix` 不得回退超过 `5%`。

### P-ACC-8.4：Filter/morphology

目标不是继续堆叠单算子 UI 循环，而是消除 box/Gaussian/derivative/morphology 中重复的
border materialization、临时 Mat 分配和 row-buffer 初始化。已经领先的 Sobel 只作回归门禁。

验收：共享 helper 至少被两个公共 API 使用；C1/C3/C4、U8/F32、ROI、anchor、border、tail
正确；每个修改后的既有 case 不得稳定回退超过 `5%`。

### P-ACC-8.5：Core GEMM/reduction

P-ACC-8.0 已证明 CVH GEMM 比 upstream 内建 CPU 路径快 `1.75x-3.41x`，因此不继续做
blocked packing 或为 Accelerate/BLAS 倍数追赶。GEMM 只保留 public shape/output overhead
诊断；本批主要处理 F32 单输入 `NORM_INF`、`meanStdDev` 和 axis reduce 中不必要的 F64
widening。

验收：GEMM 覆盖方阵、skinny/wide、NN/NT、pack-once 和 batch broadcast；归约覆盖既有
NaN/Inf、mask、channel 和精度合同；接受路径相对 scalar/baseline 至少提升 `1.5x`。

### P-ACC-8.6：收尾

Apple ARM 上运行默认 UI、UI-disabled、header-only contract、多 TU、Mode A/Mode B 全矩阵；
SSE2/AVX2 继续执行模板实例化。真实 x86 correctness 和性能必须在 x86 机器关闭，Apple
Clang 交叉编译不能替代运行验证。最终更新日期报告，并列出未收敛差距及其原因。
