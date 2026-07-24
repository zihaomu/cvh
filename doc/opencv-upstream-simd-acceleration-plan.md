# OpenCV Upstream SIMD 算子加速计划

> - 状态：实施中，P-ACC-1 第五组 Apple ARM 已完成，真实 x86 运行待验证
> - 日期：2026-07-24
> - 性能基线：`benchmark/opencv_compare/results/2026-07-24-opencv-upstream-performance.md`
> - OpenCV 参考树：`/Users/zmu/work/my_project/ocvh/opencv`
> - OpenCV 参考版本：`4.14.0-pre`，commit `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`

## 0. 实施状态

| 阶段 | 状态 | 当前范围 |
| --- | --- | --- |
| P-ACC-0 | Apple ARM 已完成，x86 运行待验证 | dispatch tag、scalar 强制模式、tail/ROI/多 TU 门禁、Mode A 路径标识 |
| P-ACC-1 | 第五组 Apple ARM 已完成，x86 运行待验证 | `PATCH_NANS/EXP/LOG/POW` 的 F32 UI fast-path 已进入公共路径 |
| P-ACC-2 | 未开始，下一阶段 | Core 归约、统计与 early-exit |
| P-ACC-3 至 P-ACC-7 | 未开始 | 按跨算子依赖顺序推进 |

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

先后顺序：向量累加与 widening，min/max 与索引，norm/normalize，early-exit，findNonZero 输出压缩。

验收除数值正确性外，必须覆盖全零、首元素命中、尾元素命中、NaN、Inf 和非连续 ROI。
`HAS_NON_ZERO` 必须保留块级 early-exit，不能退化为完整扫描后归约。

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

## 9. 下一组入口

P-ACC-1 的五组 Apple ARM 实现与测量已经完成，下一步进入 P-ACC-2。按共享依赖先拆：

1. `COUNT_NON_ZERO/HAS_NON_ZERO`：先建立 compare、widen/count 和块级 early-exit，
   当前 OpenCV 分别领先约 `33.40x` 和 `25641.03x`。
2. `SUM/MEAN/MEAN_STD_DEV`：建立可复用的 widening 与分段累加底座，避免三套独立遍历。
3. `MIN_MAX_IDX/MIN_MAX_LOC/REDUCE_ARG_*`：复用值与索引同步归约。
4. `NORM/NORMALIZE/REDUCE/FIND_NON_ZERO`：在基础归约稳定后接入，输出压缩路径单独 gate。

前五组的真实 x86 SSE/AVX correctness 与 Mode A 仍是未关闭 gate；Apple Clang
x86_64 交叉编译只用于模板实例化检查，不能替代该运行验证。
