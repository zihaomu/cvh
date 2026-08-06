# cvh v0.1 高频 NEON 热点加速计划

更新时间：2026-08-06  
状态：H0–H1 完成；H2 进行中；H3–H5 待执行

## 1. 目的与阶段定位

本文定义 v0.1 发布前一轮严格限域的 ARM64 NEON 优化，只处理三个高频
kernel family：

1. `cvtColor` 的 packed RGB/BGRA 与 YUV U8 路径；
2. `resize` 的 U8C3 bilinear 路径；
3. `Sobel`、`Scharr`、`spatialGradient` 共享的 U8 3x3 derivative 路径。

本轮不是 API 扩展，不增加颜色格式、插值模式、导数参数或数据类型。已有
scalar 和 OpenCV Universal Intrinsics（UI）实现全部保留，direct NEON 只作为
产品 `Auto` dispatch 中优先级更高的内部实现。

本计划延续以下现有合同：

- [CPU optimization and dispatch](cpu-optimization.md)；
- [OpenCV UI / specialized ISA checklist](opencv-ui-kernel-migration-checklist.md)；
- [v0.1 Imgproc performance-floor acceleration](cvh-v0.1-imgproc-performance-floor-acceleration-plan.md)。

## 2. 范围边界

### 2.1 本轮包含

#### CVTCOLOR U8

- packed channel：`BGR<->RGB`、三通道到四通道、四通道到三通道、
  `BGRA/RGBA->GRAY`；
- interleaved YUV444：`BGR/RGB<->YUV`；
- YUV420：`NV12/NV21/I420/YV12 -> BGR/RGB`，以及已有 upstream 单调用
  参考的反向编码路径；
- YUV422：`YUY2/UYVY/NV16/NV61 <-> BGR/RGB` 中已有公开支持并能建立
  upstream 对称 benchmark 的路径；
- continuous、合法 stride/ROI、vector tail 和小图 fallback。

`BGR2GRAY_u8` 已通过 UI 达到基本持平，只作为非回退对照，不重写 direct
NEON。F32 color conversion 不在本轮范围。

#### Resize U8C3

- `INTER_LINEAR`；
- downscale `0.5x`、`0.75x` 和 upscale `1.5x` 代表比例；
- continuous 与非连续 ROI；
- interior NEON、边界和 tail fallback；
- public-call end-to-end 成本，包括映射和系数准备。

`INTER_NEAREST`、`INTER_NEAREST_EXACT`、U8C1、F32、C4 和其他插值模式
保持现状。

#### Shared 3x3 derivative U8

- `Sobel`：输入 U8，C1/C3/C4，`dx/dy=(1,0)/(0,1)`，`ksize=3`；
- `Scharr`：输入 U8，C1，`dx/dy=(1,0)/(0,1)`；
- `spatialGradient`：输入 U8C1，同时输出 S16 `dx` 与 `dy`；
- S16/F32 输出 adapter、常用 `scale=1`、`delta=0`；
- `BORDER_REPLICATE`、`BORDER_REFLECT_101` 与合法 isolated ROI。

其他输入深度、kernel size、高阶导数、非常用 scale/delta 或边界组合继续
使用当前 UI/scalar 路径。Canny 调用点不在本轮迁移范围；若共享内部 primitive
影响 Canny，必须证明无正确性或性能回退，否则保持原路径。

### 2.2 明确不包含

- AVX2/SSE 专用实现；x86 继续使用现有 UI/scalar fallback；
- `GaussianBlur`、pyramid、warp、morphology、bilateral、GEMM 或其他算子；
- DEMOSAICING 重新进入 v0.1 支持面；
- 新公开宏、CMake product target 或运行时依赖；
- 修改容差、舍入、饱和、色彩系数、chroma sampling、border 或异常合同；
- 仅为 benchmark 特定尺寸写死分支；
- 用多线程、减少 case、关闭 checksum 或隐藏 scalar tail 获得性能数字；
- release packaging、版本号和发布说明。

## 3. 冻结基线

基线产物：

- [product-auto full report](../benchmark/opencv_compare/results/2026-08-06-v0.1-rc-auto-opencv-upstream-performance.en.md)；
- [raw CSV](../benchmark/opencv_compare/results/2026-08-06-v0.1-rc-auto-opencv-upstream-performance.csv)；
- [run metadata](../benchmark/opencv_compare/results/2026-08-06-v0.1-rc-auto-opencv-upstream-performance.meta.json)。

| 项目 | 基线 |
| --- | --- |
| CVH revision | `cbd5076b06f5c938f571b84a8b239c8a3165b8a9`，clean |
| OpenCV revision | `d48bf69f65444a13f8a34b8982b083c1b78fa0e8`；dirty 仅为 `.gitignore` |
| OpenCV version | `4.14.0` |
| 平台 | Apple M5 / Darwin arm64 |
| 构建 | Release，单线程，`warmup=1, iters=10, repeats=3` |
| CVH mode | product `cvh_auto` |
| 全量结果 | 370 rows；369 OK；1 expected unsupported |

相对性能继续使用以下口径：

```text
relative performance = OpenCV latency / CVH latency
1.0 = 持平；小于 1.0 = CVH 更慢
```

### 3.1 目标 case 基线

| family / case | CVH | OpenCV | relative | 等价差距 |
| --- | ---: | ---: | ---: | ---: |
| BGR2RGB U8C3 480p | `0.1154 ms` | `0.0149 ms` | `0.1288` | OpenCV `7.77x` |
| BGR2BGRA U8C3 480p | `0.2452 ms` | `0.0175 ms` | `0.0712` | OpenCV `14.05x` |
| BGRA2GRAY U8C4 480p | `0.1578 ms` | `0.0365 ms` | `0.2315` | OpenCV `4.32x` |
| BGR2YUV U8C3 480p | `0.2711 ms` | `0.0689 ms` | `0.2542` | OpenCV `3.93x` |
| YUV2BGR U8C3 480p | `0.2764 ms` | `0.0524 ms` | `0.1896` | OpenCV `5.27x` |
| BGR2I420 U8C3 480p | `0.3473 ms` | `0.0526 ms` | `0.1513` | OpenCV `6.61x` |
| I420 to BGR U8 480p | `0.8246 ms` | `0.0653 ms` | `0.0792` | OpenCV `12.62x` |
| NV12 to BGR U8 480p | `0.3765 ms` | `0.0657 ms` | `0.1744` | OpenCV `5.73x` |
| YUY2 to BGR U8 480p | `0.3730 ms` | `0.0641 ms` | `0.1719` | OpenCV `5.82x` |
| Resize linear U8C3 0.75x | `0.3336 ms` | `0.0563 ms` | `0.1687` | OpenCV `5.93x` |
| Resize linear U8C3 0.75x ROI | `0.3323 ms` | `0.0554 ms` | `0.1667` | OpenCV `6.00x` |
| Sobel U8C1 1080p | `1.7960 ms` | `0.7684 ms` | `0.4278` | OpenCV `2.34x` |
| Sobel U8C3 480p | `0.5836 ms` | `0.3032 ms` | `0.5195` | OpenCV `1.92x` |
| Sobel U8C4 480p | `0.8482 ms` | `0.3964 ms` | `0.4673` | OpenCV `2.14x` |
| Scharr U8C1 to F32 480p | `0.2753 ms` | `0.1080 ms` | `0.3921` | OpenCV `2.55x` |
| spatialGradient U8C1 to S16 480p | `0.1843 ms` | `0.0325 ms` | `0.1765` | OpenCV `5.66x` |

## 4. 当前实现审计

| family | 当前实现 | 主要缺口 | 首选方向 |
| --- | --- | --- | --- |
| BGR2GRAY U8C3 | OpenCV UI | 已基本持平 | 保持并设非回退 gate |
| packed RGB/BGRA | per-pixel header fastpath | C3/C4 load/store 与 channel shuffle 标量化 | NEON interleaved load/store |
| YUV444/420/422 | fixed-point/float scalar loops | chroma broadcast、widen MAC、pack/store 标量化 | 共享 fixed-point NEON row/block kernels |
| Resize U8C3 linear | precomputed map + scalar pixel interpolation | C3 gather、双线性 MAC、pack/store | block map + NEON arithmetic/interleaved store |
| Sobel U8C1 | UI generic derivative | 通用 filter 调度与重复 load 成本 | fused 3-row direct kernel |
| Sobel U8C3/C4 | scalar | 整个 interior 未向量化 | channel-stride NEON interior |
| Scharr U8C1 | UI generic derivative | 未使用窄 Scharr fused kernel | 共享 derivative3 NEON |
| spatialGradient | UI，同时计算 dx/dy | load/expand 次数和 32-bit 中间量偏多 | 单次 load 生成双输出 |

审计结论：三个 family 不能共用一个泛化卷积或转换模板。应共享 dispatch、
telemetry、tail 和测试合同，但分别使用针对内存布局的窄 kernel。

## 5. Dispatch 与源码结构

### 5.1 产品选择顺序

每个 eligible case 固定使用：

```text
Auto:
  accepted direct NEON
    -> existing OpenCV UI when available
    -> existing scalar/header fastpath

OpenCVUIOnly:
  existing OpenCV UI
    -> scalar fallback

ScalarOnly:
  scalar fallback

NeonOnly:
  direct NEON when the exact support predicate is satisfied
    -> scalar fallback for unsupported cases
```

Direct NEON 必须同时满足：

- `CVH_DETAIL_HAVE_NEON_KERNEL`；
- `cpu::neon_runtime_available()`；
- mode 为 `Auto` 或 `NeonOnly`；
- exact depth/channel/layout/parameter predicate；
- 经 H0 测量确定的最小 workload threshold。

`OpenCVUIOnly` 不得执行 direct NEON。`CVH_ENABLE_OPTIMIZATION=0` 必须在不
包含 ARM intrinsic 的环境中继续编译并运行。

### 5.2 建议内部文件

```text
include/cvh/imgproc/detail/cvtcolor_neon.hpp
include/cvh/imgproc/detail/resize_neon.hpp
include/cvh/imgproc/detail/derivative3_neon.hpp
```

- `cvtcolor.h`、`resize.h`、`sobel.h`、`derivatives.h` 保持公开入口与合同；
- specialized header 只暴露 `cvh::detail` 内部窄函数；
- 不从公开函数签名暴露 `uint8x16_t` 等 NEON 类型；
- scalar/UI 实现继续是独立、可直接测试的 fallback；
- 不复制一套 project-owned 通用 SIMD 抽象。

### 5.3 Dispatch 观测合同

当前报告的三层字段继续保留：

```text
algorithm_path -> dispatch_path -> isa_observed
```

本阶段为复合算子增加 stage-level `kernel_route`，示例：

```text
cvtcolor_yuv420: load=neon;convert=neon;store=neon;tail=scalar
resize_linear_u8c3: map=scalar;gather=block;interpolate=neon;tail=scalar
derivative3_u8c3: border=scalar;interior=neon;store=s16;tail=scalar
```

规则：

- 只有实际执行至少一个 direct NEON 主循环才记录 `dispatch_path=neon` 和
  `isa_observed=neon`；
- 完全落入小图或 tail fallback 的 case 必须记录真实 UI/scalar dispatch；
- `kernel_route` 记录本次 case 实际经过的阶段，不以编译能力代替执行事实；
- telemetry 保持 internal、thread-local/ODR-safe，不成为公开 API；
- CSV renderer、文档检查和历史 schema 兼容必须同步更新。

## 6. 正确性与性能门禁

### 6.1 正确性

- U8 packed/YUV、S16 derivative 输出按既有 upstream 合同逐元素比较；
- F32 derivative 保持既有 tolerance，不增加 NEON 专属放宽；
- 覆盖宽度 `1`、`lanes-1`、`lanes`、`lanes+1`、奇数宽和大图；
- YUV420/422 覆盖合法偶数宽高、UV 顺序、RGB/BGR 顺序和 plane offset；
- 覆盖 continuous、ROI、non-contiguous step、unaligned row start；
- 覆盖边界首尾行/列、scalar tail 和完全不满足 workload gate 的 fallback；
- 覆盖 `Auto`、`NeonOnly`、`OpenCVUIOnly`、`ScalarOnly`；
- OpenCV upstream differential、optimization-off、header compile 和 ODR 全部
  不得新增失败。

### 6.2 Direct NEON 保留门槛

每个 kernel 只在以下条件全部满足时保留：

1. 三轮 stable median 相对本阶段 clean baseline 至少提升 `1.25x`；
2. 至少关闭目标 case 与 OpenCV 延迟差距的 `30%`；
3. 同 family 非目标 case 不得稳定回退超过 `5%`；
4. full Imgproc 与 full compare 几何平均不得稳定回退超过 `1%`；
5. 代码仍保持 header-only，scalar/UI fallback 与 dispatch tag 均可验证。

低于门槛的 direct NEON candidate 必须回退；不得仅因已经投入开发成本而保留。

### 6.3 Family 完成目标

| family | 必达 relative floor | stretch goal |
| --- | ---: | ---: |
| packed RGB/BGRA U8 | `>= 0.70` | `>= 0.85` |
| YUV444/420/422 U8 目标路径 | `>= 0.45` | `>= 0.65` |
| Resize bilinear U8C3 | `>= 0.50` | `>= 0.67` |
| Sobel/Scharr 3x3 U8 | `>= 0.65` | `>= 0.80` |
| spatialGradient U8 | `>= 0.50` | `>= 0.70` |

Family floor 使用目标 case 三轮中位数逐项判断，不用 family 几何平均掩盖最差
case。若某项因 upstream 特殊 HAL 或平台库无法达到，必须保留原始数据、说明
绝对损失，并由用户明确批准例外；计划本身不自动批准。

## 7. H0–H5 实施批次

| 批次 | 内容 | 状态 | 主要产物 |
| --- | --- | --- | --- |
| H0 | focused matrix 与 stage dispatch 可信化 | 完成 | baseline、`kernel_route`、三模式报告 |
| H1 | CVTCOLOR packed/YUV U8 NEON | 完成 | color NEON header、tests、before/after |
| H2 | Resize bilinear U8C3 NEON | 进行中 | resize NEON header、tests、before/after |
| H3 | Sobel/Scharr/spatialGradient shared NEON | 待执行 | derivative3 NEON header、tests、before/after |
| H4 | 全量正确性与跨平台 fallback | 待执行 | ARM runtime、x86 compile/runtime、sanitizer evidence |
| H5 | product-auto full report 与文档收口 | 待执行 | date-named Markdown/CSV/metadata、完成定义 |

### H0：Focused matrix 与 telemetry

1. 在 canonical OpenCV compare runner 增加 `V01_NEON_HOT` filter，映射到
   `CVTCOLOR,RESIZE,SOBEL,SCHARR,SPATIAL_GRADIENT`，不创建阶段性 benchmark
   target。
2. 补齐 480p、720p、1080p、odd-width ROI 与 scalar-tail case。
3. Resize 增加 `0.5x/0.75x/1.5x` U8C3 bilinear；颜色转换增加主流 YUV
   decode/encode 分辨率；derivative 增加 dx/dy、S16/F32 和 C1/C3/C4。
4. 增加 `kernel_route` schema、renderer 和文档合同检查。
5. 在 clean revision 上冻结 Auto/UI/scalar 三轮 stable baseline。

H0 只改善观测，不修改目标 kernel 性能。

#### H0 实时记录（2026-08-06）

- `V01_NEON_HOT` 已接入 canonical runner，stable 每个 mode 生成 70 行：
  CVTCOLOR 36、Resize 17、Sobel 9、Scharr 4、spatialGradient 4；
- matrix 已包含 480p、720p、1080p、Phase1 240x320 对照和 odd-width ROI；
- Auto/UI 当前各为 52 scalar + 18 UI，ScalarOnly 为 70 scalar，未观察到
  direct ISA；这是 H1–H3 的真实 before dispatch 分布；
- CSV/Markdown 已增加向后兼容的 `kernel_route`；旧 CSV 缺少该字段时 renderer
  使用主 dispatch 作为 fallback；
- quick 与 stable Auto/UI/scalar smoke 均通过，所有 row 的 stage route 非空；
- targeted Imgproc correctness 84/84 通过；Imgproc 独立头编译与 ODR smoke
  均通过；
- 首次测试构建发现 compare runner 会重新配置同一 build directory 并关闭
  `CVH_BUILD_TESTS`。因此测试与 compare 从本记录起固定使用不同 build 目录，
  “无 target”不计为通过；
- H0 观测代码提交为 `3af51d2`；在该 clean revision 上完成三轮 stable
  Auto/UI/scalar baseline，每轮 210 行，metadata 均记录
  `repo_git_dirty=false`；
- Auto 三轮逐 case 中位数的 family relative 几何平均为：CVTCOLOR
  `0.2003`、Resize `0.3571`、Sobel `0.4866`、Scharr `0.3938`、
  spatialGradient `0.1775`；
- 代表性 before：BGR2RGB `0.1273`、BGR2BGRA `0.0730`、I420-to-BGR
  `0.0786`、Resize U8C3 0.75x `0.1803`、Sobel C1 1080p `0.4300`、
  Scharr `0.3800`、spatialGradient `0.1809`；
- 三轮 Auto dispatch 均为 52 scalar + 18 UI，ISA 均为 unknown；H0 所有
  correctness、dispatch、schema 与 clean baseline 条件齐备，H0 关闭。

### H1：CVTCOLOR packed/YUV U8

#### H1 实时记录（2026-08-06）

- 已新增 `detail/cvtcolor_neon.hpp`，direct selector 仅允许 ARM64
  `Auto/NeonOnly`、U8、至少一个 16-pixel vector block 且总像素不少于
  256；UIOnly、ScalarOnly、小 workload 和非目标 code 保持原路径；
- packed 第一批已覆盖 C3/C4 R/B swap、3->4 alpha fill、4->3 alpha drop、
  BGRA/RGBA->GRAY、GRAY->C3/C4；BGR/RGB->GRAY UI 路径未被替换；
- 新增 `CvtColorDispatchInternalTest`，覆盖 37 像素 tail、non-contiguous
  ROI、Auto/NeonOnly/UIOnly/ScalarOnly、短 workload fallback 和 stage route；
- packed 实现完成后，颜色专项 61/61 通过，Imgproc 独立头编译与 ODR smoke
  通过；
- 首轮 stable Auto 探测结果（尚未作为三轮 gate 结论）：BGR2RGB 480p 从
  H0 `0.117590 ms` 降至 `0.016054 ms`，`7.32x`；BGR2BGRA 从
  `0.245404 ms` 降至 `0.024856 ms`，`9.87x`；BGRA2GRAY 从
  `0.163715 ms` 降至 `0.038531 ms`，`4.25x`；对应 relative 分别为
  `1.0051`、`0.7479`、`0.9595`，均达到 packed `>=0.70` 初测 floor；
- 三个代表 packed case 均报告 `dispatch=neon`、`isa=neon` 和实际
  `load/shuffle-or-gray/store/tail` route；H1 继续实施 YUV，最终仍需三轮
  clean stable 数据才能关闭 retention gate。
- YUV decode 第一批已共用 fixed-point 8-lane conversion primitive，并接入
  NV12/NV21、I420/YV12、YUY2/UYVY 的 16-pixel block；新增 direct 大图、
  tail 与 non-contiguous step 的 scalar/NEON byte-exact 对照；
- 首轮 stable Auto 探测中，I420->BGR 480p 从 H0 `0.842492 ms` 降至
  `0.118158 ms`（`7.13x`，relative `0.5960`），YUY2->BGR 从
  `0.391808 ms` 降至 `0.081146 ms`（`4.83x`，relative `0.8554`），
  NV12->BGR 从 `0.389704 ms` 降至 `0.078004 ms`（`5.00x`，relative
  `0.9066`）；三项均超过 YUV `>=0.45` 初测 floor，并具有完整 direct
  NEON stage route；
- YUV420 planar/semi-planar 与 YUV422 packed encode 已完成；2x2/2x1
  chroma average 使用 widen pair sum 保持 `(sum+round)>>shift` 原合同；direct
  non-contiguous ROI、tail 与 scalar byte-exact 测试通过；
- encode 首轮 stable Auto：BGR->I420 480p 从 H0 `0.347540 ms` 降至
  `0.073442 ms`（`4.73x`，relative `0.7421`），BGR->YUY2 从
  `0.328054 ms` 降至 `0.078867 ms`（`4.16x`，relative `0.7449`）；
- interleaved BGR/RGB<->YUV444 使用现有 float coefficient 和 AArch64 scalar
  相同的 FMA evaluation order；随机 67x257 non-contiguous ROI 验证了 direct
  路径 byte-exact，未用 fixed-point 近似替换舍入合同；
- interleaved YUV444 首轮 stable Auto：BGR->YUV 480p `0.127002 ms`、
  relative `0.5901`，YUV->BGR `0.116490 ms`、relative `0.4869`；相对 H0
  分别提升约 `2.20x`、`2.40x`，均超过 YUV `>=0.45` 初测 floor；
- H1 当前颜色专项 65/65 通过，独立头编译、ODR 和 `git diff --check`
  通过；待 clean revision 三轮 stable Auto/UI/scalar 后关闭 H1。
- H1 实现提交为 `eb776fb`；该 clean revision 上三轮 stable 每轮均为 210
  行，metadata 均记录 `repo_git_dirty=false`；
- 三轮逐 case 中位数 gate：packed 目标最差 relative `0.7438`、最小
  candidate 提升 `4.32x`、最小 gap closure `97.3%`；YUV 目标最差
  relative `0.4854`、最小 candidate 提升 `2.37x`、最小 gap closure
  `75.2%`；全部超过 family floor、`1.25x` retention 和 `30%` gap
  closure；
- 三轮 dispatch 分布一致：Auto 为 30 NEON + 18 UI + 22 scalar；UIOnly
  为 18 UI + 52 scalar；ScalarOnly 为 70 scalar；UIOnly/ScalarOnly 未观察到
  direct NEON；
- 非 CVTCOLOR focused case 的三轮中位数没有一项回退超过 `5%`；H1
  correctness、dispatch、performance 和 non-target gate 齐备，H1 关闭。

#### H1.1 Packed channel

- 使用 `vld3q_u8` / `vld4q_u8` 与 `vst3q_u8` / `vst4q_u8` 完成 channel
  reorder、alpha fill/drop 和 BGRA/RGBA gray；
- 一个 row kernel 通过 compile-time channel order 复用，不复制每个 code；
- 小宽度和剩余像素保留现有 scalar 实现；
- BGR2GRAY UI 路径保持不变并作为回退检测。

#### H1.2 YUV444

- 固定点 widen multiply-accumulate；
- 明确沿用现有 coefficient、rounding、delta 和 saturating narrow；
- interleaved load/store 与 BGR/RGB order 使用共享 template flag。

#### H1.3 YUV420/422 decode

- 以两行 luma 和共享 chroma block 为单位处理 NV12/NV21/I420/YV12；
- 对 U/V 做 lane duplication，不按像素重复读取和计算；
- YUY2/UYVY 与 NV16/NV61 共用 fixed-point conversion primitive；
- plane offset 和 stride 继续由现有 layout validator 决定。

#### H1.4 Encode

- Y plane 与 2x2/2x1 chroma aggregation 分阶段向量化；
- 仅对 upstream 有单调用对称参考的路径设性能 gate；
- `BGR2NV12` 继续保留正确性与 dispatch 观测，但不伪造 upstream 性能参考。

### H2：Resize bilinear U8C3

#### H2 实时记录（2026-08-06）

- 已新增 `detail/resize_neon.hpp` 与 `ResizeDispatchInternalTest`；selector 仅接入
  ARM64 `Auto/NeonOnly`、U8C3、`INTER_LINEAR` 且输出 workload 不少于
  256 像素的 case，UIOnly、ScalarOnly、小图和非目标类型保持原路径；
- `0.5x` 使用 2x2 widen/pair-add 专用 kernel；通用比例每次调用只建立一次
  x/y map，并对安全的 16-source-pixel 窗口使用 `vld3q + vqtbl + float FMA +
  vst3` 处理 8 个输出，border、无法安全 overread 的尾部继续使用 scalar；
- direct correctness 当前覆盖 `0.5x/0.75x/1.5x`、36x52 与 480x640 的
  exact 0.75x、odd-size ROI、non-contiguous step、tail 以及
  Auto/NeonOnly/UIOnly/ScalarOnly；已通过 scalar byte-exact 对照；
- 最终保留源码上，Resize targeted 13/13、Imgproc 独立头编译与 ODR smoke、
  `git diff --check` 均通过；
- 当前保留候选的单轮 stable 探测：`0.5x` relative 为 `2.20–2.50`，
  `1.5x` 为 `0.96–0.99`；`0.75x` continuous/ROI 的 CVH latency 约
  `0.157–0.160 ms`，相对 H0 `0.333 ms` 提升约 `2.1x`，关闭约 `60%`
  的 OpenCV 延迟差距，但 relative 仍仅约 `0.35`；
- 已实测并删除三个未通过候选：12-output float block 因计算/存储额外 lane
  使 continuous 降至约 `0.32x` 且 ROI 无法形成安全窗口；精确 3/4 fixed-point
  kernel 为维持现有 float/std::round 合同需要 half-way tie 修正，最终仅
  `0.08–0.14x`；关闭 tie 修正虽可达到约 `0.60x`，但 checksum 与 byte-exact
  合同不一致，因此仅保留在 ignored probe 数据中，未进入源码；
- 当前通用 candidate 已超过 `1.25x` retention 与 `30%` gap-closure 门槛，
  但 0.75x continuous/ROI 尚未达到 family `>=0.50` floor。H2 保持进行中，
  在没有进一步正确性等价的优化或用户明确批准例外前不关闭。

1. 每次调用只计算一次 x index、x weight 和 y mapping；禁止 per-pixel 分配。
2. interior block 将 source index/weight 打包，NEON 完成横向、纵向 fixed-point
   interpolation、rounded narrow 和 C3 store。
3. 对连续 source window 使用向量 load/table shuffle；无法形成安全连续窗口时
   使用 block gather buffer，不做越界 overread。
4. border、短行、退化尺寸和 tail 使用现有 scalar path。
5. workload threshold 由 `0.5x/0.75x/1.5x` 三类实测共同决定，不只针对
   当前 `0.75x` case。
6. 若 gather/pack 成本使 direct NEON 未通过保留门槛，回退 candidate，并先
   保留有独立收益的数据流优化；此时不得标记 `dispatch_path=neon`。

### H3：Shared derivative3 U8

1. 建立三行、channel-stride aware 的 interior kernel；边界列独立 scalar。
2. Sobel 使用 `[1,2,1]` smoothing 与 `[-1,0,1]` derivative；Scharr 使用
   `[3,10,3]` 与同一 derivative 结构。
3. U8 widen 后使用 signed 16-bit/32-bit accumulation；Sobel/Scharr 的理论
   范围必须在实现注释和测试中证明。
4. `spatialGradient` 复用同一批 row load，同时生成 dx/dy，避免两次遍历。
5. S16 store 使用 saturating narrow；F32 store 只在最后一步转换并应用
   scale/delta。
6. C1/C3/C4 interior 共用算术表达式，通过 channel stride 决定左右邻点；
   不为每个通道复制完整 kernel。
7. Canny 默认继续使用当前已验证路径；只有无回退证据充分时才允许复用新
   primitive，且不计入 H3 完成条件。

### H4：全量与跨平台

- Apple ARM64：执行 Auto/NeonOnly/UIOnly/ScalarOnly targeted correctness；
- optimization-on/off full unit tests；
- OpenCV contract full differential；
- ASan/UBSan；
- header independent compile、ODR、install consumer；
- Linux x86_64：证明 direct NEON 完全不可达，现有 UI/scalar 路径通过；
- focused stable 三轮与 full product-auto 一轮无非目标回退。

本轮不要求新增 x86 AVX2 性能实现，但不接受因 NEON header 引入造成的 x86
编译或运行回退。

### H5：报告与收口

1. 从 clean release-candidate revision 运行单线程 product `cvh_auto` full
   profile；
2. 提交 Markdown、CSV、metadata 三件套；
3. 报告逐 case 展示 algorithm、dispatch、ISA 与 stage `kernel_route`；
4. 更新 benchmark result index、本计划状态和必要的 CPU dispatch 文档；
5. 不修改 API support matrix，因为本轮没有新增支持面。

## 8. Canonical 命令

### 8.1 Focused baseline / before-after

```bash
CVH_COMPARE_SKIP_OPENCV_SETUP=1 \
CVH_OPENCV_DIR=../opencv \
CVH_OPENCV_CONFIG_DIR=../opencv/build-slim \
CVH_COMPARE_BUILD_DIR=build-v01-neon-hot-compare \
CVH_COMPARE_THREADS=1 \
benchmark/opencv_compare/run_compare.sh \
  --profile stable \
  --impls auto,ui,scalar \
  --ops V01_NEON_HOT
```

每个批次使用相同输入和 sampling 连续运行三次，以各 case median 的中位数
判定。单轮数据不关闭 performance gate。

### 8.2 Targeted tests

```bash
cmake -S . -B build-v01-neon-hot-tests \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON \
  -DCVH_ENABLE_OPENCV_COMPARE=ON \
  -DCVH_ENABLE_OPTIMIZATION=ON \
  -DOpenCV_DIR=../opencv/build-slim

cmake --build build-v01-neon-hot-tests --parallel 2

build-v01-neon-hot-tests/cvh_test_imgproc \
  --gtest_filter='CvtColor*:Resize*:Derivatives*:MorphologyDerivativesUpstreamTest.Imgproc_Sobel*:*DispatchInternalTest*'

ctest --test-dir build-v01-neon-hot-tests --output-on-failure
```

实际 GTest filter 在 H0 按新增 suite 名称收紧；不得使用不存在的 filter 结果
作为通过证据。

### 8.3 Scalar / header-only gates

```bash
cmake -S . -B build-v01-neon-hot-scalar \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_ENABLE_OPTIMIZATION=OFF
cmake --build build-v01-neon-hot-scalar --parallel 2
ctest --test-dir build-v01-neon-hot-scalar --output-on-failure

./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
./scripts/check_docs.sh
python3 scripts/sync_opencv_intrin.py --check
git diff --check
```

## 9. 提交与回退边界

建议提交边界：

1. `bench: add v0.1 neon hot-kernel observability`；
2. `perf(imgproc): add neon packed color kernels`；
3. `perf(imgproc): add neon yuv kernels`；
4. `perf(imgproc): add neon u8c3 bilinear resize`；
5. `perf(imgproc): add shared neon derivative3 kernel`；
6. `docs(bench): publish v0.1 neon hot-kernel results`。

每个性能提交必须包含对应 correctness、dispatch 和 focused benchmark 证据，
并能独立回退。不同 family 不在同一提交中互相抵消回退。

## 10. 实时更新规则

执行期间本文是唯一阶段状态 owner：

- 开始一个批次前，将状态从“待执行”改为“进行中”；
- 每完成 correctness、dispatch、benchmark gate，立即在对应批次下追加命令、
  revision、结果和未决问题；
- candidate 被回退时记录原始数字与原因，不把失败实验留在产品路径；
- 只有所有必需证据齐备后才把批次改为“完成”；
- dated benchmark 报告一旦提交即保持不可变，更正使用新日期产物。

## 11. 完成定义

- [x] H0 focused matrix 覆盖三类 family 的主流尺寸、ROI、tail 和 fallback；
- [x] 每个 focused case 均有完整
      `algorithm_path -> dispatch_path -> isa_observed -> kernel_route`；
- [x] packed RGB/BGRA U8 达到 `relative >= 0.70`；
- [x] 目标 YUV U8 路径达到 `relative >= 0.45`；
- [ ] Resize bilinear U8C3 达到 `relative >= 0.50`；
- [ ] Sobel/Scharr 目标路径达到 `relative >= 0.65`；
- [ ] spatialGradient 达到 `relative >= 0.50`；
- [ ] 所有保留的 direct NEON kernel 通过 `1.25x` candidate retention gate；
- [ ] Auto 在 eligible Apple ARM64 case 选择 NEON；UIOnly/ScalarOnly 不执行
      direct NEON；
- [ ] 小图、unsupported parameter、ROI、non-contiguous、border 和 tail 均走
      正确 fallback；
- [ ] optimized、optimization-off、OpenCV differential、sanitizer、header
      compile、ODR 和 install consumer 全部通过；
- [ ] Linux x86_64 编译与现有 UI/scalar runtime 无新增失败；
- [ ] full Imgproc 与 full compare 无超过 `1%` 的稳定非目标回退；
- [ ] 最终 product-auto 报告来自 clean revision，并提交 Markdown/CSV/metadata；
- [ ] 本文状态、benchmark index 与必要 dispatch 文档实时同步；
- [ ] API coverage 和 v0.1 support matrix 未因本轮性能实现发生变化。
