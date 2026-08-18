# `benchmark` Framework

这个目录的目标不是临时跑几个数字，而是给 `cvh` 建立一套可以长期演进的性能判断系统。框架围绕两个模块展开：

- `core`：以 `Mat` 和基础数组算子为中心。
- `imgproc`：以图像预处理/后处理常用算子为中心。

框架同时支持两种 benchmark 模式，Pipeline 作为独立 suite 复用同一输出合同：

- **内部回归模式**：旧版本 header-only 和当前版本 header-only 对比，用来指导提速过程。
- **OpenCV 对比模式**：当前 header-only 实现和官方 OpenCV 主仓库实现对比，用来展示差距和选择优化优先级。

## Mode A: Internal Header-only Regression

内部回归模式回答的问题是：这次改动让 `cvh` 自己变快了，还是变慢了？

约束：

- 只比较 header-only 产物，不依赖 OpenCV，不依赖需要编译的 `.cpp` 扩展层。
- baseline 和 candidate 必须使用同一组 case、同一套输入生成规则、同一套编译参数和同一台机器。
- baseline 可以来自旧 commit、上一版发布产物、或同一二进制内强制 scalar fallback 的诊断行。
- baseline 和 candidate 都使用各自 commit 的 canonical
  `cvh::headers` benchmark target。
- 结果用于本项目优化决策，可以作为 CI gate。

推荐输出位置：

```text
benchmark/results/internal/<suite>/<profile>/baseline.csv
benchmark/results/internal/<suite>/<profile>/current.csv
benchmark/results/internal/<suite>/<profile>/report.md
benchmark/results/internal/<suite>/<profile>/meta.json
```

推荐实现名：

| Implementation | 含义 |
|---|---|
| `cvh_headers` | 各自 commit 的唯一公开计算 target；报告通过 baseline/current 输入区分版本。 |
| `scalar_fallback` | 同一二进制内强制 fallback 的诊断路径，只用于拆内核成本。 |
| `opencv_ui_fastpath` | 同一二进制内直接 OpenCV UI fast path 的诊断路径。 |

当前 canonical 纯 header-only benchmark：

| Target | Scope | 状态 |
|---|---|---|
| `cvh_benchmark_core_mat_header` | `Mat` 生命周期、布局、复制、转换、基础计算、random 和点变换。 | Mode A `core_mat` 聚合 target。 |
| `cvh_benchmark_imgproc_header` | 已接入聚合矩阵的 imgproc 公共 API，含区域/轮廓、形状、直方图和模板匹配。 | Mode A `imgproc` 聚合 target。 |
| `cvh_benchmark_pipeline_header` | prepared staged 与 packed/YUV F32/U8/S8、resize/letterbox fused 模型输入链；记录 execution group、完整中间图、workspace 和实际 route。 | Mode A `pipeline` canonical target。 |

当前专项诊断 benchmark：

| Target | Scope | 状态 |
|---|---|---|
| `cvh_benchmark_cvtcolor_bgr2gray_header` | `CV_8UC3` `BGR2GRAY` / `RGB2GRAY`，含 scalar/public/direct UI/micro rows。 | 可用于 imgproc 内部诊断。 |
| `cvh_benchmark_resize_bilinear_header` | `CV_8UC1` `INTER_LINEAR` exact 2x downsample，含 scalar/public/direct UI/micro rows。 | 可用于 imgproc 内部诊断。 |

最小运行示例：

```bash
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=OFF \
  -DCVH_BUILD_BENCHMARKS=ON

cmake --build build-bench -j --target \
  cvh_benchmark_core_mat_header \
  cvh_benchmark_imgproc_header \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header \
  cvh_benchmark_pipeline_header

mkdir -p benchmark/results/internal/imgproc/quick

./build-bench/cvh_benchmark_cvtcolor_bgr2gray_header \
  --profile quick \
  --warmup 3 \
  --iters 10 \
  --repeats 7 \
  --output benchmark/results/internal/imgproc/quick/cvtcolor_current.csv

./build-bench/cvh_benchmark_resize_bilinear_header \
  --profile quick \
  --warmup 3 \
  --iters 10 \
  --repeats 7 \
  --output benchmark/results/internal/imgproc/quick/resize_current.csv

./build-bench/cvh_benchmark_imgproc_header \
  --profile quick --warmup 1 --iters 3 --repeats 3 \
  --output benchmark/results/internal/imgproc/quick/imgproc_header_current.csv

./build-bench/cvh_benchmark_pipeline_header \
  --profile stable --warmup 3 --iters 3 --repeats 7 \
  --output benchmark/results/internal/pipeline/stable/current.csv
```

Pipeline target 在同一二进制、同一输入和单线程下比较 `staged_p0` 与 `fused_p1`。
它先逐 byte 验证输出，再采样 prepared `run()`；CSV 不从平台推断执行路径，而是记录
`PipelineRunInfo` 的 actual/observed route，并同时记录 execution groups、完整中间图和
workspace。指定 `--output` 时会同步生成 `<output>.meta.json`。

Core 逐元素算子还支持在同一二进制内强制 scalar，用于排除编译器和机器差异：

```bash
cmake --build build-bench -j --target cvh_benchmark_core_mat_header

./build-bench/cvh_benchmark_core_mat_header \
  --profile quick --dispatch scalar --threads 1 \
  --output benchmark/results/internal/core_mat/quick/scalar.csv

./build-bench/cvh_benchmark_core_mat_header \
  --profile quick --dispatch auto --threads 1 \
  --output benchmark/results/internal/core_mat/quick/opencv_ui.csv
```

`dispatch_path` 记录公共 API 最终命中的 `scalar` 或 `opencv_ui`，不以 target
名称推测实际执行路径。

P-ACC-8 的 GEMM 归因可以只运行 square/skinny/wide 矩阵和
end-to-end/pack-only/pack-once/kernel-only 组件：

```bash
./build-bench/cvh_benchmark_core_mat_header \
  --profile quick --ops GEMM --dispatch auto --threads 1 \
  --warmup 2 --iters 100 --repeats 3
```

`cvh_benchmark_imgproc_header` 在 640x480 anchor 上记录 pyramid 的
`public_reuse`、`public_recreate`、`precompute` 和
`precomputed_workspace` kernel 行。两者都会显式把 `cvh` 线程数设置为
CSV 中记录的线程数，不能仅依赖 OpenMP 环境变量。

`core_mat` 的逐元素矩阵同时包含 Mat-Mat、Mat-Scalar/Scalar-Mat、`inRange` 的
Mat/Scalar bounds，以及 masked Mat-Mat/Mat-Scalar bitwise 变体；这些 variant 使用相同
输入分别运行 `--dispatch scalar` 和 `--dispatch auto`，用于独立判断 broadcast、
compare/reduce 和 mask/select 的收益。

F32 数学 case 包含 `EXP`、`LOG`、`POW(power=1.75)`、`POW(power=3)` 和
`PATCH_NANS`。它们同样支持 `--dispatch scalar|auto`；通用幂与整数幂分开记录，避免
`exp(log(x) * power)` 的成本掩盖整数 exponentiation-by-squaring fast-path。

runner 已支持通过 `git worktree` 拉起旧版本，例如：

```text
benchmark/internal/run_header_regression.sh --baseline-ref <git-ref> --suite core_mat --profile quick
benchmark/internal/run_header_regression.sh --baseline-ref <git-ref> --suite imgproc --profile quick
```

## Mode B: OpenCV Upstream Compare

OpenCV 对比模式回答的问题是：当前 `cvh` header-only 实现和官方 OpenCV 主仓库还有多大差距？

约束：

- 对比对象只包括 `cvh::headers` 的 UI-forced 运行结果和官方 OpenCV `core` / `imgproc`。
- CVH 侧固定 `OpenCVUIOnly`，实现名为 `cvh_ui`；发现 NEON/AVX2 tag 或 UI-required case 未命中 `opencv_ui` 时直接失败。
- 这个模式默认是 report/log-only，不作为每个 PR 的硬 gate。
- 每份结果必须记录 `cvh` commit、OpenCV commit、编译器、平台、CPU、线程数、profile 和 CMake 选项。
- 不生成 scalar、NEON-only、AVX2-only、`native` 或 `lite` 产品实现行。

本地 OpenCV 源码或 slim benchmark checkout 通过环境变量提供，例如：

```text
CVH_OPENCV_DIR=/path/to/opencv-bench-slim
```

仓库自带 runner 的默认位置是：

```text
benchmark/opencv_compare/opencv-bench-slim
```

推荐输出位置：

```text
benchmark/results/opencv/<suite>/<profile>/compare.csv
benchmark/results/opencv/<suite>/<profile>/report.md
benchmark/results/opencv/<suite>/<profile>/meta.json
```

目标实现名：

| Implementation | 含义 |
|---|---|
| `cvh_ui` | `cvh::headers` + forced `OpenCVUIOnly`。 |
| `opencv` | 同机同编译模式下的官方 OpenCV。 |

现状：

- `benchmark/opencv_compare/` 已经可以生成 `cvh vs OpenCV` 报告。
- P2-P0 的 17 个操作族已接入 26 条同参数 upstream 对比 case，可以
  通过 `--ops PHASE2_P0` 独立运行，不必重跑全部历史矩阵。
- 该目录已经裁剪为纯 header-only compare：只用 `cvh_ui` 对比 OpenCV。

## Suites

### `core_mat`

第一优先级是 `Mat` 基础成本，而不是泛化数学库。

候选 case：

| Area | Case |
|---|---|
| Allocation/lifetime | `Mat::create`, `release`, reuse create, reallocation create。 |
| Copy/layout | `clone`, `copyTo`, continuous copy, ROI/non-contiguous copy。 |
| Fill/convert | `setTo`, `convertTo`, saturating cast。 |
| Shape/view | `reshape`, ROI construction, step/stride traversal。 |
| Basic array ops | `add`, `subtract`, `multiply`, `divide`, `compare`, `merge`, `split`，等这些进入 header-only contract 后再纳入 gate。 |

核心指标：

- `ns/op`：适合小矩阵和元数据操作。
- `MElems/s`：适合元素级算子。
- `GB/s`：适合内存带宽型路径。
- allocation count 或 reuse/recreate 标记：用于拆 `Mat::create` 影响。

目标 target：

```text
cvh_benchmark_core_mat_header
```

### `imgproc`

第一优先级是 AI vision 预处理/后处理中高频、容易被 OpenCV 用户感知的算子。

候选 case：

| Area | Case |
|---|---|
| Resize | `INTER_NEAREST`, `INTER_NEAREST_EXACT`, `INTER_LINEAR`，覆盖 downsample/upsample/非整数缩放/非对齐宽度。 |
| Color | BGR/RGB/GRAY/BGRA/RGBA，YUV encode/decode 家族按已支持布局覆盖。 |
| Threshold/LUT | `threshold`, `LUT`。 |
| Border/filter | `copyMakeBorder`, `filter2D`, `sepFilter2D`, `boxFilter`, `blur`, `GaussianBlur`。 |
| Grad/morphology | `Sobel`, `Canny`, `erode`, `dilate`, `morphologyEx`。 |

核心指标：

- `ms/call`：用户最直观的延迟。
- `MPix/s`：图像算子主指标。
- `GB/s`：内存带宽型路径辅助指标。
- `tail_ratio`：SIMD 尾部比例，尤其关注非 16/32 对齐宽度。
- `allocation_mode`：`reuse` / `recreate`，用于拆公共入口分配成本。
- `dispatch_path`：`scalar_fallback` / `opencv_ui` / `platform_fastpath`。

目标 target：

```text
cvh_benchmark_imgproc_header
```

## Common CSV Schema

新 benchmark 尽量收敛到同一批字段。现有专项 benchmark 可以先保持原 schema，但新 runner/report 应能映射到下面的标准字段。

| Field | 含义 |
|---|---|
| `schema_version` | canonical Mode A CSV 的兼容版本。 |
| `mode` | `internal` 或 `opencv_compare`。 |
| `suite` | `core_mat` 或 `imgproc`。 |
| `module` | `core` / `imgproc`。 |
| `op` | 算子名。 |
| `variant` | 插值、border、kernel size、color code 等变体。 |
| `depth` | `CV_8U` / `CV_32F` 等。 |
| `channels` | 通道数。 |
| `layout` | `continuous` / `roi` / YUV layout 等。 |
| `shape` | 人类可读尺寸。 |
| `pixels` | 输出像素数；core 非图像 case 可为元素数。 |
| `implementation` | Mode A 产品行使用 `cvh_headers`，专项诊断可使用 `scalar_fallback` / `opencv_ui_fastpath`；Mode B 只使用 `cvh_ui`, `opencv`。 |
| `dispatch_path` | 实际命中的内部路径。 |
| `allocation_mode` | `reuse` / `recreate` / `none`。 |
| `tail_ratio` | 当前 SIMD lane 下每行标量尾部比例。 |
| `warmup`, `iters`, `repeats`, `threads` | 采样参数。 |
| `min_ms`, `median_ms` | 最小值和中位数。 |
| `mpix_per_sec`, `melems_per_sec`, `gb_per_sec` | 吞吐指标。 |
| `checksum` | 防止编译器消除和粗粒度结果一致性检查。 |
| `status`, `note` | 支持状态和跳过原因。 |

## Profiles And Gates

| Profile | 用途 | 建议采样 | Gate |
|---|---|---|---|
| `quick` | 本地开发和 PR 预检查。 | 小到中尺寸，短采样。 | internal regression 可 fail。 |
| `stable` | 合并前或阶段收口。 | 更多 repeats，固定线程，固定机器。 | internal regression 可 fail。 |
| `full` | 周期性扫描。 | 全尺寸矩阵和更多边界 case。 | 默认 log-only。 |
| `micro` | 拆内核成本。 | 单内核、单职责。 | 不直接作为产品性能 gate。 |

建议 gate：

- 内部回归 quick：默认允许最多 `8%` slowdown。
- 内部回归 stable：对已接受 fast path 应收紧到 `5%` 左右；噪声大的平台可 log-only。
- OpenCV 对比：默认不 fail，只输出 `OpenCV/CVH` 和 unsupported cases。
- 只有当某个 fast path 已经以 benchmark 证据进入支持表，才为它设置硬性性能门槛。

## Measurement Rules

- Release 构建，尽量固定编译器和 CMake 选项。
- 单线程优先；多线程只在明确测试并行路径时启用。
- 同一份输入在同一 case 内复用。
- 同时记录 `reuse` 和 `recreate`，避免把 `Mat::create` 成本误判为 kernel 成本。
- micro benchmark 只解释瓶颈，不代表用户可见 API 性能。
- 每个结果必须带 metadata；没有 metadata 的 CSV 只能作为临时诊断。

## Cleanup Rules

- 新生成结果放入 `benchmark/results/` 或 `benchmark/opencv_compare/results/`，不再放在 `benchmark/` 根目录。
- `benchmark/*.csv` 视为历史阶段产物，不再作为长期文档入口。
- OpenCV compare 的滚动 `current_*` 文件是生成产物；审核过的日期命名快照将
  英文 Markdown、raw CSV 和 metadata 一起跟踪在
  `benchmark/opencv_compare/results/`。
- `core_mat` / `imgproc` 聚合 target 是产品 benchmark；专项 target 只在
  聚合覆盖等价后裁剪。

The framework is implemented. Current behavior is owned by this README, the
OpenCV compare README, executable scripts, and the result schema rather than a
completed rollout plan.
