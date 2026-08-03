# `test/smoke` 目录规划

## 目录职责

提供最快速的可用性验证，确保入口头和最小运行链路不被破坏。

## 阶段计划

### P0：基础 smoke

- `cvh_header_compile_smoke`：验证公开头可编译。
- `cvh_include_only_smoke`：验证直接 include 场景可编译运行，包含主 `include/` 和 vendored OpenCV UI include root。
- `cvh_pipeline_smoke`：验证 Imgcodecs → Imgproc 的最小纯头文件处理链路。
- `cvh_resize_dispatch_smoke`：验证默认 target 的基础 Imgproc dispatch 和输出。
- `cvh_opencv_intrin_smoke`：验证 `cvh/core/simd/opencv_ui.h` gateway 可独立 include，并能直接使用 OpenCV UI `cv::v_*` / `cv::VTraits`。
- `cvh_opencv_intrin_x86_smoke`：在 x86 AVX2 编译参数下验证 direct OpenCV UI 128-bit 和 256-bit 类型。
- `cvh_imgproc_header_odr_smoke`：两个 translation unit 同时包含并调用
  imgproc header fast-path，验证 inline/telemetry 的 ODR 链接安全。
- `cvh_core_headers_compile_smoke`：逐个编译 Core 顶层公共头，并在配置期校验头文件清单完整性。
- `cvh_imgproc_headers_compile_smoke`：逐个编译 Imgproc 顶层公共头，并在配置期校验头文件清单完整性。
- `cvh_imgcodecs_headers_compile_smoke`：逐个编译 Imgcodecs 顶层公共头，并在配置期校验头文件清单完整性。
- `cvh_highgui_headers_compile_smoke`：使用 `cvh::highgui` 编译可选 HighGUI 公共头。
- `cvh_highgui_header_odr_smoke`：以无显示测试模式验证两个 translation unit 共享 inline HighGUI 状态。
- `cvh_aggregate_headers_compile_smoke`：分别编译 `cvh/cvh.h` 与兼容转发头 `cvh.h`。

### P1：主线优先

- 增加纯 header-only 路径 smoke。

### P2：发布门禁

- 将 smoke 作为 CI 必跑项。
- 任意公共头破坏必须在 smoke 阶段被拦截。

## 完成定义（DoD）

- 新环境下可通过 smoke 快速确认项目可用性。
