# Imgproc 测试

`test/imgproc` 按算法 family 组织，不保留开发批次名。

## 目录职责

- `arithmetic/`：accumulate、blendLinear。
- `color/`：RGB/GRAY、YUV420、YUV422、YUV444、two-plane、demosaicing、colormap。
- `filtering/`：box/Gaussian、filter2D、sepFilter2D、median/bilateral/stack blur、
  derivatives、kernel、sqrBoxFilter 和 copyMakeBorder。
- `geometry/`：resize、transform matrix、convertMaps、remap、warp、rectSubPix、pyramid。
- `intensity/`：threshold、adaptiveThreshold、equalizeHist、LUT。
- `morphology/`、`feature/`、`statistics/`：各自稳定算法职责。
- `internal/`：median blur、derivatives 和 pyramid 的 scalar/UI 一致性。
- `upstream/`：OpenCV 移植 case。
- `integration/`：真实图像基础 pipeline。
- `support/`：独立 reference 和共享输入构造。
- `data/`：opencv_extra 快照及相对路径 manifest。

原来的大 `cvtColor` 文件按 RGB/GRAY、YUV420、YUV422、YUV444 拆开；共享
reference 只在 `support/cvtcolor_test_utils.hpp` 维护一份。公共测试不得调用生产
detail helper 生成期望值。

## 头文件独立编译

`test/smoke/imgproc_headers/` 为每个 `include/cvh/imgproc/*.h` 建立单独翻译单元，
由 `cvh_imgproc_headers_compile_smoke` 编译和运行。它替代了过去“同一翻译单元
include 所有头后直接 `SUCCEED()`”的弱检查。

## Fixture 与 upstream

同步固定的 opencv_extra 图片：

```bash
python3 scripts/sync_opencv_imgproc_fixtures.py \
  --opencv-extra-root /path/to/opencv_extra
```

同步 OpenCV upstream case 快照：

```bash
python3 scripts/sync_opencv_imgproc_cases.py \
  --opencv-root /path/to/opencv
```

生成的 manifest 只记录 upstream 项目、commit、相对 source/snapshot 路径和 hash，
不记录本机绝对路径。

## 运行

```bash
cmake -S . -B build-imgproc \
  -DCVH_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-imgproc \
  --target cvh_test_imgproc cvh_imgproc_headers_compile_smoke -j
ctest --test-dir build-imgproc \
  -R '^(cvh_test_imgproc|cvh_imgproc_headers_compile_smoke)$' \
  --output-on-failure
```
