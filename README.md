# cvh — cv-header-only

**An independent, header-only C++ computer vision library with familiar OpenCV-style APIs.**

`cvh` stands for **cv-header-only**. It provides a focused set of `core`,
`imgproc`, `imgcodecs`, and optional `highgui` APIs for applications that want familiar
OpenCV-style types and call patterns without linking the OpenCV libraries.
Include the headers directly or link an interface CMake target—there is no
required library build step.

`cvh` is its own project and brand. OpenCV is the compatibility reference for
API shape, naming, and behavior; `cvh` is not an OpenCV distribution and is not
affiliated with or endorsed by the OpenCV project.

Repository: [github.com/zihaomu/cvh](https://github.com/zihaomu/cvh)

> **Latest performance report:** [cvh v0.1 RC vs OpenCV upstream benchmark (2026-08-04, English)](benchmark/opencv_compare/results/2026-08-04-v0.1-rc-opencv-upstream-performance.en.md)
>
> Apple M5, single-threaded full profile: 370 cases using the single public
> `cvh::headers` target. This immutable snapshot forced `cvh_ui`; it remains UI
> diagnostic evidence and must not be used to infer product-auto GEMM
> performance. New reports default to `cvh_auto`. Raw data, metadata, and
> immutable older snapshots are listed in the
> [benchmark result index](benchmark/opencv_compare/results/README.md).

## Status

- **Project direction:** pure header-only
- **Brand:** `cvh` (cv-header-only)
- **Default target:** `cvh::headers`
- **Display target:** `cvh::highgui`
- **API scope:** an intentionally aligned subset of OpenCV `core`, `imgproc`,
  and `imgcodecs`, not a drop-in replacement for every OpenCV module
- **Product boundary:** no compiled `.cpp` backend; HighGUI is an explicit
  header-only module with platform GUI link dependencies
- **Default configure:** tests and benchmarks are opt-in, so the default build
  contains only interface targets
- **Performance goal:** benchmark-gated speedups on practical preprocessing/postprocessing hot paths

## Why this project exists

Many real-world deployments need a compact computer vision foundation:

- common image processing operators
- small dependency surface
- simple integration in constrained build environments
- predictable `Mat` memory/layout behavior
- familiar OpenCV-style APIs that reduce migration and learning cost
- fast AI vision preprocessing and postprocessing on selected hot paths

## CMake Targets

The `cvh` CMake package exposes two public header-only targets:

| Target | Role | Behavior |
|---|---|---|
| `cvh::headers` | Compute target | Enables all validated OpenCV Universal Intrinsics and architecture-specific kernels by default, with runtime dispatch and scalar fallback. It does not compile `.cpp` files or enable xsimd. |
| `cvh::highgui` | Optional display target | Inherits `cvh::headers` and provides header-only windows/event handling through AppKit, Win32, or X11. It links the corresponding system GUI libraries but does not build a cvh binary. |

New optimized paths enter `cvh::headers` only after correctness and benchmark acceptance; users do not select individual ISA implementations.

## Usage

For CMake users, `cvh::headers` propagates all required include roots. For non-CMake direct include usage, add both `include/` and `include/cvh/3rdparty/opencv_intrin/` to the compiler include path.

```cpp
#include <cvh/cvh.h>
```

CMake integration:

```cmake
find_package(cvh CONFIG REQUIRED)
target_link_libraries(app PRIVATE cvh::headers)
```

Validated CPU optimizations are enabled by default. Project builds can select
a fully scalar header configuration with `-DCVH_ENABLE_OPTIMIZATION=OFF`; the
exported `cvh::headers` target then propagates
`CVH_ENABLE_OPTIMIZATION=0`. ISA availability is detected internally.

Header-only HighGUI is opt-in and is intentionally not pulled into
`cvh/cvh.h`:

```cpp
#include <cvh/highgui/highgui.h>

cvh::namedWindow("preview");
cvh::imshow("preview", image);
cvh::waitKey(0);
cvh::destroyAllWindows();
```

```cmake
target_link_libraries(app PRIVATE cvh::highgui)
```

The first HighGUI contract supports `namedWindow`, `imshow`, `waitKey`,
`destroyWindow`, and `destroyAllWindows`. `imshow` currently accepts 2D
`CV_8U` images with 1, 3, or 4 channels. Window calls should stay on the
application UI thread. Direct-include users must link AppKit/CoreGraphics on
macOS, `user32`/`gdi32` on Windows, or X11 on Linux.

xsimd is not part of the accepted runtime path. P5.3 removed the legacy `.cpp` xsimd kernels, public adapter surface, tests, dispatch mode, and vendored xsimd tree.

## Operator Status

Legend:

- **Supported:** implemented inline in headers and covered by the header-only test path.
- **Supported + fast path:** supported by `cvh::headers`, with validated OpenCV Universal Intrinsics or architecture-specific paths and scalar fallback.
- **WIP:** target API or historical implementation exists, but it is not yet accepted as part of the pure header-only contract.
- **Out of scope:** intentionally not promised by the pure header-only product.

| Module | API / operator | Status | Current header-only scope | Optimization path |
|---|---|---|---|---|
| `core` | `Mat`, `Scalar`, `Range`, `Point`, `Size`, type/channel macros | Supported | Core data model and OpenCV-style type helpers. | Same behavior as baseline. |
| `core` | `Mat::create`, `release`, `clone`, `copyTo`, `setTo`, `convertTo`, `reshape`, 2D ROI helpers | Supported | Covers common ownership, layout, continuous/non-contiguous, and conversion paths used by imgproc. | Same behavior as baseline. |
| `core` | `parallel_for_`, thread controls | Supported | Header-only serial and standard-thread runtime. | Same behavior as baseline. |
| `core` | `add`, `subtract`, `multiply`, `divide`, `compare`, `merge`, `split` | Supported | Header-only Mat-Mat/Mat-Scalar implementations with continuous and ROI coverage. | Inherits the scalar header baseline; SIMD specialization is pending. |
| `core` | `absdiff`, `bitwise_and/not/or/xor`, `inRange`, `min`, `max` | Supported | Mat/Mat and applicable Scalar overloads; raw-bit floating bitwise, optional bitwise masks, C1/C3/C4 and ROI coverage. | Inherits the scalar header baseline; benchmark rows are established for future SIMD work. |
| `core` | `scaleAdd`, `convertScaleAbs`, `convertFp16`, `sqrt`, `pow`, `exp`, `log`, `checkRange`, `patchNaNs` | Supported | FP32/FP64 math, OpenCV-compatible FP16 bit storage, range validation, and F32 NaN patching; continuous, ROI and supported in-place paths. | Inherits the scalar header baseline; representative math benchmark rows are established. |
| `core` | `norm`, `sum`, `mean`, `meanStdDev`, non-zero predicates, extrema, `reduce`, `reduceArgMin/Max`, `normalize` | Supported | C1/C3/C4 statistics, masks, ROI, N-D extrema, axis 0/1 reductions and norm/min-max normalization for the documented type subset. | Deterministic scalar header baseline; benchmark records single-thread and project-default configurations. |
| `core` | `copyTo(mask)`, channel routing, `flip/flipND`, `rotate`, `repeat`, concat, `broadcast`, `swap`, `borderInterpolate` | Supported | Byte-preserving 2D/N-D layout operations with explicit alias handling and documented trailing-dimension broadcast rules. | Scalar header baseline; copy/layout benchmark rows are established. |
| `core` | `transpose`, `transposeND` | Supported | Header-only blocked transpose with continuous, ROI, C1/C3/C4 and non-square coverage. | Inherits the scalar header baseline. |
| `core` | `gemm`, `gemm_pack_b` | Supported + fast path | FP32 activation with FP32/FP16 weights, 2D/broadcast NN and packed-B; INT8 scales remain limited to the existing NT path. | `Auto` selects accepted NEON/AVX2 kernels, then OpenCV UI, then scalar. |
| `core` | `softmax`, `silu`, `rmsnorm`, `rope` | Out of scope | Legacy declaration-only inference APIs were removed from the installed surface. | Not applicable. |
| `imgproc` | `resize` | Supported + fast path | `CV_8U` / `CV_32F`, `C1` / `C3` / `C4`, `INTER_NEAREST`, `INTER_NEAREST_EXACT`, `INTER_LINEAR`. | `CV_8UC1` exact 2x downsample with `INTER_LINEAR`. |
| `imgproc` | `cvtColor` | Supported + fast path | `CV_8U` / `CV_32F` common BGR/RGB/GRAY/BGRA/RGBA conversions; `CV_8U` YUV encode/decode families. | `CV_8UC3` `BGR2GRAY` and `RGB2GRAY`. |
| `imgproc` | `threshold` | Supported + fast path | `CV_8U` / `CV_32F` fixed thresholds; `OTSU` / `TRIANGLE` for `CV_8UC1`. | Row-parallel `CV_32F` fixed thresholds; other modes fall back. |
| `imgproc` | `LUT` | Supported + fast path | `src=CV_8U`, `lut.total()==256`, LUT channels `1` or source channel count. | Row-parallel `CV_8U` table path. |
| `imgproc` | `copyMakeBorder` | Supported + fast path | `CV_8U` / `CV_32F`, `BORDER_CONSTANT`, `REPLICATE`, `REFLECT`, `REFLECT_101`, `WRAP`. | Row-parallel `BORDER_REPLICATE`; other borders fall back. |
| `imgproc` | `filter2D` | Supported + fast path | `CV_8U` / `CV_32F` source, `CV_32FC1` kernel, `ddepth=-1/CV_8U/CV_32F`. | Header row-parallel convolution for the accepted type matrix. |
| `imgproc` | `sepFilter2D` | Supported + fast path | `CV_8U` / `CV_32F` source, `CV_32FC1` vector kernels, `ddepth=-1/CV_8U/CV_32F`. | Header row/column fast path for the accepted type matrix. |
| `imgproc` | `boxFilter`, `blur` | Supported + fast path | `CV_8U` / `CV_32F`, common border modes, `blur` as `boxFilter` semantic wrapper. | Specialized 3x3 and generic separable header paths. |
| `imgproc` | `GaussianBlur` | Supported + fast path | `CV_8U` / `CV_32F`, odd kernel sizes and sigma-based separable path. | Specialized 3x3 and generic separable header paths. |
| `imgproc` | `Sobel` | Supported + fast path | `CV_8U` / `CV_16S` / `CV_32F` input, `CV_16S` / `CV_32F` output, `ksize=3/5`, first-order derivatives. | `CV_8U`, `ksize=3/5`, first-order header path. |
| `imgproc` | kernel generators, `integral`, `Scharr`, `Laplacian`, `spatialGradient`, `sqrBoxFilter` | Supported | F32/F64 kernel generation, U8 integral sums, derivative extensions, and wide-accumulator square filtering for the documented subset. | Scalar public-header baseline; benchmark rows are established. |
| `imgproc` | `medianBlur`, `bilateralFilter`, `stackBlur`, `adaptiveThreshold`, `thresholdWithMask`, `equalizeHist`, `applyColorMap` | Supported | U8/F32 nonlinear filters and masked/statistical intensity transforms for the documented type, channel, border, and colormap subsets. | Scalar public-header baseline; no fast path is claimed. |
| `imgproc` | accumulate family, `blendLinear`, pyramid family, `cvtColorTwoPlane` | Supported | U8/F32 accumulation and blending, fixed-kernel Gaussian pyramids, and separate-plane NV12/NV21 decode. | Scalar public-header baseline; benchmark rows are established. |
| `imgproc` | `demosaicing` | WIP | The existing U8 bilinear Bayer implementation remains callable for evaluation, but is excluded from the v0.1 support commitment because demand is low and the RC benchmark is `11.70x` behind OpenCV. | Retain correctness and benchmark evidence; reconsider only in a demand-driven later batch. |
| `imgproc` | affine/perspective matrix generators, rotation matrix, affine inverse | Supported | Point2f/Point2d matrix generation plus F32/F64 2x3 inverse with explicit degenerate-input behavior. | Small fixed-size scalar solve; no fast path is needed. |
| `imgproc` | `remap`, `convertMaps`, `warpPerspective`, `getRectSubPix` | Supported | U8/F32 C1/C3/C4 nearest/bilinear geometric sampling, three map layouts, F32/F64 perspective matrices, and sub-pixel patches. | Public scalar baseline with Mode A/B benchmark rows; SIMD is pending. |
| `imgproc` | `Canny` | Supported + fast path | Image overload for `CV_8UC1`; derivative overload for `CV_16SC1`; `apertureSize=3/5`; L1/L2 gradient. | Shared header magnitude/NMS/hysteresis path. |
| `imgproc` | `erode`, `dilate`, `morphologyEx` | Supported + fast path | `CV_8U`; `MORPH_ERODE`, `DILATE`, `OPEN`, `CLOSE`, `GRADIENT`, `TOPHAT`, `BLACKHAT`, `HITMISS`; `HITMISS` limited to `CV_8UC1`. | Shared 3x3 rectangular min/max header path; generic kernels fall back. |
| `imgcodecs` | `imread` | Supported | stb-backed `CV_8U` image load with `IMREAD_UNCHANGED`, `IMREAD_GRAYSCALE`, `IMREAD_COLOR`; OpenCV-style BGR/BGRA output for color reads. | Same behavior as baseline. |
| `imgcodecs` | `imwrite` | Supported | `CV_8U` 2D `C1` / `C3` / `C4`; writes `png`, `jpg/jpeg`, `bmp`. | Same behavior as baseline. |
| `highgui` | `namedWindow`, `imshow`, `waitKey`, `destroyWindow`, `destroyAllWindows` | Supported | Optional header-only AppKit/Win32/X11 window and event-loop subset; `imshow` accepts 2D U8 C1/C3/C4. | Uses the separate `cvh::highgui` target. |

## Header-only Contract Tests

The support table above is tied to the header-only test path:

| Contract area | Test / gate |
|---|---|
| Public header/module boundary and forbidden source dependencies | `scripts/check_public_headers.sh` |
| Installed public targets and external package consumers | `scripts/check_header_only_contract.sh` |
| `cvh::headers` macro/default behavior | `cvh_header_compile_smoke`, `cvh_include_only_smoke` |
| Imgproc multi-TU ODR | `cvh_imgproc_header_odr_smoke` |
| Core/Imgproc/Imgcodecs per-header compilation | `cvh_*_headers_compile_smoke` |
| Aggregate and forwarding headers | `cvh_aggregate_headers_compile_smoke` |
| Optional HighGUI header, ODR, lifecycle, and installed consumer | `cvh_highgui_*_smoke`, `cvh_test_highgui` |
| `core` supported baseline | `cvh_test_core` |
| Multi-translation-unit core ODR/link | `cvh_core_header_odr_smoke` |
| `imgproc` supported operators | `cvh_test_imgproc` |
| `imgcodecs` supported read/write subset | `cvh_test_imgcodecs` |

## WIP / Roadmap

These are target areas, but they are not yet supported promises in the pure header-only contract:

| Area | Candidate APIs / work | Current intent |
|---|---|---|
| Core SIMD | `add/subtract/multiply/divide/transpose/GEMM` | Add UI or platform-specific paths only after the public header baseline is measured against upstream. |
| AI preprocessing | HWC-to-CHW / CHW-to-HWC, tensor packing | Add as focused preprocessing utilities once `Mat` and imgproc behavior stay stable. |
| SIMD expansion | general `resize`, broader `cvtColor`, YUV fast paths | Use direct OpenCV Universal Intrinsics style first; add platform-specific paths only when benchmark data justifies them. |
| OpenCV compatibility | more flags, depths, borders, and edge cases | Expand only with explicit behavior contracts and regression tests. |

## Performance

Performance work is benchmark-driven. `cvh::headers` enables all accepted CPU
optimization paths while retaining scalar fallback code. The only public CPU
policy switch is `CVH_ENABLE_OPTIMIZATION`.

Current SIMD platform work is limited to ARM NEON and the x86 AVX family. RVV support is a future TODO; SSE headers/macros exist only as x86 OpenCV UI/AVX prerequisites, not as a separate current optimization track.

Current accepted fast paths:

- `cvtColor`: `CV_8UC3` `BGR2GRAY` / `RGB2GRAY`
- `resize`: `CV_8UC1` exact 2x downsample with `INTER_LINEAR`
- general U8 resize and RGB/GRAY/YUV conversion families
- threshold FP32, U8 LUT, and replicate copyMakeBorder
- box/Gaussian/filter2D/sepFilter2D/Sobel; S5 additions remain scalar baselines
- Canny image/derivative and 3x3 rectangular morphology

Compare workspace:

- [Benchmark Framework](benchmark/readme.md) - internal header-only regression and OpenCV upstream compare design
- [OpenCV Compare README](benchmark/opencv_compare/README.md) - UI-forced `cvh::headers` versus upstream OpenCV
- [OpenCV UI Kernel Migration Checklist](doc/opencv-ui-kernel-migration-checklist.md)

Scripts:

- Runner: `benchmark/opencv_compare/run_compare.sh`
- CI log-only wrapper: `scripts/ci_compare_log_only.sh`

PR admins can toggle compare jobs by comment:

    /cvh-compare on
    /cvh-compare off

`/cvh-compare on` will add the compare label and trigger the dedicated `CI Compare On Demand` workflow immediately.

## Development

Header-only validation:

```bash
./scripts/ci_headers_all.sh
```

The command runs the required UI-enabled header-only gate with
`CVH_ENABLE_OPTIMIZATION=ON`. Core, Imgproc and Imgcodecs are always
header-only; scalar, OpenCV UI, direct NEON and direct AVX2 paths are inline
header implementations. Scalar-only configuration remains available for local
diagnostics but is not a hosted CI gate.

Developer tests and benchmarks are disabled in the default product configure.
Enable them explicitly with `CVH_BUILD_TESTS=ON` or
`CVH_BUILD_BENCHMARKS=ON`.

Benchmark targets:

```bash
cmake -S . -B build-bench -DCVH_BUILD_BENCHMARKS=ON
cmake --build build-bench -j --target \
  cvh_benchmark_core_mat_header \
  cvh_benchmark_imgproc_header \
  cvh_benchmark_cvtcolor_bgr2gray_header \
  cvh_benchmark_resize_bilinear_header
```

Header-only benchmark quick smoke:

```bash
./scripts/ci_benchmark_headers_quick.sh
```

## Repository Layout

- `include/` - public headers and accepted header-only implementation path
- `test/` - correctness and regression tests
- `benchmark/` - performance benchmarks, including `benchmark/opencv_compare/`
- `doc/` - design notes and execution plans

## License

This project is licensed under the [Apache License 2.0](LICENSE).
