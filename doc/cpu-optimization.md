# CPU Optimization Configuration And Dispatch

Updated: 2026-08-06

## 1. Public Configuration

`CVH_ENABLE_OPTIMIZATION` is the only public CPU optimization policy switch:

```cmake
option(CVH_ENABLE_OPTIMIZATION
       "Enable validated CPU optimization paths in cvh headers"
       ON)
```

The equivalent direct-include definition is:

```cpp
#define CVH_ENABLE_OPTIMIZATION 0
#include <cvh/cvh.h>
```

The value must be consistently `0` or `1` across every translation unit in a
program.

| Value | Behavior |
| --- | --- |
| `1` | Compile accepted OpenCV UI and architecture-specific paths, with scalar fallback |
| `0` | Compile scalar paths only |

Consumers do not configure individual NEON, AVX2, or OpenCV UI macros.

## 2. CMake Targets

`cvh::headers` carries the optimization policy. It does not select a fixed ISA
target and does not produce a library binary.

```cmake
target_link_libraries(app PRIVATE cvh::headers)
```

`cvh::highgui` depends on `cvh::headers`, so it inherits the same CPU policy
while additionally propagating platform GUI link dependencies.

## 3. Internal Compile Capabilities

Compile capability is detected in headers and is not consumer configuration:

| Internal result | Meaning |
| --- | --- |
| `CVH_DETAIL_HAVE_OPENCV_UI` | Vendored OpenCV Universal Intrinsics may be used |
| `CVH_DETAIL_HAVE_NEON_KERNEL` | AArch64 NEON specialized kernels compile in this translation unit |
| `CVH_DETAIL_HAVE_AVX2_KERNEL` | x86 AVX2/FMA specialized kernels compile in this translation unit |

NEON capability requires an AArch64/ARM64 NEON compilation environment. For
Clang and GCC on x86, AVX2/FMA kernels use a function target attribute, allowing
the surrounding translation unit to keep its normal baseline ISA. MSVC requires
an AVX2-enabled compilation unit.

## 4. Runtime Capabilities

- AArch64 Advanced SIMD is treated as available when the specialized NEON
  kernel compiled.
- x86 AVX2/FMA checks CPU support and operating-system vector-state support
  before execution.
- Failure to satisfy a specialized runtime capability leaves the operator on
  OpenCV UI or scalar fallback.

Runtime detection does not change the public ABI because all implementations
are inline header code.

## 5. Dispatch Modes

The internal dispatch control supports:

| Mode | Intended use |
| --- | --- |
| `Auto` | Normal product execution |
| `ScalarOnly` | Correctness and fallback diagnostics |
| `OpenCVUIOnly` | UI-only tests and comparison diagnostics |
| `NeonOnly` | Forced NEON correctness tests |
| `Avx2Only` | Forced AVX2/FMA correctness tests |

Forced modes are test and benchmark controls. They are not build products and
must not become public CMake targets.

The observable dispatch tags are:

```text
scalar
opencv_ui
neon
avx2
```

## 6. Auto Selection

Each operator owns its shape, type, alignment, and workload gates. The general
selection order is:

```text
accepted specialized ISA kernel
    -> accepted OpenCV Universal Intrinsics kernel
    -> scalar implementation
```

An operator without a specialized ISA kernel can still use OpenCV UI. An
operator without either optimized path remains a valid scalar implementation.
Optimization availability must never change public API availability.

## 7. Retained Direct-ISA Routes

The following table is the current direct-ISA inventory. It describes product
`Auto`; an ineligible case continues through the operator's existing OpenCV UI
or scalar implementation.

| Public operator | Direct-kernel eligibility | Apple/AArch64 `Auto` | x86 `Auto` |
| --- | --- | --- | --- |
| `gemm` | FP32 NN/NT, `m>=2`, `n>=8`, `k>=8`, work `>=32768`; AVX2 also requires `n>=16` | direct NEON packed-B kernel | direct AVX2/FMA when available, otherwise scalar |
| `cvtColor` packed U8 | supported RGB/BGR/BGRA/RGBA shuffle, alpha/drop-alpha, BGRA/RGBA-to-gray, and gray expansion; width `>=16`, pixels `>=256` | direct NEON interior with scalar tail; existing BGR/RGB-to-gray UI route is deliberately retained | existing UI/scalar route |
| `cvtColor` YUV U8 | YUV444 interleaved, YUV420 planar/semi-planar, and packed YUV422 supported codes; pixels `>=256`, plus format-specific even dimensions | direct NEON luma/chroma conversion with scalar tail | existing UI/scalar route |
| `resize` | `INTER_LINEAR`, U8C3, destination width `>=8`, destination pixels `>=256` | direct NEON; 0.5x uses a specialized average kernel, other ratios use the NEON table-gather kernel | existing UI/scalar route |
| `Sobel` / `Scharr` | 3x3 U8 C1/C3/C4 to S16/F32, first derivative in one axis, scale 1, delta 0, replicate/reflect101 border, workload `>=256` | shared direct NEON three-row interior; scalar border and overlapping NEON tail | existing UI/scalar route |
| `spatialGradient` | U8C1, replicate/reflect101 border, pixels `>=256` | shared direct NEON three-row kernel emits dx/dy together | existing UI/scalar route |

`NeonOnly` may force eligible AArch64 kernels for correctness tests.
`OpenCVUIOnly` and `ScalarOnly` never enter a direct NEON kernel. Benchmark
rows record the actual route as
`algorithm_path -> dispatch_path -> isa_observed -> kernel_route`; the current
full product report observed 30 direct-NEON rows (10 GEMM, 10 color, 2 resize,
6 Sobel, 1 Scharr, and 1 spatial-gradient row).

## 8. OpenCV Comparison Policy

The optional OpenCV comparison defaults to product `Auto` dispatch so eligible
specialized NEON/AVX2 kernels are measured before OpenCV UI and scalar
fallbacks. Diagnostic `cvh_ui` runs force `OpenCVUIOnly` and reject `neon` or
`avx2`; `cvh_scalar` runs reject every accelerated dispatch. All modes use the
same public target and report the actual per-case dispatch tag.

## 9. Source Ownership

| Area | Source |
| --- | --- |
| Public policy | `include/cvh/detail/config.h` |
| Compile capability | `include/cvh/core/detail/isa_intrinsics.hpp` |
| Runtime CPU capability | `include/cvh/core/detail/cpu_features.hpp` |
| Forced mode and dispatch tags | `include/cvh/core/detail/dispatch_control.h` |
| OpenCV UI facade | `include/cvh/core/simd/opencv_ui.h` |
| CMake propagation | `CMakeLists.txt` |

## 10. Validation

Required checks include:

- optimized full test suite;
- optimization-disabled compile/runtime smoke;
- forced UI, NEON, AVX2, and scalar correctness where applicable;
- install-tree consumer and multi-translation-unit ODR tests;
- real x86 runtime checks before claiming AVX2/FMA coverage;
- product-auto comparison ISA observation and forced UI/scalar tag validation.

The canonical required command is:

```bash
./scripts/ci_headers_all.sh
```
