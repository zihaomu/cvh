# CPU Optimization Configuration And Dispatch

Updated: 2026-08-03

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
| `OpenCVUIOnly` | UI-only tests and OpenCV comparison |
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

## 7. UI-Only Comparison Policy

The optional OpenCV comparison forces `OpenCVUIOnly` at runtime. It verifies
that the cvh implementation reports `opencv_ui` for cases requiring a UI path
and rejects any `neon` or `avx2` result. This keeps the comparison focused on
the requested implementation family without creating a separate target.

## 8. Source Ownership

| Area | Source |
| --- | --- |
| Public policy | `include/cvh/detail/config.h` |
| Compile capability | `include/cvh/core/detail/isa_intrinsics.hpp` |
| Runtime CPU capability | `include/cvh/core/detail/cpu_features.hpp` |
| Forced mode and dispatch tags | `include/cvh/core/detail/dispatch_control.h` |
| OpenCV UI facade | `include/cvh/core/simd/opencv_ui.h` |
| CMake propagation | `CMakeLists.txt` |

## 9. Validation

Required checks include:

- optimized full test suite;
- optimization-disabled compile/runtime smoke;
- forced UI, NEON, AVX2, and scalar correctness where applicable;
- install-tree consumer and multi-translation-unit ODR tests;
- real x86 runtime checks before claiming AVX2/FMA coverage;
- UI-only comparison tag validation.

The canonical required command is:

```bash
./scripts/ci_headers_all.sh
```
