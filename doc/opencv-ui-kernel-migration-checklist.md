# OpenCV Universal Intrinsics Kernel Checklist

Updated: 2026-08-03

## 1. Purpose

This checklist governs new or modified OpenCV Universal Intrinsics (UI)
kernels in `cvh`. OpenCV UI is an internal SIMD dialect, not a public API and
not a separate CMake product target.

## 2. Scope

Current accepted scope:

- AArch64 NEON through OpenCV UI;
- x86 SSE/AVX-family support through OpenCV UI;
- direct NEON or AVX2 intrinsics only for operator-specific kernels that have
  correctness and benchmark evidence.

Deferred scope:

- RVV and scalable-vector support;
- OpenCV runtime dispatcher tables;
- IPP, OpenCL, HAL plugins, or compiled module backends;
- a project-owned SIMD facade duplicating OpenCV UI.

## 3. Source Provenance

Before adapting an upstream kernel, record:

- upstream repository and commit;
- source file and function/block;
- any material algorithm change;
- applicable OpenCV license notice.

Prefer compact `*.simd.hpp` expressions already written with `cv::v_*`,
`cv::VTraits`, `CV_SIMD`, or `vx_*`. Port the kernel expression, not OpenCV's
module dispatch or build framework.

## 4. Required Implementation Shape

Every accepted UI path must have:

1. an existing scalar correctness fallback;
2. a narrow support predicate covering type, channels, layout, flags, and
   shape restrictions;
3. compile guards based on `CVH_DETAIL_HAVE_OPENCV_UI` and the required
   `CV_SIMD*` capabilities;
4. runtime permission through `cvh::cpu::opencv_ui_allowed()` when dispatch is
   controllable;
5. explicit lane-tail behavior;
6. fallback for unsupported or too-small inputs;
7. output layout and ROI behavior matching the scalar contract.

Implementation headers include:

```cpp
#include <cvh/core/simd/opencv_ui.h>
```

Do not include long vendored OpenCV header paths from operator code.

## 5. OpenCV UI Style

- Use `cv::v_*` and `cv::vx_*` operations directly.
- Use `cv::VTraits<T>::vlanes()` for lane-dependent loops.
- Prefer existing OpenCV helpers for load/store, interleave, expand, pack,
  reinterpret, and rounded narrowing.
- Do not expose `cv::v_*` types from public cvh functions or data structures.
- Do not introduce a second project-owned vector type hierarchy.
- Keep mutable dispatch telemetry ODR-safe across translation units.

## 6. Specialized ISA Boundary

A direct NEON or AVX2 kernel is a separate implementation candidate, not an
OpenCV UI alias. It must:

- use internal compile and runtime capability checks;
- have a workload gate that avoids small-input regressions;
- remain reachable only through the normal `cvh::headers` product target;
- preserve UI and scalar fallbacks;
- report `neon` or `avx2`, never `opencv_ui`;
- stay excluded from forced UI diagnostics while remaining visible in the
  default product-auto OpenCV comparison.

## 7. Correctness Gate

Required coverage:

- accepted fast-path case;
- unsupported fallback case;
- scalar versus UI equivalence;
- integer byte-exact output or the established floating-point tolerance;
- ROI/non-contiguous rows when supported publicly;
- widths below, equal to, and above one vector width;
- non-multiple lane tails;
- in-place behavior where the public API permits it;
- multi-translation-unit compile/link safety.

Architecture-specific claims require runtime execution on the target
architecture. Cross-compilation is compile evidence only.

## 8. Benchmark Gate

Measure on identical input and sampling settings:

- public API dispatch;
- scalar fallback;
- direct UI implementation when diagnostic access exists;
- specialized ISA candidate when proposed.

Record shape, depth, channels, layout, lane/tail information, allocation mode,
threads, median time, throughput, checksum, and actual dispatch tag.

Do not accept a specialized kernel merely because it compiles. It must produce
a stable useful win on the intended hardware without regressing important
fallback cases.

## 9. Required Commands

For a normal UI kernel change:

```bash
cmake -S . -B build-ui-kernel \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=ON \
  -DCVH_ENABLE_OPTIMIZATION=ON
cmake --build build-ui-kernel --parallel 2
ctest --test-dir build-ui-kernel --output-on-failure
./scripts/check_header_only_contract.sh
./scripts/ci_headers_all.sh
python3 scripts/sync_opencv_intrin.py --check
git diff --check
```

Add the relevant operator benchmark and, for x86-specific changes, run the
real x86 correctness workflow on an x86-64 host.

For scalar fallback verification:

```bash
cmake -S . -B build-scalar \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  -DCVH_ENABLE_OPTIMIZATION=OFF
cmake --build build-scalar --parallel 2
ctest --test-dir build-scalar --output-on-failure
```

## 10. Review Checklist

- [ ] Public API support is unchanged unless explicitly documented.
- [ ] Scalar fallback remains reachable and tested.
- [ ] UI capability and runtime permission are checked separately.
- [ ] Tails and non-contiguous rows are covered.
- [ ] Dispatch tags identify the implementation actually executed.
- [ ] Specialized ISA code remains internal and header-only.
- [ ] Product-auto comparison records specialized ISA kernels; forced UI
      diagnostics cannot execute them.
- [ ] Upstream provenance and license obligations are recorded.
- [ ] Benchmark evidence is attached for performance claims.
- [ ] Current design, API coverage, and dated reports are updated only when
      their owned facts changed.
