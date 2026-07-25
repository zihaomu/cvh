# Test Design Review Findings

> Review date: 2026-07-25
>
> Review baseline: `main@4aac34f`
>
> Status: all findings closed.
>
> Follow-up: TDR-3 and TDR-5 were refined after tracing the dispatch
> instrumentation and the complete CI call chain. The closure implementation
> and executable evidence are recorded below.

## 1. Scope and conclusion

This review covers the reorganized `test/core`, `test/imgproc`, smoke tests,
fixture manifests, CMake registration, and CI entry points.

The new domain-oriented layout is a sound long-term structure:

- Public contracts, internal dispatch tests, upstream ports, integration tests,
  and compile smoke tests have distinct ownership.
- Core and Imgproc use explicit source manifests with configuration-time checks
  for missing and duplicate `*_test.cpp` files.
- Fixtures have deterministic generators, hashes, oracle descriptions, and
  consumer records.
- Test names and file paths now identify stable API families instead of rollout
  phases.

The review originally identified five gaps, including three P1 findings that
could allow incorrect results, missing header-only definitions, or disabled
SIMD paths to pass unnoticed. All five are now closed. The default green build
is supplemented by per-header compilation, explicit dispatch evidence,
UI-enabled/UI-disabled configurations, and machine-checked test inventories.

## 2. Verification snapshot

The following closure checks were run from fresh build directories on Apple ARM
in Release mode:

| Gate | Result |
|---|---|
| Installed header-only/ODR/per-header contract | `7/7` passed |
| UI-enabled default `all` build and CTest | build passed; `17/17` passed |
| UI-enabled Core GTest | `209 executed`, `209 passed`, `0 failed`, `0 skipped` |
| UI-enabled Imgproc GTest | `186 executed`, `186 passed`, `0 failed`, `0 skipped` |
| UI-disabled default `all` build and CTest | build passed; `14/14` passed |
| UI-disabled Core GTest | `209 executed`, `196 passed`, `0 failed`, `13 skipped` |
| UI-disabled Imgproc GTest | `186 executed`, `186 passed`, `0 failed`, `0 skipped` |
| Machine-readable report/inventory check | both profiles matched checked-in expectations |
| Fixture/manifest validation | Core 71, Imgproc 5, Core upstream 20, Imgproc upstream 21; valid |

The 13 UI-disabled Core skips are the pre-existing architecture-gated Core UI
kernel tests and are pinned in the CI expectation. The Imgproc dispatch tests
do not skip in that configuration; they execute and assert `Scalar`.

## 3. Findings

| ID | Priority | Area | Status | Summary |
|---|---|---|---|---|
| TDR-1 | P1 | Numeric oracle | Closed | Comparison is per scalar element, with explicit non-finite rules and diagnostics. |
| TDR-2 | P1 | Public API | Closed | Declaration-only inference APIs were removed from the supported public surface. |
| TDR-3 | P1 | SIMD dispatch | Closed | Dispatch tags now distinguish actual UI work from scalar fallback and are asserted in both configurations. |
| TDR-4 | P2 | Header contract | Closed | Every top-level Core public header has an independent compile translation unit. |
| TDR-5 | P2 | CI | Closed | CI builds `all`, runs complete CTest inventories, and validates UI-on/UI-off reports. |

### 3.1 TDR-1: Floating-point comparator can produce false passes

Affected helpers:

- `test/core/support/core_fixture_test_utils.hpp`
- `test/core/support/gemm_test_utils.hpp`

Both helpers accumulate a global maximum absolute error and a global maximum
relative error, then accept the complete matrix when:

```text
max_abs <= abs_tolerance OR max_rel <= rel_tolerance
```

This has two correctness problems:

1. `actual - expected` can be NaN. Passing that value to `std::max` can leave the
   previous finite maximum unchanged, so a NaN mismatch may be ignored.
2. Global maximum absolute/relative errors are not equivalent to applying the
   absolute-or-relative rule per element. Different elements can satisfy
   different tolerances while the global maxima reject them, or non-finite
   relative error can incorrectly allow a matrix.

Required resolution:

- Compare each element independently.
- Handle NaN explicitly according to the tested API contract.
- Accept infinity only when both values are the same signed infinity.
- Record the first failing index and both values in the failure message.
- Add focused tests for finite absolute tolerance, finite relative tolerance,
  mixed per-element tolerances, NaN, positive/negative infinity, and signed zero.

Acceptance criteria:

- A finite expected value compared with NaN or infinity always fails.
- Opposite-sign infinities fail.
- Same-sign infinities follow the explicitly documented contract.
- Every finite element passes when either its own absolute or relative tolerance
  is satisfied.
- Existing GEMM and fixture tests continue to pass.

Closure:

- `test/core/support/floating_point_test_utils.hpp` is now the single shared
  comparator used by fixture and GEMM tests.
- It walks every scalar in continuous or non-contiguous, single- or
  multi-channel `CV_32F` matrices and applies the absolute-or-relative rule per
  scalar.
- NaN always fails; infinity passes only for the same sign; signed zero passes.
  Failures report the first scalar index, actual/expected values, both errors,
  and both tolerances.
- Five focused helper tests cover finite absolute/relative behavior, mixed
  per-element tolerances, non-contiguous multi-channel ROI traversal, NaN,
  infinities, signed zero, and diagnostic contents. The complete Core suite
  passes in both UI configurations.

### 3.2 TDR-2: Declared Core APIs are not callable header-only APIs

`include/cvh/core/basic_op.h` publicly declares:

- `softmax`
- `silu`
- `rmsnorm`
- `rope`

No definitions for these functions exist in the current public headers, and
`test/core/sources.cmake` contains no active inference test. A minimal program
that includes `cvh/core/basic_op.h` and calls `cvh::softmax` compiles but fails
to link with an undefined symbol.

This must be resolved as a product-scope decision:

| Decision | Required action |
|---|---|
| These APIs remain supported | Add pure header-only definitions and public contract tests under `test/core/operations/inference_test.cpp`. |
| These APIs are outside the product boundary | Remove the declarations from the installed public surface and document the scope change. |

Acceptance criteria:

- Every installed public function declaration has a linkable header-only
  definition, or is removed from the public headers.
- A smoke or public contract test calls every retained API family.
- Retained inference APIs cover normal output, shape/type validation, numerical
  edges, and alias behavior where applicable.

Closure:

- The product-boundary decision is that `softmax`, `silu`, `rmsnorm`, and
  `rope` are outside the current header-only product.
- Their declaration-only entries were removed from
  `include/cvh/core/basic_op.h`; no installed header now promises these
  undefined functions.
- The root README, Core/Imgproc API coverage, and refactor plan all record the
  same out-of-scope decision. Repository searches confirm there are no active
  declarations or tests that imply support.

### 3.3 TDR-3: Imgproc dispatch evidence is incomplete and ambiguous

Affected tests:

- `test/imgproc/internal/derivatives_dispatch_test.cpp`
- `test/imgproc/internal/median_blur_dispatch_test.cpp`
- `test/imgproc/internal/pyramid_dispatch_test.cpp`

These tests force `ScalarOnly`, switch to `Auto`, and compare the outputs. They
do not reset and assert `cpu::last_dispatch_tag()` after the accelerated call.
When `CVH_ENABLE_OPENCV_INTRIN=0`, all 183 Imgproc tests still pass without a
skip, including these UI-named tests. Therefore, an accidentally disabled fast
path can remain green because both sides execute the scalar implementation.

The producer-side instrumentation is also incomplete:

- The scalar paths in derivative convolution and `medianBlur` do not
  consistently write `DispatchTag::Scalar`, so a successful fallback can leave
  the tag as `Unknown` or as stale evidence from an earlier call.
- Pyramid code writes `Scalar` only when the UI backend is globally
  unavailable. An `Auto` call whose type or dimensions execute no vector block
  can therefore leave `Unknown`.
- `median_blur_detail::run_u8_sorting_network` writes `OpenCVUI` and returns
  success whenever the UI build is enabled and the kernel size is 3 or 5. It
  does so even when the row is too short for the vector loop to execute. The
  current tag can therefore claim acceleration for an entirely scalar result.
- `DispatchModeGuard` restores the mode but intentionally does not reset the
  thread-local last tag. Every observation test must reset and inspect the tag
  around one call, rather than relying on suite order.

The dispatch tag contract must be made explicit before adding assertions:

1. After a successful, instrumented public operation, the tag is never
   `Unknown`.
2. `OpenCVUI` means that at least one OpenCV UI vector block processed output
   for that public operation. A scalar border or tail does not invalidate that
   tag.
3. `Scalar` means that no OpenCV UI vector block executed. This includes
   `ScalarOnly`, compile-time UI disablement, unsupported types/layouts, and
   inputs shorter than one vector block.
4. `Unknown` is only the reset/initial state. The tag after a throwing call is
   not part of the contract.
5. The tag describes one completed public call. Tests must read it immediately;
   a later call is allowed to replace it.

Required resolution:

- At each relevant public dispatch boundary, establish `Scalar` as the default
  after argument validation and before attempting UI work. A UI kernel may
  upgrade the tag to `OpenCVUI` only after its first vector block executes.
  Scalar cleanup and tail loops must not downgrade an already observed UI use.
- Apply that rule at least to `Scharr`, `Laplacian`, `spatialGradient`,
  `medianBlur`, `pyrDown`, `pyrUp`, and `buildPyramid`, including their shared
  helpers. Avoid using a helper's "handled the output" return value as proof
  that it executed a vector block.
- Make `medianBlur` reject the UI helper before processing when the vector
  interval cannot contain one full vector, or separately track
  `handled_output` and `used_ui`. Only the latter may set `OpenCVUI`.
- Initialize pyramid dispatch once at the top-level operation. Deep horizontal
  and vertical kernels may upgrade the tag when they execute vector work, but
  must not leave short or unsupported `Auto` inputs as `Unknown`.
- Add a single test-side helper for the build capability expectation, using the
  same compile-time conditions as the implementation. Do not duplicate
  architecture macro expressions throughout individual tests.
- For every accelerated public operation, reset the tag immediately before
  each call and cover these three cases independently:
  1. `ScalarOnly` with an otherwise UI-eligible input produces `Scalar`.
  2. `Auto` with an input guaranteed to contain at least one vector block
     produces `OpenCVUI` on a supported UI build and `Scalar` in the
     UI-disabled build.
  3. `Auto` with an explicit fallback input produces `Scalar` even in a
     supported UI build.
- Choose accepted-path dimensions that exceed the largest supported vector
  width and still contain a tail. Preserve non-contiguous ROI, channel, type,
  border, and alias cases, but do not assume that a short test row exercises a
  vector path.
- In the UI-disabled configuration, execute the same tests with explicit
  `Scalar` expectations. Do not skip them merely because UI is unavailable.

Acceptance criteria:

- A successful instrumented call never leaves `Unknown`.
- Removing an accepted Imgproc UI dispatch or changing it to execute zero
  vector blocks makes the default supported-UI test fail.
- A too-short accepted type is tagged `Scalar`, while a sufficiently wide input
  with a scalar tail is tagged `OpenCVUI`.
- Unsupported shapes, layouts, types, and forced-scalar calls prove
  `Scalar` fallback.
- UI-disabled tests pass with explicit scalar expectations and no new skips.
- Every output comparison remains in place; dispatch evidence supplements
  correctness checks rather than replacing them.

Closure:

- Derivative, median-blur, and pyramid public dispatch boundaries establish
  `Scalar` after successful validation. UI kernels upgrade the tag only when a
  full vector interval is available and vector work executes.
- Short `spatialGradient` rows and median-blur intervals that cannot contain
  one vector block reject the UI helper, preventing false `OpenCVUI` evidence.
- `test/support/dispatch_mode_guard.hpp` centralizes the fixed-width UI
  capability expectation.
- The three internal suites now reset and inspect the tag around each public
  call and retain output comparison. They cover forced scalar, wide UI-eligible
  input with scalar tails, short/unsupported fallback, non-contiguous ROI,
  channel/type/border variants, aliasing, and `buildPyramid`.
- UI-enabled ARM runs record the expected UI tags; the same 186 Imgproc tests
  execute with UI disabled, record explicit `Scalar`, and add no skips.

### 3.4 TDR-4: Core public headers lack self-containment coverage

`cvh_imgproc_headers_compile_smoke` compiles each Imgproc header in an
independent translation unit. Core only has umbrella-header smoke tests, where
earlier includes can hide missing dependencies.

A direct C++17 syntax check of the Core headers found that:

```cpp
#include <cvh/core/saturate.h>
```

does not compile because `saturate.h` includes `mat.h` before declaring
`saturate_cast`, while `mat` implementation code already uses
`saturate_cast`.

The `.inl.h` files may remain implementation fragments if they are explicitly
classified as non-public. That classification does not remove the need to test
standalone public entry headers such as `saturate.h`.

Required resolution:

- Define the authoritative list of standalone Core public headers.
- Add one translation unit per public Core header.
- Add a `cvh_core_headers_compile_smoke` target and CTest entry.
- Fix `saturate.h` include ordering/dependency ownership.
- Keep internal `.inl.h` fragments out of the public-header list and document
  that status.

Acceptance criteria:

- Every supported Core public header compiles first in an otherwise empty C++17
  translation unit.
- The check runs with only installed include roots.
- Adding a new public Core header without a corresponding smoke source fails
  configuration or CI.

Closure:

- `test/smoke/core_headers/sources.cmake` is the authoritative inventory for
  the 13 top-level Core `.h` entry headers; `.inl.h`, `detail/`, and `simd/`
  remain documented implementation surfaces.
- `cvh_core_headers_compile_smoke` compiles each public header first in its own
  C++17 translation unit and is registered in CTest and the installed
  header-only contract gate.
- Configuration compares discovered headers, the explicit header manifest, and
  the expected compile source names, so adding or omitting a header/source is a
  hard failure.
- `saturate.h` now owns its direct dependencies without including `mat.h`;
  all 13 translation units and the aggregate smoke executable pass.

### 3.5 TDR-5: CI does not execute the complete header-only gate

`scripts/ci_headers_all.sh` has two layers of coverage. It first calls
`scripts/check_header_only_contract.sh`, which creates its own temporary build
and already builds and runs:

- `cvh_header_compile_smoke`
- `cvh_core_header_odr_smoke`
- `cvh_imgproc_header_odr_smoke`
- `cvh_include_only_smoke`
- `cvh_headers_fast_smoke`

That nested script also verifies installed-package consumers and the absence of
legacy exports. The main `build-ci-headers-all` build then builds and directly
runs:

- `cvh_test_core`
- `cvh_test_imgproc`
- `cvh_imgproc_headers_compile_smoke`
- `cvh_test_imgcodecs`
- `cvh_test_highgui`

The earlier statement that ODR and `cvh_headers_fast_smoke` were absent from CI
was therefore incorrect. The combined call chain still does not execute the
complete CTest inventory from the main build. It currently omits:

- `cvh_mode_lite_smoke`
- `cvh_lite_pipeline_smoke`
- `cvh_resize_dispatch_lite_smoke`
- `cvh_opencv_intrin_smoke`
- `cvh_cvtcolor_opencv_intrin_smoke`
- `cvh_resize_opencv_intrin_smoke`
- `cvh_opencv_intrin_x86_smoke` on x86

There is also no continuously enforced UI-disabled lane. The
`CVH_ENABLE_OPENCV_INTRIN` name is currently only a preprocessor default in
`include/cvh/detail/config.h`; it is not a CMake option. Passing
`-DCVH_ENABLE_OPENCV_INTRIN=OFF` to the current configure command does not by
itself disable UI code. A manual build can force the macro through compiler
flags, but that is not a clear or maintainable CI contract.

Required resolution:

- Add a first-class CMake option named `CVH_ENABLE_OPENCV_INTRIN`, defaulting to
  `ON`. Preserve the existing header-level consumer override in the default
  configuration: only propagate `CVH_ENABLE_OPENCV_INTRIN=0` through
  `cvh::headers` when the CMake option is `OFF`; do not redundantly define it as
  `1` when `ON`. This avoids macro redefinition warnings and keeps the installed
  default unchanged.
- Condition positive UI-only smoke targets on that CMake option. Add a small
  UI-disabled smoke that proves the macro is 0, the reported backend is scalar,
  and representative public calls complete through the fallback path.
- Make the default header lane perform, in order:
  1. installed-package/header-only contract validation;
  2. fixture/manifest validation;
  3. configure with UI enabled;
  4. build the default `all` target rather than a hand-maintained target list;
  5. run the full registered CTest set with `--output-on-failure`.
- Add a separate header lane configured with
  `-DCVH_ENABLE_OPENCV_INTRIN=OFF`. Build its complete `all` target and run its
  full eligible CTest set. General compile, ODR, public contract, Core, Imgproc,
  Imgcodecs, and Highgui tests must remain active; only positive UI-only smokes
  are configuration-dependent.
- Keep `scripts/check_header_only_contract.sh` as the installed-package
  contract gate. Do not list its five tests as missing from the main build, and
  do not rely on that nested temporary build as a substitute for the main
  build's full CTest run.
- On x86, always build `cvh_opencv_intrin_x86_smoke` with AVX2 so compilation is
  a mandatory gate. Run it only after a runtime AVX2 capability check. If the
  host lacks AVX2, log an explicit "compile-only" result; absence of both the
  run and that result is a CI failure.
- Have the GitHub workflow call the canonical header/native scripts directly
  instead of the deprecated `ci_lite_all.sh` and `ci_full_all.sh` forwarding
  wrappers. Express UI enabled/disabled as distinct jobs or matrix entries so
  their results cannot mask one another.
- Produce machine-readable GTest XML or JSON for Core and Imgproc and validate
  executed, failed, and skipped counts against a checked-in expectation keyed
  by configuration and architecture. A count change must fail CI until the
  expectation is deliberately reviewed and updated.
- Log the architecture, compiler ID/version, CMake build type,
  `CVH_ENABLE_OPENCV_INTRIN`, target profile
  (`cvh::headers`/`cvh::headers_fast`), runtime AVX2 decision, CTest count, and
  Core/Imgproc executed and skipped counts.

Acceptance criteria:

- The default supported-UI lane builds every target in its configuration and
  executes every registered CTest entry, including all positive UI smokes.
- The UI-disabled lane is selected through the CMake option, contains no
  compiler-flag workaround or macro redefinition warning, and proves explicit
  scalar dispatch rather than merely matching scalar output.
- On x86, the AVX2 target is always compiled. It is either executed on a
  capable host or reported as an explicit compile-only gate.
- The nested package-contract coverage is visible in logs but is not
  double-counted as a missing or main-build CTest entry.
- CI fails on a test failure, a missing registered target, an unexpected
  executed/skip-count change, or an unreported SIMD configuration.

Closure:

- `CVH_ENABLE_OPENCV_INTRIN` is a first-class CMake option, default `ON`.
  Only `OFF` propagates `CVH_ENABLE_OPENCV_INTRIN=0`; the default leaves the
  header-level consumer override untouched.
- `scripts/ci_headers_all.sh` accepts `CVH_CI_OPENCV_INTRIN=ON|OFF`, logs the
  configuration fingerprint, builds the default `all` target, runs the complete
  CTest inventory, emits Core/Imgproc XML, and validates it against
  `test/ci/header_gate_expectations.json`.
- The checked-in expectations are keyed by UI profile and architecture and pin
  exact CTest names plus executed, failed, and skipped GTest counts.
- The UI-disabled profile retains compile, ODR, Core, Imgproc, Imgcodecs, and
  Highgui coverage and adds a smoke that proves the macro/backend and a public
  operation's scalar dispatch.
- On x86 UI-enabled builds, the AVX2 smoke is always part of `all`. A
  configure-time runtime probe makes its CTest entry either execute the binary
  or report an explicit compile-only result; the report checker rejects a
  missing decision.
- GitHub Actions calls the canonical header/native scripts directly and runs
  UI-on/UI-off as non-masking matrix entries. Core and Imgproc XML files are
  uploaded as artifacts.

## 4. Recommended closure order

1. Fix TDR-1 before trusting fixture and GEMM numerical results.
2. Decide the TDR-2 public API boundary; do not leave declaration-only APIs in
   the installed header surface.
3. Close TDR-3 so accepted Imgproc fast paths are observable and regression
   resistant.
4. Add the Core compile-smoke inventory and fix `saturate.h` under TDR-4.
5. Wire the complete default and UI-disabled matrices into CI under TDR-5.

All five steps were completed in this order. The status changes above are backed
by the implementation and the executable evidence in section 2.
