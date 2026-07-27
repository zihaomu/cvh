# x86 Correctness Hardening Plan

Status: implemented
Date: 2026-07-26
Validation baseline: `main@32bf37f9fc2f85b2c7cd84d7d4969d784ab5cb35`
Reference host: `local_x86`, AMD Ryzen 5 5600X, x86_64, AVX2/FMA,
GCC 11.4.0, CMake 3.22.1, WSL2

## 1. Decision

The default x86 header-only gate is green, but x86 correctness is not complete
until all of the following are true:

1. the confirmed signed-shift undefined behavior in Sobel and JPEG writing is
   removed;
2. the complete Core and Imgproc suites pass when compiled for
   `x86-64-v3`, rather than only exercising the generic x86-64/SSE2 baseline;
3. dispatch tests derive accepted and fallback dimensions from the compiled
   Universal Intrinsics vector width;
4. floating-point geometry tests accept valid FMA rounding without hiding
   meaningful numerical regressions; and
5. sanitizer and x86-v3 validation are reproducible repository gates with
   bounded resource use.

The work is correctness hardening. It must not broaden the public API, change
documented operator semantics, introduce a new SIMD abstraction, or turn scalar
fallback into an error.

## 2. Validation Snapshot

The following checks were run on the reference host.

| Configuration | Result |
|---|---|
| Canonical `scripts/ci_headers_all.sh` | header-only contract passed; CTest `18/18` passed |
| Canonical Core GTest | `209/209` passed; `0` skipped |
| Canonical Imgproc GTest | `186/186` passed; `0` skipped |
| x86 AVX2 smoke | compiled with `-mavx2` and passed at runtime |
| Full `-march=x86-64-v3` build | CTest `16/18` passed |
| Full x86-v3 Core GTest | `204/209` passed; five dispatch-expectation tests failed |
| Full x86-v3 Imgproc GTest | `185/186` passed; one affine tolerance test failed |
| ASan/UBSan Core | `209/209` passed; no sanitizer diagnostics |
| ASan/UBSan Imgproc | `186/186` assertions passed; two signed-shift UBSan diagnostics |
| ASan/UBSan Imgcodecs | `7` passed, one optional HDR fixture skipped; one signed-shift UBSan diagnostic |
| Address/Leak Sanitizer | no out-of-bounds, use-after-free, or leak diagnostic |

The x86-v3 Core failures did not contain byte or numeric mismatches. They were
caused by tests expecting `OpenCVUI` for inputs that are shorter than one AVX2
vector block. Scalar fallback produced the expected result.

The x86-v3 affine failure was deterministic. The observed absolute error was
`1.1920928955078125e-7` against a fixed `1e-7` threshold for control points
near `1e8`. Compiling the test helper with FMA contraction disabled produced
the exact expected value, confirming that the failure is a test-oracle
portability issue rather than a different affine matrix.

## 3. Findings and Required Resolutions

| ID | Priority | Area | Summary |
|---|---|---|---|
| X86-1 | P0 | Sobel/Canny | Signed negative derivative values are left-shifted in the U8 Sobel fast path. |
| X86-2 | P0 | Imgcodecs/JPEG | The vendored stb JPEG bit buffer is signed and is left-shifted after its sign bit is set. |
| X86-3 | P1 | Core dispatch tests | Fixed row widths assume 128-bit lanes and falsely fail under AVX2. |
| X86-4 | P1 | Geometry tests | A fixed absolute tolerance is too narrow for legal FMA rounding at large coordinate magnitudes. |
| X86-5 | P1 | CI | Only the dedicated smoke is forced to AVX2; the complete hosted suite is compiled for generic x86-64. |
| X86-6 | P2 | Test hygiene | Internal test includes redefine `CV_Assert`, and the NPY test loader passes `NULL` as an integer error code. |

### 3.1 X86-1: signed shift in Sobel fast path

Affected code:

- `include/cvh/imgproc/detail/sobel_impl.hpp`
- the `CV_32F` output branch around line 266;
- the `CV_16S` output branch around line 310.

Both branches contain the equivalent of:

```cpp
(p12 - p10) << 1
```

`p12 - p10` can be negative. Left-shifting a negative signed integer is
undefined behavior in C++, even though current x86 compilers commonly produce
the same bits as multiplication by two.

Required resolution:

- Replace signed shifts of derivative differences with defined arithmetic such
  as multiplication by two.
- Keep the existing accumulator range analysis explicit. U8 3x3 Sobel remains
  safely representable in `int`.
- Audit the same file and shared derivative helpers for any other shift whose
  operand can be negative.
- Add a focused regression input that produces both positive and negative
  horizontal and vertical gradients.
- Exercise both `CV_16S` and `CV_32F` destinations, direct `Sobel`, and the
  image overload of `Canny` that reaches Sobel internally.

Acceptance:

- No UBSan diagnostic is emitted from `sobel_impl.hpp`.
- Existing exact Sobel and Canny reference comparisons remain unchanged.
- Dispatch evidence still reports the same UI/scalar decision as before.

### 3.2 X86-2: signed JPEG bit-buffer shift

Affected code:

- `include/cvh/3rdparty/std/stb_image_write.h`
- `stbiw__jpg_writeBits`, `stbiw__jpg_processDU`, and the owning `bitBuf`
  declaration.

The JPEG writer stores a bit buffer in `int` and repeatedly executes:

```cpp
bitBuf <<= 8;
```

Once the sign bit is set, the buffer is negative and the shift is undefined.
The vendored header identifies itself as `stb_image_write` 1.16. Upstream
master still uses the signed buffer as of the plan date, so merely refreshing
to the current upstream file does not close this finding.

Required resolution:

- Use an explicitly unsigned 32-bit bit buffer throughout the complete JPEG
  call chain, including pointer parameters and the owning local variable.
- Preserve the current byte extraction and emitted JPEG byte stream.
- Keep the patch isolated and annotate it as a cvh sanitizer hardening delta
  from upstream 1.16.
- Record the upstream file/version or commit used for future vendor refreshes.
- Extend the JPG roundtrip test with patterns that set the high bit in the bit
  buffer and run repeated writes at more than one quality setting.

Acceptance:

- No UBSan diagnostic is emitted from `stb_image_write.h`.
- Existing JPG dimensions and lossy-difference contracts continue to pass.
- A deterministic fixture produces identical encoded bytes before and after
  the type-only hardening where the pre-fix build has defined behavior.
- PNG and BMP behavior is unchanged.

### 3.3 X86-3: lane-width-sensitive dispatch tests

Affected tests include:

- `test/core/support/array_internal_test_utils.hpp`;
- `test/core/internal/array_dispatch_test.cpp`;
- `test/core/internal/reduction_norm_dispatch_test.cpp`.

Several accepted-path cases use a row width of `19`. This is larger than the
16 U8 lanes in the generic SSE2 build but smaller than the 32 U8 lanes selected
by AVX2. Masked norm uses selected runs of 29 and 31 pixels, which have the same
problem. The implementation correctly falls back to scalar when no complete
vector block exists, but the tests unconditionally expect `OpenCVUI`.

Required resolution:

- Derive accepted-path dimensions from `cv::VTraits<Vector>::vlanes()` or a
  shared test helper that represents the widest compiled fixed-width vector.
- Make accepted inputs contain at least two complete vector blocks plus a
  non-zero scalar tail.
- Make fallback inputs explicitly shorter than one vector block.
- For masked operations, derive the longest selected run from the active
  element type and ensure accepted/fallback masks test both sides of the lane
  boundary.
- Preserve non-contiguous ROI, channel phase, alias, raw floating-point bit,
  saturation, and tail coverage.
- Compute expected dispatch from whether the specific input can execute a
  vector block; do not use global UI availability as sufficient evidence.

Acceptance:

- The same source tests pass with generic x86-64, `-mavx2`,
  `-march=x86-64-v3`, and ARM NEON builds.
- Accepted-path calls report `OpenCVUI` only after real vector work.
- Short or fragmented masked calls report `Scalar`.
- Scalar and UI results continue to match byte-for-byte or within the existing
  numeric contract.

### 3.4 X86-4: FMA-safe affine oracle

Affected test:

- `test/imgproc/geometry/transform_matrix_test.cpp`.

The current large-coordinate affine case uses one fixed absolute tolerance.
The test helper's multiply-add expression can be contracted into FMA on
`x86-64-v3`, changing the final rounding while using the same matrix.

Required resolution:

- Replace the fixed absolute-only threshold with the project's shared
  absolute-or-relative comparison rule, or an equivalent scale-aware helper
  with useful diagnostics.
- Base the tolerance on the mapped output and the conditioning of the selected
  control points, not directly on the `1e8` source magnitude.
- Retain a strict enough limit to catch a wrong coefficient, sign, point order,
  or degenerate solve.
- Add an explicit regression note that the test must pass with FMA contraction
  both enabled and disabled.
- Do not disable FMA globally or compile this test with a special floating-point
  mode merely to preserve the old rounding.

Acceptance:

- The case passes under generic x86-64, x86-64-v3/FMA, and ARM.
- Deliberately perturbing one affine coefficient by a meaningful amount causes
  the oracle to fail with the coefficient/point index in the diagnostic.

### 3.5 X86-5: reproducible x86 gates

The canonical UI gate currently validates the generic compiler target. CMake
adds `-mavx2` only to `cvh_opencv_intrin_x86_smoke`, which proves that the
vendored intrinsics types compile and run but does not compile the complete
Core and Imgproc suites with AVX2/FMA enabled.

Required resolution:

- Add a repository-owned x86 correctness entry point, for example
  `scripts/ci_x86_correctness.sh`.
- Run a Release `x86-64-v3` build with Universal Intrinsics enabled and the
  native backend disabled.
- Run the complete CTest inventory and machine-readable Core/Imgproc reports.
- Add a Debug or RelWithDebInfo ASan/UBSan build for Core, Imgproc, Imgcodecs,
  and the x86 AVX2 smoke.
- Use a default parallelism of `2` for sanitizer compilation. The reference
  8 GiB WSL2 host became unreachable while compiling sanitizer-instrumented
  Imgproc with parallelism `6`; the same build completed with parallelism `2`.
- Preserve build and test reports on failure.
- Initially land the gate as opt-in or scheduled if hosted x86 capacity is
  uncertain, then promote it to required after stable runtime and artifact
  behavior are observed.

The gate must not:

- suppress UBSan by adding the affected files to a sanitizer ignore list;
- treat a recoverable UBSan diagnostic as a pass;
- disable FMA to make the geometry test green;
- force scalar mode for tests intended to prove UI dispatch; or
- replace the existing portable header-only gate.

### 3.6 X86-6: warning cleanup

Observed warnings:

- `CV_Assert` is defined by the vendored OpenCV intrinsics `cvdef.h` and then
  redefined by `cvh/core/system.h` in internal dispatch test translation units.
- `test/utils/mat_load.cpp` passes `NULL` to an integer error-code parameter.

Required resolution:

- Establish one intentional `CV_Assert` ownership rule and make the include
  order/macro handoff explicit without weakening cvh exception behavior.
- Replace `NULL` with the intended `cvh::Error` code in the NPY test loader.
- Add a focused warning build after the correctness fixes. Do not promote the
  complete project to `-Werror` until vendored-header warnings are categorized.

## 4. Implementation Order

### Phase 0: preserve reproducers

1. Record the reference host/compiler fingerprint in the x86 gate output.
2. Add or refine focused tests so every confirmed finding reproduces before
   the implementation change.
3. Keep sanitizer recovery disabled in the acceptance command.

Exit condition: X86-1 and X86-2 fail for the recorded UBSan reason, and X86-3
and X86-4 fail only in the x86-v3 configuration for the recorded reason.

### Phase 1: eliminate undefined behavior

1. Fix X86-1 in the project-owned Sobel fast path.
2. Fix X86-2 with a minimal documented vendored-header delta.
3. Run focused sanitizer tests after each fix.
4. Run the complete Core, Imgproc, and Imgcodecs sanitizer set.

Exit condition: zero ASan, LeakSanitizer, and UBSan diagnostics.

### Phase 2: make tests architecture-width aware

1. Introduce shared lane-aware test helpers.
2. Migrate arithmetic, raw bitwise, `inRange`, and norm dispatch cases.
3. Fix the affine oracle for valid FMA rounding.
4. Run the generic and x86-v3 suites back-to-back.

Exit condition: both configurations have zero failed Core/Imgproc tests and
the expected zero skips.

### Phase 3: productize the x86 gate

1. Add the repository x86 script and report checks.
2. Add the hosted/scheduled workflow with bounded parallelism.
3. Document the gate in the root README and CI design document.
4. Promote the gate only after repeated stable runs.

Exit condition: a clean checkout can reproduce every acceptance command
without manual compiler flags or local-only scripts.

### Phase 4: close warning debt and evidence

1. Resolve X86-6.
2. Run a warning inventory on GCC and Clang x86.
3. Update this document with final commit IDs, report paths, and closure
   results.

## 5. Verification Matrix

### 5.1 Portable canonical gate

```bash
CVH_CI_PARALLEL=2 ./scripts/ci_headers_all.sh
```

Expected:

- installed header-only contract passes;
- CTest passes all registered tests;
- Core executes 209 tests with zero failures and zero skips;
- Imgproc executes 186 tests with zero failures and zero skips.

### 5.2 Full x86-64-v3 Release gate

```bash
cmake -S . -B build-x86-v3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  -DCVH_ENABLE_OPENCV_INTRIN=ON
cmake --build build-x86-v3 --parallel 2
ctest --test-dir build-x86-v3 --output-on-failure
```

Expected:

- all CTest entries pass;
- x86 AVX2 smoke runs rather than becoming compile-only;
- Core and Imgproc have zero failures and zero skips;
- accepted dispatch cases prove actual UI execution.

### 5.3 Sanitizer gate

```bash
cmake -S . -B build-x86-sanitizers \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_CXX_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCVH_BUILD_NATIVE_BACKEND=OFF \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  -DCVH_ENABLE_OPENCV_INTRIN=ON
cmake --build build-x86-sanitizers --parallel 2 --target \
  cvh_test_core \
  cvh_test_imgproc \
  cvh_test_imgcodecs \
  cvh_opencv_intrin_x86_smoke
```

Run with:

```bash
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
./build-x86-sanitizers/cvh_opencv_intrin_x86_smoke
./build-x86-sanitizers/cvh_test_core --gtest_brief=1
./build-x86-sanitizers/cvh_test_imgproc --gtest_brief=1
./build-x86-sanitizers/cvh_test_imgcodecs --gtest_brief=1
```

Expected:

- all commands return zero;
- output contains no `runtime error` or sanitizer report;
- the optional HDR fixture may remain skipped only while its absence is
  explicitly reported and all mandatory format tests execute.

### 5.4 Focused floating-point mode check

The affine regression must additionally be compiled and run once with normal
x86-v3 FMA contraction and once with `-ffp-contract=off`. Both executions must
pass the same scale-aware oracle.

## 6. Final Acceptance Criteria

The x86 hardening work is complete only when:

1. X86-1 through X86-5 are closed with executable evidence.
2. The canonical portable gate remains green.
3. The full x86-v3 Release suite is green.
4. The sanitizer gate is green with halt-on-first-error behavior.
5. No test is skipped or filtered to hide an x86-only failure.
6. Scalar fallback remains correct and observable for sub-vector inputs.
7. AVX2 accepted paths execute at least one real vector block and match the
   scalar oracle.
8. FMA remains enabled in the x86-v3 product configuration.
9. The vendored stb delta is documented so a future vendor refresh cannot
   silently reintroduce the signed buffer.
10. Final reports record the compiler, CPU flags, commit, configuration, test
    counts, failures, and skips.

## 7. Closure Record

When implementation is complete, replace `Status: proposed` with
`Status: implemented` and append:

- implementation commit(s);
- x86 host/compiler fingerprint;
- canonical gate result;
- x86-v3 result;
- sanitizer result;
- any accepted optional skip;
- hosted workflow run link or artifact identifier; and
- confirmation that the remote and local source trees are clean.

### Implementation result (2026-07-27)

- Branch: `codex/x86-correctness-hardening`.
- Primary implementation commit:
  `8e409baea2aa0a9e9bcad5036fbba26fc51f962a`.
- Host/compiler: `local_x86`, AMD Ryzen 5 5600X, WSL2 x86_64,
  GCC 11.4.0, CMake 3.22.1; AVX2 and FMA available.
- Canonical gate: `scripts/ci_headers_all.sh` passed the installed-header
  contract, CTest `18/18`, Core `209/209`, and Imgproc `187/187`, with zero
  Core/Imgproc failures or skips.
- x86-v3 Release: CTest `18/18`, Core `209/209`, Imgproc `187/187`, and the
  runtime AVX2 smoke passed. Machine-readable reports are under
  `build-x86-v3/test-reports/`.
- Sanitizers: with ASan/UBSan halt-on-first-error enabled, Core `209/209` and
  Imgproc `187/187` passed; Imgcodecs ran nine tests, with eight passing and
  only the explicitly optional missing HDR fixture skipped. The runtime AVX2
  smoke passed and no sanitizer diagnostic was emitted.
- Floating-point comparison: the affine regression passed both the normal
  x86-v3 FMA-enabled build and an x86-v3 `-ffp-contract=off` comparison build.
- Reproducible gate: `scripts/ci_x86_correctness.sh` and the manual/scheduled
  `.github/workflows/ci-x86-correctness.yml` workflow were added. No hosted
  workflow run exists yet; local reports are the current artifact identifiers.
- Resource behavior: all heavy builds used parallelism `2`; `local_x86`
  remained reachable throughout the sanitizer and canonical rebuilds.
- Tree state: after the closure-record commit, the remote branch and local
  transfer mirror are clean; build directories and reports are ignored
  artifacts.
