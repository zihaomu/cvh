# Test Organization

Tests are organized by long-lived product responsibility rather than by
development phase, version number, or historical task.

## Layers

- `core/`: Mat semantics, basic operators, types, runtime behavior, internal
  dispatch, and the selected upstream compatibility subset.
- `imgproc/`: color, filtering, geometry, intensity, morphology, and other
  image-processing contracts.
- `imgcodecs/`: image read/write behavior and failure paths.
- `highgui/`: the optional header-only window API, input constraints, and
  lifecycle behavior.
- `smoke/`: independent header compilation, ODR, target configuration, and
  minimal pipelines.
- `opencv_contract/`: optional isolated differential tests against OpenCV.
- `upstream/`: extracted OpenCV test snapshots and status manifests; snapshots
  are not compiled directly.
- `support/`: shared Core/Imgproc test-state guards.
- `utils/`: cross-module helpers, currently including the shared NPY Mat
  loader in `mat_load.*`.

Public-contract tests verify public API results, boundaries, and exceptions.
Tests that force scalar/UI paths or inspect dispatch belong in the owning
module's `internal/` directory. Ported upstream cases belong in `upstream/`
and retain their original suite/case association.

## Build And Run

```bash
cmake -S . -B build-tests \
  -DCVH_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests --target cvh_test_core cvh_test_imgproc --parallel 2
ctest --test-dir build-tests --output-on-failure
```

The canonical module-level GTest targets are `cvh_test_core`,
`cvh_test_imgproc`, `cvh_test_imgcodecs`, and `cvh_test_highgui`.
`cvh_test_gemm_isa` uses `cvh::headers` to validate specialized GEMM ISA paths
without mixing architecture-conditional skips into the default Core baseline.
`test/core/sources.cmake` and `test/imgproc/sources.cmake` list sources
explicitly; configure-time checks reject missing, duplicate, or unregistered
`*_test.cpp` files.

The complete release gate uses the optimized header-only configuration:

```bash
./scripts/ci_headers_all.sh
```

This command fixes `CVH_ENABLE_OPTIMIZATION=ON`. Core and Imgproc scalar,
OpenCV UI, NEON, and AVX2 paths remain header-only. The gate builds the default
`all` target and runs the complete CTest inventory. Core/Imgproc XML results,
CTest inventory, and executed/failed/skipped counts are validated against
`test/ci/header_gate_expectations.json`.

The scalar-only configuration remains available for local diagnostics and is
not a hosted CI gate.

## Status Ownership

- `test/ci/header_gate_expectations.json` owns expected Core and Imgproc test
  counts and architecture-specific skip expectations.
- `test/upstream/opencv/core/channel_manifest.json` and
  `test/upstream/opencv/imgproc/case_manifest.json` own upstream case status,
  provenance, and local consumers.
- Current failures remain executable failures reported by CTest/GTest. The
  repository does not maintain a handwritten failure-count ledger.
- Product-boundary cases use `OUT_OF_SCOPE` with an explicit reassessment
  condition; they are not registered as permanent skips.

## Maintenance Rules

1. File names describe stable API or algorithm responsibility, not phases,
   versions, or task numbers.
2. Every test has an observable assertion; print-only calls, empty calls, and
   permanent skips are not valid tests.
3. Failure cases spanning multiple APIs are split so the owning component is
   immediately visible.
4. Public tests do not use production `detail` helpers as their oracle.
5. Every fixture has a deterministic generator, hash, oracle, and consumer.
6. Test status, consumer paths, and the owning machine-readable manifest are
   updated together.
