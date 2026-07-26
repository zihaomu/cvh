# Documentation Index

This directory keeps only current project documents.

- [design.md](design.md): product direction, public targets, module boundaries,
  and SIMD strategy.
- [mat-contract-v1.md](mat-contract-v1.md): current `cvh::Mat` behavior
  contract.
- [opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md):
  upstream `core`/`imgproc` operation-family inventory and current cvh support
  gaps.
- [opencv-core-imgproc-three-phase-support-plan.md](opencv-core-imgproc-three-phase-support-plan.md):
  three-phase `core` and `imgproc` operator support scope, grouped by
  importance and dependency.
- [opencv-core-imgproc-phase1-implementation-plan.md](opencv-core-imgproc-phase1-implementation-plan.md):
  ordered implementation steps and acceptance gates for the first 79
  `core`/`imgproc` operation families.
- [benchmark-refactor-implementation-plan.md](benchmark-refactor-implementation-plan.md):
  implementation plan for the two-mode benchmark framework.
- [ci-ui-only-workflow-redesign.md](ci-ui-only-workflow-redesign.md):
  implemented design for a single required Universal Intrinsics header-only CI
  gate and an optional UI-only OpenCV comparison.
- [core-cpp-cleanup-plan.md](core-cpp-cleanup-plan.md): cleanup and
  header-only migration plan for core arithmetic, transpose, GEMM, and
  overlapping legacy `.cpp` implementations.
- [core-imgproc-test-refactor-plan.md](core-imgproc-test-refactor-plan.md):
  source-of-truth, naming, layout, fixture, and validation plan for
  reorganizing the `core` and `imgproc` tests.
- [test-design-review-findings.md](test-design-review-findings.md):
  resolved correctness, public API, SIMD dispatch, header self-containment, and
  CI findings from the 2026-07-25 test-suite review, including closure evidence.
- [opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md):
  checklist for porting OpenCV Universal Intrinsics kernel fragments.
- [opencv-universal-intrinsics-adapter-plan.md](opencv-universal-intrinsics-adapter-plan.md):
  current OpenCV UI SIMD status summary. The filename is kept for existing
  references, but the old adapter/facade execution log has been removed.

Historical rollout notes, old native-backend planning, old xsimd TODOs, and
stale compatibility-test plans have been removed from `doc/`. Current behavior
should be checked through README, this directory, and the header-only tests.
