# cvh Documentation

`cvh` is an independent header-only C++ computer vision library. OpenCV is an
API, behavior, and performance reference; it is not the product identity or a
binary dependency of the normal cvh package.

## Current Documents

- [design.md](design.md): product direction, header-only contract, public
  targets, module boundaries, and optimization principles.
- [ci.md](ci.md): required correctness CI, x86 validation, optional product-auto
  OpenCV comparison, artifacts, and local reproduction.
- [cpu-optimization.md](cpu-optimization.md): public optimization policy,
  compile/runtime capability detection, and dispatch modes.
- [gemm-optimization.md](gemm-optimization.md): current GEMM API, numeric
  contract, ISA/UI/scalar selection, packed weights, and real thread model.
- [mat-contract-v1.md](mat-contract-v1.md): stable `cvh::Mat` behavior contract.
- [opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md):
  pinned upstream operation-family inventory and current callable cvh subset.
- [opencv-core-imgproc-three-phase-support-plan.md](opencv-core-imgproc-three-phase-support-plan.md):
  remaining Core and Imgproc expansion roadmap.
- [opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md):
  current rules for OpenCV Universal Intrinsics and specialized ISA kernels.
- [cvh-phase2-p0-operator-acceleration-plan.md](cvh-phase2-p0-operator-acceleration-plan.md):
  implementation batches, correctness gates, upstream comparisons, and live
  status for the second-stage acceleration of the 17 Phase 2-P0 operation
  families.
- [cvh-v0.1-imgproc-performance-floor-acceleration-plan.md](cvh-v0.1-imgproc-performance-floor-acceleration-plan.md):
  third-stage performance closure, quantitative floors, SIMD/UI audit, and
  live status for the existing v0.1 Imgproc hotspots.
- [cvh-v0.1-neon-hot-kernel-acceleration-plan.md](cvh-v0.1-neon-hot-kernel-acceleration-plan.md):
  narrowly scoped direct-NEON acceleration, dispatch observability, correctness
  gates, and performance floors for packed/YUV color conversion, U8C3 bilinear
  resize, and shared 3x3 derivative kernels.
- [cvh-v0.1-release-readiness-and-next-stage-plan.md](cvh-v0.1-release-readiness-and-next-stage-plan.md):
  P7.2 onboarding, P7.3 platforms, P7.4 performance baseline, and `0.1.0`
  release gates.
- [cvh-v0.1-release-closure-plan.md](cvh-v0.1-release-closure-plan.md):
  current code/document cleanup for deprecated compatibility shims, unused
  diagnostics, phase documents, and the Phase 2 benchmark consolidation.

## Ownership Rules

| Fact | Owner |
| --- | --- |
| Public usage and current report entry | top-level `README.md` |
| Product/module/target boundary | `design.md` |
| CPU configuration and dispatch | `cpu-optimization.md` |
| GEMM implementation facts | `gemm-optimization.md` |
| Supported operation families | API coverage |
| Current code/document closure work | v0.1 release closure plan |
| Phase 2-P0 acceleration work | Phase 2-P0 operator acceleration plan |
| v0.1 Imgproc performance-floor work | v0.1 Imgproc performance-floor acceleration plan |
| v0.1 direct-NEON hot-kernel work | v0.1 NEON hot-kernel acceleration plan |
| Unfinished release work | release-readiness plan |
| Test status | test manifests and `test/failing-tests.md` |
| Performance numbers | dated benchmark report plus CSV/metadata |

Git history is the default archive for completed rollout logs. A restored
historical record must be explicitly labeled and must not become the owner of
current product facts.

## Maintenance Rules

- Documents describe current behavior or unfinished work. Git history retains
  completed implementation and acceptance records.
- Local paths use portable placeholders.
- Dated benchmark reports are immutable artifacts; corrections require a new
  dated report.
- New benchmark reports are written in English.
- Public claims must trace to code, tests, or report metadata.
- Add, remove, or rename a document and this index in the same change.
- Run `scripts/check_docs.sh` before committing documentation changes.
