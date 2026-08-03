# cvh Documentation

`cvh` is an independent header-only C++ computer vision library. OpenCV is an
API, behavior, and performance reference; it is not the product identity or a
binary dependency of the normal cvh package.

## Current Documents

- [design.md](design.md): product direction, header-only contract, public
  targets, module boundaries, and optimization principles.
- [ci.md](ci.md): required correctness CI, x86 validation, optional UI-only
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
- [opencv-core-imgproc-phase2-p0-implementation-plan.md](opencv-core-imgproc-phase2-p0-implementation-plan.md):
  selected 17-family P2-P0 scope, support matrix, implementation order, and
  acceptance gates.
- [opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md):
  current rules for OpenCV Universal Intrinsics and specialized ISA kernels.
- [cvh-v0.1-release-readiness-and-next-stage-plan.md](cvh-v0.1-release-readiness-and-next-stage-plan.md):
  P7.2 onboarding, P7.3 platforms, P7.4 performance baseline, and `0.1.0`
  release gates.
- [cvh-v0.1-release-closure-plan.md](cvh-v0.1-release-closure-plan.md):
  current code/document cleanup for deprecated compatibility shims, unused
  diagnostics, phase documents, and the Phase 2 benchmark consolidation.
## Restored Historical Records

- [opencv-core-imgproc-phase1-implementation-plan.md](opencv-core-imgproc-phase1-implementation-plan.md):
  restored Phase 1 operator implementation and acceptance record. Use API
  coverage and the three-phase plan for current support and remaining gaps.

## Ownership Rules

| Fact | Owner |
| --- | --- |
| Public usage and current report entry | top-level `README.md` |
| Product/module/target boundary | `design.md` |
| CPU configuration and dispatch | `cpu-optimization.md` |
| GEMM implementation facts | `gemm-optimization.md` |
| Supported operation families | API coverage |
| Current code/document closure work | v0.1 release closure plan |
| Unfinished release work | release-readiness plan |
| Test status | test manifests and `test/failing-tests.md` |
| Performance numbers | dated benchmark report plus CSV/metadata |

Git history is the default archive for completed rollout logs. A restored
historical record must be explicitly labeled and must not become the owner of
current product facts.

## Maintenance Rules

- Documents describe current behavior or unfinished work. Explicitly restored
  historical records are reference material only.
- Local paths use portable placeholders.
- Dated benchmark reports are immutable artifacts; corrections require a new
  dated report.
- New benchmark reports are written in English.
- Public claims must trace to code, tests, or report metadata.
- Add, remove, or rename a document and this index in the same change.
- Run `scripts/check_docs.sh` before committing documentation changes.
