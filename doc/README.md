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
- [pipeline-module-design.md](pipeline-module-design.md): proposed
  fluent-first `cvh::pipe(...)` API, ordered semantics, safe fusion,
  prepare/run and workspace contracts, model-input Recipes, and robot
  preprocessing/postprocessing boundaries.
- [opencv-core-imgproc-api-coverage.md](opencv-core-imgproc-api-coverage.md):
  pinned upstream operation-family inventory and current callable cvh subset.
- [opencv-core-imgproc-three-phase-support-plan.md](opencv-core-imgproc-three-phase-support-plan.md):
  remaining Core and Imgproc expansion roadmap.
- [opencv-ui-kernel-migration-checklist.md](opencv-ui-kernel-migration-checklist.md):
  current rules for OpenCV Universal Intrinsics and specialized ISA kernels.
- [tutorial/README.md](tutorial/README.md): bilingual Resize and Canny tutorial
  catalog plus the authoring contract for operator baselines, optimization
  steps, and reproducible OpenCV comparisons.
- [cvh-phase2-p0-operator-acceleration-plan.md](cvh-phase2-p0-operator-acceleration-plan.md):
  implementation batches, correctness gates, upstream comparisons, and live
  status for the second-stage acceleration of the 17 Phase 2-P0 operation
  families.
- [cvh-v0.1-neon-hot-kernel-acceleration-plan.md](cvh-v0.1-neon-hot-kernel-acceleration-plan.md):
  narrowly scoped direct-NEON acceleration, dispatch observability, correctness
  gates, and performance floors for packed/YUV color conversion, U8C3 bilinear
  resize, and shared 3x3 derivative kernels.
- [cvh-v0.1-resize-u8c3-fixed-point-neon-acceleration-plan.md](cvh-v0.1-resize-u8c3-fixed-point-neon-acceleration-plan.md):
  fixed-point numeric contract, flat-C3 direct-NEON kernel, correctness gates,
  and near-upstream performance closure for U8C3 bilinear resize.
- [cvh-v0.1-core-mat-native-neon-acceleration-plan.md](cvh-v0.1-core-mat-native-neon-acceleration-plan.md):
  native-NEON candidate policy, shared reduction/statistics kernels, fused
  rotate90, packed inRange, and conditional high-frequency elementwise work.
- [cvh-v0.1-release-readiness-and-next-stage-plan.md](cvh-v0.1-release-readiness-and-next-stage-plan.md):
  P7.2 onboarding, P7.3 platforms, P7.4 performance baseline, and `0.1.0`
  release gates.

## Ownership Rules

| Fact | Owner |
| --- | --- |
| Public usage and current report entry | top-level `README.md` |
| Product/module/target boundary | `design.md` |
| CPU configuration and dispatch | `cpu-optimization.md` |
| GEMM implementation facts | `gemm-optimization.md` |
| Proposed Pipeline API and execution contract | `pipeline-module-design.md` |
| Supported operation families | API coverage |
| Tutorial organization and authoring contract | `tutorial/README.md` |
| Phase 2-P0 acceleration work | Phase 2-P0 operator acceleration plan |
| v0.1 direct-NEON hot-kernel work | v0.1 NEON hot-kernel acceleration plan |
| v0.1 U8C3 fixed-point resize closure | v0.1 Resize U8C3 fixed-point NEON acceleration plan |
| v0.1 Core/Mat native-NEON acceleration | v0.1 Core/Mat native-NEON acceleration plan |
| Unfinished release work | release-readiness plan |
| Test status | `test/ci/header_gate_expectations.json` and upstream manifests |
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
