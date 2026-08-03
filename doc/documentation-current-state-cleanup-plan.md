# Documentation Current-State Cleanup Plan

Status: complete
Date: 2026-08-03

## 1. Objective

Reduce the repository documentation to a small, maintainable set that describes
only:

1. current product behavior;
2. stable public contracts;
3. unfinished roadmap work.

Completed migration logs and superseded implementation plans will be removed
from the current tree. Git history remains the archive.

## 2. Audit Baseline

The initial audit found:

- 22 Markdown files directly under `doc/`;
- 12,863 lines in that directory;
- 13 completed or superseded plans totaling 10,596 lines;
- seven broken repository-relative links in one superseded acceleration plan;
- a CI document that still described the retired optimization configuration;
- a GEMM phase document whose claimed persistent thread pool is not present in
  the current implementation;
- a release plan still reporting P7.1 as waiting for commit;
- a README performance entry pointing at a report generated for a deleted
  target model;
- local absolute paths embedded in maintained documentation.

## 3. Target Documentation Set

The maintained `doc/` set will be:

```text
doc/
├── README.md
├── documentation-current-state-cleanup-plan.md
├── design.md
├── ci.md
├── cpu-optimization.md
├── gemm-optimization.md
├── mat-contract-v1.md
├── opencv-core-imgproc-api-coverage.md
├── opencv-core-imgproc-three-phase-support-plan.md
├── opencv-ui-kernel-migration-checklist.md
└── cvh-v0.1-release-readiness-and-next-stage-plan.md
```

The cleanup plan is retained as the requested execution record. It is excluded
from checks that reject historical identifiers because it records what was
removed.

## 4. Milestones And Live Feedback

| Milestone | Status | Live feedback |
| --- | --- | --- |
| D0: Record the plan and audit baseline | Complete | Plan created from the 2026-08-03 repository audit. No product code changed. |
| D1: Correct externally visible facts | Complete | README now labels the 2026-07-25 report as a historical checkpoint. The release roadmap is based on `d96bfde`, records P7.1 as complete, and starts active work at P7.2. Link and whitespace checks passed. |
| D2: Remove superseded plans and compatibility scripts | Complete | Deleted 13 superseded plans and both deprecated CI wrappers. Active `doc/` content dropped from 12,863 to 2,085 lines before the new current-state documents are added. Remaining references are queued for D4 index/readme repair. |
| D3: Establish current design references | Complete | Added concise current-state `ci.md`, `cpu-optimization.md`, and `gemm-optimization.md`; deleted the two superseded redesign logs. The GEMM document explicitly matches the current per-call standard-thread implementation. Whitespace checks passed. |
| D4: Refresh distributed documentation | Complete | Rebuilt the central index and Core/Imgproc/Imgcodecs/HighGUI/smoke guides; updated API coverage, the three-phase roadmap, UI kernel rules, benchmark instructions, and OpenCV provenance. Coverage arithmetic revalidated at Core 57+40 and Imgproc 50+73. Current docs have no broken relative links, retired identifiers, or local absolute paths. |
| D5: Rebuild the benchmark documentation entry | Complete | Generated the 2026-08-03 English report from a full run against the default upstream OpenCV build: 344 rows, 343 OK, one unsupported, and all implementation labels `cvh_ui`. The report generator no longer emits Phase/P-ACC history. README now points to the current report, while the result index separates immutable older snapshots. The alternative CPU-only OpenCV build exposed a GEMM correctness mismatch at square 512, so its tolerance was not weakened and its failed output was not published. |
| D6: Enforce documentation consistency | Complete | Added `scripts/check_docs.py` plus the shell entry point and integrated it into the required header-only CI. Final validation passed for 31 maintained Markdown files, Python and shell syntax, whitespace, the 12-test installed-header contract, all 20 CTest targets, Core 209/209, and Imgproc 187/187 with zero failures or skips. The maintained `doc/` set now totals 1,914 lines. |

This table is updated after every completed milestone. A milestone is not marked
complete until its verification commands pass.

## 5. Files To Remove

The following completed or superseded plans will be deleted:

- `benchmark-refactor-implementation-plan.md`;
- `core-cpp-cleanup-plan.md`;
- `imgproc-cpp-cleanup-plan.md`;
- `core-imgproc-test-refactor-plan.md`;
- `opencv-core-imgproc-phase1-implementation-plan.md`;
- `test-design-review-findings.md`;
- `x86-correctness-hardening-plan.md`;
- `mat-gemm-opencv-ui-microkernel-acceleration-plan.md`;
- `native-neon-avx2-compute-kernel-phase2-acceleration-plan.md`;
- `neon-gemm-phase3-blislab-kleidiai-acceleration-plan.md`;
- `neon-gemm-phase4-multithread-runtime-scheduler-acceleration-plan.md`;
- `opencv-upstream-simd-acceleration-plan.md`;
- `opencv-universal-intrinsics-adapter-plan.md`.

The two deprecated compatibility wrappers `scripts/ci_full_all.sh` and
`scripts/ci_lite_all.sh` will also be removed. They do not represent current
CI entry points.

## 6. Current-State Document Rules

Maintained documents must:

- describe current target and configuration names;
- distinguish required correctness CI from optional OpenCV comparison CI;
- avoid claims not traceable to current code, tests, or benchmark artifacts;
- use repository-relative links or portable placeholders instead of local
  absolute paths;
- keep dated benchmark reports immutable;
- use English for newly generated benchmark reports;
- point users to one current performance report;
- avoid retaining completed implementation logs in active design documents.

## 7. Benchmark Migration Rule

Existing dated reports remain unchanged because they record the configuration
that produced their measurements. A new report must:

- be generated from the current commit;
- compare only the forced OpenCV Universal Intrinsics cvh path against upstream
  OpenCV;
- identify the cvh implementation as `cvh_ui`;
- be written in English;
- include CSV and metadata alongside the Markdown report;
- become the only report described as current by the top-level README.

## 8. Automated Acceptance Checks

The final documentation check must verify:

- all maintained relative Markdown links resolve;
- every maintained `doc/*.md` file is indexed by `doc/README.md`;
- current documentation does not expose removed targets, build modes, or public
  optimization macros;
- current documentation contains no `/Users/...` paths;
- the top-level README points to an English `cvh_ui` report;
- old benchmark result files are treated as immutable historical artifacts;
- shell syntax, CMake configuration, and the header-only contract still pass;
- `git diff --check` reports no formatting errors.

## 9. Planned Commit Boundaries

The work is organized so it can be reviewed as:

1. removal of superseded plans and dead compatibility entry points;
2. establishment of current architecture, CI, and optimization references;
3. refresh of module guides and benchmark documentation;
4. documentation consistency automation.

The user may choose to squash these changes into one commit after review.
