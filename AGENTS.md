# cvh Repository Agent Instructions

This file applies to the entire `cvh` repository. Read it before starting any
code, test, benchmark, CMake, or documentation work. If a subdirectory contains
a more specific `AGENTS.md`, apply its instructions in addition to this file.

## Before Starting Work

- Begin with read-only inspection: confirm the current branch, run
  `git status --short`, and inspect the relevant source, tests, and active plan.
  Do not overwrite, clean up, or casually commit existing user changes.
- When the user names an implementation plan, read it completely. That plan is
  the status owner for the task: update its batch status and completion criteria
  immediately when work starts, completes, is rolled back, or becomes blocked.
- Reuse existing implementations, tests, scripts, benchmark cases, and report
  generators. Do not create one-off product targets or unmaintained diagnostic
  tools.

## Product and Correctness Contracts

- `cvh` is an independent header-only implementation. Normal product targets
  must not gain a binary runtime dependency on OpenCV.
- The sibling `../opencv` repository is the API, behavior, result, and
  performance reference. Within the supported surface, outputs, boundaries,
  exceptions, ROI/step behavior, and dispatch behavior must pass upstream
  differential validation. Do not relax a frozen tolerance, remove cases, or
  disable checksums to obtain better performance results.
- When changing specialized ISA or UI paths, preserve a reliable scalar
  fallback. Cover optimization on/off, forced dispatch, tails, unaligned data,
  ROI/non-contiguous layouts, and compilation on non-target platforms.
- Benchmarks must distinguish the algorithm, dispatch path, and actually
  observed ISA. Never infer the executed path solely from the build platform.

## Reusable Build Configurations and SSD Discipline

**Distinct reusable build configurations are encouraged. Redundant throwaway
build directories are discouraged.** There is no hard limit on the number of
build directories. Development speed and configuration correctness take
priority over minimizing their count.

Each build directory must have a stable, explicit configuration identity, such
as:

```text
build-dev-release
build-dev-debug
build-opt-off
build-sanitize
build-opencv-compare
build-x86-cross
build-perf-baseline
build-perf-candidate
```

Before building:

1. List the existing top-level `build*` directories.
2. Inspect candidate `CMakeCache.txt` files without modifying them. Check the
   generator, build type, compiler/architecture, `CVH_ENABLE_OPTIMIZATION`,
   sanitizer configuration, tests/benchmarks, and `OpenCV_DIR`.
3. Reuse a cache when its configuration identity is compatible with the task,
   and continue with incremental reconfiguration and compilation.
4. Never force reuse of an incompatible, ambiguous, or untrusted cache merely
   to reduce SSD writes.
5. Prefer building affected targets during iteration. Use moderate parallelism
   by default, normally `--parallel 2`, unless the task justifies otherwise.

Creating a new build directory does not require special approval when it
represents a useful configuration that should remain available. Valid reasons
include:

- a different generator, compiler, architecture, ABI, build type, sanitizer,
  optimization mode, or upstream OpenCV configuration;
- keeping baseline and candidate binaries simultaneously available for
  performance or correctness comparison;
- comparing multiple source versions, branches, commits, or worktrees without
  repeatedly reconfiguring one cache;
- release and CI configuration validation;
- low confidence in an existing cache, suspected stale state, or a cache that
  is damaged or cannot be repaired safely; or
- an explicitly requested isolated experiment.

For multi-version comparisons, create dedicated baseline and candidate build
directories as needed. Preserve them for repeated measurements and incremental
rebuilds. A build directory created for a worktree should remain associated
with that worktree/configuration identity.

Clean builds are allowed when required for release evidence, CI parity,
toolchain or dependency validation, reproducibility, or cache-trust concerns.
After the clean configuration is established, reuse it incrementally for later
runs whenever its identity remains valid.

Avoid directories whose only identity is an attempt number, date, or vague
suffix, for example `build-try2`, `build-final2`, or one new directory per
benchmark run. Do not delete and recreate a compatible build directory merely
to avoid inspecting or incrementally updating it.

- Before a large clean rebuild, briefly state its configuration purpose and why
  an incremental cache is insufficient. This is a progress update, not a
  request for permission when the rebuild is already within task scope.
- Put small diagnostic programs and temporary artifacts in a relevant ignored
  diagnostic/build directory. Reuse that directory across related experiments;
  do not place these artifacts in the source tree or automatically promote them
  to product or CI targets.
- A "clean revision performance result" means that the source revision and
  worktree are traceable. It does not require an empty build directory unless
  the measurement methodology or cache-trust requirements call for one.

## Implementation, Validation, and Performance Evidence

- Identify the root cause and hot path before modifying the implementation. Do
  not hide a narrow task inside a broader refactor.
- Match validation depth to risk: run targeted tests first, then the affected
  module, followed by any full, header, ODR, install, sanitizer, and
  cross-platform gates required by the active plan.
- Performance conclusions must come from reproducible Release measurements on
  the same machine, with the same thread count, inputs, and sampling settings.
  A single probe may guide investigation; closing a gate requires the stable
  multi-run evidence specified by the plan.
- Do not overwrite dated benchmark reports. Corrections require a new report
  file and synchronized CSV, metadata, and index updates.
- Record the measurements and rollback reason for failed candidates, then
  remove them from the product path. Do not lower a gate because work has
  already been invested.

## Editing and Git

- Keep changes small and focused, preserving the public API and header-only/ODR
  contracts.
- When adding, deleting, or renaming documentation, update `doc/README.md` and
  run `./scripts/check_docs.sh` plus `git diff --check`.
- Unless the user explicitly requests it, do not create commits, push, merge
  branches, or modify the sibling OpenCV repository. Never use Git commands
  that can discard worktree content.
