# Continuous Integration

Updated: 2026-08-03

## 1. Scope

CI has three separate responsibilities:

| Workflow | Role | Required for normal changes |
| --- | --- | --- |
| `CI` | Complete optimized header-only correctness gate | Yes |
| `x86 correctness` | x86-64-v3, AVX2/FMA, and sanitizer coverage | Scheduled/manual |
| `CI Compare On Demand` | UI-only cvh versus upstream OpenCV performance visibility | No |

The required correctness gate and the optional OpenCV comparison are not the
same policy. The required gate enables the complete accepted optimization
configuration; only the comparison forces the OpenCV Universal Intrinsics path.

## 2. Required Header-Only Gate

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual
dispatch. Its stable check is:

```text
CI / Header-only UI
```

The stable name is retained for branch-protection compatibility. The actual
configuration is the single public optimized product target:

```text
CMAKE_BUILD_TYPE=Release
CVH_BUILD_TESTS=ON
CVH_BUILD_BENCHMARKS=OFF
CVH_ENABLE_OPTIMIZATION=ON
```

`scripts/ci_headers_all.sh` performs:

1. environment fingerprint logging;
2. documentation link, index, vocabulary, local-path, and current-report checks;
3. public-header and install-tree consumer contracts;
4. fixture manifest validation;
5. a clean Release configure and complete build;
6. full CTest execution;
7. explicit Core and Imgproc GoogleTest XML generation;
8. architecture-aware test-count and zero-skip validation.

The build covers `cvh::headers`, `cvh::highgui`, scalar fallback, OpenCV UI,
and compiled specialized ISA paths applicable to the runner. It does not build
a project binary library.

Run it locally with:

```bash
CVH_CI_PARALLEL=2 ./scripts/ci_headers_all.sh
```

Reports are written under:

```text
build-ci-headers-ui/test-reports/
```

The workflow uploads them even when tests fail.

## 3. x86 Correctness Gate

`.github/workflows/ci-x86-correctness.yml` runs weekly and by manual dispatch.
It invokes `scripts/ci_x86_correctness.sh` and contains two builds:

- x86-64-v3 Release correctness;
- sanitizer-focused correctness.

This gate owns real x86 AVX2/FMA runtime evidence. A successful compile on a
non-x86 machine is not reported as runtime coverage.

Run it on an x86-64 host with:

```bash
CVH_X86_CI_PARALLEL=2 \
CVH_X86_SANITIZER_PARALLEL=2 \
./scripts/ci_x86_correctness.sh
```

## 4. UI-Only OpenCV Comparison

`.github/workflows/ci-compare-on-demand.yml` is an optional performance
workflow. It can be triggered by:

- manual dispatch;
- the compare repository-dispatch event;
- a labeled pull request after synchronization or reopening.

The `ci/run-opencv-compare` label is managed by
`.github/workflows/ci-compare-toggle.yml` through the authorized
`/cvh-compare on` and `/cvh-compare off` commands.

The comparison configuration is fixed to:

```text
CVH_COMPARE_IMPLS=ui
CVH_COMPARE_THREADS=1
```

The cvh executable forces `OpenCVUIOnly`. It rejects specialized NEON or AVX2
dispatch tags and normalizes the cvh implementation name to `cvh_ui`.

Run a local quick comparison with:

```bash
CVH_COMPARE_PROFILE=quick \
CVH_COMPARE_IMPLS=ui \
CVH_COMPARE_THREADS=1 \
./scripts/ci_compare_log_only.sh
```

The workflow uploads Markdown, CSV, metadata, and environment artifacts. It
fails for build, execution, or malformed-report errors; raw performance
variance is not currently a required correctness gate.

## 5. Branch Protection

Only the stable required correctness check should block normal merges. The
OpenCV comparison and scheduled x86 workflow are additional evidence and must
not leave pull requests waiting for an event that does not run on every commit.

## 6. Platform Gaps

The current required workflow runs on Ubuntu. Continuous macOS ARM64 and
Windows MSVC gates remain P7.3 release-readiness work. Local Apple ARM or
cross-compilation results do not replace those hosted gates.

## 7. Maintenance Rules

- Keep tests and benchmarks opt-in in the default product configure.
- Do not introduce alternative public build modes for scalar, UI, or ISA paths.
- Forced dispatch belongs to tests and benchmarks, not consumer CMake targets.
- Preserve machine-readable reports on failure.
- Update workflow, script, expectations, and this document in the same change.
- Keep the optional comparison UI-only until a different comparison policy is
  explicitly approved and documented.
