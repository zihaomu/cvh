# UI-Only CI Workflow Redesign

Status: implemented in repository; hosted-run validation pending
Date: 2026-07-26

## 1. Decision

Hosted CI validates only the OpenCV Universal Intrinsics (UI) product path.

Core and Imgproc have no native backend. Their UI fast paths and scalar
fallbacks are both header-only implementations. The repository's legacy
`CVH_BUILD_NATIVE_BACKEND` name refers only to build-tree HighGUI `.cpp`
experiments.

The required pull-request and `main` branch gate builds and tests:

- `CVH_BUILD_NATIVE_BACKEND=OFF`
- `CVH_ENABLE_OPENCV_INTRIN=ON`
- the public header-only targets, including `cvh::headers_fast`
- all UI compile, dispatch, correctness, and installed-header contract tests

Hosted CI no longer runs:

- the `ui-off` / scalar-only matrix entry
- the legacy HighGUI `.cpp` compatibility job named `native_all`
- scalar, `full`, `lite`, or native implementations in the OpenCV comparison
  benchmark

This is a CI policy change, not a product-code removal. The scalar fallback and
the `CVH_ENABLE_OPENCV_INTRIN=OFF` configuration remain available for local
diagnostics. They remain header-only. Removing those implementations or build
options requires a separate product decision.

## 2. Target CI Model

The CI model has one required validation workflow and one optional performance
workflow:

```text
push / pull request / manual run
                 |
                 v
        required: headers_ui
        - header-only build
        - Universal Intrinsics enabled
        - full correctness suite
        - XML reports on success or failure

PR label / PR command / manual run
                 |
                 v
        optional: opencv_compare_ui
        - cvh::headers_fast only
        - upstream OpenCV baseline
        - log and report artifact
        - not a required branch-protection check
```

| Job | Purpose | Required |
| --- | --- | --- |
| `headers_ui` | Build and test the header-only UI product path | Yes |
| `opencv_compare_ui` | Compare `cvh::headers_fast` with upstream OpenCV | No |

There is no hosted `headers_scalar`, `ui-off`, or legacy HighGUI `native_all`
job.

## 3. Main Workflow

`.github/workflows/ci.yml` owns the required correctness gate.

### 3.1 Triggers

The workflow runs for:

- pushes;
- pull requests with `opened`, `synchronize`, `reopened`, and
  `ready_for_review`;
- manual `workflow_dispatch`.

`labeled` and `unlabeled` are intentionally excluded. Compare-label changes
must not rerun the required correctness gate.

Workflow concurrency cancels an older run for the same pull request or branch
when a newer commit arrives.

### 3.2 Required `headers_ui` Job

The single job:

1. checks out the requested revision;
2. configures a Release header-only build;
3. forces `CVH_ENABLE_OPENCV_INTRIN=ON`;
4. forces `CVH_BUILD_NATIVE_BACKEND=OFF` so legacy HighGUI `.cpp` experiments
   cannot enter the otherwise header-only build;
5. runs installed-header and public-header contracts;
6. builds with bounded parallelism;
7. runs the complete CTest and GoogleTest suites;
8. validates the UI test inventory and zero-skip expectations;
9. uploads machine-readable reports even if a test fails.

The stable displayed check name is:

```text
CI / Header-only UI
```

The old UI matrix and legacy HighGUI compatibility job are not part of hosted
CI. Core and Imgproc do not have an alternative native implementation.

## 4. UI Gate Script

`scripts/ci_headers_all.sh` is retained as the canonical filename to avoid a
noisy command migration, but its behavior is UI-only. It does not accept an
environment variable that can silently select a scalar-only hosted path.

It configures:

```text
CVH_ENABLE_OPENCV_INTRIN=ON
CVH_BUILD_NATIVE_BACKEND=OFF
CVH_BUILD_TESTS=ON
CVH_BUILD_BENCHMARKS=OFF
CMAKE_BUILD_TYPE=Release
```

The build directory is `build-ci-headers-ui`.

`CVH_CI_PARALLEL` remains configurable, with a bounded default of `2` to avoid
runner resource exhaustion.

### 4.1 Failure Reporting

The script:

1. creates the report directory before running tests;
2. runs CTest while preserving its exit status;
3. writes a CTest JUnit report;
4. runs the Core and Imgproc binaries with XML output even when CTest fails;
5. validates reports when both GoogleTest XML files exist;
6. returns a combined non-zero status after diagnostics are captured.

The workflow upload step uses `if: always()`. `if-no-files-found: warn` is a
final safety net for failures that occur before any test executable can run.

## 5. Test Inventory

`test/ci/header_gate_expectations.json` contains only the hosted `ui-on`
profile.

For both `x86_64` and `arm64`, the inventory continues to require:

- UI compile and runtime smoke tests;
- the architecture-specific UI smoke test where applicable;
- all Core and Imgproc correctness tests;
- zero failed tests;
- zero skipped UI tests.

Deleting the `ui-off` expectations records that scalar-only coverage is no
longer a hosted CI gate. It does not authorize removing scalar fallback code.
Both the retained UI path and the non-hosted scalar fallback are header-only.

## 6. OpenCV Comparison Workflow

The optional comparison has one product candidate:

```text
cvh::headers_fast (Universal Intrinsics enabled)
```

and one external baseline:

```text
upstream OpenCV
```

It produces no scalar, native, `full`, or `lite` CVH rows.

### 6.1 Canonical Command

The workflow and `scripts/ci_compare_log_only.sh` use:

```sh
./benchmark/opencv_compare/run_compare.sh \
  --profile quick \
  --impls headers_fast
```

The wrapper default is `headers_fast`. Benchmark metadata records the
normalized implementation as `cvh_headers_fast`.

### 6.2 Trigger Ownership

`.github/workflows/ci-compare-on-demand.yml` is the only workflow that owns the
performance comparison. It supports:

- manual dispatch;
- the `repository_dispatch` event sent by `/cvh-compare on`;
- pull-request `synchronize` and `reopened` events when the
  `ci/run-opencv-compare` label is present.

Enabling the label runs one comparison immediately. Later commits rerun the
comparison once while the label remains. Disabling the label stops future
automatic comparisons.

The main workflow does not listen for label changes or contain a comparison
job, preventing duplicate compare runs.

### 6.3 Compare Outputs

The optional job uploads:

- the rendered Markdown report;
- the CSV measurements;
- the metadata JSON;
- an environment fingerprint.

The workflow fails on build, execution, or malformed-output errors. It does
not fail solely because a performance number regresses. A statistically
defined regression gate can be introduced after stable historical data
exists.

## 7. Workflow Maintenance

The implementation:

- uses GitHub-provided action majors backed by Node 24;
- pins bounded build parallelism and explicit job timeouts;
- applies concurrency cancellation to required and optional workflows;
- keeps permissions minimal;
- retains the PR-comment actor allowlist for `/cvh-compare on|off`;
- uses stable job and artifact names.

| Job | Timeout |
| --- | ---: |
| `headers_ui` | 30 minutes |
| `opencv_compare_ui` | 60 minutes |

## 8. Current Failures and Migration Behavior

The redesign must not hide genuine UI failures. At the time of the design, the
UI job exposed correctness failures including:

- floating-point signed-zero operand-order semantics in array min/max;
- UI `convertScaleAbs` disagreement with the scalar reference on ROI, tail,
  or edge inputs.

Those failures remain blockers for the required `headers_ui` check and must be
fixed or deliberately re-specified in separate code changes.

The scalar-only signed-zero failure no longer blocks CI after `ui-off` is
removed, but any shared failure that also appears in the UI path continues to
block CI.

Local implementation validation on 2026-07-26 completed successfully on
`arm64`:

- 17 of 17 UI-enabled CTest entries passed;
- all 209 Core tests passed with zero skips;
- all 186 Imgproc tests passed with zero skips;
- CTest, Core, and Imgproc XML reports were generated and parsed.

The first hosted Linux `x86_64` run remains the authoritative validation for
the architecture-specific UI smoke and the new GitHub Actions wiring.

## 9. Branch Protection Migration

Changing workflow job names can leave pull requests waiting for old required
checks that will never be reported.

Use this order after the workflow is pushed:

1. run the new workflow and confirm `CI / Header-only UI` appears;
2. add the new check to branch protection;
3. remove `Header-only (ui-on)`, `Header-only (ui-off)`, and `native_all`;
4. verify that a new pull request can merge using only the new required check.

The optional `OpenCV Compare UI` check must not be required by branch
protection.

## 10. File Change Map

| File | Change |
| --- | --- |
| `.github/workflows/ci.yml` | One required UI job; matrix, legacy HighGUI, and compare jobs removed |
| `.github/workflows/ci-compare-on-demand.yml` | UI-only `headers_fast` compare and artifact upload |
| `.github/workflows/ci-compare-toggle.yml` | Label toggle with one immediate dispatch |
| `scripts/ci_headers_all.sh` | Fixed UI configuration and failure-report preservation |
| `scripts/ci_compare_log_only.sh` | `headers_fast` default and persistent report artifacts |
| `benchmark/opencv_compare/run_compare.sh` | Explicitly enable UI for the comparison build |
| `test/ci/header_gate_expectations.json` | UI-only hosted expectations |
| repository branch protection | Require only the stable UI check after first hosted run |

`scripts/ci_native_all.sh` is not deleted, but it is no longer a hosted CI
entry point. Its name refers to the legacy HighGUI `.cpp` build switch; it does
not provide native Core or Imgproc implementations.

## 11. Acceptance Criteria

The redesign is complete when:

1. A normal push or pull request creates exactly one required validation job.
2. Its cache shows `CVH_ENABLE_OPENCV_INTRIN=ON` and
   `CVH_BUILD_NATIVE_BACKEND=OFF`.
3. No hosted workflow runs `ui-off`, scalar-only, or the legacy HighGUI
   `native_all` job.
4. The UI CTest inventory passes and Core/Imgproc reports contain zero failures
   and zero skipped UI tests.
5. A deliberate test failure still leaves downloadable XML reports.
6. The compare workflow invokes only `--impls headers_fast`.
7. Compare metadata contains `cvh_headers_fast` and no scalar, native, `full`,
   or `lite` CVH implementation.
8. `/cvh-compare on` causes exactly one immediate compare run, and
   `/cvh-compare off` prevents later labeled-PR runs.
9. Workflow logs contain no GitHub Actions Node 20 retirement warning.
10. Branch protection requires only the stable UI validation check.

## 12. Non-Goals

This redesign does not:

- remove scalar fallback code;
- remove the UI-off CMake option from local builds;
- fix unrelated correctness failures;
- turn benchmark variance into a required correctness gate;
- redesign benchmark kernels or result presentation;
- expand CI to additional operating systems or compilers.
