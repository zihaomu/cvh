# cvh 0.1 Release Readiness And Next-Stage Plan

Updated: 2026-08-06

Current baseline: `cbd5076` (`release: align support scope and auto dispatch reporting`)

Current stage: P7.1 is complete. P7.2 user onboarding and release packaging is
the next product milestone.

## 1. Purpose

`cvh` now has a stable pure header-only product boundary and a useful subset of
Core, Imgproc, Imgcodecs, and optional HighGUI. The remaining work before
`0.1.0` is productization rather than another broad implementation expansion.

This plan owns only unfinished release and post-release work. Completed cleanup
details remain available through Git history and are not repeated here.

## 2. Current Product Baseline

### Public targets

```cmake
cvh::headers
cvh::highgui
```

- `cvh::headers` provides Core, Imgproc, and Imgcodecs.
- `cvh::highgui` depends on `cvh::headers` and propagates only the platform GUI
  libraries needed by inline HighGUI code.
- Neither target produces a cvh binary library.
- Tests and benchmarks are disabled in the default product configure.

### CPU optimization

- `CVH_ENABLE_OPTIMIZATION` is the only public CPU optimization policy.
- The default value is `ON`.
- OpenCV Universal Intrinsics, NEON, and AVX2 compile/runtime availability are
  internal implementation facts.
- Scalar fallback remains available when optimization is disabled or no
  accepted optimized path applies.

### Current validation

- public-header self-containment checks;
- multi-translation-unit ODR checks;
- install-tree CMake consumer checks;
- Core, Imgproc, Imgcodecs, HighGUI, and ISA correctness tests;
- Linux x86 correctness and sanitizer workflow;
- optional product-auto OpenCV comparison workflow with forced UI/scalar diagnostics.

## 3. Completed Release-Readiness Work

### P7.0: Product identity and fact ownership

- Project branding, namespace, CMake package, and public targets use `cvh`.
- README and design documents describe an independent OpenCV-style project,
  not an OpenCV distribution.
- API support, tests, and dated benchmark artifacts have defined owners.

Status: complete, with documentation consistency now enforced by the dedicated
cleanup plan.

### P7.1: Pure header-only product boundary

- Product implementation source files and compiled project backends are gone.
- HighGUI is an optional inline header module with platform system-library
  dependencies.
- Core and Imgproc optimized paths are inline header implementations.
- The public CPU configuration is reduced to one policy switch.
- Default CMake configuration creates interface targets only.
- The required hosted gate and optional OpenCV comparison have separate roles.

Status: complete in `d96bfde`.

## 4. P7.2: User Onboarding And Release Packaging

### 4.1 Supported integration paths

README must provide tested examples for:

1. direct include;
2. installed CMake package;
3. `FetchContent` pinned to a release tag or commit.

Each path must explain:

- the C++17 requirement;
- when to link `cvh::headers` versus `cvh::highgui`;
- how to request a scalar-only build;
- that HighGUI may require platform GUI development libraries.

### 4.2 Product examples

Provide small public-API examples for:

- Mat and Core operations;
- image read, resize/color conversion, and write;
- an in-memory preprocessing pipeline;
- optional HighGUI display.

Examples must not include internal headers, depend on repository source paths,
or require OpenCV libraries.

### 4.3 Release files

Add and validate:

- `CHANGELOG.md`;
- `CONTRIBUTING.md`;
- `SECURITY.md`;
- a release checklist;
- source archive and checksum generation.

### 4.4 Acceptance criteria

- [ ] A clean external directory can use all three supported integration paths.
- [ ] Product examples compile in CI.
- [ ] The source archive installs and is consumed without the repository build tree.
- [ ] Release documentation contains no developer-machine paths.

## 5. P7.3: Cross-Platform Quality Gates

### Required platform matrix

| Platform | Toolchain | Required coverage |
| --- | --- | --- |
| Linux x86-64 | GCC and Clang | headers, unit tests, x86 ISA, sanitizers, install consumer |
| macOS ARM64 | AppleClang | headers, NEON runtime paths, HighGUI compile, install consumer |
| Windows x64 | MSVC | headers, AVX2 compile/runtime where available, HighGUI compile, install consumer |

Additional configurations:

- C++17 required build;
- C++20 compatibility build;
- optimization-disabled scalar compile/runtime smoke;
- optional OpenCV contract and comparison builds.

Acceptance criteria:

- [ ] Required workflows run continuously rather than only on developer machines.
- [ ] Installed consumers pass on all three platforms.
- [ ] Architecture-specific tests report compile-only versus runtime execution explicitly.
- [ ] Test inventory and skip expectations remain machine-readable.

## 6. P7.4: Current Performance Baseline

Performance claims must be regenerated after the P7.1 target and dispatch
consolidation.

Required reports:

- Apple ARM64;
- Linux x86-64;
- internal baseline/current regression;
- product-auto cvh versus upstream OpenCV comparison, plus forced UI/scalar diagnostics.

Every report records:

- cvh and OpenCV revisions;
- compiler, build type, OS, architecture, and CPU;
- thread count and runtime flags;
- requested and actual dispatch path;
- sampling parameters and raw CSV location.

Acceptance criteria:

- [ ] README points to an English report generated from the release candidate.
- [ ] Product-auto reports record specialized ISA paths; forced UI diagnostics exclude them.
- [ ] Regressions are separated from accepted numerical or implementation changes.
- [ ] Dated reports remain immutable after publication.

## 7. `cvh 0.1.0` Definition Of Done

### Product and documentation

- [ ] README, design, API coverage, CI, and optimization documents agree.
- [ ] Supported API claims are traceable to tests.
- [ ] Public target and configuration examples are executable.

### Integration and release

- [ ] Direct include, install package, and FetchContent consumers pass.
- [ ] Version, tag, package version, changelog, and release notes agree.
- [ ] Source archive and checksum are published and independently verified.

### Correctness and platforms

- [ ] Linux GCC/Clang required gate passes.
- [ ] macOS ARM64 required gate passes.
- [ ] Windows MSVC required gate passes.
- [ ] Sanitizers and install-tree consumers pass.

### Performance

- [ ] The current English performance report matches the release commit.
- [ ] Scalar, UI, NEON, and AVX2 forced-path tests cover accepted GEMM dispatch.
- [ ] Important Core and Imgproc anchors have no unexplained regression.

## 8. Post-Release Priorities

After `0.1.0`, expand only APIs justified by real pipelines and shared
infrastructure.

Priority order:

1. batch data movement and channel operations;
2. quantization-friendly Core operations;
3. connected components, contour, and shape analysis;
4. feature extraction needed by practical applications;
5. additional geometry and preprocessing fusion only when benchmarks show a
   material benefit.

New optimized paths must include forced-path correctness tests, internal
regression measurements, and a documented fallback.

## 9. Delivery Order

```text
documentation current-state cleanup
    -> P7.2 onboarding and release files
    -> P7.3 cross-platform CI
    -> P7.4 release-candidate performance baseline
    -> cvh 0.1.0
    -> selective API and performance expansion
```

The release is blocked by product adoption, platform verification, and current
performance evidence—not by reaching complete OpenCV API coverage.
