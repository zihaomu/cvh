# cvh

**A compact OpenCV-style subset for common C++ image processing workloads.**

`cvh` is a focused OpenCV-style subset library for common `core` and `imgproc` workloads.  
It is designed for projects that want familiar OpenCV-like APIs with a smaller integration surface, a header-first development model, and optional compiled backends for broader coverage and higher performance.

## Why cvh

OpenCV is powerful, but many projects only need a small and stable subset of its image processing capabilities.

`cvh` focuses on that subset:

- common `core` + `imgproc` building blocks
- OpenCV-style APIs and coding model
- header-first integration for lightweight use cases
- optional compiled/full mode for better performance and broader implementation coverage
- correctness and performance comparison against OpenCV

## Project scope

### What cvh is

`cvh` is:

- a compact OpenCV-style subset
- focused on practical image processing workloads
- designed for portability and incremental adoption
- validated against OpenCV for correctness and performance on selected operators

### What cvh is not

`cvh` is **not**:

- a full replacement for OpenCV
- a wrapper around all OpenCV modules
- a promise of full API/ABI compatibility
- a project that prioritizes every long-tail operator equally
- a project that treats “pure header-only at all costs” as the primary goal

## Runtime modes

`cvh` currently supports two implementation modes:

- **Lite**  
  Header-first, minimal dependency footprint, suitable for lightweight integration and portability-first scenarios.

- **Full**  
  Uses compiled backend implementations for broader operator coverage and higher performance.

This split allows `cvh` to stay easy to integrate without blocking future optimization work such as SIMD, parallel execution, and more specialized kernels.

## Who this is for

`cvh` is intended for:

- C++ projects that mainly need common `core` / `imgproc` functionality
- teams that prefer OpenCV-like APIs but want a smaller and more controllable dependency surface
- developers who want correctness and performance evidence relative to OpenCV
- portable or embedded-oriented codebases where “subset first” is more realistic than “full framework”

## Current focus

The current priority is to make the high-frequency subset solid before expanding surface area:

- stabilize common `core` and `imgproc` operators
- improve behavior parity with OpenCV on supported paths
- maintain benchmark visibility against OpenCV
- build performance credibility on important kernels
- keep the public API compact and understandable

## Roadmap

### Phase 1 — Focused subset
- solidify common `core` and `imgproc` operators
- keep APIs small and coherent
- make Lite mode easy to adopt

### Phase 2 — Correctness parity
- strengthen result comparison against OpenCV
- add clearer behavioral guarantees for supported operators
- prevent regressions in CI

### Phase 3 — Performance credibility
- benchmark key operators against OpenCV
- improve Full mode performance on hot paths
- expand SIMD / parallel optimization where it matters most

### Phase 4 — Distribution
- improve packaging and install experience
- publish clearer release notes and support matrix
- make adoption easier for external users

## Repository layout

- `include/` — public headers and header-first implementation pieces
- `src/` — compiled/full-mode backend implementations
- `test/` — correctness and regression tests
- `benchmark/` — performance benchmarks
- `opencv_compare/` — comparison workspace and reports against OpenCV
- `example/` — usage examples

## Build

```bash
cmake -S . -B build
cmake --build build -j