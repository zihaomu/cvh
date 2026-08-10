# Smoke Test Responsibilities

Smoke tests provide fast structural evidence that is not efficiently expressed
as operator unit tests.

## Retained Categories

### Public-header compilation

- Core, Imgproc, Pipeline, Imgcodecs, and HighGUI public headers compile
  independently.
- Aggregate `cvh/cvh.h` and compatibility `cvh.h` compile independently.
- Configure-time manifests fail when a public header lacks a compile source.

### ODR and target contracts

- Core, Imgproc, Pipeline, and HighGUI are linked from multiple translation
  units.
- Inline dispatch telemetry and HighGUI state do not create duplicate symbols.
- Installed `cvh::headers` and `cvh::highgui` consumers build outside the source
  tree.

### Minimal runtime pipelines

- direct include and basic Mat behavior;
- Imgcodecs to Imgproc read/process/write flow;
- default dispatch and output sanity;
- prepared Pipeline execution performs zero heap allocations;
- optimization-disabled scalar behavior;
- OpenCV UI and architecture-specific compile/runtime capability where
  applicable.

## Boundary With Unit Tests

A smoke test should remain small and answer one structural availability
question. Broad operator matrices, numerical edge cases, and detailed dispatch
behavior belong in `test/core`, `test/imgproc`, or the dedicated ISA tests.

When a smoke grows into a large functional suite, move its cases to the owning
module and retain only the smallest end-to-end check here.

## Required Gate

The header-only install/consumer contract runs the public compile, ODR, include,
and minimal pipeline smoke set:

```bash
./scripts/check_header_only_contract.sh
```

The complete hosted gate is:

```bash
./scripts/ci_headers_all.sh
```
