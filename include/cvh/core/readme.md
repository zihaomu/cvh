# Core Module

## Responsibility

Core owns:

- `Mat`, shape, stride, ROI, storage, conversion, and expression behavior;
- `Scalar`, `Range`, `Point`, `Size`, `Rect`, `Moments`, type encoding, errors, and casts;
- array arithmetic, math, reductions, copy/channel/layout operations;
- random Mat filling and point-channel transforms;
- transpose, GEMM, and the header-only parallel runtime;
- internal CPU capability and dispatch infrastructure.

It does not own image codecs, image-processing algorithms, or GUI behavior.

## Public Headers

`include/cvh/core/*.h` files are public module entries. `detail/**`, `simd/**`,
and `*.inl.h` are implementation files and are not independent public APIs.

The stable Mat contract is documented in
[`doc/mat-contract-v1.md`](../../../doc/mat-contract-v1.md).
CPU dispatch and GEMM are documented in
[`doc/cpu-optimization.md`](../../../doc/cpu-optimization.md) and
[`doc/gemm-optimization.md`](../../../doc/gemm-optimization.md).

## Optimization Boundary

Core implementations may select specialized ISA, OpenCV Universal Intrinsics,
or scalar paths. Those paths must expose the same public type/layout contract
and remain inline header code.

Consumers configure only `CVH_ENABLE_OPTIMIZATION`. Internal dispatch controls
are reserved for tests and benchmarks.

## Validation

- `test/core/`: public contracts, runtime, upstream subset, and internal
  dispatch correctness.
- `test/smoke/core_headers/`: independent compilation for every top-level Core
  public header.
- `cvh_core_header_odr_smoke`: multi-translation-unit ODR safety.
- `cvh_test_gemm_isa`: specialized GEMM ISA correctness.

New Core APIs require a public contract test, boundary cases, and a registered
header compile source when a new public header is added.

## Phase 2 P0 Support Matrix

| API | Accepted subset |
|---|---|
| `randu`, `randn` | Preallocated U8/S8/U16/S16/S32/F32/F64 C1-C4 Mat; continuous 2D/N-D and non-contiguous 2D ROI; per-channel `Scalar` parameters. |
| `transform` | 2D F32/F64 C1-C4 source; F32/F64 C1 matrix with `src.channels()` or `src.channels()+1` columns and one to four output rows; ROI and alias-safe output. |
| `perspectiveTransform` | `N x 1`/`1 x N` F32/F64 C2/C3 point Mat with a 3x3/4x4 F32/F64 matrix; pinned zero-`w`, NaN and Inf behavior. |

Random engine classes and public seed/state control remain outside this subset.
