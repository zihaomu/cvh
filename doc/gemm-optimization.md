# GEMM Implementation And Optimization

Updated: 2026-08-03

## 1. Public API

GEMM is exposed through `cvh/core/mat.h`:

```cpp
cvh::Mat cvh::gemm(const Mat& a,
                   const Mat& b,
                   bool transA = false,
                   bool transB = false);

cvh::GemmPackedB cvh::gemm_pack_b(const Mat& b,
                                  bool transB = false);

cvh::Mat cvh::gemm(const Mat& a,
                   const GemmPackedB& packed_b,
                   bool transA = false);
```

An additional overload accepts INT8 weight scales.

## 2. Current Numeric Contract

- Activations are `CV_32F`.
- Standard and packed weights are `CV_32F` or `CV_16F`.
- Output accumulation and storage are `CV_32F`.
- Leading dimensions follow NumPy-style broadcasting.
- The final two dimensions represent matrix rows and reduction/output columns.
- INT8 weights require per-output scales and currently support only
  `transA=false, transB=true`.
- Unsupported type/layout combinations fail explicitly rather than silently
  selecting a different contract.

Transposed layouts that do not have a direct specialized path are normalized
through `transpose` before entering the main implementation.

## 3. Implementation Layers

```text
public gemm overloads
    -> layout/broadcast normalization in gemm_impl.hpp
    -> specialized ISA selection in gemm_isa.hpp
    -> OpenCV UI kernels in gemm_ui.hpp
    -> scalar fallback in gemm_impl.hpp
```

Specialized kernels live in:

- `gemm_neon.hpp` for AArch64 NEON;
- `gemm_avx2.hpp` for x86 AVX2/FMA.

All layers are inline header implementations.

## 4. Dispatch

`Auto` considers a specialized ISA only when:

- `M`, `N`, and `K` are positive;
- `M >= 2`, `N >= 8`, and `K >= 8`;
- `M * N * K >= 32768`;
- the kernel compiled and runtime CPU capability is available.

AVX2 additionally requires `N >= 16`. FP16 weights do not currently use the
AVX2 path. INT8 weights remain on the scalar scaled-dot path.

When no specialized path is selected, supported FP32 layouts try OpenCV UI and
then scalar code. Dispatch tags expose the selected family for tests and
benchmarks.

Forced dispatch modes exist only for correctness and performance diagnostics.

## 5. Packed Weights

`GemmPackedB` stores:

- canonical FP32 or FP16 row-major data;
- shape, strides, type, `K`, `N`, and batch metadata;
- an optional aligned NEON-ready FP32 panel cache.

The canonical representation is always retained as the fallback. The current
AVX2 implementation consumes canonical FP32 packed data; there is no separate
public AVX2 packed format.

Packed data is an optimization object tied to the current cvh representation.
It is not a serialized stable interchange format.

## 6. NEON And AVX2 Kernels

The NEON NN path packs B into 16-column panels and uses a 6-row micro-kernel
family, including tail handling. FP16 weights are expanded to FP32 while
packing and still accumulate in FP32.

The AVX2/FMA NN path uses 16-column blocks and row-height variants. The NT path
uses an AVX2 dot-product kernel where its layout and width requirements are
satisfied.

Kernel availability never removes the scalar or UI implementation.

## 7. Parallel Execution

GEMM can parallelize outer batches or output rows through `cvh::parallel_for_`
when workload thresholds are met.

Current facts:

- `ParallelBackend::Auto` resolves to the standard-thread backend when work is
  large enough;
- the standard-thread backend creates and joins worker threads for each
  parallel call;
- there is no persistent cvh thread pool;
- OpenMP code is available only when the consumer compilation environment
  already defines OpenMP support; project CMake does not enable it;
- small workloads remain serial to avoid thread overhead.

Any future persistent runtime must be implemented and benchmarked before this
document claims it as current behavior.

## 8. Correctness Coverage

`cvh_test_core` covers public GEMM contracts and layouts.
`cvh_test_gemm_isa` covers architecture-specific selection, forced paths,
runtime fallback, tails, and packed-weight behavior.

Required cases include:

- NN and NT layouts;
- transposed normalization paths;
- FP32 and FP16 weights;
- INT8 scaled NT behavior;
- batch broadcasting;
- packed and unpacked B;
- small-shape fallback and ISA tails;
- special floating-point values and tolerance behavior.

## 9. Known Limits

- The implementation is not a BLAS replacement.
- INT8 support does not quantize activations or use a dot-product integer
  accumulation contract.
- AVX2 FP16 weight execution is not implemented.
- Persistent worker scheduling is not implemented.
- Performance acceptance requires real target hardware; cross-compilation is
  only compile evidence.

## 10. Maintenance Rules

- Public type/layout support must remain identical across dispatch paths.
- A new kernel requires forced-path correctness tests and benchmark evidence.
- Shape gates must prevent known small-workload regressions.
- Packed-format changes require fallback and metadata validation tests.
- Current performance claims belong in dated reports, not this design document.
