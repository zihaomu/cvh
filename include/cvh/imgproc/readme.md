# Imgproc Module

## Responsibility

Imgproc provides the accepted OpenCV-style image-processing subset, including:

- color conversion and YUV input handling;
- resize, remap, affine/perspective transforms, and sub-pixel sampling;
- threshold, LUT, histogram equalization, and colormaps;
- box, Gaussian, separable, median, bilateral, stack, and derivative filters;
- pyramids, Canny, morphology, accumulation, and image blending.
- connected regions, contours, basic shape geometry, histograms, and template matching.

The precise callable-family inventory and parameter subsets are owned by
[`doc/opencv-core-imgproc-api-coverage.md`](../../../doc/opencv-core-imgproc-api-coverage.md).
This module README does not duplicate the full support matrix.

## Public Entry

```cpp
#include <cvh/imgproc/imgproc.h>
```

Top-level `include/cvh/imgproc/*.h` files are public unless the coverage document
marks them as a deferred preview. Implementation helpers in `detail/**` are
internal even though they are installed with the header-only package.

## Implementation Rules

- Scalar behavior is the correctness fallback.
- OpenCV Universal Intrinsics and specialized ISA paths must have narrow
  type/shape/layout predicates.
- Unsupported combinations fail explicitly or use the documented fallback.
- ROI, non-contiguous rows, aliasing, borders, interpolation, and lane tails
  must follow the public operator contract.
- No runtime backend registry or compiled project implementation is permitted.

The kernel migration checklist is
[`doc/opencv-ui-kernel-migration-checklist.md`](../../../doc/opencv-ui-kernel-migration-checklist.md).

## Validation

- `test/imgproc/`: public operator contracts and internal dispatch tests.
- `test/smoke/imgproc_headers/`: independent compilation of every top-level
  Imgproc public header.
- `cvh_imgproc_header_odr_smoke`: multi-translation-unit inline/telemetry
  safety.
- `cvh_benchmark_imgproc_header`: canonical internal performance suite.

New operators require accepted parameter documentation, public correctness
coverage, fallback coverage, and benchmark evidence before a performance claim.

## v0.1 Support Matrix

| Area | Accepted subset |
|---|---|
| Connected components | U8C1 binary input, 4/8 connectivity, S32 labels; stats are S32 and centroids F64. |
| Contours | Non-mutating U8C1 input; `RETR_EXTERNAL`/`RETR_LIST`, `CHAIN_APPROX_NONE`/`CHAIN_APPROX_SIMPLE`, optional offset; no hierarchy output. |
| Shapes | Integer/float point vectors for `boundingRect`, `contourArea`, `arcLength`, `approxPolyDP`, point-returning `convexHull`, `isContourConvex`, and contour `moments`; non-finite coordinates and unrepresentable rectangle extents are rejected; raster `moments` accepts U8C1. |
| Histogram | One U8/F32 2D C1/C3/C4 image and selected channel; 1D uniform dense F32 bins, optional U8C1 mask and accumulation; four base comparison methods. |
| Template matching | U8/F32 C1 image/template; SQDIFF/SQDIFF_NORMED/CCORR/CCORR_NORMED; direct spatial F32 output. |

All entries above are scalar header implementations; no SIMD fast-path claim is
made without separate benchmark evidence.

`demosaicing` is not part of the v0.1 support matrix. Its callable U8 bilinear
preview, correctness tests and benchmark row are retained for a demand-driven
later batch; the v0.1 RC measured it `11.70x` behind OpenCV.
