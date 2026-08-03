# Imgproc Module

## Responsibility

Imgproc provides the accepted OpenCV-style image-processing subset, including:

- color conversion and Bayer/YUV input handling;
- resize, remap, affine/perspective transforms, and sub-pixel sampling;
- threshold, LUT, histogram equalization, and colormaps;
- box, Gaussian, separable, median, bilateral, stack, and derivative filters;
- pyramids, Canny, morphology, accumulation, and image blending.

The precise callable-family inventory and parameter subsets are owned by
[`doc/opencv-core-imgproc-api-coverage.md`](../../../doc/opencv-core-imgproc-api-coverage.md).
This module README does not duplicate the full support matrix.

## Public Entry

```cpp
#include <cvh/imgproc/imgproc.h>
```

Top-level `include/cvh/imgproc/*.h` files are public. Implementation helpers in
`detail/**` are internal even though they are installed with the header-only
package.

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
