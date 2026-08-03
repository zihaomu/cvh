# Imgcodecs Module

## Responsibility

Imgcodecs provides the file I/O portion of the minimum cvh pipeline:

```text
read image -> Core/Imgproc processing -> write image
```

The implementation uses the vendored stb image headers and does not link an
external codec library.

## Current Surface

```cpp
#include <cvh/imgcodecs/imgcodecs.h>
```

- Decode: PNG, JPEG, BMP, GIF, PPM, and optional HDR coverage represented by
  current tests.
- Encode: PNG, JPEG, and BMP.
- Primary output is an 8-bit cvh `Mat`; behavior is a documented subset rather
  than complete OpenCV Imgcodecs compatibility.

Video codecs, animated-image writing, large-format streaming, and a general
codec plugin system are outside the current scope.

## Validation

- `test/imgcodecs/` owns functional and failure-path tests.
- `test/imgcodecs/data/manifest.json` owns synchronized fixture provenance.
- `test/smoke/imgcodecs_headers/` verifies public-header self-containment.
- `cvh_pipeline_smoke` verifies the read/process/write chain with deterministic
  generated input.

New format claims require fixtures, manifest entries, error-path coverage, and
cross-platform verification.
