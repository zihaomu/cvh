# `include/cvh` Public Header Surface

## Purpose

This directory is the installed `cvh` C++17 header surface.

## Modules

- `cvh.h`: aggregate entry for Core, Imgproc, and Imgcodecs.
- `core/`: Mat, basic types, arithmetic, reductions, layout, parallel runtime,
  and GEMM.
- `imgproc/`: color, filtering, geometry, intensity, morphology, and feature
  primitives.
- `imgcodecs/`: image file decode and encode.
- `highgui/`: optional window and event API; intentionally excluded from the
  aggregate header.
- `detail/`: cross-module internal configuration.
- `3rdparty/`: audited vendored header dependencies.

Use the aggregate compute entry with:

```cpp
#include <cvh/cvh.h>
```

Use HighGUI explicitly with:

```cpp
#include <cvh/highgui/highgui.h>
```

## Public Boundary

Top-level module `.h` files are the supported include surface. `detail/**`,
`simd/**`, implementation `.hpp`, and `.inl.h` files are installed because the
project is header-only, but they are not source-compatibility promises.

Every new public header must be registered in the per-header compile smoke.
Public code must not depend on repository source or test paths.
