# cvh Tutorials

The tutorials explain how a cvh operator evolves from a correct baseline into
an optimized implementation, and how its supported cases are validated and
measured against upstream OpenCV. They are implementation narratives, not a
replacement for the product contracts, tests, or dated benchmark reports.

## Organization

Each tutorial lives in its own directory under `doc/tutorial/` and uses
`README.md` as its primary, human-readable entry point:

```text
doc/tutorial/
├── README.md
└── <tutorial-name>/
    ├── README.md        # Default English entry
    ├── README.zh-CN.md  # Optional Simplified Chinese edition
    ├── diagrams/        # Optional editable specs, Excalidraw, and SVG
    ├── images/     # Optional diagrams, screenshots, and result plots
    ├── code/       # Optional focused or runnable examples
    └── data/       # Optional small data files used by the tutorial
```

Keep all tutorial-specific supporting material in that tutorial's directory.
Omit optional directories when they are not needed. Use relative links so that
the tutorial remains readable both in the repository and on documentation
sites.

## Available Tutorials

- Resize from first principles to near-OpenCV performance:
  [English (default)](how_to_speedup_resize/README.md) |
  [简体中文](how_to_speedup_resize/README.zh-CN.md). Starts from the familiar
  `cv::resize` call, builds nearest-neighbor and bilinear intuition, then
  introduces profiling, fixed-point arithmetic, flat-C3 NEON, dispatch,
  validation, and benchmark evidence one step at a time.
- Canny from edge intuition to a streaming implementation that reaches OpenCV:
  [English (default)](how_to_speedup_canny/README.md) |
  [简体中文](how_to_speedup_canny/README.zh-CN.md). Explains Sobel gradients,
  non-maximum suppression, double thresholds, and hysteresis before showing
  how fused gradients, a three-row magnitude ring, a padded state map, and
  observable dispatch remove full-frame work.

## Content Contract

An operator-optimization tutorial should make the following progression clear:

1. Define the supported operator case and the OpenCV behavior used as the
   correctness and performance reference.
2. Explain the simplest correct cvh implementation, including important data
   layouts, boundaries, numeric behavior, and scalar fallback.
3. Establish a reproducible correctness and performance baseline before
   optimization.
4. Identify the measured bottleneck and explain each optimization step, its
   tradeoffs, and any rejected approaches that provide a useful lesson.
5. Validate the optimized path against upstream OpenCV, including relevant
   tails, unaligned data, ROI or non-contiguous layouts, dispatch modes, and
   non-target fallbacks.
6. Show, with reproducible same-machine measurements, how the performance gap
   changes and whether the scoped case approaches or matches OpenCV.
7. Point readers to the current product source, tests, benchmark case, and
   immutable dated report that support the tutorial.

State the exact data types, channel counts, image sizes, interpolation or
border modes, platform, compiler, build type, thread count, and observed
dispatch/ISA whenever they affect a result. Do not generalize a win from one
case to an entire operator, and do not infer the executed path only from the
build platform.

## Supporting Material

- Prefer small diagrams and focused code excerpts that clarify the explanation.
- Reuse maintained repository tests, benchmarks, and scripts instead of adding
  tutorial-only product targets or duplicate harnesses.
- Mark non-runnable pseudocode clearly. Runnable examples should include their
  build and execution instructions.
- Record performance evidence in the repository's benchmark reports; link to
  it from the tutorial instead of making the tutorial the owner of current
  numbers.
- Credit and link any externally sourced material, and keep generated plots
  traceable to their source data.
