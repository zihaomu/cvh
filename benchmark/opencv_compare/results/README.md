# OpenCV Comparison Results

## Current Snapshot

- [v0.1 NEON hot-kernel product-auto English report](2026-08-06-v0.1-neon-hot-opencv-upstream-performance.en.md)
- [v0.1 NEON hot-kernel product-auto raw CSV](2026-08-06-v0.1-neon-hot-opencv-upstream-performance.csv)
- [v0.1 NEON hot-kernel product-auto run metadata](2026-08-06-v0.1-neon-hot-opencv-upstream-performance.meta.json)

This is the current reviewed full comparison. It used product `cvh_auto`
dispatch on Apple M5 with one thread and contains 370 rows. Of these, 369 are
valid and the one expected `UNSUPPORTED` row records that upstream OpenCV has
no single-call BGR-to-NV12 encoder. The clean `adac8bd` cvh snapshot records
geometric means of `0.7901` overall, `0.6434` for Core, and `0.9671` for
Imgproc. Every row records `algorithm_path`, `dispatch_path`, `isa_observed`,
and `kernel_route`; 30 rows selected direct NEON: 10 GEMM, 10 color, 2 resize,
6 Sobel, 1 Scharr, and 1 spatial-gradient case. Against the previous
identical-fingerprint product-auto snapshot, the 334 valid non-target rows
retain `0.9925x` of the normalized OpenCV/CVH geometric mean (a `0.75%`
decrease, within the `1%` gate), while the 151 non-target Imgproc rows improve
to `1.0031x`. The upstream checkout's dirty marker is limited to its
`.gitignore`; the compared OpenCV source commit is `d48bf69`.

## Historical Snapshots

- [2026-08-06 pre-hot-kernel v0.1 RC product-auto report](2026-08-06-v0.1-rc-auto-opencv-upstream-performance.en.md),
  [CSV](2026-08-06-v0.1-rc-auto-opencv-upstream-performance.csv), and
  [metadata](2026-08-06-v0.1-rc-auto-opencv-upstream-performance.meta.json):
  clean `cbd5076` baseline used for the final non-target regression comparison.
- [2026-08-04 v0.1 RC forced-UI English report](2026-08-04-v0.1-rc-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-v0.1-rc-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-v0.1-rc-opencv-upstream-performance.meta.json): this
  diagnostic snapshot excluded direct NEON/AVX2 dispatch and must not be used
  to infer product-auto GEMM performance.
- [2026-08-04 pre-closure English report](2026-08-04-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-opencv-upstream-performance.meta.json)
- [2026-07-25 English report](2026-07-25-opencv-upstream-performance.en.md)
  and its [Chinese translation](2026-07-25-opencv-upstream-performance.md)
- [2026-07-24 report](2026-07-24-opencv-upstream-performance.md)
- [2026-07-23 report](2026-07-23-opencv-upstream-performance.md)

Historical files record the configuration that produced them and are not
rewritten to match later target names or policies.

## Phase 2-P0 Acceleration Snapshots

- [Frozen stable baseline](2026-08-04-phase2-p0-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-opencv-upstream-performance.meta.json):
  26 focused rows before the second-stage operator acceleration.
- [A1 matchTemplate result](2026-08-04-phase2-p0-a1-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a1-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a1-opencv-upstream-performance.meta.json):
  stable evidence after method specialization, squared-window integrals, and
  Universal Intrinsics correlation kernels.
- [A2.1 connected-components result](2026-08-04-phase2-p0-a2-1-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a2-1-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a2-1-opencv-upstream-performance.meta.json):
  stable evidence after row-pointer scanning, compact union-find labels, and
  fused canonical relabel/statistics accumulation.
- [A2 connected-components and contours result](2026-08-04-phase2-p0-a2-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a2-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a2-opencv-upstream-performance.meta.json):
  stable evidence after completing the contour workspace and retrieval-mode
  scan split; this is the accepted A2 snapshot. The full canonical non-target
  noise adjudication is recorded in the implementation plan.
- [A3 point-transform result](2026-08-04-phase2-p0-a3-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a3-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a3-opencv-upstream-performance.meta.json):
  stable evidence after coefficient prepacking, source-channel specialization,
  and continuous point-span traversal.
- [A4 histogram result](2026-08-04-phase2-p0-a4-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a4-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a4-opencv-upstream-performance.meta.json):
  stable evidence after typed row scans, the U8 value-to-bin lookup table,
  local accumulation, and method-specialized contiguous reductions.
- [A5 random-fill result](2026-08-04-phase2-p0-a5-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a5-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a5-opencv-upstream-performance.meta.json):
  stable evidence after the lightweight internal 64-bit engine, hoisted
  distributions, channel-unrolled spans, and typed ROI rows.
- [A6 shape measurement](2026-08-04-phase2-p0-a6-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a6-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a6-opencv-upstream-performance.meta.json):
  stable evidence for the measured-no-optimize decision; candidate shape
  changes below the 20% retention threshold were reverted.
- [A7 final local result](2026-08-04-phase2-p0-a7-opencv-upstream-performance.en.md),
  [CSV](2026-08-04-phase2-p0-a7-opencv-upstream-performance.csv), and
  [metadata](2026-08-04-phase2-p0-a7-opencv-upstream-performance.meta.json):
  final 26/26 stable focused evidence after all local correctness and canonical
  regression gates; Linux x86-64 runtime evidence remains pending.

These focused snapshots use the `PHASE2_P0` filter and do not replace the
full-matrix current snapshot above.

## Retention Rule

New canonical snapshots are written in English and commit the Markdown report,
raw CSV, and metadata together. Rolling `current_*` outputs and unreviewed local
runs remain generated artifacts.
