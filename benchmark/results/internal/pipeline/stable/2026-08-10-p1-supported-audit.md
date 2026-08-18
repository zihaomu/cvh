# Pipeline P1 Supported Audit

This report is the final P1 representative performance audit. It rechecks the accepted packed
F32, letterbox, per-tensor U8/S8, NV12, and NV21 predicates in one Release binary without
overwriting the earlier candidate and rollback reports.

## Measurement contract

- Source: `3857147+working-tree-p1-supported-audit`; Release; Clang 21.0.0; Apple M5 / arm64.
- Single thread; warmup 3, iterations 3, repeats 7; median time per frame.
- Staged, forced-scalar, and Auto rows use identical inputs and parameters. Comparable rows must
  have identical checksums.
- Candidate, actual dispatch, and observed ISA are recorded independently. No route is inferred
  from the host architecture.

## Representative accepted predicates

| Predicate | Staged | Scalar fused | Auto | Result |
| --- | ---: | ---: | ---: | --- |
| Packed F32 Linear NCHW, 1280x720 to 640x640 | 1.666 ms | 1.035 ms | 1.036 ms scalar | 1.61x vs staged |
| Packed F32 Linear NHWC, 1280x720 to 640x640 | 1.218 ms | 1.010 ms | 1.015 ms scalar | 1.20x vs staged |
| Packed F32 Nearest NCHW, horizontal 2x | 0.960 ms | 0.363 ms | 0.119 ms NEON | 3.06x vs scalar; 8.09x vs staged |
| Packed F32 letterbox Nearest NCHW, horizontal content 2x | 1.251 ms | 0.602 ms | 0.255 ms NEON | 2.36x vs scalar; 4.90x vs staged |
| Packed U8 Nearest NCHW, 1280x720 to 640x640 | 1.951 ms | 1.456 ms | 1.458 ms scalar | 1.34x vs staged |
| Packed S8 Nearest NCHW, 1280x720 to 640x640 | 1.946 ms | 1.463 ms | 1.463 ms scalar | 1.33x vs staged |

All comparable staged/fused rows have matching checksums. Every fused row above reports one
execution group, zero full-frame intermediates, and zero workspace bytes.

## Two-plane coverage

| Predicate | Scalar fused | Checksum |
| --- | ---: | ---: |
| NV12 BT.709/Limited/Left to RGB F32 Linear NCHW | 10.168 ms | 9366306608337106819 |
| NV21 BT.709/Limited/Left to RGB F32 Linear NCHW | 10.128 ms | 192723119964494961 |
| NV12 BT.709/Limited/Left to RGB S8 Linear NCHW | 12.023 ms | 13423987305786661862 |
| NV21 BT.709/Limited/Left to RGB S8 Linear NCHW | 12.038 ms | 8824021885824421175 |

The two-plane API has no staged `cvh::Mat` representation, so these rows report absolute scalar
cost instead of inventing a staged speedup. They still report one group, no full-frame
intermediate, zero workspace, and an actually observed scalar route.

## Decision

The audited P1 v1 predicates are accepted. Performance claims remain bound to the rows and exact
predicates above; Linear/NHWC, arbitrary resize ratios, YUV, and quantized paths do not inherit the
narrow NEON claim. Per-channel quantization, batch greater than one, non-contiguous tensor output,
and device-memory import remain outside the Supported P1 surface.

Raw evidence: [CSV](2026-08-10-p1-supported-audit.csv) and
[metadata](2026-08-10-p1-supported-audit.meta.json).
