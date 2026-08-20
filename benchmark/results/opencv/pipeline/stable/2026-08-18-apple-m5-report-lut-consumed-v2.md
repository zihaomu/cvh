# cvh Pipeline Apple M5 normalize-u8-lut-v1 final consumed-output candidate

Status: device-bound evidence; no PF1-PF6 family or L4 claim is authorized.

## Method

- Sessions: 1, 2, 3; single-thread Release; warmup 20; 50 frames/sample; 15 samples.
- Cache modes: hot and a 64 MiB input/output ring streaming mode.
- Each CSV passed the independent scalar-oracle validation before timing.
- Stability gate: CV <= 3%; 54 of 144 implementation rows exceeded it.
- Inputs:
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-lut-consumed-session-1.csv`
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-lut-consumed-session-2.csv`
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-lut-consumed-session-3.csv`

## Predicate results

| Case | Cache | Session speedups vs OpenCV | Geomean | 95% CI | Stable sessions | Route/ISA | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| PF1 | hot | 0.297x, 0.295x, 0.296x | 0.296x | [0.295x, 0.297x] | 2/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF1 | streaming | 0.296x, 0.294x, 0.291x | 0.294x | [0.291x, 0.296x] | 2/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF2 | hot | 0.313x, 0.306x, 0.314x | 0.311x | [0.306x, 0.314x] | 0/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF2 | streaming | 0.304x, 0.305x, 0.316x | 0.308x | [0.304x, 0.316x] | 1/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF3 | hot | 0.309x, 0.310x, 0.314x | 0.311x | [0.309x, 0.314x] | 2/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF3 | streaming | 0.314x, 0.323x, 0.326x | 0.321x | [0.314x, 0.326x] | 0/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF4 | hot | 0.372x, 0.373x, 0.372x | 0.372x | [0.372x, 0.373x] | 1/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF4 | streaming | 0.360x, 0.367x, 0.377x | 0.368x | [0.360x, 0.377x] | 2/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF5 | hot | 2.678x, 2.715x, 2.685x | 2.693x | [2.678x, 2.715x] | 3/3 | neon/neon | pass: pass |
| PF5 | streaming | 2.623x, 2.582x, 2.628x | 2.611x | [2.582x, 2.628x] | 1/3 | neon/neon | fail: CV gate |
| PF6 | hot | 0.968x, 0.971x, 0.962x | 0.967x | [0.962x, 0.971x] | 1/3 | neon/neon | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF6 | streaming | 0.880x, 0.884x, 0.889x | 0.885x | [0.880x, 0.889x] | 1/3 | neon/neon | fail: CV gate; median < 1.20x; CI lower < 1.10x |

## Family result

| Cache | PF1-PF6 vs OpenCV geomean | Session range | Fused vs staged geomean | Stable | No case < 0.95x | Gate |
| --- | ---: | ---: | ---: | --- | --- | --- |
| hot | 0.550x | 0.549x–0.551x | 2.508x | no | no | fail |
| streaming | 0.540x | 0.535x–0.546x | 2.417x | no | no | fail |

## Structural evidence

- L1 fused structure gate: pass; all fused rows report one execution group, zero full-frame intermediates, zero workspace bytes, and zero planned allocations/run.
- OpenCV explicit temporary storage range: 301,056–6,144,000 bytes, excluding caller-owned output.

## Decision

- L1 is supported on this build. L2 remains supported for fusion over the cvh staged chain, subject to each row's stability marker.
- The PF1-PF6 family does not meet the OpenCV gate. PF1-PF4 Linear predicates are materially slower than the optimized OpenCV chain.
- Exact device-bound predicates passing all frozen gates: PF5 hot (2.693x, CI lower 2.678x). This does not authorize a family or edge-device claim.
- PF6 is near parity in hot mode and slower in streaming mode. No edge-device claim is permitted without two ARM Linux devices.
