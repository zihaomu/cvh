# cvh Pipeline Apple M5 stable performance proof

Status: exploratory evidence; no broad L3/L4 claim is authorized.

## Method

- Sessions: 1, 2, 3; single-thread Release; warmup 20; 50 frames/sample; 15 samples.
- Cache modes: hot and a 64 MiB input/output ring streaming mode.
- Each CSV passed the independent scalar-oracle validation before timing.
- Stability gate: CV <= 3%; 58 of 144 implementation rows exceeded it.
- Inputs:
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-lut-session-1.csv`
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-lut-session-2.csv`
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-lut-session-3.csv`

## Predicate results

| Case | Cache | Session speedups vs OpenCV | Geomean | 95% CI | Stable sessions | Route/ISA | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| PF1 | hot | 0.296x, 0.294x, 0.296x | 0.295x | [0.294x, 0.296x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF1 | streaming | 0.295x, 0.291x, 0.286x | 0.291x | [0.286x, 0.295x] | 1/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF2 | hot | 0.304x, 0.316x, 0.313x | 0.311x | [0.304x, 0.316x] | 1/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF2 | streaming | 0.308x, 0.305x, 0.303x | 0.305x | [0.303x, 0.308x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF3 | hot | 0.315x, 0.315x, 0.306x | 0.312x | [0.306x, 0.315x] | 1/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF3 | streaming | 0.332x, 0.317x, 0.314x | 0.321x | [0.314x, 0.332x] | 0/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF4 | hot | 0.373x, 0.373x, 0.372x | 0.373x | [0.372x, 0.373x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF4 | streaming | 0.367x, 0.365x, 0.367x | 0.366x | [0.365x, 0.367x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF5 | hot | 2.642x, 2.677x, 2.673x | 2.664x | [2.642x, 2.677x] | 1/3 | neon/neon | fail: CV gate |
| PF5 | streaming | 2.561x, 2.579x, 2.584x | 2.575x | [2.561x, 2.584x] | 0/3 | neon/neon | fail: CV gate |
| PF6 | hot | 0.973x, 0.966x, 0.965x | 0.968x | [0.965x, 0.973x] | 0/3 | neon/neon | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF6 | streaming | 0.886x, 0.901x, 0.899x | 0.895x | [0.886x, 0.901x] | 1/3 | neon/neon | fail: CV gate; median < 1.20x; CI lower < 1.10x |

## Family result

| Cache | PF1-PF6 vs OpenCV geomean | Session range | Fused vs staged geomean | Stable | No case < 0.95x | Gate |
| --- | ---: | ---: | ---: | --- | --- | --- |
| hot | 0.550x | 0.548x–0.552x | 2.489x | no | no | fail |
| streaming | 0.537x | 0.534x–0.541x | 2.392x | no | no | fail |

## Structural evidence

- L1 fused structure gate: pass; all fused rows report one execution group, zero full-frame intermediates, zero workspace bytes, and zero planned allocations/run.
- OpenCV explicit temporary storage range: 301,056–6,144,000 bytes, excluding caller-owned output.

## Decision

- L1 is supported on this build. L2 remains supported for fusion over the cvh staged chain, subject to each row's stability marker.
- The PF1-PF6 family does not meet the OpenCV gate. PF1-PF4 Linear predicates are materially slower than the optimized OpenCV chain.
- PF5 shows a strong NEON Nearest signal, but the formal predicate claim is withheld whenever any required session fails the 3% CV gate.
- PF6 is near parity in hot mode and slower in streaming mode. No edge-device claim is permitted without E4 and two ARM Linux devices.
