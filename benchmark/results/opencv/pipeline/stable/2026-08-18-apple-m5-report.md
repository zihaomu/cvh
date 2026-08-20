# cvh Pipeline Apple M5 stable performance proof

Status: exploratory evidence; no broad L3/L4 claim is authorized.

## Method

- Sessions: 1, 2, 3; single-thread Release; warmup 20; 50 frames/sample; 15 samples.
- Cache modes: hot and a 64 MiB input/output ring streaming mode.
- Each CSV passed the independent scalar-oracle validation before timing.
- Stability gate: CV <= 3%; 54 of 144 implementation rows exceeded it.
- Inputs:
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-session-1-pretouch.csv`
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-session-2-pretouch.csv`
  - `benchmark/results/opencv/pipeline/stable/2026-08-18-apple-m5-session-3-pretouch.csv`

## Predicate results

| Case | Cache | Session speedups vs OpenCV | Geomean | 95% CI | Stable sessions | Route/ISA | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| PF1 | hot | 0.256x, 0.255x, 0.256x | 0.256x | [0.255x, 0.256x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF1 | streaming | 0.257x, 0.258x, 0.258x | 0.258x | [0.257x, 0.258x] | 1/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF2 | hot | 0.271x, 0.271x, 0.271x | 0.271x | [0.271x, 0.271x] | 2/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF2 | streaming | 0.274x, 0.276x, 0.269x | 0.273x | [0.269x, 0.276x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF3 | hot | 0.273x, 0.271x, 0.273x | 0.272x | [0.271x, 0.273x] | 3/3 | scalar/scalar | fail: median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF3 | streaming | 0.279x, 0.286x, 0.284x | 0.283x | [0.279x, 0.286x] | 0/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF4 | hot | 0.328x, 0.325x, 0.319x | 0.324x | [0.319x, 0.328x] | 2/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF4 | streaming | 0.329x, 0.325x, 0.324x | 0.326x | [0.324x, 0.329x] | 0/3 | scalar/scalar | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF5 | hot | 2.688x, 2.677x, 2.704x | 2.690x | [2.677x, 2.704x] | 1/3 | neon/neon | fail: CV gate |
| PF5 | streaming | 2.563x, 2.600x, 2.540x | 2.568x | [2.540x, 2.600x] | 0/3 | neon/neon | fail: CV gate |
| PF6 | hot | 0.940x, 0.961x, 0.961x | 0.954x | [0.940x, 0.961x] | 1/3 | neon/neon | fail: CV gate; median < 1.20x; CI lower < 1.10x; other cache mode < 0.90x |
| PF6 | streaming | 0.890x, 0.873x, 0.897x | 0.887x | [0.873x, 0.897x] | 1/3 | neon/neon | fail: CV gate; median < 1.20x; CI lower < 1.10x |

## Family result

| Cache | PF1-PF6 geomean | Session range | Stable | No case < 0.95x | Gate |
| --- | ---: | ---: | --- | --- | --- |
| hot | 0.500x | 0.500x–0.500x | no | no | fail |
| streaming | 0.495x | 0.494x–0.497x | no | no | fail |

## Structural evidence

- L1 fused structure gate: pass; all fused rows report one execution group, zero full-frame intermediates, zero workspace bytes, and zero planned allocations/run.
- OpenCV explicit temporary storage range: 301,056–6,144,000 bytes, excluding caller-owned output.

## Decision

- L1 is supported on this build. L2 remains supported for fusion over the cvh staged chain, subject to each row's stability marker.
- The PF1-PF6 family does not meet the OpenCV gate. PF1-PF4 Linear predicates are materially slower than the optimized OpenCV chain.
- PF5 shows a strong NEON Nearest signal, but the formal predicate claim is withheld whenever any required session fails the 3% CV gate.
- PF6 is near parity in hot mode and slower in streaming mode. No edge-device claim is permitted without E4 and two ARM Linux devices.
