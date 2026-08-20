# Apple M5 Pipeline sampling attempts excluded from conclusions

The following files are retained rather than overwritten, but are excluded from
the aggregate reports:

- `2026-08-18-apple-m5-session-1.csv`: initial 10-frames/sample run; several
  streaming rows exceeded the frozen 3% CV limit.
- `2026-08-18-apple-m5-session-1-rerun-1.csv`: same parameters; instability
  reproduced.
- `2026-08-18-apple-m5-session-1-iters50.csv`: longer 50-frames/sample window,
  before explicit output-ring page pre-touch; first streaming samples carried
  demand-zero page costs.
- `2026-08-18-apple-m5-aggregate.csv` and
  `2026-08-18-apple-m5-report.md`: first reporter output, superseded by the
  non-overwriting `aggregate-v2` / `report-v2` correction that also reports L2.
- `2026-08-18-apple-m5-aggregate-lut.csv` and
  `2026-08-18-apple-m5-report-lut.md`: first candidate reporter output,
  superseded by the labelled `aggregate-lut-v2` / `report-lut-v2` correction.
- `2026-08-18-apple-m5-aggregate-lut-v2.csv` / `report-lut-v2.md` and the
  first `aggregate-lut-consumed` / `report-lut-consumed` pair are superseded by
  `aggregate-lut-consumed-v2` / `report-lut-consumed-v2`, which consume every
  timed sample output and report the passing PF5 hot predicate explicitly.

No tolerance, case, repeat count, or performance gate was changed in response
to these failures.
