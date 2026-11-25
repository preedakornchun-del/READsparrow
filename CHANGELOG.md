# Changelog

## 0.13.0 — 2026-11-25

Extends the stable counting/QC scope with the supplied-adapter `count-flex`
workflow and a public Python/Pandas API. `readsparrow.count_guides()` dispatches
to either the new `count-flex` interface or the existing anchor-bounded counter,
so notebooks and the CLI use the same streaming counting cores.

`count-flex` permits up to two edits in the supplied full upstream adapter,
requires an exact reference spacer, does not require a downstream anchor and
requires an explicit read orientation. Duplicate spacer sequences are measured
once as sequence groups. The API exposes identifiable guide counts, complete
guide audits, sequence-group counts and identifiable gene counts without
copying one observed group count to multiple guide IDs.

Added API/CLI table-equivalence tests, duplicate-spacer safety tests, input
validation, Jupyter usage documentation and a wheel-install smoke test. The
full local regression result is 226 tests plus 98 subtests. Legacy statistical
hit callers remain experimental; no biological FDR certification is implied.

## 0.12.1 — 2026-11-18

Stable release scope: read counting and descriptive QC. The counting algorithms
are unchanged from the tested0.12.1a1 toolkit. Source-level preservation records
verify that the runtime change is the version string only. Added MIT licence,
citation/archive metadata, portable small fixtures, expected counts/QC/figures,
reproduction commands, environment lock and CI.

The alpha added a Pandas API, auto/mixed orientation, opt-in anchor substitutions,
and QC plotting while retaining the original0.12 strict loop. Its anchor-tolerance
bug and correction remain documented in `docs/VALIDATION.md`. Legacy statistical
hit calling is still experimental; no inference/FDR promotion is implied.
