# Changelog

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
