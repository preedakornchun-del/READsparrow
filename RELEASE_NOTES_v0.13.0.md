# READsparrow v0.13.0

Version DOI: https://doi.org/10.5281/zenodo.24523414  
All-versions DOI: https://doi.org/10.5281/zenodo.24332350

This release adds a stable Python/Pandas interface for the supplied-adapter
`count-flex` workflow while retaining the anchor-bounded v0.12.1 counter.

## Highlights

- `readsparrow.count_guides(..., mode="count-flex")` for scripts and Jupyter.
- The Python API and CLI call the same streaming counting core.
- Up to two supplied-adapter edits; spacer matching remains exact.
- No downstream anchor is required; orientation must be explicit.
- Duplicate spacer sequences are counted once as sequence groups and are never
  silently allocated to multiple guide IDs.
- Results expose identifiable guide counts, sequence-group counts, a complete
  guide audit, identifiable gene counts, QC metrics and rejection categories.
- Full local regression: 226 tests + 98 subtests; wheel build/install smoke test
  passed under the locked Python 3.11 environment.

## Python example

```python
import readsparrow as rs

result = rs.count_guides(
    "sample.fastq.gz",
    "library.tsv",
    mode="count-flex",
    sample="sample",
    orientation="forward",
    adapter="TCTTGTGGAAAGGACGAAACACCN",
)

print(result.counts_df)
print(result.sequence_counts_df)
print(result.guide_audit_df)
print(result.qc_summary)
```

The stable scope is read counting and descriptive QC. Legacy statistical hit
callers remain experimental and this release makes no biological FDR guarantee.

Benchmark FASTQs, synthetic truth, comparison outputs and reproducibility
materials are archived separately at https://doi.org/10.5281/zenodo.24522621.
