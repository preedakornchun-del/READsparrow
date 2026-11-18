# Full-reference and short-read counting in v0.11

`count-flex` is an explicit, experimental extraction command. Existing `count`
and hit-calling commands keep v0.10 behavior. Do not assume `run` automatically
uses this new counting mode or that sequence groups are valid independent guides.

Install the local wheel (dependencies are NumPy and SciPy; optional Excel input
also needs openpyxl), or run the source without changing an existing install:

```sh
PYTHONPATH=/absolute/path/READsparrow_v011/src python -m readsparrow count-flex \
  --library full_reference.tsv --fastq sample.fastq.gz --sample sample \
  --orientation forward --adapter TCTTGTGGAAAGGACGAAACACCN \
  --max-adapter-edits 2 --output a_new_directory
```

For reverse-complemented reads use `--orientation reverse` with the **same
forward adapter**. Orientation must be known and explicit. Input FASTQ is
four-line FASTQ, optionally gzip compressed; A/C/G/T/N only. Quality lengths
are checked, but quality values are not used to correct bases.

The reference must have nonempty unique IDs and gene annotations, but DNA
sequences may repeat. All spacers must be the same positive length and contain
only A/C/G/T. Column aliases include `sgRNA/Gene/Sequence`, `ID/gene/seq`, and
the original PinAPL-Py `UID/gene_id/seq` format. TSV/CSV and optional XLSX are
supported. Duplicate IDs are still rejected; they are not the same as repeated
DNA sequences under distinct IDs.

## Extraction rule

The complete upstream adapter is matched with 0–2 Levenshtein edits, including
substitutions and indels. Adapter N positions are wildcards. At least 12
informative adapter bases are required. Exact placements are preferred; if no
exact placement exists, a literal-seed search and bounded edit-distance dynamic
program find the minimum-cost full-adapter placement. Wildcard-heavy adapters
fall back to exhaustive starts instead of silently discarding candidates.

Tied best endpoints are refused, even if one happens to lead to a reference
spacer. The next full spacer-length segment must match a reference DNA sequence
exactly. No downstream G/GTTTT scaffold base is required. Thus a 50-bp read can
be used when it actually contains the full adapter and full spacer; the command
does not reconstruct missing spacer bases or guarantee every 50-bp architecture
is recoverable. Partial upstream adapters, spacer mismatches, UMI extraction and
automatic strand inference are not supported.

## Repeated sequences are measured, not arbitrarily deconvolved

Suppose IDs A and B both have exactly the same 20-nt spacer. Twelve compatible
reads support one DNA-sequence measurement of 12. They do not establish A=12,
B=12, A=6/B=6, or any other ID allocation.

- `sequence_counts.tsv`: one row per distinct DNA sequence, with its full
  SHA256-derived group ID, member IDs/genes and one count. This is the complete
  measurement table across all distinct reference sequences.
- `guide_audit.tsv`: all original ID rows, in original order. Singleton-ID
  `resolved_count` is an integer; ambiguous-ID `resolved_count` is blank.
  `compatible_group_reads` is explanatory context repeated for members: **never
  sum this column over guide IDs**.
- `resolved_counts.tsv`: singleton-ID count matrix only. Excluded ambiguous
  measurements are not zero measurements. This output subset is not trimming
  the reference used to recognize reads: every reference sequence was indexed.
- `identifiable_gene_counts.tsv`: one contribution per sequence group when all
  its member IDs share one gene; cross-gene groups are not allocated. These are
  descriptive abundance sums, not independent-guide evidence or calibrated hits.
- `count_metrics.json`: total, identifiable-ID, duplicate-group reads; refusal
  reasons, adapter edit counts, reference membership and counting configuration.
- `provenance.json`: package source identities and input hashes, from the
  unchanged provenance machinery.

Sequence-group files deliberately lack the ordinary guide-count schema and
are rejected by existing guide-level readers. Integrating their uncertainty
into gene-level inference requires separate modelling and validation. The
resolved-ID subset may be exported for further analysis, but its missing-data
and guide-family implications must be evaluated before biological hit calling.

## Evidence and limits

The benchmark protocol and results are in the sibling `benchmark_pinapl_v011`
folder. The full six-file PinAPL-Py demonstration dataset is used; H1/H2 are not
used. Primary timings repeat both counting components five times sequentially
per file, using the same Python runtime, full reference and external CPU timer.
The PinAPL-Py compatibility score floor and warm-index scope are explicit in
that protocol. More assigned reads on real data is not known-origin accuracy.
Engineering truth tests assess extraction only, not biological FDR/sensitivity.

No v0.10 source/results were replaced and no hit-calling default was promoted.
