# Validation and scope

Counting/QC 0.13.0 retains the tested anchor-bounded v0.12.1 workflow and extends
the stable scope with supplied-adapter `count-flex` and its Python API. The API
and CLI call the same streaming counting core. The original alpha and historical
code/results/failed attempts remain preserved separately.

Local regression: 226 test items and 98 subtests pass.
An independent Hamming-scan oracle checks600constructed reads. Covered cases
include19/20nt guides, both orientations, internal downstream motifs, first-window
rules, anchor substitutions without spacer rescue, mixed reverse-complement
collisions/single assignment, malformed inputs, QC arithmetic and plotting.
The six portable tiny examples match independent constructed counts/QC.

The `count-flex` API tests independently check adapter-led exact-spacer counts,
duplicate-spacer non-allocation, complete reference audits, input validation and
row-level equivalence between all four API DataFrames and the CLI TSV outputs.

Historical alpha engineering results (one environment; no universal speed claim):
100kreads/1000guides, three fresh-process repeats/mode, median CPU seconds:
baseline0.372326; strict0.367889; DataFrameAPI0.652161; auto0.687763;
tolerant0.686060; mixed0.655448. Strict counts/rejection categories and non-timing
metrics match baseline. Small timing differences are descriptive.

The actual previous40M-record run used1000guides and80copies of a500kfixture:
32Massigned; peak counting-process RSS65.0195MiB; CPU146.30s; elapsed147.22s.
This is an engineering streaming check, not biological/FDR validation and not a
genome-wide-reference benchmark. The memory claim excludes legacy variant-audit
modes that retain distinct observed variant strings.

Environment of those historical measurements: Python3.11.16/x86_64,
NumPy2.4.6/Pandas3.1.1/Matplotlib3.11.2, MacBookAirM1/8GiB/macOS26.5.2.
No native-ARM speed claim. The package lock reproduces dependency versions;
hardware/OS and rendering can still affect numerical timing/figure bytes.

During alpha development the initial tolerant implementation let an incidental
approximate flank motif displace a valid strict cassette. It was corrected to
prioritize an already reference-valid strict cassette, then scan approximate
placements for the first reference-valid window. The original strict loop stayed
unchanged. Initial54300/80000assignment failures and corrected27-unit results
remain historical evidence; failed attempts were not pooled with qualified results.

The term stable applies to counting/descriptiveQC. Legacy hit-calling commands
remain experimental with optional SciPy; v17experimental/v18FAILED-UNPROMOTED
and the original FDR/power/independent-study limitations remain unchanged.
QC representation is guide-level representation, not cellular coverage.
