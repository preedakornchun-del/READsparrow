# Counting toolkit contract (before core-loop changes)

NEW package based on frozen READsparrow_v012; old sources/results unchanged.
counting.py, io.py and qc.py retain baseline code. orientation.py adds new modes;
api.py exposes DataFrames; plots.py lazily imports Matplotlib. SciPy becomes an
optional legacy extra, explicitly approved by the user; old inference commands
remain available. The existing optional XLSX extra is retained for compatibility.

Strict explicit forward/reverse with anchor_mismatch=0 dispatches directly to
the unchanged baseline loop. Vectors/categories/non-timing count metrics must
match exactly. Runtime cannot literally match across executions: repeated timing
will measure overhead and variability, including the new DataFrame conversion.

Tolerance: precompute original plus all3*k one-substitution variants for k-base
anchors; never edit spacers or allow anchor indels. First upstream match, first
downstream at or after upstream_end+L, then first matching guide window. Unique
policy retains its bounded span and ambiguity rejection.

Implementation correction after fixture audit: legacy tolerant mode first returns
an already reference-valid STRICT cassette; incidental approximate motifs cannot
displace it. If strict does not assign, scan approximate upstream placements in
position order until the first reference-valid window. Each downstream search
still begins at that placement's end+L. Unique tolerant policy enumerates identities.
The original strict mode and its first-upstream rule remain byte-identical.

Auto: inspect min(20000,max_reads) complete records, choose orientation with more
reference-valid matches. Ties choose forward and warn; zero support or <60%
dominance also warns. Reopen file and count once, with no sniff double counting.
Mixed: forward first, reverse only on failure; at most one assignment/read.
Use two orientation-specific dictionaries with common indices to avoid overwriting
distinct references that happen to be reverse complements. Failed mixed reads
report the furthest supported extraction stage, forward reason on stage ties.

Reference DataFrames retain row order and baseline DNA/ID uniqueness. Duplicate
DNA is never allocated silently. Existing flex/variant group modes remain intact.
QC on plain DataFrames has mapping_rate_percent=None unless total input reads
are supplied or retained in attrs. Representation is not cellular coverage.
Plots return Axes without show/save; abundance is a histogram, avoiding SciPy KDE.

Regression: 19/20nt both orientations/internal downstream motifs/first-window;
anchor substitutions/no spacer rescue; auto ambiguity/max_reads/mixed RC conflicts;
malformed FASTQ, QC arithmetic, plots, existing CLI and isolated pip installs.
Sample RSS/scaling is measured; no claim to have run40Mreads without doing so.
