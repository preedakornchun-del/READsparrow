from pathlib import Path
import readsparrow as rs

base = Path(__file__).resolve().parent
result = rs.count(base / "reads.fastq", base / "reference.tsv")
print(result.counts_df)
print(result.qc_summary)
# Optional: ax = rs.plots.abundance_distribution(result.counts_df)
