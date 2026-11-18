"""Descriptive QC only: library representation is not cellular coverage."""
import numpy as np


def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    if x.sum() == 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum()) - (len(x) + 1) / len(x))


def summarize(table):
    result = []
    for sample, x in zip(table.samples, table.counts.T):
        result.append({"sample": sample, "guides": len(x), "assigned_reads": int(x.sum()),
                       "zero_guides": int((x == 0).sum()), "representation_percent": 100 * float((x > 0).mean()),
                       "mean_count": float(x.mean()), "median_count": float(np.median(x)),
                       "raw_count_gini": gini(x), "log1p_count_gini": gini(np.log1p(x))})
    return result
