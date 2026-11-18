"""Diagnostics, not adaptive significance filters or biological bias correction."""
import numpy as np

from .inference import size_factors


def normalization_audit(table, control_mask, chosen):
    candidates, unavailable = {}, {}
    for method in ("median", "total", "control"):
        try:
            f, _ = size_factors(table.counts, method, control_mask)
            candidates[method] = f
        except ValueError as exc:
            unavailable[method] = str(exc)
    comparisons = {}
    reference = candidates.get(chosen, np.ones(len(table.samples)))
    for method, f in candidates.items():
        diff = np.log2(f / reference)
        comparisons[method] = {"factors": dict(zip(table.samples, map(float, f))),
                               "max_absolute_log2_factor_difference": float(np.abs(diff).max())}
    return {"chosen_method": chosen, "alternatives": comparisons, "unavailable": unavailable,
            "automatic_method_selection": False,
            "interpretation": "Factor disagreement is a warning, not proof of which normalization is biologically correct"}
