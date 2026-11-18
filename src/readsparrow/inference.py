"""Experimental trend-moderated log-count baseline, NOT a new NB/RRA method.

Counts are summed within genes BEFORE statistical modeling. Columns (biological
samples), not guides, determine the residual degrees of freedom. OLS on log2
normalized gene counts and empirical-Bayes residual-variance moderation follow
the scaled-F framework of Smyth (2004). Our binned abundance trend is an original
simple implementation, NOT a port of limma or a claim of numerical equivalence.
This baseline is not guide-efficacy-aware or outlier-robust.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, polygamma
from scipy.stats import t as student_t

from .io import CountTable, read_rows

CONTROL_GENES = {"NA", "na", "NTC", "Non-Targeting Control"}


@dataclass
class TestConfig:
    normalization: str = "median"
    min_count: int = 10
    min_guides: int = 2
    pseudocount: float = 0.5
    fdr: float = 0.05
    confidence: float = 0.95
    prior_df_cap: float = 100.0

    def validate(self):
        if self.normalization not in {"median", "total", "control", "none"}:
            raise ValueError("Unknown normalization")
        if self.min_count < 1 or self.min_guides < 1:
            raise ValueError("Positive integer filtering thresholds required")
        if not np.isfinite(self.pseudocount) or self.pseudocount <= 0:
            raise ValueError("Pseudocount must be finite and positive")
        if not 0 < self.fdr < 1 or not 0 < self.confidence < 1:
            raise ValueError("FDR and confidence must be between zero and one")
        if not np.isfinite(self.prior_df_cap) or self.prior_df_cap <= 0:
            raise ValueError("prior_df_cap must be positive")


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("BH expects a finite probability vector")
    if not len(p):
        return p.copy()
    order = np.argsort(p, kind="stable")
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1)
    return result


def make_design(samples, metadata, treatment, control):
    if treatment == control:
        raise ValueError("Treatment and control must differ")
    if len(metadata) != len(samples) or {r.get("sample") for r in metadata} != set(samples):
        raise ValueError("Sample sheet must identify each selected count column exactly once")
    lookup = {r["sample"]: r for r in metadata}
    if len(lookup) != len(metadata):
        raise ValueError("Duplicate sample IDs in sample sheet")
    rows = [lookup[s] for s in samples]
    if any(r.get("condition") not in {treatment, control} for r in rows):
        raise ValueError("Select only the two contrasted conditions")
    # Explicit IDs are mandatory: do not infer independence from unique FASTQ names.
    bio = [r.get("biological_replicate", "").strip() for r in rows]
    if any(not x for x in bio) or len(set(bio)) != len(bio):
        raise ValueError("Provide globally unique biological_replicate IDs; merge technical replicates first")
    is_treated = np.array([r["condition"] == treatment for r in rows])
    if min(is_treated.sum(), (~is_treated).sum()) < 2:
        raise ValueError("At least two biological replicates per condition required")
    cols = [np.ones(len(rows)), is_treated.astype(float)]
    names = ["intercept", f"{treatment}_vs_{control}"]
    for field in ("pair", "batch"):
        values = [r.get(field, "").strip() for r in rows]
        if any(values) and not all(values):
            raise ValueError(f"{field} must be complete or absent")
        if not any(values):
            continue
        levels = sorted(set(values))
        if field == "pair":
            for level in levels:
                indices = [i for i, v in enumerate(values) if v == level]
                if len(indices) != 2 or {rows[i]["condition"] for i in indices} != {treatment, control}:
                    raise ValueError("Each pair needs exactly one distinct biological sample per condition")
        for level in levels[1:]:
            cols.append(np.array([v == level for v in values], dtype=float))
            names.append(f"{field}:{level}")
    design = np.column_stack(cols)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Design is rank deficient: condition/pair/batch confounding or redundant covariates")
    df = len(rows) - design.shape[1]
    if df < 2:
        raise ValueError("Need at least two residual degrees of freedom (paired designs need >=3 pairs)")
    warnings = []
    if min(is_treated.sum(), (~is_treated).sum()) == 2:
        warnings.append("Only two biological replicates in a group: inference and outlier diagnosis are limited")
    return design, names, is_treated, df, warnings


def load_sample_sheet(path):
    header, rows = read_rows(path)
    if not {"sample", "condition", "biological_replicate"}.issubset(header):
        raise ValueError("Sample sheet needs sample, condition, biological_replicate columns")
    return rows


def size_factors(counts, method="median", control_mask=None):
    x = np.asarray(counts, dtype=float)
    if np.any(x.sum(axis=0) <= 0):
        raise ValueError("Empty sample cannot be normalized")
    warnings = []
    if method == "none":
        factors = np.ones(x.shape[1])
    elif method == "total":
        factors = x.sum(axis=0)
        warnings.append("Total-count normalization is sensitive to compositional shifts")
    elif method in {"median", "control"}:
        mask = np.all(x > 0, axis=1)
        if method == "control":
            if control_mask is None or np.sum(control_mask) < 20:
                raise ValueError("Control normalization requires >=20 declared control guides")
            mask &= control_mask
        if mask.sum() < 20:
            raise ValueError("Fewer than 20 all-positive guides for median-ratio normalization; no silent fallback")
        log_x = np.log(x[mask])
        factors = np.exp(np.median(log_x - log_x.mean(axis=1, keepdims=True), axis=0))
        if method == "control" and mask.sum() < 100:
            warnings.append("Fewer than 100 all-positive controls: normalization may be unstable")
    else:
        raise ValueError("Unknown normalization")
    factors = factors / np.exp(np.mean(np.log(factors)))
    if factors.max() / factors.min() > 3:
        warnings.append("Normalized size-factor range exceeds 3-fold; log-count baseline may be unreliable")
    return factors, warnings


def moderate_variances(variances, abundance, df, prior_df_cap=100.0):
    """Scaled-F log moment estimates with a prespecified, binned linear trend.

    Trend-bin means are used, not treatment-effect-selected features. Residual
    log-variance sampling variance is corrected for the number of fitted bins.
    A prior-df cap limits overconfidence under an estimated near-constant prior.
    Degenerate variances are floored for log moments, not removed after testing.
    """
    variances, abundance = np.asarray(variances), np.asarray(abundance)
    if len(variances) < 20 or df < 2:
        raise ValueError("Variance moderation needs >=20 tested genes and >=2 residual df")
    floor = max(1e-8, float(np.median(variances)) * 1e-8)
    e = np.log(np.maximum(variances, floor)) - digamma(df / 2) + np.log(df / 2)
    bins = min(20, max(1, len(e) // 100))
    order = np.argsort(abundance, kind="stable")
    groups = np.array_split(order, bins)
    knots = np.array([abundance[g].mean() for g in groups])
    levels = np.array([e[g].mean() for g in groups])
    # Collapse repeated knots instead of depending on undefined interpolation behavior.
    unique = np.unique(knots)
    levels = np.array([levels[knots == k].mean() for k in unique])
    trend = np.interp(abundance, unique, levels)
    resid = e - trend
    excess = np.dot(resid, resid) / max(1, len(e) - len(unique)) - polygamma(1, df / 2)
    if excess <= polygamma(1, prior_df_cap / 2):
        prior_df = float(prior_df_cap)
    else:
        prior_df = float(2 * brentq(lambda z: polygamma(1, z) - excess, 1e-8, prior_df_cap / 2))
    prior_variance = np.exp(trend + digamma(prior_df / 2) - np.log(prior_df / 2))
    posterior = (prior_df * prior_variance + df * variances) / (prior_df + df)
    return np.maximum(posterior, floor), {"prior_df": prior_df, "trend_bins": len(unique),
                                          "variance_floor": floor, "zero_residual_variances": int((variances <= floor).sum()),
                                          "trend_knots": unique.tolist(), "trend_log_levels": levels.tolist()}


def fit_log_model(y, design, prior_df_cap=100):
    """Vectorized gene-wise OLS; return a single two-sided treatment contrast."""
    n, p = design.shape
    df = n - p
    coef = np.linalg.lstsq(design, y.T, rcond=None)[0].T
    resid = y - coef @ design.T
    variances = np.sum(resid ** 2, axis=1) / df
    posterior, moderation = moderate_variances(variances, y.mean(axis=1), df, prior_df_cap)
    se = np.sqrt(posterior * np.linalg.inv(design.T @ design)[1, 1])
    statistic = coef[:, 1] / se
    total_df = min(df + moderation["prior_df"], len(y) * df)
    pvalue = 2 * student_t.sf(np.abs(statistic), total_df)
    return coef[:, 1], se, pvalue, total_df, moderation


def call_hits(table: CountTable, metadata, treatment, control, config=None, control_ids=None):
    from .diagnostics import normalization_audit
    config = config or TestConfig()
    config.validate()
    x, names, treated, df, warnings = make_design(table.samples, metadata, treatment, control)
    declared = set(control_ids or [])
    missing = declared - set(table.ids)
    if missing:
        raise ValueError(f"Unknown control guide IDs: {sorted(missing)[:5]}")
    controls = np.array([gene in CONTROL_GENES or gid in declared for gid, gene in zip(table.ids, table.genes)])
    factors, norm_warnings = size_factors(table.counts, config.normalization, controls)
    warnings += norm_warnings
    norm_audit = normalization_audit(table, controls, config.normalization)
    if any(a["max_absolute_log2_factor_difference"] > .5 for a in norm_audit["alternatives"].values()):
        warnings.append("Normalization factors disagree by >0.5 log2 with an alternative; inspect controls/compositional bias")
    groups = {}
    for i, gene in enumerate(table.genes):
        if not controls[i]:
            groups.setdefault(gene, []).append(i)
    genes = sorted(groups)
    if not genes:
        raise ValueError("No annotated target genes")
    gene_counts = np.array([table.counts[groups[g]].sum(axis=0) for g in genes])
    normalized = gene_counts / factors
    n_guides = np.array([len(groups[g]) for g in genes])
    min_samples = int(min(treated.sum(), (~treated).sum()))
    eligible = (n_guides >= config.min_guides) & ((normalized >= config.min_count).sum(axis=1) >= min_samples)
    if eligible.sum() < 20:
        raise ValueError("Fewer than 20 genes pass prespecified count/guide filters; no hit calls produced")
    y = np.log2(normalized[eligible] + config.pseudocount)
    effects, se, p, total_df, moderation = fit_log_model(y, x, config.prior_df_cap)
    q = bh_adjust(p)
    delta = student_t.ppf((1 + config.confidence) / 2, total_df) * se
    if moderation["zero_residual_variances"]:
        warnings.append("Some genes have effectively zero residual variance; examine degenerate/low-count genes")
    warnings += ["Experimental log-count baseline; not a negative-binomial, guide-efficacy or robust-outlier model",
                 "Gene sums can be dominated by a high-abundance guide; guide diagnostics do not correct p-values",
                 "No CNV correction; targeted and non-targeting controls need not share a biological null"]
    if config.normalization == "median":
        warnings.append("Median normalization assumes most guides do not undergo a directional compositional shift")
    tested_index, result = 0, []
    for i, gene in enumerate(genes):
        r = {"gene": gene, "guides": int(n_guides[i]), "tested": bool(eligible[i]),
             "status": "tested" if eligible[i] else "filtered_low_count_or_guide_number",
             "mean_control_count": float(normalized[i, ~treated].mean()),
             "mean_treatment_count": float(normalized[i, treated].mean()),
             "log2_effect": None, "standard_error": None, "ci_low": None, "ci_high": None,
             "pvalue": None, "qvalue": None, "hit": False, "direction": "not_tested",
             "guide_direction_agreement": None, "max_guide_read_share": None}
        indices = groups[gene]
        guide_norm = table.counts[indices] / factors
        means = guide_norm.sum(axis=1)
        r["max_guide_read_share"] = float(means.max() / means.sum()) if means.sum() else 0.0
        if eligible[i]:
            j = tested_index
            guide_lfc = np.log2(guide_norm[:, treated].mean(axis=1) + config.pseudocount) - np.log2(guide_norm[:, ~treated].mean(axis=1) + config.pseudocount)
            r.update(log2_effect=float(effects[j]), standard_error=float(se[j]), ci_low=float(effects[j] - delta[j]),
                     ci_high=float(effects[j] + delta[j]), pvalue=float(p[j]), qvalue=float(q[j]),
                     hit=bool(q[j] <= config.fdr), direction="enrichment" if effects[j] > 0 else "depletion" if effects[j] < 0 else "zero",
                     guide_direction_agreement=float((np.sign(guide_lfc) == np.sign(effects[j])).mean()))
            tested_index += 1
        r["baseline_zero_guides"] = int((table.counts[indices][:, ~treated].sum(axis=1) == 0).sum())
        r["absent_at_baseline"] = bool(gene_counts[i, ~treated].sum() == 0)
        flags = []
        if r["absent_at_baseline"]:
            flags.append("absent_at_baseline")
        if r["max_guide_read_share"] >= .8:
            flags.append("guide_dominance")
        if r["guide_direction_agreement"] is not None and r["guide_direction_agreement"] < .5:
            flags.append("guide_direction_disagreement")
        r["diagnostic_flags"] = ";".join(flags)
        result.append(r)
    return result, {"method": "gene-sum-trend-moderated-log-count-OLS", "release_status": "experimental-alpha",
                    "config": asdict(config), "contrast": f"{treatment} minus {control}",
                    "samples": table.samples, "design_columns": names, "design_matrix": x.tolist(),
                    "biological_samples": len(table.samples), "residual_df": df, "moderated_df": total_df,
                    "normalization_factors": dict(zip(table.samples, map(float, factors))),
                    "normalization_audit": norm_audit,
                    "control_guides_excluded": int(controls.sum()), "tested_genes": int(eligible.sum()),
                    "filtered_genes": int((~eligible).sum()), "moderation": moderation,
                    "fdr_family": "BH across all tested target genes for ONE two-sided contrast; no joint multi-contrast claim",
                    "confidence_interval": "Marginal approximate model-based intervals, not simultaneous or selective intervals",
                    "guide_diagnostics": "Descriptive unadjusted group summaries, not independent replication or robust hit calling",
                    "warnings": warnings}
