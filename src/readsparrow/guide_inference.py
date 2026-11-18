"""Guide-level biological-sample tests + directional partial conjunction.

Bonferroni partial conjunction is an existing statistical method, not a new
invention: Benjamini & Heller (2008), doi:10.1111/j.1541-0420.2007.00984.x.
It does not require independent guides IF the constituent p-values are valid.
Here those p-values come from an approximate moderated log-count model, so the
combined procedure still needs empirical calibration. It cannot remove CNV or
distinguish biology from artifacts shared by multiple guides.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import t as student_t

from .diagnostics import normalization_audit
from .inference import CONTROL_GENES, TestConfig, bh_adjust, fit_log_model, make_design, size_factors


@dataclass
class GuideConfig(TestConfig):
    min_guide_count: int = 5
    minimum_support: int = 2
    support_fraction: float = 0.5
    adjustment: str = "bh"

    def validate(self):
        super().validate()
        if self.min_guide_count < 1 or self.minimum_support < 2:
            raise ValueError("Positive guide-count threshold and >=2 supporting guides required")
        if not 0 < self.support_fraction <= 1:
            raise ValueError("Support fraction must be in (0,1]")
        if self.adjustment not in {"bh", "by"}:
            raise ValueError("Adjustment must be bh or by")


def partial_conjunction(pvalues, required):
    """P = min(1,(m-r+1)*p_(r)); valid for >=r constituent alternatives.

    Under the partial-conjunction null, <=r-1 guides may have arbitrary evidence.
    Counting r is preregistered, not selected to minimize the resulting p-value.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Constituent p-values must be finite probabilities")
    if not isinstance(required, (int, np.integer)) or required < 1:
        raise ValueError("Required support must be a positive integer")
    if required > len(p):
        return 1.0
    return float(min(1.0, (len(p) - required + 1) * np.partition(p, required - 1)[required - 1]))


def adjust_pvalues(p, method):
    q = bh_adjust(p)
    if method == "by":
        q = np.minimum(1, q * np.sum(1 / np.arange(1, len(p) + 1)))
    return q


def call_guide_hits(table, metadata, treatment, control, config=None, control_ids=None, *, _evidence_transform=None):
    config = config or GuideConfig()
    config.validate()
    design, names, treated, residual_df, warnings = make_design(table.samples, metadata, treatment, control)
    declared = set(control_ids or [])
    if declared - set(table.ids):
        raise ValueError("Unknown declared control guide IDs")
    controls = np.array([gene in CONTROL_GENES or gid in declared for gid, gene in zip(table.ids, table.genes)])
    factors, norm_warnings = size_factors(table.counts, config.normalization, controls)
    warnings += norm_warnings
    audit = normalization_audit(table, controls, config.normalization)
    if any(v["max_absolute_log2_factor_difference"] > .5 for v in audit["alternatives"].values()):
        warnings.append("Normalization factors disagree by >0.5 log2; inspect compositional bias/control validity")
    normalized = table.counts / factors
    min_samples = int(min(treated.sum(), (~treated).sum()))
    eligible_guide = ~controls & ((normalized >= config.min_guide_count).sum(axis=1) >= min_samples)
    if eligible_guide.sum() < 20:
        raise ValueError("At least 20 informative target guides required for variance moderation")
    all_y = np.log2(normalized + config.pseudocount)
    # Point estimates exist even for low-information guides; they never gain test
    # evidence through filtering. Ineligible guide p-values are set to ONE.
    coefficients = np.linalg.lstsq(design, all_y.T, rcond=None)[0].T
    beta = coefficients[:, 1]
    effects, se, two_sided, moderated_df, moderation = fit_log_model(all_y[eligible_guide], design, config.prior_df_cap)
    se_all = np.full(len(table.ids), np.inf)
    p_positive = np.ones(len(table.ids))
    p_negative = np.ones(len(table.ids))
    se_all[eligible_guide] = se
    p_positive[eligible_guide] = student_t.sf(effects / se, moderated_df)
    p_negative[eligible_guide] = student_t.cdf(effects / se, moderated_df)
    transformed = {}
    if _evidence_transform is not None:
        transformed = _evidence_transform(dict(table=table, design=design, treated=treated, controls=controls,
            normalized=normalized, all_y=all_y, beta=beta, se=se_all, eligible=eligible_guide,
            df=moderated_df, residual_df=residual_df, moderation=moderation, min_samples=min_samples))
        beta, se_all = transformed["beta"], transformed["se"]
        p_positive, p_negative = transformed["p_positive"], transformed["p_negative"]
        moderated_df = transformed["df"]
    groups = {}
    for i, gene in enumerate(table.genes):
        if not controls[i]:
            groups.setdefault(gene, []).append(i)
    if not groups:
        raise ValueError("No target genes")
    rows, guide_rows = [], []
    id_array = np.asarray(table.ids)
    critical_by_m = {}
    for gene in sorted(groups):
        indices = np.array(groups[gene])
        m = len(indices)
        required = max(config.minimum_support, math.ceil(config.support_fraction * m))
        gene_norm = normalized[indices].sum(axis=0)
        eligible = (m >= config.min_guides and eligible_guide[indices].sum() >= required
                    and (gene_norm >= config.min_count).sum() >= min_samples)
        pp = partial_conjunction(p_positive[indices], required)
        pn = partial_conjunction(p_negative[indices], required)
        p = min(1., 2 * min(pp, pn))  # account for choosing enrichment or depletion
        median_effect = float(np.median(beta[indices]))
        direction = "enrichment" if pp < pn else "depletion" if pn < pp else "ambiguous"
        target_sign = 1 if direction == "enrichment" else -1 if direction == "depletion" else 0
        # No data-dependent deletion or p-value downweighting is performed.
        center = np.median(beta[indices])
        scale = max(.25, 1.4826 * float(np.median(np.abs(beta[indices] - center))))
        outliers = np.abs(beta[indices] - center) > 4 * scale
        loo = [float(np.median(np.delete(beta[indices], j))) for j in range(m)] if m > 1 else [median_effect]
        # Conservative within-gene simultaneous-guide rectangle, propagated to
        # its median. This is NOT the interval for a gene-sum effect or a
        # simultaneous interval across all genes. Ineligible guides are unbounded.
        if m not in critical_by_m:
            critical_by_m[m] = student_t.ppf(1 - (1 - config.confidence) / (2 * m), moderated_df)
        critical = critical_by_m[m]
        if "interval_bounds" in transformed:
            lo, hi = transformed["interval_bounds"](m, config.confidence)
            lower, upper = np.median(lo[indices]), np.median(hi[indices])
        else:
            lower = np.median(beta[indices] - critical * se_all[indices])
            upper = np.median(beta[indices] + critical * se_all[indices])
        flags = []
        if outliers.any():
            flags.append("guide_effect_outlier")
        if pp < .025 and pn < .025:
            flags.append("opposing_directional_evidence")
        if target_sign != 0 and np.sign(median_effect) != target_sign:
            flags.append("median_effect_direction_disagrees_with_test")
        if (table.counts[indices][:, ~treated].sum(axis=1) == 0).any():
            flags.append("baseline_zero_guide")
        if gene_norm[~treated].sum() == 0:
            flags.append("absent_at_baseline")
        share = normalized[indices].sum(axis=1)
        max_share = float(share.max() / share.sum()) if share.sum() else 0.
        if max_share >= .8:
            flags.append("guide_read_dominance")
        row = {"gene": gene, "guides": m, "informative_guides": int(eligible_guide[indices].sum()),
               "required_support": required, "tested": bool(eligible),
               "status": "tested" if eligible else "filtered_insufficient_guide_information",
               "mean_control_count": float(gene_norm[~treated].mean()), "mean_treatment_count": float(gene_norm[treated].mean()),
               "log2_effect": median_effect, "effect_estimand": "median fitted guide log2 contrast",
               "standard_error": None, "ci_low": float(lower) if np.isfinite(lower) else None,
               "ci_high": float(upper) if np.isfinite(upper) else None,
               "ci_informative": bool(np.isfinite(lower) and np.isfinite(upper)),
               "pvalue": p if eligible else None, "qvalue": None, "hit": False,
               "pc_enrichment_pvalue": pp, "pc_depletion_pvalue": pn, "direction": direction if eligible else "not_tested",
               "guide_direction_agreement": float((np.sign(beta[indices]) == target_sign).mean()),
               "max_guide_read_share": max_share, "outlier_guides": ";".join(id_array[indices][outliers]),
               "leave_one_guide_out_max_effect_change": max(abs(v - median_effect) for v in loo),
               "diagnostic_flags": ";".join(flags)}
        rows.append(row)
        if transformed:
            row["raw_median_guide_log2_effect"] = float(np.median(transformed["raw_beta"][indices]))
            row["effect_estimand"] = "median fitted guide contrast relative to count-stratified control background"
        for local, i in enumerate(indices):
            guide_rows.append({"sgRNA": table.ids[i], "gene": gene, "informative": bool(eligible_guide[i]),
                               "log2_effect": float(beta[i]), "standard_error": float(se_all[i]) if eligible_guide[i] else None,
                               "p_enrichment": float(p_positive[i]), "p_depletion": float(p_negative[i]),
                               "effect_outlier_flag": bool(outliers[local]), "removed_from_test": False,
                               "baseline_zero": bool(table.counts[i, ~treated].sum() == 0)})
            if transformed:
                guide_rows[-1].update(transformed["guide_extra"][i])
    tested = [r for r in rows if r["tested"]]
    if not tested:
        raise ValueError("No genes pass the fixed multi-guide information requirement")
    for row, q in zip(tested, adjust_pvalues([r["pvalue"] for r in tested], config.adjustment)):
        row["qvalue"], row["hit"] = float(q), bool(q <= config.fdr)
    warnings += ["Experimental guide-evidence caller; constituent moderated p-values require calibration",
                 "Partial conjunction requires multiple same-direction guide associations; it does not prove a causal gene mechanism",
                 "One arbitrary guide cannot alone establish the configured multi-guide claim if the other guide p-values are valid",
                 "Artifacts shared by >=required_support guides, CNV toxicity and global normalization bias remain unsolved",
                 "Median guide effect is not the old gene-sum effect; no ordinary gene-level standard error is supplied",
                 "Outlier flags/leave-one-out summaries are descriptive; no guide is deleted because of its observed effect"]
    if config.normalization == "median":
        warnings.append("Median normalization assumes no predominant directional guide shift")
    info = {"method": "directional-Bonferroni-partial-conjunction-of-moderated-guide-tests",
                  "release_status": "experimental-guide-aware-alpha", "config": asdict(config),
                  "contrast": f"{treatment} minus {control}", "samples": table.samples,
                  "design_columns": names, "design_matrix": design.tolist(), "biological_samples": len(table.samples),
                  "residual_df_per_guide": residual_df, "moderated_df_per_guide": moderated_df, "moderation": moderation,
                  "normalization_factors": dict(zip(table.samples, map(float, factors))), "normalization_audit": audit,
                  "control_guides_excluded": int(controls.sum()), "informative_target_guides": int(eligible_guide.sum()),
                  "tested_genes": len(tested), "filtered_genes": len(rows) - len(tested),
                  "fdr_family": f"{config.adjustment.upper()} over tested genes for one contrast, each p accounts for both directions",
                  "effect_estimand": "Median of all annotated target guide log2 contrasts, including low-information guides",
                  "confidence_interval": "Median of within-gene Bonferroni simultaneous guide bounds; uninformative guides have unbounded intervals; approximate constituent model",
                  "guide_independence_assumed_for_combination": False,
                  "biological_replication_unit": "columns only; guides do not increase per-guide residual df",
                  "removed_outlier_guides": 0, "warnings": warnings}
    if transformed:
        info.update(transformed["info"])
        info["warnings"].extend(transformed["warnings"])
        info["moderated_df_per_guide"] = float(np.min(moderated_df))
    return rows, info, guide_rows
