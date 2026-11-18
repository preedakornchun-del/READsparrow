"""Experimental bounded-location gene scores and delete-one IUT sensitivity.

The aggregation claim differs from v0.4 partial conjunction. Subset mean tests
include biological-sample covariance (through mean-log-count residuals) and a
control-derived persistent-guide component. Independent persistent nuisance
guides are an explicit model assumption, NOT additional biological replication.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.special import digamma
from scipy.stats import t as student_t

from .empirical_inference import EmpiricalConfig, call_empirical_hits
from .guide_inference import adjust_pvalues
from .inference import fit_log_model, make_design


@dataclass
class PowerConfig(EmpiricalConfig):
    background_correlation: float = 0.0
    aggregation: str = "loo"
    clipping_threshold: float = 2.5

    def validate(self):
        super().validate()
        if self.null_model != "background":
            raise ValueError("Delete-one model uses background; empirical-rank engine remains separately available")
        if not 0 <= self.background_correlation <= 1:
            raise ValueError("Prespecified background correlation must be in [0,1]")
        if self.minimum_support != 2:
            raise ValueError("Delete-one claim requires two effective guides; use prior partial-conjunction engine for other support rules")
        if self.aggregation not in {"bounded", "loo"} or not np.isfinite(self.clipping_threshold) or self.clipping_threshold < 1:
            raise ValueError("Use bounded or loo aggregation and a prespecified clipping threshold >=1")


def nuisance_variance(tau2, bin_ids, location_variances, rho=0.):
    """Covariance-aware variance of an equal-weight guide mean.

    A null-location estimate is shared within each abundance bin: its uncertainty
    MUST NOT be divided as if every guide had an independent fitted location.
    """
    tau2 = np.asarray(tau2, float)
    n = len(tau2)
    if not n:
        raise ValueError("Nonempty guide subset required")
    independent = np.sum(tau2) / n ** 2
    persistent = (1 - rho) * independent + rho * np.sum(np.sqrt(tau2)) ** 2 / n ** 2
    shares = np.bincount(np.asarray(bin_ids, int), minlength=len(location_variances)) / n
    return float(persistent + np.dot(shares ** 2, location_variances))


def predict_measurement(y, design, moderation, residual_df):
    """Use a full-gene-trained prior for every subset; no guide-added df."""
    coefficients = np.linalg.lstsq(design, y.T, rcond=None)[0].T
    residuals = y - coefficients @ design.T
    variances = (residuals ** 2).sum(axis=1) / residual_df
    d0 = moderation["prior_df"]
    trend = np.interp(y.mean(axis=1), moderation["trend_knots"], moderation["trend_log_levels"])
    prior = np.exp(trend + digamma(d0 / 2) - np.log(d0 / 2))
    posterior = (d0 * prior + residual_df * variances) / (d0 + residual_df)
    se = np.sqrt(np.maximum(posterior, moderation["variance_floor"]) * np.linalg.inv(design.T @ design)[1, 1])
    return coefficients[:, 1], se


def combined_df(measurement_variance, nuisance, measurement_df, nuisance_df):
    """Welch–Satterthwaite approximation, NOT extra biological residual df."""
    a, b = np.asarray(measurement_variance), np.asarray(nuisance)
    return np.maximum(2., (a + b) ** 2 / np.maximum(a ** 2 / measurement_df + b ** 2 / nuisance_df, 1e-30))


def winsorized_location(effects, scales, threshold=2.5):
    """Translation-equivariant location, no clipping coherent signal to zero."""
    effects, scales = np.asarray(effects, float), np.asarray(scales, float)
    median = np.median(effects)
    return float(np.mean(median + np.clip(effects-median, -threshold*scales, threshold*scales)))


def call_power_hits(table, metadata, treatment, control, config=None, control_ids=None):
    config = config or PowerConfig()
    config.validate()
    # Reuse the EXACT previous measurement/control fit, filters and fixed split.
    # Its PC p-values are not reused as the new gene p-values.
    context = {}
    previous, info, guides = call_empirical_hits(table, metadata, treatment, control, config, control_ids, _context_sink=context)
    design, names, treated, residual_df, _ = make_design(table.samples, metadata, treatment, control)
    factors = np.array([info["normalization_factors"][s] for s in table.samples])
    normalized = table.counts / factors
    y = np.log2(normalized + config.pseudocount)
    guide_map = {r["sgRNA"]: r for r in guides}
    groups = {}
    for i, (gid, gene) in enumerate(zip(table.ids, table.genes)):
        if gid in guide_map:
            groups.setdefault(gene, []).append(i)
    bins = info["background_model"]["bins"]
    tau2 = np.array([b["extra_background_variance"] for b in bins])
    loc_var = np.array([b["location_prediction_variance"] for b in bins])
    bin_df = np.array([b["training_controls"] - 1 for b in bins])
    gids = [guide_map.get(g) for g in table.ids]
    bin_ids = np.array([g["background_bin"] if g else 0 for g in gids])
    centers = np.array([g["background_center"] if g else 0. for g in gids])
    adjusted_beta = np.array([g["log2_effect"] if g else 0. for g in gids])
    training_z = context["z"][context["train"]]

    @lru_cache(maxsize=32)
    def robust_null(m):
        # Resampling learns the location rule's null bias/variance multiplier,
        # NOT new independent controls or arbitrarily precise empirical p-tails.
        rng = np.random.default_rng(817260000 + m)
        draws = 50000
        empirical = rng.choice(training_z, size=(draws, m), replace=True)
        normal = rng.normal(size=(draws, m))
        def location(x):
            mid = np.median(x, axis=1, keepdims=True)
            return np.mean(mid + np.clip(x - mid, -config.clipping_threshold, config.clipping_threshold), axis=1)
        e, n = location(empirical), location(normal)
        inflation = max(1., m * float(np.var(e, ddof=1)), m * float(np.var(n, ddof=1)))
        # Monte Carlo precision does not replace finite-training-control error.
        bias_uncertainty = float(np.var(training_z, ddof=1) / len(training_z))
        return float(np.mean(e)), inflation, bias_uncertainty
    eligible = np.array([g["informative"] if g else False for g in gids])
    gene_names = sorted(groups)
    full_y = np.array([y[groups[g]].mean(axis=0) for g in gene_names])
    min_samples = int(min(treated.sum(), (~treated).sum()))
    good_gene = np.array([len(groups[g]) >= max(2, config.min_guides) and eligible[groups[g]].all()
        and ((normalized[groups[g]].sum(axis=0) >= config.min_count).sum() >= min_samples) for g in gene_names])
    if good_gene.sum() < 20:
        raise ValueError("Fewer than 20 fully informative genes for delete-one measurement moderation; no silent fallback")
    _, _, _, measurement_df, moderation = fit_log_model(full_y[good_gene], design, config.prior_df_cap)
    feature_y, feature_center, feature_nuisance, feature_df, feature_meta = [], [], [], [], []
    feature_indices = {}
    for gene in gene_names:
        idx = np.array(groups[gene])
        feature_indices[gene] = []
        for removed in [None] + list(idx) if len(idx) >= 2 else [None]:
            subset = idx if removed is None else idx[idx != removed]
            feature_indices[gene].append(len(feature_y))
            feature_y.append(y[subset].mean(axis=0))
            feature_center.append(centers[subset].mean())
            feature_nuisance.append(nuisance_variance(tau2[bin_ids[subset]], bin_ids[subset], loc_var, config.background_correlation))
            feature_df.append(float(bin_df[bin_ids[subset]].min()))
            feature_meta.append((gene, "none" if removed is None else table.ids[removed], len(subset)))
    raw_beta, measurement_se = predict_measurement(np.array(feature_y), design, moderation, residual_df)
    effect = raw_beta - np.array(feature_center)
    prediction_se = np.sqrt(measurement_se ** 2 + feature_nuisance)
    dfs = combined_df(measurement_se ** 2, np.array(feature_nuisance), measurement_df, np.array(feature_df))
    positive, negative = student_t.sf(effect / prediction_se, dfs), student_t.cdf(effect / prediction_se, dfs)
    rows, subset_audit = [], []
    for old, good in zip(previous, good_gene):
        row = dict(old)
        gene = row["gene"]
        fi = feature_indices[gene]
        f, loo = fi[0], fi[1:]
        pp = float(np.max(positive[loo])) if loo else 1.
        pn = float(np.max(negative[loo])) if loo else 1.
        loo_pp, loo_pn = pp, pn
        idx = np.array(groups[gene])
        null_bias, variance_inflation, null_bias_variance = robust_null(len(idx))
        median = float(np.median(adjusted_beta[idx]))
        radii = config.clipping_threshold * context["prediction_se"][idx]
        average_scale = float(np.mean(context["prediction_se"][idx]))
        bounded_score = winsorized_location(adjusted_beta[idx], context["prediction_se"][idx], config.clipping_threshold) - null_bias * average_scale
        # A full-correlation upper proxy for measurement variance avoids treating
        # guides as independent biological samples when clipping is nonlinear.
        bounded_measurement_var = variance_inflation * max(float(measurement_se[f] ** 2), float(np.mean(context["measurement_se"][idx]) ** 2))
        bounded_nuisance_var = variance_inflation * feature_nuisance[f] + null_bias_variance * average_scale ** 2
        bounded_se = float(np.sqrt(bounded_measurement_var + bounded_nuisance_var))
        bounded_df = float(combined_df(bounded_measurement_var, bounded_nuisance_var,
            min(measurement_df, context["df"]), feature_df[f]))
        if config.aggregation == "bounded":
            pp = float(student_t.sf(bounded_score / bounded_se, bounded_df)) if np.sum(adjusted_beta[idx] > 0) >= 2 else 1.
            pn = float(student_t.cdf(bounded_score / bounded_se, bounded_df)) if np.sum(adjusted_beta[idx] < 0) >= 2 else 1.
        p = min(1., 2 * min(pp, pn))
        direction = "enrichment" if pp < pn else "depletion" if pn < pp else "ambiguous"
        critical = student_t.ppf((1 + config.confidence) / 2, dfs[f])
        flags = [v for v in row["diagnostic_flags"].split(";") if v and v not in {
            "opposing_directional_evidence", "median_effect_direction_disagrees_with_test"}]
        if direction == "enrichment" and effect[f] <= 0 or direction == "depletion" and effect[f] >= 0:
            flags.append("mean_effect_direction_disagrees_with_test")
        row.pop("pc_enrichment_pvalue", None)
        row.pop("pc_depletion_pvalue", None)
        row.update(tested=bool(good), status="tested" if good else "filtered_requires_all_annotated_guides_informative",
            required_support=2, log2_effect=float(effect[f]), effect_estimand="equal-weight mean guide contrast relative to control background",
            median_guide_effect=float(np.median(adjusted_beta[idx])), raw_mean_guide_effect=float(raw_beta[f]),
            standard_error=float(prediction_se[f]) if good else None,
            ci_low=float(effect[f] - critical * prediction_se[f]) if good else None,
            ci_high=float(effect[f] + critical * prediction_se[f]) if good else None, ci_informative=bool(good),
            confidence_interval_role="Approximate full mean-effect interval, NOT dual to stronger delete-one robustness test; not outlier-robust effect interval",
            pvalue=p if good else None, qvalue=None, hit=False, direction=direction if good else "not_tested",
            loo_enrichment_pvalue=loo_pp, loo_depletion_pvalue=loo_pn,
            bounded_enrichment_pvalue=float(student_t.sf(bounded_score / bounded_se, bounded_df)),
            bounded_depletion_pvalue=float(student_t.cdf(bounded_score / bounded_se, bounded_df)),
            bounded_log2_score=bounded_score, bounded_prediction_se=bounded_se, bounded_model_df=bounded_df,
            clipped_guides=";".join(table.ids[i] for j, i in enumerate(idx) if abs(adjusted_beta[i] - median) > radii[j]),
            robust_null_bias_standardized=null_bias, robust_variance_inflation=variance_inflation,
            positive_guide_fraction=float(np.mean(adjusted_beta[idx] > 0)), negative_guide_fraction=float(np.mean(adjusted_beta[idx] < 0)),
            mean_measurement_standard_error=float(measurement_se[f]), nuisance_variance_of_mean=float(feature_nuisance[f]),
            mean_model_df=float(dfs[f]),
            worst_enrichment_omitted_guide=feature_meta[loo[int(np.argmax(positive[loo]))]][1] if loo else "",
            worst_depletion_omitted_guide=feature_meta[loo[int(np.argmax(negative[loo]))]][1] if loo else "",
            guide_direction_agreement=float(np.mean(np.sign(adjusted_beta[idx]) == (1 if direction == "enrichment" else -1 if direction == "depletion" else 0))),
            diagnostic_flags=";".join(flags))
        rows.append(row)
        for j in fi:
            subset_audit.append(dict(gene=gene, omitted_guide=feature_meta[j][1], guides_in_subset=feature_meta[j][2],
                log2_effect=float(effect[j]), measurement_se=float(measurement_se[j]), prediction_se=float(prediction_se[j]),
                nuisance_variance=float(feature_nuisance[j]), df=float(dfs[j]),
                p_enrichment=float(positive[j]), p_depletion=float(negative[j]), used_for_gene_test=bool(j != f and good)))
    tested = [r for r in rows if r["tested"]]
    for row, q in zip(tested, adjust_pvalues([r["pvalue"] for r in tested], config.adjustment)):
        row["qvalue"], row["hit"] = float(q), bool(q <= config.fdr)
    # Retain control warnings but replace PC-specific interpretations.
    info["warnings"] = [w for w in info["warnings"] if not any(s in w for s in
        ("Partial conjunction", "configured multi-guide claim", "Median guide effect", ">=required_support"))]
    info["warnings"] += [
        "Delete-one intersection-union: EVERY leave-one-guide-out mean must support a direction; no observed-effect-selected deletion",
        "Subset p-values require a valid mean-log-count model plus representative control-derived nuisance variation; this is not exact finite-sample inference",
        "Default persistent guide nuisance is independent across guides; correlated guide drift can invalidate inference; biological covariance is modeled separately through sample residuals",
        "At most one arbitrary guide artifact is addressed if the remaining-guide subset p-value is valid; coherent multi-guide artifacts and cutting/CNV remain unsolved",
        "New tested family requires ALL annotated guides informative; recall includes filtered active genes",
        "Effect is an equal-weight mean, unlike previous median/count-weighted effects; its CI is not outlier-robust or dual to delete-one p-values",
        "Guides estimate a gene mean of random nuisance components, NOT more biological replicates or residual df"]
    info.update(method=("bounded-influence-background-gene-score" if config.aggregation == "bounded" else "delete-one-intersection-union-of-covariance-aware-background-mean-tests"),
        release_status="experimental-power-alpha", config=asdict(config), tested_genes=len(tested), filtered_genes=len(rows)-len(tested),
        effect_estimand="equal-weight mean fitted guide log2 contrast relative to count-stratified control background",
        confidence_interval="Approximate full mean-effect interval including nuisance and shared location uncertainty; not dual to robust test and not outlier-robust",
        gene_measurement_moderation=moderation, gene_moderated_df=measurement_df, residual_df_per_gene=residual_df,
        technical_background_independence_assumed=config.background_correlation == 0,
        prespecified_background_correlation=config.background_correlation,
        robustness_claim="At least two same-direction guide associations under valid delete-one subset models; not the previous >=half-guides partial-conjunction claim",
        guide_pvalue_role="Descriptive v0.4 constituent values; gene calls use delete-one aggregated subset p-values",
        subset_audit=subset_audit)
    info["combined_df_method"] = "Approximate Welch-Satterthwaite of measurement and control-derived nuisance variance; not new biological residual df; ignores some hyperparameter dependence"
    info["bounded_control_fit"] = {"clipping_threshold": config.clipping_threshold,
        "centering_rule": "Winsorize deviations around each gene median; coherent shifts are NOT clipped toward zero",
        "training_only_null_location_by_guide_number": {str(m): {"bias": robust_null(m)[0], "variance_inflation": robust_null(m)[1], "bias_uncertainty_variance": robust_null(m)[2]} for m in sorted({len(groups[g]) for g in gene_names})},
        "resampling_draws_per_size": 50000, "actual_training_controls": len(training_z),
        "resampling_boundary": "Monte Carlo approximates a fitted independent-guide null population; not extra biological replicates, not distribution-free tail resolution; finite-control bias uncertainty retained",
        "measurement_variance": "maximum of residual covariance-aware mean variance and squared mean per-guide measurement SE (full-correlation upper proxy)",
        "background_variance": "unclipped nuisance variance retained, never shrunk by clipping; shared fitted-location uncertainty retained"}
    if config.aggregation == "bounded":
        info["warnings"] = [w for w in info["warnings"] if not w.startswith(("Delete-one intersection-union:", "At most one arbitrary guide artifact"))]
        info["robustness_claim"] = "Bounded influence plus two same-direction signs; NOT a partial-conjunction guarantee, not proof of >=2 true guide associations"
        info["guide_pvalue_role"] = "Per-guide p-values and delete-one p-values are descriptive sensitivity outputs; PRIMARY calls use bounded-location gene scores"
        info["warnings"] += ["PRIMARY bounded score does NOT inherit delete-one adversarial-outlier guarantees; delete-one p-values are separately exported for sensitivity",
            "Signed bounded-score Student-t tails are an approximate working model, not a rigorous concentration bound; all-null/artifact/heavy-tail calibration is required",
            "Two-sign gate can only increase directional p-values; it is descriptive concordance, not a two-effective-guide statistical claim"]
    return rows, info, guides


def loo_rows_from_power(rows, config=None):
    """Exact LOO decisions from the same model fit; avoid fitting it twice."""
    config = config or PowerConfig(aggregation="loo")
    result = []
    for original in rows:
        row = dict(original)
        pp, pn = row["loo_enrichment_pvalue"], row["loo_depletion_pvalue"]
        direction = "enrichment" if pp < pn else "depletion" if pn < pp else "ambiguous"
        row.update(pvalue=min(1., 2*min(pp, pn)) if row["tested"] else None,
                   qvalue=None, hit=False, direction=direction if row["tested"] else "not_tested")
        row["guide_direction_agreement"] = row["positive_guide_fraction"] if direction == "enrichment" else row["negative_guide_fraction"] if direction == "depletion" else 1-row["positive_guide_fraction"]-row["negative_guide_fraction"]
        flags = [f for f in row["diagnostic_flags"].split(";") if f and f != "mean_effect_direction_disagrees_with_test"]
        if direction == "enrichment" and row["log2_effect"] <= 0 or direction == "depletion" and row["log2_effect"] >= 0:
            flags.append("mean_effect_direction_disagrees_with_test")
        row["diagnostic_flags"] = ";".join(flags)
        result.append(row)
    tested = [r for r in result if r["tested"]]
    for row, q in zip(tested, adjust_pvalues([r["pvalue"] for r in tested], config.adjustment)):
        row["qvalue"], row["hit"] = float(q), bool(q <= config.fdr)
    return result
