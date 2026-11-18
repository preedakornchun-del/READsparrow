"""Experimental control-trained asymmetric tails and matched-copula scans.

Every deleted-one-guide subset must reject its all-null hypothesis. Covariance
precision scans can recover a sparse profile without assuming all guides work.
This supports >=2 non-null associations, NOT >=2 same-direction causal effects.
Fitted sinh-arcsinh tails/copula and their uncertainty are working models, not
distribution-free FDR guarantees. No HQ461 outcome enters fitting/selection.
"""
from dataclasses import asdict, dataclass
import copy
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import norm, t as student_t

from .correlated_inference import CorrelatedConfig, fit_correlation
from .empirical_inference import call_empirical_hits
from .guide_inference import adjust_pvalues, partial_conjunction
from .sparse_posterior import sparse_log_bayes_factor, posterior_decisions, three_class_decisions


@dataclass
class ShapeConfig(CorrelatedConfig):
    tail_inflation: float = 1.15
    scan_weight: float = .8
    max_scan_guides: int = 8
    matched_inference: str = "posterior"
    posterior_safety_factor: float = 2.

    def validate(self):
        super().validate()
        if not 1 <= self.tail_inflation <= 3 or not 0 < self.scan_weight < 1:
            raise ValueError("Use inflation >=1 and fixed scan weight in (0,1)")
        if not 3 <= self.max_scan_guides <= 8:
            raise ValueError("Scan supports 3..8 annotated guides; larger genes use marginal-only evidence")
        if self.matched_inference not in {"scan", "posterior"}:
            raise ValueError("Use scan or posterior matched inference")
        if self.posterior_safety_factor < 1 or not np.isfinite(self.posterior_safety_factor):
            raise ValueError("Posterior safety factor must be finite and >=1")


def sas_latent(x, parameters):
    location, log_scale, skew, log_tail = parameters
    u = np.arcsinh((np.asarray(x) - location) / np.exp(log_scale))
    return np.sinh(np.clip(np.exp(log_tail) * u - skew, -35, 35))


def sas_nll(parameters, x):
    location, log_scale, skew, log_tail = parameters
    standardized = (x - location) / np.exp(log_scale)
    u = np.exp(log_tail) * np.arcsinh(standardized) - skew
    z = np.sinh(np.clip(u, -35, 35))
    log_cosh = np.logaddexp(u, -u) - np.log(2)
    log_jacobian = log_tail - log_scale + log_cosh - .5 * np.log1p(standardized ** 2)
    return float(np.sum(.5 * z * z - log_jacobian + .5 * np.log(2 * np.pi)))


def fit_shape(x):
    """Fit only the ID-split training controls; fixed BIC selects extra shape.

    The Gaussian is nested in SAS. Multiple deterministic starts reduce local
    optimization failures. Values outside the model bounds remain limitations.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 50 or np.any(~np.isfinite(x)):
        raise ValueError("Need >=50 finite training-control effects")
    mu, sd = float(x.mean()), max(.05, float(x.std(ddof=0)))
    gaussian = np.array([mu, np.log(sd), 0., 0.])
    bounds = [(mu - 4 * sd, mu + 4 * sd), (np.log(sd) - 3, np.log(sd) + 3), (-3., 3.), (np.log(.35), np.log(2.5))]
    fits = [minimize(sas_nll, gaussian + np.array([0, 0, skew, 0]), args=(x,), method="L-BFGS-B", bounds=bounds)
            for skew in (0., -.5, .5)]
    valid = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    candidate = min(valid, key=lambda fit: fit.fun) if valid else None
    bic_gaussian = 2 * sas_nll(gaussian, x) + 2 * np.log(len(x))
    bic_shape = 2 * candidate.fun + 4 * np.log(len(x)) if candidate is not None else np.inf
    use_shape = bic_shape < bic_gaussian
    parameters = candidate.x if use_shape else gaussian
    return dict(parameters=parameters.tolist(), family="sinh-arcsinh" if use_shape else "Gaussian",
                bic_gaussian=float(bic_gaussian), bic_shape=float(bic_shape), training_controls=len(x),
                optimizer_success=bool(valid), selected_from="training-control likelihood/BIC only")


def copula_envelope(z, block_members, table, inflation):
    index = {gid: i for i, gid in enumerate(table.ids)}
    block = np.array([[index[gid] for gid in ids] for ids in block_members.values()])
    values = z[block]
    k = values.shape[1]
    # Pairwise correlations are combined across the same independent blocks;
    # they are NOT treated as independent extra block observations.
    matrix = np.corrcoef(values, rowvar=False)
    pairs = matrix[np.triu_indices(k, 1)]
    if np.any(~np.isfinite(pairs)):
        raise ValueError("Degenerate matched null copula correlation")
    point = float(np.clip(np.mean(pairs), 0., .98))
    # Conservative fixed Fisher envelope; working approximation after fitted
    # transform, not an exact confidence set or arbitrary-copula guarantee.
    radius = 3.8 / np.sqrt(max(1, len(block) - 3))
    low = float(max(0., np.tanh(np.arctanh(np.clip(pairs.min(), -.98, .98)) - radius)))
    high = float(min(.995, np.tanh(np.arctanh(np.clip(pairs.max(), -.98, .98)) + radius)))
    grid = np.unique(np.r_[low, high, np.linspace(low, high, 9)])
    return dict(point=point, lower=low, upper=high, grid=grid.tolist(), blocks=len(block), guides_per_block=k,
                pair_correlation_matrix=matrix.tolist(), working_error_budget=1e-4,
                boundary="Exchangeable nonnegative Gaussian copula, representative balanced blocks, approximate Fisher envelope; fitted marginals/parameters not exactly covered")


def subset_scan(z, rhos, df, inflation):
    """Vectorized directional sparse GLS tests with fixed subset Bonferroni.

    Each predeclared nonempty subset represents a positive latent-effect
    direction. Selection is corrected; negative precision weights mean the
    inferred direction is descriptive, not a directional conjunction claim.
    """
    z = np.asarray(z)
    n, m = z.shape
    masks = np.array([[float(i in subset) for i in range(m)]
                      for size in range(1, m + 1) for subset in combinations(range(m), size)])
    cardinality = masks.sum(axis=1)
    weights = np.array([.5 if size == m else .5 / (m - 1) / sum(cardinality == size) for size in cardinality])
    sums, total = z @ masks.T, z.sum(axis=1)[:, None]
    worst_pos, worst_neg = np.zeros(n), np.zeros(n)
    for rho in rhos:
        coefficient = rho / (1 + (m - 1) * rho)
        numerator = sums - coefficient * cardinality * total
        denominator = np.sqrt((1 - rho) * (cardinality - coefficient * cardinality ** 2))
        statistic = numerator / np.maximum(denominator, 1e-12) / inflation
        pp = np.minimum(1., (student_t.sf(statistic, df) / weights).min(axis=1))
        pn = np.minimum(1., (student_t.cdf(statistic, df) / weights).min(axis=1))
        worst_pos, worst_neg = np.maximum(worst_pos, pp), np.maximum(worst_neg, pn)
    return worst_pos, worst_neg


def call_shape_hits(table, metadata, treatment, control, config=None, control_ids=None, control_blocks=None, *, _prepared=None):
    config = config or ShapeConfig()
    config.validate()
    if _prepared is None:
        context = {}
        rows, info, guides = call_empirical_hits(table, metadata, treatment, control, config, control_ids, _context_sink=context)
    else:
        rows, info, guides, context = copy.deepcopy(_prepared)
    correlation = fit_correlation(context, table, control_blocks, config, control_ids)
    shape = fit_shape(context["z"][context["train"]])
    latent = sas_latent(context["z"], shape["parameters"])
    model_latent = latent.copy()
    # Training control count is independent of target hit count. Conservative
    # t tails plus fixed inflation acknowledge, but do not exactly integrate,
    # uncertainty of the fitted four-parameter shape.
    df = max(10., len(context["train"]) / 4 - 1)
    pp = student_t.sf(latent / config.tail_inflation, df)
    pn = student_t.cdf(latent / config.tail_inflation, df)
    # Controls can have less measurement noise than target guides. Never let
    # Gaussianization narrow the earlier total prediction model. This maximum
    # retains either tail guard; it is not a guarantee if BOTH models are wrong.
    guard_df = np.array([row["predictive_df"] for row in context["bins"]])[context["assignment"]]
    pp = np.maximum(pp, student_t.sf(context["z"], guard_df))
    pn = np.maximum(pn, student_t.cdf(context["z"], guard_df))
    latent = np.where(pp < .5, norm.isf(np.maximum(pp, 1e-300)),
                      np.where(pn < .5, norm.ppf(np.maximum(pn, 1e-300)), 0.))
    guide_lookup = {r["sgRNA"]: r for r in guides}
    eligible = np.array([guide_lookup[g]["informative"] if g in guide_lookup else False for g in table.ids])
    pp[~eligible] = pn[~eligible] = 1.
    groups = {}
    for i, gid in enumerate(table.ids):
        if gid in guide_lookup:
            groups.setdefault(table.genes[i], []).append(i)
    groups = {gene: sorted(indices, key=lambda i: table.ids[i]) for gene, indices in groups.items()}
    matched = config.correlation_mode == "matched-blocks"
    posterior_allowed = matched and config.matched_inference == "posterior" and shape["family"] == "Gaussian"
    if posterior_allowed:
        # A generative density must use its fitted marginal transform, not the
        # conservative max-of-p-values guard (which is not a Gaussian density).
        # The latter remains the frequentist marginal-only engine. Posterior
        # model validity is separately stress-tested, not borrowed from it.
        latent = model_latent
    copula = copula_envelope(latent, correlation["block_members"], table, config.tail_inflation) if matched else None
    scans = {}
    if matched and config.matched_inference == "scan":
        for m in sorted({len(indices) for indices in groups.values()}):
            if m < 3 or m > config.max_scan_guides:
                continue
            if m > copula["guides_per_block"]:
                raise ValueError("Matched null blocks have fewer guides than target gene")
            names = [gene for gene, indices in groups.items() if len(indices) == m]
            matrix = np.array([latent[groups[gene]] for gene in names])
            positives, negatives = [], []
            for omitted in range(m):
                keep = np.arange(m) != omitted
                p, n = subset_scan(matrix[:, keep], copula["grid"], min(df, copula["blocks"] - 1), config.tail_inflation)
                positives.append(p); negatives.append(n)
            for gene, p, n in zip(names, np.max(positives, axis=0), np.max(negatives, axis=0)):
                scans[gene] = (min(1., float(p) + copula["working_error_budget"]), min(1., float(n) + copula["working_error_budget"]))
    audit = []
    for row in rows:
        indices = np.array(groups[row["gene"]])
        marginal_pp, marginal_pn = partial_conjunction(pp[indices], 2), partial_conjunction(pn[indices], 2)
        scan_pp, scan_pn = scans.get(row["gene"], (1., 1.))
        if row["gene"] in scans:
            pos = min(1., scan_pp / config.scan_weight, marginal_pp / (1 - config.scan_weight))
            neg = min(1., scan_pn / config.scan_weight, marginal_pn / (1 - config.scan_weight))
        else:
            pos, neg = marginal_pp, marginal_pn
        tested = bool(row["tested"] and eligible[indices].all())
        row.update(tested=tested, status="tested" if tested else "filtered_requires_all_annotated_guides_informative",
                   required_support=2, pvalue=min(1., 2 * min(pos, neg)) if tested else None, qvalue=None, hit=False,
                   direction=("enrichment" if pos < neg else "depletion" if neg < pos else "ambiguous") if tested else "not_tested",
                   direction_role="Descriptive scan orientation; not a same-direction partial-conjunction claim when precision scan used",
                   marginal_enrichment_pvalue=marginal_pp, marginal_depletion_pvalue=marginal_pn,
                   scan_enrichment_pvalue=scan_pp, scan_depletion_pvalue=scan_pn,
                   shape_family=shape["family"], ci_low=None, ci_high=None, ci_informative=False,
                   confidence_interval_role="No shape/support-dual CI supplied; log2_effect remains descriptive median guide contrast",
                   effect_estimand="Descriptive median guide contrast relative to count-stratified background mean")
        audit.append(dict(gene=row["gene"], matched_scan_used=row["gene"] in scans, guides=len(indices),
                          latent_values=";".join(format(float(latent[i]), ".12g") for i in indices),
                          marginal_p=min(1., 2 * min(marginal_pp, marginal_pn)), scan_p=min(1., 2 * min(scan_pp, scan_pn))))
    tested_rows = [row for row in rows if row["tested"]]
    if not tested_rows:
        raise ValueError("No genes pass fully informative guide requirement")
    for row, qvalue in zip(tested_rows, adjust_pvalues([r["pvalue"] for r in tested_rows], config.adjustment)):
        row["qvalue"], row["hit"] = float(qvalue), bool(qvalue <= config.fdr)
    posterior_info = None
    if posterior_allowed:
        for row in rows:
            row.update(log_bayes_factor=None, log_single_density_ratio=None, log_multi_density_ratio=None, posterior_null_probability=None, posterior_qvalue=None)
        for m in sorted({row["guides"] for row in tested_rows}):
            if m > config.max_scan_guides or m > copula["guides_per_block"]:
                raise ValueError("Sparse posterior requires balanced matched blocks with >=target guide number and <=8 target guides")
            selected = [row for row in tested_rows if row["guides"] == m]
            matrix = np.array([latent[groups[row["gene"]]] for row in selected])
            single, multi = sparse_log_bayes_factor(matrix, copula["point"], 1., components=True)
            for row, a, b in zip(selected, single, multi):
                row.update(log_bayes_factor=float(b-a),log_single_density_ratio=float(a),log_multi_density_ratio=float(b))
        local, qvalues, posterior_info = three_class_decisions([row["log_single_density_ratio"] for row in tested_rows],
            [row["log_multi_density_ratio"] for row in tested_rows], config.fdr)
        for row, probability, qvalue in zip(tested_rows, local, qvalues):
            decision_value = min(1., config.posterior_safety_factor * float(qvalue))
            if not posterior_info["global_gate_pass"]:
                decision_value = 1.
            row.update(pvalue=None, posterior_null_probability=float(probability), posterior_qvalue=float(qvalue),
                       qvalue=decision_value, hit=bool(decision_value <= config.fdr),
                       direction="enrichment" if row["log2_effect"] > 0 else "depletion" if row["log2_effect"] < 0 else "ambiguous",
                       direction_role="Descriptive median contrast sign, not a directional posterior support claim")
        for row in rows:
            row["pvalue"] = None
        posterior_info.update(safety_factor=config.posterior_safety_factor,
            family="Safety-scaled posterior expected FDP under fitted mixture, NOT BH q-values",
            hits=sum(row["hit"] for row in tested_rows),
            selection_rule="Only training-control Gaussian BIC family permits sparse posterior; non-Gaussian shape uses guarded marginal/BH only")
    indices_by_id = {gid: i for i, gid in enumerate(table.ids)}
    for guide in guides:
        i = indices_by_id[guide["sgRNA"]]
        guide.update(shape_latent=float(latent[i]), shape_p_enrichment=float(pp[i]), shape_p_depletion=float(pn[i]))
    info.update(method="control-shape-marginal-and-matched-copula-delete-one-scan", config=asdict(config),
                release_status="experimental-shape-alpha", shape_model=shape, copula_model=copula,
                shape_tail_df=df, tail_inflation=config.tail_inflation, subset_audit=audit,
                posterior_model=posterior_info,
                inference_branch="matched-Gaussian-sparse-posterior" if posterior_allowed else "guarded-shape-marginal-BH" if not scans else "matched-delete-one-scan-BH",
                tested_genes=len(tested_rows), filtered_genes=len(rows)-len(tested_rows),
                robustness_claim=">=2 non-null guide associations conditional on fitted subset models; not directional support or causal identification",
                no_universal_FDR_guarantee=True)
    if posterior_info is not None:
        info.update(fdr_family=posterior_info["family"],
                    robustness_claim="Model-based competition of >=2-guide effects against null/single-artifact mixture; no arbitrary-outlier theorem")
    info["warnings"] = [warning for warning in info["warnings"] if not any(text in warning for text in
        ("Partial conjunction", "One arbitrary guide", "Median guide effect", "configured multi-guide"))]
    info["warnings"] += ["Sinh-arcsinh and Gaussian-copula tails extrapolate beyond controls and require calibration; no distribution-free guarantee",
        "Sparse precision scan is only available for genuine matched blocks; no fabricated NTC groups",
        "All subset/sign/branch selection corrected; finite-sample fitted-model uncertainty remains approximate",
        "Scan orientation is descriptive; support claim is >=2 non-null, not >=2 same-direction associations",
        "No new confidence interval is inferred from the adaptive support test",
        "Predeclared training-control BIC gate disables sparse posterior for non-Gaussian fits; this selection is included in calibration, not a guarantee",
        "Posterior safety multiplier was selected in development pilots, not from HQ461; posterior q and safety-scaled decision values are both exported",
        "Sparse posterior additionally requires a prespecified global working likelihood-ratio p<=.01; this is a model-based gate, not an exact selective-inference guarantee",
        "Nonrepresentative controls, multi-guide artifacts and CNV remain unresolved"]
    return rows, info, guides


def score_shape(rows, truth, guides_per_gene):
    """Shared biological truth; posterior probabilities are never p-values."""
    from .power_comparison import score_method
    compatible = [dict(row, pvalue=row["pvalue"] if row["pvalue"] is not None else 1.) for row in rows]
    metric = score_method(compatible, truth, "guide-aware", guides_per_gene)
    metric.update(interval_coverage_for_method_estimand=None, informative_interval_fraction=0.,
                  mean_finite_interval_width=None)
    if any(row.get("posterior_qvalue") is not None for row in rows):
        metric["null_p_lt_005"] = None
    return metric
