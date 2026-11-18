"""Experimental raw-effect predictive mixture with separate measurement models.

Biological samples determine measurement covariance; neutral blocks determine
persistent nuisance. A measurement-only inverse-chi-square scale mixture and
whole-block bootstrap integrate uncertainty BEFORE computing density ratios.
These are working approximations, not distribution-free FDR guarantees.
"""
from dataclasses import asdict, dataclass
import copy
from itertools import combinations

import numpy as np
from scipy.special import gammaln, logsumexp, roots_genlaguerre

from .correlated_inference import fit_correlation
from .empirical_inference import call_empirical_hits, predictive_background
from .inference import fit_log_model, make_design, moderate_variances
from .shape_inference import ShapeConfig, call_shape_hits, fit_shape
from .sparse_posterior import three_class_decisions


@dataclass
class MeasurementConfig(ShapeConfig):
    null_draws: int = 12
    null_seed: int = 18082701
    measurement_correlation_prior: float = 12.
    measurement_quadrature: int = 7
    posterior_safety_factor: float = 1.

    def validate(self):
        super().validate()
        if not 4 <= self.null_draws <= 64 or self.measurement_correlation_prior < 2 or not 3 <= self.measurement_quadrature <= 15:
            raise ValueError("Need 4..64 null draws and correlation shrinkage >=2")


def psd(matrix, floor=1e-8):
    matrix = (np.asarray(matrix) + np.asarray(matrix).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def correlation(matrix):
    cov = psd(matrix)
    d = np.sqrt(np.diag(cov))
    return cov / d[:, None] / d[None, :]


def residual_correlation(residuals, indices, df):
    """Pool residual correlations, not condition effects or apparent hits."""
    r = residuals[np.asarray(indices)]
    rms = np.sqrt(np.maximum(np.sum(r*r, axis=2)/df, 1e-12))
    standardized = r/rms[:, :, None]
    return correlation(np.einsum("bik,bjk->ij", standardized, standardized)/(len(r)*df))


def measurement_covariances(residuals, indices, se, df, total_df, prior_weight):
    pool = residual_correlation(residuals, indices, df)
    outputs = []
    for idx in indices:
        r = residuals[idx]
        local = correlation(r @ r.T / df)
        shrunk = (prior_weight*pool + df*local)/(prior_weight+df)
        # SE is a Student SCALE, not its variance. Preserve this distinction.
        d = se[idx] * np.sqrt(total_df/(total_df-2) if total_df > 2 else 10.)
        outputs.append(shrunk * d[:, None] * d[None, :])
    return np.asarray(outputs), pool


def counting_noise_floor(covariances, indices, counts, design, pseudocount):
    """Retain independent read-sampling noise when correlations approach one.

    Delta-method log-count Poisson diagonal, with 0.5-count stabilization.
    Project only the remaining covariance onto PSD: correlated biological
    noise cannot cancel independent read sampling below this explicit floor.
    This is an approximation for low counts, not a new Poisson likelihood.
    """
    weights=(np.linalg.inv(design.T@design)@design.T)[1]
    diagonal=np.sum(weights[None,:]**2/np.maximum(counts+pseudocount,.5),axis=1)/np.log(2.)**2
    out=[]
    for matrix,idx in zip(covariances,indices):
        floor=np.diag(diagonal[idx])
        out.append(floor+psd(matrix-floor,floor=0.))
    return np.asarray(out)


def reference_residuals(residuals, assignment, training):
    """Subtract the SAME sample-level control reference used for centering.

    Var(target-reference)=Var(target)+Var(reference)-Cov(target,reference)
    -Cov(reference,target). All four terms are retained by residual subtraction,
    including normalization/shared-sample covariance. No control covariance is
    automatically declared zero. Reference controls have the actual self-use;
    the block bootstrap models nuisance-center error separately.
    """
    reference = np.zeros_like(residuals)
    for b in np.unique(assignment):
        idx = training[assignment[training] == b]
        reference[assignment == b] = residuals[idx].mean(axis=0)
    return residuals-reference, reference


def refit_measurement(residuals, abundance, indices, design, residual_df, cap):
    variance = np.sum(residuals[indices]**2,axis=1)/residual_df
    posterior, prior = moderate_variances(variance,abundance[indices],residual_df,cap)
    se = np.sqrt(posterior*np.linalg.inv(design.T@design)[1,1])
    df = min(residual_df+prior["prior_df"],len(indices)*residual_df)
    return se, df, prior


def predictive_logpdf(x, means, covariances, df):
    """Batched multivariate Student log density; covariance is true variance."""
    x = np.asarray(x)
    if np.any(np.asarray(df) <= 2):
        raise ValueError("Predictive degrees of freedom must exceed two")
    scale = covariances * ((np.asarray(df)-2)/np.asarray(df))[:, None, None]
    precision = np.linalg.inv(scale)
    delta = x-means
    quad = np.einsum("ni,nij,nj->n", delta, precision, delta)
    m = x.shape[1]
    return (gammaln((df+m)/2)-gammaln(df/2)-m/2*np.log(df*np.pi)
            -.5*np.linalg.slogdet(scale)[1]-(df+m)/2*np.log1p(quad/df))


def component_densities(x, mean, covariance, df):
    """Clean, broad single association, and sparse coherent multi association.

    Fixed log2-effect priors; guide efficacy, simulated truth and HQ outcomes
    never enter the fit. Single and multi classes use the SAME null density.
    """
    n, m = x.shape
    scale = covariance*((df-2)/df)[:, None, None]
    precision = np.linalg.inv(scale)
    delta = x-mean
    pz = np.einsum("nij,nj->ni", precision, delta)
    quad = np.einsum("ni,ni->n", pz, delta)
    clean = predictive_logpdf(x, mean, covariance, df)
    def shifted(mask, effect):
        projection = pz @ mask
        norm = np.einsum("i,nij,j->n", mask, precision, mask)
        q = np.maximum(0., quad-2*effect*projection+effect*effect*norm)
        return clean - (df+m)/2*(np.log1p(q/df)-np.log1p(quad/df))
    single = []
    for j in range(m):
        mask = np.eye(m)[j]
        broad = covariance.copy(); broad[:, j, j] += 16.
        single.append(predictive_logpdf(x, mean, broad, df)-np.log(2*m))
        for effect in (.75, 1.5, 2.5, 4., 6.):
            for sign in (-1., 1.):
                single.append(shifted(mask, sign*effect)-np.log(20*m))
    multi = []
    for cardinality in range(2, m+1):
        subsets = list(combinations(range(m), cardinality))
        weight = np.log((m-1)*len(subsets)*8)
        for subset in subsets:
            mask = np.zeros(m); mask[list(subset)] = 1.
            for effect in (.75, 1.5, 2.5, 4.):
                for sign in (-1., 1.):
                    multi.append(shifted(mask, sign*effect)-weight)
    return clean, logsumexp(single, axis=0), logsumexp(multi, axis=0)


def gaussian_components(x, mean, covariance):
    """Exact conditional Gaussian densities for every sparse class."""
    n,m = x.shape
    precision = np.linalg.inv(covariance)
    delta = x-mean
    pz = np.einsum("nij,nj->ni",precision,delta)
    quad = np.einsum("ni,ni->n",pz,delta)
    clean = -.5*(m*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]+quad)
    single = []
    diagonal = np.diagonal(precision,axis1=1,axis2=2)
    artifact = clean[:,None]+.5*16*pz*pz/(1+16*diagonal)-.5*np.log1p(16*diagonal)
    single.append(logsumexp(artifact,axis=1)-np.log(2*m))
    for effect in (.75,1.5,2.5,4.,6.):
        for sign in (-1.,1.):
            single.append(logsumexp(clean[:,None]+sign*effect*pz-.5*effect**2*diagonal,axis=1)-np.log(20*m))
    multi=[]
    for k in range(2,m+1):
        masks=np.array([[float(j in subset) for j in range(m)] for subset in combinations(range(m),k)])
        projection=pz@masks.T
        norm=np.einsum('ki,nij,kj->nk',masks,precision,masks)
        weight=np.log((m-1)*len(masks)*8)
        for effect in (.75,1.5,2.5,4.):
            for sign in (-1.,1.):
                multi.append(logsumexp(clean[:,None]+sign*effect*projection-.5*effect**2*norm,axis=1)-weight)
    return np.array([clean,logsumexp(single,axis=0),logsumexp(multi,axis=0)])


def measurement_mixture(df, points):
    """Inverse-chi-square scale mixture acts on MEASUREMENT only.

    For target measurement variance M, conditional variance is
    M*(df-2)/ChiSquare(df). Nuisance remains an independent Gaussian block
    component with bootstrap parameter uncertainty; it is not t-inflated.
    """
    nodes,weights=roots_genlaguerre(points,df/2-1)
    weights=weights/np.exp(gammaln(df/2))
    return (df-2)/(2*nodes),np.log(weights)


def nuisance_draws(values, measurement, draws, seed):
    """Whole-block bootstrap, subtract each block's OWN measurement matrix.

    Include a point fit plus deterministic bootstrap fits. Estimated nuisance
    center uncertainty is added as covariance/n, never independent per guide.
    PSD projection is an approximation near the zero-nuisance boundary.
    """
    rng = np.random.default_rng(seed)
    n, m = values.shape
    outputs = []
    for draw in range(draws):
        idx = np.arange(n) if draw == 0 else rng.integers(0, n, n)
        v = values[idx]
        observed = np.cov(v, rowvar=False, ddof=1)
        nuisance = psd(observed-measurement[idx].mean(axis=0), floor=0.)
        # Exchangeable guide positions: the same mean is shared by every guide.
        # The actual reference mean is subtracted from target sample residuals.
        # Do NOT resample its measurement error a second time as a random mean.
        # Persistent-nuisance center uncertainty is integrated analytically.
        center = np.zeros(m)
        # Reference MEASUREMENT error is already in residualized target noise.
        # Only persistent nuisance-center uncertainty belongs here.
        center_cov = np.ones((m, m))*float(nuisance.sum()/(m*m*n))
        outputs.append((center, nuisance, center_cov))
    return outputs


def call_measurement_hits(table, metadata, treatment, control, config=None,
                          control_ids=None, control_blocks=None, *, _prepared=None):
    config = config or MeasurementConfig()
    config.validate()
    if _prepared is None:
        context = {}
        old, info, guides = call_empirical_hits(table, metadata, treatment, control,
                                               config, control_ids, _context_sink=context)
        prepared = (old, info, guides, context)
    else:
        prepared = copy.deepcopy(_prepared)
        old, info, guides, context = prepared
    shape = fit_shape(context["z"][context["train"]])
    # Non-Gaussian controls or absent real blocks retain the frozen v0.7 guard.
    # No made-up neutral groups and no Gaussian likelihood on transformed SEs.
    if config.correlation_mode != "matched-blocks" or shape["family"] != "Gaussian":
        rows, info, guides = call_shape_hits(table, metadata, treatment, control,
            config, control_ids, control_blocks, _prepared=prepared)
        info.update(method="measurement-aware-guarded-fallback", config=asdict(config),
            release_status="experimental-measurement-alpha",
            measurement_model=None, null_uncertainty_model=None,
            fallback_reason="No genuine matched blocks" if config.correlation_mode != "matched-blocks" else "Training-control BIC rejects Gaussian nuisance branch",
            confidence_interval="Not supplied: descriptive median effect only",
            legacy_background_diagnostics_role="Inherited guide/background audits refer to the v0.4 layer, not final gene inference")
        return rows, info, guides
    blocks = fit_correlation(context, table, control_blocks, config, control_ids)
    index = {g:i for i,g in enumerate(table.ids)}
    control_idx = np.array([[index[g] for g in ids] for ids in blocks["block_members"].values()])
    design, _, _, residual_df, _ = make_design(table.samples, metadata, treatment, control)
    factors = np.array([info["normalization_factors"][s] for s in table.samples])
    y = np.log2(table.counts/factors+config.pseudocount)
    coefficients = np.linalg.lstsq(design, y.T, rcond=None)[0].T
    residuals = y-coefficients@design.T
    # The control measurement prior is fitted separately, NOT borrowed from
    # lower/higher-noise targets. Targets keep the frozen EB scale predictions.
    unique_controls = np.unique(control_idx)
    _, control_se, _, control_df, control_prior = fit_log_model(y[unique_controls], design, config.prior_df_cap)
    se = context["measurement_se"].copy(); se[unique_controls] = control_se
    # One global persistent-nuisance distribution. Unlike measurement noise,
    # its covariance is assumed abundance-invariant in this explicit branch.
    # Raw units avoid fixed noisy per-bin scales and near-zero tau division.
    assignment=np.zeros(len(table.ids),dtype=int)
    center=np.repeat(float(context["beta"][unique_controls].mean()),len(table.ids))
    bins=[dict(bin=0,training_controls=len(unique_controls),null_center=float(center[0]),
        model="Global raw-log2 nuisance covariance; abundance dependence is a limitation, not silently fitted from targets")]
    raw_residuals = residuals.copy()
    residuals, reference = reference_residuals(residuals,assignment,unique_controls)
    target_ids = {r["sgRNA"] for r in guides}
    target_indices = np.array([i for i,gid in enumerate(table.ids) if gid in target_ids])
    target_se,target_df,target_prior = refit_measurement(residuals,y.mean(axis=1),
        target_indices,design,residual_df,config.prior_df_cap)
    control_se,control_df,control_prior = refit_measurement(residuals,y.mean(axis=1),
        unique_controls,design,residual_df,config.prior_df_cap)
    if min(target_df,control_df) <= 4:
        raise ValueError("Insufficient measurement df for stable variance-mixture prediction (>4 required)")
    se[target_indices] = target_se; se[unique_controls] = control_se
    control_cov, control_pool = measurement_covariances(residuals, control_idx, se,
        residual_df, control_df, config.measurement_correlation_prior)
    control_cov=counting_noise_floor(control_cov,control_idx,table.counts,design,config.pseudocount)
    nuisance_sd = np.ones(len(table.ids))
    values = (context["beta"][control_idx]-center[control_idx])/nuisance_sd[control_idx]
    measurement_standard = control_cov/nuisance_sd[control_idx][:,:,None]/nuisance_sd[control_idx][:,None,:]
    draws = nuisance_draws(values, measurement_standard, config.null_draws, config.null_seed)
    lookup = {r["sgRNA"]:r for r in guides}
    groups = {}
    for i,gid in enumerate(table.ids):
        if gid in lookup:
            groups.setdefault(table.genes[i], []).append(i)
    groups = {g:sorted(v,key=lambda i:table.ids[i]) for g,v in groups.items()}
    rows = copy.deepcopy(old)
    selected = [r for r in rows if r["tested"] and all(lookup[table.ids[i]]["informative"] for i in groups[r["gene"]])]
    if len(selected) < 20:
        raise ValueError("Fewer than 20 fully informative genes")
    audit = []; pools = {}
    for m in sorted({len(groups[r["gene"]]) for r in selected}):
        if m > config.max_scan_guides or m > control_idx.shape[1]:
            raise ValueError("Need matched blocks with at least target guide count, target <=8 guides")
        current = [r for r in selected if len(groups[r["gene"]]) == m]
        indices = np.array([groups[r["gene"]] for r in current])
        target_cov, pool = measurement_covariances(residuals, indices, se,
            residual_df, target_df, config.measurement_correlation_prior)
        target_cov=counting_noise_floor(target_cov,indices,table.counts,design,config.pseudocount)
        pools[str(m)] = pool.tolist()
        x = context["beta"][indices]-center[indices]
        amplitudes = nuisance_sd[indices]
        densities = []
        scales,logweights = measurement_mixture(target_df,config.measurement_quadrature)
        for mu, nuisance, center_cov in draws:
            nvar = (nuisance[:m,:m]+center_cov[:m,:m])*amplitudes[:,:,None]*amplitudes[:,None,:]
            conditional=[]
            for scale,logweight in zip(scales,logweights):
                total = target_cov*scale+nvar+np.eye(m)[None,:,:]*1e-8
                conditional.append(gaussian_components(x,amplitudes*mu[:m],total)+logweight)
            densities.append(logsumexp(conditional,axis=0))
        integrated = logsumexp(np.asarray(densities),axis=0)-np.log(len(draws))
        single, multi = integrated[1]-integrated[0], integrated[2]-integrated[0]
        for j,(r,a,b) in enumerate(zip(current,single,multi)):
            r.update(log_single_density_ratio=float(a),log_multi_density_ratio=float(b))
            audit.append(dict(gene=r["gene"], target_measurement_covariance=target_cov[j].tolist(),
                nuisance_amplitudes=amplitudes[j].tolist(), null_draws=len(draws)))
    local, qvalues, posterior = three_class_decisions([r["log_single_density_ratio"] for r in selected],
        [r["log_multi_density_ratio"] for r in selected], config.fdr)
    selected_names = {r["gene"] for r in selected}
    for r in rows:
        for key in ("pc_enrichment_pvalue", "pc_depletion_pvalue", "standard_error"):
            r.pop(key,None)
        r.update(tested=r["gene"] in selected_names, pvalue=None,qvalue=None, hit=False,
            posterior_null_probability=None,posterior_qvalue=None,decision_value=None,
            ci_low=None,ci_high=None,ci_informative=False,
            confidence_interval_role="Not supplied for sparse support; effect is descriptive median",
            direction_role="Descriptive median sign, not directional posterior inference")
        r["status"]="tested" if r["tested"] else "filtered_requires_all_annotated_guides_informative"
    for r,p,q in zip(selected,local,qvalues):
        decision = min(1.,config.posterior_safety_factor*float(q)) if posterior["global_gate_pass"] else 1.
        r.update(posterior_null_probability=float(p),posterior_qvalue=float(q),decision_value=decision,
            qvalue=decision,hit=bool(decision<=config.fdr),direction="enrichment" if r["log2_effect"]>0 else "depletion")
    info.update(method="measurement-separated-uncertain-null-sparse-mixture",config=asdict(config),
        release_status="experimental-measurement-alpha",
        inference_branch="raw-effect-measurement-aware-predictive-posterior",
        fdr_family="Safety-scaled posterior expected FDP; NOT BH q-values or guaranteed frequentist FDR",
        confidence_interval="Not supplied: descriptive median effect only",shape_model=shape,
        measurement_model=dict(target_correlation=pools,control_correlation=control_pool.tolist(),
            target_measurement_df=float(target_df),control_measurement_df=float(control_df),
            biological_residual_df=residual_df, control_moderation=control_prior,target_moderation=target_prior,
            target_reference_mean_cross_product=float(np.mean(np.sum(raw_residuals[target_indices]*reference[target_indices],axis=1)/residual_df)),
            reference_covariance_handling="Residualize target and controls by actual sample-level all-block-control reference before covariance estimation; retains both cross terms. Measurement reference error is NOT bootstrapped a second time.",
            correlation_prior_weight=config.measurement_correlation_prior,
            independent_counting_noise_floor="Delta-method Poisson diagonal + PSD projection of remaining measurement covariance",
            estimation="Residual-only separate target/control pooled correlations, shrunk gene correlation; per-guide moderated Student-scale SE converted to variance"),
        null_uncertainty_model=dict(draws=len(draws),seed=config.null_seed,independent_blocks=len(control_idx),
            point_nuisance_covariance=draws[0][1].tolist(),point_center_covariance=draws[0][2].tolist(),
            approach="Whole-neutral-block bootstrap covariance/center mixture convolved with target-only inverse-chi-square measurement uncertainty",
            quadrature_points=config.measurement_quadrature,
            boundary="Approximate bootstrap + measurement-scale model, not exact posterior, selective FDR theorem, or arbitrary-tail coverage"),
        background_model_v08=dict(bins=bins), posterior_model=posterior,subset_audit=audit,
        tested_genes=len(selected),filtered_genes=len(rows)-len(selected),
        legacy_background_diagnostics_role="Inherited guide/background audits refer to v0.4; raw measurement and final decision exported separately",
        robustness_claim="Model-dependent competition of >=2 coherent-guide effects against clean/single class; no arbitrary-outlier guarantee",
        no_universal_FDR_guarantee=True)
    posterior.update(safety_factor=config.posterior_safety_factor,
        decision_role="qvalue is a compatibility alias for decision_value; posterior_qvalue is NOT a BH q-value",
        hits=sum(r["hit"] for r in selected))
    for guide in guides:
        i=index[guide["sgRNA"]]
        guide.update(v08_reference_residual_measurement_se=float(se[i]),
            v08_nuisance_sd=float(nuisance_sd[i]),
            v08_measurement_role="Reference-residualized SE; prior p_enrichment/p_depletion are legacy diagnostics, not new guide tests")
    return rows, info, guides
