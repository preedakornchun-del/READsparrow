"""Experimental covariance-envelope LOO plus sparse partial conjunction.

No target effects estimate nuisance correlation. Genuine matched negative
control blocks may narrow an exchangeable, nonnegative covariance envelope.
Without blocks the default envelope is [0,1], NOT independent guides.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import f as f_dist, t as student_t

from .empirical_inference import EmpiricalConfig, call_empirical_hits
from .guide_inference import adjust_pvalues, partial_conjunction
from .inference import CONTROL_GENES, fit_log_model, make_design
from .power_inference import combined_df, nuisance_variance, predict_measurement


@dataclass
class CorrelatedConfig(EmpiricalConfig):
    correlation_mode: str = "envelope"
    correlation_upper: float = 1.0
    correlation_error: float = 1e-5
    min_control_blocks: int = 40
    coherent_weight: float = .8

    def validate(self):
        super().validate()
        if self.null_model != "background" or self.minimum_support != 2:
            raise ValueError("Use background and a fixed two-guide support claim")
        if self.correlation_mode not in {"envelope", "matched-blocks"}:
            raise ValueError("Use envelope or matched-blocks correlation")
        if not 0 <= self.correlation_upper <= 1 or not 0 < self.coherent_weight < 1:
            raise ValueError("Correlation upper must be [0,1]; coherent weight strictly between zero and one")
        if not 0 < self.correlation_error <= .01 or self.min_control_blocks < 30:
            raise ValueError("Need correlation error in (0,.01] and >=30 matched null blocks")


def fit_correlation(context, table, blocks, config, declared_null_ids=None):
    """Upper ICC envelope from balanced matched null blocks, never targets.

    F intervals assume Gaussian exchangeable standardized block effects.
    Division by minimum nuisance fraction protects against attenuation by
    independent measurement noise. Nonnegative measurement covariance is an
    explicit assumption. This is a working-model CI, NOT distribution-free.
    """
    if config.correlation_mode == "envelope":
        if blocks:
            raise ValueError("Control blocks supplied but mode is not matched-blocks")
        return dict(mode="prespecified-envelope", lower=0., upper=config.correlation_upper,
                    error_budget=0., blocks=0, estimated=False)
    if not blocks:
        raise ValueError("Matched-block mode requires genuine experimentally matched null control blocks; no random NTC grouping")
    id_index = {g: i for i, g in enumerate(table.ids)}
    null_indices = set(context["train"].tolist()+context["calibration"].tolist())
    declared=set(declared_null_ids or [])
    all_null={i for i,(gid,gene) in enumerate(zip(table.ids,table.genes)) if gene in CONTROL_GENES or gid in declared}
    by_block = {}
    for gid, block in blocks.items():
        if gid not in id_index or id_index[gid] not in all_null:
            raise ValueError("Control block IDs must be known informative declared null controls")
        if not block:
            raise ValueError("Empty control block label")
        by_block.setdefault(block, []).append(id_index[gid])
    sizes = {len(v) for v in by_block.values()}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError("Need balanced matched null blocks with >=2 guides each and sufficient blocks; no automatic fallback")
    supplied_blocks=len(by_block)
    by_block={b:idx for b,idx in by_block.items() if all(i in null_indices for i in idx)}
    if len(by_block)<config.min_control_blocks:
        raise ValueError("Need sufficient blocks after whole-block information filtering; no automatic fallback")
    indices = np.array([sorted(by_block[b], key=lambda i: table.ids[i]) for b in sorted(by_block)])
    z = context["z"][indices]
    n, k = z.shape
    means = z.mean(axis=1)
    between = float(k * np.var(means, ddof=1))
    within = float(np.sum((z-means[:, None])**2)/(n*(k-1)))
    if within <= 1e-15:
        upper_total, point = 1., 1.
    else:
        ratio = between/within
        upper_ratio = ratio/f_dist.ppf(config.correlation_error, n-1, n*(k-1))
        upper_total = float(np.clip((upper_ratio-1)/(upper_ratio+k-1), 0., 1.))
        point = float(np.clip((ratio-1)/(ratio+k-1), 0., 1.))
    tau = np.array([b["extra_background_variance"] for b in context["bins"]])[context["assignment"]]
    total = context["prediction_se"]**2
    nuisance_fraction = tau[indices]/total[indices]
    minimum_fraction = float(nuisance_fraction.min())
    upper = 1. if minimum_fraction < .01 else min(1., upper_total/minimum_fraction)
    return dict(mode="matched-null-block-ICC-envelope", lower=0., upper=float(upper),
        error_budget=config.correlation_error, blocks=n, guides_per_block=k, estimated=True,
        supplied_blocks=supplied_blocks,excluded_low_information_blocks=supplied_blocks-n,
        observed_standardized_icc=point, total_icc_upper=upper_total,
        minimum_nuisance_variance_fraction=minimum_fraction,
        coverage_model="One-way Gaussian balanced random-intercept F upper interval; attenuation correction; approximate after fitted standardization",
        boundary="Matched control structure must represent target nuisance. Nonnegative measurement covariance, shared exchangeable nuisance rho and valid marginal background model required. Unequal covariance and fitted-scale uncertainty are not exactly covered.",
        control_use="All specified informative block controls fit covariance; former held-out marginal audits are NOT independent of this covariance fit",
        block_members={b:[table.ids[i] for i in sorted(v,key=lambda i:table.ids[i])] for b,v in sorted(by_block.items())})


def adaptive_directional(coherent, sparse, weight):
    """Prespecified weighted union; subset/sign choice is never uncorrected."""
    return min(1., coherent/weight, sparse/(1-weight))


def matched_null_population(context, table, correlation):
    """One vector per REAL null block, not all overlapping pseudo-subsets.

    Standardize by fitted guide nuisance SD, then retain the complete sample
    covariance. Remove modeled diagonal control measurement variance; retain
    off-diagonal control measurement covariance (conservative if nonnegative).
    A PSD projection prevents a negative nuisance variance. Target biological
    measurement covariance is then added separately, not counted as nuisance.
    All projected subsets use the same block population; df never exceeds
    independent block count minus one. Normal predictive tails are approximate.
    """
    if not correlation["estimated"]:
        return None
    id_index={gid:i for i,gid in enumerate(table.ids)}
    idx=np.array([[id_index[g] for g in gs] for gs in correlation["block_members"].values()])
    bins=context["bins"];assignment=context["assignment"]
    tau=np.array([b["extra_background_variance"] for b in bins])[assignment]
    if np.any(tau[idx]<=1e-8):
        correlation["predictive_status"]="No positive nuisance scale for every matched control; conservative envelope used, never division by zero"
        return None
    center=np.array([b["null_center"] for b in bins])[assignment]
    z=(context["beta"][idx]-center[idx])/np.sqrt(tau[idx])
    inflation=context["df"]/(context["df"]-2) if context["df"]>2 else 10.
    measurement_diag=np.mean(context["measurement_se"][idx]**2/tau[idx],axis=0)*inflation
    observed=np.cov(z,rowvar=False,ddof=1)
    raw=observed-np.diag(measurement_diag)
    eigenvalues,eigenvectors=np.linalg.eigh(raw)
    covariance=(eigenvectors*np.maximum(eigenvalues,0.))@eigenvectors.T
    correlation["predictive_status"]="matched block vector nuisance population; diagonal control measurement variance removed, off-diagonal measurement covariance retained"
    correlation["projection_population_fit"]=dict(blocks=len(idx),observed_covariance=observed.tolist(),
        modeled_measurement_diagonal=measurement_diag.tolist(),raw_eigenvalues=eigenvalues.tolist(),
        nuisance_covariance=covariance.tolist(),standardized_mean=z.mean(axis=0).tolist(),
        uncertainty_boundary="Method-of-moments noise subtraction and PSD projection plus Welch control df are working approximations, not exact Wishart/t inference; negative measurement covariance is outside the model")
    return dict(mean=z.mean(axis=0),covariance=covariance,blocks=len(idx),size=idx.shape[1])


def projected_nuisance(subset,tau,assignment,location_variances,population):
    j=len(subset)
    if j>population["size"]:
        raise ValueError("Matched null blocks have fewer guides than target subset; explicit matched structure required")
    a=np.sqrt(tau[assignment[subset]])/j
    cov=population["covariance"][:j,:j]
    shared_location=nuisance_variance(tau[assignment[subset]],assignment[subset],location_variances,0.)-np.sum(a*a)
    prediction_variance=float(a@cov@a)*(1+1/population["blocks"])+shared_location
    null_center=float(a@population["mean"][:j])
    return max(0.,prediction_variance),null_center


def call_correlated_hits(table, metadata, treatment, control, config=None, control_ids=None, control_blocks=None):
    config = config or CorrelatedConfig()
    config.validate()
    context = {}
    previous, info, guides = call_empirical_hits(table, metadata, treatment, control, config, control_ids, _context_sink=context)
    correlation = fit_correlation(context, table, control_blocks, config,control_ids)
    population=matched_null_population(context,table,correlation)
    correlation["main_variance_model"]="matched-projection-predictive" if population is not None else "correlation-upper-envelope"
    if population is not None:
        correlation["error_budget"]=0.
        correlation["icc_role"]="Diagnostic ONLY; main variance learned from matched null block projections, finite independent-block df"
    design, _, treated, residual_df, _ = make_design(table.samples, metadata, treatment, control)
    factors = np.array([info["normalization_factors"][s] for s in table.samples])
    normalized = table.counts/factors
    y = np.log2(normalized+config.pseudocount)
    guide_map = {r["sgRNA"]:r for r in guides}
    groups = {}
    for i,(gid,gene) in enumerate(zip(table.ids,table.genes)):
        if gid in guide_map:
            groups.setdefault(gene,[]).append(i)
    # Guide order is fixed by ID, never by input rows or observed efficacy.
    groups={g:sorted(idx,key=lambda i:table.ids[i]) for g,idx in groups.items()}
    gene_names = sorted(groups)
    gids = [guide_map.get(g) for g in table.ids]
    eligible = np.array([g["informative"] if g else False for g in gids])
    good = np.array([len(groups[g])>=max(2,config.min_guides) and eligible[groups[g]].all()
        and (normalized[groups[g]].sum(axis=0)>=config.min_count).sum()>=min(treated.sum(),(~treated).sum()) for g in gene_names])
    if good.sum()<20:
        raise ValueError("Fewer than 20 fully informative genes for covariance-aware measurement moderation")
    full_y = np.array([y[groups[g]].mean(axis=0) for g in gene_names])
    _,_,_,mean_df,moderation=fit_log_model(full_y[good],design,config.prior_df_cap)
    bins=context["bins"]
    tau=np.array([b["extra_background_variance"] for b in bins])
    loc=np.array([b["location_prediction_variance"] for b in bins])
    null_df=np.array([b["training_controls"]-1 for b in bins])
    assignment=context["assignment"]
    centers=np.array([b["null_center"] for b in bins])[assignment]
    features,sets,feature_ids=[],[],{}
    for gene in gene_names:
        idx=np.array(groups[gene])
        feature_ids[gene]=[]
        for removed in [None]+list(idx) if len(idx)>=2 else [None]:
            subset=idx if removed is None else idx[idx!=removed]
            feature_ids[gene].append(len(features));features.append(y[subset].mean(axis=0));sets.append(subset)
    raw,se=predict_measurement(np.array(features),design,moderation,residual_df)
    effect=raw-np.array([centers[s].mean() for s in sets])
    nuisance=np.array([nuisance_variance(tau[assignment[s]],assignment[s],loc,correlation["upper"]) for s in sets])
    nuisance_df=np.array([null_df[assignment[s]].min() for s in sets])
    if population is not None:
        projections=[projected_nuisance(s,tau,assignment,loc,population) for s in sets]
        nuisance=np.array([v[0] for v in projections]);effect-=np.array([v[1] for v in projections])
        nuisance_df=np.minimum(nuisance_df,population["blocks"]-1)
    dfs=combined_df(se**2,nuisance,mean_df,nuisance_df)
    scale=np.sqrt(se**2+nuisance)
    feature_pp=student_t.sf(effect/scale,dfs);feature_pn=student_t.cdf(effect/scale,dfs)
    # Sparse evidence uses individually valid marginal guide tests, which do not
    # require independence. Welch df replaces the old min-component shortcut.
    guide_nuisance=tau[assignment]+loc[assignment]
    guide_df=combined_df(context["measurement_se"]**2,guide_nuisance,context["df"],null_df[assignment])
    guide_pp=student_t.sf(context["z"],guide_df);guide_pn=student_t.cdf(context["z"],guide_df)
    guide_pp[~eligible]=1.;guide_pn[~eligible]=1.
    rows,audit=[],[]
    for old,is_good in zip(previous,good):
        row=dict(old);idx=np.array(groups[row["gene"]]);fi=feature_ids[row["gene"]];full,loo=fi[0],fi[1:]
        coherent_pp=min(1.,float(max(feature_pp[loo]))+correlation["error_budget"]) if loo else 1.
        coherent_pn=min(1.,float(max(feature_pn[loo]))+correlation["error_budget"]) if loo else 1.
        sparse_pp=partial_conjunction(guide_pp[idx],2);sparse_pn=partial_conjunction(guide_pn[idx],2)
        pp=adaptive_directional(coherent_pp,sparse_pp,config.coherent_weight)
        pn=adaptive_directional(coherent_pn,sparse_pn,config.coherent_weight)
        direction="enrichment" if pp<pn else "depletion" if pn<pp else "ambiguous"
        critical=student_t.ppf((1+config.confidence)/2,dfs[full])
        flags=[f for f in row["diagnostic_flags"].split(";") if f and f not in {"median_effect_direction_disagrees_with_test","opposing_directional_evidence"}]
        if direction=="enrichment" and effect[full]<=0 or direction=="depletion" and effect[full]>=0:
            flags.append("mean_effect_direction_disagrees_with_test")
        row.pop("pc_enrichment_pvalue",None);row.pop("pc_depletion_pvalue",None)
        row.update(tested=bool(is_good),status="tested" if is_good else "filtered_requires_all_annotated_guides_informative",
            required_support=2,log2_effect=float(effect[full]),effect_estimand="equal-weight mean guide contrast relative to fitted control background",
            standard_error=float(scale[full]) if is_good else None,ci_low=float(effect[full]-critical*scale[full]) if is_good else None,
            ci_high=float(effect[full]+critical*scale[full]) if is_good else None,ci_informative=bool(is_good),
            confidence_interval_role="Approximate full mean-effect CI, not robust to guide artifacts, not dual to adaptive support test; plug-in covariance bound",
            pvalue=min(1.,2*min(pp,pn)) if is_good else None,qvalue=None,hit=False,direction=direction if is_good else "not_tested",
            coherent_enrichment_pvalue=coherent_pp,coherent_depletion_pvalue=coherent_pn,
            sparse_enrichment_pvalue=sparse_pp,sparse_depletion_pvalue=sparse_pn,
            adaptive_enrichment_pvalue=pp,adaptive_depletion_pvalue=pn,
            nuisance_correlation_upper=correlation["upper"],nuisance_variance_of_mean=float(nuisance[full]),mean_model_df=float(dfs[full]),
            diagnostic_flags=";".join(flags))
        rows.append(row)
        for j in fi:
            removed=sorted(set(idx.tolist())-set(sets[j].tolist()))
            audit.append(dict(gene=row["gene"],omitted_guide=table.ids[removed[0]] if removed else "none",guides_in_subset=len(sets[j]),
                log2_effect=float(effect[j]),measurement_se=float(se[j]),nuisance_variance=float(nuisance[j]),prediction_se=float(scale[j]),
                df=float(dfs[j]),p_enrichment=float(feature_pp[j]),p_depletion=float(feature_pn[j]),used_in_coherent_component=bool(is_good and j!=full)))
    tested=[r for r in rows if r["tested"]]
    for row,q in zip(tested,adjust_pvalues([r["pvalue"] for r in tested],config.adjustment)):
        row["qvalue"],row["hit"]=float(q),bool(q<=config.fdr)
    id_index={gid:i for i,gid in enumerate(table.ids)}
    for g in guides:
        i=id_index[g["sgRNA"]]
        g.update(sparse_welch_df=float(guide_df[i]),sparse_p_enrichment=float(guide_pp[i]),sparse_p_depletion=float(guide_pn[i]),
            removed_from_test=False)
    info.update(method="covariance-aware-LOO-and-sparse-two-guide-weighted-union",release_status="experimental-correlated-alpha",
        config=asdict(config),tested_genes=len(tested),filtered_genes=len(rows)-len(tested),correlation_model=correlation,
        gene_measurement_moderation=moderation,residual_df_per_gene=residual_df,gene_moderated_df=mean_df,
        subset_audit=audit,effect_estimand="equal-weight mean fitted guide contrast, NOT effective-guide-only effect",
        robustness_claim="At least two guide associations, conditional on valid marginal guide and delete-one subset nulls; not >=half-guides support",
        guide_pvalue_role="Sparse branch uses Welch constituent p-values; previous guide p-values retained descriptively",
        no_universal_FDR_guarantee=True)
    info["warnings"]=[w for w in info["warnings"] if not any(s in w for s in ("Partial conjunction",">=required_support","Median guide effect","configured multi-guide claim"))]+[
        "Default envelope allows any common nonnegative exchangeable nuisance correlation in [0,1]; not independent guides",
        "Matched null blocks must reflect target-gene nuisance; arbitrary NTC grouping is invalid and may be anti-conservative",
        "Coherent and sparse evidence combined with prespecified weights, two directions corrected; no unpenalized guide selection",
        "Sparse fixed-r=2 partial conjunction tolerates ineffective guides but cannot identify efficacy or prove causality",
        "One arbitrary guide cannot alone establish the support claim IF constituent null models are valid; two-guide artifacts/CNV remain unresolved",
        "Matched projection covariance uses one vector per independent neutral block; guide positions/subsets are NOT extra null blocks or biological replicates",
        "Matched predictive-t/Welch tails and fallback ICC envelope are approximate after fitted heteroscedastic control standardization; no distribution-free or exact Berger-Boos guarantee",
        "All annotated guides must be informative; no outcome-dependent deletion. Inactive but well-counted guides are different from low-information guides",
        "Correlated common-mode nuisance and coherent true effects are not identifiable from one contrast without matched neutral structure; power loss may be unavoidable",
        "Per-gene approximate predictive mean CIs are not dual to adaptive p-values or robust to an artifact",
        "BH needs its usual dependence assumptions across genes; BY is available but cannot repair invalid constituent p-values"]
    return rows,info,guides
