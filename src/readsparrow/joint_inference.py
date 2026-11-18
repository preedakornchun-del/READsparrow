"""Experimental v0.9 shared-null model; frozen v0.8 remains callable."""
import copy
from dataclasses import asdict,dataclass
import numpy as np
from .measurement_inference import (MeasurementConfig,measurement_covariances,
    counting_noise_floor,reference_residuals,refit_measurement)
from .correlated_inference import fit_correlation
from .empirical_inference import call_empirical_hits
from .inference import make_design
from .shape_inference import call_shape_hits,fit_shape
from .joint_null import SharedNullLikelihood


@dataclass
class JointConfig(MeasurementConfig):
    shared_draws:int=128
    shared_seed:int=270901

    def validate(self):
        super().validate()
        if not 8<=self.shared_draws<=256:raise ValueError('Need 8..256 shared parameter draws')
        if self.minimum_support!=2 or self.support_fraction!=.5:
            raise ValueError('Joint-null model has a fixed >=2 coherent-guide claim; support overrides are unsupported')


def call_joint_hits(table,metadata,treatment,control,config=None,control_ids=None,control_blocks=None,*,_prepared=None):
    config=config or JointConfig();config.validate()
    if _prepared is None:
        context={};old,info,guides=call_empirical_hits(table,metadata,treatment,control,config,control_ids,_context_sink=context)
    else:old,info,guides,context=copy.deepcopy(_prepared)
    shape=fit_shape(context['z'][context['train']])
    if config.correlation_mode!='matched-blocks' or shape['family']!='Gaussian':
        rows,info,guides=call_shape_hits(table,metadata,treatment,control,config,control_ids,control_blocks,
            _prepared=(old,info,guides,context))
        info.update(method='joint-null-guarded-fallback',config=asdict(config),release_status='experimental-joint-null-alpha',
            joint_null_model=None,fallback_reason='No genuine matched blocks' if config.correlation_mode!='matched-blocks' else 'Training-control BIC rejects Gaussian nuisance branch')
        return rows,info,guides
    blocks=fit_correlation(context,table,control_blocks,config,control_ids)
    index={g:i for i,g in enumerate(table.ids)}
    ci=np.array([[index[g] for g in ids] for ids in blocks['block_members'].values()])
    design,_,_,residual_df,_=make_design(table.samples,metadata,treatment,control)
    factors=np.array([info['normalization_factors'][s] for s in table.samples])
    y=np.log2(table.counts/factors+config.pseudocount)
    coeff=np.linalg.lstsq(design,y.T,rcond=None)[0].T
    residuals=y-coeff@design.T;uc=np.unique(ci)
    center=float(context['beta'][uc].mean())
    residuals,_=reference_residuals(residuals,np.zeros(len(table.ids),dtype=int),uc)
    lookup={r['sgRNA']:r for r in guides};ti=np.array([i for i,g in enumerate(table.ids) if g in lookup])
    tse,tdf,tprior=refit_measurement(residuals,y.mean(axis=1),ti,design,residual_df,config.prior_df_cap)
    cse,cdf,cprior=refit_measurement(residuals,y.mean(axis=1),uc,design,residual_df,config.prior_df_cap)
    if min(tdf,cdf)<=4:raise ValueError('Insufficient measurement df')
    se=context['measurement_se'].copy();se[ti]=tse;se[uc]=cse
    cm,cpool=measurement_covariances(residuals,ci,se,residual_df,cdf,config.measurement_correlation_prior)
    cm=counting_noise_floor(cm,ci,table.counts,design,config.pseudocount)
    groups={}
    for i,gid in enumerate(table.ids):
        if gid in lookup:groups.setdefault(table.genes[i],[]).append(i)
    groups={g:sorted(v,key=lambda i:table.ids[i]) for g,v in groups.items()}
    rows=copy.deepcopy(old)
    selected=[r for r in rows if r['tested'] and all(lookup[table.ids[i]]['informative'] for i in groups[r['gene']])]
    if len(selected)<20:raise ValueError('Fewer than 20 fully informative genes')
    inputs=[];ordered=[];audit=[];pools={}
    for m in sorted({len(groups[r['gene']]) for r in selected}):
        if not 2<=m<=min(config.max_scan_guides,ci.shape[1]):raise ValueError('Need matched blocks with at least target guide count, target 2..8 guides')
        current=[r for r in selected if len(groups[r['gene']])==m];indices=np.array([groups[r['gene']] for r in current])
        tm,pool=measurement_covariances(residuals,indices,se,residual_df,tdf,config.measurement_correlation_prior)
        tm=counting_noise_floor(tm,indices,table.counts,design,config.pseudocount)
        inputs.append((context['beta'][indices]-center,tm));ordered.extend(current);pools[str(m)]=pool.tolist()
        audit.extend(dict(gene=r['gene'],target_measurement_covariance=matrix.tolist()) for r,matrix in zip(current,tm))
    engine=SharedNullLikelihood(inputs,context['beta'][ci]-center,cm)
    local,q,posterior=engine.decide(config.fdr,config.shared_draws,config.shared_seed)
    selected_names={r['gene'] for r in ordered}
    for r in rows:
        for key in ('pc_enrichment_pvalue','pc_depletion_pvalue','standard_error'):r.pop(key,None)
        r.update(tested=r['gene'] in selected_names,pvalue=None,qvalue=None,hit=False,
            posterior_null_probability=None,posterior_qvalue=None,decision_value=None,
            ci_low=None,ci_high=None,ci_informative=False,
            confidence_interval_role='Not supplied for sparse support; effect is descriptive median',
            direction_role='Descriptive median sign, not directional posterior inference')
        r['status']='tested' if r['tested'] else 'filtered_requires_all_annotated_guides_informative'
    for r,p,d in zip(ordered,local,q):
        r.update(posterior_null_probability=float(p),posterior_qvalue=float(d),decision_value=float(d),qvalue=float(d),
            hit=bool(d<=config.fdr),direction='enrichment' if r['log2_effect']>0 else 'depletion')
    info.update(method='shared-null-joint-sparse-mixture',config=asdict(config),release_status='experimental-joint-null-alpha',
        inference_branch='raw-effect-shared-null-joint-posterior',fdr_family=posterior['family'],shape_model=shape,
        confidence_interval='Not supplied: descriptive median effect only',posterior_model=posterior,
        joint_null_model=dict(control_blocks=len(ci),block_size=ci.shape[1],exchangeability_assumption=True,
            center_model='Shared reference shift estimated in joint target/control likelihood; measurement reference covariance retained by residualization',
            covariance='Two nonnegative exchangeable nuisance eigenvariances, distinct from heteroscedastic measurement covariance',
            measurement_approximation='Gaussian with moderated covariance (moment matching); measurement-tail uncertainty not separately integrated',
            alternative='Continuous nonlocal delta^2*N(0,1) coherent effect; learned shared all-guide versus partial-subset fraction; single half nonlocal/half N(0,16)',
            target_use='Full target mixture informs shared nuisance parameters: NOT control-only null estimation'),
        measurement_model=dict(target_correlation=pools,control_correlation=cpool.tolist(),biological_residual_df=residual_df,
            target_measurement_df=float(tdf),control_measurement_df=float(cdf),target_moderation=tprior,control_moderation=cprior),
        subset_audit=audit,tested_genes=len(ordered),filtered_genes=len(rows)-len(ordered),
        robustness_claim='Model-dependent coherent >=2 support, not arbitrary artifact rejection',no_universal_FDR_guarantee=True,
        legacy_background_diagnostics_role='Inherited p-values and descriptive median effects refer to old guide/background layer, not this joint inference')
    posterior['hits']=sum(r['hit'] for r in ordered)
    return rows,info,guides
