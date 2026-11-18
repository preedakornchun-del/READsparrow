"""Paired frozen evaluation. Truth is only passed to scoring, never inference."""
from dataclasses import asdict,replace
from pathlib import Path
import time
import numpy as np
from .calibration import binomial_interval
from .correlated_inference import CorrelatedConfig,call_correlated_hits
from .empirical_inference import EmpiricalConfig,call_empirical_hits
from .io import CountTable,write_json,write_tsv,sha256
from .measurement_inference import MeasurementConfig,call_measurement_hits
from .shape_comparison import cases as previous_cases,generate as previous_generate
from .shape_inference import ShapeConfig,call_shape_hits,score_shape


def cases():
    result={n:dict(c,shared_measurement=False) for n,c in previous_cases().items()}
    for rho in (0.,.5,1.):
        base=result[f'matched_null_rho{rho:g}']
        result[f'few_controls_null_rho{rho:g}']=dict(base,config=replace(base['config'],controls=200))
    for tag in ('ineffective_rho0.5','ineffective_rho0.5_effect2'):
        base=result['matched_'+tag]
        result['few_controls_'+tag]=dict(base,config=replace(base['config'],controls=200))
    # Shared sample-wise measurement creates genuine target/reference covariance.
    # Effects and loadings are generated before counting, not inferred from hits.
    for tag in ('null_rho0','null_rho0.5','ineffective_rho0.5','ineffective_rho0.5_effect2'):
        result['shared_measurement_'+tag]=dict(result['matched_'+tag],shared_measurement=True)
    return result


def generate(config,case):
    table,meta,truth,blocks=previous_generate(config,case)
    if not case.get('shared_measurement'):
        return table,meta,truth,blocks
    rng=np.random.default_rng(config.seed+858301)
    target=config.genes*config.guides_per_gene
    shared=rng.normal(0,.45,len(table.samples))
    loading=np.r_[np.repeat(rng.uniform(.6,1.4,config.genes),config.guides_per_gene),rng.uniform(.6,1.4,config.controls)]
    x=rng.poisson(table.counts*np.exp2(loading[:,None]*shared[None,:]))
    return CountTable(table.ids,table.genes,table.samples,x),meta,truth,blocks


def summarize(records,case):
    results=[]
    for method in dict.fromkeys(r['method'] for r in records):
        group=[r for r in records if r['method']==method];ok=[r for r in group if r['status']=='ok']
        row=dict(scenario=group[0]['scenario'],method=method,repeats=len(group),valid_repetitions=len(ok),unsupported_repetitions=len(group)-len(ok))
        keys=list(dict.fromkeys(k for r in ok for k,v in r.items() if isinstance(v,(int,float)) and k not in {'seed','repetition'}))
        for key in keys:
            values=[r[key] for r in ok if r.get(key) is not None]
            row[key]=float(np.mean(values)) if values else None
        for key in ('fdp','recall','support_claim_fdp','support_claim_recall'):
            values=[r[key] for r in ok if r.get(key) is not None]
            row[key+'_mc_se']=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        if case['config'].active_fraction==0:
            row['all_null_exact_95_interval']=binomial_interval(sum(r['any_hit'] for r in ok),len(ok)) if ok else None
        results.append(row)
    return results


def run_suite(out,names,repeats=50,null_repeats=100,genes=1000,seed=1900000000):
    all_cases=cases();records=[];summaries=[];started=time.perf_counter()
    write_json(out/'protocol.json',dict(seed=seed,repeats=repeats,null_repeats=null_repeats,genes=genes,
        scenarios={n:dict(config=asdict(all_cases[n]['config']),normalization=all_cases[n]['normalization'],matched=all_cases[n]['matched'],shared_measurement=all_cases[n]['shared_measurement']) for n in names},
        seed_rule='base + global scenario index*100000 + rep; focused median/control retain shared index14',
        source_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        methods='v06, v07, v08; matched ONLY when genuine simulated neutral blocks are supplied, otherwise conservative envelope/guard',
        primary='Mean biological FDP and all-active recall at decision<=.05; all-null FDP=P(any hit); support>=2 truth also scored',
        boundary='Previously seen generator families and HQ461; fresh seeds are not new external biology. Posterior decision values are NOT BH q-values.',
        config=asdict(MeasurementConfig())))
    for name in names:
        case=all_cases[name];n=null_repeats if case['config'].active_fraction==0 else repeats
        index=14 if name.startswith('focused_shift') else list(all_cases).index(name)
        current=[]
        for rep in range(n):
            cfg=replace(case['config'],genes=genes,seed=seed+index*100000+rep)
            table,meta,truth,blocks=generate(cfg,case)
            prepared=None
            for method in ('v06','v07','v08'):
                tick=time.perf_counter()
                try:
                    options=dict(normalization=case['normalization'],correlation_mode='matched-blocks' if case['matched'] else 'envelope')
                    if method=='v06':
                        rows,info,_=call_correlated_hits(table,meta,'treatment','control',CorrelatedConfig(**options),control_blocks=blocks)
                    else:
                        if prepared is None:
                            ctx={};a,b,c=call_empirical_hits(table,meta,'treatment','control',EmpiricalConfig(normalization=case['normalization']),_context_sink=ctx)
                            prepared=(a,b,c,ctx)
                        if method=='v07':
                            rows,info,_=call_shape_hits(table,meta,'treatment','control',ShapeConfig(**options),control_blocks=blocks,_prepared=prepared)
                        else:
                            rows,info,_=call_measurement_hits(table,meta,'treatment','control',MeasurementConfig(**options),control_blocks=blocks,_prepared=prepared)
                    metric=score_shape(rows,truth,cfg.guides_per_gene)
                    record=dict(scenario=name,method=method,repetition=rep,seed=cfg.seed,status='ok',**metric,
                        seconds=time.perf_counter()-tick,branch=info.get('inference_branch',info.get('method')),
                        shape_family=info.get('shape_model',{}).get('family'),
                        posterior_prevalence=(info.get('posterior_model') or {}).get('estimated_alternative_prevalence'),
                        em_converged=(info.get('posterior_model') or {}).get('em_converged'))
                except ValueError as exc:
                    if not any(s in str(exc) for s in ('informative declared null controls','fully informative','sufficient blocks','No genes pass','Insufficient measurement df')):
                        raise
                    record=dict(scenario=name,method=method,repetition=rep,seed=cfg.seed,status='unsupported',error=str(exc))
                records.append(record);current.append(record)
            if (rep+1)%10==0:print(dict(scenario=name,completed=rep+1,total=n,seconds=round(time.perf_counter()-started,1)),flush=True)
        summaries.extend(summarize(current,case))
        write_tsv(out/'runs.tsv',list(dict.fromkeys(k for r in records for k in r)),records)
        write_json(out/'summary.json',dict(scenarios=summaries,complete=False))
        for summary in summaries[-3:]:print(summary,flush=True)
    write_json(out/'summary.json',dict(scenarios=summaries,complete=True,attempted_decisions=len(records),
        scenario_screens=len({(r['scenario'],r['seed']) for r in records}),seconds=time.perf_counter()-started,production_validated=False))
