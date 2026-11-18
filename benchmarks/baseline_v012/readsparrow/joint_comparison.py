"""Paired v0.7/v0.8/v0.9 calibration; truth enters scoring only."""
from dataclasses import replace,asdict
from pathlib import Path
import time
from .measurement_comparison import cases,generate,summarize
from .empirical_inference import call_empirical_hits,EmpiricalConfig
from .shape_inference import call_shape_hits,ShapeConfig,score_shape
from .measurement_inference import call_measurement_hits,MeasurementConfig
from .joint_inference import call_joint_hits,JointConfig
from .io import new_output_dir,write_json,write_tsv,save_counts,sha256

METHODS=('v07','v08','v09')
REPLAY_CASES=('matched_null_rho0','matched_null_rho1','matched_ineffective_rho0.5',
    'matched_ineffective_rho0.5_effect2','few_controls_ineffective_rho0.5',
    'few_controls_ineffective_rho0.5_effect2','shared_measurement_null_rho0.5','nb_mixture_n3')

def infer(table,meta,case,blocks,method,prepared=None,config_override=None):
    options=dict(normalization=case['normalization'],correlation_mode='matched-blocks' if case['matched'] else 'envelope')
    cls,fn={'v07':(ShapeConfig,call_shape_hits),'v08':(MeasurementConfig,call_measurement_hits),'v09':(JointConfig,call_joint_hits)}[method]
    return fn(table,meta,'treatment','control',config_override or cls(**options),control_blocks=blocks,_prepared=prepared)

def run_suite(out,names,repeats=50,null_repeats=100,genes=1000,seed=2300000000):
    all_cases=cases();records=[];summaries=[]
    sources={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    write_json(out/'protocol.json',dict(seed=seed,repeats=repeats,null_repeats=null_repeats,genes=genes,
        source_sha256=sources,scenarios={n:dict(**{k:v for k,v in all_cases[n].items() if k!='config'},simulation=asdict(all_cases[n]['config'])) for n in names},
        methods=METHODS,seed_rule='base + global scenario index*100000 + repetition; focused cases share index14',
        configs={m:asdict(c()) for m,c in [('v07',ShapeConfig),('v08',MeasurementConfig),('v09',JointConfig)]},
        selection_boundary='Frozen after separately retained pilots. Fresh disjoint seeds, not new generator families or external biology.',
        primary='Mean biological FDP, all-active recall; separate >=2-support claim FDP/recall. All-null FDR=P(any hit). Refusals are unsupported, never zero-error success.',
        inference_boundary='New model learns shared nuisance from explicit target/control mixture. Approximate posterior expected FDP is NOT BH or a frequentist guarantee.'))
    for name in names:
        case=all_cases[name];n=null_repeats if case['config'].active_fraction==0 else repeats
        current=[];index=14 if name.startswith('focused_shift') else list(all_cases).index(name)
        for rep in range(n):
            cfg=replace(case['config'],genes=genes,seed=seed+index*100000+rep)
            table,meta,truth,blocks=generate(cfg,case);prepared=None
            retained=rep==0 and name in REPLAY_CASES
            if retained:
                folder=new_output_dir(out/'full_replays'/name)
                save_counts(folder/'counts.tsv',table);write_tsv(folder/'samples.tsv',list(meta[0]),meta);write_tsv(folder/'truth.tsv',list(truth[0]),truth)
                write_json(folder/'generation.json',dict(config=asdict(cfg),case=name))
                if blocks:write_tsv(folder/'blocks.tsv',['sgRNA','block'],[dict(sgRNA=g,block=b) for g,b in blocks.items()])
            for method in METHODS:
                tick=time.perf_counter()
                try:
                    if prepared is None:
                        ctx={};a,b,c=call_empirical_hits(table,meta,'treatment','control',EmpiricalConfig(normalization=case['normalization']),_context_sink=ctx)
                        prepared=(a,b,c,ctx)
                    rows,info,guides=infer(table,meta,case,blocks,method,prepared)
                    score=score_shape(rows,truth,cfg.guides_per_gene);post=info.get('posterior_model') or {}
                    record=dict(scenario=name,method=method,repetition=rep,seed=cfg.seed,status='ok',**score,seconds=time.perf_counter()-tick,
                        branch=info.get('inference_branch',info['method']),shape_family=info.get('shape_model',{}).get('family'),
                        em_converged=post.get('em_converged'),posterior_prevalence=post.get('estimated_alternative_prevalence'),
                        model_probability=post.get('multi_model_probability'),
                        full_importance_ess=(post.get('full_integration') or {}).get('importance_ess'),
                        null_importance_ess=(post.get('null_integration') or {}).get('importance_ess'))
                    if retained:
                        target=new_output_dir(folder/method);write_tsv(target/'gene_results.tsv',list(dict.fromkeys(k for r in rows for k in r)),rows)
                        write_json(target/'analysis.json',info);write_json(target/'score.json',score)
                except ValueError as exc:
                    if not any(s in str(exc) for s in ('informative declared null controls','fully informative','sufficient blocks','No genes pass','Insufficient measurement df','Shared-null Laplace draws failed')):raise
                    record=dict(scenario=name,method=method,repetition=rep,seed=cfg.seed,status='unsupported',error=str(exc))
                    if retained:
                        target=new_output_dir(folder/method);write_json(target/'unsupported.json',dict(error=str(exc)))
                records.append(record);current.append(record)
            if (rep+1)%10==0:print(dict(scenario=name,completed=rep+1,total=n),flush=True)
        summaries.extend(summarize(current,case))
        write_tsv(out/'runs.tsv',list(dict.fromkeys(k for r in records for k in r)),records)
        write_json(out/'summary.json',dict(scenarios=summaries,complete=False))
    write_json(out/'summary.json',dict(scenarios=summaries,complete=True,decisions=len(records)))
