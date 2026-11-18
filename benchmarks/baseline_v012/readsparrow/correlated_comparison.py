"""Matched calibration: v0.4 background, v0.5 LOO, new correlation/sparse.

Old cases retain their exact generators. New matched-null groups have genuinely
shared simulated nuisance, never random post-hoc partitions of unrelated NTCs.
"""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .calibration import binomial_interval
from .correlated_inference import CorrelatedConfig, call_correlated_hits
from .empirical_inference import EmpiricalConfig, call_empirical_hits
from .io import CountTable, sha256, write_json, write_tsv
from .power_comparison import cases as previous_cases, generate_screen, score_method
from .power_inference import PowerConfig, call_power_hits
from .simulation import SimulationConfig, simulate


def cases():
    result={n:dict(v,matched=False) for n,v in previous_cases().items()}
    for active,ineffective,tag,rhos in ((0.,0.,"null",(0.,.25,.5,1.)),(.1,0.,"mixture",(0.,.25,.5)),(.1,.5,"ineffective",(0.,.25,.5))):
        for rho in rhos:
            result[f"matched_{tag}_rho{rho:g}"]=dict(config=SimulationConfig(controls=1000,active_fraction=active,ineffective_fraction=ineffective),
                normalization="median",stress=None,drift="gaussian",correlation=rho,matched=True,control_correlation=rho)
    for rho in (0.,.25,.5):
        result[f"matched_effect2_rho{rho:g}"]=dict(config=SimulationConfig(controls=1000,effect=2.),normalization="median",
            stress=None,drift="gaussian",correlation=rho,matched=True,control_correlation=rho)
    result["matched_wrong_controls_rho0.5_LIMIT"]=dict(config=SimulationConfig(controls=1000,active_fraction=0),
        normalization="median",stress=None,drift="gaussian",correlation=.5,matched=True,control_correlation=0.)
    for k,ineffective in ((6,.5),(8,.5),(8,.75)):
        result[f"ineffective_nb_guides{k}_fraction{ineffective:g}"]=dict(config=SimulationConfig(controls=1000,guides_per_gene=k,ineffective_fraction=ineffective),
            normalization="median",stress=None,drift=None,correlation=None,matched=False)
    result["matched_ineffective_rho0.5_effect2"]=dict(config=SimulationConfig(controls=1000,effect=2.,ineffective_fraction=.5),
        normalization="median",stress=None,drift="gaussian",correlation=.5,matched=True,control_correlation=.5)
    return result


def generate(config,case):
    table,meta,truth=generate_screen(config,case)
    if not case["matched"]:
        return table,meta,truth,None
    k,r=config.guides_per_gene,config.replicates
    target=config.genes*k
    if config.controls%k:
        raise ValueError("Matched simulation requires complete null blocks")
    rng=np.random.default_rng(config.seed+198231)
    x=table.counts.copy()
    mu=np.maximum(1.,rng.choice(x[:target,:r].mean(axis=1),config.controls))
    # Reuse existing library depth structure; the new controls are deliberately
    # Poisson measurements so nuisance exchangeability, not assay complexity,
    # is isolated. Target measurement covariance remains gene/sample-shared.
    # Reproduce the seeded generator's TRUE library size factors, not estimated
    # treatment effects or an assumption of equal depths across conditions.
    _,_,_,manifest=simulate(config)
    depths=np.asarray(manifest["true_size_factors"])
    baseline=rng.poisson(mu[:,None]*depths[None,:])
    rho=case["control_correlation"]
    common=rng.normal(0,.8,config.controls//k)
    delta=np.sqrt(rho)*np.repeat(common,k)+np.sqrt(1-rho)*rng.normal(0,.8,config.controls)
    baseline[:,r:]=rng.poisson(baseline[:,r:]*np.exp2(np.clip(delta,-12,12)[:,None]))
    x[target:]=baseline
    blocks={gid:f"matched_null_{i//k:05d}" for i,gid in enumerate(table.ids[target:])}
    return CountTable(table.ids,table.genes,table.samples,x),meta,truth,blocks


def run_suite(out,repeats=50,null_repeats=100,genes=1000,seed=826261000,scenario_names=None,progress=None,coherent_weight=.8):
    all_cases=cases();names=list(scenario_names or all_cases)
    if repeats<2 or null_repeats<2 or genes<100 or any(n not in all_cases for n in names):
        raise ValueError("Known scenarios, >=2 repetitions and >=100 genes required")
    protocol=dict(seed=seed,repeats=repeats,null_repeats=null_repeats,genes=genes,coherent_weight=coherent_weight,
        source_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        scenarios={n:dict(**{k:v for k,v in all_cases[n].items() if k!="config"},simulation=asdict(all_cases[n]["config"])) for n in names},
        seed_rule="base + global case index*100000 + rep; original focused median/control share index14",
        methods="background and power-loo (rho0 legacy), corr-envelope (rho<=1); corr-matched additionally ONLY when genuine control block labels were simulated",
        boundary="Previously seen generator families/HQ461; fresh seeds not external biology. Matched blocks are extra experimental information; compare both methods on the SAME matrix. Unknown rho1 common-mode limits moderate signal identifiability.",
        primary="Biological mean FDP and all-active recall at q=.05; support-claim metrics also retained; null=P(anyhit), exact binomial interval",
        refusal_policy="Insufficient controls/blocks/full informative genes retained as unsupported, never zero-FDR success",
        control_block_model="Poisson null measurement, shared Gaussian block nuisance rho; labels fixed before analysis. Wrong-controls case deliberately violates representativeness.")
    write_json(out/"protocol.json",protocol)
    records,summaries=[],[];started=time.perf_counter()
    for name in names:
        case=all_cases[name];index=14 if name.startswith("focused_shift") else list(all_cases).index(name)
        n=null_repeats if case["config"].active_fraction==0 else repeats
        methods=["background","power-loo","corr-envelope"]+(["corr-matched"] if case["matched"] else [])
        current=[]
        for rep in range(n):
            cfg=replace(case["config"],genes=genes,seed=seed+index*100000+rep)
            table,meta,truth,blocks=generate(cfg,case)
            for method in methods:
                try:
                    if method=="background":
                        rows,info,_=call_empirical_hits(table,meta,"treatment","control",EmpiricalConfig(normalization=case["normalization"]))
                    elif method=="power-loo":
                        rows,info,_=call_power_hits(table,meta,"treatment","control",PowerConfig(normalization=case["normalization"]))
                    else:
                        config=CorrelatedConfig(normalization=case["normalization"],coherent_weight=coherent_weight,
                            correlation_mode="matched-blocks" if method=="corr-matched" else "envelope")
                        rows,info,_=call_correlated_hits(table,meta,"treatment","control",config,control_blocks=blocks if method=="corr-matched" else None)
                    metric=score_method(rows,truth,"power-loo" if method!="background" else method,cfg.guides_per_gene)
                    record=dict(scenario=name,method=method,repetition=rep+1,seed=cfg.seed,status="ok",**metric)
                    if method.startswith("corr-"):
                        record.update(rho_upper=info["correlation_model"]["upper"],control_blocks=info["correlation_model"]["blocks"])
                except ValueError as exc:
                    if not any(v in str(exc) for v in ("informative declared null controls","fully informative genes","sufficient blocks","known informative declared null controls")):
                        raise
                    record=dict(scenario=name,method=method,repetition=rep+1,seed=cfg.seed,status="unsupported",error=str(exc))
                records.append(record);current.append(record)
            if progress and (rep+1)%10==0:
                progress(dict(scenario=name,completed=rep+1,total=n,seconds=round(time.perf_counter()-started,1)))
        for method in methods:
            good=[r for r in current if r["method"]==method and r["status"]=="ok"]
            keys=list(dict.fromkeys(k for r in good for k in r if k not in {"scenario","method","repetition","seed","status"}))
            summary=dict(scenario=name,method=method,repeats=n,valid_repetitions=len(good),unsupported_repetitions=n-len(good),
                **{k:float(np.mean([r[k] for r in good if r.get(k) is not None])) if any(r.get(k) is not None for r in good) else None for k in keys})
            summary["fdp_mc_se"]=float(np.std([r["fdp"] for r in good],ddof=1)/np.sqrt(len(good))) if len(good)>1 else None
            if case["config"].active_fraction==0:
                summary["all_null_exact_95_interval"]=binomial_interval(sum(r["any_hit"] for r in good),len(good)) if good else None
            summaries.append(summary)
            if progress:progress(summary)
        write_tsv(out/"runs.tsv",list(dict.fromkeys(k for r in records for k in r)),records)
        write_json(out/"summary.json",dict(protocol=protocol,scenarios=summaries,complete=False))
    result=dict(protocol=protocol,scenarios=summaries,complete=True,attempted_decisions=len(records),
        scenario_screens=len({(r["scenario"],r["seed"]) for r in records}),elapsed_seconds=time.perf_counter()-started,production_validated=False)
    write_json(out/"summary.json",result)
    return result
