"""Frozen paired old/new calibration; additional asymmetric/copula stresses."""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .calibration import binomial_interval
from .correlated_comparison import cases as old_cases, generate as old_generate
from .correlated_inference import CorrelatedConfig, call_correlated_hits
from .empirical_inference import EmpiricalConfig, call_empirical_hits
from .io import CountTable, sha256, write_json, write_tsv
from .power_comparison import score_method
from .power_inference import PowerConfig, call_power_hits
from .shape_inference import ShapeConfig, call_shape_hits, score_shape
from .simulation import SimulationConfig, simulate


def cases():
    result = {name: dict(case, asymmetric=None) for name, case in old_cases().items()}
    for direction in (1, -1):
        for active in (0., .1):
            for rho in (0., .5):
                tag = "null" if active == 0 else "ineffective"
                name = f"sas_{'right' if direction == 1 else 'left'}_skew_{tag}_rho{rho:g}"
                result[name] = dict(config=SimulationConfig(controls=1000, active_fraction=active, ineffective_fraction=.5 if active else 0.),
                                    normalization="median", stress=None, drift=None, correlation=None, matched=True,
                                    control_correlation=rho, asymmetric=dict(direction=direction, rho=rho, sigma=.7))
    for tag, artifact, active in (("single_artifact_null", .05, 0.), ("single_artifact_mixture", .05, .1)):
        result[f"matched_{tag}_rho0.5"] = dict(config=SimulationConfig(controls=1000, active_fraction=active, artifact_fraction=artifact,
                                                                       ineffective_fraction=.5 if active else 0.),
            normalization="median", stress=None, drift="gaussian", correlation=.5, matched=True, control_correlation=.5, asymmetric=None)
    return result


def generate(config, case):
    if case["asymmetric"] is None:
        return old_generate(config, case)
    table, meta, truth, manifest = simulate(config)
    x = table.counts.copy()
    k, r, target = config.guides_per_gene, config.replicates, config.genes * config.guides_per_gene
    rng = np.random.default_rng(config.seed + 495373)
    mu = np.maximum(1., rng.choice(x[:target, :r].mean(axis=1), config.controls))
    x[target:] = rng.poisson(mu[:, None] * np.array(manifest["true_size_factors"])[None, :])
    rho, sigma, sign = (case["asymmetric"][key] for key in ("rho", "sigma", "direction"))
    block_count = config.genes + config.controls // k
    latent = np.sqrt(rho) * np.repeat(rng.normal(size=block_count), k) + np.sqrt(1-rho) * rng.normal(size=len(table.ids))
    delta = sign * .8 * (np.exp(sigma * latent - sigma**2/2) - 1) / np.sqrt(np.expm1(sigma**2))
    x[:, r:] = rng.poisson(x[:, r:] * np.exp2(np.clip(delta, -12, 12))[:, None])
    blocks = {gid: f"matched_null_{i//k:05d}" for i, gid in enumerate(table.ids[target:])}
    return CountTable(table.ids, table.genes, table.samples, x), meta, truth, blocks


def run_suite(out, repeats=50, null_repeats=100, genes=1000, seed=1027261000, scenario_names=None, progress=None):
    all_cases = cases()
    names = list(scenario_names or all_cases)
    if repeats < 2 or null_repeats < 2 or genes < 100 or any(n not in all_cases for n in names):
        raise ValueError("Need known cases, >=2 repetitions and >=100 genes")
    protocol = dict(seed=seed, repeats=repeats, null_repeats=null_repeats, genes=genes,
        source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        scenarios={name: dict(**{k:v for k,v in all_cases[name].items() if k != "config"}, simulation=asdict(all_cases[name]["config"])) for name in names},
        seed_rule="base + global case index*100000 + rep; focused cases share index14",
        methods="v0.4 background, v0.5 power-loo, v0.6 corr-envelope, v0.7 shape-envelope; corr-matched/shape-matched ONLY with genuine simulated null blocks",
        primary="Biological mean FDP and ALL-active recall at declared decision threshold .05; all-null FDR=P(anyhit)",
        inference_boundary="Shape-envelope uses guarded marginal partial-conjunction/BH. Shape-matched uses fitted sparse-mixture POSTERIOR expected FDP, NOT BH p/q values. Their frequentist FDR is independently measured against generated truth.",
        selection_boundary="Development pilots retained separately. Frozen new seeds and old generator families; not independent external biological validation. HQ461 not used for model selection.",
        refusal_policy="Invalid control/target information retained as unsupported, never counted as successful zero FDR",
        config=asdict(ShapeConfig()))
    write_json(out / "protocol.json", protocol)
    records, summaries = [], []
    started = time.perf_counter()
    for name in names:
        case = all_cases[name]
        index = 14 if name.startswith("focused_shift") else list(all_cases).index(name)
        n = null_repeats if case["config"].active_fraction == 0 else repeats
        methods = ["background", "power-loo", "corr-envelope", "shape-envelope"] + (["corr-matched", "shape-matched"] if case["matched"] else [])
        current = []
        for rep in range(n):
            cfg = replace(case["config"], genes=genes, seed=seed + index * 100000 + rep)
            table, meta, truth, blocks = generate(cfg, case)
            prepared = None
            for method in methods:
                try:
                    if method == "background":
                        context = {}
                        rows, info, guides = call_empirical_hits(table, meta, "treatment", "control", EmpiricalConfig(normalization=case["normalization"]), _context_sink=context)
                        prepared = (rows, info, guides, context)
                    elif method == "power-loo":
                        rows, info, _ = call_power_hits(table, meta, "treatment", "control", PowerConfig(normalization=case["normalization"]))
                    else:
                        matched = method.endswith("matched")
                        options = dict(normalization=case["normalization"], correlation_mode="matched-blocks" if matched else "envelope")
                        if method.startswith("shape"):
                            rows, info, _ = call_shape_hits(table, meta, "treatment", "control", ShapeConfig(**options),
                                control_blocks=blocks if matched else None, _prepared=prepared)
                        else:
                            rows, info, _ = call_correlated_hits(table, meta, "treatment", "control", CorrelatedConfig(**options), control_blocks=blocks if matched else None)
                    metric = score_shape(rows, truth, cfg.guides_per_gene) if method.startswith("shape") else score_method(rows, truth, method if method == "background" else "power-loo", cfg.guides_per_gene)
                    record = dict(scenario=name, method=method, repetition=rep+1, seed=cfg.seed, status="ok", **metric)
                    if method.startswith("shape"):
                        record.update(shape_family=info["shape_model"]["family"],
                                      posterior_prevalence=info["posterior_model"]["estimated_alternative_prevalence"] if info["posterior_model"] else None)
                except ValueError as exc:
                    if not any(text in str(exc) for text in ("informative declared null controls", "fully informative", "sufficient blocks", "No genes pass")):
                        raise
                    record = dict(scenario=name, method=method, repetition=rep+1, seed=cfg.seed, status="unsupported", error=str(exc))
                records.append(record); current.append(record)
            if progress and (rep+1) % 10 == 0:
                progress(dict(scenario=name, completed=rep+1, total=n, elapsed_seconds=round(time.perf_counter()-started, 1)))
        for method in methods:
            good = [r for r in current if r["method"] == method and r["status"] == "ok"]
            exclude = {"scenario", "method", "repetition", "seed", "status", "shape_family"}
            keys = list(dict.fromkeys(k for r in good for k in r if k not in exclude))
            summary = dict(scenario=name, method=method, repeats=n, valid_repetitions=len(good), unsupported_repetitions=n-len(good),
                **{key:float(np.mean([r[key] for r in good if r.get(key) is not None])) if any(r.get(key) is not None for r in good) else None for key in keys})
            summary["fdp_mc_se"] = float(np.std([r["fdp"] for r in good], ddof=1)/np.sqrt(len(good))) if len(good)>1 else None
            summary["recall_mc_se"] = float(np.std([r["recall"] for r in good], ddof=1)/np.sqrt(len(good))) if len(good)>1 and good[0]["recall"] is not None else None
            if case["config"].active_fraction == 0:
                summary["all_null_exact_95_interval"] = binomial_interval(sum(r["any_hit"] for r in good), len(good)) if good else None
            summaries.append(summary)
            if progress:
                progress(summary)
        write_tsv(out / "runs.tsv", list(dict.fromkeys(k for r in records for k in r)), records)
        write_json(out / "summary.json", dict(protocol=protocol, scenarios=summaries, complete=False))
    result = dict(protocol=protocol, scenarios=summaries, complete=True, attempted_decisions=len(records),
                  scenario_screens=len({(r["scenario"], r["seed"]) for r in records}), elapsed_seconds=time.perf_counter()-started,
                  production_validated=False)
    write_json(out / "summary.json", result)
    return result
