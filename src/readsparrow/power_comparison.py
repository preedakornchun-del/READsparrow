"""Matched power/calibration development and frozen tests, no HQ tuning."""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .calibration import binomial_interval
from .comparison import score
from .empirical_inference import EmpiricalConfig, call_empirical_hits
from .guide_inference import call_guide_hits, GuideConfig
from .io import CountTable, sha256, write_json, write_tsv
from .null_comparison import cases as previous_cases, generate
from .power_inference import PowerConfig, call_power_hits, loo_rows_from_power
from .simulation import SimulationConfig, simulate

METHODS = ("guide-aware", "background", "power-bounded", "power-loo")


def cases():
    result = {n: dict(v, correlation=None) for n, v in previous_cases().items()}
    for effect in (.75, 1., 2.):
        result[f"moderate_drift_effect{effect:g}"] = dict(config=SimulationConfig(controls=1000, effect=effect),
            normalization="median", stress=None, drift="gaussian", correlation=None)
    for k in (6, 8):
        result[f"moderate_drift_guides{k}"] = dict(config=SimulationConfig(controls=1000, guides_per_gene=k),
            normalization="median", stress=None, drift="gaussian", correlation=None)
    for rho in (.25, .5, 1.):
        result[f"correlated_drift_null_rho{rho:g}_LIMIT"] = dict(config=SimulationConfig(controls=1000, active_fraction=0),
            normalization="median", stress=None, drift="gaussian", correlation=rho)
    return result


def generate_screen(config, case):
    if case["correlation"] is None:
        return generate(config, case["stress"], case["drift"])
    table, metadata, truth, manifest = simulate(config)
    rng = np.random.default_rng(config.seed + 29137)
    x, r, target = table.counts.copy(), config.replicates, config.genes * config.guides_per_gene
    control_mu = np.maximum(1., rng.choice(table.counts[:target, :r].mean(axis=1), config.controls))
    x[target:] = rng.poisson(control_mu[:, None] * np.asarray(manifest["true_size_factors"])[None, :])
    delta = rng.normal(0, .8, len(table.ids))
    common = np.random.default_rng(config.seed + 7189).normal(0, .8, config.genes)
    rho = case["correlation"]
    delta[:target] = np.sqrt(1-rho)*delta[:target] + np.sqrt(rho)*np.repeat(common, config.guides_per_gene)
    x[:, r:] = rng.poisson(x[:, r:] * np.exp2(np.clip(delta, -12, 12)[:, None]))
    return CountTable(table.ids, table.genes, table.samples, x), metadata, truth


def score_method(rows, truth, method, guides_per_gene):
    if method in {"power-bounded", "power-loo"}:
        # Equal guide mean is NOT the old count-weighted or median estimand.
        reference = [dict(r, true_median_guide_effect=r["nominal_effect"]*r["effective_guides"]/guides_per_gene) for r in truth]
    else:
        reference = truth
    metric = score(rows, reference, "guide-aware")
    if method == "power-bounded":
        metric["two_effective_guide_context_fdp"] = metric["support_claim_fdp"]
        metric["two_effective_guide_context_recall"] = metric["support_claim_recall"]
        # A sign gate is NOT a valid partial-conjunction/support claim.
        metric["support_claim_fdp"] = metric["support_claim_recall"] = None
    return metric


def comparison_suite(out, repeats=50, null_repeats=100, genes=1000, seed=714261000, scenario_names=None, progress=None):
    all_cases = cases()
    names = list(scenario_names or all_cases)
    if repeats < 2 or null_repeats < 2 or genes < 100 or any(n not in all_cases for n in names):
        raise ValueError("Known scenarios, >=2 repeats, >=100 genes required")
    protocol = dict(seed=seed, repeats=repeats, null_repeats=null_repeats, genes=genes,
        methods=METHODS, config=asdict(PowerConfig()),
        scenarios={n: dict(**{k: v for k, v in all_cases[n].items() if k != "config"}, simulation=asdict(all_cases[n]["config"])) for n in names},
        primary="Delete-one IUT vs v0.4 background and v0.3 guide-aware; matched biological FDP and recall at q=.05",
        secondary="Bounded-location candidate is retained but NOT promoted: development null calibration failed under skew/heavy tails. Selection made before fresh frozen seeds and before HQ461.",
        seed_rule="base+global scenario index*100000+rep; focused median/control share index14",
        source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        estimands="Previous median guide vs new equal-weight mean guide effects; biological truth and all-active recall denominator unchanged",
        boundary="Fresh simulation seeds, previously used generator families; HQ461 reused development data. No model or threshold selected from HQ hit count. Independence of persistent nuisance is explicitly stress-tested.",
        refusal_policy="Insufficient informative controls/full-gene measurement moderation: unsupported, never zero-FDR success",
        resampling="50k training-control and normal pseudo-groups per guide number fit null bias/variance inflation only; not biological replication or distribution-free empirical p-tail precision")
    write_json(out / "protocol.json", protocol)
    records, summaries = [], []
    started = time.perf_counter()
    for name in names:
        case = all_cases[name]
        index = 14 if name.startswith("focused_shift") else list(all_cases).index(name)
        n = null_repeats if not case["config"].active_fraction else repeats
        current = []
        for rep in range(n):
            config = replace(case["config"], genes=genes, seed=seed + index*100000 + rep)
            table, meta, truth = generate_screen(config, case)
            power_rows = None
            power_error = None
            for method in METHODS:
                try:
                    if method == "guide-aware":
                        rows, info, _ = call_guide_hits(table, meta, "treatment", "control", GuideConfig(normalization=case["normalization"]))
                    elif method == "background":
                        rows, info, _ = call_empirical_hits(table, meta, "treatment", "control", EmpiricalConfig(normalization=case["normalization"]))
                    elif method == "power-bounded":
                        rows, info, _ = call_power_hits(table, meta, "treatment", "control", PowerConfig(normalization=case["normalization"], aggregation="bounded"))
                        power_rows = rows
                    else:
                        if power_error:
                            raise ValueError(power_error)
                        rows = loo_rows_from_power(power_rows, PowerConfig(normalization=case["normalization"], aggregation="loo"))
                    metric = dict(scenario=name, method=method, repetition=rep+1, seed=config.seed, status="ok",
                        **score_method(rows, truth, method, config.guides_per_gene))
                except ValueError as exc:
                    if not any(s in str(exc) for s in ("informative declared null controls", "fully informative genes")):
                        raise
                    if method == "power-bounded":
                        power_error = str(exc)
                    metric = dict(scenario=name, method=method, repetition=rep+1, seed=config.seed, status="unsupported", error=str(exc))
                current.append(metric)
                records.append(metric)
            if progress and (rep+1) % 10 == 0:
                progress(dict(scenario=name, completed=rep+1, total=n, seconds=round(time.perf_counter()-started, 1)))
        for method in METHODS:
            subset = [r for r in current if r["method"] == method and r["status"] == "ok"]
            metrics = list(dict.fromkeys(k for r in subset for k in r if k not in {"scenario", "method", "repetition", "seed", "status"}))
            summary = dict(scenario=name, method=method, repeats=n, valid_repetitions=len(subset), unsupported_repetitions=n-len(subset),
                **{k: float(np.mean([r[k] for r in subset if r.get(k) is not None])) if any(r.get(k) is not None for r in subset) else None for k in metrics})
            if subset:
                summary["fdp_mc_se"] = float(np.std([r["fdp"] for r in subset], ddof=1)/np.sqrt(len(subset))) if len(subset)>1 else None
                if not case["config"].active_fraction:
                    summary["all_null_exact_95_interval"] = binomial_interval(sum(r["any_hit"] for r in subset), len(subset))
            summaries.append(summary)
            if progress:
                progress(summary)
        write_tsv(out / "runs.tsv", list(dict.fromkeys(k for r in records for k in r)), records)
        write_json(out / "summary.json", dict(protocol=protocol, scenarios=summaries, complete=False))
    result = dict(protocol=protocol, scenarios=summaries, complete=True, attempted_evaluations=len(records),
                  scenario_screens=len({(r["scenario"],r["seed"]) for r in records}), elapsed_seconds=time.perf_counter()-started,
                  production_validated=False)
    write_json(out / "summary.json", result)
    return result
