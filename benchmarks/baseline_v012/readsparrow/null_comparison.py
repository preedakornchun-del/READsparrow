"""Frozen matched calibration of all four research engines; retain failures."""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .calibration import binomial_interval
from .comparison import cases as legacy_cases, generate_case, score
from .empirical_inference import EmpiricalConfig, call_empirical_hits
from .guide_inference import GuideConfig, call_guide_hits
from .inference import TestConfig, call_hits
from .io import CountTable, sha256, write_json, write_tsv
from .simulation import SimulationConfig, simulate

METHODS = ("baseline", "guide-aware", "background", "empirical")


def cases():
    result = {k: dict(config=c, normalization=n, stress=s, drift=None) for k, (c, n, s) in legacy_cases().items()}
    for name, active, n, controls, drift in (
        ("drift_null_n3", 0., 3, 1000, "gaussian"),
        ("drift_mixture_n3", .1, 3, 1000, "gaussian"),
        ("drift_null_n2", 0., 2, 1000, "gaussian"),
        ("drift_mixture_n2", .1, 2, 1000, "gaussian"),
        ("drift_mixture_controls100", .1, 3, 100, "gaussian"),
        ("drift_strong_controls10000", .1, 3, 10000, "gaussian"),
        ("heavy_tail_null_LIMIT", 0., 3, 1000, "student3"),
        ("heavy_tail_mixture_LIMIT", .1, 3, 1000, "student3"),
        ("abundance_drift_null", 0., 3, 1000, "abundance"),
        ("skew_drift_null_LIMIT", 0., 3, 1000, "skew"),
        ("contaminated_controls_LIMIT", .1, 3, 1000, "contaminated"),
        ("nonexchangeable_null_LIMIT", 0., 3, 1000, "nonexchangeable"),
    ):
        result[name] = dict(config=SimulationConfig(active_fraction=active, replicates=n, controls=controls,
            effect=3. if "strong" in name else 1.5), normalization="median", stress=None, drift=drift)
    return result


def generate(config, stress=None, drift=None):
    if drift is None:
        return generate_case(config, stress)
    table, metadata, truth, manifest = simulate(config)
    rng = np.random.default_rng(config.seed + 29137)
    x, r, target = table.counts.copy(), config.replicates, config.genes * config.guides_per_gene
    # Unlike older simple Poisson NTCs of nearly equal abundance, these controls
    # span counts independently drawn from the target-guide baseline distribution.
    # Truth and target counts remain identical across all four analysis methods.
    control_mu = np.maximum(1., rng.choice(table.counts[:target, :r].mean(axis=1), config.controls))
    depths = np.asarray(manifest["true_size_factors"])
    x[target:] = rng.poisson(control_mu[:, None] * depths[None, :])
    delta = rng.normal(0, .8, len(table.ids))
    if drift == "student3":
        delta = .8 / np.sqrt(3) * rng.standard_t(3, len(table.ids))
    elif drift == "skew":
        sigma = .7
        delta = .8 * (rng.lognormal(-sigma ** 2 / 2, sigma, len(table.ids)) - 1) / np.sqrt(np.expm1(sigma ** 2))
    elif drift == "abundance":
        mu = x[:, :r].mean(axis=1)
        delta *= (.3 + 1.2 / (1 + mu / 50)) / .8
    elif drift == "contaminated":
        selected = target + rng.choice(config.controls, int(.1 * config.controls), replace=False)
        delta[selected] += 2.5
    elif drift == "nonexchangeable":
        delta[target:] *= .25  # negative controls do not represent target nuisance
    elif drift != "gaussian":
        raise ValueError("Unknown drift")
    # Avoid overflow for exceptionally rare t3 tails. This physical count limit
    # is specified before runs and recorded, not selected from analysis results.
    delta = np.clip(delta, -12, 12)
    x[:, r:] = rng.poisson(x[:, r:] * np.exp2(delta[:, None]))
    return CountTable(table.ids, table.genes, table.samples, x), metadata, truth


def analyze(table, metadata, treatment, control, method, normalization="median"):
    if method == "baseline":
        rows, info = call_hits(table, metadata, treatment, control, TestConfig(normalization=normalization))
        return rows, info, None
    if method == "guide-aware":
        return call_guide_hits(table, metadata, treatment, control, GuideConfig(normalization=normalization))
    return call_empirical_hits(table, metadata, treatment, control, EmpiricalConfig(normalization=normalization, null_model=method))


def comparison_suite(out, repeats=50, genes=1000, seed=604221000, scenario_names=None, progress=None, null_repeats=100):
    all_cases = cases()
    names = list(scenario_names or all_cases)
    if any(n not in all_cases for n in names) or repeats < 2 or genes < 100 or null_repeats < 2:
        raise ValueError("Known scenarios, >=2 repeats and >=100 genes required")
    protocol = {"development_status": "Post-HQ461-informed model development; fresh seeds are not independent external biological validation",
        "seed_base": seed, "genes": genes, "mixture_repeats": repeats, "all_null_repeats": null_repeats,
        "scenarios": {n: {**{k: v for k, v in all_cases[n].items() if k != "config"}, "simulation": asdict(all_cases[n]["config"])} for n in names},
        "methods": list(METHODS), "new_config": asdict(EmpiricalConfig()),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "seed_rule": "base + original case index*100000 + repetition; focused median/control share case index14",
        "drift_scale": "persistent per-guide technical/nuisance log2 SD .8; shared across treatment biological samples; clip at +/-12 before integer count generation",
        "drift_magnitude_origin": "Previous HQ461 exploratory diagnostic, not an independently discovered biological mechanism",
        "primary": "Mean biological FDP at q<=.05 plus recall; all-null FDR=P(any hit), exact binomial interval",
        "boundary": "Model-based prediction intervals and empirical scores need control/target null exchangeability. A small hit count or uniform calibration-control ranks is not accuracy evidence.",
        "estimands": "Old gene-sum vs new median-guide effects; simulated nuisance mean zero, biological pre-nuisance truth retained",
        "decision": "Fixed all four methods; do not choose favorable normalization/model or hide LIMIT scenarios"}
    write_json(out / "comparison_protocol.json", protocol)
    records, summaries = [], []
    start = time.perf_counter()
    for name in names:
        case = all_cases[name]
        count = null_repeats if case["config"].active_fraction == 0 else repeats
        index = list(all_cases).index(name)
        index = 14 if name.startswith("focused_shift") else index
        current = []
        for rep in range(count):
            conf = replace(case["config"], genes=genes, seed=seed + index * 100000 + rep)
            table, meta, truth = generate(conf, case["stress"], case["drift"])
            for method in METHODS:
                try:
                    rows, info, _ = analyze(table, meta, "treatment", "control", method, case["normalization"])
                except ValueError as exc:
                    if method not in {"background", "empirical"} or "informative declared null controls" not in str(exc):
                        raise
                    metric = dict(scenario=name, method=method, repetition=rep + 1, seed=conf.seed,
                                  status="unsupported_insufficient_controls", error=str(exc))
                    records.append(metric)
                    current.append(metric)
                    continue
                # Background-relative median guide effects have the median-guide
                # biological estimand when the generating null has zero mean.
                metric = score(rows, truth, "baseline" if method == "baseline" else "guide-aware")
                metric.update(scenario=name, method=method, repetition=rep + 1, seed=conf.seed, status="ok")
                if method in {"background", "empirical"}:
                    bg = info["background_model"]
                    metric.update(held_out_control_p_lt_005=bg["held_out_background_p_lt_005"],
                        background_tail_alarm=int(any("tail_excess" in v for v in bg["advisory_alarms"])),
                        one_sided_rank_p_floor=bg["one_sided_rank_p_floor"])
                records.append(metric)
                current.append(metric)
            if progress and (rep + 1) % 10 == 0:
                progress({"scenario": name, "completed": rep + 1, "total": count, "elapsed_seconds": round(time.perf_counter() - start, 1)})
        for method in METHODS:
            subset = [r for r in current if r["method"] == method and r["status"] == "ok"]
            keys = list(dict.fromkeys(k for r in subset for k in r if k not in {"scenario", "method", "repetition", "seed", "status", "error"}))
            row = dict(scenario=name, method=method, repeats=count, valid_repetitions=len(subset), unsupported_repetitions=count-len(subset),
                **{k: float(np.mean([r[k] for r in subset if r.get(k) is not None])) if any(r.get(k) is not None for r in subset) else None for k in keys})
            row["fdp_mc_se"] = float(np.std([r["fdp"] for r in subset], ddof=1) / np.sqrt(len(subset))) if len(subset) > 1 else None
            if case["config"].active_fraction == 0:
                row["all_null_exact_95_interval"] = binomial_interval(sum(r["any_hit"] for r in subset), len(subset)) if subset else None
            summaries.append(row)
            if progress:
                progress(row)
        write_tsv(out / "comparison_runs.tsv", list(dict.fromkeys(k for r in records for k in r)), records)
        write_json(out / "comparison_summary.json", dict(protocol=protocol, scenarios=summaries, complete=False))
    result = dict(protocol=protocol, scenarios=summaries, complete=True, production_validated=False,
                  screens=sum(null_repeats if all_cases[n]["config"].active_fraction == 0 else repeats for n in names),
                  method_evaluations=len(records), elapsed_seconds=time.perf_counter() - start)
    write_json(out / "comparison_summary.json", result)
    return result
