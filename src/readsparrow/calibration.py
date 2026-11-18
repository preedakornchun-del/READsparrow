"""Monte Carlo diagnostics. Failure remains visible; never tune on held-out seeds."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
from scipy.stats import beta, kstest

from .inference import TestConfig, call_hits
from .io import sha256, write_json, write_tsv
from pathlib import Path
from .simulation import SimulationConfig, simulate


def evaluate(results, truth, fdr=0.05):
    expected = {r["gene"]: r for r in truth}
    tested = [r for r in results if r["tested"]]
    hits = [r for r in tested if r["qvalue"] <= fdr]
    false = sum(not expected[r["gene"]]["active"] for r in hits)
    active_all = sum(r["active"] for r in truth)
    null = [r for r in tested if not expected[r["gene"]]["active"]]
    true_hits = [r for r in hits if expected[r["gene"]]["active"]]
    errors = [r["log2_effect"] - expected[r["gene"]]["true_log2_effect"] for r in tested]
    return {"tested": len(tested), "hits": len(hits), "false_hits": false, "fdp": false / max(1, len(hits)),
            "recall": len(true_hits) / active_all if active_all else None,
            "any_hit": int(bool(hits)), "active_genes": active_all,
            "null_p_lt_005": float(np.mean([r["pvalue"] < .05 for r in null])) if null else None,
            "null_p_ks_statistic": float(kstest([r["pvalue"] for r in null], "uniform").statistic) if null else None,
            "ci_coverage": float(np.mean([r["ci_low"] <= expected[r["gene"]]["true_log2_effect"] <= r["ci_high"] for r in tested])),
            "effect_bias": float(np.mean(errors)), "effect_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "wrong_direction_true_hits": sum(np.sign(r["log2_effect"]) != np.sign(expected[r["gene"]]["true_log2_effect"]) for r in true_hits)}


def binomial_interval(successes, total, confidence=.95):
    a = (1 - confidence) / 2
    return [0.0 if successes == 0 else float(beta.ppf(a, successes, total - successes + 1)),
            1.0 if successes == total else float(beta.ppf(1 - a, successes + 1, total - successes))]


def calibration_suite(out, repeats=100, genes=1000, seed=202610220, progress=None):
    if repeats < 2 or genes < 100 or seed < 0:
        raise ValueError("Use >=2 Monte Carlo repetitions, >=100 genes and non-negative seed")
    test_config = TestConfig()
    scenarios = {
        "nb_null_n3": SimulationConfig(active_fraction=0, replicates=3),
        "nb_mixture_n3": SimulationConfig(replicates=3),
        "nb_mixture_n2": SimulationConfig(replicates=2),
        "lognormal_mixture_n3": SimulationConfig(model="lognormal"),
        "paired_nb_n3": SimulationConfig(paired=True),
        "low_counts_stress": SimulationConfig(mean_count=8, dispersion=.2),
        "guide_artifact_stress": SimulationConfig(artifact_fraction=.05),
        "unequal_depth_stress": SimulationConfig(depth_sd=.8),
        "focused_shift_median_stress": SimulationConfig(active_fraction=.7, positive_fraction=1),
        "focused_shift_control_stress": SimulationConfig(active_fraction=.7, positive_fraction=1),
    }
    # Frozen diagnostic margins: engineering smoke gates, NOT proof of FDR control.
    protocol = {"scenarios": {k: asdict(replace(v, genes=genes)) for k, v in scenarios.items()},
        "test_config": asdict(test_config), "repeats": repeats, "seed": seed,
        "normalization_override": {"focused_shift_control_stress": "control"},
        "seed_rule": "seed + scenario_index*100000 + repetition; both focused-shift methods reuse scenario_index=8 for a paired comparison",
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        "diagnostic_margins": {"mean_fdp_upper": .08, "null_p_lt_005_upper": .075, "ci_coverage_lower": .90},
        "decision": "Nominal q=.05; margins only flag major miscalibration, not establish statistical control at .05",
        "stress_scenarios": [k for k in scenarios if "stress" in k],
        "scope": "Count-level simulation only; no molecular truth in public data; no hold-out external validation here"}
    # Write before generating any screen or computing an outcome.
    write_json(out / "calibration_protocol.json", protocol)
    rows, summaries = [], []
    for scenario_index, (scenario, base) in enumerate(scenarios.items()):
        records = []
        for rep in range(repeats):
            paired_index = 8 if scenario.startswith("focused_shift_") else scenario_index
            config = replace(base, seed=seed + paired_index * 100000 + rep, genes=genes)
            table, metadata, truth, _ = simulate(config)
            setting = replace(test_config, normalization="control") if scenario == "focused_shift_control_stress" else test_config
            results, info = call_hits(table, metadata, "treatment", "control", setting)
            metric = {"scenario": scenario, "repetition": rep + 1, "seed": config.seed,
                      **evaluate(results, truth, test_config.fdr), "prior_df": info["moderation"]["prior_df"]}
            rows.append(metric)
            records.append(metric)
        get_mean = lambda name: float(np.mean([r[name] for r in records if r[name] is not None]))
        fdp = np.array([r["fdp"] for r in records])
        summary = {"scenario": scenario, "repetitions": repeats, "mean_fdp_empirical_fdr": get_mean("fdp"),
            "fdp_monte_carlo_se": float(fdp.std(ddof=1) / np.sqrt(repeats)),
            "recall": get_mean("recall") if base.active_fraction else None,
            "mean_null_p_lt_005": get_mean("null_p_lt_005"), "mean_ci_coverage": get_mean("ci_coverage"),
            "mean_effect_bias": get_mean("effect_bias"), "mean_effect_rmse": get_mean("effect_rmse"),
            "mean_hits": get_mean("hits"), "stress_test": "stress" in scenario}
        if not base.active_fraction:
            successes = sum(r["any_hit"] for r in records)
            summary["all_null_probability_any_discovery"] = successes / repeats
            summary["all_null_fdr_exact_95_interval"] = binomial_interval(successes, repeats)
        flags = []
        if summary["mean_fdp_empirical_fdr"] > .08:
            flags.append("empirical_fdr_exceeds_diagnostic_margin")
        if summary["mean_null_p_lt_005"] > .075:
            flags.append("liberal_null_pvalues")
        if summary["mean_ci_coverage"] < .90:
            flags.append("interval_undercoverage")
        summary["diagnostic_flags"] = flags
        summaries.append(summary)
        if progress:
            progress(summary)
    write_tsv(out / "calibration_runs.tsv", list(rows[0]), rows)
    report = {"protocol": protocol, "scenarios": summaries,
        "baseline_diagnostic_pass": all(not r["diagnostic_flags"] for r in summaries if not r["stress_test"]),
        "production_validated": False,
        "interpretation": "Diagnostic pass does not prove FDR<=.05; use Monte Carlo error and held-out studies. Stress failures are expected limitations, never hidden."}
    write_json(out / "calibration_summary.json", report)
    return report
