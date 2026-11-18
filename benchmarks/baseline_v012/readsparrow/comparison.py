"""Paired old/new calibration. No outcome-based threshold or model selection."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .calibration import binomial_interval
from .guide_inference import GuideConfig, call_guide_hits
from .inference import TestConfig, call_hits
from .io import CountTable, sha256, write_json, write_tsv
from .simulation import SimulationConfig, simulate


def score(rows, truth, method, q=.05):
    lookup = {r["gene"]: r for r in truth}
    tested = [r for r in rows if r["tested"]]
    hits = [r for r in tested if r["qvalue"] <= q]
    biologically_active = sum(r["active"] for r in truth)
    false_hits = sum(not lookup[r["gene"]]["active"] for r in hits)
    true_hits = sum(lookup[r["gene"]]["active"] for r in hits)
    true_parameter = "true_median_guide_effect" if method == "guide-aware" else "true_log2_effect"
    errors, coverage, widths = [], [], []
    for r in tested:
        actual = lookup[r["gene"]][true_parameter]
        errors.append(r["log2_effect"] - actual)
        low = -np.inf if r["ci_low"] is None else r["ci_low"]
        high = np.inf if r["ci_high"] is None else r["ci_high"]
        coverage.append(low <= actual <= high)
        if np.isfinite(low) and np.isfinite(high):
            widths.append(high - low)
    null_rows = [r for r in tested if not lookup[r["gene"]]["active"]]
    metric = {"tested": len(tested), "hits": len(hits), "false_hits": false_hits,
              "fdp": false_hits / max(1, len(hits)), "any_hit": int(bool(hits)),
              "recall": true_hits / biologically_active if biologically_active else None,
              "active_genes": biologically_active,
              "null_p_lt_005": float(np.mean([r["pvalue"] < .05 for r in null_rows])) if null_rows else None,
              "interval_coverage_for_method_estimand": float(np.mean(coverage)),
              "informative_interval_fraction": len(widths) / len(tested),
              "mean_finite_interval_width": float(np.mean(widths)) if widths else None,
              "effect_bias_for_method_estimand": float(np.mean(errors)),
              "wrong_direction_true_hits": sum(r["direction"] != ("enrichment" if lookup[r["gene"]]["nominal_effect"] > 0 else "depletion")
                                               for r in hits if lookup[r["gene"]]["active"])}
    if method == "guide-aware":
        # Distinguish 'any effective guide' biological truth from the stronger
        # configured multiple-effective-guide claim (one effective guide cannot
        # satisfy the latter even when its targeted gene is truly biological).
        support_truth = lambda r: lookup[r["gene"]]["effective_guides"] >= r["required_support"]
        claim_false = sum(not support_truth(r) for r in hits)
        support_active = sum(support_truth(r) for r in rows)
        metric["support_claim_fdp"] = claim_false / max(1, len(hits))
        metric["support_claim_recall"] = sum(support_truth(r) for r in hits) / support_active if support_active else None
    else:
        metric["support_claim_fdp"] = None
        metric["support_claim_recall"] = None
    return metric


def generate_case(config, stress=None):
    table, metadata, truth, manifest = simulate(config)
    if stress is None:
        return table, metadata, truth
    rng = np.random.default_rng(config.seed + 7919)
    x = table.counts.copy()
    null = np.array([i for i, r in enumerate(truth) if not r["active"]])
    selected = rng.choice(null, max(1, int(len(null) * .05)), replace=False)
    k, n = config.guides_per_gene, config.replicates
    if stress == "two_guide_artifacts":
        for g in selected:
            x[g * k:g * k + 2, n:] *= 8
    elif stress == "single_replicate_artifact":
        for g in selected:
            x[g * k:(g + 1) * k, n] *= 8
    elif stress == "guide_noise":
        # Independent guide/sample heterogeneity beyond shared gene-level noise.
        target = config.genes * k
        x[:target] = rng.poisson(x[:target] * rng.lognormal(-.5 * .4**2, .4, x[:target].shape))
    else:
        raise ValueError("Unknown stress case")
    for i in selected:
        truth[i]["technical_artifact"] = True
    return CountTable(table.ids, table.genes, table.samples, x), metadata, truth


def cases():
    # Fixed before HQ461 candidate results. Sensitivity cases are not model selection.
    return {
        "nb_null_n3": (SimulationConfig(active_fraction=0), "median", None),
        "nb_null_n2": (SimulationConfig(active_fraction=0, replicates=2), "median", None),
        "nb_mixture_n3": (SimulationConfig(), "median", None),
        "nb_mixture_n2": (SimulationConfig(replicates=2), "median", None),
        "lognormal_mixture_n3": (SimulationConfig(model="lognormal"), "median", None),
        "paired_nb_n3": (SimulationConfig(paired=True), "median", None),
        "single_guide_artifact": (SimulationConfig(artifact_fraction=.05), "median", None),
        "single_guide_artifact_n2": (SimulationConfig(artifact_fraction=.05, replicates=2), "median", None),
        "ineffective_guides": (SimulationConfig(ineffective_fraction=.5), "median", None),
        "low_counts": (SimulationConfig(mean_count=8, dispersion=.2), "median", None),
        "unequal_depth": (SimulationConfig(depth_sd=.8), "median", None),
        "two_guide_artifacts_LIMIT": (SimulationConfig(), "median", "two_guide_artifacts"),
        "single_replicate_artifact_LIMIT": (SimulationConfig(), "median", "single_replicate_artifact"),
        "guide_specific_noise": (SimulationConfig(), "median", "guide_noise"),
        "focused_shift_median_LIMIT": (SimulationConfig(active_fraction=.7, positive_fraction=1), "median", None),
        "focused_shift_control": (SimulationConfig(active_fraction=.7, positive_fraction=1), "control", None),
    }


def comparison_suite(out, repeats=100, genes=1000, seed=202610240, scenario_names=None, progress=None):
    if repeats < 2 or genes < 100:
        raise ValueError("Use >=2 replicates of screens and >=100 genes")
    all_cases = cases()
    names = list(all_cases) if not scenario_names else list(scenario_names)
    if any(n not in all_cases for n in names):
        raise ValueError("Unknown scenario")
    protocol = {"seed": seed, "repeats": repeats, "genes": genes,
                "cases": {n: {"simulation": asdict(all_cases[n][0]), "normalization": all_cases[n][1], "stress": all_cases[n][2]} for n in names},
                "baseline_config": asdict(TestConfig()), "guide_config": asdict(GuideConfig()),
                "primary_endpoints": ["biological empirical FDR", "recall including filtered active genes", "support-claim empirical FDR"],
                "secondary": ["interval coverage for each method's DIFFERENT estimand", "informative interval fraction", "HQ461 counts/hits, not accuracy"],
                "seed_rule": "seed + original scenario index*100000 + repetition; matched focused normalization uses index14 for both",
                "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                "decision": "No tuning to match MAGeCK hit number or remove unfavorable calibration results",
                "FDR": "Mean FDP across screens. All-null FDR=P(any discovery); do not substitute p<.05 frequency.",
                "limitations": "Constituent log-count tests approximate; BH needs across-gene dependence assumptions; BY available as sensitivity, not selected by results",
                "theory_reference": "Benjamini & Heller 2008 doi:10.1111/j.1541-0420.2007.00984.x"}
    write_json(out / "comparison_protocol.json", protocol)
    records, summaries = [], []
    for name in names:
        base, normalization, stress = all_cases[name]
        index = list(all_cases).index(name)
        index = 14 if name.startswith("focused_shift") else index
        scenario_rows = []
        for rep in range(repeats):
            config = replace(base, genes=genes, seed=seed + index * 100000 + rep)
            table, metadata, truth = generate_case(config, stress)
            for method in ("baseline", "guide-aware"):
                if method == "baseline":
                    result, info = call_hits(table, metadata, "treatment", "control", TestConfig(normalization=normalization))
                else:
                    result, info, _ = call_guide_hits(table, metadata, "treatment", "control", GuideConfig(normalization=normalization))
                metric = {"scenario": name, "method": method, "repetition": rep + 1, "seed": config.seed,
                          **score(result, truth, method)}
                records.append(metric)
                scenario_rows.append(metric)
            if progress and (rep + 1) % 25 == 0:
                progress({"scenario": name, "completed": rep + 1, "total": repeats})
        for method in ("baseline", "guide-aware"):
            subset = [r for r in scenario_rows if r["method"] == method]
            metrics = [key for key in subset[0] if key not in {"scenario", "method", "seed", "repetition"}]
            row = {"scenario": name, "method": method, "repeats": repeats,
                   **{key: float(np.mean([r[key] for r in subset if r[key] is not None])) if any(r[key] is not None for r in subset) else None for key in metrics}}
            row["fdp_mc_se"] = float(np.std([r["fdp"] for r in subset], ddof=1) / np.sqrt(repeats))
            if base.active_fraction == 0:
                row["all_null_exact_95_interval"] = binomial_interval(sum(r["any_hit"] for r in subset), repeats)
            summaries.append(row)
            if progress:
                progress(row)
        # Checkpoint each completed case. Interrupted runs retain completed evidence.
        fields = list(dict.fromkeys(key for record in records for key in record))
        write_tsv(out / "comparison_runs.tsv", fields, records)
        write_json(out / "comparison_summary.json", {"protocol": protocol, "scenarios": summaries, "complete": False})
    report = {"protocol": protocol, "scenarios": summaries, "complete": True, "production_validated": False,
              "interpretation": "Smaller hit sets are not accuracy evidence; quantify false discoveries, sensitivity loss and unsolved coherent artifacts"}
    write_json(out / "comparison_summary.json", report)
    return report
