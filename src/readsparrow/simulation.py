"""Seeded gene-truth count simulations, not a substitute for biological validation.

NB gene totals followed by multinomial guide allocation supply one model;
Poisson-lognormal totals supply a misspecified alternative. Technical-artifact
scenarios retain BIOLOGICAL truth labels even when observed counts are biased.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .counting import Library
from .io import CountTable, save_counts, write_json, write_tsv


@dataclass
class SimulationConfig:
    seed: int = 20261022
    genes: int = 1000
    guides_per_gene: int = 4
    replicates: int = 3
    controls: int = 100
    active_fraction: float = 0.1
    effect: float = 1.5
    mean_count: float = 400.0
    dispersion: float = 0.05
    model: str = "nb"
    paired: bool = False
    artifact_fraction: float = 0.0
    ineffective_fraction: float = 0.0
    depth_sd: float = 0.15
    positive_fraction: float = 0.5

    def validate(self):
        if min(self.genes, self.guides_per_gene, self.replicates, self.controls) < 1 or self.replicates < 2:
            raise ValueError("Positive simulation sizes and >=2 biological replicates required")
        for x in (self.active_fraction, self.artifact_fraction, self.ineffective_fraction, self.positive_fraction):
            if not 0 <= x <= 1:
                raise ValueError("Fractions must be in [0,1]")
        if self.model not in {"nb", "lognormal"}:
            raise ValueError("Simulation model must be nb or lognormal")
        if not all(np.isfinite(x) and x > 0 for x in (self.effect, self.mean_count, self.dispersion)) or not np.isfinite(self.depth_sd) or self.depth_sd < 0:
            raise ValueError("Invalid simulation effect/count/dispersion/depth parameters")


def simulate(config=None):
    config = config or SimulationConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    g, k, r = config.genes, config.guides_per_gene, config.replicates
    samples = [f"{condition}_{i + 1}" for condition in ("control", "treatment") for i in range(r)]
    treated = np.array([False] * r + [True] * r)
    metadata = [{"sample": s, "condition": "treatment" if treated[i] else "control", "biological_replicate": f"bio_{i + 1}",
                 **({"pair": f"pair_{i % r + 1}"} if config.paired else {})} for i, s in enumerate(samples)]
    target_effect = np.zeros(g)
    active = rng.choice(g, int(g * config.active_fraction), replace=False)
    n_positive = int(round(len(active) * config.positive_fraction))
    target_effect[active] = config.effect * np.where(np.arange(len(active)) < n_positive, 1, -1)
    proportions = rng.dirichlet(np.repeat(2.0, k), size=g)
    efficacy = (rng.random((g, k)) >= config.ineffective_fraction).astype(float)
    fold = np.exp2(target_effect[:, None] * efficacy)
    true_effect = np.log2(np.sum(proportions * fold, axis=1))
    # Floating point sum(proportions) may differ from 1 by ~1e-16. Such roundoff
    # must NEVER turn a null gene into an active truth label (especially all-null).
    true_effect[(target_effect == 0) | (~np.any(efficacy > 0, axis=1))] = 0.0
    base = rng.lognormal(np.log(config.mean_count), 0.7, g)
    alpha = config.dispersion * rng.lognormal(0, 0.35, g)
    depths = np.exp(rng.normal(0, config.depth_sd, 2 * r))
    pair_multiplier = rng.lognormal(0, 0.6, (g, r)) if config.paired else np.ones((g, r))
    gene_counts = np.zeros((g, k, 2 * r), dtype=np.int64)
    for j in range(2 * r):
        mu = base * depths[j] * pair_multiplier[:, j % r]
        fractions = proportions.copy()
        if treated[j]:
            mu *= np.exp2(true_effect)
            fractions *= fold
            fractions /= fractions.sum(axis=1, keepdims=True)
        if config.model == "nb":
            total = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
        else:
            variance = np.log1p(alpha)
            total = rng.poisson(mu * np.exp(rng.normal(-variance / 2, np.sqrt(variance))))
        for i in range(g):
            gene_counts[i, :, j] = rng.multinomial(total[i], fractions[i])
    # Inject a consistent condition-specific technical guide artifact into null genes.
    null_genes = np.flatnonzero(true_effect == 0)
    artifact_genes = rng.choice(null_genes, int(len(null_genes) * config.artifact_fraction), replace=False)
    for i in artifact_genes:
        gene_counts[i, 0, r:] *= 8
    control_counts = rng.poisson(np.maximum(1, config.mean_count / k) * depths, size=(config.controls, 2 * r))
    counts = np.vstack([gene_counts.reshape(g * k, 2 * r), control_counts])
    ids = [f"g{i + 1:05d}_sg{j + 1}" for i in range(g) for j in range(k)] + [f"NTC_{i + 1:04d}" for i in range(config.controls)]
    genes = [f"GENE_{i + 1:05d}" for i in range(g) for _ in range(k)] + ["NA"] * config.controls
    truth = [{"gene": f"GENE_{i + 1:05d}", "true_log2_effect": float(true_effect[i]), "active": bool(true_effect[i] != 0),
              "technical_artifact": bool(i in artifact_genes), "nominal_effect": float(target_effect[i]),
              "true_median_guide_effect": float(np.median(target_effect[i] * efficacy[i])),
              "effective_guides": int(np.sum(target_effect[i] * efficacy[i] != 0))} for i in range(g)]
    return CountTable(ids, genes, samples, counts), metadata, truth, {"config": asdict(config), "true_size_factors": depths.tolist(),
        "truth_definition": "Expected biological gene-total abundance ratio before technical artifacts; not guide knockout efficacy", 
        "generator": "gene total -> multinomial guide allocation; biological variation shared within gene/sample",
        "seed": config.seed}


def write_simulation(out, config):
    table, metadata, truth, manifest = simulate(config)
    save_counts(out / "counts.tsv", table)
    write_tsv(out / "samples.tsv", list(metadata[0]), metadata)
    write_tsv(out / "gene_truth.tsv", list(truth[0]), truth)
    # Small unique sequence reference for count-matrix examples; no huge FASTQs generated.
    rng = np.random.default_rng(config.seed + 1)
    sequences, used = [], set()
    for _ in table.ids:
        while True:
            seq = "".join(rng.choice(list("ACGT"), size=20))
            if seq not in used:
                used.add(seq)
                sequences.append(seq)
                break
    Library(table.ids, table.genes, [s.encode() for s in sequences])
    write_tsv(out / "library.tsv", ["sgRNA", "Gene", "Sequence"],
              ({"sgRNA": gid, "Gene": gene, "Sequence": seq} for gid, gene, seq in zip(table.ids, table.genes, sequences)))
    write_json(out / "simulation.json", manifest)
    return table, metadata, truth
