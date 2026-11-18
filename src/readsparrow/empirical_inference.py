"""Control-derived nuisance variation and held-out empirical signed tails.

Both are experimental calibration models, NOT CNV correction. No target effect
or HQ461 hit count is used to fit the nuisance distribution. Empirical ranks
never extrapolate beyond observed calibration controls. Their finite resolution
is a real power limitation, not a reason to report zero probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from functools import lru_cache

import numpy as np
from scipy.special import digamma
from scipy.stats import t as student_t

from .guide_inference import GuideConfig, call_guide_hits


@dataclass
class EmpiricalConfig(GuideConfig):
    null_model: str = "background"
    min_null_controls: int = 100
    max_background_bins: int = 3
    min_training_per_bin: int = 40
    split_salt: str = "READsparrow-null-v1"

    def validate(self):
        super().validate()
        if self.null_model not in {"background", "empirical"}:
            raise ValueError("null_model must be background or empirical")
        if self.min_null_controls < 100 or self.min_training_per_bin < 30:
            raise ValueError("Need >=100 informative null controls and >=30 training controls per bin")
        if not 1 <= self.max_background_bins <= 10:
            raise ValueError("Use 1..10 prespecified background bins")
        if not self.split_salt:
            raise ValueError("Control split salt must not be empty")


def split_controls(ids, mask, salt):
    """ID-hashed balanced split, independent of counts/effects and row order."""
    indices = np.flatnonzero(mask)
    order = sorted(indices, key=lambda i: (hashlib.sha256((salt + "\0" + ids[i]).encode()).digest(), ids[i]))
    half = len(order) // 2
    return np.array(order[:half], dtype=int), np.array(order[half:], dtype=int)


def signed_rank_p(reference, scores):
    """Conservative (1 + # controls at least as extreme)/(n+1), ties included."""
    ref = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(scores, dtype=float)
    if not len(ref) or np.any(~np.isfinite(ref)) or np.any(~np.isfinite(values)):
        raise ValueError("Finite scores and nonempty calibration controls required")
    pp = (1 + len(ref) - np.searchsorted(ref, values, side="left")) / (len(ref) + 1)
    pn = (1 + np.searchsorted(ref, values, side="right")) / (len(ref) + 1)
    return pp, pn


def predictive_background(beta, measurement_se, abundance, training, config, measurement_df):
    """Method-of-moments nuisance variance, never shrinks measurement SE.

    beta_control = mu(bin) + nuisance + measurement error. The nuisance variance
    is max(0, sample variance of beta_control - mean measurement variance).
    A prediction variance term accounts approximately for estimated location.
    Student tails limit, but do not eliminate, finite-control uncertainty.
    """
    bins = min(config.max_background_bins, max(1, len(training) // config.min_training_per_bin))
    edges = np.unique(np.quantile(abundance[training], np.arange(1, bins) / bins)) if bins > 1 else np.array([])
    assignment = np.searchsorted(edges, abundance, side="right")
    # Degenerate/tied count distributions may leave small or empty bins. Collapse
    # globally rather than choose bins based on treatment effects or discoveries.
    if any(np.sum(assignment[training] == j) < config.min_training_per_bin for j in range(len(edges) + 1)):
        edges = np.array([])
        assignment = np.zeros(len(beta), dtype=int)
    center, tau, center_var, dfs = [np.zeros(len(beta)) for _ in range(4)]
    diagnostics = []
    for j in range(len(edges) + 1):
        idx = training[assignment[training] == j]
        mean = float(np.mean(beta[idx]))
        observed = float(np.var(beta[idx], ddof=1))
        # A t scale SE is not its variance. Retain the old measurement model's
        # variance inflation when estimating the extra nuisance component.
        inflate = measurement_df / (measurement_df - 2) if measurement_df > 2 else 10.0
        expected = float(np.mean(measurement_se[idx] ** 2)) * inflate
        extra = max(0., observed - expected)
        mask = assignment == j
        center[mask], tau[mask], center_var[mask] = mean, extra, observed / len(idx)
        dfs[mask] = min(measurement_df, len(idx) - 1)
        diagnostics.append({"bin": j, "training_controls": len(idx), "control_log2_abundance_min": float(abundance[idx].min()),
            "control_log2_abundance_max": float(abundance[idx].max()), "null_center": mean,
            "observed_control_effect_variance": observed, "expected_measurement_variance": expected,
            "extra_background_variance": extra, "background_sd": float(np.sqrt(extra)),
            "location_prediction_variance": observed / len(idx), "predictive_df": float(dfs[mask][0])})
    prediction_se = np.sqrt(np.maximum(measurement_se ** 2 + tau + center_var, 1e-12))
    return center, prediction_se, dfs, assignment, diagnostics


def call_empirical_hits(table, metadata, treatment, control, config=None, control_ids=None, *, _context_sink=None):
    config = config or EmpiricalConfig()
    config.validate()
    audit_rows = []

    def transform(ctx):
        y, design, beta, eligible = ctx["all_y"], ctx["design"], ctx["beta"].copy(), ctx["eligible"]
        controls, normalized = ctx["controls"], ctx["normalized"]
        # Fit controls using exactly the target-trained variance trend, not a
        # separately optimized model and not the held-out controls' effects.
        coef = np.linalg.lstsq(design, y.T, rcond=None)[0].T
        residual = y - coef @ design.T
        variances = (residual ** 2).sum(axis=1) / ctx["residual_df"]
        prior = ctx["moderation"]
        trend = np.interp(y.mean(axis=1), prior["trend_knots"], prior["trend_log_levels"])
        prior_variance = np.exp(trend + digamma(prior["prior_df"] / 2) - np.log(prior["prior_df"] / 2))
        posterior = (prior["prior_df"] * prior_variance + ctx["residual_df"] * variances) / (prior["prior_df"] + ctx["residual_df"])
        measurement_se = np.sqrt(np.maximum(posterior, prior["variance_floor"]) * np.linalg.inv(design.T @ design)[1, 1])
        # Exact legacy values for targets; only control predictions were missing.
        measurement_se[eligible] = ctx["se"][eligible]
        informative_control = controls & ((normalized >= config.min_guide_count).sum(axis=1) >= ctx["min_samples"])
        if informative_control.sum() < config.min_null_controls:
            raise ValueError(f"Need >={config.min_null_controls} informative declared null controls; found {informative_control.sum()}; no silent legacy fallback")
        train, calibration = split_controls(table.ids, informative_control, config.split_salt)
        # Baseline-only abundance avoids binning on the observed treatment effect.
        abundance = np.log2(normalized[:, ~ctx["treated"]].mean(axis=1) + config.pseudocount)
        center, prediction_se, dfs, assignment, bins = predictive_background(beta, measurement_se, abundance, train, config, ctx["df"])
        adjusted = beta - center
        z = adjusted / prediction_se
        pp_background, pn_background = student_t.sf(z, dfs), student_t.cdf(z, dfs)
        if config.null_model == "empirical":
            pp, pn = signed_rank_p(z[calibration], z)
        else:
            pp, pn = pp_background.copy(), pn_background.copy()
        pp[~eligible], pn[~eligible] = 1., 1.
        se = prediction_se.copy()
        se[~eligible] = np.inf
        reference = np.sort(z[calibration])

        @lru_cache(maxsize=32)
        def bounds(m, confidence):
            alpha = (1 - confidence) / (2 * m)
            if config.null_model == "background":
                quantile = student_t.ppf(1 - alpha, dfs)
                lo, hi = adjusted - quantile * prediction_se, adjusted + quantile * prediction_se
            else:
                # Exact inversion of the conservative signed finite-rank tails.
                # If fewer than one tail observation is resolved, the interval
                # is unbounded; never replace the missing tail by Gaussian tails.
                k = int(np.floor(alpha * (len(reference) + 1) - 1 + 1e-12))
                if k < 0:
                    lo, hi = np.full(len(beta), -np.inf), np.full(len(beta), np.inf)
                else:
                    lower_z, upper_z = reference[k], reference[-k - 1]
                    lo, hi = adjusted - upper_z * prediction_se, adjusted - lower_z * prediction_se
            lo[~eligible], hi[~eligible] = -np.inf, np.inf
            return lo, hi

        train_set, calibration_set = set(train.tolist()), set(calibration.tolist())
        control_pp, control_pn = signed_rank_p(z[calibration], z[calibration])
        # Audit p-values for empirical controls use LEAVE-ONE-OUT reference, so
        # their own observed score is not included among the null comparisons.
        n = len(calibration)
        loo_pp = (control_pp * (n + 1) - 1) / n
        loo_pn = (control_pn * (n + 1) - 1) / n
        for local, i in enumerate(calibration):
            audit_rows.append({"sgRNA": table.ids[i], "role": "held_out_calibration", "background_bin": int(assignment[i]),
                "raw_log2_effect": float(beta[i]), "null_center": float(center[i]), "measurement_se": float(measurement_se[i]),
                "prediction_se": float(prediction_se[i]), "standardized_effect": float(z[i]),
                "uncalibrated_two_sided_p": float(2 * student_t.sf(abs(beta[i] / measurement_se[i]), ctx["df"])),
                "background_two_sided_p": float(min(1, 2 * min(pp_background[i], pn_background[i]))),
                "rank_leave_one_out_two_sided_p": float(min(1, 2 * min(loo_pp[local], loo_pn[local])))})
        p0 = np.array([r["background_two_sided_p"] for r in audit_rows])
        background_p_lt_005 = float(np.mean(p0 < .05))
        # A deliberately simple, prespecified advisory alarm, NOT a proof that
        # passing implies gene-level FDR control or a trigger for model switching.
        alarm_threshold = .05 + 3 * np.sqrt(.05 * .95 / len(calibration))
        alarms = []
        if background_p_lt_005 > alarm_threshold:
            alarms.append("held_out_background_tail_excess")
        if (p0 < .01).mean() > .01 + 3 * np.sqrt(.01 * .99 / len(calibration)):
            alarms.append("held_out_background_extreme_tail_excess")
        outside = ((abundance < abundance[train].min()) | (abundance > abundance[train].max())) & eligible
        if outside.mean() > .1:
            alarms.append("many_targets_outside_training_control_abundance_range")
        if config.normalization == "control":
            norm_note = "Normalization uses all declared controls, including calibration controls; audit is effect-fit-held-out, not a fully independent end-to-end holdout"
        else:
            norm_note = "Target variance moderation and normalization share data; ranks require approximately exchangeable standardized control/target null effects, not an unconditional distribution-free guarantee"
        floor = 1 / (len(calibration) + 1)
        guide_extra = [{"raw_log2_effect": float(beta[i]), "background_center": float(center[i]),
            "measurement_standard_error": float(measurement_se[i]), "background_prediction_scale": float(prediction_se[i]),
            "background_bin": int(assignment[i]), "control_abundance_extrapolation": bool(outside[i]),
            "null_model": config.null_model, "one_sided_rank_p_floor": floor if config.null_model == "empirical" else None}
            for i in range(len(beta))]
        warnings = ["Null controls must be experimentally valid and representative; non-targeting controls do not model cutting/CNV toxicity",
            "Background model includes persistent guide variation missed by replicate residuals; it does not identify its biological cause",
            "Background Student-t tails are model-based extrapolation; parameter uncertainty is only approximately accounted for",
            "Control-based nuisance centering changes the estimand; raw and background-relative effects are both exported",
            norm_note, "Held-out empirical-control rank uniformity is largely structural, not independent validation of target-gene FDR",
            "No post-hoc model switching, guide deletion, significance-threshold tuning or MAGeCK hit-count matching"]
        if config.null_model == "empirical":
            warnings.append(f"Finite rank resolution: one-sided guide p >= {floor:.6g}; this can make multi-guide/BH discoveries impossible despite strong effects")
        warnings.extend(alarms)
        if _context_sink is not None:
            _context_sink.update(beta=beta, measurement_se=measurement_se, prediction_se=prediction_se, z=z,
                train=train, calibration=calibration, assignment=assignment, bins=bins, df=ctx["df"])
        return dict(beta=adjusted, raw_beta=beta, se=se, df=dfs, p_positive=pp, p_negative=pn,
            interval_bounds=bounds, guide_extra=guide_extra, warnings=warnings,
            info={"method": config.null_model + "-control-calibrated-directional-partial-conjunction",
                "release_status": "experimental-background-empirical-alpha", "config": asdict(config),
                "effect_estimand": "median biological-guide contrast relative to estimated count-stratified null-control location",
                "confidence_interval": "Median of approximate simultaneous predictive guide bounds; uncertainty includes nuisance variation; empirical unresolved tails unbounded",
                "background_model": {"bins": bins, "training_controls": len(train), "calibration_controls": len(calibration),
                    "split_rule": "balanced ID-SHA256 ordering; first half training, second half calibration", "split_salt": config.split_salt,
                    "training_control_ids": [table.ids[i] for i in train], "calibration_control_ids": [table.ids[i] for i in calibration],
                    "fitted_from_target_effects": False, "one_sided_rank_p_floor": floor,
                    "held_out_background_p_lt_005": background_p_lt_005, "held_out_background_p_lt_001": float((p0 < .01).mean()),
                    "advisory_tail_alarm_threshold_005": float(alarm_threshold), "advisory_alarms": alarms,
                    "target_abundance_extrapolation_fraction": float(outside[eligible].mean()),
                    "rank_tail_pooling": "all calibration controls after training-only abundance standardization; approximate exchangeability assumption"},
                "no_gene_FDR_guarantee_from_controls_alone": True})

    rows, info, guides = call_guide_hits(table, metadata, treatment, control, config, control_ids, _evidence_transform=transform)
    info["control_audit"] = audit_rows
    return rows, info, guides
