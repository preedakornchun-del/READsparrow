# Boundary-stable shared-null candidate

The scientific v0.9 modules are retained byte-for-byte; the new code lives in
stable_null.py and stable_inference.py. No MAGeCK/R subprocess is used for
standalone inference. Counting and the default background caller are unchanged.

## Integration

The shared nuisance covariance is a(I-J/K)+bJ/K, a,b in [0,100]. Measurement
covariance is separate. The reference shift is in [-8,8]; single-guide
prevalence s and all-guide fraction eta are in [0,1]; multi prevalence p is in
[0,0.5]. Each bounded coordinate theta = low + width*expit(u) is transformed to
R with the exact log-Jacobian log(width)-softplus(u)-softplus(-u).

The complete control/target family likelihood and physical priors are evaluated
at every shared draw. BFGS locates a transformed-density mode. A finite-
difference analytic-gradient Hessian defines a proposal covariance with
eigenvalues floored at 0.05 (proposal only). Draws come from a defensive mixture
0.9*t_8(C)+0.1*t_8(4C). Exact mixture proposal densities correct all importance
weights. No truncation rejection or Monte Carlo support-mass estimate is used.
The p prior remains Uniform(0,0.5), including its density factor 2. Equal prior
mass is assigned to explicit no-multi and multi models. Priors common to both
models cancel in evidence ratios. Finite-draw ESS/MCSE are diagnostic estimates;
there is no guarantee that all modes/tails are covered.

## Continuous coherent effects

The selected guides share a signed effect delta with normalized nonlocal prior
delta^2*N(delta;0,v)/v. With learning enabled, omega=log(v) has a properly
normalized truncated N(0,1) prior over [-4,4] and is integrated jointly with all
other full-model parameters. The inference receives neither true effects nor
guide efficacies. A physical-coordinate MAP and transformed integration mode
are both exported; neither is a posterior-mean effect estimate. Single-guide
components retain half fixed nonlocal N(0,1), half broad N(0,16). The learned
eta and partial-subset prior are unchanged from v0.9.

## Measurement-scale approximation

Three generalized Gauss-Laguerre nodes integrate an inverse-chi-square scale
mixture on the moderated measurement covariance only. Its scale multiplier is
(nu-2)/(2*x), x~Gamma(nu/2), so the exact mixing distribution preserves the
moment covariance. Target and control nu are separately estimated from their
biological residuals. Nuisance covariance is not multiplied. The finite-node
quadrature is approximate, not a new exact count likelihood, and the entire
moderated measurement matrix (including its read-sampling floor) is treated as
uncertain. Reference covariance is retained by the same sample residualization
as v0.9. Student tails and a shared scale per guide vector are working assumptions.
Configuration stable_measurement_quadrature=0 restores the Gaussian moment
approximation; learn_effect_scale=False restores fixed v=1 for ablation.

## Decision and scope

Local posterior null means clean OR exactly-one-associated guide, with explicit
no-multi model averaging. Sorted cumulative mean local null probabilities are
posterior expected-FDP decision values at threshold 0.05, NOT BH-adjusted
p-values or guaranteed frequentist FDR. Matched new inference claims >=2
coherent guides and explicitly reports required_support=2, including target
guide counts 6/8. It is not a robustness guarantee against >=2 coherent artifacts.
No inferential effect interval is supplied; median effects/signs are descriptive.

Without genuine neutral groups, or when training-control shape selection
rejects a Gaussian nuisance, the unchanged v0.7 guarded branch is used. This
fallback retains its original ceil(0.5*m) support definition; its metadata is not
relabeled as a two-guide claim. Non-Gaussian nuisance and representative controls
remain substantive limitations. All candidates are research alpha.
