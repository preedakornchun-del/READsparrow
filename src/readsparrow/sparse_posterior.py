"""Model-based sparse-mixture posterior: NOT a frequentist p-value engine.

An explicit broad single-guide artifact component competes with multi-guide
alternatives. The null shape/correlation come exclusively from declared null
controls; only mixture prevalence is estimated from target likelihood ratios.
Posterior expected FDP relies on the entire fitted model. Frequentist FDR must
be evaluated separately, including single-guide and misspecification stresses.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, expit
from scipy.stats import chi2


def sparse_log_bayes_factor(z, rho, inflation=1.15, artifact_probability=.05, *, components=False):
    z = np.asarray(z, dtype=float)
    n, m = z.shape
    if m < 2 or not 0 <= rho < 1 or inflation < 1:
        raise ValueError("Need >=2 guides, rho in [0,1), inflation>=1")
    variance = inflation ** 2
    precision = (np.eye(m) - rho / (1 + (m - 1) * rho) * np.ones((m, m))) / ((1 - rho) * variance)
    precision_z = z @ precision
    diagonal = np.diag(precision)
    # Gaussian random artifact (SD=4 latent units) at exactly one unknown guide.
    artifact_variance = 16.
    artifact_lr = .5 * artifact_variance * precision_z ** 2 / (1 + artifact_variance * diagonal) - .5 * np.log1p(artifact_variance * diagonal)
    log_null = np.logaddexp(np.log1p(-artifact_probability), np.log(artifact_probability) + logsumexp(artifact_lr, axis=1) - np.log(m))
    terms = []
    # Fixed effect grid and cardinality prior; no nominal simulation effect or
    # true guide efficacy is supplied to inference.
    effects = (.75, 1.5, 2.5, 4.)
    for cardinality in range(2, m + 1):
        subsets = list(combinations(range(m), cardinality))
        log_weight = -np.log(m - 1) - np.log(len(subsets)) - np.log(len(effects)) - np.log(2)
        masks = np.array([[float(j in subset) for j in range(m)] for subset in subsets])
        projection = precision_z @ masks.T
        norm_squared = np.einsum("ij,jk,ik->i", masks, precision, masks)
        for effect in effects:
            for sign in (-1., 1.):
                terms.append(logsumexp(sign * effect * projection - .5 * effect ** 2 * norm_squared, axis=1) + log_weight)
    log_alternative = logsumexp(np.array(terms), axis=0)
    if components:
        # A flexible single-guide association class also absorbs biological
        # one-effective-guide positives, which cannot support a >=2 claim.
        single_terms = [logsumexp(artifact_lr, axis=1)-np.log(m)+np.log(.5)]
        single_effects = (.75, 1.5, 2.5, 4., 6.)
        for effect in single_effects:
            for sign in (-1., 1.):
                single_terms.append(logsumexp(sign*effect*precision_z-.5*effect**2*diagonal, axis=1)
                                    -np.log(m)-np.log(2*len(single_effects))+np.log(.5))
        return logsumexp(np.array(single_terms), axis=0), log_alternative
    return log_alternative - log_null


def three_class_decisions(log_single, log_multi, fdr=.05):
    """Jointly fit clean/single/multi prevalence, with no forced discoveries."""
    logs = np.column_stack([np.zeros(len(log_single)), log_single, log_multi])
    if len(logs) < 20 or np.any(~np.isfinite(logs)):
        raise ValueError("Need >=20 finite target log density ratios")
    weights = np.array([.95, .03, .02])
    iterations = 0
    for iterations in range(1000):
        log_posterior = logs + np.log(np.maximum(weights, 1e-300))
        posterior = np.exp(log_posterior-logsumexp(log_posterior, axis=1)[:, None])
        update = posterior.mean(axis=0)
        if update[2] > .5:
            update[:2] *= .5/update[:2].sum(); update[2] = .5
        if np.max(np.abs(update-weights)) < 1e-10:
            weights = update
            break
        weights = update
    # Include the explicit no-multi boundary; finite EM convergence must not
    # manufacture a mandatory positive alternative prevalence.
    single_fit = minimize_scalar(lambda p:-float(np.logaddexp(np.log1p(-p), np.log(p)+logs[:, 1]).sum()),
                                 bounds=(1e-10, 1-1e-10), method="bounded")
    boundary_weight,boundary_likelihood = max([(0.,0.),(1.,float(logs[:,1].sum())),
                                               (float(single_fit.x),-float(single_fit.fun))],key=lambda item:item[1])
    likelihood = float(logsumexp(logs+np.log(np.maximum(weights,1e-300)),axis=1).sum())
    if boundary_likelihood >= likelihood-1e-7:
        weights = np.array([1-boundary_weight,boundary_weight,0.])
    likelihood_ratio = max(0.,2*(likelihood-boundary_likelihood))
    global_pvalue = float(chi2.sf(likelihood_ratio,1))
    posterior = np.exp(logs+np.log(np.maximum(weights,1e-300))-logsumexp(logs+np.log(np.maximum(weights,1e-300)),axis=1)[:,None])
    local = np.clip(1-posterior[:,2],0,1)
    order = np.argsort(local,kind="stable");q=np.empty(len(local))
    q[order] = np.cumsum(local[order])/np.arange(1,len(local)+1)
    return local,q,dict(estimated_alternative_prevalence=float(weights[2]), estimated_single_association_prevalence=float(weights[1]),
        clean_prevalence=float(weights[0]),em_iterations=iterations+1,em_max_iterations=1000,
        global_likelihood_ratio=likelihood_ratio,global_working_pvalue=global_pvalue,
        global_gate_threshold=.01,global_gate_pass=global_pvalue<=.01,
        global_gate_boundary="Conservative ordinary chi-square1 working gate for boundary mixture test; fitted-null/EM/model uncertainty not exactly covered",
        em_converged=iterations<999,upper_prevalence_bound=.5,
        family="Posterior expected FDP under fitted clean/single/multi mixture, NOT BH q-values",
        target_use="Clean/single/multi prevalence only; targets never fit null shape/correlation",hits=int(np.sum(q<=fdr)))


def posterior_decisions(log_bf, fdr=.05):
    """Empirical-Bayes prevalence, monotone posterior expected-FDP values.

    No unadjusted posterior is mislabeled as a p-value. This is model-dependent
    Bayesian FDR, not BH or a theorem about the real screen's frequentist FDR.
    """
    log_bf = np.asarray(log_bf, dtype=float)
    if log_bf.ndim != 1 or len(log_bf) < 20 or np.any(~np.isfinite(log_bf)):
        raise ValueError("Need >=20 finite target log likelihood ratios")
    def objective(prevalence):
        return -float(np.logaddexp(np.log1p(-prevalence), np.log(prevalence) + log_bf).sum())
    fit = minimize_scalar(objective, bounds=(1e-8, .5), method="bounded", options={"xatol": 1e-10})
    candidates = [(0., 0.), (.5, objective(.5)), (float(fit.x), float(fit.fun))]
    prevalence = min(candidates, key=lambda item: item[1])[0]
    if prevalence == 0:
        local = np.ones(len(log_bf))
    else:
        local = expit(np.log1p(-prevalence) - np.log(prevalence) - log_bf)
    order = np.argsort(local, kind="stable")
    q = np.empty(len(local))
    q[order] = np.cumsum(local[order]) / np.arange(1, len(local) + 1)
    return local, q, dict(estimated_alternative_prevalence=prevalence, upper_prevalence_bound=.5,
                          family="Posterior expected FDP under fitted sparse mixture, NOT BH q-values",
                          target_use="Prevalence only; targets never fit null shape/correlation",
                          hits=int(np.sum(q <= fdr)))
