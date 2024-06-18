import numpy as np
from scipy import optimize as opt
from scipy.linalg import lstsq

import utils


def update_weights(losses, tol=1e-3, maxiter=100):
    '''Optimize Bernoulli probabilities'''
    weights = 0.95 * np.ones_like(losses)
    new_weights = np.copy(weights)
    for _ in range(maxiter):
        eps = 1 - np.mean(weights)
        ratio = eps / (1 - eps)
        new_weights = np.exp(-losses) / (ratio + np.exp(-losses))
        error = np.linalg.norm(new_weights - weights)
        weights = np.copy(new_weights)
        if error < tol:
            break
    return new_weights


def update_weights_constrained(losses, n_eff, tol=1e-3, maxiter=100):
    """
    Constraint optimization of Bernoulli probabilities
    for the unbounded likelihood case.
    Uses the constraint: `sum(weights) >= n_eff`.
    """

    n = len(losses)
    # First, solve the unconstrained optimization
    new_weights = update_weights(losses, tol=tol, maxiter=maxiter)
    # Then, correct based on the KKT condition
    def shift_obj(s):
        return np.square(
            np.sum(
                np.exp(-losses + s) / ((n - n_eff)/n_eff + np.exp(-losses + s))
            ) - n_eff
        )
    if np.sum(new_weights) < n_eff:
        shift = opt.minimize_scalar(shift_obj)['x']
        new_weights = np.exp(-losses + shift) / ((n - n_eff)/n_eff + np.exp(-losses + shift))
    return new_weights


def mean(sample, maxiter=100, tol=1e-3):
    weights = np.ones(sample.shape[0])
    theta = weights @ sample / np.sum(weights)
    residuals = np.linalg.norm(theta - sample, axis=1)**2
    sigma2 = weights @ residuals / np.sum(weights)
    losses = 0.5 * residuals/sigma2

    for _ in range(maxiter):
        weights = update_weights(losses)
        prev_theta = np.copy(theta)
        theta = weights @ sample / np.sum(weights)
        residuals = np.linalg.norm(theta - sample, axis=1)**2
        sigma2 = weights @ residuals / np.sum(weights)
        losses = 0.5 * residuals/sigma2

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def linear_regression(X, y, maxiter=100, tol=1e-3):
    weights = np.ones(X.shape[0])
    w_sqrt = np.diag(np.sqrt(weights))
    theta = lstsq(w_sqrt @ X, w_sqrt @ y)[0]
    residuals = (y - X @ theta)**2
    sigma2 = weights @ residuals / np.sum(weights)
    losses = 0.5 * residuals/sigma2

    for _ in range(maxiter):
        weights = update_weights(losses)
        prev_theta = np.copy(theta)
        w_sqrt = np.diag(np.sqrt(weights))
        theta = lstsq(w_sqrt @ X, w_sqrt @ y)[0]
        residuals = (y - X @ theta)**2
        sigma2 = weights @ residuals / np.sum(weights)
        losses = 0.5 * residuals/sigma2

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def logistic_regression(X, y, maxiter=100, tol=1e-2):
    weights = np.ones(X.shape[0])
    theta, losses = utils.sklearn_log_reg(X, y, weights)
    # theta, losses = utils.mm_log_reg(X, y, weights)

    for _ in range(maxiter):
        weights = update_weights(losses)

        prev_theta = np.copy(theta)
        theta, losses = utils.sklearn_log_reg(X, y, weights)
        # theta, losses = utils.mm_log_reg(X, y, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def pca(sample, maxiter=100, tol=1e-2, theta_init=None):
    weights = np.ones(sample.shape[0])
    theta, losses = utils.pca(sample, weights, theta_init)

    for _ in range(maxiter):
        weights = update_weights(losses)

        prev_theta = np.copy(theta)
        theta, losses = utils.pca(sample, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def covariance(sample, eps, maxiter=100, tol=1e-2):
    n, d = sample.shape
    n_eff = n * (1 - eps)

    weights = np.ones(n)
    theta, losses = utils.covariance(sample, weights)

    for _ in range(maxiter):
        weights = update_weights_constrained(losses, n_eff)
        prev_theta = np.copy(theta)
        theta, losses = utils.covariance(sample, weights)

        discrepancy = np.linalg.norm(theta - prev_theta, ord='fro') / np.linalg.norm(prev_theta, ord='fro')
        if discrepancy <= tol:
            break

    return theta
