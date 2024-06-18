"""
Based on paper Robust Risk Minimization for Statistical Learning
Osama et al., 2020
"""
import numpy as np
from scipy import optimize as opt
from scipy.linalg import lstsq

import utils


def update_weights(losses, eps):
    '''Optimize sample weights for Robust Risk Minimization'''
    res = np.copy(losses)
    t = -np.log((1 - eps) * res.shape[0])
    numeric_cutoff = 1e-16
    def objective(xi):
        phi = np.exp(-res * np.exp(-xi))
        phi[phi < numeric_cutoff] = numeric_cutoff
        sum_phi = np.sum(phi)
        return np.exp(xi) * (np.log(sum_phi) + t)

    opt_res = opt.minimize_scalar(objective)
    opt_xi = opt_res['x']
    opt_alpha = np.exp(opt_xi)

    phi = np.exp(-res / opt_alpha)
    phi[phi < numeric_cutoff] = numeric_cutoff
    sum_phi = np.sum(phi)
    opt_beta_over_alpha = np.log(sum_phi) - 1

    opt_weights = np.exp(-res / opt_alpha) * np.exp(-opt_beta_over_alpha - 1)
    return opt_weights


def mean(sample, eps, maxiter=100, tol=1e-3):
    weights = np.ones(sample.shape[0]) / sample.shape[0]
    theta = weights @ sample / np.sum(weights)
    losses = np.linalg.norm(theta - sample, axis=1)**2

    for _ in range(maxiter):
        weights = update_weights(losses, eps)

        prev_theta = np.copy(theta)
        theta = weights @ sample / np.sum(weights)
        losses = np.linalg.norm(theta - sample, axis=1)**2

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def linear_regression(X, y, eps, maxiter=100, tol=1e-3):
    weights = np.ones(X.shape[0]) / X.shape[0]
    w_sqrt = np.diag(np.sqrt(weights))
    theta = lstsq(w_sqrt @ X, w_sqrt @ y)[0]
    losses = (y - X @ theta)**2

    for _ in range(maxiter):
        weights = update_weights(losses, eps)

        prev_theta = np.copy(theta)
        w_sqrt = np.diag(np.sqrt(weights))
        theta = lstsq(w_sqrt @ X, w_sqrt @ y)[0]
        losses = (y - X @ theta)**2

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def logistic_regression(X, y, eps, maxiter=100, tol=1e-2):
    weights = np.ones(X.shape[0]) / X.shape[0]
    theta, losses = utils.sklearn_log_reg(X, y, weights)
    # theta, losses = utils.mm_log_reg(X, y, weights)
    for _ in range(maxiter):
        weights = update_weights(losses, eps)

        prev_theta = np.copy(theta)
        theta, losses = utils.sklearn_log_reg(X, y, weights)
        # theta, losses = utils.mm_log_reg(X, y, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def pca(sample, eps, maxiter=100, tol=1e-2, theta_init=None):
    weights = np.ones(sample.shape[0]) / sample.shape[0]
    theta, losses = utils.pca(sample, weights, theta_init)
    for _ in range(maxiter):
        weights = update_weights(losses, eps)

        prev_theta = np.copy(theta)
        theta, losses = utils.pca(sample, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def covariance(sample, eps, maxiter=100, tol=1e-2):
    weights = np.ones(sample.shape[0]) / sample.shape[0]
    theta, losses = utils.covariance(sample, weights)

    for _ in range(maxiter):
        weights = update_weights(losses, eps)

        prev_theta = np.copy(theta)
        theta, losses = utils.covariance(sample, weights)

        discrepancy = np.linalg.norm(theta - prev_theta, ord='fro') / np.linalg.norm(prev_theta, ord='fro')
        if discrepancy <= tol:
            break

    return theta
