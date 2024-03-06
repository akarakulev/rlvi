import numpy as np
from scipy.linalg import lstsq

import utils


def update_weights(residuals, tol=1e-3, maxiter=100):
    '''Coordinate descent for Bernoulli probabilities'''
    # Scale residuals under exponential function
    scale = 2 * np.sqrt(np.mean(residuals) * np.median(residuals))
    if scale > 1:
        residuals /= scale
    exp_res = np.exp(-residuals)
    weights = 0.95 * np.ones_like(residuals)
    for _ in range(maxiter):
        avg_weight = np.mean(weights)
        ratio = avg_weight / (1 - avg_weight)
        new_weights = ratio * exp_res / (1 + ratio * exp_res)
        error = np.linalg.norm(new_weights - weights)
        weights = np.copy(new_weights)
        if error < tol:
            break
    # Scale weights from [0; w_max] to [0; 1] for convinience
    new_weights = new_weights / np.max(new_weights)
    return new_weights


def mean(sample, maxiter=100, tol=1e-3):
    weights = np.ones(sample.shape[0]) / sample.shape[0]
    theta = weights @ sample / np.sum(weights)
    residuals = np.linalg.norm(theta - sample, axis=1)**2

    for _ in range(maxiter):
        weights = update_weights(residuals)

        prev_theta = np.copy(theta)
        theta = weights @ sample / np.sum(weights)
        residuals = np.linalg.norm(theta - sample, axis=1)**2

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def linear_regression(X, y, maxiter=100, tol=1e-3):
    weights = np.ones(X.shape[0]) / X.shape[0]
    w_sqrt = np.diag(np.sqrt(weights))
    theta = lstsq(w_sqrt @ X, w_sqrt @ y)[0]
    residuals = (y - X @ theta)**2

    for _ in range(maxiter):
        weights = update_weights(residuals)

        prev_theta = np.copy(theta)
        w_sqrt = np.diag(np.sqrt(weights))
        theta = lstsq(w_sqrt @ X, w_sqrt @ y)[0]
        residuals = (y - X @ theta)**2

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def logistic_regression(X, y, maxiter=100, tol=1e-2):
    weights = np.ones(X.shape[0]) / X.shape[0]
    theta, residuals = utils.sklearn_log_reg(X, y, weights)
    # theta, residuals = utils.mm_log_reg(X, y, weights)
    for _ in range(maxiter):
        weights = update_weights(residuals)

        prev_theta = np.copy(theta)
        theta, residuals = utils.sklearn_log_reg(X, y, weights)
        # theta, residuals = utils.mm_log_reg(X, y, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def pca(sample, maxiter=100, tol=1e-2, theta_init=None):
    weights = np.ones(sample.shape[0]) / sample.shape[0]
    theta, residuals = utils.pca(sample, weights, theta_init)

    for _ in range(maxiter):
        weights = update_weights(residuals)

        prev_theta = np.copy(theta)
        theta, residuals = utils.pca(sample, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(prev_theta)
        if discrepancy <= tol:
            break

    return theta


def covariance(sample, maxiter=100, tol=1e-2):
    weights = np.ones(sample.shape[0]) / sample.shape[0]
    theta, residuals = utils.covariance(sample, weights)

    for _ in range(maxiter):
        weights = update_weights(residuals)

        prev_theta = np.copy(theta)
        theta, residuals = utils.covariance(sample, weights)

        discrepancy = np.linalg.norm(theta - prev_theta, ord='fro') / np.linalg.norm(prev_theta, ord='fro')
        if discrepancy <= tol:
            break

    return theta


def clf_breast_cancer(X, y, maxiter=100, tol=1e-2):
    weights = np.ones(X.shape[0]) / X.shape[0]
    # theta, residuals = utils.sklearn_log_reg(X, y, weights)
    theta, residuals = utils.mm_log_reg(X, y, weights)
    for _ in range(maxiter):
        weights = update_weights(residuals)

        prev_theta = np.copy(theta)
        # theta, residuals = utils.sklearn_log_reg(X, y, weights)
        theta, residuals = utils.mm_log_reg(X, y, weights)

        discrepancy = np.linalg.norm(theta - prev_theta) / np.linalg.norm(weights)
        if discrepancy <= tol:
            break

    return theta
