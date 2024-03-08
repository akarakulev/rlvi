import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def sigmoid(x):
    '''Numerically stable logistic function'''
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def cross_entropy(X, theta, y):
    phi = X @ theta
    return -y * phi + phi + np.log1p(np.exp(-phi))


def clf_predict(X, theta, augment=True):
    if augment:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    phi = X @ theta
    proba = sigmoid(phi)
    return np.array(proba > 0.5, dtype=int)


def mm_log_reg(X, y, weights=None):
    if weights is None:
        weights = np.ones_like(y)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    theta0 = np.zeros(X.shape[1])

    Q = X.T @ np.diag(weights) @ X / 4.
    Q_inv = np.linalg.inv(Q)

    def grad(theta):
        return X.T @ np.diag(weights) @ (sigmoid(X @ theta) - y)

    def delta(theta):
        g = grad(theta)
        return -Q_inv @ g

    theta1 = theta0 + delta(theta0)

    while True:
        if np.linalg.norm(theta1 - theta0) > 1e-2:
            theta0 = theta1
            theta1 = theta0 + delta(theta0)
        else:
            theta_best = theta1
            break

    residuals = cross_entropy(X, theta_best, y)
    return theta_best, residuals


def sklearn_log_reg(X, y, weights=None, reg_coeff=1e2):
    def get_residuals(classifier):
        log_proba = classifier.predict_log_proba(X).T[0]
        return -y * log_proba - (1 - y) * log_proba

    if weights is None:
        weights = np.ones_like(y, dtype=float)
    weights /= np.max(weights)
    clf = LogisticRegression(solver="liblinear", C=reg_coeff)
    clf.fit(X, y, sample_weight=weights)
    theta = np.hstack([clf.intercept_.flatten(), clf.coef_.flatten()])

    # Compute residuals
    residuals = get_residuals(clf)
    return theta, residuals


def pca(samples, weights=None, theta=None):
    def get_residuals(princ_comp):
        proj = samples @ princ_comp.reshape((-1, 1))
        return np.sum(samples**2, axis=1) - np.sum(proj**2, axis=1)

    if weights is None:
        weights = np.ones(len(samples))

    if theta is None:
        pca_ = PCA(n_components=1)
        pca_.fit(np.diag(weights) @ samples)
        theta = pca_.components_[0]
        theta /= np.linalg.norm(theta)

    # Compute residuals
    residuals = get_residuals(theta)
    return theta, residuals


def covariance(samples, weights=None, theta=None):
    def get_residuals(cov):
        mean = np.mean(samples, axis=0)
        scaled_samples = np.linalg.inv(cov) @ (samples - mean.T).T
        res = np.diag((samples - mean.T) @ scaled_samples)
        (sign, logabsdet) = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Non PSD covariance matrix")
        return res + logabsdet

    if weights is None:
        weights = np.ones(len(samples), 1) / len(samples)

    if theta is None:
        mean = samples.T @ weights / np.sum(weights)
        centered_samples = samples - mean
        weighted_cov = centered_samples.T @ np.diag(weights) @ centered_samples / np.sum(weights)
        theta = weighted_cov

    # Compute residuals
    residuals = get_residuals(theta)
    return theta, residuals
