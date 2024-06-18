import numpy as np
from scipy.linalg import lstsq
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


def mm_log_reg(X, y, weights):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    theta0 = np.zeros(X.shape[1])

    X_tilde = 0.5 * X * np.sqrt(weights)[:, None]
    Q = X_tilde.T @ X_tilde
    Q_inv = np.linalg.inv(Q)

    def grad(theta):
        return X.T @ (weights * (sigmoid(X @ theta) - y))

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

    losses = cross_entropy(X, theta_best, y)
    return theta_best, losses


def sklearn_log_reg(X, y, weights, reg_coeff=1e2):
    def get_losses(classifier):
        log_proba = classifier.predict_log_proba(X).T[0]
        return -y * log_proba - (1 - y) * log_proba

    weights /= np.max(weights)
    clf = LogisticRegression(solver="liblinear", C=reg_coeff)
    clf.fit(X, y, sample_weight=weights)
    theta = np.hstack([clf.intercept_.flatten(), clf.coef_.flatten()])

    # Compute loss on samples
    losses = get_losses(clf)
    return theta, losses


def pca(samples, weights, theta=None):
    def get_losses(princ_comp):
        proj = samples @ princ_comp
        return np.sum(samples**2, axis=1) - proj**2

    if theta is None:
        pca_ = PCA(n_components=1)
        pca_.fit(np.diag(weights) @ samples)
        theta = pca_.components_[0]
        theta /= np.linalg.norm(theta)

    # Compute loss on samples
    losses = get_losses(theta)
    return theta, losses


def covariance(samples, weights, mean=None):
    def get_losses(mean, cov, weights):
        d = mean.shape[0]
        centered_samples = samples - mean
        scaled_samples = lstsq(cov, centered_samples.T)[0]
        residuals = np.sum(centered_samples * scaled_samples.T, axis=1)
        (sign, logabsdet) = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Singular covariance matrix")
        return 0.5*(residuals + logabsdet + d*np.log(2*np.pi))

    mean = samples.T @ weights / np.sum(weights)
    centered_samples = samples - mean
    cov = centered_samples.T @ np.diag(weights) @ centered_samples / np.sum(weights)
    # Compute loss on samples
    losses = get_losses(mean, cov, weights)
    return cov, losses
