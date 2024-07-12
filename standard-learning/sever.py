"""
Based on paper SEVER: a robust meta-algorithm for stochastic optimization
Ilias Diakonikolas et al., 2019
"""
import numpy as np
from scipy.linalg import lstsq
import utils


def linear_regression(X, y, eps, numiter=4):
    n = len(y)
    s = range(n)  # active set
    # X and y corresponding to active set
    Xs = X[s, :]
    ys = y[s]
        
    for _ in range(numiter):
        # base learner: least-square
        theta = lstsq(Xs, ys)[0]
        
        # gradients
        G_uncen = 2 * (Xs @ theta - ys)[:, np.newaxis] * Xs
        G_cen = G_uncen - np.mean(G_uncen, axis=0)

        # top right singular vector
        V = np.linalg.svd(G_cen)[-1]
        v = V[:, 0]
        
        # outlier scores
        tau = (G_cen @ v)**2
        
        # remove p points with highest scores
        p = int(eps / 2 * Xs.shape[0])
        idx = np.argsort(-tau)
        
        # new active set
        s = idx[p:]
        Xs = Xs[s, :]
        ys = ys[s]

    return theta


def logistic_regression(X, y, eps, numiter=4):
    n = len(y)
    s = range(n)  # active set
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    # X and y corresponding to active set
    Xs = X[s, :]
    ys = y[s]

    for _ in range(numiter):
        # base learner: logistic regression
        theta, _ = utils.sklearn_log_reg(Xs[:, 1:], ys, weights=np.ones(len(ys)))
        # theta, _ = utils.mm_log_reg(Xs[:, 1:], ys, weights=np.ones(len(ys)))

        # gradients
        phi = Xs @ theta
        proba = utils.sigmoid(phi)
        G_uncen = (-ys * (Xs @ theta) + proba)[:, np.newaxis] * Xs
        G_cen = G_uncen - np.mean(G_uncen, axis=0)

        # top right singular vector
        V = np.linalg.svd(G_cen)[-1]
        v = V[:, 0]

        # outlier scores
        tau = (G_cen @ v)**2
        
        # remove p points with highest scores
        p = int(eps / 2 * Xs.shape[0])
        idx = np.argsort(-tau)
        
        # new active set
        s = idx[p:]
        Xs = Xs[s, :]
        ys = ys[s]

    return theta


def pca(samples, eps, numiter=4, theta_init=None):
    n = len(samples)
    s = range(n)  # active set
    # samples corresponding to active set
    samples_s = samples[s, :]

    for k in range(numiter):
        # base learner: pca
        if (theta_init is not None) and (k == 0):
            theta = theta_init
        else:
            theta, _ = utils.pca(samples_s, weights=np.ones(len(samples_s)))

        # gradients
        G_uncen = -2 * ((samples_s @ theta)[:, np.newaxis] * samples_s)
        G_cen = G_uncen - np.mean(G_uncen, axis=0)

        # top right singular vector
        V = np.linalg.svd(G_cen)[-1]
        v = V[:, 0]

        # outlier scores
        tau = (G_cen @ v)**2
        
        # remove p points with highest scores
        p = int(eps / 2 * samples_s.shape[0])
        idx = np.argsort(-tau)
        
        # new active set
        s = idx[p:]
        samples_s = samples_s[s, :]

    return theta


def covariance(samples, eps, numiter=4):
    n, d = samples.shape
    s = range(n)  # active set
    # samples corresponding to active set
    samples_s = samples[s, :]

    for _ in range(numiter):
        # base learner: sample covariance
        mean = np.mean(samples_s, axis=0)
        cov = np.cov(samples_s, rowvar=False)
        cov_inv = np.linalg.inv(cov)

        # gradients
        G_uncen = np.zeros((samples_s.shape[0], d + d**2))
        for row in range(len(samples_s)):
            G_uncen[row, :d] = cov_inv @ (mean - samples_s[row, :])
            centered = (samples_s[row, :] - mean).reshape((-1, 1))
            row_cov = centered @ centered.T
            g = cov_inv @ (np.eye(d) - row_cov / cov) / 2.
            g = g.reshape(-1)
            G_uncen[row, d:] = g

        G_cen = G_uncen - np.mean(G_uncen, axis=0)

        # top right singular vector
        V = np.linalg.svd(G_cen)[-1]
        v = V[:, 0]

        # outlier scores
        tau = (G_cen @ v)**2
        
        # remove p points with highest scores
        p = int(eps / 2 * samples_s.shape[0])
        idx = np.argsort(-tau)
        
        # new active set
        s = idx[p:]
        samples_s = samples_s[s, :]

    return cov