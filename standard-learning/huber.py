import numpy as np
from scipy.stats import chi2
from scipy.linalg import lstsq

def psi_huber(vec, c):
    psi = np.zeros(len(vec))
    mask = np.abs(vec) <= c
    psi[mask] = vec[mask]
    psi[~mask] = c * np.sign(vec[~mask])
    return psi


def linear_regression(X, y, c=0.7317, sigma0=1):
    n = X.shape[0]
    alpha = c**2 * (1 - chi2.cdf(c**2, df=1)) + chi2.cdf(c**2, df=3)

    theta0 = np.linalg.solve(X.T @ X, X.T @ y)

    while True:
        # update residual
        res = y - X @ theta0
        # update pesudo residual
        res_psi = sigma0 * psi_huber(res / sigma0, c)
        # update scale
        sigma = 1 / np.sqrt(n*2 * alpha) * np.linalg.norm(res_psi)
        # (re)update the pseudo residual
        res_psi = sigma * psi_huber(res / sigma, c)
        # regress X on r_psi
        # delta = np.linalg.solve(X.T @ X, X.T @ res_psi)
        delta = lstsq(X, res_psi)[0]
        # update theta
        theta = theta0 + delta
        # stopping criteria
        if (np.linalg.norm(theta - theta0) / np.linalg.norm(theta0)) > 1e-6:
            sigma0 = sigma
            theta0 = theta
        else:
            break

    return theta
