import os
import numpy as np
from scipy.linalg import lstsq
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

import huber
import sever
import rrm
import rlvi

import utils


# For reproducibility
SEED = 0
random_gen = np.random.default_rng(seed=SEED)


# Properties for boxplots
box_plot_props = dict(
    widths=0.6,
    whis=[0, 100],
    meanprops={'marker': 'o', 'markersize': 8, 'markerfacecolor': 'red', 'markeredgewidth': 0},
    medianprops={'linestyle': '-', 'linewidth': 1, 'color': 'r'},
    whiskerprops={'linestyle': (0, (10, 10))},
    boxprops={'c': 'blue', 'lw': 0.8},
    showfliers=False, autorange=True, showmeans=True,
)
textbox_props = dict(boxstyle='round', facecolor='white')

# Use an open source Roboto font from Google (download first)
if os.path.isfile("../Roboto.ttf"):
    custom_font = fm.FontProperties(fname="../Roboto.ttf", size=22)
else:
    custom_font = fm.FontProperties(size=22)


# To save results
if not os.path.exists('plots'):
    os.makedirs('plots')


def generate_data_mean(size=100, eps=0.2):
    n2 = random_gen.binomial(n=size, p=eps)  # number of corrupted samples
    n1 = size - n2

    mean = np.array([200, 200])
    mean = mean.reshape((-1, 1))
    sigma = np.array([
        [400, 50],
        [50, 400]
    ])
    sigma_root = np.linalg.cholesky(sigma)
    sample1 = mean + sigma_root @ random_gen.normal(size=(2, n1))

    nu = 2.5
    C = 2 * (nu / (nu - 2)) * sigma
    C_root = np.linalg.cholesky(C)
    u = random_gen.chisquare(df=nu, size=n2) / nu
    u = u[None, :]
    y = C_root @ random_gen.normal(size=(2, n2))
    sample2 = mean + y / np.sqrt(u)

    sample = np.hstack([sample1, sample2])
    return sample.T


def generate_data_linear_regression(size=40, eps=0.2, nu=1.5):
    X = -5 + 10 * random_gen.random(size=(size, 10))
    n2 = random_gen.binomial(n=size, p=eps)  # number of corrupted samples
    n1 = size - n2

    theta = np.ones(X.shape[-1])
    sigma = 0.25
    y1 = X[:n1] @ theta + sigma * random_gen.normal(size=n1)

    C = 1
    u = random_gen.chisquare(df=nu, size=n2) / nu
    u = u[None, :]
    v = np.sqrt(C) * random_gen.normal(size=n2)
    y2 = X[n1:] @ theta + v / np.sqrt(u)

    y = np.vstack([y1.reshape((-1, 1)), y2.reshape((-1, 1))]).flatten()
    return X, y


def generate_data_logistic_regression(size=100, eps=0.05):
    n2 = random_gen.binomial(n=size, p=eps)  # number of corrupted samples
    n1 = size - n2
    theta = np.array([-1, 1, 1])

    mean1 = np.array([0.5, 0.5])
    mean1 = mean1.reshape((-1, 1))
    rho = 0.99
    sigma1 = np.array([
        [0.25, -0.25 * rho],
        [-0.25 * rho, 0.25]
    ])
    sigma_root1 = np.linalg.cholesky(sigma1)
    X1 = mean1 + sigma_root1 @ random_gen.normal(size=(2, n1))
    X1 = X1.T
    X1_aug = np.hstack([
        np.ones((X1.shape[0], 1)),
        X1,
    ])
    inverse_sigmoid = 1 + np.exp(-X1_aug @ theta)
    y1 = np.ones(X1.shape[0])
    y1[inverse_sigmoid > 2] = 0

    mean2 = np.array([0.5, 1.25])
    mean2 = mean2.reshape((-1, 1))
    sigma_root2 = np.array([
        [0.1, 0],
        [0, 0.1]
    ])
    X2 = mean2 + sigma_root2 @ random_gen.normal(size=(2, n2))
    X2 = X2.T
    y2 = np.zeros(X2.shape[0])

    X = np.vstack([X1, X2])
    y = np.vstack([y1.reshape((-1, 1)), y2.reshape((-1, 1))]).flatten()
    return X, y


def generate_data_pca(size=200, eps=0.2):
    n2 = random_gen.binomial(n=size, p=eps)  # number of corrupted samples
    n1 = size - n2

    z1 = random_gen.normal(size=n1)
    sigma = 0.25
    z2 = 2 * z1 + sigma * random_gen.normal(size=n1)
    z_true = np.column_stack([z1, z2])

    nu = 1.5
    u = random_gen.chisquare(df=nu, size=n2) / nu
    u = u[None, :]
    v = random_gen.normal(size=(2, n2))
    z_false = v / np.sqrt(u)
    z_false = z_false.T

    sample = np.vstack([z_true, z_false])
    return sample


def generate_data_cov(size=50, eps=0.2):
    n2 = random_gen.binomial(n=size, p=eps)  # number of corrupted samples
    n1 = size - n2

    true_mean = np.zeros((2, 1))
    true_sigma = np.array([
        [1, 0.8],
        [0.8, 1]
    ])
    true_sigma_root = np.linalg.cholesky(true_sigma)

    sample1 = true_mean + true_sigma_root @ random_gen.normal(size=(2, n1))

    nu = 1.5

    u = random_gen.chisquare(df=nu, size=n2) / nu
    u = u[None, :]
    y = true_sigma_root @ random_gen.normal(size=(2, n2))
    sample2 = true_mean + y / np.sqrt(u)

    sample = np.hstack([sample1, sample2])
    return sample.T


def test_mean():
    print("Running test for mean estimation")

    true_mean = np.array([200, 200])
    MC_runs = 100
    eps_values = np.linspace(0, 0.4, 20)
    ml_errors = [0 for _ in range(len(eps_values))]
    rrm_errors = [0 for _ in range(len(eps_values))]
    rlvi_errors = [0 for _ in range(len(eps_values))]

    for _ in range(MC_runs):
        for i, eps in enumerate(eps_values):
            samples = generate_data_mean(size=100, eps=eps)
            mean_ml = np.mean(samples, axis=0)
            mean_rrm = rrm.mean(samples, eps=0.4)
            mean_rlvi = rlvi.mean(samples)
            ml_errors[i] += np.linalg.norm(true_mean - mean_ml) / np.linalg.norm(true_mean) / MC_runs
            rrm_errors[i] += np.linalg.norm(true_mean - mean_rrm) / np.linalg.norm(true_mean) / MC_runs
            rlvi_errors[i] += np.linalg.norm(true_mean - mean_rlvi) / np.linalg.norm(true_mean) / MC_runs

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.plot(eps_values, ml_errors, lw=3, c='brown', label='ML')
    plt.plot(eps_values, rrm_errors, lw=3, c='green', label='RRM')
    plt.plot(eps_values, rlvi_errors, lw=3, c='blue', label='RLVI')
    plt.xlabel(r"corruption level $\varepsilon$", font_properties=custom_font)
    # plt.ylabel("$Average\ relative\ error$", fontsize=22)
    plt.ylabel("Mean\navg. relative error", font_properties=custom_font)
    plt.xticks(np.arange(0, 0.5, 0.1), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 0.4)
    # plt.ylim(0, 0.4)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/mean.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_mean_fixed_eps():
    print("Running test for mean estimation (fixed eps.)")

    true_mean = np.array([200, 200])
    MC_runs = 100
    
    ml_errors = []
    rrm_errors = []
    rlvi_errors = []

    for _ in range(MC_runs):
        samples = generate_data_mean(size=100, eps=0.2)
        mean_ml = np.mean(samples, axis=0)
        mean_rrm = rrm.mean(samples, eps=0.4)
        mean_rlvi = rlvi.mean(samples)
        ml_errors.append(np.linalg.norm(true_mean - mean_ml) / np.linalg.norm(true_mean))
        rrm_errors.append(np.linalg.norm(true_mean - mean_rrm) / np.linalg.norm(true_mean))
        rlvi_errors.append(np.linalg.norm(true_mean - mean_rlvi) / np.linalg.norm(true_mean))

    errors_to_plot = {
        "ML": ml_errors,
        "RRM": rrm_errors,
        "RLVI": rlvi_errors
    }

    plt.rcParams["mathtext.fontset"] = "cm"
    for i, errors in enumerate(errors_to_plot.values()):
        plt.boxplot(errors, positions=[i + 1], **box_plot_props)
    plt.xticks(range(1, len(errors_to_plot) + 1), errors_to_plot.keys(), font_properties=custom_font)
    plt.ylabel("Mean\nrelative error", font_properties=custom_font)
    plt.ylim(0, 0.12)
    plt.yticks(np.arange(0, 0.20, 0.05), fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/mean_boxplots.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_linear_regression():
    print("Running test for linear regression")

    true_theta = np.ones(10)
    eps_values = np.linspace(0, 0.4, 20)

    ml_loss = []
    huber_loss = []
    sever_loss = []
    rrm_loss = []
    rlvi_loss = []

    MC_runs = 100
    for eps in eps_values:
        ml_loss_avg = 0
        huber_loss_avg = 0
        sever_loss_avg = 0
        rrm_loss_avg = 0
        rlvi_loss_avg = 0
        for _ in range(MC_runs):
            X, y = generate_data_linear_regression(eps=eps, nu=2.5)
            theta_ml = lstsq(X, y)[0]
            theta_huber = huber.linear_regression(X, y)
            theta_sever = sever.linear_regression(X, y, eps=0.4)
            theta_rrm = rrm.linear_regression(X, y, eps=0.4)
            theta_rlvi = rlvi.linear_regression(X, y)

            ml_loss_avg += np.linalg.norm(true_theta - theta_ml)
            huber_loss_avg += np.linalg.norm(true_theta - theta_huber)
            sever_loss_avg += np.linalg.norm(true_theta - theta_sever)
            rrm_loss_avg += np.linalg.norm(true_theta - theta_rrm)
            rlvi_loss_avg += np.linalg.norm(true_theta - theta_rlvi)

        ml_loss.append(ml_loss_avg / np.linalg.norm(true_theta) / MC_runs)
        huber_loss.append(huber_loss_avg / np.linalg.norm(true_theta) / MC_runs)
        sever_loss.append(sever_loss_avg / np.linalg.norm(true_theta) / MC_runs)
        rrm_loss.append(rrm_loss_avg / np.linalg.norm(true_theta) / MC_runs)
        rlvi_loss.append(rlvi_loss_avg / np.linalg.norm(true_theta) / MC_runs)

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.plot(eps_values, ml_loss, lw=3, c='brown', label='ML')
    plt.plot(eps_values, huber_loss, lw=3, c='chocolate', label='Huber')
    plt.plot(eps_values, sever_loss, lw=3, c='orange', label='SEVER')
    plt.plot(eps_values, rrm_loss, lw=3, c='teal', label='RRM')
    plt.plot(eps_values, rlvi_loss, lw=3, c='#4477aa', label='RLVI')

    plt.xlabel(r"corruption level, $\varepsilon$", font_properties=custom_font)#, labelpad=10)
    plt.ylabel("LinReg\navg. relative error", font_properties=custom_font)
    plt.xticks(np.arange(0, 0.5, 0.1), fontsize=20)
    plt.xlim(0, 0.4)
    plt.yticks(np.arange(0.010, 0.08, 0.01), fontsize=20)
    plt.ylim(0.015, 0.08)
    plt.legend(fontsize=18, framealpha=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/linear_regression.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_linear_regression_fixed_eps():
    print("Running test for linear regression (fixed eps.)")

    true_theta = np.ones(10)

    ml_errors = []
    huber_errors = []
    sever_errors = []
    rrm_errors = []
    rlvi_errors = []

    MC_runs = 100
    for _ in range(MC_runs):
        X, y = generate_data_linear_regression(eps=0.2, nu=2.5)
        theta_ml = np.linalg.solve(X.T @ X, X.T @ y)
        theta_huber = huber.linear_regression(X, y)
        theta_sever = sever.linear_regression(X, y, eps=0.4)
        theta_rrm = rrm.linear_regression(X, y, eps=0.4)
        theta_rlvi = rlvi.linear_regression(X, y)
        
        ml_errors.append(np.linalg.norm(true_theta - theta_ml) / np.linalg.norm(true_theta))
        huber_errors.append(np.linalg.norm(true_theta - theta_huber) / np.linalg.norm(true_theta))
        sever_errors.append(np.linalg.norm(true_theta - theta_sever) / np.linalg.norm(true_theta))
        rrm_errors.append(np.linalg.norm(true_theta - theta_rrm) / np.linalg.norm(true_theta))
        rlvi_errors.append(np.linalg.norm(true_theta - theta_rlvi) / np.linalg.norm(true_theta))

    errors_to_plot = {
        "ML": ml_errors,
        "Huber": huber_errors,
        "SEVER": sever_errors,
        "RRM": rrm_errors,
        "RLVI": rlvi_errors
    }
    fig, ax = plt.subplots()
    plt.rcParams["mathtext.fontset"] = "cm"
    for i, errors in enumerate(errors_to_plot.values()):
        plt.boxplot(errors, positions=[i + 1], **box_plot_props)
    plt.xticks(range(1, len(errors_to_plot) + 1), errors_to_plot.keys(), fontsize=20)
    plt.ylabel("LinReg\nrelative error", font_properties=custom_font)
    plt.ylim(0, 0.15)
    plt.yticks(np.arange(0, 0.16, 0.05), font_properties=custom_font)

    plt.text(0.95, 0.9, r"$\varepsilon=20\%$",
             transform=ax.transAxes, fontsize=22,
             ha='right', va='top', bbox=textbox_props)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/linear_regression_boxplots.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_logistic_regression():
    print("Running test for logistic regression")

    true_theta = np.array([-1, 1, 1])
    eps_values = np.linspace(0, 0.4, 20)

    ml_loss = []
    sever_loss = []
    rrm_loss = []
    rlvi_loss = []

    MC_runs = 100
    for eps in eps_values:
        ml_loss_avg = 0
        sever_loss_avg = 0
        rrm_loss_avg = 0
        rlvi_loss_avg = 0
        for _ in range(MC_runs):
            X, y = generate_data_logistic_regression(eps=eps)
            theta_ml, _ = utils.sklearn_log_reg(X, y)
            # theta_ml, _ = utils.mm_log_reg(X, y)
            theta_sever = sever.logistic_regression(X, y, eps=0.1)
            theta_rrm = rrm.logistic_regression(X, y, eps=0.4)
            theta_rlvi = rlvi.logistic_regression(X, y)

            ml_loss_avg += np.arccos(
                theta_ml @ true_theta / (np.linalg.norm(theta_ml) * np.linalg.norm(true_theta))
            ) * 180 / np.pi
            sever_loss_avg += np.arccos(
                theta_sever @ true_theta / (np.linalg.norm(theta_sever) * np.linalg.norm(true_theta))
            ) * 180 / np.pi
            rrm_loss_avg += np.arccos(
                theta_rrm @ true_theta / (np.linalg.norm(theta_rrm) * np.linalg.norm(true_theta))
            ) * 180 / np.pi
            rlvi_loss_avg += np.arccos(
                theta_rlvi @ true_theta / (np.linalg.norm(theta_rlvi) * np.linalg.norm(true_theta))
            ) * 180 / np.pi

        ml_loss.append(ml_loss_avg / MC_runs)
        sever_loss.append(sever_loss_avg / MC_runs)
        rrm_loss.append(rrm_loss_avg / MC_runs)
        rlvi_loss.append(rlvi_loss_avg / MC_runs)

    plt.rcParams["mathtext.fontset"] = "cm"
    # plt.plot(eps_values, ml_loss, lw=3, c='brown', label='ML')
    plt.plot(eps_values, sever_loss, lw=3, c='orange', label='SEVER')
    plt.plot(eps_values, rrm_loss, lw=3, c='green', label='RRM')
    plt.plot(eps_values, rlvi_loss, lw=3, c='blue', label='RLVI')
    plt.xlabel(r"corruption level $\varepsilon$", font_properties=custom_font)
    plt.ylabel("LogReg\navg. angle in degree", font_properties=custom_font)
    
    plt.xticks(np.arange(0, 0.5, 0.1), fontsize=20)
    plt.xlim(0, 0.4)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/logistic_regression.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_logistic_regression_fixed_eps():
    print("Running test for logistic regression (fixed eps.)")

    true_theta = np.array([-1, 1, 1])
    ml_angles = []
    sever_angles = []
    rrm_angles = []
    rlvi_angles = []
    MC_runs = 100
    for _ in range(MC_runs):
        X, y = generate_data_logistic_regression(size=200, eps=0.05)
        theta_ml, _ = utils.sklearn_log_reg(X, y, weights=np.ones(len(y)))
        # theta_ml, _ = utils.mm_log_reg(X, y, weights=np.ones(len(y)))
        theta_sever = sever.logistic_regression(X, y, eps=0.3)
        theta_rrm = rrm.logistic_regression(X, y, eps=0.3)
        theta_rlvi = rlvi.logistic_regression(X, y)
        ml_angles.append(
            np.arccos(theta_ml @ true_theta / (np.linalg.norm(theta_ml) * np.linalg.norm(true_theta))) * 180 / np.pi
        )
        sever_angles.append(
            np.arccos(theta_sever @ true_theta / (np.linalg.norm(theta_sever) * np.linalg.norm(true_theta))) * 180 / np.pi
        )
        rrm_angles.append(
            np.arccos(theta_rrm @ true_theta / (np.linalg.norm(theta_rrm) * np.linalg.norm(true_theta))) * 180 / np.pi
        )
        rlvi_angles.append(
            np.arccos(theta_rlvi @ true_theta / (np.linalg.norm(theta_rlvi) * np.linalg.norm(true_theta))) * 180 / np.pi
        )

    errors_to_plot = {
        "ML": ml_angles,
        "SEVER": sever_angles,
        "RRM": rrm_angles,
        "RLVI": rlvi_angles
    }
    fig, ax = plt.subplots()
    plt.rcParams["mathtext.fontset"] = "cm"
    for i, errors in enumerate(errors_to_plot.values()):
        plt.boxplot(errors, positions=[i + 1], **box_plot_props)
    plt.xticks(range(1, len(errors_to_plot) + 1), errors_to_plot.keys(), font_properties=custom_font)
    plt.ylabel("LogReg\nangle in degree", font_properties=custom_font)
    plt.ylim(0, 30)
    plt.yticks(np.arange(0, 35, 10), fontsize=20)
    plt.text(0.95, 0.9, r"$\varepsilon=5\%$",
             transform=ax.transAxes, fontsize=22,
             ha='right', va='top', bbox=textbox_props)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/logistic_regression_boxplots.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_pca():
    print("Running test for PCA")

    true_theta = np.array([1 / np.sqrt(5), 2 / np.sqrt(5)])
    ml_errors = []
    rrm_errors = []
    sever_errors = []
    rlvi_errors = []
    MC_runs = 100

    # Original benchmark in Osama et al. (2020) used the initial guess for RRM – so do we.
    # For a fair comparison we use it for all methods: SEVER, RRM, and RLVI
    theta_init = np.array([0.1, 1])
    theta_init /= np.linalg.norm(theta_init)
    for _ in range(MC_runs):
        sample = generate_data_pca(size=40, eps=0.2)
        theta_ml, _ = utils.pca(sample, weights=np.ones(len(sample)))
        theta_rrm = rrm.pca(sample, eps=0.4, theta_init=theta_init)
        theta_sever = sever.pca(sample, eps=0.4, theta_init=theta_init)
        theta_rlvi = rlvi.pca(sample, theta_init=theta_init)

        ml_errors.append(1 - np.abs(theta_ml @ true_theta))
        rrm_errors.append(1 - np.abs(theta_rrm @ true_theta))
        sever_errors.append(1 - np.abs(theta_sever @ true_theta))
        rlvi_errors.append(1 - np.abs(theta_rlvi @ true_theta))

    errors_to_plot = {
        "ML": ml_errors,
        "SEVER": sever_errors,
        "RRM": rrm_errors,
        "RLVI": rlvi_errors
    }

    fig, ax = plt.subplots()
    plt.rcParams["mathtext.fontset"] = "cm"
    for i, errors in enumerate(errors_to_plot.values()):
        plt.boxplot(errors, positions=[i + 1], **box_plot_props)
    plt.xticks(range(1, len(errors_to_plot) + 1), errors_to_plot.keys(), font_properties=custom_font)
    plt.ylabel("PCA\nmisalignment error", font_properties=custom_font)
    plt.ylim(0, 0.061)
    plt.yticks(np.arange(0, 0.061, 0.02), fontsize=20)
    plt.text(0.95, 0.9, r"$\varepsilon=20\%$",
             transform=ax.transAxes, fontsize=22,
             ha='right', va='top', bbox=textbox_props)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/pca_boxplots.png", dpi=300)
    plt.close()
    print("\tDone\n")


def test_covariance():
    print("Running test for covariance estimation")

    true_cov = np.array([
        [1, 0.8],
        [0.8, 1]
    ])

    ml_errors = []
    rrm_errors = []
    sever_errors = []
    rlvi_errors = []
    MC_runs = 100
    for _ in range(MC_runs):
        samples = generate_data_cov(size=50, eps=0.2)
        cov_ml = np.cov(samples, rowvar=False)
        cov_rrm = rrm.covariance(samples, eps=0.4)
        cov_sever = sever.covariance(samples, eps=0.4, numiter=3)
        cov_rlvi = rlvi.covariance(samples, eps=0.4)  # use constraint due to unbounded likelihood
        ml_errors.append(np.linalg.norm(true_cov - cov_ml, ord='fro') / np.linalg.norm(true_cov, ord='fro'))
        rrm_errors.append(np.linalg.norm(true_cov - cov_rrm, ord='fro') / np.linalg.norm(true_cov, ord='fro'))
        sever_errors.append(np.linalg.norm(true_cov - cov_sever, ord='fro') / np.linalg.norm(true_cov, ord='fro'))
        rlvi_errors.append(np.linalg.norm(true_cov - cov_rlvi, ord='fro') / np.linalg.norm(true_cov, ord='fro'))

    errors_to_plot = {
        "ML": ml_errors,
        "SEVER": sever_errors,
        "RRM": rrm_errors,
        "RLVI": rlvi_errors
    }

    fig, ax = plt.subplots()
    plt.rcParams["mathtext.fontset"] = "cm"
    for i, errors in enumerate(errors_to_plot.values()):
        plt.boxplot(errors, positions=[i + 1], **box_plot_props)
    plt.xticks(range(1, len(errors_to_plot) + 1), errors_to_plot.keys(), font_properties=custom_font)
    plt.ylabel("Covariance\nrelative error", font_properties=custom_font)
    plt.ylim(0, 3)
    plt.yticks(list(range(4)), fontsize=20)
    plt.text(0.95, 0.9, r"$\varepsilon=20\%$",
             transform=ax.transAxes, fontsize=22,
             ha='right', va='top', bbox=textbox_props)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/cov_boxplots.png", dpi=300)
    plt.close()
    print("\tDone\n")


def main():
    test_mean()
    test_linear_regression()
    test_linear_regression_fixed_eps()
    test_logistic_regression_fixed_eps()
    test_pca()
    test_covariance()


if __name__ == "__main__":
    main()
