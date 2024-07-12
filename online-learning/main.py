import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

from scipy.optimize import minimize_scalar
from scipy.io import loadmat

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing


random_gen = np.random.default_rng(seed=1)

partial_fit_classifiers = {
    'SGD': SGDClassifier(loss='log_loss', random_state=1),
    'RRM': SGDClassifier(loss='log_loss', random_state=1),
    'RLVI': SGDClassifier(loss='log_loss', random_state=1),
}

CLF_COLORS = {
    'SGD': '#b85460',
    'RRM': '#e69e00',
    'RLVI': '#4477aa',
}

CLF_ALPHA = {
    'SGD': 1.,
    'RRM': 1.,
    'RLVI': 1.,
}

# Use an open source Roboto font from Google (download first)
if os.path.isfile("../Roboto.ttf"):
    custom_font = fm.FontProperties(fname="../Roboto.ttf", size=22)
else:
    custom_font = fm.FontProperties(size=22)

# To save results
if not os.path.exists('plots'):
    os.makedirs('plots')


def update_weights_rlvi(losses, tol=1e-3, maxiter=100):
    '''Optimize Bernoulli probabilities'''
    exp_loss = np.exp(-losses)
    weights = 0.5 * np.ones_like(losses)
    for _ in range(maxiter):
        avg_weight = np.mean(weights)
        ratio = avg_weight / (1 - avg_weight)
        new_weights = ratio * exp_loss / (1 + ratio * exp_loss)
        error = np.linalg.norm(new_weights - weights)
        if error < tol:
            break
        weights = np.copy(new_weights)
    new_weights /= np.max(new_weights) * len(new_weights)
    return new_weights


def update_weights_rrm(residuals, eps):
    '''Optimize sample weights for Robust Risk Minimization'''
    res = np.copy(residuals)
    t = -np.log((1 - eps) * res.shape[0])
    numeric_cutoff = 1e-16

    def objective(xi):
        phi = np.exp(-res * np.exp(-xi))
        phi[phi < numeric_cutoff] = numeric_cutoff
        sum_phi = np.sum(phi)
        return np.exp(xi) * (np.log(sum_phi) + t)

    opt_res = minimize_scalar(objective)
    opt_xi = opt_res['x']
    opt_alpha = np.exp(opt_xi)
    phi = np.exp(-res / opt_alpha)
    phi[phi < numeric_cutoff] = numeric_cutoff
    sum_phi = np.sum(phi)
    opt_beta_over_alpha = np.log(sum_phi) - 1
    opt_weights = np.exp(-res / opt_alpha) * np.exp(-opt_beta_over_alpha - 1)
    return opt_weights


def cross_entropy(log_proba, targets):
    return -targets * log_proba - (1 - targets) * log_proba


def smooth_triangle(data, degree=1):
    '''Moving average filter (triangle scheme)'''
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
    smoothed = []
    for i in range(degree, len(data) - degree * 2):
        point = data[i : i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))
    # Boundaries
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def plot_score_evolution(clf_stats, smooth_deg=1):
    def plot_score(x, y, name):
        '''Plot accuracy vs number of samples'''
        x = np.array(x)
        y = np.array(y) * 100
        y = smooth_triangle(y, smooth_deg)
        plt.plot(x, y, lw=3, c=CLF_COLORS[name], alpha=CLF_ALPHA[name])

    clf_names = list(clf_stats.keys())
    plt.figure()
    for name, stats in clf_stats.items():
        # Plot accuracy evolution with #examples
        score, n_examples = zip(*stats['acc_history'])
        # Scale by thousands
        n_examples = np.array(n_examples) / 1000.
        plot_score(n_examples, score, name)
        plt.gca()

    plt.xlabel(r'observed samples $\times 10^3$', font_properties=custom_font)
    plt.ylabel(r'Accuracy, %', font_properties=custom_font)
    plt.xlim(1, 10.8)
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(80, 101, 5), fontsize=20)
    plt.ylim(85, 100)
    plt.grid(True)
    plt.legend(clf_names, loc='lower left', fontsize=19)
    plt.tight_layout()
    plt.savefig("plots/hard_score.png", dpi=300)
    plt.close()


def plot_tp_evolution(clf_stats, smooth_deg=1):
    def plot_recall(x, y, name):
        '''Plot recall for pos. class vs number of samples'''
        x = np.array(x)
        y = np.array(y)
        y = smooth_triangle(y, smooth_deg)
        plt.plot(x, y, lw=3, c=CLF_COLORS[name], alpha=CLF_ALPHA[name])

    clf_names = list(clf_stats.keys())
    plt.figure()
    for name, stats in clf_stats.items():
        # Plot recall evolution with #examples
        recall, n_examples = zip(*stats['tp_history'])
        # Scale by thousands
        n_examples = np.array(n_examples) / 1000.
        plot_recall(n_examples, recall, name)
        plt.gca()

    plt.xlabel(r'observed samples $\times 10^3$', font_properties=custom_font)
    plt.ylabel('Recall', font_properties=custom_font)
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0.7, 1.01, 0.1), fontsize=20)
    plt.ylim(0.75, 1)
    plt.xlim(1, 10.8)
    plt.grid(True)
    plt.legend(clf_names, loc='lower left', fontsize=19)
    plt.tight_layout()
    plt.savefig("plots/hard_tp.png", dpi=300)
    plt.close()


def plot_tn_evolution(clf_stats, smooth_deg=1):
    def plot_recall(x, y, name):
        '''Plot recall for neg. class vs number of samples'''
        x = np.array(x)
        y = np.array(y)
        y = smooth_triangle(y, smooth_deg)
        plt.plot(x, y, lw=3, c=CLF_COLORS[name], alpha=CLF_ALPHA[name])

    clf_names = list(clf_stats.keys())
    plt.figure()
    for name, stats in clf_stats.items():
        # Plot recall evolution with #examples
        recall, n_examples = zip(*stats['tn_history'])
        # Scale by thousands
        n_examples = np.array(n_examples) / 1000.
        plot_recall(n_examples, recall, name)
        plt.gca()

    plt.xlabel(r'observed samples $\times 10^3$', font_properties=custom_font)
    plt.ylabel('TN rate', font_properties=custom_font)
    plt.xlim(1, 10.8)
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0.8, 1.01, 0.05), fontsize=20)
    plt.ylim(0.8, 1)
    plt.grid(True)
    plt.legend(clf_names, loc='lower left', fontsize=19)
    plt.tight_layout()
    plt.savefig("plots/hard_tn.png", dpi=300)
    plt.close()


def plot_f1_evolution(clf_stats, smooth_deg=1):
    def plot_f1(x, y, name):
        '''Plot f1-score vs number of samples'''
        x = np.array(x)
        y = np.array(y)
        y = smooth_triangle(y, smooth_deg)
        plt.plot(x, y, lw=3, c=CLF_COLORS[name], alpha=CLF_ALPHA[name])

    clf_names = list(clf_stats.keys())
    plt.figure()
    for name, stats in clf_stats.items():
        # Plot recall evolution with #examples
        recall, n_examples = zip(*stats['f1_history'])
        # Scale by thousands
        n_examples = np.array(n_examples) / 1000.
        plot_f1(n_examples, recall, name)
        plt.gca()

    plt.xlabel(r'observed samples $\times 10^3$', font_properties=custom_font)
    plt.ylabel(r'$F_1{-}$score', font_properties=custom_font)
    plt.xlim(1, 10.8)
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0.8, 1.01, 0.05), fontsize=20)
    plt.ylim(0.8, 1)
    plt.grid(True)
    plt.legend(clf_names, loc='lower left', fontsize=19)
    plt.tight_layout()
    plt.savefig("plots/hard_f1.png", dpi=300)
    plt.close()


def read_hard_dataset(file):
    data = loadmat(file)
    X = data['feat']
    y = data['actid'].flatten()
    y = np.array(y > 2, dtype=int)
    return X, y


def pert(a, b, c, lmbda=4):
    r = c - a
    alpha = 1 + lmbda * (b - a) / r
    beta = 1 + lmbda * (c - b) / r
    return a + float(random_gen.beta(alpha, beta)) * r


def iter_minibatches(X, y, batch_size=100, test_size=0.2, noise_rate=0, dynamic=False):
    size = len(X)
    for idx in range(0, size, batch_size):
        left = idx
        right = min(idx + batch_size, size)
        X_batch = X[left:right]
        y_batch = y[left:right]
        # Split the batch into training and test part
        X_test = X_batch[:int(test_size * len(y_batch))]
        y_test = y_batch[:int(test_size * len(y_batch))]
        X_train = X_batch[int(test_size * len(y_batch)):]
        y_train = y_batch[int(test_size * len(y_batch)):]
        # Normalize features based on training data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # Corrupt training data
        if noise_rate > 0:
            y_positive = np.where(y_train == 1)[0]
            if dynamic:
                n_corrupted = int(pert(a=0, b=min(0.1, noise_rate), c=noise_rate) * len(y_positive))
            else:
                n_corrupted = int(noise_rate * len(y_positive))
            corrupted_labels = random_gen.choice(y_positive, size=n_corrupted, replace=False)
            y_train[corrupted_labels] = 0
        yield (X_train, y_train), (X_test, y_test)


def check_fitted(clf):
    return hasattr(clf, "classes_")


data_X, data_y = read_hard_dataset('./humanactivity.mat')
data_X, data_y = shuffle(data_X, data_y, random_state=1)


def main():
    clf_stats = {}
    for clf_name in partial_fit_classifiers:
        stats = {'n_train': 0, 'acc': 0.0, 'acc_history': [(0, 0)],
                 'tp': 0, 'tn': 0, 'tp_history': [(0, 0)], 'tn_history': [(0, 0)],
                 'f1': 0, 'f1_history': [(0, 0)],
                }
        clf_stats[clf_name] = stats

    minibatch_iterators = iter_minibatches(data_X, data_y, batch_size=200, test_size=0.5, noise_rate=0.3, dynamic=True)
    # Main loop : iterate on mini-batchs of examples
    for i, batch in enumerate(minibatch_iterators):
        (X_batch, y_batch), (X_test, y_test) = batch
        for clf_name, clf in partial_fit_classifiers.items():
            if clf_name == 'RLVI':
                if (not check_fitted(clf)):
                    log_proba = np.log(0.5 * np.ones_like(y_batch))
                else:
                    log_proba = clf.predict_log_proba(X_batch)[:, 1]
                residuals = cross_entropy(log_proba, y_batch)
                sample_weight = update_weights_rlvi(residuals)
                # update estimator with examples in the current mini-batch
                clf.partial_fit(X_batch, y_batch, classes=[0, 1], sample_weight=sample_weight)

            elif clf_name == 'RRM':
                if not check_fitted(clf):
                    log_proba = np.log(0.5 * np.ones_like(y_batch))
                else:
                    log_proba = clf.predict_log_proba(X_batch)[:, 1]
                residuals = cross_entropy(log_proba, y_batch)
                sample_weight = update_weights_rrm(residuals, eps=0.3)
                # update estimator with examples in the current mini-batch
                clf.partial_fit(X_batch, y_batch, classes=[0, 1], sample_weight=sample_weight)

            else:
                # update estimator with examples in the current mini-batch
                clf.partial_fit(X_batch, y_batch, classes=[0, 1])

            # accumulate test accuracy stats
            clf_stats[clf_name]['n_train'] += len(X_batch)

            y_pred = clf.predict(X_test)
            
            true_positive = np.count_nonzero(np.logical_and(y_pred == 1, y_test == 1))
            clf_stats[clf_name]['tp'] = true_positive / np.count_nonzero(y_test == 1)
            tp_history = (clf_stats[clf_name]['tp'], clf_stats[clf_name]['n_train'])
            clf_stats[clf_name]['tp_history'].append(tp_history)
            
            true_negative = np.count_nonzero(np.logical_and(y_pred == 0, y_test == 0))
            clf_stats[clf_name]['tn'] = true_negative / np.count_nonzero(y_test == 0)
            tn_history = (clf_stats[clf_name]['tn'], clf_stats[clf_name]['n_train'])
            clf_stats[clf_name]['tn_history'].append(tn_history)
            
            false_positive = np.count_nonzero(np.logical_and(y_pred == 1, y_test == 0))
            false_negative = np.count_nonzero(np.logical_and(y_pred == 0, y_test == 1))
            f1 = true_positive / (true_positive + (false_positive + false_negative) / 2.)
            clf_stats[clf_name]['f1'] = f1
            f1_history = (clf_stats[clf_name]['f1'], clf_stats[clf_name]['n_train'])
            clf_stats[clf_name]['f1_history'].append(f1_history)

            clf_stats[clf_name]['acc'] = clf.score(X_test, y_test)
            acc_history = (clf_stats[clf_name]['acc'], clf_stats[clf_name]['n_train'])
            clf_stats[clf_name]['acc_history'].append(acc_history)

    smooth_deg = 10
    plot_score_evolution(clf_stats, smooth_deg=smooth_deg)
    plot_tp_evolution(clf_stats, smooth_deg=smooth_deg)
    plot_tn_evolution(clf_stats, smooth_deg=smooth_deg)
    plot_f1_evolution(clf_stats, smooth_deg=smooth_deg)

if __name__ == '__main__':
    main()
