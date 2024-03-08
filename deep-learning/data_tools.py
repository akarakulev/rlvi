import os
import hashlib
import errno
import codecs

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats

import torch
import torch.nn.functional as F
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        parsed = np.array(parsed)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        parsed = np.array(parsed)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, actual_noise,P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, actual_noise,P

def noisify_trid(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the trid
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1], P[0, nb_classes-1] = 1. - n, n / 2, n /2
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1], P[i, i - 1]  = 1. - n, n / 2, n /2
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0], P[nb_classes-1, nb_classes-2]  = 1. - n, n / 2, n /2

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        y_train = y_train_noisy

    return y_train, actual_noise, P



def noisify_multiclass_asymmetric_mnist(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.eye(10)
    n = noise

    # 2 -> 7
    P[2, 2], P[2, 7] = 1. - n, n

    # 5 <-> 6
    P[5, 5], P[5, 6] = 1. - n, n
    P[6, 6], P[6, 5] = 1. - n, n

    # 3 -> 8
    P[3, 3], P[3, 8] = 1. - n, n

    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)

    y_train = y_train_noisy

    return y_train, actual_noise, P


def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def noisify_multiclass_asymmetric_cifar10(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    source_class = [9, 2, 3, 5, 4]
    target_class = [1, 0, 5, 3, 7]
    y_train_ = np.copy(np.array(y_train))
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(np.array(y_train) == s)[0]
        n_noisy = int(noise * cls_idx.shape[0])
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        for idx in noisy_sample_index:
            y_train_[idx] = t
    actual_noise = (y_train_ != np.array(y_train)).mean()
    print('Actual noise %.2f' % actual_noise)
    return y_train_, actual_noise, target_class

def noisify_multiclass_asymmetric_cifar100(y_train, noise, random_state=None, nb_classes=100):
    """mistakes:
        flip in the symmetric way
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

            y_train_noisy = multiclass_noisify(np.array(y_train), P=P, random_state=random_state)
            actual_noise = (y_train_noisy != np.array(y_train)).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        targets = y_train_noisy
    return targets, actual_noise, P


def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=1, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P


def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, 
                  dataset='mnist', noise_type='symmetric', noise_rate=0.5, 
                  split_per=0.9, random_seed=1, num_classes=10
                ):
    
    clean_train_labels = train_labels[:, np.newaxis]
    
    if noise_type == 'symmetric':
         noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels, 
                                                                                        noise=noise_rate, 
                                                                                        random_state=random_seed, 
                                                                                        nb_classes=num_classes)
    if noise_type == 'pairflip':
        noisy_labels, real_noise_rate, transition_matrix = noisify_pairflip(clean_train_labels,
                                                                            noise=noise_rate,
                                                                            random_state=random_seed,
                                                                            nb_classes=num_classes)
    if noise_type == 'asymmetric' and dataset == 'mnist':
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_asymmetric_mnist(clean_train_labels,
                                                                                            noise=noise_rate,
                                                                                            random_state=random_seed,
                                                                                            nb_classes=num_classes)
    
    if noise_type == 'asymmetric' and dataset == 'cifar10':
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_asymmetric_cifar10(clean_train_labels,
                                                                                                noise=noise_rate,
                                                                                                random_state=random_seed,
                                                                                                nb_classes=num_classes)
        
    if noise_type == 'asymmetric' and dataset == 'cifar100':
        noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_asymmetric_cifar100(clean_train_labels,
                                                                                                noise=noise_rate,
                                                                                                random_state=random_seed,
                                                                                                nb_classes=num_classes)

    if noise_type == 'instance' and dataset == 'mnist':
        noisy_labels = get_instance_noisy_label(noise_rate, 
                                                images=train_images,
                                                labels=train_labels,
                                                num_classes=10, 
                                                feature_size=784, 
                                                norm_std=0.1, 
                                                seed=random_seed)
    
    if noise_type == 'instance' and dataset == 'cifar10':
        noisy_labels = get_instance_noisy_label(noise_rate, 
                                                images=train_images,
                                                labels=train_labels,
                                                num_classes=10, 
                                                feature_size=3072, 
                                                norm_std=0.1, 
                                                seed=random_seed)
        
    if noise_type == 'instance' and dataset == 'cifar100':
        noisy_labels = get_instance_noisy_label(noise_rate, 
                                                images=train_images,
                                                labels=train_labels,
                                                num_classes=100, 
                                                feature_size=3072, 
                                                norm_std=0.1, 
                                                seedcle=random_seed)

    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    noise_mask = np.transpose(clean_train_labels[train_set_index]) == np.transpose(train_labels)

    return train_set, val_set, train_labels, val_labels, noise_mask


def dataset_split_without_noise(train_images, train_labels, split_per=0.9, random_seed=1):
    total_labels = train_labels[:, np.newaxis]
    num_samples = int(total_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = total_labels[train_set_index], total_labels[val_set_index]

    noise_mask = np.full(len(train_labels), True)

    return train_set, val_set, train_labels.squeeze(), val_labels.squeeze(), noise_mask


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


def get_instance_noisy_label(noise_rate, images, labels, num_classes, feature_size, norm_std, seed):
    """
    images: mnist, cifar10, cifar100
    label_num: class number
    feature_size: the size of input images (e.g. 28*28)
    norm_std: default 0.1
    """

    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate, scale=norm_std)
    # flip_distribution = stats.beta(a=0.01, b=(0.01 / n) - 0.01, loc=0, scale=1)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    images = torch.Tensor(images).to(torch.float)
    if not torch.is_tensor(labels):
        labels = torch.FloatTensor(labels)
    labels = labels.to(DEVICE)

    W = np.random.randn(label_num, feature_size, label_num)
    W = torch.FloatTensor(W).to(DEVICE)

    for i, (x, y) in enumerate(zip(images, labels)):
        x = x.to(DEVICE)
        y = int(y)
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -np.inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    return np.array(new_label)
