import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import data_tools as tools


class Mnist(Data.Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, 
                 train=True, transform=None, target_transform=None, download=False,
                 dataset='mnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
        self.root = os.path.join(os.path.expanduser(root), 'MNIST')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.train_data, self.train_labels = torch.load(
            os.path.join(self.root, self.processed_folder, self.training_file))
        self.train_labels = np.array(self.train_labels)

        # clean images and noisy labels (training and validation)
        if noise_rate > 0:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split(
                self.train_data, self.train_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class
            )
        else:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split_without_noise(
                self.train_data, self.train_labels, split_per, random_seed
            )

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img.numpy())
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index

    def __len__(self):            
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            tools.read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            tools.read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            tools.read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            tools.read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
 

class MnistTest(Data.Dataset):
    processed_folder = 'processed'
    test_file = 'test.pt'

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.join(os.path.expanduser(root), 'MNIST')
        self.transform = transform
        self.target_transform = target_transform

        self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        self.test_labels = np.array(self.test_labels)

        
    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.numpy())
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)
  
    
class Cifar10(Data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 dataset='cifar10', noise_type='symmetric', noise_rate=0.5, 
                 split_per=0.9, random_seed=1, num_class=10
                ):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        self.train_data = np.concatenate(self.train_data)
        self.train_labels = np.array(self.train_labels)
        # self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        # clean images and noisy labels (training and validation)
        if noise_rate > 0:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split(
                self.train_data, self.train_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class
            )
        else:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split_without_noise(
                self.train_data, self.train_labels, split_per, random_seed
            )

        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))
        
    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not tools.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        tools.download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

        
class Cifar10Test(Data.Dataset):
    base_folder = 'cifar-10-batches-py'
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform

        f = self.test_list[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        self.test_data = entry['data']
        if 'labels' in entry:
            self.test_labels = entry['labels']
        else:
            self.test_labels = entry['fine_labels']
        fo.close()
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))
        self.test_labels = np.array(self.test_labels)

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)

    
class Cifar100(Data.Dataset):

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 dataset='cifar100', noise_type='symmetric', noise_rate=0.5, 
                 split_per=0.9, random_seed=1, num_class=100
                ):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        self.train_data = np.concatenate(self.train_data)
        self.train_labels = np.array(self.train_labels)

        # clean images and noisy labels (training and validation)
        if noise_rate > 0:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split(
                self.train_data, self.train_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class
            )
        else:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split_without_noise(
                self.train_data, self.train_labels, split_per, random_seed
            )

        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]   
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not tools.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        tools.download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
        
        
class Cifar100Test(Data.Dataset):

    base_folder = 'cifar-100-python'
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform

        f = self.test_list[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        self.test_data = entry['data']
        if 'labels' in entry:
            self.test_labels = entry['labels']
        else:
            self.test_labels = entry['fine_labels']
        fo.close()
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))
        self.test_labels = np.array(self.test_labels)

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)
