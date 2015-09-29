import theano
import numpy as np
import pickle as pkl
import gzip
import tarfile
import fnmatch
import os
import cPickle


def load_mnist_realval():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pkl.load(f)
    f.close()
    x_train, targets_train = train_set[0], train_set[1]
    x_valid, targets_valid = valid_set[0], valid_set[1]
    x_test, targets_test = test_set[0], test_set[1]
    return x_train, targets_train, x_valid, targets_valid, x_test, targets_test


def cifar10(datasets_dir='data', num_val=5000):
    # this code is largely cp from Kyle Kastner
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_file = os.path.join(datasets_dir, 'cifar-10-python.tar.gz')
    data_dir = os.path.join(datasets_dir, 'cifar-10-batches-py')

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    if not os.path.isfile(data_file):
        urllib.urlretrieve(url, data_file)
        with tarfile.open(data_file) as tar:
            os.chdir(datasets_dir)
            tar.extractall()

    train_files = []
    for filepath in fnmatch.filter(os.listdir(data_dir), 'data*'):
        train_files.append(os.path.join(data_dir, filepath))

    test_file = os.path.join(data_dir, 'test_batch')


    name2label = {k: v for v, k in enumerate(
        unpickle(os.path.join(data_dir, 'batches.meta'))['label_names'])}
    label2name = {v: k for k, v in name2label.items()}

    train_files = sorted(train_files, key=lambda x: x.split("_")[-1])
    x_train = []
    targets_train = []
    for f in train_files:
        d = unpickle(f)
        x_train.append(d['data'])
        targets_train.append(d['labels'])
    x_train = np.array(x_train, dtype='uint8')
    shp = x_train.shape
    x_train = x_train.reshape(shp[0] * shp[1], 3, 32, 32)
    targets_train = np.array(targets_train)
    targets_train = targets_train.ravel()

    d = unpickle(test_file)
    x_test = d['data']
    targets_test = d['labels']
    x_test = np.array(x_test, dtype='uint8')
    x_test = x_test.reshape(-1, 3, 32, 32)
    targets_test = np.array(targets_test)
    targets_test = targets_test.ravel()

    if num_val is not None:
        perm = np.random.permutation(x_train.shape[0])
        x = x_train[perm]
        y = targets_train[perm]

        x_valid = x[:num_val]
        targets_valid = y[:num_val]
        x_train = x[num_val:]
        targets_train = y[num_val:]
        return (x_train, targets_train,
                x_valid, targets_valid,
                x_test, targets_test)
    else:
        return x_train, targets_train, x_test, targets_test


def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d