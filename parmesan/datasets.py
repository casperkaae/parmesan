import numpy as np
import pickle as pkl
import gzip
import tarfile
import fnmatch
import os
import urllib
from scipy.io import loadmat

def _unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def _download_frey_faces(dataset):
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    origin = (
        'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset+'.mat')
    matdata = loadmat(dataset)
    f = gzip.open(dataset +'.pkl.gz', 'w')
    pkl.dump([matdata['ff'].T],f)



def _download_mnist_realval(dataset):
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    origin = (
        'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset)


def _download_mnist_binarized(datapath):
    """
    Download the fized binzarized MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    datafiles = {
        "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
        "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
        "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat"
    }
    datasplits = {}
    for split in datafiles.keys():
        print "Downloading %s data..." %(split)
        local_file = datapath + '/binarized_mnist_%s.npy'%(split)
        datasplits[split] = np.loadtxt(urllib.urlretrieve(datafiles[split])[0])

    f = gzip.open(datapath +'/mnist.pkl.gz', 'w')
    pkl.dump([datasplits['train'],datasplits['valid'],datasplits['test']],f)


def _get_datafolder_path():
    full_path = os.path.abspath('.')
    path = full_path +'/data'
    return path


def load_mnist_realval(dataset=_get_datafolder_path()+'/mnist_real/mnist.pkl.gz'):
    '''
    Loads the real valued MNIST dataset
    :param dataset: path to dataset file
    :return: None
    '''
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_realval(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f)
    f.close()
    x_train, targets_train = train_set[0], train_set[1]
    x_valid, targets_valid = valid_set[0], valid_set[1]
    x_test, targets_test = test_set[0], test_set[1]
    return x_train, targets_train, x_valid, targets_valid, x_test, targets_test


def load_mnist_binarized(dataset=_get_datafolder_path()+'/mnist_binarized/mnist.pkl.gz'):
    '''
    Loads the fixed binarized MNIST dataset provided by Hugo Larochelle.
    :param dataset: path to dataset file
    :return: None
    '''
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_binarized(datasetfolder)

    f = gzip.open(dataset, 'rb')
    x_train, x_valid, x_test = pkl.load(f)
    f.close()
    return x_train, x_valid, x_test


def cifar10(datasets_dir=_get_datafolder_path(), num_val=5000):
    raise Warning('cifar10 loader is untested!')
    # this code is largely cp from Kyle Kastner:
    #
    # https://gist.github.com/kastnerkyle/f3f67424adda343fef40
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
        org_dir = os.getcwd()
        with tarfile.open(data_file) as tar:
            os.chdir(datasets_dir)
            tar.extractall()
        os.chdir(org_dir)

    train_files = []
    for filepath in fnmatch.filter(os.listdir(data_dir), 'data*'):
        train_files.append(os.path.join(data_dir, filepath))
    train_files = sorted(train_files, key=lambda x: x.split("_")[-1])

    test_file = os.path.join(data_dir, 'test_batch')

    x_train, targets_train = [], []
    for f in train_files:
        d = _unpickle(f)
        x_train.append(d['data'])
        targets_train.append(d['labels'])
    x_train = np.array(x_train, dtype='uint8')
    shp = x_train.shape
    x_train = x_train.reshape(shp[0] * shp[1], 3, 32, 32)
    targets_train = np.array(targets_train)
    targets_train = targets_train.ravel()

    d = _unpickle(test_file)
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



def load_frey_faces(dataset=_get_datafolder_path()+'/frey_faces/frey_faces', normalize=True):
    '''
    Loads the frey faces dataset
    :param dataset: path to dataset file
    '''
    if not os.path.isfile(dataset + '.pkl.gz'):
        datasetfolder = os.path.dirname(dataset+'.pkl.gz')
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_frey_faces(dataset)

    f = gzip.open(dataset+'.pkl.gz', 'rb')
    data = pkl.load(f)[0].astype('float32')
    f.close()
    if normalize:
        data = (data - np.min(data)) / (np.max(data)-np.min(data))
    return data
