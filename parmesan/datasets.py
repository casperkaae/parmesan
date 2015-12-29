import numpy as np
import pickle as pkl
import cPickle as cPkl

import gzip
import tarfile
import fnmatch
import os
import urllib
from scipy.io import loadmat
from sklearn.datasets import fetch_lfw_people

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer as EnglishStemmer
from nltk.tokenize import wordpunct_tokenize as wordpunct_tokenize
import re
import string


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

def _download_lwf(dataset,size):
    '''
    :param dataset:
    :return:
    '''
    lfw_people = fetch_lfw_people(color=True,resize=size)
    f = gzip.open(dataset, 'w')
    cPkl.dump([lfw_people.images.astype('uint8'),lfw_people.target],f,protocol=cPkl.HIGHEST_PROTOCOL)
    f.close()


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



def load_frey_faces(dataset=_get_datafolder_path()+'/frey_faces/frey_faces', normalize=True, dequantify=True):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013 "RNADE: The real-valued neural autoregressive density-estimator"
    :return:
    '''
    datasetfolder = os.path.dirname(dataset+'.pkl.gz')
    if not os.path.isfile(dataset + '.pkl.gz'):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_frey_faces(dataset)

    if not os.path.isfile(datasetfolder + '/fixed_split.pkl'):
        urllib.urlretrieve('https://raw.githubusercontent.com/casperkaae/'
                           'extra_parmesan/master/data_splits/'
                           'frey_faces_fixed_split.pkl',
                           datasetfolder + '/fixed_split.pkl')

    f = gzip.open(dataset+'.pkl.gz', 'rb')
    data = pkl.load(f)[0].reshape(-1,28,20).astype('float32')
    f.close()
    if dequantify:
        data = data + np.random.uniform(0,1,size=data.shape).astype('float32')
    if normalize:
        data = data / 256.
    return data

def load_lfw(dataset=_get_datafolder_path()+'/lfw/lfw', normalize=True, dequantify=True, size=0.25):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013 "RNADE: The real-valued neural autoregressive density-estimator"
    :param size: rescaling factor
    :return:
    '''

    dataset="%s_%0.2f.cpkl"%(dataset,size)
    datasetfolder = os.path.dirname(dataset)
    if not os.path.isfile(dataset):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_lwf(dataset,size)

    if not os.path.isfile(datasetfolder + '/fixed_split.pkl'):
        urllib.urlretrieve('https://raw.githubusercontent.com/casperkaae/'
                           'extra_parmesan/master/data_splits/'
                           'lfw_fixed_split.pkl',
                           datasetfolder + '/fixed_split.pkl')


    f = gzip.open(dataset, 'rb')
    data = cPkl.load(f)[0].astype('float32')
    f.close()
    if dequantify:
        data = data + np.random.uniform(0,1,size=data.shape).astype('float32')
    if normalize:
        data = data / 256.
    return data


def load_svhn(dataset=_get_datafolder_path()+'/svhn/', normalize=True, dequantify=True, extra=False):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013 "RNADE: The real-valued neural autoregressive density-estimator"
    :param extra: include extra svhn samples
    :return:
    '''
    if not os.path.isfile(dataset +'svhn_train.cpkl'):
        datasetfolder = os.path.dirname(dataset +'svhn_train.cpkl')
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_svhn(dataset)

    with open(dataset +'svhn_train.cpkl', 'rb') as f:
        train_x,train_y = cPkl.load(f)
    with open(dataset +'svhn_test.cpkl', 'rb') as f:
        test_x,test_y = cPkl.load(f)

    if extra:
        with open(dataset +'svhn_extra.cpkl', 'rb') as f:
            extra_x,extra_y = cPkl.load(f)
        train_x = np.concatenate([train_x,extra_x])
        train_y = np.concatenate([train_y,extra_y])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_y = train_y.astype('int32')
    test_y = test_y.astype('int32')

    if dequantify:
        train_x = train_x + np.random.uniform(0,1,size=train_x.shape).astype('float32')
        test_x = test_x + np.random.uniform(0,1,size=test_x.shape).astype('float32')

    if normalize:
        train_x = train_x / 256.
        test_x = test_x / 256.

    return train_x, train_y, test_x, test_y



def _download_svhn(dataset):
    """
    """
    print 'Downloading data from http://ufldl.stanford.edu/housenumbers/, this may take a while...'
    print "Downloading train data..."
    urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', dataset+'train_32x32.mat')
    print "Downloading test data..."
    urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', dataset+'test_32x32.mat')
    print "Downloading extra data..."
    urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat', dataset+'extra_32x32.mat')

    train = loadmat(dataset+'train_32x32.mat')
    train_x = train['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
    train_y = train['y'].reshape((-1)) - 1
    test = loadmat(dataset+'test_32x32.mat')
    test_x = test['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
    test_y = test['y'].reshape((-1)) - 1
    extra = loadmat(dataset+'extra_32x32.mat')
    extra_x = extra['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
    extra_y = extra['y'].reshape((-1)) - 1
    print "Saving train data"
    with open(dataset +'svhn_train.cpkl', 'w') as f:
        cPkl.dump([train_x,train_y],f,protocol=cPkl.HIGHEST_PROTOCOL)
    print "Saving test data"
    with open(dataset +'svhn_test.cpkl', 'w') as f:
        pkl.dump([test_x,test_y],f,protocol=cPkl.HIGHEST_PROTOCOL)
    print "Saving extra data"
    with open(dataset +'svhn_extra.cpkl', 'w') as f:
        pkl.dump([extra_x,extra_y],f,protocol=cPkl.HIGHEST_PROTOCOL)
    os.remove(dataset+'train_32x32.mat'),os.remove(dataset+'test_32x32.mat'),os.remove(dataset+'extra_32x32.mat')


def load_20newsgroup(dataset=_get_datafolder_path()+'/20newsgroup/',max_features=1000,normalize_by_doc_len=True):
    '''
    Loads 20 newsgroup dataset
    :param dataset: path to dataset file
    :return: None
    '''

    if not os.path.isfile(dataset + '20newsgroup.pkl'):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)

    if not os.path.isfile(dataset + '20newsgroup_mf'+ str(max_features) + '.pkl'):
        train_set,test_set = _download_20newsgroup()
        train_set, test_set, vocab_train = _bow(train_set, test_set, max_features=max_features)
        with open(dataset + '20newsgroup_mf'+ str(max_features) + '.pkl','w') as f:
            pkl.dump([train_set, test_set, vocab_train],f)

    with open(dataset + '20newsgroup_mf'+ str(max_features) + '.pkl','r') as f:
        train_set, test_set, vocab_train = pkl.load(f)

    x_train, y_train = train_set
    x_test, y_test = test_set

    if normalize_by_doc_len:
        x_train = x_train / (x_train + 1e-8).sum(keepdims=True, axis=1)
        x_test = x_test / (x_test + 1e-8).sum(keepdims=True, axis=1)

    return x_train, y_train, x_test, y_test

def _download_20newsgroup():
    """
    Download the 20 newsgroups dataset from scikitlearn.
    :return: The train, test and validation set.
    """
    print "downloading 20 newsgroup train data...."
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    print "downloading 20 newsgroup test data...."
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    train_set = (newsgroups_train.data, newsgroups_train.target)
    test_set = (newsgroups_test.data, newsgroups_test.target)

    return train_set,test_set

def _bow(train, test, max_features=1000):
    x_train, y_train = train
    x_test, y_test = test


    stemmer = EnglishStemmer()
    lemmatizer = WordNetLemmatizer()
    for i in range(len(x_train)):
        x_train[i] = " ".join([lemmatizer.lemmatize(stemmer.stem(token.lower())) for token in wordpunct_tokenize(
            re.sub('[%s]' % re.escape(string.punctuation), '', x_train[i]))])

    vectorizer_train = CountVectorizer(strip_accents='ascii', stop_words='english',
                                 token_pattern=r"(?u)\b\w[a-z]\w+[a-z]\b", max_features=max_features,
                                 vocabulary=None, dtype='float32')
    x_train = vectorizer_train.fit_transform(x_train).toarray()


    vocab_train = vectorizer_train.get_feature_names()
    vectorizer_test = CountVectorizer(strip_accents='ascii', stop_words='english',
                                 token_pattern=r"(?u)\b\w[a-z]\w+[a-z]\b", max_features=max_features,
                                 vocabulary=vocab_train, dtype='float32')
    x_test = vectorizer_test.fit_transform(x_test).toarray()

    # remove documents with no words
    r = np.where(x_train.sum(axis=1) > 0.)[0]
    x_train = x_train[r, :]
    y_train = y_train[r]

    r = np.where(x_test.sum(axis=1) > 0.)[0]
    x_test = x_test[r, :]
    y_test = y_test[r]

    return (x_train, y_train),(x_test, y_test), vocab_train