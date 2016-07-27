import numpy as np
import pickle as pkl
import cPickle as cPkl
import gzip, zipfile, tarfile
import os, shutil, re, string, urllib, fnmatch

def _get_datafolder_path():
    full_path = os.path.abspath('.')
    path = full_path +'/data'
    return path

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
    from scipy.io import loadmat
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

def _download_omniglot_iwae(dataset):
    """
    Download the MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    origin = (
        'https://github.com/yburda/iwae/raw/'
        'master/datasets/OMNIGLOT/chardata.mat'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset + '/chardata.mat')


def _download_norb_small(dataset):
    """
    Download the Norb dataset
    """
    from scipy.io import loadmat
    print 'Downloading small resized norb data'

    urllib.urlretrieve('http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-'
                       '5x46789x9x18x6x2x32x32-training-dat-matlab-bicubic.mat',
                       dataset + '/smallnorb_train_x.mat')
    urllib.urlretrieve('http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-'
                       '5x46789x9x18x6x2x96x96-training-cat-matlab.mat',
                       dataset + '/smallnorb_train_t.mat')

    urllib.urlretrieve('http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-'
                       '5x01235x9x18x6x2x32x32-testing-dat-matlab-bicubic.mat',
                       dataset + '/smallnorb_test_x.mat')
    urllib.urlretrieve('http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-'
                       '5x01235x9x18x6x2x96x96-testing-cat-matlab.mat',
                       dataset + '/smallnorb_test_t.mat')

    data = loadmat(dataset + '/smallnorb_train_x.mat')['traindata']
    train_x = np.concatenate([data[:,0,:].T, data[:,0,:].T]).astype('float32')
    data = loadmat(dataset + '/smallnorb_train_t.mat')
    train_t = data['trainlabels'].flatten().astype('float32')
    train_t = np.concatenate([train_t, train_t])

    data = loadmat(dataset + '/smallnorb_test_x.mat')['testdata']
    test_x = np.concatenate([data[:,0,:].T, data[:,0,:].T]).astype('float32')
    data = loadmat(dataset + '/smallnorb_test_t.mat')
    test_t = data['testlabels'].flatten().astype('float32')
    test_t = np.concatenate([test_t, test_t])
    with open(dataset+'/norbsmall32x32.cpkl','w') as f:
        cPkl.dump([train_x, train_t, test_x, test_t], f,
                  protocol=cPkl.HIGHEST_PROTOCOL)


def _download_rotten_tomatoes(dataset):
    origin = ('http://www.cs.cornell.edu/people/pabo/'
              'movie-review-data/rt-polaritydata.tar.gz')

    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset + '/rt-polaritydata.tar.gz')


def load_norb_small(
        dataset=_get_datafolder_path()+'/norb_small/norbsmall32x32.cpkl',
        dequantify=True,
        normalize=True ):
    '''
    Loads the real valued MNIST dataset
    :param dataset: path to dataset file
    :return: None
    '''
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_norb_small(datasetfolder)

    with open(dataset,'r') as f:
        train_x, train_t, test_x, test_t = cPkl.load(f)

    if dequantify:
        train_x += np.random.uniform(0,1,size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0,1,size=test_x.shape).astype('float32')
    if normalize:
        normalizer = train_x.max().astype('float32')
        train_x = train_x / normalizer
        test_x = test_x / normalizer

    return train_x, train_t, test_x, test_t


def _download_omniglot(dataset):
    """
    Download the omniglot dataset if it is not present.
    :return: The train, test and validation set.
    """
    from scipy.misc import imread,imresize
    origin_eval = (
        "https://github.com/brendenlake/omniglot/"
        "raw/master/python/images_evaluation.zip"
    )
    origin_back = (
        "https://github.com/brendenlake/omniglot/"
        "raw/master/python/images_background.zip"
    )
    print 'Downloading data from %s' % origin_eval
    urllib.urlretrieve(origin_eval, dataset + '/images_evaluation.zip')
    print 'Downloading data from %s' % origin_back
    urllib.urlretrieve(origin_back, dataset + '/images_background.zip')

    with zipfile.ZipFile(dataset + '/images_evaluation.zip', "r") as z:
        z.extractall(dataset)
    with zipfile.ZipFile(dataset + '/images_background.zip', "r") as z:
        z.extractall(dataset)

    background =  dataset + '/images_background'
    evaluation =  dataset + '/images_evaluation'
    matches = []
    for root, dirnames, filenames in os.walk(background):
        for filename in fnmatch.filter(filenames, '*.png'):
            matches.append(os.path.join(root, filename))
    for root, dirnames, filenames in os.walk(evaluation):
        for filename in fnmatch.filter(filenames, '*.png'):
            matches.append(os.path.join(root, filename))

    train = []
    test = []

    def _load_image(fn):
        image = imread(fn, True)
        image = imresize(image, (32, 32), interp='bicubic')
        image = image.reshape((-1))
        image = np.abs(image-255.)/255.
        return image

    for p in matches:
        if any(x in p for x in ['16.png','17.png','18.png','19.png','20.png']):
            test.append(_load_image(p))
        else:
            train.append(_load_image(p))

    shutil.rmtree(background+'/')
    shutil.rmtree(evaluation+'/')

    test = np.asarray(test)
    train = np.asarray(train)
    with open(dataset+'/omniglot.cpkl','w') as f:
        cPkl.dump([train, test],f,protocol=cPkl.HIGHEST_PROTOCOL)


def _download_lwf(dataset,size):
    from sklearn.datasets import fetch_lfw_people
    '''
    :param dataset:
    :return:
    '''
    lfw_people = fetch_lfw_people(color=True,resize=size)
    f = gzip.open(dataset, 'w')
    cPkl.dump([lfw_people.images.astype('uint8'),lfw_people.target], f,
              protocol=cPkl.HIGHEST_PROTOCOL)
    f.close()


def _download_mnist_binarized(datapath):
    """
    Download the fized binzarized MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    datafiles = {
        "train": "http://www.cs.toronto.edu/~larocheh/public/"
                 "datasets/binarized_mnist/binarized_mnist_train.amat",
        "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                 "binarized_mnist/binarized_mnist_valid.amat",
        "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                "binarized_mnist/binarized_mnist_test.amat"
    }
    datasplits = {}
    for split in datafiles.keys():
        print "Downloading %s data..." %(split)
        local_file = datapath + '/binarized_mnist_%s.npy'%(split)
        datasplits[split] = np.loadtxt(urllib.urlretrieve(datafiles[split])[0])

    f = gzip.open(datapath +'/mnist.pkl.gz', 'w')
    pkl.dump([datasplits['train'],datasplits['valid'],datasplits['test']],f)



def load_omniglot(dataset=_get_datafolder_path()+'/omniglot'):
    '''
    Loads the real valued MNIST dataset
    :param dataset: path to dataset file
    :return: None
    '''
    if not os.path.exists(dataset):
        os.makedirs(dataset)
        _download_omniglot(dataset)

    with open(dataset+'/omniglot.cpkl', 'rb') as f:
        train, test = cPkl.load(f)

    train = train.astype('float32')
    test = test.astype('float32')

    return train, test


def load_omniglot_iwae(dataset=_get_datafolder_path()+'/omniglot_iwae'):
    '''
    Loads the real valued MNIST dataset
    :param dataset: path to dataset file
    :return: None
    '''
    from scipy.io import loadmat
    if not os.path.exists(dataset):
        os.makedirs(dataset)
        _download_omniglot_iwae(dataset)

    data = loadmat(dataset+'/chardata.mat')

    train_x = data['data'].astype('float32').T
    train_t = np.argmax(data['target'].astype('float32').T,axis=1)
    train_char = data['targetchar'].astype('float32')
    test_x = data['testdata'].astype('float32').T
    test_t = np.argmax(data['testtarget'].astype('float32').T,axis=1)
    test_char = data['testtargetchar'].astype('float32')


    return train_x, train_t, train_char, test_x, test_t, test_char


def load_mnist_realval(
        dataset=_get_datafolder_path()+'/mnist_real/mnist.pkl.gz'):
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


def load_mnist_binarized(
        dataset=_get_datafolder_path()+'/mnist_binarized/mnist.pkl.gz'):
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

def _download_rcv1():
    """
    Download the rcv1 dataset from scikitlearn.
    :return: The train, test and validation set.
    """
    from sklearn.datasets import fetch_rcv1
    print "downloading rcv1 train data...."
    newsgroups_train = fetch_rcv1(subset='train')
    print "downloading rcv1 test data...."
    newsgroups_test = fetch_rcv1(subset='test')
    train_set = (newsgroups_train.data, newsgroups_train.target)
    test_set = (newsgroups_test.data, newsgroups_test.target)

    return train_set,test_set


def _download_20newsgroup():
    """
    Download the 20 newsgroups dataset from scikitlearn.
    :return: The train, test and validation set.
    """
    from sklearn.datasets import fetch_20newsgroups
    print "downloading 20 newsgroup train data...."
    newsgroups_train = fetch_20newsgroups(
        subset='train', remove=('headers', 'footers', 'quotes'))
    print "downloading 20 newsgroup test data...."
    newsgroups_test = fetch_20newsgroups(
        subset='test', remove=('headers', 'footers', 'quotes'))
    train_set = (newsgroups_train.data, newsgroups_train.target)
    test_set = (newsgroups_test.data, newsgroups_test.target)

    return train_set,test_set

def _bow(train, test, max_features=1000):
    '''
    bag-of-words encoding helper function
    '''
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer as EnglishStemmer
    from nltk.tokenize import wordpunct_tokenize as wordpunct_tokenize

    x_train, y_train = train
    x_test, y_test = test


    stemmer = EnglishStemmer()
    lemmatizer = WordNetLemmatizer()
    for i in range(len(x_train)):
        x_train[i] = " ".join([lemmatizer.lemmatize(stemmer.stem(token.lower()))
                               for token in wordpunct_tokenize(
            re.sub('[%s]' % re.escape(string.punctuation), '', x_train[i]))])

    vectorizer_train = CountVectorizer(strip_accents='ascii',
                                       stop_words='english',
                                       token_pattern=r"(?u)\b\w[a-z]\w+[a-z]\b",
                                       max_features=max_features,
                                       vocabulary=None, dtype='float32')
    x_train = vectorizer_train.fit_transform(x_train).toarray()


    vocab_train = vectorizer_train.get_feature_names()
    vectorizer_test = CountVectorizer(strip_accents='ascii',
                                      stop_words='english',
                                      token_pattern=r"(?u)\b\w[a-z]\w+[a-z]\b",
                                      max_features=max_features,
                                      vocabulary=vocab_train,
                                      dtype='float32')
    x_test = vectorizer_test.fit_transform(x_test).toarray()

    # remove documents with no words
    r = np.where(x_train.sum(axis=1) > 0.)[0]
    x_train = x_train[r, :]
    y_train = y_train[r]

    r = np.where(x_test.sum(axis=1) > 0.)[0]
    x_test = x_test[r, :]
    y_test = y_test[r]

    return (x_train, y_train),(x_test, y_test), vocab_train


def _download_cifar10(dataset):
    """
    Download the Cifar10 dataset if it is not present.
    """
    origin = (
        'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset)


def load_cifar10(
        dataset=_get_datafolder_path()+'/cifar10/cifar-10-python.tar.gz',
        normalize=True,
        dequantify=True):
    '''
    Loads the cifar10 dataset
    :param dataset: path to dataset file
    :param normalize: normalize the x data to the range [0,1]
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
    :return: train and test data
    '''
    datasetfolder = os.path.dirname(dataset)
    batch_folder = datasetfolder+ '/cifar-10-batches-py/'
    if not os.path.isfile(dataset):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_cifar10(dataset)

    if not os.path.isfile(batch_folder + 'data_batch_5'):
        with tarfile.open(dataset) as tar:
            tar.extractall(os.path.dirname(dataset))

    train_x, train_y = [],[]
    for i in ['1','2','3','4','5']:
        with open(batch_folder + 'data_batch_'+ i,'r') as f:
            data = cPkl.load(f)
            train_x += [data['data']]
            train_y += [data['labels']]
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)


    with open(batch_folder + 'test_batch','r') as f:
        data = cPkl.load(f)
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    if dequantify:
        train_x += np.random.uniform(0,1,size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0,1,size=test_x.shape).astype('float32')
    if normalize:
        normalizer = train_x.max().astype('float32')
        train_x = train_x / normalizer
        test_x = test_x / normalizer

    train_x = train_x.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_x = test_x.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    return train_x.astype('float32'), train_y, test_x.astype('float32'), test_y


def load_frey_faces(
        dataset=_get_datafolder_path()+'/frey_faces/frey_faces',
        normalize=True,
        dequantify=True):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
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
        normalizer = data.max().astype('float32')
        data = data / normalizer
    return data

def load_lfw(
        dataset=_get_datafolder_path()+'/lfw/lfw',
        normalize=True,
        dequantify=True,
        size=0.25):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
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
        normalizer = data.max().astype('float32')
        data = data / normalizer
    return data


def load_svhn(
        dataset=_get_datafolder_path()+'/svhn/',
        normalize=True,
        dequantify=True,
        extra=False):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
    :param extra: include extra svhn samples
    :return:
    '''

    if not os.path.isfile(dataset +'svhn_train.cpkl'):
        datasetfolder = os.path.dirname(dataset +'svhn_train.cpkl')
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_svhn(dataset, extra=False)

    with open(dataset +'svhn_train.cpkl', 'rb') as f:
        train_x,train_y = cPkl.load(f)
    with open(dataset +'svhn_test.cpkl', 'rb') as f:
        test_x,test_y = cPkl.load(f)

    if extra:
        if not os.path.isfile(dataset +'svhn_extra.cpkl'):
            datasetfolder = os.path.dirname(dataset +'svhn_train.cpkl')
            if not os.path.exists(datasetfolder):
                os.makedirs(datasetfolder)
            _download_svhn(dataset, extra=True)

        with open(dataset +'svhn_extra.cpkl', 'rb') as f:
            extra_x,extra_y = cPkl.load(f)
        train_x = np.concatenate([train_x,extra_x])
        train_y = np.concatenate([train_y,extra_y])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_y = train_y.astype('int32')
    test_y = test_y.astype('int32')

    if dequantify:
        train_x += np.random.uniform(0,1,size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0,1,size=test_x.shape).astype('float32')

    if normalize:
        normalizer = train_x.max().astype('float32')
        train_x = train_x / normalizer
        test_x = test_x / normalizer

    return train_x, train_y, test_x, test_y



def _download_svhn(dataset, extra):
    """
    Download the SVHN dataset
    """
    from scipy.io import loadmat

    print 'Downloading data from http://ufldl.stanford.edu/housenumbers/, ' \
          'this may take a while...'
    if extra:
        print "Downloading extra data..."
        urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',
                           dataset+'extra_32x32.mat')
        extra = loadmat(dataset+'extra_32x32.mat')
        extra_x = extra['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
        extra_y = extra['y'].reshape((-1)) - 1

        print "Saving extra data"
        with open(dataset +'svhn_extra.cpkl', 'w') as f:
            pkl.dump([extra_x,extra_y],f,protocol=cPkl.HIGHEST_PROTOCOL)
        os.remove(dataset+'extra_32x32.mat')

    else:
        print "Downloading train data..."
        urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                           dataset+'train_32x32.mat')
        print "Downloading test data..."
        urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                           dataset+'test_32x32.mat')

        train = loadmat(dataset+'train_32x32.mat')
        train_x = train['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
        train_y = train['y'].reshape((-1)) - 1
        test = loadmat(dataset+'test_32x32.mat')
        test_x = test['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
        test_y = test['y'].reshape((-1)) - 1

        print "Saving train data"
        with open(dataset +'svhn_train.cpkl', 'w') as f:
            cPkl.dump([train_x,train_y],f,protocol=cPkl.HIGHEST_PROTOCOL)
        print "Saving test data"
        with open(dataset +'svhn_test.cpkl', 'w') as f:
            pkl.dump([test_x,test_y],f,protocol=cPkl.HIGHEST_PROTOCOL)
        os.remove(dataset+'train_32x32.mat')
        os.remove(dataset+'test_32x32.mat')



def load_20newsgroup(
        dataset=_get_datafolder_path()+'/20newsgroup/',
        max_feat=1000,
        normalize_by_doc_len=True):
    '''
    Loads 20 newsgroup dataset
    :param dataset: path to dataset file
    :return: None
    '''

    datasetfolder = os.path.dirname(dataset)
    if not os.path.exists(datasetfolder):
        os.makedirs(datasetfolder)

    if not os.path.isfile(dataset + '20newsgroup_mf'+ str(max_feat) + '.pkl'):
        train_set,test_set = _download_20newsgroup()
        train_set, test_set, vocab_train \
            = _bow(train_set, test_set, max_features=max_feat)
        with open(dataset + '20newsgroup_mf'+ str(max_feat) + '.pkl','w') as f:
            pkl.dump([train_set, test_set, vocab_train],f)

    with open(dataset + '20newsgroup_mf'+ str(max_feat) + '.pkl','r') as f:
        train_set, test_set, vocab_train = pkl.load(f)

    x_train, y_train = train_set
    x_test, y_test = test_set

    if normalize_by_doc_len:
        x_train = x_train / (x_train).sum(keepdims=True, axis=1)
        x_test = x_test / (x_test).sum(keepdims=True, axis=1)

    return x_train.astype('float32'), \
           y_train.astype('float32'), \
           x_test.astype('float32'), \
           y_test.astype('float32')


def load_rcv1(
        dataset=_get_datafolder_path()+'/rcv1/',
        max_features=10000,
        normalize_by_doc_len=True):
    '''
    Loads 20 newsgroup dataset
    :param dataset: path to dataset file
    :return: None
    '''

    datasetfolder = os.path.dirname(dataset)
    if not os.path.exists(datasetfolder):
        os.makedirs(datasetfolder)

    if not os.path.isfile(dataset + 'rcv1_mf'+ str(max_features) + '.cpkl'):
        train_set,test_set = _download_rcv1()
        train_set, test_set, vocab_train = \
            _bow(train_set, test_set, max_features=max_features)
        with open(dataset + 'rcv1_mf'+ str(max_features) + '.cpkl','w') as f:
            cPkl.dump([train_set, test_set, vocab_train],f)

    with open(dataset + 'rcv1_mf'+ str(max_features) + '.cpkl','r') as f:
        train_set, test_set, vocab_train = cPkl.load(f)

    x_train, y_train = train_set
    x_test, y_test = test_set

    if normalize_by_doc_len:
        x_train = x_train / (x_train).sum(keepdims=True, axis=1)
        x_test = x_test / (x_test).sum(keepdims=True, axis=1)

    return x_train.astype('float32'), \
           y_train.astype('float32'), \
           x_test.astype('float32'), \
           y_test.astype('float32')


def load_rotten_tomatoes(dataset=_get_datafolder_path()+'/rotten_tomatoes/',
                         vocab_size=40, minimum_len=None, maximum_len=None,
                         seed=1234):
    """Loader for rotten tomatoes sentiment analysis dataset

    Loads dataset as characters

    :param dataset: str
        path to dataset file
    :param vocab_size: int or str
         number of vocab_size in VOCAB. Defaults to the 40 most frequent
         vocab_size if str uses str as vocab
    :param minimum_len: None or int
        if None no filtering
    :param maximum_len: None or int
        if None no filtering
    :param seed: int
         random seed
    :return: X, y, mask, vocab

    Notes
    -----
    The vocab output can be used to encode several character dataset with
    the same encoding.
    """
    from collections import Counter
    try:
        import pandas as pd
    except:
        raise ImportError('Pandas is required')

    datasetfolder = os.path.dirname(dataset)

    if not os.path.exists(datasetfolder):
        os.makedirs(datasetfolder)

    fn_pos = dataset + '/rt-polaritydata/rt-polarity.pos'
    fn_neg = dataset + '/rt-polaritydata/rt-polarity.neg'
    if not os.path.isfile(dataset + '/rt-polaritydata.tar.gz'):
        _download_rotten_tomatoes(dataset)

    with tarfile.open(dataset + '/rt-polaritydata.tar.gz', "r:gz") as tar:
        tar.extractall(path=dataset)

    # load review data
    pos = pd.read_csv(
        fn_pos, delimiter="\t", quoting=3, header=None)
    neg = pd.read_csv(
        fn_neg, header=None, delimiter="\t", quoting=3)
    pos.columns = ["review"]
    neg.columns = ["review"]

    # helper clean function
    def clean(l):
        l = l.strip('\n')
        l = l.rstrip(' ')
        l = l.lower()
        return l

    pos_lst = [clean(l) for l in pos['review']]
    neg_lst = [clean(l) for l in neg['review']]

    # count character counts and return the "character" most frequently
    # used vocab_size as list

    if isinstance(vocab_size, int):
        char_counts = Counter("".join(pos_lst + neg_lst))
        char_counts_sorted = sorted(
            char_counts.items(), key=lambda x:x[1], reverse=True)

        chars, counts = zip(*char_counts_sorted[:vocab_size])

        print "Using VOCAB with the %i most frequent characters \n |%s|" % (
            vocab_size, "".join(chars))
        print "Character counts", ", ".join(map(str, counts))

        vocab = sorted(chars)
        char2idx = {char:idx for idx, char in enumerate(vocab)}
        unk_idx = max(char2idx.values()) + 1
    else:
        vocab = vocab_size
        print "Using VOCAB: |%s|" % vocab_size
        char2idx = {char:idx for idx, char in enumerate(vocab_size)}
        unk_idx = max(char2idx.values()) + 1

    # find maximum sequence length
    max_len = max(map(len, pos_lst + neg_lst))

    # helper function for converting chars to matrix format
    def create_matrix(reviews, y_cls):
        num_seqs = len(reviews)
        X = np.zeros((num_seqs, max_len), dtype='int32') -1  # set all to -1
        for row in range(num_seqs):
            review = reviews[row]
            for col in range(len(review)):
                # try to look up key otherwise use unk_idx
                if review[col] in char2idx:
                    char_idx = char2idx[review[col]]
                else:
                    char_idx = unk_idx
                X[row, col] = char_idx

        mask = (X != -1).astype('float32')
        X[X==-1] = 0
        y = np.ones(num_seqs, dtype='int32')*y_cls
        return X, y, mask

    X_pos, y_pos, mask_pos = create_matrix(pos_lst, 1)
    X_neg, y_neg, mask_neg = create_matrix(neg_lst, 0)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    mask = np.concatenate([mask_pos, mask_neg])

    print "-"*40
    print "Minium length filter :", minimum_len
    print "Maximum length filter:", maximum_len
    if minimum_len is not None:
        seq_lens = mask.sum(axis=1)
        keep = seq_lens >= minimum_len
        print "Seqs below minimum   : %i" % np.invert(keep).sum()
        X = X[keep, :]
        y = y[keep]
        mask = mask[keep, :]

    if maximum_len is not None:
        seq_lens = mask.sum(axis=1)
        keep = seq_lens <= maximum_len
        print "Seqs above maximum   : %i" % np.invert(keep).sum()
        X = X[keep, :]
        y = y[keep]
        mask = mask[keep, :]

    np.random.seed(seed)
    p = np.random.permutation(X.shape[0])
    X = X[p]
    y = y[p]
    mask = mask[p]

    seq_lens = mask.sum(axis=1).astype('int32')
    print "X                    :", X.shape, X.dtype
    print "y                    :", y.shape, y.dtype
    print "mask                 :", mask.shape, mask.dtype
    print "MIN length           : ", seq_lens.min()
    print "MAX length           : ", seq_lens.max()
    print "MEAN length          : ", seq_lens.mean()
    print "UNKOWN chars         : ", np.sum(X==unk_idx)
    print "-"*40

    # check that idx's in X is the number of vocab_size + unk_idx
    n = vocab_size if isinstance(vocab_size, int) else len(vocab_size)
    assert len(np.unique(X)) == n  + 1
    assert sum(np.unique(y)) == 1 # check that y is 0,1
    return X, y, mask, vocab


def _one_hot(x,n_labels=None):
    if n_labels is None:
        n_labels = np.max(x)
    return np.eye(n_labels)[x]

def _download_and_extract_stl10(dest_directory):
    """
    SOURCE: https://github.com/mttk/STL10
    Download and extract the STL-10 dataset
    :return: None
    """
    import sys
    origin = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = origin.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(origin, filepath, reporthook=_progress)
        print('Downloaded', filename)

    binary_directory = os.path.join(dest_directory, 'stl10_binary')
    if not os.path.exists(binary_directory):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    return binary_directory


def load_stl10(
        dataset=_get_datafolder_path()+'/stl10/stl10_binary.tar.gz',
        normalize=False,
        dequantify=False):
    '''
    Loads the stl10 dataset
    :param dataset: path to dataset file
    :param normalize: Not supported. For normalization we would need to
                      convert the dataset to float32 which would increase
                      the dataset size further
    :param dequantify: not supported
    :return: data. Note that the data will be returned as uint8 to save memory.
            You'll need to convert it to float32.



    '''
    if normalize is True:
        raise ValueError('Normalization with STL10 loader is not supported. '
                         'Create an iterator that normalizes on the fly')
    if dequantify is True:
        raise ValueError('Dequantify is not supported with STL10 loader. '
                         'Create an iterator that dequantifies on the fly')

    def read_all_images(path_to_data):
        print "Loading %s" % path_to_data,
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
            print "shp", images.shape, "dtype", images.dtype
            return images

    def read_labels(path_to_labels):
        print "Loading %s" % path_to_labels,
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            labels -= 1 # from 1...10 to 0...9
            print "shp", labels.shape, "dtype", labels.dtype
        return labels

    datasetfolder = os.path.dirname(dataset)
    # download and extract if nessesary
    binary_directory = _download_and_extract_stl10(datasetfolder)

    data_path_train = os.path.join(binary_directory, 'train_X.bin')
    data_path_test = os.path.join(binary_directory, 'test_X.bin')
    data_path_unlab = os.path.join(binary_directory, 'unlabeled_X.bin')
    label_path_train = os.path.join(binary_directory, 'train_y.bin')
    label_path_test = os.path.join(binary_directory, 'test_y.bin')

    x_train = read_all_images(data_path_train)
    x_test = read_all_images(data_path_test)
    x_unlab = read_all_images(data_path_unlab)
    y_train = read_labels(label_path_train)
    y_test = read_labels(label_path_test)
    return x_train, y_train, x_test, y_test, x_unlab


if __name__ == "__main__":
    out = load_svhn()
    print "done"
