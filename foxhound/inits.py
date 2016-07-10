import os
import numpy as np
from time import time
from sklearn.externals import joblib

import theano
import theano.tensor as T

from theano_utils import sharedX, floatX, intX
from rng import np_rng

class W2VEmbedding(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, vocab, name=None):
        t = time()
        w2v_vocab = joblib.load(os.path.join(self.data_dir, '3m_w2v_gn_vocab.jl'))
        w2v_embed = joblib.load(os.path.join(self.data_dir, '3m_w2v_gn.jl'))
        mapping = {}
        for i, w in enumerate(w2v_vocab):
            w = w.lower()
            if w in mapping:
                mapping[w].append(i)
            else:
                mapping[w] = [i]
        widxs = []
        w2vidxs = []
        for i, w in enumerate(vocab):
            w = w.replace('`', "'")
            if w in mapping:
                w2vi = min(mapping[w])
                w2vidxs.append(w2vi)
                widxs.append(i)
        # w = np_rng.uniform(low=-0.05, high=0.05, size=(len(vocab), w2v_embed.shape[1]))
        w = np.zeros((len(vocab), w2v_embed.shape[1]))
        w[widxs, :] = w2v_embed[w2vidxs, :]/2.
        return sharedX(w, name=name)

class Uniform(object):
    def __init__(self, scale=0.05):
        self.scale = 0.05

    def __call__(self, shape):
        return sharedX(np_rng.uniform(low=-self.scale, high=self.scale, size=shape))

class Normal(object):
    def __init__(self, loc=0., scale=0.05):
        self.scale = scale
        self.loc = loc

    def __call__(self, shape, name=None):
        return sharedX(np_rng.normal(loc=self.loc, scale=self.scale, size=shape), name=name)

class Orthogonal(object):
    """ benanne lasagne ortho init (faster than qr approach)"""
    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, shape, name=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np_rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return sharedX(self.scale * q[:shape[0], :shape[1]], name=name)

class Frob(object):

    def __init__(self):
        pass

    def __call__(self, shape, name=None):
        r = np_rng.normal(loc=0, scale=0.01, size=shape)
        r = r/np.sqrt(np.sum(r**2))*np.sqrt(shape[1])
        return sharedX(r, name=name)

class Constant(object):

    def __init__(self, c=0.):
        self.c = c

    def __call__(self, shape):
        return sharedX(np.ones(shape) * self.c)

class Identity(object):

    def __init__(self, scale=0.25):
        self.scale = scale

    def __call__(self, shape):
        return sharedX(np.identity(shape[0]) * self.scale)

class ReluInit(object):

    def __init__(self):
        pass

    def __call__(self, shape):
        if len(shape) == 2:
            scale = np.sqrt(2./shape[0])
        elif len(shape) == 4:
            scale = np.sqrt(2./np.prod(shape[1:]))
        else:
            raise NotImplementedError
        return sharedX(np_rng.normal(size=shape, scale=scale))