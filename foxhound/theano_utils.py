import numpy as np
import theano
import theano.tensor as T

def euclidean(x, y, e1=1e-3, e2=1e-3):
    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e1))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e1))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist + e2)
    return dist

def diag_gaussian(X, m, c):
    """theano version of function from sklearn/mixture/gmm/gmm.py"""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + T.sum(T.log(c), 1)
                  + T.sum((m ** 2) / c, 1)
                  - 2 * T.dot(X, (m / c).T)
                  + T.dot(X ** 2, (1.0 / c).T))
    return lpr

def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d

def pair_cosine(a, b, e=1e-8):
    return T.sum(a*b, axis=1)/(l2norm(a, e=e)*l2norm(b, e=e))

def pair_euclidean(a, b, axis=1, e=1e-8):
    return T.sqrt(T.sum(T.sqr(a - b), axis=axis) + e)

def l2norm(x, axis=1, e=1e-8):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=True) + e)

def l1norm(x, axis=1):
    return T.sum(T.abs_(x), axis=axis)

def intX(X):
    return np.asarray(X, dtype=np.int32)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape)*n, dtype=dtype, name=name)

def downcast_float(X):
    return np.asarray(X, dtype=np.float32)