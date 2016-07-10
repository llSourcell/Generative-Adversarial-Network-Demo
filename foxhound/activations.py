import theano
import theano.tensor as T

class Softmax(object):

    def __init__(self):
        pass

    def __call__(self, x):
        e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

class Maxout(object):

    def __init__(self, n_pool=2):
        self.n_pool = n_pool

    def __call__(self, x):
        if x.ndim == 2:
            x = T.max([x[:, n::self.n_pool] for n in range(self.n_pool)], axis=0)
        elif x.ndim == 4:
            x = T.max([x[:, n::self.n_pool, :, :] for n in range(self.n_pool)], axis=0)
        elif x.ndim == 3:
            print 'assuming standard rnn 3tensor'
            x = T.max([x[:, :, n::self.n_pool] for n in range(self.n_pool)], axis=0)
        return x

class ConvRMSPool(object):

    def __init__(self):
        pass

    def __call__(self, x):
        x = x**2
        return T.sqrt(x[:, ::2, :, :]+x[:, 1::2, :, :]+1e-8)

class ConvSoftmax(object):

    def __init__(self):
        pass

    def __call__(self, x):
        e_x = T.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

class MaskedConvSoftmax(object):

    def __init__(self):
        pass

    def __call__(self, x, m):
        x = x*m.dimshuffle(0, 'x', 1, 'x')
        e_x = T.exp(x - x.max(axis=2, keepdims=True))
        e_x = e_x*m.dimshuffle(0, 'x', 1, 'x')
        return e_x / e_x.sum(axis=2, keepdims=True)
        # return e_x / T.clip(e_x.sum(axis=2, keepdims=True), 1e-6, 1e6)

class ELU(object):
    def __init__(self):
        pass

    def __call__(self, x):
        # t = x >= 0
        # return t*x+(1-t)*(T.exp(x)-1)
        return T.switch(T.ge(x, 0), x, T.exp(x)-1)

class Rectify(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return (x + abs(x)) / 2.0

class ClippedRectify(object):

    def __init__(self, clip=10.):
        self.clip = clip

    def __call__(self, x):
        return T.clip((x + abs(x)) / 2.0, 0., self.clip)

class LeakyRectify(object):

    def __init__(self, leak=0.2):
        self.leak = leak

    def __call__(self, x):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * x + f2 * abs(x)

class Prelu(object):

    def __init__(self):
        pass

    def __call__(self, x, leak):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        if leak.ndim == 1:
            return T.flatten(f1, 1)[0] * x + T.flatten(f2, 1)[0] * abs(x)
        else:
            return f1 * x + f2 * abs(x)

class Tanh(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.tanh(x)

class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.nnet.sigmoid(x)

class Linear(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return x

class SteeperSigmoid(object):

    def __init__(self, scale=3.75):
        self.scale = scale

    def __call__(self, x):
        return 1./(1. + T.exp(-self.scale * x))

class HardSigmoid(object):

    def __init__(self):
        pass

    def __call__(self, X):
        return T.clip(X + 0.5, 0., 1.)