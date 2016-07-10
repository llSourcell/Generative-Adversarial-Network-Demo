import theano
import theano.tensor as T
import inits
import activations
import updates
import numpy as np
from theano.tensor.extra_ops import repeat
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from utils import instantiate
from theano_utils import shared0s, sharedX
from rng import t_rng

def same_pad(n):
    return int(np.floor(n / 2.))

class Input(object):

    def __init__(self, shape, dtype=theano.config.floatX):
        self.X = T.TensorType(dtype, (False,)*(len(shape)))()
        print self.X.type
        self.out_shape = shape
        self.dtype = dtype

    def op(self, state):
        return self.X

class Flatten(object):

    def __init__(self, axes):
        self.axes = axes

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        if self.axes == len(self.in_shape):
            self.out_shape = self.in_shape
        else:
            self.out_shape = self.in_shape[:self.axes-1] + [np.prod(self.in_shape[self.axes-1:])]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return T.flatten(X, outdim=self.axes)

class Embedding(object):

    def __init__(self, dim, n_embed, init_fn='uniform', update_fn='nag'):
        self.dim = dim
        self.n_embed = n_embed
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.in_shape[1],
            self.dim
        ]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.n_embed, self.dim))
        self.params = [self.w]

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.w[X]

    def update(self, cost):
        return self.update_fn(self.params, cost)

class MaxPool(object):

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.in_shape[1],
            int(np.ceil(float(self.in_shape[2]) / self.shape[0])),
            int(np.ceil(float(self.in_shape[3]) / self.shape[1]))
        ]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return max_pool_2d(X, self.shape)

class CUDNNPool(object):

    def __init__(self, shape, stride, pad=(0, 0), mode='max'):
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(pad, int):
            pad = (pad, pad)
        self.pad = pad

        self.mode = mode

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.in_shape[1],
            (self.in_shape[2] - self.shape[0] + self.pad[0]*2) // self.stride[0] + 1,
            (self.in_shape[3] - self.shape[1] + self.pad[1]*2) // self.stride[1] + 1
        ]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return dnn_pool(X, self.shape, self.stride, self.mode, self.pad)

class FilterPool2D(object):

    def __init__(self, fn=lambda x:T.mean(x, axis=2)):
        if isinstance(fn, basestring):
            if fn == 'rms':
                fn = lambda x:T.sqrt(T.mean(x*x, axis=2) + 1e-6)
            elif fn == 'max':
                fn = lambda x:T.max(x, axis=2)
            elif fn == 'mean' or fn == 'avg':
                fn = lambda x:T.mean(x, axis=2)
            else:
                raise NotImplementedError
        self.fn = fn

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.in_shape[1],
        ]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.fn(X.reshape((X.shape[0], X.shape[1], -1)))

class ConvLPNorm(object):

    def __init__(self, init_fn='normal', update_fn='nag'):
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.in_shape[1],
        ]
        print self.out_shape

    def init(self):
        self.lpn = self.init_fn((self.in_shape[1]))
        # self.lpn = inits.Constant(c=-5)((self.in_shape[1]))
        self.params = [self.lpn]

    def op(self, state):
        X = self.l_in.op(state=state)
        lpn = 1.+T.log(1.+T.exp(self.lpn))
        lpnb = lpn.dimshuffle('x', 0, 'x', 'x')
        X = T.abs_(X)**lpnb
        X = T.mean(X, axis=[2, 3])
        X = T.pow(X, 1/lpn)
        return X

    def update(self, cost):
        return self.update_fn(self.params, cost)

class EmbeddingLPNorm(object):

    def __init__(self, init_fn='normal', update_fn='nag'):
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[1],
            self.in_shape[2],
        ]
        print self.out_shape

    def init(self):
        self.lpn = self.init_fn((self.in_shape[2]))
        # self.lpn = inits.Constant(c=-5)((self.in_shape[1]))
        self.params = [self.lpn]

    def op(self, state):
        X = self.l_in.op(state=state)
        lpn = 1.+T.log(1.+T.exp(self.lpn))
        lpnb = lpn.dimshuffle('x', 'x', 0)
        X = T.abs_(X)**lpnb
        X = T.mean(X, axis=[0])
        X = T.pow(X, 1/lpn)
        return X

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Conv(object):

    def __init__(self, n=32, shape=(3, 3), pad='same', stride=(1, 1), init_fn='orthogonal', update_fn='nag'):
        self.n = n

        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape

        if isinstance(pad, int):
            pad = (pad, pad) 
        elif pad == 'same':
            pad = (same_pad(shape[0]), same_pad(shape[1]))
        self.pad = pad

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.n, 
            (self.in_shape[2] - self.shape[0] + self.pad[0] * 2)/self.stride[0] + 1, 
            (self.in_shape[3] - self.shape[1] + self.pad[1] * 2)/self.stride[1] + 1
        ]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.n, self.in_shape[1], self.shape[0], self.shape[1]))
        self.params = [self.w]

    def op(self, state):
        X = self.l_in.op(state=state)
        return dnn_conv(X, self.w, subsample=self.stride, border_mode=self.pad)

    def update(self, cost):
        return self.update_fn(self.params, cost)

class CPUConv(object):

    def __init__(self, n=32, shape=(3, 3), pad='same', stride=(1, 1), init_fn='orthogonal', update_fn='nag'):
        self.n = n

        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape

        if pad != 'same':
            raise NotImplementedError('Only same pad supported right now!')
        self.pad = pad

        if isinstance(stride, int):
            stride = (stride, stride)
        if stride != (1, 1):
            raise NotImplementedError('Only (1, 1) stride supported right now!')
        self.stride = stride

        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = self.l_in.out_shape
        self.out_shape = [
            self.in_shape[0],
            self.n, 
            self.in_shape[2], 
            self.in_shape[3]
        ]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.n, self.in_shape[1], self.shape[0], self.shape[1]))
        self.params = [self.w]

    def op(self, state):
        """ Benanne lasange same for cpu """
        X = self.l_in.op(state=state)
        out = T.nnet.conv2d(X, self.w, subsample=self.stride, border_mode='full')
        shift_x = (self.shape[0] - 1) // 2
        shift_y = (self.shape[1] - 1) // 2
        return out[:, :, shift_x:self.out_shape[2] + shift_x, shift_y:self.out_shape[3] + shift_y]

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Variational(object):

    def __init__(self, dim=256, init_fn='orthogonal', update_fn='nag'):
        self.dim = dim
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)        

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = [self.in_shape[0], self.dim]
        print self.out_shape

    def init(self):
        self.wmu = self.init_fn((self.in_shape[-1], self.out_shape[-1]))
        self.wsigma = self.init_fn((self.in_shape[-1], self.out_shape[-1]))
        self.params = [self.wmu, self.wsigma]

    def op(self, state):
        X = self.l_in.op(state=state)
        self.mu = T.dot(X, self.wmu)
        self.log_sigma = 0.5 * T.dot(X, self.wsigma) 
        if state['sample']:
            Z = state['sample']
            # Z = t_rng.normal(self.log_sigma.shape)
        else:
            Z = self.mu + T.exp(self.log_sigma) * t_rng.normal(self.log_sigma.shape)
        return Z

    def cost(self):
        return -0.5 * T.sum(1 + 2*self.log_sigma - self.mu**2 - T.exp(2*self.log_sigma))

class Project(object):

    def __init__(self, dim=256, init_fn='orthogonal', update_fn='nag'):
        self.dim = dim
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        naxes = len(self.in_shape)
        if naxes == 3:
            self.out_shape = self.in_shape[:-1] + [self.dim]
        else:
            self.out_shape = [self.in_shape[0], self.dim]
        print self.out_shape

    def init(self):
        self.w = self.init_fn((self.in_shape[-1], self.out_shape[-1]))
        self.params = [self.w]

    def op(self, state):
        X = self.l_in.op(state=state)
        return T.dot(X, self.w)

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Dropout(object):

    def __init__(self, p_drop=0.5):
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        retain_prob = 1 - self.p_drop  
        if state['dropout']:
            X = X / retain_prob * t_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        return X

class Shift(object):

    def __init__(self, init_fn='constant', update_fn='nag'):
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape
        n_dim_in = len(l_in.out_shape)
        if n_dim_in == 4:
            self.conv = True
        elif n_dim_in == 2:
            self.conv = False
        else:
            raise NotImplementedError

    def init(self):
        if self.conv:
            self.b = self.init_fn(self.out_shape[1])
        else:
            self.b = self.init_fn(self.out_shape[-1])

        self.params = [self.b]

    def op(self, state):
        X = self.l_in.op(state=state)
        if self.conv:
            return X + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            return X + self.b  

    def update(self, cost):
        return self.update_fn(self.params, cost)

class Activation(object):

    def __init__(self, activation, init_fn=inits.Constant(c=0.25), update_fn='nag'):
        self.activation = instantiate(activations, activation)
        self.init_fn = instantiate(inits, init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        # if isinstance(self.activation, activations.ConvMaxout) or isinstance(self.activation, activations.ConvRMSPool):
        #     print 'called'
        #     self.out_shape = self.in_shape
        #     self.out_shape[1] = self.out_shape[1]/2
        # else:
        self.out_shape = self.in_shape

    def init(self):
        self.params = []

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.activation(X)

    def update(self, cost):
        return self.update_fn(self.params, cost)

class BatchNormalize(object):

    def __init__(self, update_fn='nag', e=1e-8):
        self.update_fn = instantiate(updates, update_fn)
        self.e = e

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape

    def init(self):
        naxes = len(self.out_shape)
        if naxes == 2 or naxes == 4:
            dim = self.out_shape[1]
        elif naxes == 3:
            dim = self.out_shape[-1]
        else:
            raise NotImplementedError
        self.g = inits.Constant(c=1.)(dim)
        self.b = inits.Constant(c=0.)(dim)
        self.u = inits.Constant(c=0.)(dim)
        self.s = inits.Constant(c=0.)(dim)
        self.n = sharedX(0.)
        self.params = [self.g, self.b]
        self.other_params = [self.u, self.s, self.n]

    def op(self, state):
        X = self.l_in.op(state=state)
        naxes = len(self.out_shape)
        if naxes == 4: #CNN
            if state['bn_active']:
                u = T.mean(X, axis=[0, 2, 3])
            else:
                u = self.u/self.n
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            if state['bn_active']:
                s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3])
            else:
                s = self.s/self.n
            X = (X - b_u) / T.sqrt(s.dimshuffle('x', 0, 'x', 'x') + self.e)
            X = self.g.dimshuffle('x', 0, 'x', 'x')*X + self.b.dimshuffle('x', 0, 'x', 'x')
        elif naxes == 3: #RNN
            if state['bn_active']:
                u = T.mean(X, axis=[0, 1])
            else:
                u = self.u/self.n
            b_u = u.dimshuffle('x', 'x', 0)
            if state['bn_active']:
                s = T.mean(T.sqr(X - b_u), axis=[0, 1])
            else:
                s = self.s/self.n       
            X = (X - b_u) / T.sqrt(s.dimshuffle('x', 'x', 0) + self.e)
            X = self.g.dimshuffle('x', 'x', 0)*X + self.b.dimshuffle('x', 'x', 0)     
        elif naxes == 2: #FC
            if state['bn_active']:
                u = T.mean(X, axis=0)
            else:
                u = self.u/self.n
            if state['bn_active']:
                s = T.mean(T.sqr(X - u), axis=0)
            else:
                s = self.s/self.n
            X = (X - u) / T.sqrt(s + self.e)
            X = self.g*X + self.b
        else:
            raise NotImplementedError

        if state['infer']:
            self.infer_update = [
                [self.u, self.u + u],
                [self.s, self.s + s],
                [self.n, self.n + 1.]
            ]
        return X

    def update(self, cost):
        return self.update_fn(self.params, cost)

    def reset_update(self):
        return [
                [self.u, self.u * 0.],
                [self.s, self.s * 0.],
                [self.n, self.n * 0.]
            ]

class Dimshuffle(object):

    def __init__(self, shuffle):
        self.shuffle = shuffle

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = [self.in_shape[idx] for idx in self.shuffle]
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return X.dimshuffle(*self.shuffle)

class GaussianNoise(object):

    def __init__(self, scale=0.3):
        self.scale = scale

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape

    def op(self, state):
        X = self.l_in.op(state=state)  
        if state['dropout']:
            X += t_rng.normal(X.shape, std=self.scale, dtype=theano.config.floatX)
        return X

class Slice(object):

    def __init__(self, fn=lambda x:x[-1], shape_fn=lambda x:x[1:]):
        self.fn = fn
        self.shape_fn = shape_fn

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.shape_fn(self.in_shape)
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.fn(X) 

class L2Norm(object):

    def __init__(self, axis=1, e=1e-8):
        self.e = e
        self.axis = axis

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape       
        self.out_shape = self.in_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return X/T.sqrt(T.sum(T.sqr(X), axis=self.axis, keepdims=True) + self.e)

class Op(object):

    def __init__(self, fn, shape_fn):
        self.fn = fn
        self.shape_fn = shape_fn

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.shape_fn(self.in_shape)
        print self.out_shape

    def op(self, state):
        X = self.l_in.op(state=state)
        return self.fn(X) 

class RNN(object):

    def __init__(self, dim=256, activation='rectify', proj_init_fn='orthogonal', rec_init_fn='identity',
                 bias_init_fn='constant', update_fn='nag'):
        self.dim = dim
        self.activation = instantiate(activations, activation)
        self.proj_init_fn = instantiate(inits, proj_init_fn)
        self.rec_init_fn = instantiate(inits, rec_init_fn)
        self.bias_init_fn = instantiate(inits, bias_init_fn)
        self.update_fn = instantiate(updates, update_fn)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape[:-1] + [self.dim]
        print self.out_shape

    def init(self):
        self.w = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.u = self.rec_init_fn((self.dim, self.dim))
        self.b = self.bias_init_fn((self.dim))
        # self.h0 = shared0s((1, self.dim))
        # self.params = [self.w, self.u, self.b, self.h0]
        self.params = [self.w, self.u, self.b]

    def step(self, x_t, h_tm1):
        h_t = self.activation(x_t + T.dot(h_tm1, self.u))
        return h_t

    def op(self, state):
        X = self.l_in.op(state=state)
        x = T.dot(X, self.w) + self.b
        out, _ = theano.scan(self.step,
            sequences=[x],
            # outputs_info=[repeat(self.h0, x.shape[1], axis=0)],
            outputs_info=[T.zeros((x.shape[1], self.dim), dtype=theano.config.floatX)],
        )
        return out

    def update(self, cost):
        return self.update_fn(self.params, cost)

class GRU(object):

    def __init__(self, dim=256, activation='rectify', gate_activation='SteeperSigmoid', proj_init_fn='orthogonal',
                 rec_init_fn='identity', bias_init_fn='constant', update_fn='nag'):
        self.dim = dim
        self.proj_init_fn = instantiate(inits, proj_init_fn)
        self.rec_init_fn = instantiate(inits, rec_init_fn)
        self.bias_init_fn = instantiate(inits, bias_init_fn)
        self.update_fn = instantiate(updates, update_fn)
        self.activation = instantiate(activations, activation)
        self.gate_activation = instantiate(activations, gate_activation)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape[:-1] + [self.dim]
        print self.out_shape

    def init(self):
        self.w_z = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_r = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_h = self.proj_init_fn((self.in_shape[-1], self.dim))

        self.u_z = self.rec_init_fn((self.dim, self.dim))
        self.u_r = self.rec_init_fn((self.dim, self.dim))
        self.u_h = self.rec_init_fn((self.dim, self.dim))

        self.b_z = self.bias_init_fn((self.dim))
        self.b_r = self.bias_init_fn((self.dim))
        self.b_h = self.bias_init_fn((self.dim))

        self.params = [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

    def step(self, xz_t, xr_t, xh_t, h_tm1):
        z = self.gate_activation(xz_t + T.dot(h_tm1, self.u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, self.u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, self.u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def op(self, state):
        X = self.l_in.op(state=state)
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
            outputs_info=[T.zeros((x_h.shape[1], self.dim), dtype=theano.config.floatX)], 
        )
        return out

    def update(self, cost):
        return self.update_fn(self.params, cost)

class LSTM(object):

    def __init__(self, dim=256, activation='rectify', gate_activation='SteeperSigmoid', proj_init_fn='orthogonal',
                 rec_init_fn='identity', bias_init_fn='constant', update_fn='nag'):
        self.dim = dim
        self.proj_init_fn = instantiate(inits, proj_init_fn)
        self.rec_init_fn = instantiate(inits, rec_init_fn)
        self.bias_init_fn = instantiate(inits, bias_init_fn)
        self.update_fn = instantiate(updates, update_fn)
        self.activation = instantiate(activations, activation)
        self.gate_activation = instantiate(activations, gate_activation)

    def connect(self, l_in):
        self.l_in = l_in
        self.in_shape = l_in.out_shape
        self.out_shape = self.in_shape[:-1] + [self.dim]
        print self.out_shape

    def init(self):

        self.w_i = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_f = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_o = self.proj_init_fn((self.in_shape[-1], self.dim))
        self.w_c = self.proj_init_fn((self.in_shape[-1], self.dim))

        self.b_i = self.bias_init_fn((self.dim))
        self.b_f = self.bias_init_fn((self.dim))
        self.b_o = self.bias_init_fn((self.dim))
        self.b_c = self.bias_init_fn((self.dim))

        self.u_i = self.rec_init_fn((self.dim, self.dim))
        self.u_f = self.rec_init_fn((self.dim, self.dim))
        self.u_o = self.rec_init_fn((self.dim, self.dim))
        self.u_c = self.rec_init_fn((self.dim, self.dim))

        self.params = [self.w_i, self.w_f, self.w_o, self.w_c, 
            self.u_i, self.u_f, self.u_o, self.u_c,  
            self.b_i, self.b_f, self.b_o, self.b_c] 

    def step(self, xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1, u_i, u_f, u_o, u_c):
        i_t = self.gate_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.gate_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.gate_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def op(self, state):
        X = self.l_in.op(state=state)
        x_i = T.dot(X, self.w_i) + self.b_i
        x_f = T.dot(X, self.w_f) + self.b_f
        x_o = T.dot(X, self.w_o) + self.b_o
        x_c = T.dot(X, self.w_c) + self.b_c
        [out, cells], _ = theano.scan(self.step, 
            sequences=[x_i, x_f, x_o, x_c], 
            outputs_info=[
                T.zeros((x_i.shape[1], self.dim), dtype=theano.config.floatX), 
                T.zeros((x_i.shape[1], self.dim), dtype=theano.config.floatX)
            ], 
            non_sequences=[self.u_i, self.u_f, self.u_o, self.u_c],
        )
        return out
