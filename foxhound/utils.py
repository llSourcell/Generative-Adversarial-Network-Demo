import inspect
import types
import numpy as np
from sklearn import utils as skutils

from rng import np_rng

def numpy_array(X):
    return type(X).__module__ == np.__name__

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data]) 

def iter_indices(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1
    for b in range(batches):
        yield b

def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], basestring):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

def case_insensitive_import(module, name):
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name.lower()])

def classes_of(module):
	return tuple(x[1] for x in inspect.getmembers(module, inspect.isclass))

def instantiate(module, obj):
    if isinstance(obj, basestring):
        obj = case_insensitive_import(module, obj)
        if isinstance(obj, types.FunctionType):
            return obj
        else:
            return obj()
    elif isinstance(obj, classes_of(module)):
    	return obj
    elif inspect.isfunction(obj):
    	return obj
    else:
        raise TypeError