import numpy as np
import costs

from utils import numpy_array
import string
from collections import Counter

def one_hot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def flatten(l):
    return [item for sublist in l for item in sublist]

def lbf(l,b):
    return [el for el, condition in zip(l, b) if condition]

def list_index(l, idxs):
    return [l[idx] for idx in idxs]

punctuation = set(string.punctuation)
punctuation.add('\n')
punctuation.add('\t')
punctuation.add('')

def merge_tokens(tokens):
    merged = [tokens[0]]
    for t in tokens[1:]:
        m = merged[-1]
        if t in punctuation and m[-1] == t:
            merged[-1] += t
            m += t
        elif m.count(m[0]) == len(m) and len(m) > 1 and m[0] in punctuation:
            merged[-1] = m[:4]
            merged.append(t)
        else:
            merged.append(t)
    return merged

# def tokenize(text):
#     tokenized = []
#     w = ''
#     for t in text:
#         if t in punctuation:
#             tokenized.append(w)
#             tokenized.append(t)
#             w = ''
#         elif t == ' ':
#             tokenized.append(w)
#             w = ''
#         else:
#             w += t
#     if w != '':
#         tokenized.append(w)
#     tokenized = [token for token in tokenized if token]
#     # Don't merge if already empty
#     if len(tokenized) > 0:
#         tokenized = merge_tokens(tokenized)
#     return tokenized

def tokenize2(text):
    tokenized = []
    text = text.replace('\n', ' \n ')
    text = text.replace('\t', ' \t ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.replace(':', ' : ')
    text = text.replace(';', ' ; ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')
    text = text.replace('>', ' > ')
    text = text.replace('<', ' < ')
    text = text.replace('"', ' " ')
    text = text.replace("(", ' ( ')
    text = text.replace(")", ' ) ')
    text = text.replace("[", ' [ ')
    text = text.replace("]", ' ] ')
    text = text.replace("'s", " 's")
    text = text.replace("'t", " 't")
    text = text.replace("'re"," 're")
    text = text.replace("'ve"," 've")
    text = text.replace("'ll"," 'll")
    text = text.replace("'m"," 'm")
    text = text.replace("'d"," 'd")
    text = text.replace(" '", " `")
    text = text.replace("'", " ' ")
    tokens = text.split(' ')
    tokens = [token for token in tokens if token]
    return tokens

# import re

# chars = set(string.punctuation)
# chars.add('\t')
# chars.add('\n')
# chars.remove("'")
# chars = list(chars)
# rep = dict(zip(chars, [' '+c+' ' for c in chars]))

# # use these three lines to do the replacement
# rep = dict((re.escape(k), v) for k, v in rep.iteritems())
# print "|".join(rep.keys())
# pattern = re.compile("|".join(rep.keys()))
# # text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

punc = set(string.punctuation)
punc.add('\t')
punc.add('\n')
punc.remove("'")
contractions = ["'s", "'t", "'re", "'ve", "'ll", "'m", "'d"]

def tokenize(text):
    tokenized = []
    for c in punc:
        text = text.replace(c, ' '+c+' ')
    for c in contractions:
        text = text.replace(c, ' |'+c)
    text = text.replace(" |'", " `")
    text = text.replace("'", " ' ")
    tokens = text.split(' ')
    tokens = [token for token in tokens if token]
    return tokens

# def token_encoder(texts, max_features=9997, min_df=10):
#     df = {}
#     for text in texts:
#         tokens = set(text)
#         for token in tokens:
#             if token in df:
#                 df[token] += 1
#             else:
#                 df[token] = 1
#     k, v = df.keys(), np.asarray(df.values())
#     valid = v >= min_df
#     k = lbf(k, valid)
#     v = v[valid]
#     sort_mask = np.argsort(v)[::-1]
#     k = list_index(k, sort_mask)[:max_features]
#     v = v[sort_mask][:max_features]
#     xtoi = dict(zip(k, range(3, len(k)+3)))
#     return xtoi

def token_encoder(texts, character=False, max_features=9997, min_df=10):
    df = {}
    for text in texts:
        if character:
            text = list(text)
        else:
            text = tokenize(text)
        tokens = set(text)
        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1
    k, v = df.keys(), np.asarray(df.values())
    valid = v >= min_df
    k = lbf(k, valid)
    v = v[valid]
    sort_mask = np.argsort(v)[::-1]
    k = list_index(k, sort_mask)[:max_features]
    v = v[sort_mask][:max_features]
    xtoi = dict(zip(k, range(3, len(k)+3)))
    return xtoi

def standardize_X(shape, X):
	if not numpy_array(X):
		X = np.asarray(X)

	if len(shape) == 4 and len(X.shape) == 2:
		return X.reshape(-1, shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2)
	else:
		return X

def standardize_Y(shape, Y):
	if not numpy_array(Y):
		Y = np.asarray(Y)
	if len(Y.shape) == 1:
		Y = Y.reshape(-1, 1)
	if len(Y.shape) == 2 and len(shape) == 2:
		if shape[-1] != Y.shape[-1]:
			return one_hot(Y, n=shape[-1])
		else:
			return Y
	else:
		return Y

def one_hot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh
    
class Tokenizer(object):
    """
    For converting lists of text into tokens used by Passage models.
    max_features sets the maximum number of tokens (all others are mapped to UNK)
    min_df sets the minimum number of documents a token must appear in to not get mapped to UNK
    lowercase controls whether the text is lowercased or not
    character sets whether the tokenizer works on a character or word level

    Usage:
    >>> from passage.preprocessing import Tokenizer
    >>> example_text = ['This. is.', 'Example TEXT', 'is text']
    >>> tokenizer = Tokenizer(min_df=1, lowercase=True, character=False)
    >>> tokenized = tokenizer.fit_transform(example_text)
    >>> tokenized
    [[7, 5, 3, 5], [6, 4], [3, 4]]
    >>> tokenizer.inverse_transform(tokenized)
    ['this . is .', 'example text', 'is text']
    """

    def __init__(self, max_features=9997, min_df=10, lowercase=True, character=False):
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.character = character

    def fit(self, texts):
        if self.lowercase:
            texts = [text.lower() for text in texts]
        self.encoder = token_encoder(texts, character=self.character, max_features=self.max_features-3, min_df=self.min_df)
        self.encoder['PAD'] = 0
        self.encoder['END'] = 1
        self.encoder['UNK'] = 2
        self.decoder = dict(zip(self.encoder.values(), self.encoder.keys()))
        self.n_features = len(self.encoder)
        return self

    def transform(self, texts):
        if self.lowercase:
            texts = [text.lower() for text in texts]
        if self.character:
            texts = [list(text) for text in texts]
        else:
            texts = [tokenize(text) for text in texts]
        tokens = [[self.encoder.get(token, 2) for token in text] for text in texts]
        return tokens

    def fit_transform(self, texts):
        self.fit(texts)
        tokens = self.transform(texts)
        return tokens

    def inverse_transform(self, codes):
        if self.character:
            joiner = ''
        else:
            joiner = ' '
        return [joiner.join([self.decoder[token] for token in code]) for code in codes]

class LenFilter(object):

    def __init__(self, max_len=1000, min_max_len=100, percentile=99):
        self.max_len = max_len
        self.percentile = percentile
        self.min_max_len = min_max_len

    def filter(self, *data):
        lens = [len(seq) for seq in data[0]]
        if self.percentile > 0:
            max_len = np.percentile(lens, self.percentile)
            max_len = np.clip(max_len, self.min_max_len, self.max_len)
        else:
            max_len = self.max_len
        valid_idxs = [i for i, l in enumerate(lens) if l <= max_len]
        if len(data) == 1:
            return list_index(data[0], valid_idxs)
        else:
            return tuple([list_index(d, valid_idxs) for d in data])

