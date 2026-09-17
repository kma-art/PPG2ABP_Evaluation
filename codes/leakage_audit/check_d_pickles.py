# -*- coding: utf-8 -*-
"""Check D — the overlap survives at the level of the arrays actually fed to the model.

Checks G and H work on data.hdf5. This one works on the pickles the training
and evaluation code consumes: train9.p, val9.p and test_original.p, all three
produced locally by running the authors' data_handling.py over their data.hdf5.

The training and test parts are normalized with different min/max, so an exact
byte comparison would find nothing. Segments are therefore compared through a
fingerprint that is invariant to scale and offset (per-channel standardization,
rounded), and a sample of the matches is then verified strictly by fitting the
affine relation and reporting the residual. A negative control uses random
pairs.

Fold 9 is used here only because those are the files at hand; check_h shows the
share does not depend on the fold.
"""

import argparse
import gc
import hashlib
import pickle
import random

import numpy as np

from _common import default_path, require


def fingerprints(x, y):
    """Fingerprint of a (PPG, ABP) pair, invariant to per-channel scale and offset."""
    out = []
    for i in range(x.shape[0]):
        digest = hashlib.sha256()
        for array in (x[i].reshape(-1), y[i].reshape(-1)):
            a = array.astype(np.float64)
            sd = a.std()
            z = np.zeros_like(a) if sd == 0 else (a - a.mean()) / sd
            digest.update(np.round(z, 9).tobytes())
        out.append(digest.digest())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--train9', default=default_path('train9.p'), metavar='PATH')
    ap.add_argument('--val9', default=default_path('val9.p'), metavar='PATH')
    ap.add_argument('--test-original', default=default_path('test_original.p'), metavar='PATH')
    args = ap.parse_args()

    print('loading train9.p ...')
    with open(require(args.train9, 'train9.p'), 'rb') as stream:
        train = pickle.load(stream)
    x_train, y_train = np.asarray(train['X_train']), np.asarray(train['Y_train'])
    print('  X_train %s  Y_train %s' % (x_train.shape, y_train.shape))
    train_index = {}
    for i, key in enumerate(fingerprints(x_train, y_train)):
        train_index.setdefault(key, i)
    print('  distinct fingerprints in train9: %d of %d' % (len(train_index), x_train.shape[0]))

    print('loading val9.p ...')
    with open(require(args.val9, 'val9.p'), 'rb') as stream:
        validation = pickle.load(stream)
    x_validation, y_validation = np.asarray(validation['X_val']), np.asarray(validation['Y_val'])
    validation_index = {}
    for i, key in enumerate(fingerprints(x_validation, y_validation)):
        validation_index.setdefault(key, i)
    print('  distinct fingerprints in val9: %d of %d'
          % (len(validation_index), x_validation.shape[0]))
    del validation
    gc.collect()

    print('loading test_original.p ...')
    with open(require(args.test_original, 'test_original.p'), 'rb') as stream:
        test = pickle.load(stream)
    x_test, y_test = np.asarray(test['X_test']), np.asarray(test['Y_test'])
    print('  X_test %s  Y_test %s' % (x_test.shape, y_test.shape))
    test_fingerprints = fingerprints(x_test, y_test)
    del test
    gc.collect()

    hits_train = [(i, train_index[k]) for i, k in enumerate(test_fingerprints) if k in train_index]
    hits_validation = [(i, validation_index[k]) for i, k in enumerate(test_fingerprints)
                       if k in validation_index]
    hits_any = sum(1 for k in test_fingerprints if k in train_index or k in validation_index)
    n = len(test_fingerprints)
    print()
    print('test_original.p against the fold-9 training files, N = %d' % n)
    print('  matches a segment of train9.p : %6d (%.1f %%)'
          % (len(hits_train), 100.0 * len(hits_train) / n))
    print('  matches a segment of val9.p   : %6d (%.1f %%)'
          % (len(hits_validation), 100.0 * len(hits_validation) / n))
    print('  matches train9 or val9        : %6d (%.1f %%)' % (hits_any, 100.0 * hits_any / n))

    random.seed(11)
    print('\nstrict verification of 8 matches: affine relation between the segments')
    for test_i, train_i in random.sample(hits_train, 8):
        for name, a, b in (('PPG', x_test[test_i].reshape(-1), x_train[train_i].reshape(-1)),
                           ('ABP', y_test[test_i].reshape(-1), y_train[train_i].reshape(-1))):
            slope, offset = np.polyfit(b.astype(np.float64), a.astype(np.float64), 1)
            residual = np.max(np.abs(a - (slope * b + offset)))
            print('  test #%-6d <-> train #%-6d  %s: a=%.12f b=%.12f residual=%.2e'
                  % (test_i, train_i, name, slope, offset, residual))

    print('\nnegative control: 5 random pairs')
    for _ in range(5):
        test_i = random.randrange(x_test.shape[0])
        train_i = random.randrange(x_train.shape[0])
        a = x_test[test_i].reshape(-1).astype(np.float64)
        b = x_train[train_i].reshape(-1).astype(np.float64)
        slope, offset = np.polyfit(b, a, 1)
        print('  test #%-6d <-> train #%-6d  PPG residual=%.3e'
              % (test_i, train_i, np.max(np.abs(a - (slope * b + offset)))))


if __name__ == '__main__':
    main()
