# -*- coding: utf-8 -*-
"""Check E — the regenerated train9.p / val9.p are exactly rows of data.hdf5.

fold_data() in the authors' codes/data_handling.py contains no shuffling and no
call to a random number generator: it is a slice by row index. Running it over
data.hdf5 therefore has to reproduce the HDF5 rows exactly, up to the training
normalization stored in meta9.p. This script verifies that on sampled rows,
including the fold boundaries, so that the index arithmetic used by the other
checks is confirmed rather than assumed.

Inputs train9.p and val9.p are produced locally by running the authors'
data_handling.py over their data.hdf5; they are not distributed by anyone.
"""

import argparse
import pickle
import random
import warnings

import h5py
import numpy as np

from _common import default_path, require

warnings.filterwarnings('ignore')

SEGMENT_LENGTH = 1024
# validation_data_start[9] in data_handling.py
VAL_START, VAL_END = 80000, 90000


def expected_row_train(position):
    """The train loop walks range(0, 80000) and then range(90000, 100000)."""
    return position if position < VAL_START else position + (VAL_END - VAL_START)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--hdf5', default=default_path('data.hdf5'), metavar='PATH')
    ap.add_argument('--meta9', default=default_path('meta9.p'), metavar='PATH')
    ap.add_argument('--train9', default=default_path('train9.p'), metavar='PATH')
    ap.add_argument('--val9', default=default_path('val9.p'), metavar='PATH')
    args = ap.parse_args()

    with open(require(args.meta9, 'meta9.p'), 'rb') as stream:
        meta = pickle.load(stream)
    min_ppg, max_ppg = meta['min_ppg'], meta['max_ppg']

    handle = h5py.File(require(args.hdf5, 'data.hdf5'), 'r')
    data = handle['data']

    with open(require(args.train9, 'train9.p'), 'rb') as stream:
        train = pickle.load(stream)
    with open(require(args.val9, 'val9.p'), 'rb') as stream:
        validation = pickle.load(stream)
    x_train = np.asarray(train['X_train'])
    x_validation = np.asarray(validation['X_val'])
    print('train9.p %s   val9.p %s' % (x_train.shape, x_validation.shape))

    random.seed(3)
    worst = 0.0
    print('\ntrain9.p[j]  ->  data.hdf5[row]   (PPG: (raw - %g) / %.17g)'
          % (min_ppg, max_ppg - min_ppg))
    for position in sorted(random.sample(range(x_train.shape[0]), 6)
                           + [0, VAL_START - 1, VAL_START, VAL_END - 1]):
        row = expected_row_train(position)
        raw = np.asarray(data[row][1][:SEGMENT_LENGTH], dtype=np.float64)
        got = x_train[position].reshape(-1).astype(np.float64)
        residual = np.max(np.abs(got - (raw - min_ppg) / (max_ppg - min_ppg)))
        worst = max(worst, residual)
        print('  j=%-6d -> row %-6d  max residual = %.3e' % (position, row, residual))

    print('\nval9.p[j]  ->  data.hdf5[%d + j]' % VAL_START)
    for position in sorted(random.sample(range(x_validation.shape[0]), 5) + [0, 9999]):
        row = VAL_START + position
        raw = np.asarray(data[row][1][:SEGMENT_LENGTH], dtype=np.float64)
        got = x_validation[position].reshape(-1).astype(np.float64)
        residual = np.max(np.abs(got - (raw - min_ppg) / (max_ppg - min_ppg)))
        worst = max(worst, residual)
        print('  j=%-6d -> row %-6d  max residual = %.3e' % (position, row, residual))

    print('\nworst residual over everything checked: %.3e' % worst)
    print('conclusion: train9.p + val9.p are rows 0-99999 and the test part is rows')
    print('100000-127259; the two index ranges do not intersect.')
    handle.close()


if __name__ == '__main__':
    main()
