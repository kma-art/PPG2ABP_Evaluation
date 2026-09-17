# -*- coding: utf-8 -*-
"""Check H — does the overlap depend on which cross-validation fold is used?

fold_data() in the authors' codes/data_handling.py assigns rows 0-99999 to
training and validation for every fold_id and moves only the boundary between
those two parts; rows 100000+ are the test part in all ten cases. This script
measures the overlap separately for each fold, so the reported share does not
rest on any assumption about which fold produced the released weights.

Only the published data.hdf5 is read.
"""

import argparse
import hashlib

import h5py
import numpy as np

from _common import default_path, require

TEST_START = 100000
# validation_data_start, copied verbatim from the authors' data_handling.py
VALIDATION_DATA_START = {0: 90000, 1: 0, 2: 10000, 3: 20000, 4: 30000,
                         5: 40000, 6: 50000, 7: 60000, 8: 70000, 9: 80000}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--hdf5', default=default_path('data.hdf5'), metavar='PATH',
                    help='data.hdf5 published by the original authors')
    args = ap.parse_args()

    handle = h5py.File(require(args.hdf5, 'data.hdf5'), 'r')
    data = handle['data']
    n = data.shape[0]

    groups = {}
    for start in range(0, n, 2000):
        batch = np.asarray(data[start:min(start + 2000, n)])
        for offset in range(batch.shape[0]):
            key = hashlib.sha256(
                np.ascontiguousarray(batch[offset, 0, :1024]).tobytes()
                + np.ascontiguousarray(batch[offset, 1, :1024]).tobytes()).digest()
            groups.setdefault(key, []).append(start + offset)
    handle.close()

    duplicated = [indices for indices in groups.values() if len(indices) > 1]
    n_test = n - TEST_START
    print('rows %d, test rows %d, groups containing a repeat %d'
          % (n, n_test, len(duplicated)))
    print()
    print('fold | validation rows | copy in train | copy in val | copy in rows 0-99999')
    print('-----|-----------------|---------------|-------------|---------------------')

    for fold in range(10):
        val_start = VALIDATION_DATA_START[fold]
        val_end = val_start + 10000
        in_train = in_validation = in_either = 0
        for indices in duplicated:
            test_rows = [i for i in indices if i >= TEST_START]
            if not test_rows:
                continue
            has_train = any(i < TEST_START and not (val_start <= i < val_end) for i in indices)
            has_validation = any(val_start <= i < val_end for i in indices)
            for _ in test_rows:
                in_train += has_train
                in_validation += has_validation
                in_either += (has_train or has_validation)
        print('  %d  | %6d-%-8d | %5d (%4.1f %%) | %4d (%3.1f %%) |    %5d (%4.1f %%)'
              % (fold, val_start, val_end - 1,
                 in_train, 100.0 * in_train / n_test,
                 in_validation, 100.0 * in_validation / n_test,
                 in_either, 100.0 * in_either / n_test))

    print()
    print('The last column is identical for all ten folds: rows 0-99999 are consumed by')
    print('training and validation whatever fold_id is chosen, and only the boundary moves.')


if __name__ == '__main__':
    main()
