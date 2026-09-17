# -*- coding: utf-8 -*-
"""Check G — duplicate episodes in the published data.hdf5 and the resulting
overlap between the training/validation rows and the test rows.

This is the headline check. It reads only the file the original authors
distribute (Google Drive id 1IxN2sX2TX0uK6CFDh8eudb8haz3RlF7X, linked from
codes/README.md and cells 0 and 6 of their notebook) and applies their own
split rule from codes/data_handling.py: rows 0-99999 are used for training and
validation, rows 100000-127259 form the test part.

Reported:
  * SHA-256 of the file, compared with the value we downloaded twice;
  * how many rows are byte-identical repeats of another row;
  * how many test rows have a byte-identical copy among rows 0-99999,
    at 1024 samples (the length the model consumes) and at the stored 1250;
  * whether the min/max of rows 0-99999 reproduce the authors' meta9.p;
  * six random matches re-compared byte by byte, plus a negative control.

Runtime is dominated by hashing 2.5 GB; expect several minutes.
"""

import argparse
import hashlib
import pickle
import random

import h5py
import numpy as np

from _common import default_path, require

# SHA-256 of data.hdf5 as distributed by the original authors. Two independent
# downloads (2025-12-04 and 2026-09-11) produced this same digest.
PUBLISHED_SHA256 = '4d1a3085cd91f8725682c9290c0527f28651f6ce290d6d2a2da796b316adbf25'

TEST_START = 100000
# validation_data_start[9] in data_handling.py; only the train/validation
# boundary depends on the fold, never the 0-99999 versus 100000+ split.
VAL_LO, VAL_HI = 80000, 90000


def role(index):
    if index >= TEST_START:
        return 'test'
    return 'validation' if VAL_LO <= index < VAL_HI else 'train'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--hdf5', default=default_path('data.hdf5'), metavar='PATH',
                    help='data.hdf5 published by the original authors')
    ap.add_argument('--meta9', default=default_path('meta9.p'), metavar='PATH',
                    help="meta9.p published by the original authors")
    ap.add_argument('--skip-hash', action='store_true',
                    help='skip the SHA-256 pass over the 2.5 GB file')
    args = ap.parse_args()

    hdf5_path = require(args.hdf5, 'data.hdf5')
    meta_path = require(args.meta9, 'meta9.p')

    if args.skip_hash:
        print('sha256: skipped (--skip-hash)')
    else:
        digest = hashlib.sha256()
        with open(hdf5_path, 'rb') as stream:
            while True:
                block = stream.read(1 << 24)
                if not block:
                    break
                digest.update(block)
        actual = digest.hexdigest()
        print('sha256 of this file : %s' % actual)
        print('sha256 as published : %s' % PUBLISHED_SHA256)
        print('MATCH: %s' % (actual == PUBLISHED_SHA256))

    handle = h5py.File(hdf5_path, 'r')
    data = handle['data']
    n = data.shape[0]
    print('\nshape=%s dtype=%s' % (data.shape, data.dtype))

    groups_1024, groups_1250 = {}, {}
    abp_lo = abp_hi = ppg_lo = ppg_hi = None
    for start in range(0, n, 2000):
        batch = data[start:min(start + 2000, n)]
        abp = np.asarray(batch[:, 0, :])
        ppg = np.asarray(batch[:, 1, :])
        if start < TEST_START:
            end = min(start + 2000, TEST_START) - start
            if end > 0:
                abp_lo = abp[:end].min() if abp_lo is None else min(abp_lo, abp[:end].min())
                abp_hi = abp[:end].max() if abp_hi is None else max(abp_hi, abp[:end].max())
                ppg_lo = ppg[:end].min() if ppg_lo is None else min(ppg_lo, ppg[:end].min())
                ppg_hi = ppg[:end].max() if ppg_hi is None else max(ppg_hi, ppg[:end].max())
        for offset in range(batch.shape[0]):
            index = start + offset
            key_1024 = hashlib.sha256(
                np.ascontiguousarray(abp[offset][:1024]).tobytes()
                + np.ascontiguousarray(ppg[offset][:1024]).tobytes()).digest()
            key_1250 = hashlib.sha256(
                np.ascontiguousarray(abp[offset]).tobytes()
                + np.ascontiguousarray(ppg[offset]).tobytes()).digest()
            groups_1024.setdefault(key_1024, []).append(index)
            groups_1250.setdefault(key_1250, []).append(index)

    n_test = n - TEST_START
    matched_pairs = []
    for label, groups in (('first 1024 samples', groups_1024),
                          ('all 1250 stored samples', groups_1250)):
        multiplicity = {}
        for indices in groups.values():
            multiplicity[len(indices)] = multiplicity.get(len(indices), 0) + 1
        in_train = in_validation = in_either = 0
        pairs = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            test_rows = [i for i in indices if role(i) == 'test']
            train_rows = [i for i in indices if role(i) == 'train']
            validation_rows = [i for i in indices if role(i) == 'validation']
            for _ in test_rows:
                in_train += bool(train_rows)
                in_validation += bool(validation_rows)
                in_either += bool(train_rows or validation_rows)
            if train_rows and test_rows:
                pairs.append((train_rows[0], test_rows[0]))
        print('\n--- %s ---' % label)
        print('  distinct segments: %d,  multiplicities: %s'
              % (len(groups), dict(sorted(multiplicity.items()))))
        print('  test row with a copy in train      : %6d (%.1f %%)'
              % (in_train, 100.0 * in_train / n_test))
        print('  test row with a copy in validation : %6d (%.1f %%)'
              % (in_validation, 100.0 * in_validation / n_test))
        print('  test row with a copy in rows 0-99999: %6d (%.1f %%)'
              % (in_either, 100.0 * in_either / n_test))
        if label.startswith('first'):
            matched_pairs = pairs

    with open(meta_path, 'rb') as stream:
        meta = pickle.load(stream)
    print("\nstatistics of rows 0-99999 against the authors' meta9.p:")
    print('  ABP  computed %.17g / %.17g   meta9 %.17g / %.17g'
          % (abp_lo, abp_hi, meta['min_abp'], meta['max_abp']))
    print('  PPG  computed %.17g / %.17g   meta9 %.17g / %.17g'
          % (ppg_lo, ppg_hi, meta['min_ppg'], meta['max_ppg']))
    print('  match: ABP %s, PPG %s'
          % (abp_lo == meta['min_abp'] and abp_hi == meta['max_abp'],
             ppg_lo == meta['min_ppg'] and ppg_hi == meta['max_ppg']))

    random.seed(5)
    confirmed = sum(1 for train_row, test_row in random.sample(matched_pairs, 6)
                    if np.array_equal(np.asarray(data[train_row]), np.asarray(data[test_row])))
    print('\nbyte-for-byte re-comparison of 6 matched train/test pairs: %d of 6' % confirmed)
    negative = sum(1 for _ in range(6)
                   if np.array_equal(np.asarray(data[random.randrange(0, TEST_START)]),
                                     np.asarray(data[random.randrange(TEST_START, n)])))
    print('negative control, 6 random pairs: %d of 6 equal (expected 0)' % negative)
    handle.close()


if __name__ == '__main__':
    main()
