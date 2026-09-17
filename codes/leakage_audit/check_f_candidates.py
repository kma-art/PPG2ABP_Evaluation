# -*- coding: utf-8 -*-
"""Check F — repeated entries in the authors' candidates.p.

candidates.p is the direct output of downsample_data() in the authors'
codes/data_processing.py, published on Google Drive (id
1u-yvkqJmmrYCbuSw3lnS8mIIcEHRQxFD, linked from cell 11 of their notebook).
Each entry is a triple [file_no, record_no, episode_start] naming one episode
of the Kachuee MIMIC-III subset.

The script counts how often the same triple appears, and compares the overlap
this would produce after a shuffle and an index cut with the value check_g
measures directly on data.hdf5.
"""

import argparse
import hashlib
import io
import pickle
from collections import Counter

from _common import default_path, require

# SHA-256 of candidates.p as distributed by the original authors.
PUBLISHED_SHA256 = '4c22fe9d3ec1268cdaefe3220200e41b799e6702f8d231e51582ec82c3389a38'
# Test rows in data.hdf5: 127260 - 100000.
N_TEST = 27260
# Value that check_g measures byte for byte on data.hdf5, quoted for comparison.
OBSERVED_OVERLAP = 16265


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--candidates', default=default_path('candidates.p'), metavar='PATH',
                    help='candidates.p published by the original authors')
    args = ap.parse_args()

    path = require(args.candidates, 'candidates.p')
    raw = io.open(path, 'rb').read()
    print('file   : %s' % path)
    print('size   : %d bytes' % len(raw))
    print('sha256 : %s' % hashlib.sha256(raw).hexdigest())
    print('as published: %s' % PUBLISHED_SHA256)

    try:
        obj = pickle.loads(raw)
    except Exception:
        obj = pickle.loads(raw, encoding='latin1')

    print('\ntop-level type: %s' % type(obj).__name__)
    print('entries       : %d' % len(obj))
    print('first three   : %s' % [list(x) for x in obj[:3]])

    items = [tuple(int(v) for v in x) for x in obj]
    counts = Counter(items)
    multiplicity = Counter(counts.values())
    print('\ntotal entries            : %d' % len(items))
    print('distinct triples         : %d' % len(counts))
    print('multiplicity distribution: %s' % dict(sorted(multiplicity.items())))

    extra = len(items) - len(counts)
    print('surplus copies           : %d (%.1f %% of all entries)'
          % (extra, 100.0 * extra / len(items)))

    exactly_twice = multiplicity.get(2, 0)
    total = len(items)
    share_test = float(N_TEST) / total
    expected = 2 * exactly_twice * share_test * (1 - share_test)
    print('\nEach duplicated episode ends up on both sides of the index cut with')
    print('probability 2p(1-p), p = %d/%d. Expected test rows with a copy outside' % (N_TEST, total))
    print('the test part: %.0f (%.1f %%).' % (expected, 100.0 * expected / N_TEST))
    print('check_g measures on data.hdf5: %d (%.1f %%); difference %.2f %%.'
          % (OBSERVED_OVERLAP, 100.0 * OBSERVED_OVERLAP / N_TEST,
             100.0 * abs(expected - OBSERVED_OVERLAP) / OBSERVED_OVERLAP))

    print('\nexamples of repeated triples [file_no, record_no, episode_start]:')
    for triple, count in [(t, c) for t, c in counts.items() if c > 1][:5]:
        print('   %s  -> %d times' % (list(triple), count))


if __name__ == '__main__':
    main()
