# -*- coding: utf-8 -*-
"""Check I — candidates.p has exactly the structure the two-stage selection predicts.

downsample_data() fills its candidate list in two passes. The DBP pass appends
episodes and registers each one in a look-up table. The SBP pass looks each
episode up in that table inside a try/except and appends it when the lookup
succeeds, that is when the episode is already in the list:

    try:
        dumi = lut[...][...][...]
    except:
        sbps_dict[sbp].pop(indix)
        continue
    candidates.append([...])

If that is what happens, the published list must split into a prefix of
pairwise distinct triples (the DBP pass) followed by a suffix in which every
triple repeats one from the prefix (the SBP pass), and no triple can occur more
than twice. This script tests that prediction on the authors' file.
"""

import argparse
import pickle
from collections import Counter

from _common import default_path, require


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--candidates', default=default_path('candidates.p'), metavar='PATH',
                    help='candidates.p published by the original authors')
    args = ap.parse_args()

    with open(require(args.candidates, 'candidates.p'), 'rb') as stream:
        obj = pickle.load(stream)
    items = [tuple(int(v) for v in x) for x in obj]

    counts = Counter(items)
    distinct = len(counts)
    multiplicity = Counter(counts.values())
    print('entries %d, distinct triples %d, multiplicities %s'
          % (len(items), distinct, dict(sorted(multiplicity.items()))))

    seen = set()
    boundary = None
    for position, triple in enumerate(items):
        if triple in seen:
            boundary = position
            break
        seen.add(triple)
    print('first repeat at position %s' % boundary)
    print('number of distinct triples %d' % distinct)
    print('the two coincide: %s' % (boundary == distinct))

    prefix, suffix = items[:boundary], items[boundary:]
    print()
    print('--- prefix (predicted: the DBP pass, all triples distinct) ---')
    print('  length %d, distinct %d, repeats inside: %d'
          % (len(prefix), len(set(prefix)), len(prefix) - len(set(prefix))))

    prefix_set = set(prefix)
    from_prefix = sum(1 for triple in suffix if triple in prefix_set)
    print()
    print('--- suffix (predicted: the SBP pass, every triple repeats the prefix) ---')
    print('  length %d, distinct %d' % (len(suffix), len(set(suffix))))
    print('  repeating a prefix triple: %d of %d (%.2f %%)'
          % (from_prefix, len(suffix), 100.0 * from_prefix / len(suffix)))
    print('  repeats inside the suffix itself: %d' % (len(suffix) - len(set(suffix))))

    print()
    print('total: %d + %d = %d' % (len(prefix), len(suffix), len(prefix) + len(suffix)))
    print('no triple occurs more than twice: %s' % (max(counts.values()) == 2))


if __name__ == '__main__':
    main()
