"""Shared path handling for the independence-audit scripts.

Every script resolves each input in this order:

1. an explicit command-line option;
2. the environment variable listed in ``ENV_VARS``;
3. ``<repository root>/codes/data/<file name>``.

Nothing here is machine specific: the defaults follow the repository layout
documented in the top-level README.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
DATA_DIR = os.path.join(ROOT, "codes", "data")

ENV_VARS = {
    "data.hdf5": "PPG2ABP_DATA_HDF5",
    "candidates.p": "PPG2ABP_CANDIDATES",
    "meta9.p": "PPG2ABP_META9",
    "train9.p": "PPG2ABP_TRAIN9",
    "val9.p": "PPG2ABP_VAL9",
    "test_original.p": "PPG2ABP_TEST_ORIGINAL",
}


def default_path(name):
    """Environment override when set, otherwise ``codes/data/<name>``."""
    override = os.environ.get(ENV_VARS.get(name, ""), "")
    return override if override else os.path.join(DATA_DIR, name)


def require(path, name):
    """Fail with an actionable message instead of a bare IOError."""
    if not os.path.isfile(path):
        raise SystemExit(
            "missing input {0}\n  looked for: {1}\n  pass the matching option or set {2}".format(
                name, path, ENV_VARS.get(name, "the environment override")))
    return path
