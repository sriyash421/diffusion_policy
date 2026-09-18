"""Registers the one custom marker used here, so `-m 'not slow'` runs without a warning.

`slow` means "loads data/pusht_cchi_v7_replay.zarr or trains a few hundred steps". Everything
unmarked runs on CPU stubs in well under a second.
"""


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slow: loads the PushT zarr or runs an optimisation loop')
