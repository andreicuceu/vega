"""Pytest configuration shared by editable-checkout and installed-wheel test runs."""

import gc
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]

# These modules inspect the source tree (package sources, ``bin/`` scripts, ``docs/``) and
# cannot run when the tests are staged outside a checkout to exercise an installed wheel.
SOURCE_TREE_MODULES = [
    "test_package_metadata.py",
    "test_redshift_weights.py",
    "test_documentation_headings.py",
]

collect_ignore = [] if (SOURCE / "vega" / "redshift_weights.py").is_file() else SOURCE_TREE_MODULES


@pytest.fixture(autouse=True)
def collect_garbage():
    """Run a full garbage collection after each test.

    A ``VegaInterface`` lies in reference cycles (its ``Minimizer`` holds the bound ``chi2``,
    the analytic marginalization holds the interface), so reference counting cannot free it.
    The interfaces of the full test configurations hold ~1.3 GB each, but the cyclic collector
    is triggered by object counts rather than memory, so without an explicit collection
    several discarded interfaces accumulate and exceed the 16 GB of the CI runners.

    Yields
    ------
    None
        Control returns to the test; the collection runs at teardown.
    """
    yield
    gc.collect()
