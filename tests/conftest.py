"""Pytest configuration shared by editable-checkout and installed-wheel test runs."""

from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]

# These modules inspect the source tree (package sources, ``bin/`` scripts, ``docs/``) and
# cannot run when the tests are staged outside a checkout to exercise an installed wheel.
SOURCE_TREE_MODULES = [
    "test_package_metadata.py",
    "test_redshift_weights.py",
    "test_documentation_headings.py",
]

collect_ignore = [] if (SOURCE / "vega" / "redshift_weights.py").is_file() else SOURCE_TREE_MODULES
