"""Top-level package for Vega."""

__author__ = """Andrei Cuceu"""
__email__ = 'andreicuceu@gmail.com'

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vega")
except PackageNotFoundError:
    # If the package is not installed (e.g., just cloned without pip install)
    __version__ = "unknown"

# ruff: noqa: I001
from vega.build_config import BuildConfig
from vega.plots.plot import VegaPlots
from vega.plots.rt_wedges import RtWedge
from vega.plots.shell import Shell

# from .sampler_interface import Sampler
from vega.plots.wedges import Wedge
from vega.postprocess.fit_results import FitResults
from vega.vega_interface import VegaInterface

from vega.scripts.run_vega import run_vega
