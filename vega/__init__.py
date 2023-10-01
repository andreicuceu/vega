"""Top-level package for Vega."""

__author__ = """Andrei Cuceu"""
__email__ = 'andreicuceu@gmail.com'
__version__ = '0.5.2'

from vega.vega_interface import VegaInterface
from vega.build_config import BuildConfig
# from .sampler_interface import Sampler
from vega.plots.wedges import Wedge
from vega.plots.shell import Shell
from vega.plots.rt_wedges import RtWedge
from vega.plots.plot import VegaPlots
from vega.postprocess.fit_results import FitResults
