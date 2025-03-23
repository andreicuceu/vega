import numpy as np
import scipy.stats as stats
from astropy.io import fits
from getdist import MCSamples
from dataclasses import dataclass
from numpy.typing import ArrayLike

from vega.utils import find_file
from vega.parameters.param_utils import build_names


@dataclass
class CorrelationOutput:
    model: ArrayLike
    model_mask: ArrayLike
    data: ArrayLike
    data_mask: ArrayLike
    variance: ArrayLike
    rp: ArrayLike
    rt: ArrayLike
    z: ArrayLike


class FitResults:
    def __init__(self, path, results_only=False):
        hdul = fits.open(find_file(path))

        self.chisq = hdul[2].header['FVAL']
        self.valid = hdul[2].header['VALID']
        self.accurate = hdul[2].header['ACCURATE']
        self.names = hdul[2].data['names']
        self.mean = hdul[2].data['values']
        self.cov = hdul[2].data['covariance']
        self.params = {name: value for name, value in zip(self.names, self.mean)}
        self.sigmas = {name: value for name, value in zip(self.names, hdul[2].data['errors'])}
        self.num_pars = len(self.names)

        if not results_only:
            self.read_correlations(hdul[1])

        hdul.close()

        if not results_only:
            self.chain = self.make_chain(self.names, self.mean, self.cov)

    @staticmethod
    def make_chain(names, mean, cov):
        labels = build_names(names)
        gaussian_samples = np.random.multivariate_normal(mean, cov, size=1000000)
        return MCSamples(samples=gaussian_samples, names=names, labels=list(labels.values()))

    def read_correlations(self, hdu):
        if len(hdu.data.columns) % 9 != 0:
            raise ValueError('Vega output format has changed. Please update fit reader.')

        self.correlations = {}
        self.num_data_points = 0
        for i in range(len(hdu.data.columns) // 9):
            model_name = hdu.data.columns[i * 9].name
            assert model_name[-6:] == '_MODEL'
            corr_name = model_name[:-6]

            model = hdu.data[model_name]
            model_mask = hdu.data[corr_name + '_MODEL_MASK']
            data = hdu.data[corr_name + '_DATA']
            data_mask = hdu.data[corr_name + '_MASK']
            self.num_data_points += len(data[data_mask])

            variance = hdu.data[corr_name + '_VAR']
            rp = hdu.data[corr_name + '_RP']
            rt = hdu.data[corr_name + '_RT']
            z = hdu.data[corr_name + '_Z']

            self.correlations[corr_name] = CorrelationOutput(model, model_mask, data, data_mask,
                                                             variance, rp, rt, z)

        self.p_value = 1 - stats.chi2.cdf(self.chisq, self.num_data_points - self.num_pars)
        self.reduced_chisq = self.chisq / (self.num_data_points - self.num_pars)
