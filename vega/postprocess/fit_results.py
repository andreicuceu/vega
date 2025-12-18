import numpy as np
import scipy.stats as stats
from astropy.io import fits
from getdist import MCSamples
from dataclasses import dataclass, field
from typing import Union
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

    size: Union[int, None] = None
    chisq: Union[float, None] = None
    reduced_chisq: Union[float, None] = None
    p_value: Union[float, None] = None
    bestfit_marg_coeff: Union[ArrayLike, None] = None


class FitResults:
    def __init__(self, path, results_only=False):
        hdul = fits.open(find_file(path))

        self.chisq = hdul['BESTFIT'].header['FVAL']
        self.valid = hdul['BESTFIT'].header['VALID']
        self.accurate = hdul['BESTFIT'].header['ACCURATE']
        self.names = hdul['BESTFIT'].data['names']
        self.mean = hdul['BESTFIT'].data['values']
        self.cov = hdul['BESTFIT'].data['covariance']
        self.params = {name: value for name, value in zip(self.names, self.mean)}
        self.sigmas = {
            name: value for name, value in zip(self.names, hdul['BESTFIT'].data['errors'])}
        self.num_pars = len(self.names)

        if not results_only:
            self.read_correlations(hdul)

        hdul.close()

        if not results_only:
            self.chain = self.make_chain(self.names, self.mean, self.cov)

    @staticmethod
    def make_chain(names, mean, cov):
        labels = build_names(names)
        gaussian_samples = np.random.multivariate_normal(mean, cov, size=1000000)
        return MCSamples(samples=gaussian_samples, names=names, labels=list(labels.values()))

    def read_correlations(self, hdul):
        model_hdus = [hdu for hdu in hdul if hdu.name.startswith('MODEL')]
        if len(model_hdus) == 0:
            raise ValueError('No model HDUs found in the fit results file.')
        elif model_hdus[0].name == 'MODEL':
            self.old_read_correlations(model_hdus[0])
            return

        self.correlations = {}
        self.num_data_points = 0
        for hdu in model_hdus:
            corr_name = hdu.name.split('_')[1]

            model = hdu.data[corr_name + '_MODEL']
            model_mask = hdu.data[corr_name + '_MODEL_MASK']
            data = hdu.data[corr_name + '_DATA']
            data_mask = hdu.data[corr_name + '_MASK']
            self.num_data_points += len(data[data_mask])

            variance = hdu.data[corr_name + '_VAR']
            rp = hdu.data[corr_name + '_RP']
            rt = hdu.data[corr_name + '_RT']
            z = hdu.data[corr_name + '_Z']

            size = hdu.header.get('HIERARCH SIZE', None)
            chisq = hdu.header.get('HIERARCH CHISQ', None)
            reduced_chisq = hdu.header.get('HIERARCH REDUCED_CHISQ', None)
            p_value = hdu.header.get('HIERARCH P_VALUE', None)

            bestfit_marg_coeff = []
            if 'HIERARCH marg_coeff_0' in hdu.header:
                i = 0
                while f'HIERARCH marg_coeff_{i}' in hdu.header:
                    bestfit_marg_coeff.append(hdu.header[f'HIERARCH marg_coeff_{i}'])
                    i += 1
            bestfit_marg_coeff = np.array(bestfit_marg_coeff)

            lowercase_name = corr_name.lower()
            self.correlations[lowercase_name] = CorrelationOutput(
                model, model_mask, data, data_mask, variance, rp, rt, z,
                size=size, chisq=chisq, reduced_chisq=reduced_chisq,
                p_value=p_value, bestfit_marg_coeff=bestfit_marg_coeff
            )

        self.p_value = 1 - stats.chi2.cdf(self.chisq, self.num_data_points - self.num_pars)
        self.reduced_chisq = self.chisq / (self.num_data_points - self.num_pars)

    def old_read_correlations(self, hdu):
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
