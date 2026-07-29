import numpy as np
import scipy.stats as stats
from astropy.io import fits
from getdist import MCSamples
from dataclasses import dataclass
from typing import Union
from numpy.typing import ArrayLike

from vega.utils import find_file
from vega.parameters.param_utils import build_names


@dataclass
class CorrelationOutputRtRp:
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


@dataclass
class CorrelationOutputElls:
    model: ArrayLike
    model_mask: ArrayLike
    data: ArrayLike
    data_mask: ArrayLike
    variance: ArrayLike
    ells: ArrayLike
    r: ArrayLike
    z: ArrayLike
    nell: int

    size: Union[int, None] = None
    chisq: Union[float, None] = None
    reduced_chisq: Union[float, None] = None
    p_value: Union[float, None] = None
    bestfit_marg_coeff: Union[ArrayLike, None] = None


def readBestfitMargCoeff(hdu):
    bestfit_marg_coeff = []
    if 'HIERARCH marg_coeff_0' in hdu.header:
        i = 0
        while f'HIERARCH marg_coeff_{i}' in hdu.header:
            bestfit_marg_coeff.append(hdu.header[f'HIERARCH marg_coeff_{i}'])
            i += 1
    return np.array(bestfit_marg_coeff)


def makeCorrelationOutputRtRp(corr_name, hdu):
    model = hdu.data[corr_name + '_MODEL']
    model_mask = hdu.data[corr_name + '_MODEL_MASK']
    data = hdu.data[corr_name + '_DATA']
    data_mask = hdu.data[corr_name + '_MASK']

    variance = hdu.data[corr_name + '_VAR']
    rp = hdu.data[corr_name + '_RP']
    rt = hdu.data[corr_name + '_RT']
    z = hdu.data[corr_name + '_Z']

    chisq = hdu.header.get('HIERARCH CHISQ', None)
    reduced_chisq = hdu.header.get('HIERARCH REDUCED_CHISQ', None)
    p_value = hdu.header.get('HIERARCH P_VALUE', None)
    size = hdu.header.get(f"HIERARCH {corr_name}_datasize", data.size)

    bestfit_marg_coeff = readBestfitMargCoeff(hdu)
    return CorrelationOutputRtRp(
        model, model_mask, data, data_mask, variance, rp, rt, z,
        size=size, chisq=chisq, reduced_chisq=reduced_chisq,
        p_value=p_value, bestfit_marg_coeff=bestfit_marg_coeff
    )


def makeCorrelationOutputElls(corr_name, hdu):
    model = hdu.data[corr_name + '_MODEL']
    model_mask = hdu.data[corr_name + '_MODEL_MASK']
    data = hdu.data[corr_name + '_DATA']
    data_mask = hdu.data[corr_name + '_MASK']

    variance = hdu.data[corr_name + '_VAR']
    ells = hdu.data[corr_name + '_ELL']
    r = hdu.data[corr_name + '_R']
    z = hdu.data[corr_name + '_Z']

    chisq = hdu.header.get('HIERARCH CHISQ', None)
    reduced_chisq = hdu.header.get('HIERARCH REDUCED_CHISQ', None)
    p_value = hdu.header.get('HIERARCH P_VALUE', None)
    size = hdu.header.get(f"HIERARCH {corr_name}_datasize", data.size)
    nell = hdu.header.get(f"HIERARCH {corr_name}_nell", None)

    bestfit_marg_coeff = readBestfitMargCoeff(hdu)
    return CorrelationOutputElls(
        model, model_mask, data, data_mask, variance, ells, r, z, nell,
        size=size, chisq=chisq, reduced_chisq=reduced_chisq,
        p_value=p_value, bestfit_marg_coeff=bestfit_marg_coeff
    )


class FitResults:
    def __init__(self, path, results_only=False, no_chain=False):
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

        self.marg_coeff = {}
        if not results_only:
            self.read_correlations(hdul)

        hdul.close()

        if not results_only and not no_chain:
            self.chain = self.make_chain(self.names, self.mean, self.cov)

    @staticmethod
    def make_chain(names, mean, cov):
        labels = build_names(names)
        gaussian_samples = np.random.multivariate_normal(mean, cov, size=100000)
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
            corr_name = hdu.name.split('_', 1)[1]
            lowercase_name = corr_name.lower()
            is_multipoles = hdu.header.get(f"HIERARCH {corr_name}_multipoles", None)

            this_corr = None
            if is_multipoles:
                this_corr = makeCorrelationOutputElls(corr_name, hdu)
            else:
                this_corr = makeCorrelationOutputRtRp(corr_name, hdu)

            self.num_data_points += len(this_corr.data[this_corr.data_mask])
            self.marg_coeff[lowercase_name] = this_corr.bestfit_marg_coeff
            self.correlations[lowercase_name] = this_corr

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
            size = hdu.header[corr_name + '_size']
            hdu_data = hdu.data[:size]

            model = hdu_data[model_name]
            model_mask = hdu_data[corr_name + '_MODEL_MASK']
            data = hdu_data[corr_name + '_DATA']
            data_mask = hdu_data[corr_name + '_MASK']
            self.num_data_points += len(data[data_mask])

            variance = hdu_data[corr_name + '_VAR']
            rp = hdu_data[corr_name + '_RP']
            rt = hdu_data[corr_name + '_RT']
            z = hdu_data[corr_name + '_Z']

            self.correlations[corr_name] = CorrelationOutputRtRp(
                model, model_mask, data, data_mask, variance, rp, rt, z
            )

        self.p_value = 1 - stats.chi2.cdf(self.chisq, self.num_data_points - self.num_pars)
        self.reduced_chisq = self.chisq / (self.num_data_points - self.num_pars)
