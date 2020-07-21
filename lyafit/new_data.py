import numpy as np
from astropy.io import fits
from scipy import linalg
from scipy.sparse import csr_matrix


class Data:
    """Class for handling lya forest correlation function data
    An instance of this is required for each cf component
    """
    def __init__(self, data_config):
        """Read the config file and initialize the data

        Parameters
        ----------
        data_config : ConfigParser
            Config file for a cf component
        """
        # First read the config and initialize some variables we need
        self._name = data_config['data'].get('name')
        self._tracer1 = {}
        self._tracer2 = {}
        self._tracer1['name'] = data_config['data'].get('tracer1')
        self._tracer1['type'] = data_config['data'].get('tracer1-type')
        self._tracer2['name'] = data_config['data'].get('tracer2',
                                                        self._tracer1['name'])
        self._tracer2['type'] = data_config['data'].get('tracer2-type',
                                                        self._tracer1['type'])

        # Read the data file
        data_path = data_config['data'].get('filename')
        self._read_data(data_path, data_config['cuts'])

    def _read_data(self, data_path, cuts_config):
        """Read the data, mask it and prepare the environment

        Parameters
        ----------
        data_path : string
            Path to fits data file
        cuts_config : ConfigParser
            cuts section from the config file
        """
        hdul = fits.open(data_path)

        self.data_vec = hdul[1].data['DA'][:]
        self.cov_mat = hdul[1].data['CO'][:]
        self.distortion_mat = csr_matrix(hdul[1].data['DM'][:])
        rp = hdul[1].data['RP'][:]
        rt = hdul[1].data['RT'][:]
        z = hdul[1].data['Z'][:]

        if len(hdul) > 2:
            dist_rp = hdul[2].data['DMRP'][:]
            dist_rt = hdul[2].data['DMRT'][:]
            dist_z = hdul[2].data['DMZ'][:]
        else:
            dist_rp = rp.copy()
            dist_rt = rt.copy()
            dist_z = z.copy()
        self.coeff_binning_model = np.sqrt(dist_rp.size / rp.size)

        # Compute the mask and use it on the data
        self.mask = self._build_mask(rp, rt, cuts_config, hdul[1].header)
        self.masked_data_vec = np.zeros(self.mask.sum())
        self.masked_data_vec[:] = self.data_vec[self.mask]

        # TODO This section can be massively optimized by only performing one
        # TODO Cholesky decomposition for the masked cov (instead of 3)
        # Compute inverse and determinant of the covariance matrix
        masked_cov = self.cov_mat[:, self.mask]
        masked_cov = masked_cov[self.mask, :]
        try:
            linalg.cholesky(self.cov_mat)
            print('LOG: Full matrix is positive definite')
        except linalg.LinAlgError:
            print('WARNING: Full matrix is not positive definite')
        try:
            linalg.cholesky(masked_cov)
            print('LOG: Reduced matrix is positive definite')
        except linalg.LinAlgError:
            print('WARNING: Reduced matrix is not positive definite')
        self.inv_masked_cov = linalg.inv(masked_cov)

        # Compute the log determinant using and LDL^T decomposition
        # |C| = Product of Diagonal components of D
        _, d, __ = linalg.ldl(masked_cov)
        self.log_co_det = np.log(d.diagonal()).sum()

        # ! Why are these named square? Is there a better name?
        self.r_square = np.sqrt(rp**2 + rt**2)
        self.mu_square = np.zeros(self.r_square.size)
        w = self.r_square > 0.
        self.mu_square[w] = rp[w] / self.r_square[w]

        # Save the coordinate grid
        self.rp = dist_rp
        self.rt = dist_rt
        self.z = dist_z
        self.r = np.sqrt(self.rp**2 + self.rt**2)
        self.mu = np.zeros(self.r.size)
        w = self.r > 0.
        self.mu[w] = self.rp[w] / self.r[w]

    def _build_mask(self, rp, rt, cuts_config, data_header):
        """Build the mask for the data by comparing
        the cuts from config with the data limits

        Parameters
        ----------
        rp : 1D Array
            Vector of data rp coordinates
        rt : 1D Array
            Vector of data rt coordinates
        cuts_config : ConfigParser
            cuts section from config
        data_header : fits header
            Data file header

        Returns
        -------
        ND Array
            Mask
        """
        # Read the cuts
        rp_min = cuts_config.getfloat('rp-min', 0.)
        rp_max = cuts_config.getfloat('rp-max', 200.)

        rt_min = cuts_config.getfloat('rt-min', 0.)
        rt_max = cuts_config.getfloat('rt-max', 200.)

        r_min = cuts_config.getfloat('r-min', 10.)
        r_max = cuts_config.getfloat('r-max', 180.)

        mu_min = cuts_config.getfloat('mu-min', -1.)
        mu_max = cuts_config.getfloat('mu-max', +1.)

        # TODO If RTMIN is ever added to the cf data files this needs modifying
        # Get the data bin size
        bin_size_rp = (data_header['RPMAX'] - data_header['RPMIN'])
        bin_size_rp /= data_header['NP']
        bin_size_rt = data_header['RTMAX'] / data_header['NT']

        # Compute bin centers
        bin_center_rp = np.zeros(rp.size)
        for i, rp_value in enumerate(rp):
            bin_index = np.floor((rp_value - data_header['RPMIN'])
                                 / bin_size_rp)
            bin_center_rp[i] = data_header['RPMIN'] \
                + (bin_index + 0.5) * bin_size_rp

        bin_center_rt = np.zeros(rt.size)
        for i, rt_value in enumerate(rt):
            bin_index = np.floor(rt_value / bin_size_rt)
            bin_center_rt[i] = (bin_index + 0.5) * bin_size_rt

        bin_center_r = np.sqrt(bin_center_rp**2 + bin_center_rt**2)
        bin_center_mu = bin_center_rp / bin_center_r

        # Build the mask by comparing the data bins to the cuts
        mask = (bin_center_rp > rp_min) & (bin_center_rp < rp_max)
        mask &= (bin_center_rt > rt_min) & (bin_center_rt < rt_max)
        mask &= (bin_center_r > r_min) & (bin_center_r < r_max)
        mask &= (bin_center_mu > mu_min) & (bin_center_mu < mu_max)

        return mask
