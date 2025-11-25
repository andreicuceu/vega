import numpy as np
from scipy.sparse import coo_array
from picca import constants as picca_constants
from functools import reduce


class CorrelationItem:
    """Class for handling the info and config of
    each correlation function component.
    """
    cosmo = None
    model_coordinates = None
    dist_model_coordinates = None
    data_coordinates = None
    low_mem_mode = False

    def __init__(self, config, model_pk=False):
        """

        Parameters
        ----------
        config : ConfigParser
            parsed config file
        """
        # Save the config and read the tracer info
        self.config = config
        self.model_pk = model_pk
        self.name = config['data'].get('name')
        self.tracer1 = {}
        self.tracer2 = {}
        self.tracer1['name'] = config['data'].get('tracer1')
        self.tracer1['type'] = config['data'].get('tracer1-type')
        self.tracer2['name'] = config['data'].get('tracer2', self.tracer1['name'])
        self.tracer2['type'] = config['data'].get('tracer2-type', self.tracer1['type'])

        self.cov_rescale = config['data'].getfloat('cov_rescale', None)
        self.has_distortion = config['data'].getboolean('distortion', True)

        self.has_data = config['data'].getboolean('has_datafile', True)
        if 'filename' not in config['data']:
            self.has_data = False

        self.new_metals = config['model'].getboolean('new_metals', False)
        if self.new_metals:
            self.tracer1['weights-path'] = config['data'].get('weights-tracer1')
            self.tracer2['weights-path'] = config['data'].get('weights-tracer2', None)
            if self.tracer2['weights-path'] is None:
                self.tracer2['weights-path'] = self.tracer1['weights-path']

        self.test_flag = config['data'].getboolean('test', False)

        marg_rs = [
            config['model'].getfloat("marginalize-below-rtmax", 0),
            config['model'].getfloat("marginalize-above-rtmin", 0),
            config['model'].getfloat("marginalize-below-rpmax", 0),
            config['model'].getfloat("marginalize-above-rpmin", 0)
        ]
        self.marginalize_small_scales_prior_sigma = config['model'].getfloat(
            "marginalize-prior-sigma", 10.0)
        self.marginalize_small_scales = {}
        for i, name in enumerate(['rtmax', 'rtmin', 'rpmax', 'rpmin']):
            if marg_rs[i] > 0:
                self.marginalize_small_scales[name] = marg_rs[i]

        self.marginalize_small_scales_with_cuts = config['model'].getboolean(
            "marginalize-small-scales-with-cuts", False)

        self.has_metals = False
        self.has_bb = False

    def init_metals(self, tracer_catalog, metal_correlations):
        """Initialize the metal config.

        This should be called from the data object if we have metal matrices.

        Parameters
        ----------
        tracer_catalog : dict
            Dictionary containing all tracer objects (metals and the core ones)
        metal_correlations : list
            list of all metal correlations we need to compute
        """
        self.tracer_catalog = tracer_catalog
        self.metal_correlations = []
        for corr in metal_correlations:
            corr_hash = tuple(sorted([corr[0], corr[1]]))

            # If only one tracer is given, assume auto-correlation
            if len(corr_hash) != 2:
                corr_hash = (corr[0], corr[0])

            # Make sure main tracers are in the correct position in the tuple
            if corr_hash[0] == self.tracer2['name'] or corr_hash[1] == self.tracer1['name']:
                corr_hash = (corr_hash[1], corr_hash[0])

            # Avoid duplicates
            if corr_hash not in self.metal_correlations:
                self.metal_correlations.append(corr_hash)

        self.has_metals = True

    def init_broadband(self, coeff_binning_model):
        """Initialize the parameters we need to compute
        the broadband functions

        Parameters
        ----------
        coeff_binning_model : float
            Ratio of distorted coordinate grid bin size to undistorted bin size
        """
        self.coeff_binning_model = coeff_binning_model
        self.has_bb = True

    def init_coordinates(
            self, model_coordinates, dist_model_coordinates=None, data_coordinates=None):
        """Initialize the coordinate grids.

        Parameters
        ----------
        model_coordinates : Coordinates
            Coordinates for the model
        dist_model_coordinates : Coordinates, optional
            Distorted coordinates for the model, by default None
        data_coordinates : Coordinates, optional
            Coordinates for the data, by default None
        """
        self.model_coordinates = model_coordinates
        self.data_coordinates = model_coordinates if data_coordinates is None else data_coordinates
        self.dist_model_coordinates = (model_coordinates if dist_model_coordinates is None
                                       else dist_model_coordinates)

    def init_cosmo(self, cosmo_params):
        self.cosmo_params = cosmo_params

        self.cosmo = picca_constants.Cosmo(
                Om=cosmo_params['Omega_m'], Ok=cosmo_params['Omega_k'],
                Or=cosmo_params['Omega_r'], wl=cosmo_params['wl'], verbose=False
            )

    def check_if_blind_corr(self, blind_tracers):
        if 'all' in blind_tracers:
            return True

        for tracer in blind_tracers:
            if tracer in self.tracer1['name'] or tracer in self.tracer2['name']:
                return True

        return False

    def get_undist_xi_marg_templates(self):
        """Calculate undistorted correlation function marginalization templates.
        Degenerate modes are removed in the (relevant) distorted space in
            data.get_dist_xi_marg_templates function.

        Returns
        -------
        sparse array, likely csc_array
            Prior sigma is multiplied to each vector.
        """
        if not self.marginalize_small_scales_with_cuts:
            indeces = []
            if 'rtmax' in self.marginalize_small_scales:
                rtmax = self.marginalize_small_scales['rtmax']
                indeces += [np.nonzero(
                    self.model_coordinates.rt_regular_grid < rtmax
                )[0]]

            if 'rtmin' in self.marginalize_small_scales:
                rtmin = self.marginalize_small_scales['rtmin']
                indeces += [np.nonzero(
                    self.model_coordinates.rt_regular_grid > rtmin
                )[0]]

            if 'rpmax' in self.marginalize_small_scales:
                rpmax = self.marginalize_small_scales['rpmax']
                indeces += [np.nonzero(
                    np.abs(self.model_coordinates.rp_regular_grid) < rpmax
                )[0]]

            if 'rpmin' in self.marginalize_small_scales:
                rpmin = self.marginalize_small_scales['rpmin']
                indeces += [np.nonzero(
                    np.abs(self.model_coordinates.rp_regular_grid) > rpmin
                )[0]]

            common_idx = reduce(np.intersect1d, indeces)
            if common_idx.size == 0:
                raise ValueError(
                    "No common indices found for small-scale marginalization templates."
                )
        else:
            mask = self.model_coordinates.get_mask_scale_cuts(
                self.config['cuts']
            )
            common_idx = np.nonzero(~mask)[0]
            print(
                f"Marginalizing distortion scales with {common_idx.size} points "
                "based on scale cuts."
            )

        N = self.model_coordinates.rt_regular_grid.size
        d = np.ones(common_idx.size)

        templates = coo_array(
            (d, (np.arange(d.size), common_idx)), shape=(d.size, N)
        ).tocsr().T

        a = self.marginalize_small_scales_prior_sigma
        return a * templates
