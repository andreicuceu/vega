class CorrelationItem:
    """Class for handling the info and config of
    each correlation function component
    """

    def __init__(self, config):
        """Read the config and get tracer info

        Parameters
        ----------
        config : ConfigParser
            parsed config file
        """
        # Save the config and read the tracer info
        self.config = config
        self.name = config['data'].get('name')
        self.tracer1 = {}
        self.tracer2 = {}
        self.tracer1['name'] = config['data'].get('tracer1')
        self.tracer1['type'] = config['data'].get('tracer1-type')
        self.tracer2['name'] = config['data'].get('tracer2',
                                                  self.tracer1['name'])
        self.tracer2['type'] = config['data'].get('tracer2-type',
                                                  self.tracer1['type'])
        self.has_metals = False

    def init_metals(self, tracer_catalog, metal_correlations):
        """Initialize the metal config
        This should be called from the data object if we have metal matrices

        Parameters
        ----------
        tracer_catalog : dict
            Dictionary containing all tracer objects (metals and the core ones)
        metal_correlations : list
            list of all metal correlations we need to compute
        """
        self.tracer_catalog = tracer_catalog
        self.metal_correlations = metal_correlations
        self.has_metals = True

  
    # def compute_model(self, pars, pk_full, pk_smooth):
    #     """Compute correlation function model using input
    #     P(k) components and pars

    #     Parameters
    #     ----------
    #     pars : dict
    #         computation parameters
    #     pk_full : 1D Array
    #         Full fiducial Power Spectrum
    #     pk_smooth : 1D Array
    #         Smooth component Power Spectrum

    #     Returns
    #     -------
    #     1D Array
    #         Correlation Function Model
    #     """
    #     return self.model.compute(pars, pk_full, pk_smooth)

    # def chi2(self, pars, pk_full, pk_smooth):
    #     """Compute component chi2 by computing a model
    #     correlation function

    #     Parameters
    #     ----------
    #     pars : dict
    #         computation parameters
    #     pk_full : 1D Array
    #         Full fiducial Power Spectrum
    #     pk_smooth : 1D Array
    #         Smooth component Power Spectrum

    #     Returns
    #     -------
    #     float
    #         chi^2
    #     """
    #     xi_model = self.compute_model(pars, pk_full, pk_smooth)
    #     diff = self.data.masked_data_vec - xi_model[self.data.mask]
    #     chi2 = diff.T.dot(self.data.inv_masked_cov.dot(diff))

    #     return chi2

    # def log_lik(self, pars, pk_full, pk_smooth):
    #     """Compute component log likelihood by computing a model
    #     correlation function

    #     Parameters
    #     ----------
    #     pars : dict
    #         computation parameters
    #     pk_full : 1D Array
    #         Full fiducial Power Spectrum
    #     pk_smooth : 1D Array
    #         Smooth component Power Spectrum

    #     Returns
    #     -------
    #     float
    #         log Likelihood
    #     """
    #     chi2 = self.chi2(pars, pk_full, pk_smooth)
    #     log_lik = - 0.5 * len(self.data.masked_data_vec) * np.log(2 * np.pi)
    #     log_lik -= 0.5 * self.data.log_cov_det
    #     log_lik -= 0.5 * chi2

    #     return log_lik
