class CorrelationItem:
    """Class for handling the info and config of
    each correlation function component.
    """
    cosmo_params = None
    model_coordinates = None
    dist_model_coordinates = None
    data_coordinates = None

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
        self.old_fftlog = config['model'].getboolean('old_fftlog', False)

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
        self.metal_correlations = metal_correlations
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
