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
