from configparser import ConfigParser
from vega.utils import find_file
from pathlib import Path
import os


class BuildConfig:
    """Build and manage config files based on available templates
    """

    _params_template = None
    recognised_fits = ['lyalya_lyalya', 'lyalya_lyalyb', 'lyalya_qso', 'lyalyb_qso',
                       'lyalya_dla', 'lyalyb_dla', 'qso_qso', 'qso_dla', 'dla_dla',
                       'lyalya_lyalya__lyalya_lyalyb', 'lyalya_lyalya__lyalya_qso',
                       'lyalya_lyalyb__lyalyb_qso', 'lyalya_qso__lyalyb_qso',
                       'lyalya_lyalya__lyalya_dla', 'lyalya_lyalyb__lyalyb_dla',
                       'lyalya_dla__lyalyb_dla',
                       'lyalya_lyalya__lyalya_lyalyb__lyalya_qso__lyalyb_qso',
                       'lyalya_lyalya__lyalya_lyalyb__lyalya_dla__lyalyb_dla']

    def __init__(self, options={}):
        """Initialize the model options that are not tracer or correlation specific.

        Parameters
        ----------
        options : dict, optional
            Dictionary with model options, by default {}
        """
        self.options = {}

        self.options['scale_params'] = options.get('scale_params', 'ap_at')
        self.options['template'] = options.get('template', 'PlanckDR16/PlanckDR16.fits')
        self.options['full_shape'] = options.get('full_shape', False)
        self.options['smooth_scaling'] = options.get('smooth_scaling', False)

        self.options['small_scale_nl'] = options.get('small_scale_nl', False)
        self.options['bao_broadening'] = options.get('bao_broadening', False)
        self.options['uv_background'] = options.get('uv_background', False)
        self.options['velocity_dispersion'] = options.get('velocity_dispersion', None)
        self.options['radiation_effects'] = options.get('radiation_effects', False)
        self.options['hcd'] = options.get('hcd', None)
        self.options['fvoigt_model'] = options.get('fvoigt_model', 'exp')
        self.options['fullshape_smoothing'] = options.get('fullshape_smoothing', None)

        metals = options.get('metals', None)
        if metals == 'all':
            metals = ['SiII(1190)', 'SiII(1193)', 'SiIII(1207)', 'SiII(1260)', 'CIV(eff)']
        self.options['metals'] = metals

    def build(self, correlations, fit_type, fit_info, out_path, parameters={}):
        # Check if we know the fit combination
        if fit_type not in self.recognised_fits:
            raise ValueError('Unknown fit: {}'.format(fit_type))

        # Save some of the info
        self.fit_info = fit_info

        # Check if we need sampler or fitter or both
        self.fitter = fit_info.get('fitter', True)
        self.sampler = fit_info.get('sampler', False)

        # get the relevant paths
        self.config_path = Path(os.path.expandvars(out_path))
        assert self.config_path.is_dir()
        if self.fitter:
            self.fitter_out_path = self.config_path / 'output_fitter'
            if not self.fitter_out_path.exists():
                os.mkdir(self.fitter_out_path)
        if self.sampler:
            self.sampler_out_path = self.config_path / 'output_sampler'
            if not self.sampler_out_path.exists():
                os.mkdir(self.sampler_out_path)

        # Build config files for each correlation
        components = fit_type.split('__')
        self.corr_paths = []
        self.corr_names = []
        for name in components:
            # Check if we have info on the correlation
            if name not in correlations:
                raise ValueError('You asked for a fit combination with an unknown'
                                 ' correlation: {}'.format(name))

            # Build the config file for the correlation and save the path
            corr_path, tracer1, tracer2 = self._build_corr_config(name, correlations[name])
            self.corr_paths.append(corr_path)
            if tracer1 not in self.corr_names:
                self.corr_names.append(tracer1)
            if tracer2 not in self.corr_names:
                self.corr_names.append(tracer2)

        self.parameters = parameters
        main_path = self._build_main_config(fit_type, fit_info)

        return main_path

    def _build_corr_config(self, name, corr_info):
        """Build config file for a correlation based on a template

        Parameters
        ----------
        name : string
            Name of the correlation. Must be the same as corresponding template file name
        corr_info : dict
            Correlation information. The paths to the data and metal files are required.

        Returns
        -------
        [type]
            [description]
        """
        # Read template
        config = ConfigParser()
        config.optionxform = lambda option: option
        template_path = find_file('vega/templates/{}.ini'.format(name))
        config.read(template_path)

        # get tracer info
        tracer1 = config['data']['tracer1']
        tracer2 = config['data']['tracer2']
        type1 = config['data']['tracer1-type']
        type2 = config['data']['tracer2-type']

        # Write the basic info
        config['data']['filename'] = corr_info.get('corr_path')
        config['cuts']['r-min'] = str(corr_info.get('r-min', 10))
        config['cuts']['r-max'] = str(corr_info.get('r-max', 180))
        config['parameters']['par binsize {}'.format(name)] = str(corr_info.get('binsize', 4))
        config['parameters']['per binsize {}'.format(name)] = str(corr_info.get('binsize', 4))

        # Write the model options
        # Things that require both tracers to be LYA
        if tracer1 == 'LYA' and tracer2 == 'LYA':
            if self.options['small_scale_nl']:
                config['model']['small scale nl'] = 'dnl_arinyo'

        # Things that require at least one tracer to be continuous
        if type1 == 'continuous' or type2 == 'continuous':
            if self.options['uv_background']:
                config['model']['add uv'] = 'True'

            if self.options['hcd'] is not None:
                assert self.options['hcd'] in ['mask', 'Rogers2018', 'sinc']
                config['model']['model-hcd'] = self.options['hcd']
                if self.options['hcd'] == 'mask':
                    config['model']['fvoigt_model'] = self.options['fvoigt_model']

            if self.options['metals'] is not None:
                config['metals'] = {}
                config['metals']['filename'] = corr_info.get('metal_path')
                config['metals']['z evol'] = 'bias_vs_z_std'
                if type1 == 'continuous':
                    config['metals']['in tracer1'] = ' '.join(self.options['metals'])
                if type2 == 'continuous':
                    config['metals']['in tracer2'] = ' '.join(self.options['metals'])

        # Things that require at least one discrete tracer
        if type1 == 'discrete' or type2 == 'discrete':
            if self.options['velocity_dispersion'] is not None:
                assert self.options['velocity_dispersion'] in ['lorentz', 'gaussian']
                config['model']['velocity dispersion'] = self.options['velocity_dispersion']

                if self.options['metals'] is not None and type1 != type2:
                    config['metals']['velocity dispersion'] = self.options['velocity_dispersion']

        # Only for the LYA - QSO cross
        if 'LYA' in [tracer1, tracer2] and 'QSO' in [tracer1, tracer2]:
            if self.options['radiation_effects']:
                config['model']['radiation effects'] = 'True'

        # General things
        if self.options['fullshape_smoothing'] is not None:
            assert self.options['fullshape_smoothing'] in ['gauss', 'exp']
            config['model']['fullshape smoothing'] = self.options['fullshape_smoothing']

        corr_path = self.config_path / '{}.ini'.format(name)
        with open(corr_path, 'w') as configfile:
            config.write(configfile)

        return corr_path, tracer1, tracer2

    def _build_main_config(self, fit_type, fit_info):
        # Initialize the config
        config = ConfigParser()
        config.optionxform = lambda option: option

        # Write the paths to the correlation configs
        config['data sets'] = {}
        config['data sets']['zeff'] = str(fit_info['zeff'])
        corr_paths = [str(path) for path in self.corr_paths]
        config['data sets']['ini files'] = ' '.join(corr_paths)

        # Write the scale parameters functions
        config['cosmo-fit type'] = {}
        config['cosmo-fit type']['cosmo fit func'] = self.options['scale_params']

        # Write the template info
        config['fiducial'] = {}
        config['fiducial']['filename'] = self.options['template']
        config['fiducial']['full-shape'] = str(self.options['full_shape'])
        config['fiducial']['smooth-scaling'] = str(self.options['smooth_scaling'])

        # Write the output path
        config['output'] = {}
        config['output']['filename'] = str(self.fitter_out_path / fit_type)

        sample_params = fit_info['sample_params']
        config['sample'] = {}
        for param in sample_params:
            config['sample'][param] = 'True'

        # Write the parameters
        config['parameters'] = {}
        for name, value in self.parameters.items():
            config['parameters'][name] = str(value)

        # Check all sampled parameters are defined
        for param in sample_params:
            assert param in config['parameters']

        # Check if we need the sampler
        if self.sampler:
            config['control'] = {}
            config['control']['sampler'] = 'True'

            config['Polychord'] = {}
            config['Polychord']['path'] = str(self.sampler_out_path)
            config['Polychord']['name'] = fit_type
            config['Polychord']['nlive'] = fit_info['Polychord'].get('nlive',
                                                                     str(25 * len(sample_params)))
            config['Polychord']['num_repeats'] = fit_info['Polychord'].get('num_repeats',
                                                                           str(len(sample_params)))
            config['Polychord']['do_clustering'] = 'True'
            config['Polychord']['boost_posterior'] = str(3)

        # Write main config
        main_path = self.config_path / 'main.ini'
        with open(main_path, 'w') as configfile:
            config.write(configfile)

        return main_path

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if self._params_template is None:
            # Read template
            config = ConfigParser()
            config.optionxform = lambda option: option
            template_path = find_file('vega/templates/parameters.ini')
            config.read(template_path)
            self._params_template = config['parameters']

        def get_par(name):
            if name not in parameters and name not in self._params_template:
                raise ValueError('Unknown parameter: {}, please pass a default value.'.format(name))
            return parameters.get(name, self._params_template[name])

        new_params = {}

        # Scale parameters
        if self.options['scale_params'] == 'ap_at':
            new_params['ap'] = get_par('ap')
            new_params['at'] = get_par('at')
        elif self.options['scale_params'] == 'phi_gamma':
            new_params['phi'] = get_par('phi')
            new_params['gamma'] = get_par('gamma')
            if self.options['smooth_scaling']:
                new_params['phi_smooth'] = get_par('phi_smooth')
                new_params['gamma_smooth'] = get_par('gamma_smooth')
        elif self.options['scale_params'] == 'aiso_epsilon':
            new_params['aiso'] = get_par('aiso')
            new_params['1+epsilon'] = get_par('1+epsilon')
        else:
            raise ValueError('Unknown scale parameters: {}'.format(self.options['scale_params']))

        # Peak parameters
        if self.options['bao_broadening']:
            new_params['sigmaNL_per'] = get_par('sigmaNL_per')
            new_params['sigmaNL_par'] = get_par('sigmaNL_par')
        else:
            new_params['sigmaNL_per'] = 0.
            new_params['sigmaNL_par'] = 0.
        new_params['bao_amp'] = get_par('bao_amp')

        # bias beta model
        for name in self.corr_names:
            if self.fit_info['use_bias_eta'].get(name, False):
                new_params['growth_rate'] = get_par('growth_rate')
                new_params['bias_eta_{}'.format(name)] = get_par('bias_eta_{}'.format(name))
            else:
                new_params['bias_{}'.format(name)] = get_par('bias_{}'.format(name))
            new_params['beta_{}'.format(name)] = get_par('beta_{}'.format(name))
            new_params['alpha_{}'.format(name)] = get_par('alpha_{}'.format(name))

        # Small scale non-linear model
        if self.options['small_scale_nl']:
            new_params['dnl_arinyo_q1'] = get_par('dnl_arinyo_q1')
            new_params['dnl_arinyo_kv'] = get_par('dnl_arinyo_kv')
            new_params['dnl_arinyo_av'] = get_par('dnl_arinyo_av')
            new_params['dnl_arinyo_bv'] = get_par('dnl_arinyo_bv')
            new_params['dnl_arinyo_kp'] = get_par('dnl_arinyo_kp')

        # HCDs
        if self.options['hcd'] is not None:
            new_params['bias_hcd'] = get_par('bias_hcd')
            new_params['beta_hcd'] = get_par('beta_hcd')
            new_params['L0_hcd'] = get_par('L0_hcd')

        # Delta_rp
        if 'QSO' in self.corr_names:
            new_params['drp_QSO'] = get_par('drp_QSO')

        # Velocity dispersion parameters
        if self.options['velocity_dispersion'] is not None:
            if self.options['velocity_dispersion'] == 'lorentz':
                new_params['sigma_velo_disp_lorentz_QSO'] = get_par('sigma_velo_disp_lorentz_QSO')
            else:
                new_params['sigma_velo_disp_gauss_QSO'] = get_par('sigma_velo_disp_gauss_QSO')

        # QSO radiation effects
        if self.options['radiation_effects']:
            new_params['qso_rad_strength'] = get_par('qso_rad_strength')
            new_params['qso_rad_asymmetry'] = get_par('qso_rad_asymmetry')
            new_params['qso_rad_lifetime'] = get_par('qso_rad_lifetime')
            new_params['qso_rad_decrease'] = get_par('qso_rad_decrease')

        # UV background parameters
        if self.options['uv_background']:
            new_params['bias_gamma'] = get_par('bias_gamma')
            new_params['bias_prim'] = get_par('bias_prim')
            new_params['lambda_uv'] = get_par('lambda_uv')

        # Metals
        if self.options['metals'] is not None:
            for name in self.options['metals']:
                new_params['bias_eta_{}'.format(name)] = get_par('bias_eta_{}'.format(name))
                new_params['beta_{}'.format(name)] = get_par('beta_{}'.format(name))
                new_params['alpha_{}'.format(name)] = get_par('alpha_{}'.format(name))

        # Full-shape smoothing
        if self.options['fullshape_smoothing'] is not None:
            new_params['par_sigma_smooth'] = get_par('par_sigma_smooth')
            new_params['per_sigma_smooth'] = get_par('per_sigma_smooth')
            if self.options['fullshape_smoothing'] == 'exp':
                new_params['par_exp_smooth'] = get_par('par_exp_smooth')
                new_params['per_exp_smooth'] = get_par('per_exp_smooth')

        self._parameters = new_params
