import os
import git
import numpy as np
from pathlib import Path
from astropy.io import fits
from datetime import datetime
from configparser import ConfigParser

import vega
from vega.utils import find_file


class BuildConfig:
    """Build and manage config files based on available templates
    """

    _params_template = None
    recognised_correlations = ['lyaxlya', 'lyaxlyb', 'lyaxqso', 'lybxqso',
                               'lyaxdla', 'lybxdla', 'qsoxqso', 'qsoxdla', 'dlaxdla']

    def __init__(self, options={}, overwrite=False):
        """Initialize the model options that are not tracer or correlation specific.

        Parameters
        ----------
        options : dict, optional
            Dictionary with model options, by default {}
            Here is a list of options:
                scale_params: string from ['ap_at', 'phi_alpha', 'aiso_epsilon'], default 'ap_at'
                template: string with custom_link, default 'PlanckDR16/PlanckDR16.fits'
                full_shape: Bool, default False
                full_shape_alpha: Bool, default False
                smooth_scaling: Bool, default False

                small_scale_nl: Bool, default False
                small_scale_nl_cross: Bool, default False
                bao_broadening: Bool, default False
                uv_background: Bool, default False
                velocity_dispersion: string from [None, 'lorentz', 'gauss'], default None
                radiation_effects: Bool, default False

                hcd_model: string from [None, 'Rogers', 'nomask', 'sinc'], default None
                fvoigt_model: string, default exp
                fullshape_smoothing: string from [None, 'gauss', 'gauss_iso', 'exp'], default None
                metals: List can include ['all', 'SiII(1190)', 'SiII(1193)', 'SiIII(1207)',
                    'SiII(1260)', 'CIV(eff)'], default None
        """
        self.overwrite = overwrite
        self.options = {}

        self.options['scale_params'] = options.get('scale_params', 'ap_at')
        self.options['template'] = options.get('template', 'PlanckDR16/PlanckDR16.fits')
        self.options['full_shape'] = options.get('full_shape', False)
        self.options['full_shape_alpha'] = options.get('full_shape_alpha', False)
        self.options['smooth_scaling'] = options.get('smooth_scaling', False)

        self.options['small_scale_nl'] = options.get('small_scale_nl', False)
        self.options['small_scale_nl_cross'] = options.get('small_scale_nl_cross', False)
        self.options['bao_broadening'] = options.get('bao_broadening', False)
        self.options['uv_background'] = options.get('uv_background', False)
        self.options['velocity_dispersion'] = options.get('velocity_dispersion', None)
        self.options['radiation_effects'] = options.get('radiation_effects', False)

        self.options['hcd_model'] = options.get('hcd_model', None)
        self.options['fvoigt_model'] = options.get('fvoigt_model', 'exp')
        self.options['fullshape_smoothing'] = options.get('fullshape_smoothing', None)
        self.options['desi-instrumental-systematics'] = options.get('desi-instrumental-systematics',
                                                                    False)
        self.options['test'] = options.get('test', False)

        metals = options.get('metals', None)
        if metals is not None:
            if 'all' in metals:
                metals = ['SiII(1190)', 'SiII(1193)', 'SiIII(1207)', 'SiII(1260)', 'CIV(eff)']
        self.options['metals'] = metals

    def build(self, correlations, fit_type, fit_info, out_path, parameters={}, name_extension=None):
        """Build Vega config files and write them to an output directory

        Parameters
        ----------
        correlations : dict
            Information for each correlation. It must contain the path to the measured correlation,
            and the path to metal files if metals were requested. Optionally specify scale cuts.
            List of options:
                corr_path: string
                metal_path: string
                r-min: float, default 10
                r-max: float, default 180
                rt-min: float, default 0
                fast_metals: bool, default False
                binsize: int, default 4 (deprecated)
                broadband: broadband string configuration
        fit_type : string
            Name of the fit. Includes the name of the correlations with the two
            tracers separated by an "x" (e.g. lyaxqso), and different correlations
            separated by an underscore (e.g. lyaxlya_lyaxqso). If unsure check
            the templates folder to see all possibilities.
        fit_info : dict
            Fit information. Must contain a list of sampled parameters and the effective redshift.
            List of options:
                fitter: bool, default True
                sampler: bool, default False
                bias_beta_config: dict with 'tracer': 'bias_beta', 'bias_eta_beta', 'bias_bias_eta'
                zeff: float, default None
                zeff_rmin: float, default 0
                zeff_rmax: float, default 300
                sample_params: list or dict with vega setup (par_name: min max start step)
                priors: dict with vega setup (par_name: 'gaussian mean sigma')
                Polychord: dict with Polychord setup
        out_path : string
            Path to directory where to write the config files
        parameters : dict, optional
            Parameter values to write to the main config, by default {}
        name_extension : string, optional
            Optional string to add to the config file names, by default None

        Returns
        -------
        string
            Path to the main config file
        """
        # Save some of the info
        self.fit_info = fit_info
        self.name_extension = name_extension

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

        # Check if we know the correlation types
        components = fit_type.split('_')
        for corr in components:
            if corr not in self.recognised_correlations:
                raise ValueError(f'Unknown correlation {corr}, part of fit type {fit_type}.')

        if len(components) != len(set(components)):
            print(f'Warning! fit type {fit_type} has duplicates')

        # Get git hash
        vega_path = Path(os.path.dirname(vega.__file__))
        git_hash = git.Repo(vega_path.parents[0]).head.object.hexsha

        # Build config files for each correlation
        self.corr_paths = []
        self.corr_names = []
        self.data_paths = []
        for name in components:
            # Check if we have info on the correlation
            if name not in correlations:
                raise ValueError(f'You asked for correlation {name} but did not provide'
                                 ' its configuration in the "correlations" dictionary.')

            # Build the config file for the correlation and save the path
            corr_path, data_path, tracer1, tracer2 = self._build_corr_config(name,
                                                                             correlations[name],
                                                                             git_hash)
            self.corr_paths.append(corr_path)
            self.data_paths.append(data_path)
            if tracer1 not in self.corr_names:
                self.corr_names.append(tracer1)
            if tracer2 not in self.corr_names:
                self.corr_names.append(tracer2)

        main_path = self._build_main_config(fit_type, fit_info, parameters, git_hash)

        return main_path

    def _build_corr_config(self, name, corr_info, git_hash):
        """Build config file for a correlation based on a template

        Parameters
        ----------
        name : string
            Name of the correlation. Must be the same as corresponding template file name
        corr_info : dict
            Correlation information. The paths to the data and metal files are required.

        Returns
        -------
        string
            Path to the config file for the correlation.
        """
        # Read template
        config = ConfigParser()
        config.optionxform = lambda option: option
        template_path = find_file(f'vega/templates/{name}.ini')
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
        config['cuts']['rt-min'] = str(corr_info.get('rt-min', 0))
        if self.options['test']:
            config['data']['test'] = 'True'

        if 'binsize' in corr_info:
            config['parameters'] = {}
            config['parameters']['par binsize {}'.format(name)] = str(corr_info.get('binsize', 4))
            config['parameters']['per binsize {}'.format(name)] = str(corr_info.get('binsize', 4))

        # Write the model options
        # Things that require LYA
        if tracer1 == 'LYA' and tracer2 == 'LYA':
            if self.options['small_scale_nl']:
                config['model']['small scale nl'] = 'dnl_arinyo'
        elif tracer1 == 'LYA' or tracer2 == 'LYA':
            if self.options['small_scale_nl_cross']:
                config['model']['small scale nl'] = 'dnl_arinyo'

        # Things that require both tracers to be continuous
        if type1 == 'continuous' and type2 == 'continuous':
            if self.options['desi-instrumental-systematics']:
                config['model']['desi-instrumental-systematics'] = 'True'

        # Things that require at least one tracer to be continuous
        if type1 == 'continuous' or type2 == 'continuous':
            if self.options['uv_background']:
                config['model']['add uv'] = 'True'

            if self.options['hcd_model'] is not None:
                assert self.options['hcd_model'] in ['mask', 'Rogers2018', 'sinc']
                config['model']['model-hcd'] = self.options['hcd_model']
                if self.options['hcd_model'] == 'mask':
                    config['model']['fvoigt_model'] = self.options['fvoigt_model']

            if self.options['metals'] is not None:
                config['metals'] = {}
                config['metals']['filename'] = corr_info.get('metal_path')
                config['metals']['z evol'] = 'bias_vs_z_std'
                if type1 == 'continuous':
                    config['metals']['in tracer1'] = ' '.join(self.options['metals'])
                if type2 == 'continuous':
                    config['metals']['in tracer2'] = ' '.join(self.options['metals'])

                if 'fast_metals' in corr_info:
                    config['model']['fast_metals'] = corr_info.get('fast_metals', 'False')

        # Things that require at least one discrete tracer
        if type1 == 'discrete' or type2 == 'discrete':
            if self.options['velocity_dispersion'] is not None:
                assert self.options['velocity_dispersion'] in ['lorentz', 'gauss']
                config['model']['velocity dispersion'] = self.options['velocity_dispersion']

                if self.options['metals'] is not None and type1 != type2:
                    config['metals']['velocity dispersion'] = self.options['velocity_dispersion']

        # Only for the LYA - QSO cross
        if 'LYA' in [tracer1, tracer2] and 'QSO' in [tracer1, tracer2]:
            if self.options['radiation_effects']:
                config['model']['radiation effects'] = 'True'

        # General things
        if 'broadband' in corr_info:
            config['broadband'] = {}
            for key, item in corr_info['broadband'].items():
                config['broadband'][key] = item

        if self.options['fullshape_smoothing'] is not None:
            assert self.options['fullshape_smoothing'] in ['gauss', 'gauss_iso', 'exp']
            config['model']['fullshape smoothing'] = self.options['fullshape_smoothing']

        if self.name_extension is None:
            corr_path = self.config_path / '{}.ini'.format(name)
        else:
            corr_path = self.config_path / '{}-{}.ini'.format(name, self.name_extension)

        if corr_path.is_file() and not self.overwrite:
            raise ValueError(f'File {corr_path} already exists. Please change the name extension.')

        with open(corr_path, 'w') as configfile:
            configfile.write(f'# File written on {datetime.now()} \n')
            configfile.write(f'# Vega git hash: {git_hash} \n\n')
            config.write(configfile)

        return corr_path, config['data']['filename'], tracer1, tracer2

    @staticmethod
    def get_zeff(data_paths, rmin=0., rmax=300.):
        """Compute effective redshift of all correlations

        Parameters
        ----------
        data_paths : List[string]
            List of paths to exported picca correlations
        rmin : float, optional
            Minimum separation, by default 0.
        rmax : float, optional
            Maximum separation, by default 300.

        Returns
        -------
        float
            Effective redshift
        """
        zeff_list = []
        weights = []
        for path in data_paths:
            hdul = fits.open(path)

            r_arr = np.sqrt(hdul[1].data['RP']**2 + hdul[1].data['RT']**2)
            cells = (r_arr > rmin) * (r_arr < rmax)

            zeff = np.average(hdul[1].data['Z'][cells], weights=hdul[1].data['NB'][cells])
            weight = np.sum(hdul[1].data['NB'][cells])

            hdul.close()

            zeff_list.append(zeff)
            weights.append(weight)

        zeff = np.average(zeff_list, weights=weights)
        return zeff

    def _build_main_config(self, fit_type, fit_info, parameters, git_hash):
        """Build the main vega configuration file

        Parameters
        ----------
        fit_type : string
            Name of the fit. Includes the name of the correlations with the two
            tracers separated by a single underscore (e.g. lyalya_qso),
            and different correlations separated by a double underscore
            (e.g. lyalya_lyalya__lyalya_qso). If unsure check the templates
            folder to see all possibilities.
        fit_info : dict
            Fit information. Must contain a list of sampled parameters and the effective redshift.
        parameters : dict, optional
            Parameter values to write to the main config

        Returns
        -------
        Path
            Path to written 'main.ini' file to be used with vega
        """
        # Initialize the config
        config = ConfigParser()
        config.optionxform = lambda option: option

        # Check the effective redshift
        self.zeff_in = fit_info.get('zeff', None)
        zeff_rmin = fit_info.get('zeff_rmin', 0.)
        zeff_rmax = fit_info.get('zeff_rmax', 300.)

        zeff_comp = self.get_zeff(self.data_paths, zeff_rmin, zeff_rmax)
        if self.zeff_in is None:
            self.zeff_in = zeff_comp
        elif self.zeff_in != zeff_comp:
            print('Warning: Input zeff and computed zeff are different. Will write input zeff.')

        # Write the paths to the correlation configs
        config['data sets'] = {}
        config['data sets']['zeff'] = str(self.zeff_in)
        corr_paths = [str(path) for path in self.corr_paths]
        config['data sets']['ini files'] = ' '.join(corr_paths)

        # Write the scale parameters functions
        config['cosmo-fit type'] = {}
        config['cosmo-fit type']['cosmo fit func'] = self.options['scale_params']
        config['cosmo-fit type']['full-shape'] = str(self.options['full_shape'])
        config['cosmo-fit type']['full-shape-alpha'] = str(self.options['full_shape_alpha'])
        config['cosmo-fit type']['smooth-scaling'] = str(self.options['smooth_scaling'])

        # Write the template info
        config['fiducial'] = {}
        config['fiducial']['filename'] = self.options['template']

        # Write the output path
        run_name = fit_type
        if self.name_extension is not None:
            run_name += '-{}'.format(self.name_extension)
        config['output'] = {}
        config['output']['filename'] = str(self.fitter_out_path / run_name)

        # Write the sampled parameters
        sample_params = fit_info['sample_params']
        config['sample'] = {}
        if type(sample_params) is list:
            for param in sample_params:
                config['sample'][param] = 'True'
        elif type(sample_params) is dict:
            for param, setup in sample_params.items():
                config['sample'][param] = setup
        else:
            raise TypeError('The sample_params object has to be either a list or a dict.')

        # Write the priors
        if 'priors' in fit_info:
            config['priors'] = {}
            for par, prior in fit_info['priors'].items():
                assert par in config['sample'], 'Cannot add prior for parameter that is not sampled'
                config['priors'][par] = prior

        # Write the parameters
        self.parameters = parameters
        config['parameters'] = {}
        for name, value in self.parameters.items():
            config['parameters'][name] = str(value)

        # Check all sampled parameters are defined
        for param in sample_params:
            if param not in config['parameters']:
                raise ValueError(f'Asked for unknown parameter "{param}". This does not exist in '
                                 'the current configuration. Please check the vega configuration '
                                 'you requested is correct. If this is a new parameter that does '
                                 'not have a default value yet, please add it to the parameters '
                                 'dictionary when calling BuildConfig.')

        # Check if we need the sampler
        if self.sampler:
            config['control'] = {}
            config['control']['sampler'] = 'True'

            config['Polychord'] = {}
            config['Polychord']['path'] = str(self.sampler_out_path)

            config['Polychord']['name'] = run_name
            config['Polychord']['num_live'] = fit_info['Polychord'].get('num_live',
                                                                        str(25*len(sample_params)))
            config['Polychord']['num_repeats'] = fit_info['Polychord'].get('num_repeats',
                                                                           str(len(sample_params)))
            config['Polychord']['do_clustering'] = 'True'
            config['Polychord']['boost_posterior'] = str(3)

        # Write main config
        if self.name_extension is None:
            main_path = self.config_path / 'main.ini'
        else:
            main_path = self.config_path / 'main-{}.ini'.format(self.name_extension)

        if main_path.is_file() and not self.overwrite:
            raise ValueError(f'File {main_path} already exists. Please change the name extension.')

        with open(main_path, 'w') as configfile:
            configfile.write(f'# File written on {datetime.now()} \n')
            configfile.write(f'# Vega git hash: {git_hash} \n\n')
            config.write(configfile)

        return main_path

    @property
    def parameters(self):
        """Parameters property

        Returns
        -------
        dict
            Parameters
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """Setter for parameters property

        Parameters
        ----------
        parameters : dict
            Parameters dictionary
        """
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
        elif self.options['scale_params'] == 'phi_alpha':
            new_params['phi'] = get_par('phi')
            new_params['alpha'] = get_par('alpha')
            if self.options['full_shape']:
                new_params['phi_full'] = get_par('phi_full')
                if self.options['full_shape_alpha']:
                    new_params['alpha_full'] = get_par('alpha_full')
            if self.options['smooth_scaling']:
                new_params['phi_smooth'] = get_par('phi_smooth')
                new_params['alpha_smooth'] = get_par('alpha_smooth')
        elif self.options['scale_params'] == 'aiso_epsilon':
            new_params['aiso'] = get_par('aiso')
            new_params['epsilon'] = get_par('epsilon')
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

        def add_bias_beta(new_params, tracer, bias_beta_config, bias, bias_eta, beta, growth_rate):
            if bias_beta_config == 'bias_beta':
                new_params[f'bias_{tracer}'] = bias
                new_params[f'beta_{tracer}'] = beta
            elif bias_beta_config == 'bias_bias_eta':
                new_params[f'bias_{tracer}'] = bias
                new_params[f'bias_eta_{tracer}'] = bias_eta
                new_params['growth_rate'] = growth_rate
            elif bias_beta_config == 'bias_eta_beta':
                new_params[f'beta_{tracer}'] = beta
                new_params[f'bias_eta_{tracer}'] = bias_eta
                new_params['growth_rate'] = growth_rate
            else:
                raise ValueError(f'Option {bias_beta_config} not a valid bias_beta_config. '
                                 'Choose from ["bias_beta", "bias_eta_beta", "bias_bias_eta"].')

        # bias beta model
        for name in self.corr_names:
            bias_beta_config = self.fit_info['bias_beta_config'].get(name, 'bias_beta')

            growth_rate = parameters.get('growth_rate', None)
            if growth_rate is None:
                growth_rate = self.get_growth_rate(self.zeff_in)

            if (name == 'LYA') or (name == 'LYB'):
                bias = parameters.get(f'bias_{name}', self.get_lya_bias(self.zeff_in))
                bias_eta = parameters.get(f'bias_eta_{name}', None)
                beta = float(get_par(f'beta_{name}'))

                if bias_eta is None:
                    bias_eta = bias * beta / growth_rate
            elif name == 'QSO':
                bias = parameters.get('bias_QSO', self.get_qso_bias(self.zeff_in))
                beta = parameters.get('beta_QSO', None)
                bias_eta = 1

                if beta is None:
                    beta = growth_rate / bias
            else:
                raise ValueError(f'Tracer {name} not supported yet. Please open an issue')

            add_bias_beta(new_params, name, bias_beta_config, bias, bias_eta, beta, growth_rate)

            new_params['alpha_{}'.format(name)] = get_par('alpha_{}'.format(name))

        # Small scale non-linear model
        if self.options['small_scale_nl']:
            new_params['dnl_arinyo_q1'] = get_par('dnl_arinyo_q1')
            new_params['dnl_arinyo_kv'] = get_par('dnl_arinyo_kv')
            new_params['dnl_arinyo_av'] = get_par('dnl_arinyo_av')
            new_params['dnl_arinyo_bv'] = get_par('dnl_arinyo_bv')
            new_params['dnl_arinyo_kp'] = get_par('dnl_arinyo_kp')

        # HCDs
        if self.options['hcd_model'] is not None:
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
            if self.options['fullshape_smoothing'] in ['gauss', 'exp']:
                new_params['per_sigma_smooth'] = get_par('per_sigma_smooth')
            if self.options['fullshape_smoothing'] == 'exp':
                new_params['par_exp_smooth'] = get_par('par_exp_smooth')
                new_params['per_exp_smooth'] = get_par('per_exp_smooth')

        # DESI instrumental systematics amplitude
        if self.options['desi-instrumental-systematics']:
            new_params['desi_inst_sys_amp'] = get_par('desi_inst_sys_amp')

        # Check for broadband parameters
        for name, value in parameters.items():
            if 'BB' in name and name not in new_params:
                new_params[name] = value

        self._parameters = new_params

    @staticmethod
    def get_lya_bias(z):
        """Compute default Lya bias at redshift z

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Default Lya bias
        """
        return -0.1167 * ((1 + z) / (1 + 2.334))**2.9

    @staticmethod
    def get_qso_bias(z):
        """Compute default QSO bias at redshift z

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Default QSO bias
        """
        return 3.91 * ((1 + z) / (1 + 2.39))**1.7133

    @staticmethod
    def get_growth_rate(z, Omega_m=0.3153):
        """Compute default growth rate at redshift z

        Parameters
        ----------
        z : float
            Redshift
        Omega_m : float, optional
            Matter fraction, by default 0.3153

        Returns
        -------
        float
            Default growth rate
        """
        Omega_m_z = Omega_m * ((1 + z)**3) / (Omega_m * ((1 + z)**3) + 1 - Omega_m)
        Omega_lambda_z = 1 - Omega_m_z
        growth_rate = (Omega_m_z**0.6) + (Omega_lambda_z / 70.) * (1 + Omega_m_z / 2.)
        return growth_rate
