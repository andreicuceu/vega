import argparse
from vega import BuildConfig


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Create config files for vega.')

    parser.add_argument('--fit-name', type=str, default='lyalya_lyalya', required=True,
                        help=('Name of the fit. Includes the name of the correlations with the two'
                              ' tracers separated by a single underscore (e.g. lyalya_qso),'
                              ' and different correlations separated by a double underscore'
                              ' (e.g. lyalya_lyalya__lyalya_qso). If unsure check the templates'
                              ' folder in Vega to see all possibilities.'))

    parser.add_argument('--corr-paths', type=str, nargs='*', default=None, required=True,
                        help='Paths to all measured correlations files.')

    parser.add_argument('--out-path', type=str, default=None, required=True,
                        help='Directory to write the config files into')

    parser.add_argument('--zeff', type=float, default=None, required=True,
                        help='Effective redshift')

    parser.add_argument('--sample-params', type=str, nargs='*', default=None, required=True,
                        help='List of parameters to sample/fit.')

    parser.add_argument('--sampler', type=bool, default=False, required=False,
                        help='Run the sampler.')

    parser.add_argument('--rmin-values', type=float, nargs='*', default=[40.], required=False,
                        help='Minimum separation',)

    parser.add_argument('--rmax-values', type=float, nargs='*', default=[160.], required=False,
                        help='Maximum separation',)

    parser.add_argument('--scale-params', type=str, default='ap_at', required=False,
                        help='Scale parameters model. Choose from [ap_at, phi_gamma]')

    parser.add_argument('--metals', type=str, nargs='*', default=None, required=False,
                        help=('Names of metals to include. Choose from [all, SiII(1190),'
                              'SiII(1193), SiIII(1207), SiII(1260), CIV(eff)]'))

    parser.add_argument('--metal-paths', type=str, nargs='*', default=None, required=False,
                        help='Paths to all metal matrices that are needed.')

    parser.add_argument('--template', type=str, default='PlanckDR16/PlanckDR16.fits',
                        required=False, help='Path to the template. Can be relative to Vega.')

    parser.add_argument('--small-scale-nl', type=bool, default=False, required=False,
                        help='Small scale non-linear model for the Lya Auto. (Arinyo model)')

    parser.add_argument('--bao-broadening', type=bool, default=False, required=False,
                        help='Non linear broadening of the BAO peak')

    parser.add_argument('--uv-background', type=bool, default=False, required=False,
                        help='Directory to write the config files into')

    parser.add_argument('--velocity-dispersion', type=str, default=None, required=False,
                        help='Model velocity dispersion for discrete tracers.')

    parser.add_argument('--radiation-effects', type=bool, default=False, required=False,
                        help='QSO radiation effects')

    parser.add_argument('--hcd-model', type=str, default=None, required=False,
                        help='HCD model. Choose from [Rogers2018, mask, sinc]')

    parser.add_argument('--fvoigt-model', type=str, default=None, required=False,
                        help='Name of fvoigt model. Must be in the models folder in Vega.')

    parser.add_argument('--fullshape-smoothing', type=str, default=None, required=False,
                        help='Full-shape smoothing model. Choose from [gauss, exp]')

    parser.add_argument('--binsizes', type=float, default=[4.], required=False,
                        help='Binsizes for each correlation.')

    parser.add_argument('--full-shape', type=bool, default=False, required=False,
                        help='Run full shape fit')

    parser.add_argument('--smooth-scaling', type=bool, default=False, required=False,
                        help='Run with rescaling of the smooth component')

    args = parser.parse_args()

    options = {}
    options['scale_params'] = args.scale_params
    options['metals'] = args.metals
    options['template'] = args.template
    options['small_scale_nl'] = args.small_scale_nl
    options['bao_broadening'] = args.bao_broadening
    options['uv_background'] = args.uv_background
    options['velocity_dispersion'] = args.velocity_dispersion
    options['radiation_effects'] = args.radiation_effects
    options['hcd_model'] = args.hcd_model
    options['fvoigt_model'] = args.fvoigt_model
    options['fullshape_smoothing'] = args.fullshape_smoothing
    options['full_shape'] = args.full_shape
    options['smooth_scaling'] = args.smooth_scaling

    corr_names = args.fit_name.split('__')
    correlations = {}
    for i, name in enumerate(corr_names):
        correlations[name] = {}
        correlations[name]['corr_path'] = args.corr_paths[i]

        if len(args.rmin_values) > 1:
            correlations[name]['r-min'] = args.rmin_values[i]
        else:
            correlations[name]['r-min'] = args.rmin_values[0]
        if len(args.rmax_values) > 1:
            correlations[name]['r-max'] = args.rmax_values[i]
        else:
            correlations[name]['r-max'] = args.rmax_values[0]

        if len(args.binsizes) > 1:
            correlations[name]['binsize'] = args.binsizes[i]
        else:
            correlations[name]['binsize'] = args.binsizes[0]

        if args.metals is not None:
            correlations[name]['metal_path'] = args.metal_paths[i]

    fit_info = {}
    fit_info['fitter'] = True
    fit_info['zeff'] = args.zeff
    fit_info['sample_params'] = args.sample_params
    fit_info['use_bias_eta'] = {}
    fit_info['sampler'] = args.sampler
    if args.sampler:
        fit_info['Polychord'] = {}

    print('\nBuilding config files for Vega in: {} \n'.format(args.out_path))

    config_builder = BuildConfig(options=options)
    main_path = config_builder.build(correlations=correlations, fit_type=args.fit_name,
                                     fit_info=fit_info, out_path=args.out_path)

    print('Successfully built Vega config files. The main.ini file is: {} \n'.format(main_path))
