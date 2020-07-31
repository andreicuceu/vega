from lyafit import lyafit, sampler_interface
import argparse

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Lyafit in parallel.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    # Initialize LyaFit
    lf = lyafit.LyaFit(args.config)

    # Run sampler
    if lf.has_sampler:
        sampler = sampler_interface.Sampler(lf.main_config['Polychord'],
                                            lf.sample_params['limits'],
                                            lf.log_lik)

        sampler.run()
