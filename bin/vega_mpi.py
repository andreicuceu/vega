from vega import VegaInterface
from vega.sampler_interface import Sampler
import argparse

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    # Initialize Vega
    vega = VegaInterface(args.config)

    # Run sampler
    if vega.has_sampler:
        sampler = Sampler(vega.main_config['Polychord'],
                          vega.sample_params['limits'], vega.log_lik)

        sampler.run()
