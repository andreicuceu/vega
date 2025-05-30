#!/usr/bin/env python
import matplotlib.pyplot as plt

from vega import VegaInterface


def run_vega(config_path):
    # Initialize Vega
    vega = VegaInterface(config_path)

    # run compute_model once to initialize all the caches
    _ = vega.compute_model(run_init=False)

    # Check if we need to run over a Monte Carlo mock
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if run_montecarlo and vega.mc_config is not None:
        _ = vega.initialize_monte_carlo()
    elif run_montecarlo:
        raise ValueError('You asked to run over a Monte Carlo simulation,'
                         ' but no "[monte carlo]" section provided.')

    # Run minimizer
    vega.minimize()

    # Run chi2scan
    scan_results = None
    if 'chi2 scan' in vega.main_config:
        scan_results = vega.analysis.chi2_scan()

    # Write output
    if vega.minimizer is not None:
        for par, val in vega.bestfit.values.items():
            vega.params[par] = val
    corr_funcs = vega.compute_model(vega.params, run_init=False)
    vega.output.write_results(corr_funcs, vega.params, vega.minimizer, scan_results, vega.models)

    plt.rc('axes', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    num_pars = len(vega.sample_params['limits'])
    for name in vega.plots.data:
        bestfit_legend = f'Correlation: {name}, Total '
        bestfit_legend += r'$\chi^2_\mathrm{best}/(N_\mathrm{data}-N_\mathrm{pars})$'
        bestfit_legend += f': {vega.chisq:.1f}/({vega.total_data_size}-{num_pars}) '
        bestfit_legend += f'= {vega.reduced_chisq:.3f}, PTE={vega.p_value:.2f}'
        if not vega.bestfit.fmin.is_valid:
            bestfit_legend = 'Invalid fit! Disregard these results.'

        vega.plots.plot_4wedges(models=[vega.bestfit_model[name]], corr_name=name, title=None,
                                mu_bin_labels=True, no_font=True, model_colors=['r'], xlim=None)
        vega.plots.fig.suptitle(bestfit_legend, fontsize=18, y=1.03)
        vega.plots.fig.savefig(f'{vega.output.outfile[:-5]}_{name}.png', dpi='figure',
                               bbox_inches='tight', facecolor='white')
