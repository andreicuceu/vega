#!/usr/bin/env python
"""Build the Vega configs for the BAO fit of the true-continuum LyaCoLoRe 2LPT mocks.

Joint BAO fit of the four Lya forest correlation functions, Lya(Lya)xLya(Lya),
Lya(Lya)xLya(Lyb), Lya(Lya)xQSO and Lya(Lyb)xQSO, measured from the stack of
realisations 200-300 of the LyaCoLoRe 2LPT DESI DR2-like mocks (qq_desi_y3 v2.8.5),
without astrophysical contaminants (loa-0), with the joint covariance matrix of the stack.

In the true-continuum analysis the deltas are computed from the true unabsorbed quasar
continua instead of from a continuum fit, so the measured correlations are not distorted
by the continuum fitting and no distortion matrix is needed. The raw analysis (deltas
computed directly from the transmitted flux fraction) has the same model; to fit raw
correlations only the correlation and covariance paths below need to be changed.

The cosmological setup is the standard anisotropic BAO fit: only the peak component is
rescaled, by alpha_parallel (ap) and alpha_transverse (at), relative to the linear power
spectrum used to generate the mocks (Abacus-Colore-2LPT template, z = 2.33). The growth
rate is fixed to the value of the template.

The model of the correlation functions is the Kaiser model with power-law bias evolution
and non-linear broadening of the BAO peak, without the components that describe effects
absent from these mocks (metals, HCDs, UV background fluctuations, instrumental
systematics, QSO redshift errors and proximity effect). The smoothing of the LyaCoLoRe
transmission field on small scales is modelled by a Gaussian damping of the power
spectrum, with one pair of smoothing scales (par_sigma_smooth, per_sigma_smooth) for all
tracers.

The configs are written to this directory with paths relative to it, so Vega must be run
from this directory:

    python build_configs.py
    python ../../../bin/run_vega.py main.ini          # minimizer
    srun python ../../../bin/run_vega_mpi.py main.ini # PolyChord sampler (compute nodes)

Do not edit the generated .ini files by hand; modify this script and re-run it.
"""

import os
from pathlib import Path

from vega import BuildConfig

EXAMPLE_DIR = Path(__file__).resolve().parent

# Stacked true-continuum correlations of realisations 200-300 of the LyaCoLoRe 2LPT mocks
MOCK_STACK_DIR = Path(
    "/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/lyacolore_2lpt/"
    "qq_desi_y3/v2.8.5/analysis-stack/loa-0/true_cont/correlations"
)
GLOBAL_COVARIANCE = MOCK_STACK_DIR / "full_cov_v200-300.fits"

# Correlations in the joint fit, following the Vega naming convention
FIT_TYPE = "lyaxlya_lyaxlyb_lyaxqso_lybxqso"

# Separation range of the fit [Mpc/h], as in the DESI data BAO example. Without a distortion
# matrix the bins below r-min do not affect the model, so they are not marginalised.
RMIN_AUTO = 30.0
RMIN_CROSS = 40.0
RMAX = 180.0

# The full rp range of the measurement is kept: |rp| <= 200 Mpc/h in the stack, and the cut
# at 300 Mpc/h removes no bins.
RP_MIN = -300.0


def get_correlation_setup():
    """Collect the measured correlations and scale cuts of each correlation function.

    Returns
    -------
    dict
        Correlation name (e.g. 'lyaxqso') -> dictionary with the path of the measured
        correlation ('corr_path') and the scale cuts ('r-min', 'r-max', 'rp-min'), in the
        format expected by ``BuildConfig.build``. No distortion matrix is given, as the
        true-continuum correlations are not distorted by continuum fitting.
    """
    correlation_files = {
        "lyaxlya": "cf_lya_lya_0_10_v200-300-exp.fits.gz",
        "lyaxlyb": "cf_lya_lyb_0_10_v200-300-exp.fits.gz",
        "lyaxqso": "xcf_lya_qso_0_10_v200-300-exp.fits.gz",
        "lybxqso": "xcf_lyb_qso_0_10_v200-300-exp.fits.gz",
    }

    correlations = {}
    for name, corr_file in correlation_files.items():
        is_cross = "qso" in name

        correlations[name] = {
            "corr_path": str(MOCK_STACK_DIR / corr_file),
            "r-min": RMIN_CROSS if is_cross else RMIN_AUTO,
            "r-max": RMAX,
            "rp-min": RP_MIN,
        }

    return correlations


# Model options common to all correlations (see the BuildConfig docstring)
MODEL_OPTIONS = {
    # Cosmological setup: BAO-only fit, rescaling of the peak by (ap, at), with the linear
    # power spectrum used to generate the mocks as template
    "scale_params": "ap_at",
    "template": "Abacus-Colore-2LPT/Abacus-Colore-2LPT.fits",
    "full_shape": False,
    "full_shape_alpha": False,
    "smooth_scaling": False,
    # Non-linear broadening of the BAO peak (sigmaNL_per, sigmaNL_par)
    "bao_broadening": True,
    # Gaussian damping of the power spectrum, describing the smoothing of the LyaCoLoRe
    # field on small scales. A single global pair of smoothing scales is used because only
    # par_sigma_smooth and per_sigma_smooth are given in PARAMETERS.
    "fullshape_smoothing": "gauss",
}

# Fiducial (starting) values of the model parameters. Parameters not listed here take the
# values of vega/templates/parameters.ini.
PARAMETERS = {
    # BAO
    "ap": 1.0,
    "at": 1.0,
    "sigmaNL_per": 3.24,
    "sigmaNL_par": 6.36984,
    "bao_amp": 1.0,
    # Lya and QSO biases and RSD. The growth rate is replaced by the value of the template
    # (F_ZREF) when Vega is initialised.
    "bias_LYA": -0.11356,
    "beta_LYA": 1.67,
    "alpha_LYA": 2.9,
    "bias_QSO": 3.4,
    "bias_eta_QSO": 1.0,
    "growth_rate": 0.970470511472463,
    "alpha_QSO": 1.44,
    # No QSO redshift errors in these mocks (zcat.fits catalogues): no shift of the cross
    "drp_QSO": 0.0,
    # Gaussian smoothing scales [Mpc/h] along and across the line of sight
    "par_sigma_smooth": 2.0,
    "per_sigma_smooth": 2.0,
}

# Sampled parameters. 'True' uses the default limits of vega/parameters/default_values.txt.
SAMPLE_PARAMS = {
    "ap": "True",
    "at": "True",
    "sigmaNL_per": "True",
    "sigmaNL_par": "True",
    "bias_LYA": "True",
    "beta_LYA": "True",
    "bias_QSO": "True",
    "par_sigma_smooth": "True",
    "per_sigma_smooth": "True",
}

FIT_INFO = {
    # zeff is not given, so BuildConfig computes it from the measured correlations
    "global_cov_file": str(GLOBAL_COVARIANCE),
    # Lya: (bias, beta) sampled directly; QSO: beta_QSO = bias_eta_QSO * f / bias_QSO
    "bias_beta_config": {"LYA": "bias_beta", "QSO": "bias_bias_eta"},
    "sample_params": SAMPLE_PARAMS,
    "run_sampler": True,
    "sampler": "Polychord",
    "Polychord": {
        "num_live": "512",
        "num_repeats": "50",
        "do_clustering": "True",
        "boost_posterior": "5",
    },
}


def main():
    """Write the main and correlation configs of the true-continuum mock BAO fit.

    The working directory is changed to the example directory so that BuildConfig writes
    paths relative to it (correlation configs, output_fitter/ and output_sampler/).
    Existing configs are overwritten.

    Returns
    -------
    pathlib.Path
        Path of the main config, relative to the example directory.
    """
    os.chdir(EXAMPLE_DIR)

    config_builder = BuildConfig(MODEL_OPTIONS, overwrite=True)
    main_config_path = config_builder.build(
        correlations=get_correlation_setup(),
        fit_type=FIT_TYPE,
        fit_info=FIT_INFO,
        out_path=".",
        parameters=PARAMETERS,
    )

    print(f"Wrote {EXAMPLE_DIR / main_config_path}")
    return main_config_path


if __name__ == "__main__":
    main()
