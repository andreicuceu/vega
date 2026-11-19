#!/usr/bin/env python
"""Build the Vega configs for the BAO fit of the uncontaminated LyaCoLoRe 2LPT mocks.

Joint BAO fit of the four Lya forest correlation functions, Lya(Lya)xLya(Lya),
Lya(Lya)xLya(Lyb), Lya(Lya)xQSO and Lya(Lyb)xQSO, measured from the stack of the 100
realisations (0-99) of the LyaCoLoRe 2LPT DESI DR2-like mocks (qq_desi_y3 v3.0.5) without
astrophysical contaminants (loa-0), with the joint covariance matrix of the stack.

The mocks are analysed as the data: the deltas are computed from a continuum fit, so the
correlations are distorted by the continuum fitting and the model is multiplied by the
distortion matrices. The distortion matrices are only computed for realisation 0, and are
used for the stack. The QSO redshifts in these mocks have no errors (zcat.fits catalogues).

The cosmological setup is the standard anisotropic BAO fit: only the peak component is
rescaled, by alpha_parallel (ap) and alpha_transverse (at), relative to the linear power
spectrum used to generate the mocks (Abacus-Colore-2LPT template, z = 2.33). The growth
rate is fixed to the value of the template.

The model is the same as for the true-continuum mocks (../../raw_and_true-cont): Kaiser
model with power-law bias evolution and non-linear broadening of the BAO peak, without
metals, HCDs, UV background fluctuations, instrumental systematics, QSO redshift errors or
proximity effect. The small-scale smoothing of the LyaCoLoRe transmission field is
modelled by a Gaussian damping of the power spectrum, with one pair of smoothing scales
(par_sigma_smooth, per_sigma_smooth) for all tracers. The alternative treatment, based on
the LyaCoLoRe cell size, is in ../los_smoothing. In addition, the bins below r-min are
analytically marginalised, as in the data examples, because through the distortion matrix
they contribute to the model at larger separations.

The configs are written to this directory with paths relative to it, so Vega must be run
from this directory:

    python build_configs.py
    python ../../../../bin/run_vega.py main.ini          # minimizer
    srun python ../../../../bin/run_vega_mpi.py main.ini # PolyChord sampler (compute nodes)

Do not edit the generated .ini files by hand; modify this script and re-run it.
"""

import os
from pathlib import Path

from vega import BuildConfig

EXAMPLE_DIR = Path(__file__).resolve().parent

# LyaCoLoRe 2LPT mocks, v3.0.5, without contaminants: stack of realisations 0-99, and the
# single realisation for which the distortion matrices were computed
MOCK_ANALYSIS_DIR = Path(
    "/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/lyacolore_2lpt/"
    "qq_desi_y3/v3.0.5"
)
STACK_DIR = MOCK_ANALYSIS_DIR / "analysis-stack/loa-0/baseline/correlations"
DISTORTION_DIR = MOCK_ANALYSIS_DIR / "analysis-0/loa-0/baseline/correlations"
GLOBAL_COVARIANCE = STACK_DIR / "full_cov.fits"

# Correlations in the joint fit, following the Vega naming convention
FIT_TYPE = "lyaxlya_lyaxlyb_lyaxqso_lybxqso"

# Separation range of the fit [Mpc/h], as in the other BAO examples
RMIN_AUTO = 30.0
RMIN_CROSS = 40.0
RMAX = 180.0

# The full rp range of the measurement is kept: |rp| <= 200 Mpc/h in the stack, and the cut
# at 300 Mpc/h removes no bins.
RP_MIN = -300.0


def get_correlation_setup():
    """Collect the measured correlations, distortion matrices and scale cuts.

    Returns
    -------
    dict
        Correlation name (e.g. 'lyaxqso') -> dictionary with the paths of the stacked
        correlation ('corr_path') and of the distortion matrix of realisation 0
        ('distortion-file'), together with the scale cuts ('r-min', 'r-max', 'rp-min'),
        in the format expected by ``BuildConfig.build``.
    """
    # Stacked correlation and distortion matrix of each correlation function
    data_products = {
        "lyaxlya": ("cf_lya_lya_0_10-exp.fits.gz", "dmat_lya_lya_0_10.fits"),
        "lyaxlyb": ("cf_lya_lyb_0_10-exp.fits.gz", "dmat_lya_lyb_0_10.fits"),
        "lyaxqso": ("xcf_lya_qso_0_10-exp.fits.gz", "xdmat_lya_qso_0_10.fits"),
        "lybxqso": ("xcf_lyb_qso_0_10-exp.fits.gz", "xdmat_lyb_qso_0_10.fits"),
    }

    correlations = {}
    for name, (corr_file, dmat_file) in data_products.items():
        is_cross = "qso" in name

        correlations[name] = {
            "corr_path": str(STACK_DIR / corr_file),
            "distortion-file": str(DISTORTION_DIR / dmat_file),
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
    # Analytic marginalisation of all bins below r-min, keeping them in the model and
    # matching the data binning
    "marginalize-all-rmin-cuts": True,
    "fit-marginalized-scales": True,
    "marginalize-match-data-bins": True,
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
    """Write the main and correlation configs of the uncontaminated mock BAO fit.

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
