#!/usr/bin/env python
"""Build the Vega configs for the BAO fit of the contaminated LyaCoLoRe 2LPT mocks.

Joint BAO fit of the four Lya forest correlation functions, Lya(Lya)xLya(Lya),
Lya(Lya)xLya(Lyb), Lya(Lya)xQSO and Lya(Lyb)xQSO, measured from the stack of realisations
0-399 of the LyaCoLoRe 2LPT DESI DR2-like mocks (qq_desi_y3 v3.0.5) with astrophysical
contaminants and QSO redshift errors (loa-124), with the joint covariance matrix of the
stack.

The quickquasars spectra of these mocks include:
- DLAs, masked in the analysis using a DLA catalogue (NHI > 10^20.3 cm^-2, S/N > 2,
  completeness 0.75), so that undetected DLAs and lower column density HCDs remain;
- metal absorption by SiII(1190), SiII(1193), SiIII(1207) and SiII(1260);
- BALs, masked in the analysis;
- Gaussian QSO redshift errors with a dispersion of 400 km/s (zcat_gauss_400.fits),
  applied to the QSO positions in the cross-correlations.

The deltas are computed from a continuum fit, so the model is multiplied by the distortion
matrices. The distortion matrices are only computed for realisation 0, and are used for
the stack. The metal matrices are computed by Vega from the forest weights of realisation 0
and the QSO catalogue of realisation 0.

The cosmological setup is the standard anisotropic BAO fit: only the peak component is
rescaled, by alpha_parallel (ap) and alpha_transverse (at), relative to the linear power
spectrum used to generate the mocks (Abacus-Colore-2LPT template, z = 2.33). The growth
rate is fixed to the value of the template.

The model extends that of the uncontaminated mocks (../uncontaminated/gauss_smoothing),
i.e. Kaiser model with power-law bias evolution, non-linear broadening of the BAO peak,
Gaussian damping of the smooth component with global smoothing scales, and analytic
marginalisation of the bins below r-min, with:
- HCD contamination (Rogers et al. 2018 template), with Gaussian priors on beta_hcd
  (as in the data analysis) and L0_hcd;
- the four Si metals, with free biases and RSD parameters fixed to 1.5, as the metal
  absorption follows the Lya transmission field of the mocks, and the same Gaussian
  smoothing as the Lya forest;
- a Gaussian velocity dispersion of the QSOs (sigma_velo_disp_gauss_QSO), describing the
  Gaussian redshift errors, and a free shift of the cross-correlations (drp_QSO) with the
  prior of the data analysis.

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

# LyaCoLoRe 2LPT mocks, v3.0.5, with contaminants: stack of realisations 0-399, and the
# single realisation for which the distortion matrices were computed
MOCK_ANALYSIS_DIR = Path(
    "/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/lyacolore_2lpt/"
    "qq_desi_y3/v3.0.5"
)
STACK_DIR = MOCK_ANALYSIS_DIR / "analysis-stack/loa-124/baseline/correlations"
REALISATION_0_DIR = MOCK_ANALYSIS_DIR / "analysis-0/loa-124/baseline"
DISTORTION_DIR = REALISATION_0_DIR / "correlations"
GLOBAL_COVARIANCE = STACK_DIR / "full_cov.fits"

# Weights used to compute the metal matrices: forest weights and QSO catalogue (with
# redshift errors, as used in the cross-correlations) of realisation 0
LYA_DELTA_ATTRIBUTES = REALISATION_0_DIR / "deltas_lya/Log/delta_attributes.fits.gz"
LYB_DELTA_ATTRIBUTES = REALISATION_0_DIR / "deltas_lyb/Log/delta_attributes.fits.gz"
QSO_CATALOG = Path(
    "/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/"
    "v3.0.5/mock-0/loa-124/zcat_gauss_400.fits"
)

# Correlations in the joint fit, following the Vega naming convention
FIT_TYPE = "lyaxlya_lyaxlyb_lyaxqso_lybxqso"

# Metal lines included in the quickquasars spectra of the mocks
METALS = ["SiII(1190)", "SiII(1193)", "SiIII(1207)", "SiII(1260)"]

# Separation range of the fit [Mpc/h], as in the other BAO examples
RMIN_AUTO = 30.0
RMIN_CROSS = 40.0
RMAX = 180.0

# The full rp range of the measurement is kept: |rp| <= 200 Mpc/h in the stack, and the cut
# at 300 Mpc/h removes no bins.
RP_MIN = -300.0


def get_correlation_setup():
    """Collect the data products and scale cuts of each correlation function.

    Returns
    -------
    dict
        Correlation name (e.g. 'lyaxqso') -> dictionary with the paths of the stacked
        correlation ('corr_path'), of the distortion matrix of realisation 0
        ('distortion-file') and of the weight files used to compute the metal matrices
        ('weights-tracer1' and 'weights-tracer2'), together with the scale cuts ('r-min',
        'r-max', 'rp-min') and the 'fast_metals' flag, in the format expected by
        ``BuildConfig.build``.
    """
    # Stacked correlation, distortion matrix, and weight files of tracer1 and tracer2.
    # In the Vega convention the forest is tracer1 and the quasars tracer2 in the crosses.
    data_products = {
        "lyaxlya": (
            "cf_lya_lya_0_10-exp.fits.gz",
            "dmat_lya_lya_0_10.fits",
            LYA_DELTA_ATTRIBUTES,
            LYA_DELTA_ATTRIBUTES,
        ),
        "lyaxlyb": (
            "cf_lya_lyb_0_10-exp.fits.gz",
            "dmat_lya_lyb_0_10.fits",
            LYA_DELTA_ATTRIBUTES,
            LYB_DELTA_ATTRIBUTES,
        ),
        "lyaxqso": (
            "xcf_lya_qso_0_10-exp.fits.gz",
            "xdmat_lya_qso_0_10.fits",
            LYA_DELTA_ATTRIBUTES,
            QSO_CATALOG,
        ),
        "lybxqso": (
            "xcf_lyb_qso_0_10-exp.fits.gz",
            "xdmat_lyb_qso_0_10.fits",
            LYB_DELTA_ATTRIBUTES,
            QSO_CATALOG,
        ),
    }

    correlations = {}
    for name, (corr_file, dmat_file, weights1, weights2) in data_products.items():
        is_cross = "qso" in name

        correlations[name] = {
            "corr_path": str(STACK_DIR / corr_file),
            "distortion-file": str(DISTORTION_DIR / dmat_file),
            "weights-tracer1": str(weights1),
            "weights-tracer2": str(weights2),
            "r-min": RMIN_CROSS if is_cross else RMIN_AUTO,
            "r-max": RMAX,
            "rp-min": RP_MIN,
            # Compute the metal correlations with the fast (approximate) method
            "fast_metals": "True",
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
    # Apply the Gaussian small-scale smoothing to the smooth component only, not to the
    # BAO peak
    "skip-nl-model-in-peak": True,
    # Gaussian damping of the power spectrum, describing the smoothing of the LyaCoLoRe
    # field on small scales. A single global pair of smoothing scales is used because only
    # par_sigma_smooth and per_sigma_smooth are given in PARAMETERS. The metal absorption
    # is computed from the same transmission field, so the smoothing is also applied to the
    # metal correlations.
    "fullshape_smoothing": "gauss",
    "fullshape_smoothing_metals": True,
    # Gaussian velocity dispersion of the QSOs, describing the Gaussian redshift errors
    "velocity_dispersion": "gauss",
    # HCD contamination
    "hcd_model": "Rogers2018",
    # Metals, with distortion matrices computed by Vega from the forest and QSO weights
    "metals": METALS,
    "new_metals": True,
    "rp_only_metal_mats": False,
    "use_metal_autos": True,
    # Analytic marginalisation of all bins below r-min, keeping them in the model and
    # matching the data binning
    "marginalize-all-rmin-cuts": True,
    "fit-marginalized-scales": True,
    "marginalize-match-data-bins": True,
}

# Fiducial (starting) values of the model parameters, from a previous fit of the v2.8.5
# contaminated mocks. Parameters not listed here take the values of
# vega/templates/parameters.ini.
PARAMETERS = {
    # BAO
    "ap": 1.0,
    "at": 1.0,
    "sigmaNL_per": 3.24,
    "sigmaNL_par": 6.36984,
    "bao_amp": 1.0,
    # Lya and QSO biases and RSD. The growth rate is replaced by the value of the template
    # (F_ZREF) when Vega is initialised.
    "bias_LYA": -0.15202,
    "beta_LYA": 1.195,
    "alpha_LYA": 2.9,
    "bias_QSO": 3.4,
    "bias_eta_QSO": 1.0,
    "growth_rate": 0.970470511472463,
    "alpha_QSO": 1.44,
    # QSO redshift errors: Gaussian velocity dispersion [Mpc/h] and shift of the cross
    # along the line of sight [Mpc/h]
    "sigma_velo_disp_gauss_QSO": 5.09,
    "drp_QSO": -0.572,
    # HCDs
    "bias_hcd": -0.0172,
    "beta_hcd": 1.15,
    "L0_hcd": 6.5,
    # Metals: bias, RSD parameter and redshift evolution exponent of the bias. The metal
    # absorption follows the Lya transmission field, so the RSD parameters are fixed to a
    # value close to beta_LYA.
    "bias_SiII(1190)": -0.00289,
    "beta_SiII(1190)": 1.5,
    "alpha_SiII(1190)": 1.0,
    "bias_SiII(1193)": -0.00204,
    "beta_SiII(1193)": 1.5,
    "alpha_SiII(1193)": 1.0,
    "bias_SiIII(1207)": -0.00594,
    "beta_SiIII(1207)": 1.5,
    "alpha_SiIII(1207)": 1.0,
    "bias_SiII(1260)": -0.00341,
    "beta_SiII(1260)": 1.5,
    "alpha_SiII(1260)": 1.0,
    # Gaussian smoothing scales [Mpc/h] along and across the line of sight
    "par_sigma_smooth": 2.0,
    "per_sigma_smooth": 2.0,
}

# Sampled parameters. 'True' uses the default limits of vega/parameters/default_values.txt;
# otherwise the value is "min max start sigma".
SAMPLE_PARAMS = {
    "ap": "True",
    "at": "True",
    "sigmaNL_per": "True",
    "sigmaNL_par": "True",
    "bias_LYA": "True",
    "beta_LYA": "True",
    "bias_QSO": "True",
    "sigma_velo_disp_gauss_QSO": "True",
    "drp_QSO": "True",
    "bias_hcd": "-1.0 0.0 -0.0172 0.01",
    "beta_hcd": "0.0 3.0 1.15 0.1",
    "L0_hcd": "0.0 10.0 6.5 0.1",
    "bias_SiII(1190)": "True",
    "bias_SiII(1193)": "True",
    "bias_SiIII(1207)": "True",
    "bias_SiII(1260)": "True",
    "par_sigma_smooth": "True",
    "per_sigma_smooth": "True",
}

# Gaussian priors, "gaussian mean sigma". The priors on drp_QSO and beta_hcd are those of
# the DESI data analysis; the prior on L0_hcd is wider than in the data (sigma = 1 Mpc/h).
PRIORS = {
    "drp_QSO": "gaussian 0.0 1.0",
    "beta_hcd": "gaussian 0.50 0.09",
    "L0_hcd": "gaussian 5.0 2.0",
}

FIT_INFO = {
    # zeff is not given, so BuildConfig computes it from the measured correlations
    "global_cov_file": str(GLOBAL_COVARIANCE),
    # Lya: (bias, beta) sampled directly; QSO: beta_QSO = bias_eta_QSO * f / bias_QSO
    "bias_beta_config": {"LYA": "bias_beta", "QSO": "bias_bias_eta"},
    "sample_params": SAMPLE_PARAMS,
    "priors": PRIORS,
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
    """Write the main and correlation configs of the contaminated mock BAO fit.

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
