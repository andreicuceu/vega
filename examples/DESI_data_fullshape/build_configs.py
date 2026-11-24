#!/usr/bin/env python
"""Build the Vega configs for the DESI DR2 Lya full-shape analysis.

Joint fit of the four Lya forest correlation functions measured from the DESI DR2 (loa)
data set with the full-shape (y3-fs v1-4-0-0) data products: Lya(Lya)xLya(Lya),
Lya(Lya)xLya(Lyb), Lya(Lya)xQSO and Lya(Lyb)xQSO, with the smoothed joint covariance matrix.

The cosmological setup is the Lya full-shape (Alcock-Paczynski) fit of Cuceu et al. (2021,
arXiv:2103.14075): the peak and smooth components are rescaled independently, with
(phi, alpha) for the peak and (phi_smooth, alpha_smooth) for the smooth component, where
phi = at / ap and alpha = sqrt(ap * at). The AP information comes from phi_smooth. The
growth rate f is sampled, rather than taken from the Planck 2018 template at z = 2.33,
and is constrained by the QSO redshift-space distortions (beta_QSO = f / bias_QSO).

The model of the correlation functions follows the DESI DR3 BAO baseline:
- Kaiser model with power-law bias evolution, and Arinyo-i-Prats et al. (2015) small-scale
  non-linear correction in the Lya auto-correlations (applied to the smooth component only);
- non-linear broadening of the BAO peak;
- HCD contamination (Rogers et al. 2018 template);
- eight metal contaminants, with the metal distortion matrices computed by Vega
  (new_metals) from the redshift distributions of the forest pixels and quasars;
- UV background fluctuations;
- QSO velocity dispersion (Lorentzian), redshift errors (drp_QSO) and transverse proximity
  effect (radiation effects) in the cross-correlations;
- DESI instrumental systematics (correlated sky-subtraction noise) in the autos;
- analytic marginalisation of the small-scale bins below the r-min cut.

The configs are written to this directory with paths relative to it, so Vega must be run
from this directory:

    python build_configs.py
    python ../../bin/run_vega.py main.ini          # minimizer
    srun python ../../bin/run_vega_mpi.py main.ini # PolyChord sampler (compute nodes)

Do not edit the generated .ini files by hand; modify this script and re-run it.
"""

import os
from pathlib import Path

from vega import BuildConfig

EXAMPLE_DIR = Path(__file__).resolve().parent

# DESI DR2 (loa) Lya full-shape data products (y3-fs validation tests, version v1-4-0-0)
FULLSHAPE_DIR = Path(
    "/global/cfs/cdirs/desicollab/science/lya/y3-fs/validation-tests/v1-4-0-0/baseline"
)
CORRELATIONS_DIR = FULLSHAPE_DIR / "correlations/results"
LYA_DELTA_ATTRIBUTES = FULLSHAPE_DIR / "deltas/results/lya/Log/delta_attributes.fits.gz"
LYB_DELTA_ATTRIBUTES = FULLSHAPE_DIR / "deltas/results/lyb/Log/delta_attributes.fits.gz"
# Catalogue with the alternative BAL masking used for the full-shape measurement
QSO_CATALOG = Path(
    "/global/cfs/cdirs/desicollab/users/martini/bal-catalogs/loa/"
    "QSO_cat_loa_main_dark_healpix_v3-altbal.fits"
)
GLOBAL_COVARIANCE = FULLSHAPE_DIR / "fits/results/full-covariance-smoothed.fits"

# Correlations in the joint fit, following the Vega naming convention
FIT_TYPE = "lyaxlya_lyaxlyb_lyaxqso_lybxqso"

# Metal lines contaminating the Lya and Lyb regions of the forest. FeII(1145), NV(1239) and
# SiII(1304) are only weakly constrained, so their biases are given bounded ranges below.
METALS = [
    "FeII(1145)",
    "SiII(1190)",
    "SiII(1193)",
    "SiIII(1207)",
    "NV(1239)",
    "SiII(1260)",
    "SiII(1304)",
    "CIV(eff)",
]

# Separation range of the fit [Mpc/h]. The QSO cross-correlations start at larger r-min
# because of the stronger non-linearities (velocity dispersion, proximity effect) of the QSO
# environment. The bins below r-min are analytically marginalised (see MODEL_OPTIONS).
# The full-shape fit extends to r-max = 200 Mpc/h (180 Mpc/h in the BAO fit) to include more
# of the broadband information.
RMIN_AUTO = 30.0
RMIN_CROSS = 40.0
RMAX = 200.0

# The full rp range of the measurement is kept: |rp| <= 200 Mpc/h in the data, and the cut
# at 300 Mpc/h removes no bins.
RP_MIN = -300.0


def get_correlation_setup():
    """Collect the data products and scale cuts of each correlation function.

    Returns
    -------
    dict
        Correlation name (e.g. 'lyaxqso') -> dictionary with the paths of the measured
        correlation ('corr_path'), its distortion matrix ('distortion-file') and the
        weight files used to compute the metal matrices ('weights-tracer1' and
        'weights-tracer2'), together with the scale cuts ('r-min', 'r-max', 'rp-min') and
        the 'fast_metals' flag, in the format expected by ``BuildConfig.build``.
    """
    # Measured correlation and distortion matrix, and weight files of tracer1 and tracer2.
    # In the Vega convention the forest is tracer1 and the quasars tracer2 in the crosses.
    data_products = {
        "lyaxlya": (
            "lyalya_lyalya/cf_exp.fits.gz",
            "lyalya_lyalya/dmat.fits.gz",
            LYA_DELTA_ATTRIBUTES,
            LYA_DELTA_ATTRIBUTES,
        ),
        "lyaxlyb": (
            "lyalya_lyalyb/cf_exp.fits.gz",
            "lyalya_lyalyb/dmat.fits.gz",
            LYA_DELTA_ATTRIBUTES,
            LYB_DELTA_ATTRIBUTES,
        ),
        "lyaxqso": (
            "qso_lyalya/xcf_exp.fits.gz",
            "qso_lyalya/xdmat.fits.gz",
            LYA_DELTA_ATTRIBUTES,
            QSO_CATALOG,
        ),
        "lybxqso": (
            "qso_lyalyb/xcf_exp.fits.gz",
            "qso_lyalyb/xdmat.fits.gz",
            LYB_DELTA_ATTRIBUTES,
            QSO_CATALOG,
        ),
    }

    correlations = {}
    for name, (corr_file, dmat_file, weights1, weights2) in data_products.items():
        is_cross = "qso" in name

        correlations[name] = {
            "corr_path": str(CORRELATIONS_DIR / corr_file),
            "distortion-file": str(CORRELATIONS_DIR / dmat_file),
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
    # Cosmological setup: full-shape AP fit, with independent rescaling of the peak
    # (phi, alpha) and of the smooth component (phi_smooth, alpha_smooth)
    "scale_params": "phi_alpha",
    "template": "Planck18/DESI-2024_z_2.33.fits",
    "full_shape": False,
    "full_shape_alpha": False,
    "smooth_scaling": True,
    # Non-linear broadening of the BAO peak (sigmaNL_per, sigmaNL_par)
    "bao_broadening": True,
    # Arinyo-i-Prats et al. (2015) correction in the Lya autos, not applied to the peak
    "small_scale_nl": True,
    "skip-nl-model-in-peak": True,
    # QSO velocity dispersion and transverse proximity effect in the crosses
    "velocity_dispersion": "lorentz",
    "radiation_effects": True,
    # UV background fluctuations. BuildConfig also switches on the UV shot-noise term in the
    # autos, but its amplitude (uv_shotnoise_amp) is fixed to 0 below, so it does not
    # contribute to the model.
    "UVB-fluctuations": True,
    # HCD contamination
    "hcd_model": "Rogers2018",
    # Metals, with distortion matrices computed by Vega from the forest and QSO weights
    "metals": METALS,
    "new_metals": True,
    "rp_only_metal_mats": False,
    "use_metal_autos": True,
    # DESI instrumental systematics in the autos (desi_inst_sys_amp)
    "desi-instrumental-systematics": True,
    # Analytic marginalisation of all bins below r-min, keeping them in the model and
    # matching the data binning
    "marginalize-all-rmin-cuts": True,
    "fit-marginalized-scales": True,
    "marginalize-match-data-bins": True,
}

# Fiducial (starting) values of the model parameters. Parameters not listed here take the
# values of vega/templates/parameters.ini.
PARAMETERS = {
    # Scale parameters of the peak and smooth components, and BAO peak
    "phi": 1.0,
    "alpha": 1.0,
    "phi_smooth": 1.0,
    "alpha_smooth": 1.0,
    "sigmaNL_per": 3.24,
    "sigmaNL_par": 6.36984,
    "bao_amp": 1.0,
    # Lya and QSO biases and RSD
    "bias_LYA": -0.11629442782749021,
    "beta_LYA": 1.67,
    "alpha_LYA": 2.9,
    "bias_QSO": 3.792183285869361,
    "bias_eta_QSO": 1.0,
    "growth_rate": 0.9674624900622072,
    "alpha_QSO": 1.44,
    # Arinyo-i-Prats et al. (2015) small-scale correction
    "dnl_arinyo_q1": 0.303,
    "dnl_arinyo_q2": 0.267,
    "dnl_arinyo_kv": 0.576,
    "dnl_arinyo_av": 0.443,
    "dnl_arinyo_bv": 1.66,
    "dnl_arinyo_kp": 11.062,
    # HCDs
    "bias_hcd": -0.05,
    "beta_hcd": 0.7,
    "L0_hcd": 6.3,
    # QSO redshift errors, velocity dispersion and proximity effect
    "drp_QSO": 0.2,
    "sigma_velo_disp_lorentz_QSO": 3.8,
    "qso_rad_strength": 0.36,
    "qso_rad_asymmetry": 0.0,
    "qso_rad_lifetime": 9e99,
    "qso_rad_decrease": 300.0,
    # Metals: bias, RSD parameter and redshift evolution exponent of the bias
    "bias_FeII(1145)": -0.0052,
    "beta_FeII(1145)": 0.5,
    "alpha_FeII(1145)": 1.0,
    "bias_SiII(1190)": -0.0052,
    "beta_SiII(1190)": 0.5,
    "alpha_SiII(1190)": 1.0,
    "bias_SiII(1193)": -0.0024,
    "beta_SiII(1193)": 0.5,
    "alpha_SiII(1193)": 1.0,
    "bias_SiIII(1207)": -0.0074,
    "beta_SiIII(1207)": 0.5,
    "alpha_SiIII(1207)": 1.0,
    "bias_NV(1239)": 0.0,
    "beta_NV(1239)": 0.5,
    "alpha_NV(1239)": 1.0,
    "bias_SiII(1260)": -0.0046,
    "beta_SiII(1260)": 0.5,
    "alpha_SiII(1260)": 1.0,
    "bias_SiII(1304)": 0.0,
    "beta_SiII(1304)": 0.5,
    "alpha_SiII(1304)": 1.0,
    # CIV(eff) is an effective tracer of the CIV absorbers with no redshift evolution
    "bias_CIV(eff)": -0.01,
    "beta_CIV(eff)": 0.5,
    "alpha_CIV(eff)": 0.0,
    # DESI instrumental systematics
    "desi_inst_sys_amp": 1e-4,
    # UV background fluctuations
    "bias_gamma": 0.1125,
    "bias_prim": -0.66,
    "lambda_uv": 300.0,
    "uv_shotnoise_amp": 0.0,
}

# Sampled parameters. 'True' uses the default limits of vega/parameters/default_values.txt;
# otherwise the value is "min max start sigma".
SAMPLE_PARAMS = {
    "phi": "True",
    "alpha": "True",
    "phi_smooth": "True",
    "alpha_smooth": "True",
    "growth_rate": "True",
    "bias_LYA": "True",
    "beta_LYA": "True",
    "bias_QSO": "True",
    "sigma_velo_disp_lorentz_QSO": "True",
    "drp_QSO": "True",
    "qso_rad_strength": "True",
    "bias_hcd": "-1.0 0.0 -0.05 0.01",
    "beta_hcd": "0.0 3.0 0.7 0.1",
    "L0_hcd": "0.0 10.0 6.5 0.1",
    "bias_SiII(1190)": "True",
    "bias_SiII(1193)": "True",
    "bias_SiIII(1207)": "True",
    "bias_SiII(1260)": "True",
    "bias_CIV(eff)": "True",
    # Weak metals: biases restricted to negative values
    "bias_FeII(1145)": "-0.1 0.0 -0.00001 0.001",
    "bias_NV(1239)": "-0.1 0.0 -0.00001 0.001",
    "bias_SiII(1304)": "-0.1 0.0 -0.00001 0.001",
    "desi_inst_sys_amp": "True",
    "bias_gamma": "True",
}

# Gaussian priors: "gaussian mean sigma"
PRIORS = {
    # CIV(eff) bias measured from the CIV forest region
    "bias_CIV(eff)": "gaussian -0.019 0.005",
    # No prior on bias_QSO (unlike the BAO fit): with f sampled, a prior on bias_QSO would
    # inform the growth rate through beta_QSO = f / bias_QSO.
    "drp_QSO": "gaussian 0.0 1.0",
    # HCD parameters from mock-based studies of the HCD contamination
    "beta_hcd": "gaussian 0.50 0.09",
    "L0_hcd": "gaussian 5.0 1.0",
}

FIT_INFO = {
    "zeff": 2.33,
    "global_cov_file": str(GLOBAL_COVARIANCE),
    # Lya: (bias, beta) sampled directly; QSO: beta_QSO = bias_eta_QSO * f / bias_QSO
    "bias_beta_config": {"LYA": "bias_beta", "QSO": "bias_bias_eta"},
    # Sample the growth rate instead of fixing it to the value of the fiducial template
    "use_template_growth_rate": "False",
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
    """Write the main and correlation configs of the DESI DR2 full-shape fit to this directory.

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
