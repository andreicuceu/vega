=======
History
=======

1.4.2 (2025-10-01)
------------------
* Unblinded DR2 CIV BAO analysis
* Added Arinyo q2 parameter
* Added explicit call limit for iminuit
* Minor bug fixes

1.4.1 (2025-08-25)
------------------
* Add support for shell plots in the plotting module
* Automatic shell plots after fits when using run_vega.py
* Add option to fit independent full-shape smoothing parameters for each tracer
* Add MC functionality to the ConfigBuilder
* Some minor bug fixes related to MC mocks

1.4.0 (2025-07-01)
------------------
* Overhaul of the Monte Carlo mocks functionality, now supports global covariances
* Minor improvements to blinding functionality, and added blinding for DR2 full-shape
* Update flat priors to have more realistic ranges
* Added options to compute shell compression as a function of mu, mu^2, and theta
* Fixed a bug in the plotting of the Fisher information as a function of rp and rt

1.3.1 (2024-12-11)
------------------
* Unblind Y3 BAO analysis
* Some small improvements to FitResults and full-shape blinding

1.3.0 (2024-11-14)
------------------
* Some updates to MC mocks. Now it is possible to read a previous fit and use it to generate mocks.
* Updated metal formatrix computation allowing for rt dependecy.
Previous configuration can still be used, as the new one is quite slow to initialize.

1.2.1 (2024-10-14)
------------------
* Fixed metal caching bug. Only metal autos are cached now.
* Updated blinding for Y1 and Y3 full-shape analyses

1.2.0 (2024-06-04)
------------------

* Added blinding for DESI Y3
* BAO blinding still done through a blinding template
* AP blinding done at the parameter level in Vega
* Added two AP blinding files for DESI Y1 and DESI Y3

1.1.0 (2024-04-27)
------------------

* Added AP blinding for DESI DR1 (can only be run at NERSC)
* Updated some variable names to reflect the DESI BAO paper
* Added option to run Monte Carlo mocks from the global covariance

1.0.0 (2024-02-09)
------------------

* This is the version used to run the DESI Y1 BAO analysis
* New template: Planck18/DESI-2024_z_2.33.fits
* The growth_rate parameter is now automatically read from the template file by default. 
This means the value passed will be ignored from now on. There is a way to turn this off by setting
use_template_growth_rate=False option in [control].
* Added option to use bias_metal (instead of bias_eta_metal) in ConfigBuilder
* Defaults values updated: beta_CIV=0.5, alpha_CIV=0
* We now recommand sampling bias_metal instead of bias_eta_metal
* Smoothing parameters are now fixed to input value for metal correlations. This fixes the bug
that made fits fail when varying smoothing parameters with fast_metals turned on. (only matters for mocks)
* Growth rate is now fixed for metal correlations by default (only matters for full-shape analyses).
* Fixed a bug where the DESI instrumental systematics model was being added twice.
This only impacts the value of the fitted amplitude for that model.
* Monte Carlo mocks now support the use of the full covariance matrix.
* Added option to turn off extrapolation when doing the FHT to go from xi to pk in the make_template script.
We recommend turning this off when computing templates for non-Planck cosmologies.
* Added example config files for the DESI Y1 analysis under examples/DESI_data_setup for runs on data,
and under examples/DESI_mock_setup for runs on mocks.

0.6.3 (2023-12-10)
------------------

* DESI Y1 is now unblinded. Vega supports both DA_BLIND and DA column names for desi_y1 blinding
* Distortion matrix blinding status is now ignored

0.6.2 (2023-12-06)
------------------

* Added new template at the effective redshift of the DESI Y1 correlations
* Update method used to compute the effective redshift
* Added new metal configuration to the Config Builder
* Added new low memory mode for sampler runs


0.6.1 (2023-12-04)
------------------

* Added option to use the full covariance matrix, including all cross-covariances


0.6.0 (2023-12-02)
------------------

* New metal modelling:
    - Metal matrices are now computed on the fly in vega
    - To compute metal matrices vega now requires weights 
    (delta-attributes file for forests and catalogs for discrete objects)
    - Backwards compatibility with old metal matrices is maintained

* New Monte-Carlo mock functionality, including a new MPI parallelized script
* Overhaul of coordinate handling in vega

0.5.2 (2023-10-01)
------------------

* New more accurate model for desi instrumental systematics
* Added option to skip all metal auto-correlations (turned off by default)

0.5.1 (2023-09-19)
------------------

* Small fixes for non-standard model binning
* Metal smoothing for mocks
* Fixed interpolation bounds for fvoigt model

0.5.0 (2023-08-24)
------------------

* Update HCD modelling config and defaults
    - To use the Fvoigt model, use keyword "fvoigt" instead of the old "mask"
    - "L0_hcd" parameter only applies to Rogers model from now on
    - Fvoigt model now has its own "L0_fvoigt" parameter, by default set to 1

* Add parameter sensitivity and information calculations and plots.

0.4.3 (2023-07-26)
------------------

* Minor fix for vega installations from tarballs instead of git

0.4.2 (2023-06-19)
------------------

* Minor updates to maintain future compatibility with numba

0.4.1 (2023-06-19)
------------------

* Fixed some minor issues from v0.4.0

0.4.0 (2023-06-19)
------------------

* First version used for the DESI Y1 analysis.
* New plotting module
* New metal computation and smart chaching

0.2.0 (2022-01-07)
------------------

* First version used in DESI, and for most of the early analyses. Includes blinding.

0.1.0 (2020-03-03)
------------------

* First version of Vega
