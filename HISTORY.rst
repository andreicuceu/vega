=======
History
=======

1.8.1 (2026-11-25)
------------------
* Added example config builders for the standard DESI analyses. Each example folder
  contains a ``build_configs.py`` script that generates the configs with ``BuildConfig``,
  and the generated configs, with paths relative to the folder:

  * ``examples/DESI_data_BAO`` and ``examples/DESI_data_fullshape``: BAO and full-shape
    (Alcock-Paczynski) fits of the four DESI DR2 Ly-alpha correlations, with the DESI DR3
    BAO model of the correlations.
  * ``examples/DESI_mocks``: BAO fits of stacks of LyaCoLoRe 2LPT DR2-like mocks, for the
    true-continuum analysis (``raw_and_true-cont``), the uncontaminated mocks with two
    treatments of the small-scale smoothing (``uncontaminated/gauss_smoothing`` and
    ``uncontaminated/los_smoothing``), and the mocks with DLAs, metals, BALs and QSO
    redshift errors (``contaminated``).

* Added the ``Abacus-Colore-2LPT/Abacus-Colore-2LPT.fits`` linear power spectrum template
  of the LyaCoLoRe 2LPT mocks.
* Rewrote the annotated ``examples/main.ini``, ``examples/lyaxlya.ini`` and
  ``examples/qsoxlya.ini`` to document every option Vega reads. This corrects ``ell-max``
  to ``ell_max``, removes inline comments (which ConfigParser reads as part of the value)
  and options that are not read, and marks ``examples/qsoxqso.ini`` as experimental.
* Added ``examples/AGENTS.md`` with instructions to generate configs with ``BuildConfig``
  instead of writing them by hand, and updated ``examples/README.rst``.
* Moved the DESI DR1 (Y1) example configs to ``examples/deprecated``, and removed the
  LyaCoLoRe and picca fitter2 benchmark examples. The Vega configs of the picca benchmark
  test moved to ``tests/picca_benchmarks``, so the installed-wheel tests no longer stage
  files from ``examples``.
* Added ``vega/analytic_marginalization.py``, which holds the analytic marginalization
  of linear model parameters. The small-scale template marginalization moved there from
  ``Data`` and ``VegaInterface``; the chi2, likelihood and fit results of existing
  configurations are bit-identical.
* Additive broadband polynomials can be marginalized analytically with
  ``marginalize-broadbands = True`` in the ``[model]`` section of a correlation config,
  with a flat prior or a Gaussian prior of width ``broadband-prior-sigma``.
* The best-fit small-scale coefficients, and the best-fit model in the output file, now
  come from the joint solution with the Gaussian prior and, when a global covariance is
  used, the covariance between correlations. Previously each correlation was solved
  separately, with a flat prior when ``fit-marginalized-scales`` and
  ``marginalize-match-data-bins`` were both set.
* ``marginalize-in-fit = True`` no longer changes the likelihood: it only makes the
  marginalized coefficients available at every evaluation, e.g. as derived parameters of
  a chain. Chains run without it no longer contain these derived parameters.
* The counts entering the degrees of freedom are stored in ``VegaInterface.dof_counts``
  and written to the ``BESTFIT`` header (``NDATA``, ``NFITPAR``, ``NSMALL``, ``NMARGOTH``,
  ``NMARG``).
* ``FitResults`` reads ``NSMALL`` and ``NMARGOTH`` and computes the degrees of freedom,
  reduced chi2 and PTE as ``VegaInterface.minimize`` does. For output files written from
  now on, the small-scale modes are therefore subtracted from the number of data points.
* The Percival correction counts analytically marginalized broadband coefficients as
  fitted parameters.
* Linear parameters listed in the new ``[analytic marginalization]`` section of the main
  config are marginalized analytically: ``desi_inst_sys_amp``, ``qso_rad_strength``, the
  metal biases and the metal x metal amplitudes. One parameter is one coefficient whose
  template may enter several correlations; templates that depend on the sampled parameters
  are rebuilt at every evaluation. Their priors are read from ``[priors]`` (flat otherwise).
* New ``metal-auto-amplitudes`` option in ``[model]``: every metal x metal correlation is
  scaled by its own amplitude ``amp_<m1>_<m2>`` instead of the product of the metal biases,
  which makes the metal biases linear parameters. With amplitudes equal to the bias
  products the model is unchanged.
* Removed the ``separate-metal-auto-biases`` option (``bias_<m1>_<m2>`` factors on the
  metal x metal bias products); configurations that set it are refused with a pointer to
  ``metal-auto-amplitudes``.
* Changed convention for analytically marginalized broadband coefficients: the model is
  evaluated with them set to zero, and they are marginalized with a zero-mean prior, as
  all marginalized coefficients now are. Their ``[parameters]`` values no longer enter the
  likelihood and may be omitted. Results are unchanged for flat priors and for zero
  ``[parameters]`` values; with ``broadband-prior-sigma`` and non-zero values, the prior
  is now centred on zero instead of on those values; the previous results correspond to a
  Gaussian prior centred on the ``[parameters]`` values. The reported coefficients are the
  values themselves, not deviations from ``[parameters]``.
* ``BuildConfig`` writes configs for the analytic marginalization options: with
  ``metal-auto-amplitudes`` it writes ``amp_<m1>_<m2>`` to ``[parameters]`` for every metal x
  metal correlation Vega builds (given values, or the bias products by default), so that
  they can be sampled; ``fit_info["analytic_marginalization"]`` writes and validates the
  ``[analytic marginalization]`` section, whose parameters may have entries in
  ``fit_info["priors"]``; the correlation entries ``marginalize-broadbands`` and
  ``broadband-prior-sigma`` are written to the ``[model]`` section of that correlation.
  Configs built without these options are unchanged.
* ``BuildConfig`` writes ``fit_info["marginalize_in_fit"]`` as ``marginalize-in-fit`` to
  ``[control]``, and the PolyChord settings ``precision``, ``seed``, ``resume``,
  ``write_dead``, ``cluster_posteriors`` and ``maximise`` given in ``fit_info["Polychord"]``
  to ``[Polychord]``. Values that Vega cannot read as the expected type are refused. They
  are written only when given, so configs built without them are unchanged.
* ``BuildConfig`` no longer lists ``CIV(eff)`` in the ``[metals]`` section of
  cross-correlations with a discrete tracer (e.g. ``lyaxqso``). Vega only uses CIV(eff)
  through CIV(eff) x CIV(eff) and ignored it there, so the models are unchanged. It is kept
  if it is the only metal.
* Fixed Monte Carlo mock creation without a global covariance when scale cuts are applied:
  ``Data.create_monte_carlo`` now accepts the fiducial model restricted to the fitted bins
  that ``VegaInterface.initialize_monte_carlo`` passes to it, instead of raising
  "Could not match fiducial model to data or model size".
* The relativistic (Doppler) term of continuous x discrete cross-correlations now uses the
  Doppler kernel ``nu_ell(r) = 1/(2 pi^2) int dk k P_lin(k) j_ell(kr)``, computed with mcfit:
  ``xi_rel = Arel1 nu_1 L1(mu) + Arel3 nu_3 L3(mu)``, with ``Arel1 = -H f X1`` and
  ``Arel3 = H f X3``. The previous FFTLog (``tform="rel"``) did not lower the power of k and
  returned ``int dk k^2 P j_ell``, so values of ``Arel1``/``Arel3`` from earlier fits are not
  comparable; typical Lya x QSO values are now ``|Arel1| ~ 1e-3``. The asymmetry term keeps
  its kernels and normalisation, without the 2-3% scale-dependent error of the old FFTLog.
  Both terms now rescale coordinates as the core correlation does (correlation-specific
  ``ap``/``at`` and the data-template correction). ``Arel1``, ``Arel3`` and ``Aasy0/2/3``
  default to 0 in the template parameters and have entries in ``default_values.txt``. The
  picca benchmark datasets that use the relativistic correction (``test_2``, ``test_7``)
  are no longer compared with picca.
* Fixed ``marginalize-all-rmin-cuts`` with ``rt-min`` (or ``rp-min``) cuts: the marginalized
  bins are now those removed by the small-scale cuts inside the large-scale cuts (``r-max``,
  ``rt-max``, ``rp-max`` and the mu range). Before, the data and distorted-model grids of a
  cross-correlation, which cover different r_par ranges, marginalized different numbers of
  bins and initialization failed. Configs with ``r-min`` alone are unchanged.
* Fixed ``velocity dispersion = lorentz_gauss``, which applied the Gaussian kernel alone; it
  now applies both the Lorentzian and the Gaussian kernels.

1.8.0 (2026-10-09)
------------------
* Changed the new-metal distortion matrices (``new_metals = True``) to the expectation of
  the pair estimator, selected with the new ``metal_matrix_convention`` option in
  ``[model]`` (``estimator`` by default, or ``legacy``). Each observed bin is normalized by
  the estimator weight of all its pairs, the amplitude evolution of each absorber is
  applied once, pair by pair at the true redshift, and the undistorted metal correlation
  is evaluated at ``zeff`` with the effective coordinates of the true bin. The previous
  matrices normalized each true bin to unit sum, counted the redshift evolution twice,
  and used the effective coordinates of the observed bin, biasing e.g. the CIV(eff)xCIV(eff)
  amplitude low by about 11%.
* The new default changes the model of every existing ``new_metals`` configuration. Set
  ``metal_matrix_convention = legacy`` to reproduce previous results bit for bit. In
  DESI-like Ly-alpha auto and Ly-alpha-QSO cross fits the change is 0.2--0.5 in chi2 at
  fixed parameters, absorbed by 2--6% shifts in the metal biases, with BAO shifts of at
  most 0.08 sigma.
* Moved the amplitude evolution exponents of the metal matrices to ``alpha_<X>`` in
  ``[parameters]``, now required for every absorber of the metal correlations (including
  ``alpha_LYA`` and ``alpha_QSO``). The estimator weighting exponents in ``[metal-matrix]``
  are renamed ``weight_evol_LYA`` (previously ``alpha_LYA``) and ``weight_evol_QSO``
  (previously ``z_evol_objects``, also used by catalogue bias evolution and direct
  multipoles). The old keys are accepted with a deprecation warning, and conflicting old
  and new weighting keys raise an error.
* Under the estimator convention, sampling or scanning an exponent built into the
  matrices raises an error, and Monte Carlo input models with different exponents are
  computed with matrices built from those exponents. ``new-bias-evolution`` is disabled
  for these metal correlations, since the matrices already include the evolution of both
  legs.
* ``fast_metals = True`` now raises an error for configurations that would reuse a stale
  cached metal x metal correlation: a varied metal beta, a varied metal ``alpha`` under
  ``legacy`` or with ``new_metals = False``, ``no-metal-decomp = False``, or
  ``metal-scaling = True``. The picca metal-matrix path (``new_metals = False``) is
  unchanged but now warns that it double counts the redshift evolution.
* The data-template cosmology correction introduced in 1.7.9 is no longer applied
  automatically; enable it with ``apply-data-template-correction = True`` in ``[model]``.
* ``BuildConfig`` accepts ``metal_matrix_convention`` and writes the new
  ``[metal-matrix]`` layout, as do the example configurations. The examples now use
  ``alpha_CIV(eff) = 0`` from ``[parameters]`` instead of the inconsistent value of 1
  previously set in ``[metal-matrix]``.
* The 2D metal matrices are assembled as a sparse Kronecker product, yielding identical
  matrices without the dense temporary (about 7 GB per species for Ly-alpha-QSO) and a
  5--10 times faster initialization.
* Added ``utils.normalized_growth_factor``, shared by ``CorrelationFunction`` and the metal
  matrices, and removed ``compute_growth_old`` and the ``old_growth_func`` option (setting
  it now raises an error).
* Removed the unused ``VegaInterface.set_fast_metals``; set ``fast_metals`` in ``[model]``
  instead. The interface checks and ``compute_sensitivity`` moved to
  ``vega/interface_utils.py``, with ``VegaInterface.compute_sensitivity`` kept as a wrapper.
* Added ``tests/test_metals.py``, covering the matrix normalization against an independent
  stacked-pair expectation, a ``legacy`` regression, equivalence of the old and new key
  layouts, the exponent and fast-metals errors, and the Monte Carlo input models.

1.7.10 (2026-10-04)
-------------------
* Raised the minimum supported Python version to 3.11 (3.9 and 3.10 are no longer supported
  or tested) and added Python 3.14. The CI matrix covers Python 3.11--3.14, and Python 3.14
  is the default version for linting, documentation, distribution validation, and releases.
* Declared every directly imported dependency: ``GitPython`` is now a core dependency
  (required by ``BuildConfig``). Added the ``[pocomc]`` extra (``pocomc``, ``mpi4py``,
  ``schwimmbad``), with ``[samplers]`` kept as an identical alias, and the ``[templates]``
  extra (``camb``, ``fitsio``) for ``make_template.py``. The README documents each extra.
* The Sphinx documentation now builds without warnings: the tutorial notebooks are
  rendered, broken API references and malformed docstrings were fixed, and optional
  sampler dependencies are mocked. The PolyChord installation instructions, including
  the NERSC ``mpi4py`` build, are included in the documentation.
* Added CI jobs that build the documentation with warnings treated as errors, install the
  exact validated wheel with each optional extra in a clean environment and probe its
  imports and scripts, and run the test suite against the installed wheel from outside
  the source tree.
* The ordinary CI test matrix no longer installs MPI or the ``mpi`` extra. The CI workflow
  has an explicit read-only token, every action (including Codecov) is pinned to a commit
  SHA, and the redundant artifact-transfer job was removed.
* Aligned the ``Makefile`` targets with CI (Ruff, strict Sphinx build, pytest coverage),
  removed the unused ``tox`` and ``servedocs`` targets, and stopped ``make docs`` from
  deleting ``docs/modules.rst``.
* Documented the reason for each dependency lower bound and added a dependency policy to
  ``CONTRIBUTING.rst``. Test and documentation CI jobs upload their resolved package
  versions as ``freeze-*`` artifacts.

1.7.9 (2026-10-04)
------------------
* Added automatic correction for differences between the cosmology used to compute the
  correlation function from the data (the ``picca`` cosmology) and the fiducial template
  cosmology. The ratios of the template and catalogue distances, ``DM/D_M(z_fid)`` and
  ``DH/D_H(z_fid)``, are applied as additional scalings of the Alcock-Paczynski
  parameters ``at`` and ``ap`` when rescaling the model coordinates. A warning is printed
  when the two cosmologies differ by more than 1% or when ``z_eff`` and ``z_fid`` differ
  by more than 0.025.
* Enabled Ruff linting and formatting across the package and tests (notebooks excluded),
  aligned the Ruff version between pre-commit and CI, and documented the pre-commit
  development workflow.
* Updated the GitHub Actions workflows for Node.js 24 and bumped the pinned actions
  (``codecov-action``) via Dependabot.
* Switched to SPDX license metadata (``GPL-3.0-or-later``, PEP 639), requiring
  ``setuptools>=77.0.3``, and added canonical installation instructions to the README and
  documentation.

1.7.8 (2026-09-14)
------------------
* Added tested source distributions, pure-Python wheels, and SHA256 checksums to
  GitHub Releases.
* Added Git-archive version metadata for ``setuptools-scm`` so tagged archives can
  recover their exact version without a ``.git`` directory.
* Added CI validation of checkout and Git-archive builds, isolated source-distribution
  rebuilds, clean wheel installations, scripts, and packaged scientific resources.
* Updated release automation and installation documentation to use the validated
  GitHub Release artifacts as the canonical distributions.

1.7.7 (2026-09-14)
------------------
* Added support for correlation functions binned in (r, mu) and Legendre multipole
  compression, including covariance and distortion matrices and metal correlations.
* Added direct input of QSO auto-correlation multipoles and their covariance for
  joint Ly-alpha auto, Ly-alpha-QSO cross, and QSO auto analyses, with consistent
  separation cuts and multipole selection in the global covariance.
* Added catalog-based redshift weights and bias evolution for QSO multipoles and
  cross-correlations.
* Added support for externally computed two-dimensional P(k, mu) models, including
  metal contributions, with example notebooks for custom models and configurations.
* Added the aiso/aap Alcock-Paczynski parametrization used in DESI galaxy analyses.
* Added optional Hartlap corrections for individual and global covariances, with
  the Percival correction applied to saved parameter errors and covariance for
  global fits using the Hartlap correction.
* Extended FITS output and fit postprocessing for multipoles, fit p-values, and
  covariance corrections; added multipole plotting and improved four-wedge plots.
* Fixed fiducial-model masking and Monte Carlo mock generation with global
  covariances, and improved coordinate-binning checks.
* Added a model-binning option to the configuration builder and updated small-scale
  marginalization parameter handling.
* Added tests and example configurations for QSO multipoles and catalog redshift
  weights, updated the README, and required SciPy >= 1.12.0.

1.7.6 (2026-07-31)
------------------
* Migrated package setup from setup.py to modern pyproject.toml standard.
* Replaced flake8/isort with Ruff and implemented pre-commit hooks (need to be activated).
* Updated CI/CD pipeline to use modern GitHub Actions with dependency caching.
* Refactored documentation to use sphinx-book-theme and MyST-NB for Jupyter notebook rendering.
* Replaced hardcoded version strings with dynamic setuptools_scm versioning.

1.7.5 (2026-07-29)
------------------
* Version used for DESI DR2 full-shape KP
* Minor updates to metal parameters

1.7.4 (2026-04-08)
------------------
* Catch PackageNotFoundError when initializing Vega

1.7.3 (2026-04-07)
------------------
* Some protections for log_lik and marg_coeff behaviour

1.7.2 (2026-04-06)
------------------
* Added printing of vega version when initializing VegaInterface
* Some protections for log_lik and marg_coeff behaviour
* New modelling options for binning
* New modelling options for HCDs

1.7.1 (2026-03-24)
------------------
* Minor bug fix in ConfigBuilder

1.7.0 (2026-03-23)
------------------
* Added UVB shotnoise and HeII reionization models
* Updated some flag names for consistency
* Added support for the redshift bin analysis

1.6.3 (2026-03-11)
------------------
* Added new options for small-scale marginalization:

  - Option to fit the marginalized bins.
  - Option to marginalize over data bins instead of model bins.
  - Option to fit marginalized parameters at every step of the likelihood
    instead of marginalizing in the covariance matrix.

1.6.2 (2026-03-04)
------------------
* Updates and improvements to the small-scale marginaliztion functionality
* Improvements to the sampler performance
* Add SSM parameters as derived parameters in PolyChord

1.6.1 (2026-02-23)
------------------
* Unblind DR2 alpha_smooth parameter
* Add support for alpha_full analyses while phi is still fitted separately
* Some minor fixes to the low memory mode for sampler runs

1.6.0 (2026-01-06)
------------------
* Add analytic marginalization over small-scale marginalization parameters

1.5.1 (2025-11-24)
------------------
* Fix bug with QSO-metal computation of metal matrices

1.5.0 (2025-11-21)
------------------
* Optimize metal computation by caching more correlations
* Added small scale marginalization for distorted analyses
* Add option for k^n smoothing to match EFT behavior
* Minor updates to the ConfigBuilder
* Minor bug fixes

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
* The growth_rate parameter is now automatically read from the template file by default. This means the value passed will be ignored from now on. There is a way to turn this off by setting use_template_growth_rate=False option in [control].
* Added option to use bias_metal (instead of bias_eta_metal) in ConfigBuilder
* Defaults values updated: beta_CIV=0.5, alpha_CIV=0
* We now recommand sampling bias_metal instead of bias_eta_metal
* Smoothing parameters are now fixed to input value for metal correlations. This fixes the bug that made fits fail when varying smoothing parameters with fast_metals turned on. (only matters for mocks)
* Growth rate is now fixed for metal correlations by default (only matters for full-shape analyses).
* Fixed a bug where the DESI instrumental systematics model was being added twice. This only impacts the value of the fitted amplitude for that model.
* Monte Carlo mocks now support the use of the full covariance matrix.
* Added option to turn off extrapolation when doing the FHT to go from xi to pk in the make_template script. We recommend turning this off when computing templates for non-Planck cosmologies.
* Added example config files for the DESI Y1 analysis under examples/DESI_data_setup for runs on data, and under examples/DESI_mock_setup for runs on mocks.

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

  - Metal matrices are now computed on the fly in Vega.
  - To compute metal matrices, Vega now requires weights (delta-attributes
    files for forests and catalogs for discrete objects).
  - Backwards compatibility with old metal matrices is maintained.

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
