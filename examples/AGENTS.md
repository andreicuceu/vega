# Agent instructions for Vega configurations

## Running Vega

A Vega run is defined by one main config (`main.ini`) and one config per correlation
function (e.g. `lyaxlya.ini`, `qsoxlya.ini`), listed under `ini files` in the `[data sets]`
section of the main config. The main config sets the data-set level options (BAO/full-shape
parametrisation, fiducial template, parameters, sampled parameters and priors, sampler,
output). Each correlation config sets the data, scale cuts and model of that correlation.
See `main.ini`, `lyaxlya.ini` and `qsoxlya.ini` in this directory for the annotated list of
all options.

The scripts in `bin/` take the path to the main config:

- `run_vega.py main.ini`: run the minimizer (and the chi2 scan, if a `[chi2 scan]` section
  is present) and write the best-fit model and results to the `[output]` file.
- `run_vega_mpi.py main.ini`: run the sampler (PolyChord or PocoMC). Run it with MPI on
  compute nodes; typical runs need 10^2 - 10^4 core hours.
- `run_vega_mc_mpi.py` / `run_vega_mc_fits_mpi.py`: generate and fit Monte Carlo
  realisations (see `Monte_carlo_sims/README.md`).

Vega can also be run interactively through `vega.VegaInterface(path_to_main_ini)`.

## Choosing an example to start from

All the examples in this directory are starting points, not definitive configurations.
Whether their model, scale cuts, priors and sampled parameters are appropriate depends on
the data or mocks being fitted, and must be validated case by case.

To set up a new analysis, start from the closest example below: copy its `build_configs.py`,
change the data paths and options, and run it to generate the configs. Each of these
directories contains a `build_configs.py` (with a docstring describing the data and model
in detail) and the configs it generates, with paths relative to the directory. All fits
use the four Lya correlations (lyaxlya, lyaxlyb, lyaxqso, lybxqso); to fit a subset, change
`FIT_TYPE` and the correlation setup in the script.

| Task | Directory |
|---|---|
| BAO fit of DESI data | `DESI_data_BAO/` |
| Full-shape (AP / growth rate) fit of DESI data | `DESI_data_fullshape/` |
| BAO fit of raw or true-continuum mocks | `DESI_mocks/raw_and_true-cont/` |
| BAO fit of mocks without contaminants (continuum-fitted) | `DESI_mocks/uncontaminated/` |
| BAO fit of mocks with contaminants and/or QSO redshift errors | `DESI_mocks/contaminated/` |

- `DESI_data_BAO/`: DESI DR2 (loa) data with the current baseline model of the Lya
  correlations (DR3 BAO setup): metals with Vega metal matrices, HCDs, UV background
  fluctuations, QSO velocity dispersion and proximity effect, DESI instrumental
  systematics, analytic marginalisation of the bins below r-min. Only the BAO peak is
  rescaled (`ap_at`). Use it for any BAO fit of real data, or as the reference for the
  baseline model options.
- `DESI_data_fullshape/`: the same correlation model as `DESI_data_BAO/` with the full-shape
  cosmological setup (`phi_alpha` with independent peak and smooth rescaling, free growth
  rate, r-max = 200 Mpc/h). Comparing it with `DESI_data_BAO/` shows exactly which options
  change between a BAO and a full-shape fit.
- `DESI_mocks/raw_and_true-cont/`: mocks analysed with the true continuum (or raw deltas,
  computed directly from the transmitted flux): no distortion matrix, no contaminants, no
  QSO redshift errors. Kaiser model with BAO broadening and a Gaussian smoothing of the
  LyaCoLoRe field, with the mock template `Abacus-Colore-2LPT/Abacus-Colore-2LPT.fits`. For
  raw mocks, only the correlation and covariance paths change.
- `DESI_mocks/uncontaminated/`: mocks without astrophysical contaminants, analysed like the
  data (continuum fitting), so the distortion matrices and the analytic marginalisation of
  small scales are included. Two versions of the small-scale smoothing of the LyaCoLoRe
  field: `gauss_smoothing/` (Gaussian damping, free par/per_sigma_smooth) and
  `los_smoothing/` (LyaCoLoRe cell kernel with a free line-of-sight amplitude).
- `DESI_mocks/contaminated/`: mocks with DLAs, metals, BALs and Gaussian QSO redshift errors.
  Adds to `uncontaminated/gauss_smoothing/` the HCD model, the metals (with Vega metal
  matrices) and a Gaussian QSO velocity dispersion with a free drp_QSO.

To tell which mock example applies: the mock analysis flavour (raw, true continuum, or
continuum-fitted "baseline") sets the first choice; for continuum-fitted mocks, the
contaminants in the quickquasars run (e.g. `loa-0` has none, `loa-124` has DLAs, metals
and BALs) and the QSO catalogue used in the cross-correlations (`zcat.fits` has no
redshift errors; `zcat_gauss_400.fits` has Gaussian errors of 400 km/s) set the second.

Other material in this directory:

- `main.ini`, `lyaxlya.ini`, `qsoxlya.ini`: annotated reference of every option Vega reads.
  Use them to look up an option, not as configs to run. `qsoxqso.ini` is an experimental
  QSO auto-correlation example.
- `Monte_carlo_sims/`: Monte Carlo realisations and forecasts (see its README).
- `P3D_support/`: correlation function models for a custom input P(k).
- `eBOSS_DR16/`: reproduction of the eBOSS DR16 BAO analysis.
- `deprecated/`: DESI DR1 (Y1) configs, kept for reference only. Do not start new analyses
  from them.

## Creating and modifying configs

- Never create or edit Vega `.ini` files by hand, including small changes to existing
  files.
- Generate them procedurally with a script (or notebook). To change a configuration, change
  the script and re-run it, so that every config can be reproduced from its script.
- Always generate the configs with Vega's config builder, `vega.BuildConfig`
  (`vega/build_config.py`), through `BuildConfig(options).build(...)`.
- If a needed option is not yet supported by `BuildConfig`, check the Vega issues
  (https://github.com/andreicuceu/vega/issues) and open an issue requesting it if none
  exists. Until it is implemented, add code to the same script that modifies the generated
  `.ini` files after `BuildConfig` has written them (e.g. with `configparser`). Use this only
  for missing features. Document each such post-processing step in the script, with a
  comment explaining what is changed and why, and a link to the corresponding issue, so it
  can be removed once `BuildConfig` supports the feature.
