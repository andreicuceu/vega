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
