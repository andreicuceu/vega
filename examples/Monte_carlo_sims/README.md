# Tutorial for running Monte Carlo Simulations

## Scripts

Monte carlo (MC) simulations can be run with a few different scripts:

- `run_vega.py`: Used for single iteration runs (e.g. when using forecast mode).
- `run_vega_mc_mpi.py`: Used to create and fit sets of MC simulations (parallelized with MPI).
- `run_vega_mc_fits_mpi.py`: Used to fit sets of MC simulations created with the previous script (parallelized with MPI).

Note that if you plan to use any form of analytic marginalization, only a certain combination of these scripts will work. This is described in more detail in the final section below.

## General setup

This section applies to all MC runs, independent of which script you are using.

Your main.ini files should have two extra sections, which *must* exist even if left empty:

```
[mc parameters]
# This is treated as a second parameters section.
# If empty, the usual parameters section will be used
# Any parameter added here will overwrite the fiducial model generation (independent of the source of the model)
# This section does not apply when using the "use_measured_fiducial" option

# E.g. if you want your model to based on a fit, but with fixed BAO to 1, add:
ap = 1
at = 1

[monte carlo]
# This section functions identically to the [sample] section
# The parameters set here will be the ones that are sampled when fitting the MC mock/forecast
```

The usual `[sample]` and `[parameters]` sections must still exist even if left empty. If there are parameters in both the `[sample]` and `[monte carlo]` sections, an initial fit will be done on the actual data passed, and the resulting best-fit parameters will be used to produce the fiducial model for MC simulations. We highly recommend not doing this when using the MPI scripts (see an alternative below). 

The MC module offers a variety of different options, which can be set in the `[control]` section of the main.ini file:

```
[control]

# Main flag for turning on MC mode
run_montecarlo = True

# Seed to be used for noise generation in MC simulations
mc_seed = 0

# Toggle forecast mode
# No noise will be injected to the mock correlations
forecast = False

# Genearate the fiducial model using the set of parameters from an existing fit
# The path must point to a previous vega fit results file
# This variable is optional (don't pass it unless needed)
mc_start_from_fit = /path/to/vega/fit_results.fits

# Use a measured correlation function instead of a fiducial model
# to generate forecasts and MC simulations
# When this is used, it's generally the stack of many mocks
# This variable is optional (don't pass it unless needed)
use_measured_fiducial = False
# Use "mc_fiducial_NAME" to pass the paths to the measured correlations
# when using the flag above
# NAME must be the name given to each correlation in it's respective ini file
# These variables are optional (don't pass them unless needed)
mc_fiducial_lyaxlya = /path/to/picca/cf.fits
mc_fiducial_lyaxqso = /path/to/picca/xcf.fits

# Flag for direct inference analyses
# Only use with the Cobaya interface for Vega
# This variable is optional (don't pass it unless needed)
use_full_pk_for_mc = False

# Used to rescale global covariance when it is read
# This factor multiplies the global covariance,
# and affects both MC production and fits
# Only applies to the global covariance
# This variable is optional (don't pass it unless needed)
cov_scale = None

# Similar to the previous variable, but only affects the global covariance
# when generating noise for MC simulations
# Has no effect on the covariance used when fitting
# This variable is optional (don't pass it unless needed)
global_cov_rescale = None
```

Finally, MC mocks have an option for specifying their own output path:
```
[output]
# This is optional, but we highly recommend to use it
mc_output = /path/to/mc_output
```
Unlike the usual vega output path, this path is specifying a directory. If running MC mocks in parallel with MPI, each thread will write its own output file in that directory, with files named `monte_carlo_{i}.fits`. Otherwise, there will one file named `monte_carlo.fits`.

## Using run_vega_mc_mpi.py

The script run_vega_mc_mpi.py can be used either for a full MC analysis, or as the generation step if doing a two step analysis. The relevant options for this script are:
```
[control]
# Set to True to fit the MC mocks
# Set to False to generate MC mocks and write them to a file, without fitting
# Note that this option is turned on by default! 
run_mc_fits = True

# Number of MC mocks to generate
num_mc_mocks = 1024
```
These options are in addition to those described in the General section above.

If using it for a full MC analysis (i.e. with `run_mc_fits=True`), we recommend using MPI to parallelize the run. At NERSC, the relevant command is:
```
srun --tasks-per-node=64 run_vega_mc_mpi.py /path/to/vega/main.ini
```
We recommend 32 to 64 MPI threads per node. Note that in this mode, MC mocks will be generated and fitted separately for each thread, with each thread having a different seed. The algorithm is simply `cpu_seed = seed + cpu_id`. Therefore, if running multiple consecutive runs to generate more MC mocks, make sure to increase the seed by at least `num_cpu` (i.e. `num_nodes x num_threads_per_node`) after each run.

If using this script to generate mocks without fitting them (i.e. with `run_mc_fits=False`), simply run:
```
run_vega_mc_mpi.py /path/to/vega/main.ini
```

## Using run_vega_mc_fits_mpi.py

The purpose of this script is to fit MC mocks that were generated with the previous script. This is useful when the setup for generating and fitting mocks is different. An example of this is when MC mocks are to be fitted with analytic marginalization, a case that is described in the next section.

The relevant options for this script are:
```
[control]
# Path to Vega MC mocks output file
# This is required, and should be the output of running the previous script
mc_mocks = /path/to/output_fitter/monte_carlo/monte_carlo.fits

# These options are used to slice the MC mocks when fitting only a subset of the full mock
# It is meant to be used when computing cross-correlations between different data sets
# This use case is not well documented yet. If interested in trying it, ask either Andrei or Hiram
# These flags are optional (don't pass them unless needed)
slice_start1 = None
slice_end1 = None
slice_start2 = None
slice_end2 = None
```
These options are in addition to those described in the General section above.

To run this script, see the MPI instructions for the `run_vega_mc_mpi.py` script above.

## MC mocks with analytic marginalization

We describe the case where MC mocks are to be fitted with analytic marginalization separately, as this is a special case. The reason for this is that analytic marginalization is normally done directly in the covariance matrices; for each correlation separately and also within the global covariance matrix. As unalatered covariance matrices are needed to generate the noise for MC mocks, special care should be taken when running this. To run a setup with analytic marginalization, take the following steps:

1. Use the `run_vega_mc_mpi.py` script to generate MC mocks, but do not fit them (see [Using run_vega_mc_mpi.py](#using-run_vega_mc_mpipy)). This should be done with all marginalization options turned off, so not to alter the covariance matrices.
2. Use the `run_vega_mc_fits_mpi.py` script to fit the mocks generated in the privous step (see [Using run_vega_mc_fits_mpi.py](#using-run_vega_mc_fits_mpipy)). In this case you can use the exact setup that you want for the fits, including analytic marginalization.

This approach is also useful anytime you want to generate MC mocks with one setup and fit them with a different setup, but it requires two different sets of Vega configuration files. An example of this setup for the DR2 full-shape analysis is shown in the files here. The files named
