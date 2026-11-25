==============
Vega Tutorials
==============

Here you can find some examples on how to run Vega. If you can't find what you are looking for, also have a look at the `documentation`_, or open an issue.

.. _documentation: https://vega.readthedocs.io/en/latest/?badge=latest

Config files
------------

Vega needs one "main.ini" file with the data-set level configuration, and at least one correlation config file. The main config sets the cosmological (BAO or full-shape) parametrisation, the fiducial template, the model parameters, the sampled parameters and priors, the sampler and the output, and lists the correlation configs under ``ini files`` in the ``[data sets]`` section. Each correlation config sets the measured correlation, the scale cuts and the model of that correlation. They are named following the Vega convention, with the two tracers separated by an "x": "lyaxlya.ini" for the Lyman alpha forest auto-correlation, "lyaxqso.ini" for its cross-correlation with quasars, and "lyaxlyb.ini" and "lybxqso.ini" for the correlations involving Lyman alpha absorption in the Lyman beta region of the forest.

The files "main.ini", "lyaxlya.ini" and "qsoxlya.ini" in this directory document every option Vega reads, with comments explaining each option, its default and its alternatives. "qsoxqso.ini" is an example for the QSO auto-correlation multipoles, which is an experimental feature.

Config files for real analyses should not be written by hand. Generate them with ``vega.BuildConfig`` from a script, and modify the script instead of the generated files. All the DESI examples below follow this approach (see also "AGENTS.md").

Using the terminal
------------------
You can call Vega from a terminal using the scripts in the bin folder, and pointing them to a "main.ini" file like this:

.. code-block:: console

    $ python run_vega.py path_to/main.ini

The "run_vega.py" script can be used for computing model correlations and for running the fitter. However, these can also be run interactively (see next section).

On the other hand the samplers (PolyChord or PocoMC) cannot be run interactively and need to be called using the second script like this:

.. code-block:: console

    $ srun python run_vega_mpi.py path_to/main.ini

We strongly suggest you run the sampler in parallel on many cores, as normal run-times are of the order :math:`10^2` - :math:`10^4` core hours.

Monte Carlo simulations are run with "run_vega_mc_mpi.py" and "run_vega_mc_fits_mpi.py" (see the Monte_carlo_sims folder).

Interactive use
---------------

You can run Vega interactively using Ipython or a Jupyter notebook. The "Vega_tutorial" notebook takes you through the steps of initializing Vega, computing a model and performing a fit.

The other notebooks cover specific parts of Vega:

- "SimpleModelTutorial": computing model correlations, including on a user-defined coordinate grid.
- "FitResultsTutorial": reading fit results and splitting the model into its components.
- "VegaPlots", "Plots_tutorial" and "VegaPlots2Datasets": plotting correlations and models.
- "Sensitivity_tutorial": calculating and plotting the model sensitivity and distribution of information available on each parameter over (rt,rp).
- "config_creation": a minimal example of building configs with ``BuildConfig``.
- "P3D_support": building configs and computing correlation function models for a custom input P(k).

This process is much more powerful compared to running in terminal as you directly have access to all the output, model components and fit results. Additionally, Vega was built in a modular structure with the aim of the user being able to call each module independently. Therefore, you have access to much more functionality this way. The `documentation`_ is the best source on how to run these modules independently, but if you can't find something there, please open an issue and we will try to help you and also improve the documentation.

DESI analyses
-------------

Each of the following folders contains a "build_configs.py" script, which generates the configs with ``BuildConfig``, together with the configs it generates. The configs use paths relative to their folder, so Vega must be run from inside it. The data and mock products are on NERSC.

- **DESI_data_BAO**: BAO fit of the four DESI DR2 Lya correlations (Lya(Lya)xLya(Lya), Lya(Lya)xLya(Lyb), Lya(Lya)xQSO and Lya(Lyb)xQSO), with the DESI DR3 BAO model of the correlations (metals, HCDs, UV background fluctuations, QSO velocity dispersion and proximity effect, instrumental systematics, analytic marginalisation of small scales). The linear parameters (instrumental systematics amplitude, QSO proximity strength, metal biases and metal x metal amplitudes) are marginalized analytically.
- **DESI_data_fullshape**: full-shape (Alcock-Paczynski) fit of the DESI DR2 correlations, with the same correlation model as the BAO example and independent rescaling of the peak and smooth components and a free growth rate.
- **DESI_mocks**: BAO fits of stacks of LyaCoLoRe 2LPT DESI DR2-like mocks:

  - **raw_and_true-cont**: true-continuum analysis of the mocks without contaminants (no distortion matrix). The raw analysis uses the same configs with different correlation files.
  - **uncontaminated**: analysis of the mocks without contaminants, with the distortion matrix. Two versions for the small-scale smoothing of the LyaCoLoRe field: a Gaussian damping (gauss_smoothing) or the LyaCoLoRe cell kernel with a free line-of-sight amplitude (los_smoothing).
  - **contaminated**: analysis of the mocks with DLAs, metals, BALs and QSO redshift errors, with the corresponding HCD, metal and QSO velocity dispersion models. The metal biases and metal x metal amplitudes are marginalized analytically.

Analytic marginalization of linear parameters
---------------------------------------------

The DESI_data_BAO, DESI_data_fullshape and DESI_mocks/contaminated examples marginalize the linear parameters of the model analytically instead of sampling them, which reduces the number of sampled parameters and the cost of fits and chains (see the `documentation`_ and ``docs/analytic_marginalization.rst``). Every metal x metal correlation gets its own amplitude ``amp_<m1>_<m2>`` (``metal-auto-amplitudes``), so that the metal biases enter the model linearly. The marginalized parameters are listed in the ``[analytic marginalization]`` section of "main.ini", with Gaussian priors in ``[priors]`` (flat otherwise), and do not appear in ``[sample]``. The "build_configs.py" scripts describe the chosen parameters and priors.

The priors of the marginalized parameters are for example purposes only: they should be updated for the data or mocks being fitted and validated before the results are trusted.

Running on the eBOSS DR16 correlations
--------------------------------------

In the eBOSS_DR16 folder you can find example config files for reproducing the eBOSS DR16 BAO analysis (table 6 of https://arxiv.org/abs/2007.08995). These have been adapted from the fitter2 config files. To run these you just need to put in the right paths to the measured correlations and metal matrices. You can download these from here: https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/dataveccov/lya_forest/.

Deprecated examples
-------------------

The deprecated folder contains the configs of the DESI DR1 (Y1) BAO analysis, on data (DESI_DR1_data) and on mocks (DESI_DR1_mocks). They are kept for reference, but use older data products and model choices, and absolute paths to the original analysis folders.
