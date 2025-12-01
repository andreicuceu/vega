==============
Vega Tutorials
==============

Here you can find some examples on how to run Vega. If you can't find what you are looking for, also have a look at the `documentation`_, or open an issue.

.. _documentation: https://vega.readthedocs.io/en/latest/?badge=latest

Config files
------------

Vega needs one "main.ini" file with the configuration, and at least one correlation config file. These correlation config files are generally of the form "lyaxlya.ini" for the Lyman alpha forest auto-correlation, or "qsoxlya.ini" for its cross-corelation with quasars. More complex cases also appear if we use the part of the Lyman alpha forest that appears left of the Lyman beta peak (i.e. in the Lyman beta part of the forest). These are generally called lyalyaxlyalyb.ini, which means we correlate Lya absorption in the Lya forest, denoted Lya(Lya), with Lya absorption in the Lyb part of the forest, denoted Lya(Lyb).

Here you can find examples of these config files with a lot of comments explaining what each option does. If you don't understand something, or we missed something, please open an issue.

Using the terminal
------------------
You can call Vega from a terminal using the scripts in the bin folder, and pointing them to a "main.ini" file like this:

.. code-block:: console

    $ python run_vega.py path_to/main.ini

The "run_vega.py" script can be used for computing model correlations and for running the fitter. However, these can also be run interactively (see next section).

On the other hand the sampler (PolyChord) cannot be run interactively and needs to be called using the second script like this:

.. code-block:: console

    $ python run_vega_mpi.py path_to/main.ini

We strongly suggest you run the sampler in parallel on many cores, as normal run-times are of the order :math:`10^2` - :math:`10^4` core hours.

Interactive use
---------------

You can run Vega interactively using Ipython or a Jupyter notebook. The "Vega_tutorial" notebook takes you through the steps of intializing Vega, computing a model and performing a fit.

The "Sensitivity_tutorial" notebook shows how to calculate and plot the model sensitivity and distribution of information
available on each parameter over (rt,rp).

This process is much more powerful compared to running in terminal as you directly have access to all the output, model components and fit results. Additionally, Vega was built in a modular structure with the aim of the user being able to call each module independently. Therefore, you have access to much more functionality this way. The `documentation`_ is the best source on how to run these modules independently, but if you can't find something there, please open an issue and we will try to help you and also improve the documentation.

.. _documentation: https://vega.readthedocs.io/en/latest/?badge=latest

Running on the eBOSS DR16 correlations
--------------------------------------

In the eBOSS_DR16 folder you can find example config files for reproducing the eBOSS DR16 BAO analysis (table 6 of https://arxiv.org/abs/2007.08995). These have been adapted from the fitter2 config files. To run these you just need to put in the right paths to the measured correlations and metal matrices. You can download these from here: https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/dataveccov/lya_forest/.
