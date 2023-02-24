====
Vega
====

.. image:: https://github.com/andreicuceu/vega/actions/workflows/python_package.yml/badge.svg?branch=master
    :target: https://github.com/andreicuceu/vega/actions/workflows/python_package.yml

.. image:: https://readthedocs.org/projects/lyafit/badge/?version=latest
        :target: https://vega.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/andreicuceu/Vega/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/andreicuceu/Vega


.. |alpha| image:: https://latex.codecogs.com/svg.image?{\color{Red}&space;\alpha}

Vega is a tool for computing 3D correlation function models for tracers used by the Ly-|alpha| forest group (such as Ly-|alpha| flux, Quasar positions or different metal lines) and for fitting data produced by `picca <https://github.com/igmhub/picca>`__ primarily to measure Baryon Acoustic Oscillations (BAO).

* Free software: GPL-3.0 License
* Documentation: https://vega.readthedocs.io.
* Referencing: If you use Vega in a publication please give the link to this repository (https://github.com/andreicuceu/vega). Right now there is no Vega paper. The best descriptions of what the code does are found in Cuceu et al. 2022 (https://arxiv.org/abs/2209.12931) and du Mas des Bourboux et al. 2020 (https://arxiv.org/abs/2007.08995).

Installation
------------

We recommend to start by creating a fresh conda environment. The following code will create this and also install all the dependencies:

.. code-block:: console

    $ conda create --name vega pip ipython jupyter jupyterlab ipykernel numpy scipy astropy numba h5py setuptools "iminuit>=2.0.0" cachetools matplotlib
    $ conda activate vega
    $ pip install mcfit

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/andreicuceu/vega.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/andreicuceu/Vega/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ cd vega
    $ pip install -e .

Installing the sampler
----------------------

First install mpi4py using the NERSC specific commmand:

.. code-block:: console

    $ MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

If you want to run the sampler, you will need `Polychord`_. Older instructions for Cori at NERSC can be found `here`_. Here are instructions for installing it on Perlmutter. Note that this requires the default Perlmutter envrinoment with no changes (except module load python). Start by following the steps above to install vega and its dendencies. After that clone `Polychord`_ and revert to an older stable commit:

.. code-block:: console

    $ git clone https://github.com/PolyChord/PolyChordLite.git
    $ cd PolyChordLite
    $ git checkout 3bad756
    
In the PolyChordLite folder you will find a make file named "Makefile_gnu". You need to open and edit this file by changing lines 2-4 from:

.. code-block:: make

    FC = mpifort
    CC = mpicc
    CXX = mpicxx
    
to

.. code-block:: make

    FC = ftn
    CC = CC
    CXX = CC
    
After that, you can install PolyChord:

.. code-block:: console

    $ make veryclean
    $ make COMPILER_TYPE=gnu
    $ pip install -e .

If you have any problems or questions about the sampler please raise an issue or email Andrei.


.. _tarball: https://github.com/andreicuceu/Vega/tarball/master
.. _Polychord: https://github.com/PolyChord/PolyChordLite
.. _here: https://github.com/andreicuceu/fitter2_tutorial

Usage
-----

Vega needs one "main.ini" file with the configuration, and at least one correlation config file. These correlation config files are generally of the form "lyaxlya.ini" for the Lyman alpha forest auto-correlation, or "qsoxlya.ini" for its cross-corelation with quasars. More complex cases also appear if we use the part of the Lyman alpha forest that appears left of the Lyman beta peak (i.e. in the Lyman beta part of the forest). These are generally called lyalyaxlyalyb.ini, which means we correlate Lya absorption in the Lya forest, denoted Lya(Lya), with Lya absorption in the Lyb part of the forest, denoted Lya(Lyb).

In the `examples`_ folder you can find examples of these config files with a lot of comments explaining what each option does. If you don't understand something, or we missed something, please open an issue.

Vega now has a Config `Builder`_ that is designed to create full Vega config files with minimal input. This is now the preffered way of interacting with Vega, as it automates fits and reduces the chance of mistakes. You can use the BuildConfig class interactively (e.g. in a notebook) as shown in this `tutorial`_.

.. _documentation: https://vega.readthedocs.io/en/latest/?badge=latest
.. _examples: https://github.com/andreicuceu/Vega/tree/master/examples
.. _Builder: https://github.com/andreicuceu/vega/blob/master/vega/build_config.py
.. _tutorial: https://github.com/andreicuceu/vega/blob/master/examples/config_creation.ipynb

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

You can run Vega interactively using Ipython or a Jupyter notebook. This `example`_ notebook takes you through the steps of intializing Vega, computing a model and performing a fit.

This process is much more powerful compared to running in terminal as you directly have access to all the output, model components and fit results. Additionally, Vega was built in a modular structure with the aim of the user being able to call each module independently. Therefore, you have access to much more functionality this way. The `documentation`_ is the best source on how to run these modules independently, but if you can't find something there, please open an issue and we will try to help you and also improve the documentation.

.. _example: https://github.com/andreicuceu/Vega/blob/master/examples/Vega_tutorial.ipynb

Credits
-------

This package is based on picca fitter2 found here: https://github.com/igmhub/picca/tree/master/py/picca/fitter2, and was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
