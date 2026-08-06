====
Vega
====

.. image:: https://github.com/andreicuceu/vega/actions/workflows/python_package.yml/badge.svg?branch=master
    :target: https://github.com/andreicuceu/vega/actions/workflows/python_package.yml

.. image:: https://readthedocs.org/projects/lyafit/badge/?version=latest
        :target: https://vega.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/andreicuceu/Vega/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/andreicuceu/Vega

Vega is a tool for computing 3D correlation function and power spectrum models primarily for Lyman-α (Lyα) forest analyses. It is built to be modular and highly flexible in terms of the tracers being used. So far, Vega has been used to analyze the Lyα forest auto-correlation and its cross-correlation with galaxies and quasars (e.g., `Gerardi et al. 2022 <https://doi.org/10.1093/mnras/stac3257>`__, `Gordon et al. 2023 <https://doi.org/10.1088/1475-7516/2023/11/045>`__, `Herrera-Alcantar et al. 2025 <https://doi.org/10.1088/1475-7516/2025/12/053>`__, `Karaçaylı et al. 2026 <https://doi.org/10.48550/arXiv.2603.04281>`__), as well as auto- and cross-correlations of metal lines such as CIV and SiIV (e.g., `Guy et al. 2025 <https://doi.org/10.1088/1475-7516/2025/01/140>`__, `Bault et al. 2026 <https://doi.org/10.48550/arXiv.2601.08103>`__), Damped Lyman-α (DLA) absorbers (`Pérez-Ràfols et al. 2023 <https://doi.org/10.1093/mnras/stad1994>`__), and Strong Blended Lyman-α (SBLA) absorbers (`Pérez-Ràfols et al. 2023 <https://doi.org/10.1093/mnras/stad1994>`__). 

Vega is currently being used by the Lyα forest working group in DESI to measure Baryon Acoustic Oscillations (BAO) and perform full-shape analyses of Lyα forest auto- and cross-correlations (e.g., `DESI et al. 2025a <https://doi.org/10.1088/1475-7516/2025/01/124>`__, `DESI et al. 2025b <https://doi.org/10.1103/2wwn-xjm5>`__, `Cuceu et al. 2025 <https://doi.org/10.48550/arXiv.2509.15308>`__).

* Free software: GPL-3.0 License
* Documentation: https://vega.readthedocs.io.
* Referencing: If you use Vega in a publication, please give the link to this repository (https://github.com/andreicuceu/vega). The best descriptions of what the code does are found in `Cuceu et al. (2022) <https://doi.org/10.1093/mnras/stad1546>`__ and `Cuceu et al. (2025) <https://doi.org/10.48550/arXiv.2509.15308>`__.

Installation
------------

We recommend to start by creating a fresh conda environment:

.. code-block:: console

    conda create --name vega python=3.13
    conda activate vega

You can either clone the public repository:

.. code-block:: console

    git clone https://github.com/andreicuceu/vega.git

Or download the `tarball`_:

.. code-block:: console

    curl -OJL https://github.com/andreicuceu/Vega/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    cd vega
    pip install -e .

If you want to install the development version with all optional dependencies, you should run:

.. code-block:: console

    pip install -e .[dev]

If you are at NERSC and want your vega environment to show up as Jupyter kernel, you can run the following command:

.. code-block:: console

    python -m ipykernel install --user --name vega --display-name Vega

The sampler and a few other modules in Vega need mpi4py. If you are at NERSC, you should install this using the NERSC-specific command:

.. code-block:: console

    MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

Vega currently has interfaces for one sampler: `Polychord`_. You do not need to install it to run the iminuit minimizer. You can find the instructions for installing at NERSC Polychord below.

.. _tarball: https://github.com/andreicuceu/Vega/tarball/master
.. _Polychord: https://github.com/PolyChord/PolyChordLite

Installing Polychord
--------------------

Here are instructions for installing Polychord at NERSC. Note that this requires the default Perlmutter environment with no changes (except module load python). Start by following the steps above to install vega and its dendencies. After that clone `Polychord`_:

.. code-block:: console

    git clone https://github.com/PolyChord/PolyChordLite.git
    cd PolyChordLite
    
In the PolyChordLite folder, you will find a make file named "Makefile_gnu". You need to open and edit this file by changing lines 2-4 from:

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

    make veryclean
    make COMPILER_TYPE=gnu
    pip install -e .

You can test if PolyChord works by running the test script on an interactive node:

.. code-block:: console

    srun -n 2 python run_pypolychord.py

Finally, you should add this line to your :code:`.bashrc` file, or at the beginning of your scripts (make sure to replace it with the correct path to your version of PolyChord):

.. code-block:: console

    export LD_LIBRARY_PATH=/path/to/PolyChordLite/lib:${LD_LIBRARY_PATH}

.. _Polychord: https://github.com/PolyChord/PolyChordLite

.. Installing PocoMC
.. -----------------

.. Here are instructions for installing PocoMC at NERSC. First, install Pytorch in CPU mode (see `this`_ for more details):

.. .. code-block:: console

..     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

.. Finally, install `PocoMC`_:

.. .. code-block:: console

..     pip install pocomc

.. .. _this: https://pytorch.org/get-started/locally/
.. .. _PocoMC: https://github.com/minaskar/pocomc

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

    python run_vega.py path_to/main.ini

The "run_vega.py" script can be used for computing model correlations and for running the fitter. However, these can also be run interactively (see next section).

On the other hand the sampler (PolyChord) cannot be run interactively and needs to be called using the second script like this:

.. code-block:: console

    python run_vega_mpi.py path_to/main.ini

We strongly suggest you run the sampler in parallel on many cores, as normal run-times are of the order :math:`10^2` - :math:`10^4` core hours.

Interactive use
---------------

You can run Vega interactively using Ipython or a Jupyter notebook. This `example`_ notebook takes you through the steps of intializing Vega, computing a model and performing a fit.

This process is much more powerful compared to running in terminal as you directly have access to all the output, model components and fit results. Additionally, Vega was built in a modular structure with the aim of the user being able to call each module independently. Therefore, you have access to much more functionality this way. The `documentation`_ is the best source on how to run these modules independently, but if you can't find something there, please open an issue and we will try to help you and also improve the documentation.

Vega also has a FitResults module for analysing the results of a fit. You can find example usage of it in this `notebook`_.

.. _example: https://github.com/andreicuceu/Vega/blob/master/examples/Vega_tutorial.ipynb
.. _notebook: https://github.com/andreicuceu/Vega/blob/master/examples/FitResultsTutorial.ipynb

Credits
-------

This package is based on picca fitter2 found here: https://github.com/igmhub/picca/tree/v4/py/picca/fitter2, and was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
