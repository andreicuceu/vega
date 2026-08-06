Welcome to Vega's documentation!
======================================

Vega is a tool for computing 3D correlation function and power spectrum models primarily for Lyman-α (Lyα) forest analyses. It is built to be modular and highly flexible in terms of the tracers being used. So far, Vega has been used to analyze the Lyα forest auto-correlation, its cross-correlation with galaxies and quasars (e.g., `Gerardi et al. 2022 <https://doi.org/10.1093/mnras/stac3257>`__, `Gordon et al. 2023 <https://doi.org/10.1088/1475-7516/2023/11/045>`__, `Herrera-Alcantar et al. 2025 <https://doi.org/10.1088/1475-7516/2025/12/053>`__, `Karaçaylı et al. 2026 <https://doi.org/10.48550/arXiv.2603.04281>`__), as well as auto- and cross-correlations of metal lines such as CIV and SiIV (e.g., `Guy et al. 2025 <https://doi.org/10.1088/1475-7516/2025/01/140>`__, `Bault et al. 2026 <https://doi.org/10.48550/arXiv.2601.08103>`__), Damped Lyman-α (DLA) absorbers (`Pérez-Ràfols et al. 2023 <https://doi.org/10.1093/mnras/stad1994>`__), and Strong Blended Lyman-α (SBLA) absorbers (`Pérez-Ràfols et al. 2023 <https://doi.org/10.1093/mnras/stad1994>`__). 

Vega is currently being used by the Lyα forest working group in DESI to measure Baryon Acoustic Oscillations (BAO) and perform full-shape analyses of Lyα forest auto- and cross-correlations (e.g., `DESI et al. 2025a <https://doi.org/10.1088/1475-7516/2025/01/124>`__, `DESI et al. 2025b <https://doi.org/10.1103/2wwn-xjm5>`__, `Cuceu et al. 2025 <https://doi.org/10.48550/arXiv.2509.15308>`__).

* Free software: GPL-3.0 License
* Documentation: https://vega.readthedocs.io.
* Referencing: If you use Vega in a publication, please give the link to this repository (https://github.com/andreicuceu/vega). The best descriptions of what the code does are found in `Cuceu et al. (2022) <https://doi.org/10.1093/mnras/stad1546>`__ and `Cuceu et al. (2025) <https://doi.org/10.48550/arXiv.2509.15308>`__.

.. toctree::
   :maxdepth: 2

   Introduction and Installation <intro>

There are several tutorials and examples available to help you get started with Vega:

.. toctree::
   :maxdepth: 1
   :caption: Tutorials & Examples:

   Old Vega Tutorial <examples/Vega_tutorial.ipynb>
   Simple Model Tutorial <examples/SimpleModelTutorial.ipynb>
   Fit Results Tutorial <examples/FitResultsTutorial.ipynb>
   Vega Plots Module <examples/VegaPlots.ipynb>
   Old Plots Tutorial <examples/Plots_tutorial.ipynb>
   Vega Plots 2 Datasets <examples/VegaPlots2Datasets.ipynb>
   Config Creation <examples/config_creation.ipynb>
   Sensitivity Tutorial <examples/Sensitivity_tutorial.ipynb>

Vega modules are documented here:

.. toctree::
   :maxdepth: 2

   Vega Modules <modules>

Other stuff:

.. toctree::
   :maxdepth: 1

   contributing
   authors
   history
   Some parameter descriptions (old) <config>

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
