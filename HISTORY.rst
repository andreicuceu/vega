=======
History
=======

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

* First release on PyPI.
