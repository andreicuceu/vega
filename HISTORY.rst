=======
History
=======

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
