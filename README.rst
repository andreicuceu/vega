====
Vega
====


.. image:: https://img.shields.io/travis/andreicuceu/Vega.svg
        :target: https://travis-ci.com/andreicuceu/Vega

.. image:: https://readthedocs.org/projects/lyafit/badge/?version=latest
        :target: https://vega.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Vega is a tool for computing 3D correlation function models for tracers used by the Ly-:math:`\alpha` forest group (such as Ly-:math:`\alpha` flux, Quasar positions or different metal lines) and for fitting data produced by `picca <https://github.com/igmhub/picca>`__ primarily to measure Baryon Acoustic Oscillations (BAO).


* Free software: MIT license
* Documentation: https://vega.readthedocs.io.

Stuff from fitter2 yet to be implemented
----------------------------------------

* More output in .h5 files (easy)
* Large scale fastmc (easy)

Stuff that needs improving
--------------------------

* Coordinate rescaling models (in utils.py)
* Fast Hankel Transform
* Cache stuff that doesn't change between different components (e.g. peak_nl)

Features
--------

* TODO

Credits
-------

This package is based on picca fitter2 found here: https://github.com/igmhub/picca/tree/master/py/picca/fitter2

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
