======
lyafit
======


.. image:: https://img.shields.io/pypi/v/lyafit.svg
        :target: https://pypi.python.org/pypi/lyafit

.. image:: https://img.shields.io/travis/andreicuceu/lyafit.svg
        :target: https://travis-ci.com/andreicuceu/lyafit

.. image:: https://readthedocs.org/projects/lyafit/badge/?version=latest
        :target: https://lyafit.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Package for modeling and fitting the 3D Correlation Function for the Lyman-alpha forest.


* Free software: MIT license
* Documentation: https://lyafit.readthedocs.io.

Stuff from fitter2 yet to be implemented
--------

* More output in .h5 files (easy)
* Large scale fastmc (easy)

Stuff that needs improving
--------

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
