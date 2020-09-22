====
Vega
====


.. image:: https://img.shields.io/travis/andreicuceu/Vega.svg
        :target: https://travis-ci.com/andreicuceu/Vega

.. image:: https://readthedocs.org/projects/lyafit/badge/?version=latest
        :target: https://vega.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Vega is a tool for computing 3D correlation function models for tracers used by the Ly-:math:`\alpha` forest group (such as Ly-:math:`\alpha` flux, Quasar positions or different metal lines) and for fitting data produced by `picca <https://github.com/igmhub/picca>`__ primarily to measure Baryon Acoustic Oscillations (BAO).


* Free software: GPL-3.0 License
* Documentation: https://vega.readthedocs.io.

Installation
------------

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/andreicuceu/Vega

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/andreicuceu/Vega/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ cd Vega
    $ pip install -r requirements.txt
    $ pip install .

If you want to run the sampler you will need `Polychord`_. Instructions can be found `here`_, and will be added to this repo soon.

.. _tarball: https://github.com/andreicuceu/Vega/tarball/master
.. _Polychord: https://github.com/PolyChord/PolyChordLite
.. _here: https://github.com/andreicuceu/fitter2_tutorial


Credits
-------

This package is based on picca fitter2 found here: https://github.com/igmhub/picca/tree/master/py/picca/fitter2, and was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
