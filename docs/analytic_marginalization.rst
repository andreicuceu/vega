Analytic marginalization
========================

Some nuisance parameters enter the correlation function model linearly. Vega can
marginalize over them analytically instead of sampling them, which reduces the
number of parameters in fits and chains without making likelihood evaluations
slower. The implementation lives in ``vega/analytic_marginalization.py``.

Two families of linear parameters are supported:

* the small-scale templates, one per bin of the undistorted correlation function
  at small separation;
* the coefficients of the additive broadband polynomials.

Method
------

The data vector of each correlation is modelled as

.. math::

    d = m(\theta) + T a + n ,

where :math:`m(\theta)` is the model for the sampled parameters, the columns of
:math:`T` are the linear templates propagated through the distortion matrix,
:math:`a` are their coefficients and :math:`n` is noise with covariance :math:`C`.
The templates do not depend on the sampled parameters, so all of the algebra is
done once at initialization.

Coefficients with a zero-mean Gaussian prior of covariance :math:`\Sigma` are
marginalized by the covariance update

.. math::

    C \rightarrow C + T \Sigma T^T .

Coefficients with a flat prior are then marginalized by projecting their
templates :math:`F` out of the resulting precision matrix :math:`P`,

.. math::

    P \rightarrow P - P F (F^T P F)^{-1} F^T P .

The best-fit coefficients for a residual :math:`r` are the solution of the joint
normal equations with the original covariance,

.. math::

    \hat a = (T^T C^{-1} T + \Lambda)^{-1} T^T C^{-1} r ,

where :math:`\Lambda` is the diagonal prior precision, zero for flat priors.

``chi2()`` computed with the marginalized precision equals the chi2 at the
best-fit coefficients, including the penalty of their Gaussian priors.
``log_lik()`` is the likelihood integrated over the coefficients. With a global
covariance the same operations are applied to the joint matrix, so the
covariance between correlations is taken into account.

Flat-prior templates must be linearly independent within the fitted bins;
otherwise Vega raises an error. The integral over a flat prior is defined up to
a constant that depends on the units of the coefficients, so evidences are only
comparable between runs that marginalize the same flat-prior templates.

Small-scale templates
---------------------

The existing options in the ``[model]`` section of a correlation config select
the templates (``marginalize-below-rtmax``, ``marginalize-above-rtmin``,
``marginalize-below-rpmax``, ``marginalize-above-rpmin``,
``marginalize-all-rmin-cuts``, ``marginalize-match-data-bins`` and
``fit-marginalized-scales``). All small-scale coefficients have a Gaussian prior
of width ``marginalize-prior-sigma`` (10 by default).

Additive broadbands
-------------------

To marginalize all additive polynomial coefficients of a correlation, add to the
``[model]`` section of its config::

    [model]
    marginalize-broadbands = True
    # Optional Gaussian prior width, the same for all coefficients of this
    # correlation. Without it the prior is flat.
    broadband-prior-sigma = 1e-4

    [broadband]
    1 = add pre r,mu 0:2:1 0:2:2

The following rules apply:

* The coefficients must still be listed in ``[parameters]`` of the main config.
  The model uses those values, and the marginalization is over deviations from
  them; a Gaussian prior is centred on them. With a flat prior the result does
  not depend on these values.
* The coefficients must not appear in ``[sample]``, ``[chi2 scan]`` or
  ``[priors]``.
* In the peak/smooth decomposition a coefficient :math:`a` contributes
  :math:`(1 + \mathrm{bao\_amp})\, a` times its polynomial, as it does when it is
  sampled. The templates are built with the ``bao_amp`` of ``[parameters]``, so
  ``bao_amp`` cannot be sampled or scanned while broadbands are marginalized.
* For the same reason, a Gaussian broadband prior cannot be used when the model is
  computed directly from the full power spectrum (``direct_pk``), which does not
  have this factor; Vega raises an error. Flat priors are not affected.
* Multiplicative broadbands are not applied to the marginalized templates. Vega
  prints a warning when a correlation has both.
* Only polynomial terms are marginalized; sky broadbands keep their usual
  treatment. Only rp/rt data are supported.

Best-fit coefficients
---------------------

The likelihood does not need the values of the marginalized coefficients. They
are recovered from the residual in one of two ways, selected in the ``[control]``
section of the main config:

* ``marginalize-in-fit = False`` (default): the coefficients are computed once,
  at the best fit, when ``minimize()`` is called.
* ``marginalize-in-fit = True``: the coefficients are also returned at every
  likelihood evaluation that requests them, and samplers save them as derived
  parameters.

The likelihood is identical in the two modes. ``bestfit_model`` and the model
written to the output file include the template contribution. From a script::

    chi2, coefficients = vega.chi2(params, return_marg_coeff=True)
    model = vega.compute_model(params, run_init=False, marg_coeff=coefficients)

``coefficients`` is a dictionary of arrays keyed by correlation name, ordered as
``vega.data[name].marginalization.coefficient_ids``.

Degrees of freedom
------------------

Each small-scale mode removes the information of one fitted bin, so these modes
are subtracted from the number of data points. Other analytically marginalized
coefficients are counted as model parameters. ``minimize()`` stores the
individual numbers in ``vega.dof_counts``, and they are written to the header of
the ``BESTFIT`` extension of the output file:

============ ==========================================================
``NDATA``    fitted data points, before any subtraction
``NFITPAR``  directly fitted parameters
``NSMALL``   small-scale marginalization modes
``NMARGOTH`` analytically marginalized parameters other than small-scale
``NMARG``    all analytically marginalized parameters
============ ==========================================================

The degrees of freedom are ``NDATA - NSMALL - NFITPAR - NMARGOTH``.
