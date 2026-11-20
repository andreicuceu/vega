Analytic marginalization
========================

Some nuisance parameters enter the correlation function model linearly. Vega can
marginalize over them analytically instead of sampling them, which reduces the
number of parameters in fits and chains. The implementation lives in
``vega/analytic_marginalization.py``.

Two kinds of linear parameters are supported:

* Families of coefficients selected per correlation, with fixed templates: the
  small-scale templates, one per bin of the undistorted correlation function at
  small separation, and the coefficients of the additive broadband polynomials.
  They cost nothing at likelihood evaluations.
* Individual parameters listed in the ``[analytic marginalization]`` section of the
  main config: ``desi_inst_sys_amp``, ``qso_rad_strength``, the metal biases and the
  metal x metal amplitudes. One parameter is one coefficient, whose template may
  enter several correlations, and some of these templates depend on the sampled
  parameters. They add a small cost to each likelihood evaluation (see below).

All marginalized coefficients follow one convention: the model is evaluated with
the coefficients set to zero, and the coefficients themselves are marginalized
with their prior. Values given for them in ``[parameters]`` do not enter the
likelihood and may be omitted (missing values are set to zero); when given, they
are used where the model is needed at fixed parameters, e.g. for the fiducial
model of Monte Carlo mocks, and Vega prints a warning if they are not zero.

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

* The coefficients do not need to be listed in ``[parameters]``. The Gaussian
  prior has zero mean; ``broadband-prior-sigma`` sets the same width for all
  marginalized coefficients of the correlation, so that they need no entries in
  ``[priors]``.
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

Parameters of the ``[analytic marginalization]`` section
--------------------------------------------------------

The parameters to marginalize are moved from ``[sample]`` to a new section of the
main config::

    [analytic marginalization]
    desi_inst_sys_amp = True
    qso_rad_strength = True
    bias_SiII(1190) = True
    amp_SiII(1190)_SiII(1193) = True

    [priors]
    desi_inst_sys_amp = gaussian 3e-4 1e-4

A parameter with an entry in ``[priors]`` has that Gaussian prior (the mean need not
be zero); otherwise its prior is flat, without bounds. These priors are applied by
the marginalization and do not enter the prior chi2 of the sampled parameters.
A parameter cannot be both marginalized and sampled or scanned.

Each parameter is one coefficient. Its template, the response of the model to a
unit value with all other parameters fixed, is computed by the model itself and
propagated through the distortion matrix of every correlation it enters:

* ``desi_inst_sys_amp``: auto-correlations with ``desi-instrumental-systematics``;
  fixed template.
* ``qso_rad_strength``: Lya x QSO correlations with ``radiation effects``; template
  rebuilt at each evaluation, as it depends on ``drp_QSO``.
* ``amp_<m1>_<m2>``: auto-correlations with ``metal-auto-amplitudes``; fixed template
  with ``fast_metals``, otherwise rebuilt at each evaluation (with a warning).
* ``bias_<m>`` or ``bias_eta_<m>``: all correlations with the metal <m>; template
  rebuilt at each evaluation, as it depends on the parameters of the main tracers.

All these terms enter the smooth (or full) component, so their templates carry no
``bao_amp`` factor. Multiplicative broadbands are not applied to them (warning).
Only rp/rt data are supported, and ``model_pk`` is not.

With :math:`P` the precision after the per-correlation marginalizations above,
:math:`r` the residual of the model with these parameters set to zero, :math:`T`
their templates and a Gaussian prior of mean :math:`\mu` and diagonal precision
:math:`\Lambda` (zero for flat priors), each evaluation computes

.. math::

    F = T^T P T + \Lambda , \qquad b = T^T P r + \Lambda \mu ,

    \chi^2 = r^T P r + \mu^T \Lambda \mu - b^T F^{-1} b ,

the chi2 at the best-fit coefficients :math:`\hat a = F^{-1} b` including the prior
penalty, and adds :math:`\log\det F + \sum \log \sigma^2 - n_{\rm flat} \log 2\pi`
(sum over the Gaussian coefficients) to :math:`-2 \log L`. The covariance is not
modified. Fixed templates are multiplied by the precision once; the templates
that change cost one product of the precision with the residual and these
templates. With templates that depend on the sampled parameters,
:math:`\log\det F` depends on them too, so ``chi2()`` (the profile over the
coefficients, used by the minimizer) and ``log_lik()`` (the marginal likelihood,
used by the samplers) are no longer related by a constant.

Templates with flat priors that are degenerate within the fitted bins are refused
at initialization; Gaussian priors remove the degeneracy.

Metal x metal amplitudes
~~~~~~~~~~~~~~~~~~~~~~~~

The metal biases enter the main x metal correlations linearly, but the metal x
metal correlations quadratically. To marginalize the metal biases, every metal x
metal correlation of the correlations involved needs its own amplitude, set with::

    [model]
    metal-auto-amplitudes = True

Each metal x metal correlation is then ``amp_<m1>_<m2>`` times the correlation
computed with unit biases, with the metal names sorted (e.g.
``amp_SiII(1190)_SiII(1193)``, ``amp_CIV(eff)_CIV(eff)``). An amplitude is shared by
all correlations that contain the metal pair (e.g. Lya x Lya and Lya x Lyb). The
amplitudes can also be sampled or fixed like any other parameter; with values
equal to the bias products the model is unchanged. The option needs
``fast_metal_bias = True`` (the default). It replaces the removed
``separate-metal-auto-biases`` option and its ``bias_<m1>_<m2>`` factors.

Marginalized metal biases also require ``no-metal-decomp = True`` (the default),
a given ``beta_<m>`` (or ``beta_metals``), and, for ``bias_eta_<m>``, no
``bias_<m>`` entry, which would replace it. A model computed directly from the full
power spectrum (``direct_pk``) has no metal correlations, so marginalized metal
parameters are refused there.

Generating configs with BuildConfig
-----------------------------------

``vega.BuildConfig`` writes all the options above:

* ``metal-auto-amplitudes`` (``BuildConfig`` option): ``amp_<m1>_<m2>`` is written to
  ``[parameters]`` for every metal x metal correlation that Vega builds for the
  requested correlations (the same pair selection as ``vega.data.Data``: CIV(eff) only
  with itself, and no pair of two names containing ``SiII``, which includes
  SiIII(1207), without ``use_metal_autos``). Values given in the ``parameters``
  argument of ``build`` take precedence; the others are set to the product of the
  written metal biases, which leaves the model unchanged. With
  ``use_metal_bias_eta`` there is no default: the amplitudes must be given, unless
  they are marginalized (then they are written as zero). The amplitudes can be
  sampled through ``fit_info["sample_params"]`` like any other parameter.
* ``fit_info["analytic_marginalization"]``: list of parameters written as
  ``<name> = True`` to ``[analytic marginalization]``. Their Gaussian priors go in
  ``fit_info["priors"]``. ``build`` refuses parameters that are also sampled, that are
  not among those listed above, or that are not part of the configuration (e.g.
  ``desi_inst_sys_amp`` without ``desi-instrumental-systematics``).
* ``marginalize-broadbands`` and ``broadband-prior-sigma``: entries of the dictionary
  of each correlation, next to its ``broadband`` entry, written to the ``[model]``
  section of that correlation config.
* ``fit_info["marginalize_in_fit"]`` (bool): written as ``marginalize-in-fit`` to
  ``[control]`` (see `Best-fit coefficients`_); absent by default.

For example::

    builder = BuildConfig({"metals": metals, "metal-auto-amplitudes": True})
    correlations["lyaxlya"]["marginalize-broadbands"] = True
    correlations["lyaxlya"]["broadband-prior-sigma"] = 1e-4
    fit_info["analytic_marginalization"] = ["desi_inst_sys_amp", "bias_SiII(1190)"]
    builder.build(correlations, "lyaxlya_lyaxqso", fit_info, out_path, parameters)

Best-fit coefficients
---------------------

The likelihood does not need the values of the marginalized coefficients. They
are recovered from the residual in one of two ways, selected in the ``[control]``
section of the main config:

* ``marginalize-in-fit = False`` (default): the coefficients are computed once,
  at the best fit, when ``minimize()`` is called.
* ``marginalize-in-fit = True``: the coefficients are also returned at every
  likelihood evaluation that requests them, and samplers save them as derived
  parameters. ``BuildConfig`` writes it with ``fit_info["marginalize_in_fit"] = True``.

The likelihood is identical in the two modes. ``bestfit_model`` and the model
written to the output file include the template contribution. From a script::

    chi2, coefficients = vega.chi2(params, return_marg_coeff=True)
    model = vega.compute_model(params, run_init=False, marg_coeff=coefficients)

``coefficients`` is a dictionary of arrays keyed by correlation name, ordered as
``vega.data[name].marginalization.coefficient_ids``, and, under ``"linear"``, the
coefficients of the ``[analytic marginalization]`` parameters, ordered as
``vega.marginalization.linear.names``. The latter are solved first; the
per-correlation coefficients are then solved for the model including their
contribution, which gives the joint best fit. Their errors in the output are
conditional on the ``[analytic marginalization]`` coefficients, and the errors of
these are marginalized over the per-correlation coefficients; both are at fixed
sampled parameters.

The output file has a ``MARGCOEFF`` extension with one row per coefficient (the
correlations a shared coefficient enters are comma-separated), and samplers name
the derived parameters of the ``[analytic marginalization]`` section after the
parameters. After a fit, ``vega.bestfit_marginalized_values`` holds the best-fit
values of the broadband and ``[analytic marginalization]`` coefficients;
``run_vega`` writes them with the other parameters.

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
