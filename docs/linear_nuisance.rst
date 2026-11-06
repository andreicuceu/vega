Linear nuisance elimination (phase 1)
====================================

This opt-in module eliminates coefficients of a linear template matrix jointly
at each nonlinear parameter point. Without a ``[linear nuisance]`` section,
Vega retains its existing covariance-inflation and ``marginalize-in-fit`` paths.
Do not combine the new section with ``marginalize-in-fit = True``.

Scope and model convention
--------------------------

Phase 1 supports rp/rt correlation-function data, the existing undistorted
small-separation bin templates, and additive polynomial broadbands before or
after distortion. Metals, other physics terms, sky broadbands, multiplicative
broadbands, power-spectrum fits and multipole data are not selected for analytic
elimination. Physics, sky terms and multiplicative broadbands can still be part
of the base model and retain their existing parameter treatment. Multipole and
power-spectrum adapters are explicitly rejected in this phase.

The convention is ``d = m(theta) + T(theta) a + noise``. Polynomial coefficients
selected for elimination are zeroed in the base model. Their response follows
the current ``Model`` operations exactly: pre-additive polynomials pass through
distortion and the post-multiplicative broadband; post-additive polynomials are
added after both. In peak/smooth mode, both responses include ``1 + bao_amp``;
with ``direct_pk``, this factor is one. Templates are recomputed at each point,
including the current multiplicative broadband parameters.

Small-scale coefficients describe **one additive correction to the full
undistorted correlation**, inserted after the pre-broadband operations. They
pass through distortion and the post-multiplicative broadband, with no
``1 + bao_amp`` factor. This amplitude convention is explicit and differs from
changing both peak and smooth components. The established small-scale bin
selection, matching to data bins, and ``fit-marginalized-scales`` mask expansion
are reused. The legacy SVD compression, covariance inflation, and independent
coefficient solve are bypassed; no covariance is inflated a second time.

Configuration
-------------

Add this to the main INI to integrate over the configured small-scale modes and
all additive polynomial broadband coefficients::

    [linear nuisance]
    mode = integrate
    broadbands = True
    # Optional relative SVD rank threshold:
    # rank-rtol = 1e-12

``mode`` must be ``profile`` (default) or ``integrate``. Set ``broadbands = False``
to leave broadbands under ordinary parameter control and eliminate only the
configured small-scale modes. Existing per-component options such as
``marginalize-below-rtmax`` and ``marginalize-match-data-bins`` select small-scale
columns. Their default priors are zero-mean Gaussians with the component's
``marginalize-prior-sigma`` (default 10), **including** when
``fit-marginalized-scales`` and matching to data bins are enabled. Flat priors
require an explicit ``inf`` override in this new path.

Broadband IDs are the exact existing parameter names, for example
``BB-lyalya-0 add pre r,mu (0,0)``. Small-scale IDs are
``small-scale-lyalya (0)``, ``small-scale-lyalya (1)``, etc., in the column order
of ``CorrelationItem.get_undist_xi_marg_templates()``. IDs and column order are
stable for a fixed component configuration/grid, independent of the fitted
output mask; changing bin selection or bin matching can change their meaning.
The final joint ordering is lexicographically sorted and available as
``vega.linear_nuisance.ids``. Names are case-sensitive.

Additive polynomial coefficients have flat priors by default. Specify each
Gaussian as ``mean sigma``, or use ``0 inf`` for an explicitly flat prior::

    [linear nuisance priors]
    BB-lyalya-0 add pre r,mu (0,0) = 0 0.1
    small-scale-lyalya (0) = 0 inf

Remove eliminated coefficients from ``[sample]``, ``[monte carlo]`` sampled
parameters, ordinary ``[priors]`` and ``[chi2 scan]``. These combinations are rejected, rather
than silently double counting their priors. Initial fixed values in
``[parameters]`` do not set the eliminated coefficient values or prior means.
Use ``[linear nuisance priors]`` for those means.

Coefficients are independent across components unless explicitly aliased::

    [linear nuisance ids]
    BB-lyalya-0 add pre r,mu (0,0) = shared-offset
    BB-lyalyb-0 add pre r,mu (0,0) = shared-offset

    [linear nuisance priors]
    shared-offset = 0 0.1

Aliased responses are summed into one column, and the prior is applied once.
Prior overrides refer to the final aliased ID. Conflicting default priors for
an alias require an explicit shared prior. Unknown IDs and priors are rejected.

Objectives and covariance
-------------------------

For residual ``r = d - m``, precision ``P``, diagonal prior precision ``Lambda``
and prior mean ``mu``, the profile objective is the minimum of
``(r-Ta)^T P (r-Ta) + (a-mu)^T Lambda (a-mu)``. It is a penalized residual
quadratic, omitting parameter-independent prior normalizations.

With ``A = T^T P T + Lambda``, the integrated objective adds
``log(det(A)) + sum_proper(log(sigma**2)) - n_flat*log(2*pi)``. Vega's ordinary
data likelihood normalization is then applied by ``log_lik``. Finite Gaussian
priors are normalized: the result agrees with a covariance update
``C + T diag(sigma**2) T^T`` for proper independent priors. The determinant term
is retained at every nonlinear point, even when the template matrix depends
on ``bao_amp`` or multiplicative broadband parameters.

Flat priors integrate over unit Lebesgue measure in the stated coefficient
coordinates. Their likelihood has an arbitrary absolute normalization and must
not be used for absolute evidence comparisons. Rescaling such a template
changes its integrated volume. Rank-deficient flat directions are errors;
provide finite priors or remove redundancy. Proper Gaussian priors regularize
null/degenerate modes. An augmented SVD avoids forming the normal equations;
there is no pseudoinverse or silent truncation. The default relative rank
threshold is machine precision times the largest augmented matrix dimension.

All component templates and masked residuals are assembled in ``corr_items``
order. With a global covariance, the full masked global precision is used,
including off-diagonal blocks. Otherwise the precision is block diagonal.
Shared coefficients are solved jointly in either case. Monte Carlo evaluations
use the current global mock or the individual mocks and their scaled precision,
matching the existing likelihood's covariance convention. Priors are specified
in coefficient units and do not rescale with mock noise.

In ``integrate`` mode, ``chi2()`` and the minimizer's ``fval`` are effective
minus-two-log-likelihood objectives and may be negative. They are not ordinary
chi-square statistics. The new path reports PTE and reduced chi-square as the
existing unavailable sentinel ``-1`` in both modes; priors and shared
coefficients do not define simple per-component degrees of freedom. Individual
residual quadratic values remain descriptive and do not partition a global
covariance objective.

Notebook and script use
-----------------------

The existing script accepts the new section without a new command::

    python bin/run_vega.py main.ini

From a notebook or another script::

    from vega import VegaInterface

    vega = VegaInterface("main.ini")
    result = vega.compute_linear_nuisance({"bao_amp": 1.05})
    print(dict(zip(result.ids, result.coefficients)))
    conditional_covariance = result.covariance
    print(result.profile, result.objective)

    # compute_model alone returns the base model with eliminated terms zeroed.
    corrected = vega.compute_model(
        {"bao_amp": 1.05}, run_init=False,
        marg_coeff=dict(zip(result.ids, result.coefficients)),
    )
    vega.minimize()
    best = vega.bestfit_linear_nuisance
    corrected_bestfit = vega.bestfit_model

``LinearResult.covariance`` is conditional on the supplied nonlinear parameters;
it does not propagate their uncertainty. ``log_lik(return_marg_coeff=True)``
returns the coefficient array in final ID order, and sampler derived dimensions
use that same ordering. ``chi2(return_marg_coeff=True)`` wraps that array in a
``{"linear_nuisance": ...}`` dictionary. ``compute_marg_coeff`` remains the
legacy API and raises in the new mode; use ``compute_linear_nuisance`` instead.

The standard fit output contains the corrected best-fit model and FITS headers
``linear_nuisance_mode`` and ``linear coefficient <ID>``. The stored original
``[parameters]`` broadband values are inputs, not the solved coefficients.
Saved individual model components remain the base model components. Conditional
coefficient covariance is available on the notebook result, not written to a
new FITS extension in phase 1.

The numerical solver is usable independently of a Vega instance::

    from vega.linear_nuisance import solve_linear

    result = solve_linear(
        residual, templates, precision,
        ids=("offset", "slope"), means=[0, 0], sigmas=[0.1, float("inf")],
        mode="integrate",
    )

The current implementation uses dense joint template/precision arrays and a
cached precision Cholesky factor. Large-scale performance and scientific
validation on production covariances are deferred to NERSC. Focused laptop
tests cover covariance-update equivalence, explicit minimization, numerical
integration, rank/prior behavior, distortion and BAO response, joint masks,
shared IDs, Monte Carlo inputs, and real FITS/config fits and outputs.
