Configuration-space Fisher forecasts
===================================

``vega.fisher`` provides a small forecast for a supplied, fixed covariance.
It differentiates Vega's final correlation model and computes
``F = J.T @ solve(C, J) + F_prior``. It does not fit the data, generate mocks,
construct a survey covariance, or depend on FishHighZ or the new marginalization
work. The ordinary model, distortion, broadband, multipole, and blinding paths
remain active.

Notebook API
------------

.. code-block:: python

    from vega import VegaInterface
    from vega.fisher import compute_fisher

    vega = VegaInterface("main.ini")
    result = compute_fisher(
        vega,
        ["ap", "at"],
        fiducial={"ap": 1.0, "at": 1.0},
        steps={"ap": 0.001, "at": 0.001},
    )
    print(result.fisher, result.rank, result.step_changes)
    if result.sigma is not None:
        print(dict(zip(result.parameters, result.sigma)))
    else:
        print(result.null_directions)
    result.save("fisher.npz")

Parameter order is explicit; omitting the list uses ``[sample]`` order. The
fiducial starts from ``vega.params`` (the ``[parameters]`` values), with optional
overrides. It does not use minimizer start values, a best fit, or Monte Carlo
settings. Other model parameters are held fixed. To marginalize over nuisance
parameters in the local Gaussian approximation, include them in the forecast
list and read the desired submatrix of ``result.covariance``. Forecasting only
the cosmological parameters gives errors conditional on fixed nuisances.

Steps are required, positive, absolute central-difference displacements in input
parameter units. Both endpoints must lie inside any configured sample limits.
No one-sided substitution or automatic fit is performed. By default the code
compares derivatives at ``h`` and ``h/2`` in the covariance-weighted norm, warns
when their relative difference exceeds 0.05, and returns the ``h`` derivatives.
Inspect ``step_changes`` and repeat with several step sizes before using a
forecast scientifically. Zero sensitivity is exposed by the rank diagnostics.

Covariance and ordering
-----------------------

By default the forecast uses the global covariance configured under
``[data sets] global-cov-file``, retaining cross-correlations between components.
It reuses Vega's global data mask, including full-grid direct-multipole selection
and existing multipole transformations. Model vectors use each component's
``model_mask``. The common ordering is ``vega.corr_items`` order, followed by
the selected bins of each component in their native order (ell-major for direct
multipoles). Data masks and model masks may have different lengths but must
select the same number of bins.

Without a global covariance, the configured per-component covariances are
treated as independent blocks. A real covariance must be supplied in the data
file or via ``covariance-file``; the identity fallback used elsewhere in ``Data``
is rejected. Explicitly supplied identity covariances are allowed.

The notebook API also accepts ``covariance=C_selected``. This must already be
the selected joint covariance in the order above, with shape
``(sum(data.data_mask.sum()), sum(data.data_mask.sum()))`` over components.
It is used exactly as supplied: no Vega scaling, cuts, Hartlap correction,
projection, or small-scale covariance update is applied again. Select full
external matrices yourself, or use the ordinary global-covariance INI path for
Vega's full-grid and provenance handling.

Configured covariance paths retain Vega's covariance rescaling, Hartlap factors,
and any existing fixed small-scale covariance updates. Low-memory global mode
uses Vega's retained selected precision matrix. No additional correction is
applied. In particular, the reported parameter covariance has **no Percival
correction**; the configured factor is recorded in metadata for manual use.
The covariance must be finite, symmetric, and positive definite. There is no
covariance-derivative term and no automatic regularization.

Priors, rank, and blinding
-------------------------

``vega.priors`` supplies independent Gaussian prior widths by default. Pass
``priors={}`` to disable them or ``priors={"ap": (1.0, 0.02)}`` to replace them.
Each forecast prior contributes ``1/sigma**2`` to the diagonal; means do not
affect local curvature. Priors on fixed parameters do not change the forecast.
``fisher_data`` and ``fisher_prior`` are returned separately.

Rank uses eigenvalues of the diagonally normalized total Fisher matrix, with
relative cutoff ``rcond=1e-10``. ``eigenvalues`` and ``condition_number`` refer to
that normalized matrix. ``null_directions`` contains parameter-space columns in
the original parameter units. When numerical rank is deficient, ``covariance``
and ``sigma`` are ``None``: no pseudoinverse is reported as an uncertainty.
For full rank, ``sigma`` is the square root of the inverse Fisher diagonal.
Units, degeneracies, near-null directions, and the cutoff still need scientific
review; full rank alone does not establish forecast reliability.

Vega's normal additive parameter blinding is preserved. On blind data, all
forecast parameters must already have passed Vega's ``[sample]`` validation.
Fiducials are input-coordinate values; no unblinding offsets are written.
Fourier-space models, active Monte Carlo mode, and ``marginalize-in-fit`` are
rejected. Analytic elimination of linear nuisance templates is a future phase.

Command-line interface
----------------------

Append this section to an otherwise ordinary main INI:

.. code-block:: ini

    [fisher]
    parameters = ap at
    steps = 0.001 0.001
    fiducial = 1.0 1.0
    output = fisher.npz
    include-priors = True
    check-steps = True
    step-tolerance = 0.05
    rcond = 1e-10

``steps`` and optional ``fiducial`` contain one value per parameter, in exactly
the listed order. Omitting ``fiducial`` retains the normal parameter values;
omitting ``parameters`` uses the sampled parameters. Unknown options are rejected.
Paths follow the usual Vega INI conventions; the output is relative to the
working directory. Run from a checkout or installed environment:

.. code-block:: console

    python bin/run_vega_fisher.py main.ini
    python -m vega.scripts.run_vega_fisher main.ini --output alternative.npz

The installed script is ``run_vega_fisher.py``. Both entry points call the same
engine as the notebook API. The output path must not exist already.

The compressed NPZ contains parameter names, fiducial values, steps, selected
model, Jacobian, data/prior/total Fisher matrices, covariance, sigma, rank,
normalized eigenvalues, null directions, condition number, step checks, and JSON
metadata. Read it with ``numpy.load(path, allow_pickle=False)``. For singular
forecasts, saved covariance/sigma are empty arrays. Metadata records component
order, selected bin counts, covariance source, included priors, blinding flag,
and numerical settings.

Validation at NERSC
-------------------

The laptop tests cover analytic models, joint cross-covariance, distinct data
and model masks, full-grid direct-multipole selection, low-memory precision,
Gaussian priors, degeneracies, step checks, CLI dispatch, and an actual Vega
model's noiseless likelihood curvature. Before scientific use at NERSC:

* Run with the intended global covariance and verify its component/ell/bin
  ordering, units, cuts, and corrections against the data production pipeline.
* Repeat with smaller and larger steps; inspect every ``step_changes`` entry
  and any null directions, including weak nuisance directions.
* Compare local curvature and errors with a noiseless likelihood scan or fit,
  then inspect non-Gaussianity and degeneracies for the full analysis model.
* Verify prior conventions and manually handle any justified Percival factor.

API reference
-------------

.. automodule:: vega.fisher
   :members:
