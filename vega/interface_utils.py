"""Helpers of ``VegaInterface``, kept separate to limit the size of ``vega_interface.py``.

The functions take the ``VegaInterface`` object as first argument. This module must not import
``vega.vega_interface``: the latter imports this module.
"""

import copy

import numpy as np

from vega.model import Model


def varied_parameter_names(vega_interface):
    """Names of all parameters that can take values different from [parameters].

    This is the union of the sampled parameters ([sample]), the parameters sampled in the
    Monte Carlo fits ([monte carlo], if present) and the parameters of the [chi2 scan]
    section (if present). The keys of the ``limits`` dictionaries are used for the sampled
    parameters, because ``_read_sample`` drops entries that are set to False and
    parameters missing from [parameters], so only parameters that are actually varied
    appear there.

    Parameters
    ----------
    vega_interface : vega.VegaInterface
        Interface whose ``sample_params``, ``mc_config`` and ``main_config`` are read.

    Returns
    -------
    set of str
        Names of the varied parameters.
    """
    varied_names = set(vega_interface.sample_params["limits"])

    if vega_interface.mc_config is not None:
        varied_names |= set(vega_interface.mc_config["sample"]["limits"])

    if "chi2 scan" in vega_interface.main_config:
        varied_names |= set(vega_interface.main_config["chi2 scan"])

    return varied_names


def matrix_exponent_names(vega_interface):
    """Names of the parameters frozen into the new-metal matrices, per correlation.

    Only correlations with metals, ``new_metals = True`` and
    ``metal_matrix_convention = estimator`` are included. For them the amplitude evolution
    of the metal terms is built into the metal matrices at initialization from the
    [parameters] values of ``alpha_<main tracer>`` (both main tracers) and
    ``alpha_<metal>`` (every metal species, i.e. every absorber of the metal correlations
    that is not a main tracer). The names are matched exactly, never by prefix.

    Parameters
    ----------
    vega_interface : vega.VegaInterface
        Interface whose ``corr_items`` and ``models`` are read. The models must be built.

    Returns
    -------
    dict
        Correlation name -> list of the parameter names, in the order main tracers then
        metal species.
    """
    exponent_names = {}
    for name, corr_item in vega_interface.corr_items.items():
        model = vega_interface.models.get(name)
        metals = None if model is None else model.metals
        if metals is None or not metals.new_metals:
            continue
        if metals.metal_matrix_convention != "estimator":
            continue

        main_tracers = list(dict.fromkeys(metals.main_tracers))
        species = []
        for corr_hash in corr_item.metal_correlations:
            for absorber in corr_hash:
                if absorber not in main_tracers and absorber not in species:
                    species.append(absorber)

        exponent_names[name] = [f"alpha_{absorber}" for absorber in main_tracers + species]

    return exponent_names


def check_metal_model_options(vega_interface):
    """Raise an error for model options that conflict with the varied parameters.

    All problems found are collected and reported in a single ValueError. Currently:

    - Under ``metal_matrix_convention = estimator`` with ``new_metals = True``, the
      amplitude evolution of the metal terms is built into the metal matrices at
      initialization from the [parameters] values of ``alpha_<main tracer>`` and
      ``alpha_<metal>``. These exponents therefore cannot be varied, whether sampled,
      in the Monte Carlo ([monte carlo]) or scanned ([chi2 scan]).
    - With ``fast_metals = True`` (both matrix conventions and both matrix paths) each
      metal x metal correlation is computed once, at the first model evaluation, and
      afterwards only rescaled by the metal biases. This is wrong if anything that enters
      its shape changes between evaluations, so the following are refused for every
      correlation with metals and fast metals: varied ``beta_metals``, ``beta_<metal>`` or
      (where the first check above does not already refuse it, i.e. under the legacy
      convention or with ``new_metals = False``) ``alpha_<metal>``; ``no-metal-decomp =
      False`` in [model] (the smooth evaluation would reuse the cached peak correlation);
      and ``metal-scaling = True`` in [cosmo-fit type]. The varied ``bias_<metal>``,
      ``bias_eta_<metal>``, ``bias_<metal1>_<metal2>`` and all parameters of the main
      tracers are allowed: they act after the cache, or only in the main x metal
      correlations, which are recomputed at every evaluation.

    This function is called at the end of ``VegaInterface.__init__``.

    Parameters
    ----------
    vega_interface : vega.VegaInterface
        Interface with the models built, whose varied parameters and model options are checked.

    Raises
    ------
    ValueError
        If a conflicting parameter is varied or a conflicting option is set.
    """
    varied_names = varied_parameter_names(vega_interface)
    messages = []

    # Matrix exponents are fixed at initialization (estimator convention, new metals)
    offending_exponents = {}
    for name, exponent_names in matrix_exponent_names(vega_interface).items():
        varied_exponents = [par for par in exponent_names if par in varied_names]
        if varied_exponents:
            offending_exponents[name] = varied_exponents

    if offending_exponents:
        details = "; ".join(
            f"'{name}': {', '.join(pars)}" for name, pars in offending_exponents.items()
        )
        messages.append(
            "With metal_matrix_convention = estimator (the default) and new_metals = True, "
            "the amplitude evolution of the metal terms is built into the metal matrices at "
            "initialization from the [parameters] values of alpha_<tracer> and alpha_<metal>. "
            "These exponents cannot be varied (sampled, in [monte carlo] or in [chi2 scan]). "
            f"Offending parameters per correlation: {details}. Fix them to their "
            "[parameters] values, or set metal_matrix_convention = legacy in [model] if "
            "they must vary (the legacy convention evaluates the evolution at runtime)."
        )

    # Fast metals: the metal x metal correlations are cached at the first evaluation
    exponent_names_per_correlation = matrix_exponent_names(vega_interface)
    for name, corr_item in vega_interface.corr_items.items():
        model = vega_interface.models.get(name)
        metals = None if model is None else model.metals
        if metals is None or not metals.fast_metals:
            continue

        # Metal species: absorbers of the metal correlations that are not main tracers
        main_tracers = [corr_item.tracer1["name"], corr_item.tracer2["name"]]
        species = []
        for corr_hash in corr_item.metal_correlations:
            for absorber in corr_hash:
                if absorber not in main_tracers and absorber not in species:
                    species.append(absorber)

        # Exact names of the metal shape parameters. The alpha_<metal> are left out where
        # the check above already refuses them (they then are refused with slow metals too,
        # so asking for fast_metals = False would not help).
        shape_names = ["beta_metals"] + [f"beta_{absorber}" for absorber in species]
        if name not in exponent_names_per_correlation:
            shape_names += [f"alpha_{absorber}" for absorber in species]
        varied_shape_names = [par for par in shape_names if par in varied_names]

        items = []
        if varied_shape_names:
            items.append(f"varied {', '.join(varied_shape_names)}")
        if not model.no_metal_decomp:
            items.append("no-metal-decomp = False")
        if vega_interface.scale_params.metal_scaling:
            items.append("metal-scaling = True in [cosmo-fit type]")

        if items:
            messages.append(
                f"fast_metals = True in [model] of '{name}' computes each metal×metal "
                "correlation once, at the first model evaluation, and afterwards only "
                f"rescales it by the metal biases. It cannot be combined with: "
                f"{'; '.join(items)}. To keep these options, set fast_metals = False in "
                f"[model] of '{name}' (the metal term is then slower, ≈10× for Lyα×Lyα)."
            )

    if messages:
        raise ValueError("\n".join(messages))


def build_models_for_mc_exponents(vega_interface, mc_params, print_func=print):
    """Build temporary models whose metal matrices use the Monte Carlo exponents.

    Under ``metal_matrix_convention = estimator`` the amplitude exponents ``alpha_<tracer>``
    and ``alpha_<metal>`` enter the new-metal matrices, which are built at initialization
    from ``corr_item.parameters``. If one of them differs between the Monte Carlo input
    parameters and [parameters], new ``Model`` objects are built with
    ``corr_item.parameters = vega_interface.params | mc_params``.

    Side effect: ``corr_item.parameters`` of every correlation is changed in that case. The
    caller must restore it to ``vega_interface.params`` and put back the original
    ``vega_interface.models``.

    Parameters
    ----------
    vega_interface : vega.VegaInterface
        Interface whose ``corr_items``, ``params``, ``fiducial``, ``scale_params`` and ``data``
        are used to build the models.
    mc_params : dict
        Final Monte Carlo input parameters (after merging with the best fit, if any).
    print_func : callable, optional
        Function used for log output, by default print

    Returns
    -------
    dict or None
        Temporary models keyed by component name, or None if no matrix exponent differs
        from [parameters] (the existing models are then valid).
    """
    changed_exponents = []
    for exponent_names in matrix_exponent_names(vega_interface).values():
        for par in exponent_names:
            if par in mc_params and mc_params[par] != vega_interface.params.get(par):
                if par not in changed_exponents:
                    changed_exponents.append(par)

    if not changed_exponents:
        return None

    print_func(
        "Monte Carlo input exponents differ from [parameters]: "
        + ", ".join(
            f"{par} = {mc_params[par]} ({vega_interface.params.get(par)})"
            for par in changed_exponents
        )
        + ". Building the metal matrices of the MC input model with the MC values."
    )

    mc_all_params = vega_interface.params | dict(mc_params)
    for corr_item in vega_interface.corr_items.values():
        corr_item.parameters = mc_all_params

    return {
        name: Model(
            corr_item,
            vega_interface.fiducial,
            vega_interface.scale_params,
            vega_interface.data[name],
        )
        for name, corr_item in vega_interface.corr_items.items()
    }


def compute_sensitivity(vega_interface, nominal=None, frac=0.1, verbose=True):
    """Compute the model sensitivity to each floating parameter.

    Calculate numerical partial derivatives of the model with respect to each floating
    pararameter, evaluated at a specified point in parameter space. Calculate Fisher information
    distributed over bins of (rt,rp).  Results are stored in a dictionary attribute
    named `sensitivity` with keys `nominal`, `partials`, and `fisher`.

    Parameters
    ----------
    vega_interface : vega.VegaInterface
        Initialized interface, with data and models built.
    nominal : dict or None
        Dictionary of (value,error) tuples for each floating parameter. Uses the results
        of the last call to minimize when None, or raises a RuntimeError when minimize
        has not yet been called.
    frac : float
        Estimate partial derivatives of the likelihood using central finite differences
        at value +/- frac * error for each floating parameter.
    verbose : bool
        Print progress of the computation when True.

    Raises
    ------
    RuntimeError
        If ``nominal`` is None and minimize has not been called.

    Notes
    -----
    Side effects: the results are stored in ``vega_interface.sensitivity`` (replacing any
    previous value), and ``vega_interface.fiducial["save-components"]`` is set to True, which
    stays set after the call. The model is evaluated with ``run_init=True`` at every point.
    """
    # Copy the baseline parameters to use.
    if nominal is None:
        if vega_interface.bestfit.params is None:
            raise RuntimeError("No nominal parameter values provided or saved by minimize()")
        nominal = {p.name: (p.value, p.error) for p in vega_interface.bestfit.params}

    params = copy.deepcopy(vega_interface.params)
    for pname, (pvalue, _) in nominal.items():
        params[pname] = pvalue

    # Initialize the sensitivity results.
    vega_interface.sensitivity = dict(nominal=copy.deepcopy(nominal), partials={}, fisher={})
    for name in vega_interface.corr_items:
        vega_interface.sensitivity["partials"][name] = {}
        vega_interface.sensitivity["fisher"][name] = {}

    # Loop over fit parameters
    vega_interface.fiducial["save-components"] = True
    bao_amp = vega_interface.params["bao_amp"]
    for pindex, (pname, (pvalue, perror)) in enumerate(nominal.items()):
        if verbose:
            print(f"Calculating sensitivity for [{pindex}] {pname} at {pvalue:.4f} ± {perror:.4f}")

        # Compute partial derivatives wrt to p for each multipole
        delta = frac * perror
        for sign in (+1, -1):
            params[pname] = pvalue + sign * delta
            # Compute the model for all datasets.
            cfs = vega_interface.compute_model(params, run_init=True)

            # Loop over datasets to update the partial derivative calculations.
            for n in cfs:
                if pname not in vega_interface.sensitivity["partials"][n]:
                    rp = vega_interface.corr_items[n].model_coordinates.rp_grid
                    vega_interface.sensitivity["partials"][n][pname] = np.zeros((2, 2, len(rp)))

                model = vega_interface.models[n]
                # Distorted peak
                vega_interface.sensitivity["partials"][n][pname][0, 0] += (
                    sign * bao_amp * model.xi_distorted["peak"]["core"]
                )

                # Distorted smooth
                vega_interface.sensitivity["partials"][n][pname][0, 1] += (
                    sign * model.xi_distorted["smooth"]["core"]
                )

                # Undistorted peak
                vega_interface.sensitivity["partials"][n][pname][1, 0] += (
                    sign * bao_amp * model.xi["peak"]["core"]
                )

                # Distorted smooth
                vega_interface.sensitivity["partials"][n][pname][1, 1] += (
                    sign * model.xi["smooth"]["core"]
                )

        # Normalize the partial derivatives.
        for n in vega_interface.corr_items:
            vega_interface.sensitivity["partials"][n][pname] /= 2 * delta

        # Restore the fitted parameter value.
        params[pname] = pvalue

    # Loop over pairs of fit parameters.
    if verbose:
        print("Computing Fisher information for each pair of parameters...")
    for pindex1, pname1 in enumerate(nominal):
        for pindex2, pname2 in enumerate(nominal):
            if pindex1 > pindex2:
                continue

            # Loop over datasets.
            for n in vega_interface.corr_items:
                if (pname1, pname2) not in vega_interface.sensitivity["fisher"][n]:
                    rp = vega_interface.corr_items[n].model_coordinates.rp_grid
                    vega_interface.sensitivity["fisher"][n][(pname1, pname2)] = np.zeros(
                        (2, len(rp))
                    )

                fisher = vega_interface.sensitivity["fisher"][n][(pname1, pname2)]
                # Lookup the data vector mask for this dataset
                mask = vega_interface.data[n].data_mask

                # Loop over distorted / non-distorted.
                for idistort in range(2):
                    # Combine peak + smooth partials.
                    partial1 = vega_interface.sensitivity["partials"][n][pname1][idistort].sum(
                        axis=0
                    )
                    partial2 = vega_interface.sensitivity["partials"][n][pname2][idistort].sum(
                        axis=0
                    )

                    # Calculate the Fisher info for all unmasked correlation bins.
                    masked_info = partial1[mask] * vega_interface.data[n].inv_masked_cov.dot(
                        partial2[mask]
                    )
                    fisher[idistort, mask] = masked_info
                    # Calculate the predicted inverse covariance for this parameter pair.
                    # ivar[idistort] = np.sum(fisher[idistort])
                    # ferror = ivar ** -0.5 if ivar > 0 else np.nan
                    # Set unused bins to NaN for plotting
                    fisher[idistort, ~mask] = np.nan
