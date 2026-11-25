import copy
import os
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

import git
import numpy as np
from astropy.io import fits

import vega
from vega.metals import Metals
from vega.utils import check_removed_z_options, find_file


class BuildConfig:
    """Build and manage config files based on available templates"""

    _params_template = None
    recognised_correlations = [
        "lyaxlya",
        "lyaxlyb",
        "lyaxqso",
        "lybxqso",
        "lyaxdla",
        "lybxdla",
        "lyaxsbla",
        "lybxsbla",
        "qsoxqso",
        "qsoxdla",
        "dlaxdla",
        "civxciv",
        "civxqso",
        "civxlya",
    ]

    # Optional [Polychord] settings read by ``vega.samplers.polychord.Polychord``
    # (``get_sampler_settings``), with the type it reads them as. They are written only
    # when given in ``fit_info["Polychord"]``, so that the sampler defaults apply otherwise.
    optional_polychord_settings = {
        "precision": "float",
        "seed": "int",
        "resume": "bool",
        "write_dead": "bool",
        "cluster_posteriors": "bool",
        "maximise": "bool",
    }

    def __init__(self, options=None, overwrite=False):
        """Initialize the model options that are not tracer or correlation specific.

        Parameters
        ----------
        options : dict, optional
            Dictionary with model options, by default None
            Here is a list of options:
                scale_params: string from ['ap_at', 'phi_alpha', 'aiso_epsilon'], default 'ap_at'
                template: string with custom_link, default 'PlanckDR16/PlanckDR16.fits'
                full_shape: Bool, default False
                full_shape_alpha: Bool, default False
                smooth_scaling: Bool, default False

                small_scale_nl: Bool, default False
                small_scale_nl_cross: Bool, default False
                bao_broadening: Bool, default False
                uv_background: Bool, default False
                velocity_dispersion: string from [None, 'lorentz', 'gauss'], default None
                radiation_effects: Bool, default False

                hcd_model: string from [None, 'Rogers', 'fvoigt', 'sinc'], default None
                fvoigt_model: string, default exp
                fullshape_smoothing: string from [None, 'gauss', 'gauss_iso', 'exp'], default None
                metals: List can include ['all', 'SiII(1190)', 'SiII(1193)', 'SiIII(1207)',
                    'SiII(1260)', 'CIV(eff)'], default None. CIV(eff) only enters through
                    CIV(eff) x CIV(eff), so it is not written to cross-correlations with a
                    discrete tracer (unless it is the only metal).
                metal_matrix_convention: string from ['estimator', 'legacy'], default
                    'estimator'. Convention of the new-metal matrices (only used with
                    new_metals = True); 'legacy' reproduces the matrices of previous versions.
                use_metal_autos: Bool, default True. Include the SiII x SiII metal
                    correlations.
                use_metal_bias_eta: Bool, default False. Write the metal biases as
                    bias_eta_<metal> instead of bias_<metal>.
                metal-auto-amplitudes: Bool, default False. Scale every metal x metal
                    correlation by its own amplitude amp_<metal1>_<metal2> (metal names
                    sorted) instead of the product of the metal biases. The amplitudes of
                    all metal x metal correlations that Vega builds for the requested
                    correlations are written to [parameters]; see ``build``.
        """
        if options is None:
            options = {}
        check_removed_z_options(options, "BuildConfig")

        self.overwrite = overwrite
        self.options = {}

        self.options["scale_params"] = options.get("scale_params", "ap_at")
        self.options["template"] = options.get("template", "PlanckDR16/PlanckDR16.fits")
        self.options["full_shape"] = options.get("full_shape", False)
        self.options["full_shape_alpha"] = options.get("full_shape_alpha", False)
        self.options["smooth_scaling"] = options.get("smooth_scaling", False)
        self.options["model-binning"] = options.get("model-binning", True)

        self.options["small_scale_nl"] = options.get("small_scale_nl", False)
        self.options["small_scale_nl_cross"] = options.get("small_scale_nl_cross", False)
        self.options["bao_broadening"] = options.get("bao_broadening", False)
        self.options["skip-nl-model-in-peak"] = options.get("skip-nl-model-in-peak", False)
        self.options["UVB-fluctuations"] = options.get("UVB-fluctuations", False)
        self.options["UVB-SN-cross"] = options.get("UVB-SN-cross", False)
        self.options["HeII-reionization"] = options.get("HeII-reionization", False)
        self.options["mock-bin-size"] = options.get("mock-bin-size", None)
        self.options["mock-los-smoothing"] = options.get("mock-los-smoothing", None)

        self.options["velocity_dispersion"] = options.get("velocity_dispersion", None)
        self.options["radiation_effects"] = options.get("radiation_effects", False)
        self.options["pk-damping-scale"] = options.get("pk-damping-scale", None)
        self.options["pk-damping-power"] = options.get("pk-damping-power", 2)

        self.options["marginalize-below-rtmax"] = options.get("marginalize-below-rtmax", None)
        self.options["marginalize-above-rtmin"] = options.get("marginalize-above-rtmin", None)
        self.options["marginalize-below-rpmax"] = options.get("marginalize-below-rpmax", None)
        self.options["marginalize-above-rpmin"] = options.get("marginalize-above-rpmin", None)

        self.options["marginalize-all-rmin-cuts"] = options.get("marginalize-all-rmin-cuts", False)
        self.options["marginalize-prior-sigma"] = options.get("marginalize-prior-sigma", 10.0)
        self.options["fit-marginalized-scales"] = options.get("fit-marginalized-scales", True)
        self.options["marginalize-match-data-bins"] = options.get(
            "marginalize-match-data-bins", True
        )

        self.options["hcd_model"] = options.get("hcd_model", None)
        self.options["fvoigt_model"] = options.get("fvoigt_model", "exp")
        self.options["fullshape_smoothing"] = options.get("fullshape_smoothing", None)
        self.options["fullshape_smoothing_metals"] = options.get(
            "fullshape_smoothing_metals", False
        )
        self.options["desi-instrumental-systematics"] = options.get(
            "desi-instrumental-systematics", False
        )
        self.options["test"] = options.get("test", False)
        self.options["use_metal_autos"] = options.get("use_metal_autos", True)
        self.options["new_metals"] = options.get("new_metals", False)
        self.options["rp_only_metal_mats"] = options.get("rp_only_metal_mats", False)
        self.options["metal_matrix_convention"] = options.get(
            "metal_matrix_convention", "estimator"
        )
        if self.options["metal_matrix_convention"] not in ("estimator", "legacy"):
            raise ValueError(
                f"Invalid metal_matrix_convention = '{self.options['metal_matrix_convention']}'. "
                "Use 'estimator' or 'legacy'."
            )
        self.options["metal-matrix"] = options.get("metal-matrix", {})
        self.options["rebin-metals"] = options.get("rebin-metals", None)
        self.options["use_metal_bias_eta"] = options.get("use_metal_bias_eta", False)
        if "separate-metal-auto-biases" in options:
            raise ValueError(
                "The option 'separate-metal-auto-biases' was removed. Use "
                "'metal-auto-amplitudes', which gives every metal x metal correlation its own "
                "amplitude amp_<metal1>_<metal2>."
            )
        self.options["metal-auto-amplitudes"] = options.get("metal-auto-amplitudes", False)
        self.options["single-metal-beta"] = options.get("single-metal-beta", False)

        metals = options.get("metals", None)
        if metals is not None:
            if "all" in metals:
                metals = ["SiII(1190)", "SiII(1193)", "SiIII(1207)", "SiII(1260)", "CIV(eff)"]
        self.options["metals"] = metals

    def build(
        self, correlations, fit_type, fit_info, out_path, parameters=None, name_extension=None
    ):
        """Build Vega config files and write them to an output directory

        Parameters
        ----------
        correlations : dict
            Information for each correlation. It must contain the path to the measured correlation,
            and the path to metal files if metals were requested. Optionally specify scale cuts.
            Options include ``corr_path`` (str), ``metal_path`` (str),
            ``r-min`` (float, default 10), ``r-max`` (float, default 180),
            ``rt-min`` (float, default 0), ``fast_metals`` (bool, default False),
            ``binsize`` (int, default 4; deprecated), ``broadband``
            (dict of broadband configuration strings), ``marginalize-broadbands``
            (bool, default False; analytically marginalize all additive polynomial
            broadband coefficients of this correlation, requires ``broadband``), and
            ``broadband-prior-sigma`` (float, optional; width of the zero-mean Gaussian
            prior of these coefficients, flat prior if absent; requires
            ``marginalize-broadbands``).
        fit_type : string
            Name of the fit. Includes the name of the correlations with the two
            tracers separated by an "x" (e.g. lyaxqso), and different correlations
            separated by an underscore (e.g. lyaxlya_lyaxqso). If unsure check
            the templates folder to see all possibilities.
        fit_info : dict
            Fit information. Must contain a list of sampled parameters and the effective redshift.
            Options include ``run_sampler`` (bool, default False),
            ``bias_beta_config`` (mapping tracers to ``bias_beta``,
            ``bias_eta_beta``, or ``bias_bias_eta``), ``zeff`` (float, default
            None), ``zeff_rmin`` (float, default 0), ``zeff_rmax`` (float,
            default 300), ``sample_params`` (list or parameter configuration
            dict), ``priors`` (parameter prior dict; sampled or analytically
            marginalized parameters only), ``analytic_marginalization`` (list of
            parameter names written as ``<name> = True`` to the ``[analytic
            marginalization]`` section), ``marginalize_in_fit`` (bool, optional;
            written as ``marginalize-in-fit`` to ``[control]``, so that the
            marginalized coefficients are computed at every likelihood evaluation and
            saved as derived parameters by the samplers; not written if absent), and
            ``Polychord`` (PolyChord configuration dict, required with ``run_sampler``).
            The ``Polychord`` dict may contain ``num_live`` (default
            ``25 * len(sample_params)``), ``num_repeats`` (default ``len(sample_params)``),
            ``do_clustering`` (default True) and ``boost_posterior`` (default 0), given
            as strings, and ``precision`` (float), ``seed`` (int), ``resume``,
            ``write_dead``, ``cluster_posteriors`` and ``maximise`` (bool), which are
            written only when given, so that the defaults of
            ``vega.samplers.polychord.Polychord`` apply otherwise.
        out_path : string
            Path to directory where to write the config files
        parameters : dict, optional
            Parameter values to write to the main config, by default None.
            With ``metal-auto-amplitudes``, the values of the amplitudes
            ``amp_<metal1>_<metal2>`` are taken from here; missing amplitudes are set to
            the product ``bias_<metal1> * bias_<metal2>`` of the written metal biases, which
            leaves the model unchanged. Amplitudes of metal pairs that Vega does not build
            for the requested correlations are not written.
        name_extension : string, optional
            Optional string to add to the config file names, by default None

        Returns
        -------
        string
            Path to the main config file

        Raises
        ------
        ValueError
            If a correlation, parameter or option is unknown or inconsistent, e.g. a
            parameter listed in ``fit_info["analytic_marginalization"]`` that is also
            sampled, or that Vega cannot marginalize analytically in this configuration,
            or a ``marginalize_in_fit`` or optional ``Polychord`` value that Vega cannot
            read as the expected type.
        """
        if parameters is None:
            parameters = {}

        # Save some of the info
        self.fit_info = fit_info
        self.name_extension = name_extension

        # Check if we need sampler or fitter or both
        self.fitter = fit_info.get("fitter", True)
        self.run_sampler = fit_info.get("run_sampler", False)

        # get the relevant paths
        self.config_path = Path(os.path.expandvars(out_path))
        assert self.config_path.is_dir()
        if self.fitter:
            self.fitter_out_path = self.config_path / "output_fitter"
            if not self.fitter_out_path.exists():
                os.mkdir(self.fitter_out_path)
        if self.run_sampler:
            self.sampler = fit_info.get("sampler", "Polychord")
            self.sampler_out_path = self.config_path / "output_sampler"
            if not self.sampler_out_path.exists():
                os.mkdir(self.sampler_out_path)

        # Check if we know the correlation types
        components = fit_type.split("_")
        for corr in components:
            if corr not in self.recognised_correlations:
                raise ValueError(f"Unknown correlation {corr}, part of fit type {fit_type}.")

        if len(components) != len(set(components)):
            print(f"Warning! fit type {fit_type} has duplicates")

        # Get git hash
        vega_path = Path(os.path.dirname(vega.__file__))
        try:
            git_hash = git.Repo(vega_path.parents[0]).head.object.hexsha
        except git.InvalidGitRepositoryError:
            git_hash = "None"

        # Build config files for each correlation
        self.corr_paths = []
        self.corr_names = []
        self.data_paths = []
        # Amplitudes amp_<m1>_<m2> of the metal x metal correlations, collected over all
        # correlations (filled only with metal-auto-amplitudes)
        self.metal_auto_amplitude_names = []
        for name in components:
            # Check if we have info on the correlation
            if name not in correlations:
                raise ValueError(
                    f"You asked for correlation {name} but did not provide"
                    ' its configuration in the "correlations" dictionary.'
                )

            # Build the config file for the correlation and save the path
            corr_path, data_path, tracer1, tracer2 = self._build_corr_config(
                name, correlations[name], git_hash
            )

            self.corr_paths.append(corr_path)
            self.data_paths.append(data_path)
            if tracer1 not in self.corr_names:
                self.corr_names.append(tracer1)
            if tracer2 not in self.corr_names:
                self.corr_names.append(tracer2)

        main_path = self._build_main_config(fit_type, fit_info, parameters, git_hash)

        return main_path

    def _build_corr_config(self, name, corr_info, git_hash):
        """Build config file for a correlation based on a template

        Parameters
        ----------
        name : string
            Name of the correlation. Must be the same as corresponding template file name
        corr_info : dict
            Correlation information. The paths to the data and metal files are required.
        git_hash : str
            Git hash of the current Vega version, written as a comment to the config file

        Returns
        -------
        string
            Path to the config file for the correlation.
        """
        # Read template
        config = ConfigParser()
        config.optionxform = lambda option: option
        template_path = find_file(f"vega/templates/{name}.ini")
        config.read(template_path)

        # get tracer info
        tracer1 = config["data"]["tracer1"]
        tracer2 = config["data"]["tracer2"]
        type1 = config["data"]["tracer1-type"]
        type2 = config["data"]["tracer2-type"]

        # Write the basic info
        config["data"]["filename"] = corr_info.get("corr_path")
        if "distortion-file" in corr_info:
            config["data"]["distortion-file"] = corr_info.get("distortion-file")
        if "covariance-file" in corr_info:
            config["data"]["covariance-file"] = corr_info.get("covariance-file")
        if "cov_rescale" in corr_info:
            config["data"]["cov_rescale"] = corr_info.get("cov_rescale")

        config["cuts"]["r-min"] = str(corr_info.get("r-min", 10))
        config["cuts"]["r-max"] = str(corr_info.get("r-max", 180))
        config["cuts"]["rt-min"] = str(corr_info.get("rt-min", 0))
        config["cuts"]["rp-min"] = str(corr_info.get("rp-min", -300))
        config["cuts"]["mu-min"] = str(corr_info.get("mu-min", -1))
        config["cuts"]["mu-max"] = str(corr_info.get("mu-max", 1))
        if self.options["test"]:
            config["data"]["test"] = "True"

        if "binsize" in corr_info:
            config["parameters"] = {}
            config["parameters"]["par binsize {}".format(name)] = str(corr_info.get("binsize", 4))
            config["parameters"]["per binsize {}".format(name)] = str(corr_info.get("binsize", 4))

        # Write the model options
        config["model"]["model binning"] = str(self.options["model-binning"])

        # Things that require LYA
        if tracer1 == "LYA" and tracer2 == "LYA":
            if self.options["small_scale_nl"]:
                config["model"]["small scale nl"] = "dnl_arinyo"
        elif tracer1 == "LYA" or tracer2 == "LYA":
            if self.options["small_scale_nl_cross"]:
                config["model"]["small scale nl"] = "dnl_arinyo"

        # Things that require both tracers to be continuous
        if type1 == "continuous" and type2 == "continuous":
            config["model"]["use_metal_autos"] = str(self.options["use_metal_autos"])
            if self.options["desi-instrumental-systematics"]:
                config["model"]["desi-instrumental-systematics"] = "True"

        # Things that require at least one tracer to be continuous
        if type1 == "continuous" or type2 == "continuous":
            if self.options["UVB-fluctuations"]:
                config["model"]["UVB-fluctuations"] = "True"

                # UV shotnoise is added to auto by default, and to cross only with extra flag
                if type1 == type2 or self.options["UVB-SN-cross"]:
                    config["model"]["UVB-shotnoise"] = "True"

            if self.options["HeII-reionization"]:
                config["model"]["HeII-reionization"] = "True"

            if self.options["hcd_model"] is not None:
                assert self.options["hcd_model"] in ["fvoigt", "Rogers2018", "sinc"]
                config["model"]["model-hcd"] = self.options["hcd_model"]
                if self.options["hcd_model"] == "fvoigt":
                    config["model"]["fvoigt_model"] = self.options["fvoigt_model"]

            if self.options["metals"] is not None:
                config["metals"] = {}
                config["metals"]["filename"] = corr_info.get("metal_path", "None")
                config["metals"]["z evol"] = "bias_vs_z_std"
                # CIV(eff) enters only through CIV(eff) x CIV(eff) (vega.data.Data), which a
                # cross-correlation with a discrete tracer does not have, so it is left out
                # there (unless it is the only metal, to keep a valid [metals] section)
                metals_in_correlation = self.options["metals"]
                if type1 != type2:
                    metals_without_civ = [
                        metal for metal in metals_in_correlation if metal != "CIV(eff)"
                    ]
                    if metals_without_civ:
                        metals_in_correlation = metals_without_civ

                if type1 == "continuous":
                    config["metals"]["in tracer1"] = " ".join(metals_in_correlation)
                if type2 == "continuous":
                    config["metals"]["in tracer2"] = " ".join(metals_in_correlation)

                if "fast_metals" in corr_info:
                    config["model"]["fast_metals"] = corr_info.get("fast_metals", "False")

                if self.options["metal-auto-amplitudes"]:
                    config["model"]["metal-auto-amplitudes"] = "True"

                    # Metal x metal correlations exist only if both tracers carry metals
                    if type1 == "continuous" and type2 == "continuous":
                        # Same comparison of the tracer dictionaries as in vega.data.Data
                        # (name and type; the weights paths of new_metals only reorder the
                        # pairs, which leaves the sorted amplitude names unchanged).
                        same_tracer = tracer1 == tracer2 and type1 == type2
                        amplitude_names = self.get_metal_auto_amplitude_names(
                            self.options["metals"],
                            self.options["metals"],
                            same_tracer,
                            self.options["use_metal_autos"],
                        )
                        for amplitude_name in amplitude_names:
                            if amplitude_name not in self.metal_auto_amplitude_names:
                                self.metal_auto_amplitude_names.append(amplitude_name)

                if self.options["single-metal-beta"]:
                    config["model"]["single-metal-beta"] = "True"

                new_metals_flag = self.options.get("new_metals", False)
                if new_metals_flag:
                    config["model"]["new_metals"] = "True"
                    config["model"]["rp_only_metal_mats"] = str(self.options["rp_only_metal_mats"])
                    config["model"]["metal_matrix_convention"] = self.options[
                        "metal_matrix_convention"
                    ]

                    config["data"]["weights-tracer1"] = corr_info.get("weights-tracer1")
                    config["data"]["weights-tracer2"] = corr_info.get("weights-tracer2")

                    config["metal-matrix"] = {}
                    if self.options["rebin-metals"] is not None:
                        config["metal-matrix"]["rebin_factor"] = str(
                            int(self.options["rebin-metals"])
                        )
                    else:
                        config["metal-matrix"]["rebin_factor"] = self.options["metal-matrix"].get(
                            "rebin_factor", "3"
                        )

                    # Estimator weighting exponents of the main tracers (how picca weighted the
                    # data). The deprecated names alpha_LYA and z_evol_objects are still read.
                    metal_matrix_options = self.options["metal-matrix"]
                    config["metal-matrix"]["weight_evol_LYA"] = metal_matrix_options.get(
                        "weight_evol_LYA", metal_matrix_options.get("alpha_LYA", "2.9")
                    )
                    config["metal-matrix"]["weight_evol_QSO"] = metal_matrix_options.get(
                        "weight_evol_QSO", metal_matrix_options.get("z_evol_objects", "1.44")
                    )

                    # The amplitude exponents alpha_<metal> are read from [parameters]. The legacy
                    # convention still reads them from [metal-matrix] if present there (alpha_LYA
                    # also sets the amplitude exponent of the Lya leg), so explicitly given values
                    # are kept to reproduce previous configurations.
                    if self.options["metal_matrix_convention"] == "legacy":
                        for option, value in metal_matrix_options.items():
                            if option.startswith("alpha_"):
                                config["metal-matrix"][option] = value

                    metal_alpha_options = [
                        option
                        for option in metal_matrix_options
                        if option.startswith("alpha_") and option != "alpha_LYA"
                    ]
                    if (
                        self.options["metal_matrix_convention"] == "estimator"
                        and metal_alpha_options
                    ):
                        print(
                            "WARNING: the [metal-matrix] options "
                            + ", ".join(metal_alpha_options)
                            + " are ignored. The amplitude exponents of the metals are taken "
                            "from [parameters] (alpha_<metal>)."
                        )

                    config["metal-matrix"]["z_ref_objects"] = self.options["metal-matrix"].get(
                        "z_ref_objects", "2.25"
                    )
                    config["metal-matrix"]["z_bins_objects"] = self.options["metal-matrix"].get(
                        "z_bins_objects", "1000"
                    )

        # Things that require at least one discrete tracer
        if type1 == "discrete" or type2 == "discrete":
            if self.options["velocity_dispersion"] is not None:
                assert self.options["velocity_dispersion"] in ["lorentz", "gauss"]
                config["model"]["velocity dispersion"] = self.options["velocity_dispersion"]

                if self.options["metals"] is not None and type1 != type2:
                    config["metals"]["velocity dispersion"] = self.options["velocity_dispersion"]

        # Only for the LYA - QSO cross
        if "LYA" in [tracer1, tracer2] and "QSO" in [tracer1, tracer2]:
            if self.options["radiation_effects"]:
                config["model"]["radiation effects"] = "True"

        # Marginalize small scales
        has_marg = False
        if self.options["marginalize-below-rtmax"] is not None:
            config["model"]["marginalize-below-rtmax"] = str(
                self.options["marginalize-below-rtmax"]
            )
            has_marg = True
        if self.options["marginalize-above-rtmin"] is not None:
            config["model"]["marginalize-above-rtmin"] = str(
                self.options["marginalize-above-rtmin"]
            )
            has_marg = True
        if self.options["marginalize-below-rpmax"] is not None:
            config["model"]["marginalize-below-rpmax"] = str(
                self.options["marginalize-below-rpmax"]
            )
            has_marg = True
        if self.options["marginalize-above-rpmin"] is not None:
            config["model"]["marginalize-above-rpmin"] = str(
                self.options["marginalize-above-rpmin"]
            )
            has_marg = True

        # This should appear even if turned off to inform user
        config["model"]["marginalize-all-rmin-cuts"] = str(
            self.options["marginalize-all-rmin-cuts"]
        )

        # These options are only needed if marginalization is turned on
        if has_marg or self.options["marginalize-all-rmin-cuts"]:
            config["model"]["marginalize-prior-sigma"] = str(
                self.options["marginalize-prior-sigma"]
            )
            config["model"]["fit-marginalized-scales"] = str(
                self.options["fit-marginalized-scales"]
            )
            config["model"]["marginalize-match-data-bins"] = str(
                self.options["marginalize-match-data-bins"]
            )

        if self.options["skip-nl-model-in-peak"]:
            config["model"]["skip-nl-model-in-peak"] = str(self.options["skip-nl-model-in-peak"])

        # P(k) damping scale
        if self.options["pk-damping-scale"] is not None:
            config["model"]["pk-damping-scale"] = str(self.options["pk-damping-scale"])
            config["model"]["pk-damping-power"] = str(self.options["pk-damping-power"])

        # General things
        if "broadband" in corr_info:
            config["broadband"] = {}
            for key, item in corr_info["broadband"].items():
                config["broadband"][key] = item

        # Analytic marginalization of the additive broadband polynomials. These are
        # per-correlation entries, like the broadbands they act on; the options are written
        # only when requested, so configs built without them are unchanged.
        marginalize_broadbands = bool(corr_info.get("marginalize-broadbands", False))
        if marginalize_broadbands:
            if "broadband" not in corr_info:
                raise ValueError(
                    f"marginalize-broadbands is set for {name}, but no broadband was given "
                    "in its correlation dictionary."
                )
            config["model"]["marginalize-broadbands"] = "True"

        if "broadband-prior-sigma" in corr_info:
            if not marginalize_broadbands:
                raise ValueError(
                    f"broadband-prior-sigma is set for {name} without marginalize-broadbands; "
                    "Vega would ignore it."
                )
            broadband_prior_sigma = float(corr_info["broadband-prior-sigma"])
            if not broadband_prior_sigma > 0:
                raise ValueError(f"broadband-prior-sigma must be positive ({name}).")
            config["model"]["broadband-prior-sigma"] = str(corr_info["broadband-prior-sigma"])

        if self.options["fullshape_smoothing"] is not None:
            assert self.options["fullshape_smoothing"] in ["gauss", "gauss_iso", "exp"]
            config["model"]["fullshape smoothing"] = self.options["fullshape_smoothing"]

            condition = type1 == "continuous" or type2 == "continuous"
            condition &= self.options["metals"] is not None
            condition &= self.options["fullshape_smoothing_metals"]
            if condition:
                config["metals"]["fullshape smoothing"] = self.options["fullshape_smoothing"]

        if self.options["mock-bin-size"] is not None:
            config["model"]["mock-bin-size"] = str(self.options["mock-bin-size"])
            if self.options["metals"] is not None:
                config["metals"]["mock-bin-size"] = str(self.options["mock-bin-size"])
            if self.options["mock-los-smoothing"] is not None:
                config["model"]["mock-los-smoothing"] = self.options["mock-los-smoothing"]
                if self.options["metals"] is not None:
                    config["metals"]["mock-los-smoothing"] = self.options["mock-los-smoothing"]

        if self.name_extension is None:
            corr_path = self.config_path / "{}.ini".format(name)
        else:
            corr_path = self.config_path / "{}-{}.ini".format(name, self.name_extension)

        if corr_path.is_file() and not self.overwrite:
            raise ValueError(f"File {corr_path} already exists. Please change the name extension.")

        with open(corr_path, "w") as configfile:
            configfile.write(f"# File written on {datetime.now()} \n")
            configfile.write(f"# Vega git hash: {git_hash} \n\n")
            config.write(configfile)

        return corr_path, config["data"]["filename"], tracer1, tracer2

    @staticmethod
    def get_zeff(data_paths, rmin=0.0, rmax=300.0):
        """Compute effective redshift of all correlations

        Parameters
        ----------
        data_paths : List[string]
            List of paths to exported picca correlations
        rmin : float, optional
            Minimum separation, by default 0.
        rmax : float, optional
            Maximum separation, by default 300.

        Returns
        -------
        float
            Effective redshift
        """
        zeff_list = []
        weights = []
        for path in data_paths:
            hdul = fits.open(path)

            r_arr = np.sqrt(hdul[1].data["RP"] ** 2 + hdul[1].data["RT"] ** 2)
            cells = (r_arr > rmin) & (r_arr < rmax)

            inverse_variance = 1 / np.diag(hdul[1].data["CO"])
            zeff = np.average(hdul[1].data["Z"][cells], weights=inverse_variance[cells])
            weight = np.sum(inverse_variance[cells])

            hdul.close()

            zeff_list.append(zeff)
            weights.append(weight)

        zeff = np.average(zeff_list, weights=weights)
        return zeff

    @staticmethod
    def get_metal_auto_amplitude_names(
        metals_in_tracer1, metals_in_tracer2, same_tracer, use_metal_autos=True
    ):
        """Amplitude names of the metal x metal correlations of one correlation.

        Mirrors the pair selection of ``vega.data.Data._init_metal_correlations`` and
        ``Data._use_correlation``: all pairs (metal1, metal2) of the metals of the two
        tracers, with metal2 not before metal1 in the list when the two main tracers are
        the same; CIV(eff) only with itself; SiII x SiII pairs only with
        ``use_metal_autos``. Names follow ``vega.metals.Metals.amplitude_name``.

        Parameters
        ----------
        metals_in_tracer1 : list of str
            Metals contributing to the first tracer.
        metals_in_tracer2 : list of str
            Metals contributing to the second tracer.
        same_tracer : bool
            Whether the two main tracers are the same (auto-correlation).
        use_metal_autos : bool, optional
            Whether the SiII x SiII correlations are used, by default True.

        Returns
        -------
        list of str
            Unique names ``amp_<m1>_<m2>`` (metal names sorted), in the order in which
            Vega builds the correlations.
        """

        def use_pair(metal1, metal2):
            # Same rules as Data._use_correlation
            if metal1 == "CIV(eff)" or metal2 == "CIV(eff)":
                return metal1 == metal2
            return use_metal_autos or not ("SiII" in metal1 and "SiII" in metal2)

        amplitude_names = []
        for i, metal1 in enumerate(metals_in_tracer1):
            first_index = i if same_tracer else 0
            for metal2 in metals_in_tracer2[first_index:]:
                if not use_pair(metal1, metal2):
                    continue
                amplitude_name = Metals.amplitude_name(metal1, metal2)
                if amplitude_name not in amplitude_names:
                    amplitude_names.append(amplitude_name)

        return amplitude_names

    def _build_main_config(self, fit_type, fit_info, parameters, git_hash):
        """Build the main vega configuration file

        Parameters
        ----------
        fit_type : string
            Name of the fit. Includes the name of the correlations with the two
            tracers separated by an 'x' (e.g. lyaxqso),
            and different correlations separated by an underscore
            (e.g. lyaxlya_lyaxqso). If unsure check the templates
            folder to see all possibilities.
        fit_info : dict
            Fit information. Must contain a list of sampled parameters and the effective
            redshift. See ``build`` for the options.
        parameters : dict, optional
            Parameter values to write to the main config

        Returns
        -------
        Path
            Path to written 'main.ini' file to be used with vega
        """
        # Initialize the config
        config = ConfigParser()
        config.optionxform = lambda option: option

        # Check the effective redshift
        self.zeff_in = fit_info.get("zeff", None)
        zeff_rmin = float(fit_info.get("zeff_rmin", 0.0))
        zeff_rmax = float(fit_info.get("zeff_rmax", 300.0))

        if self.zeff_in is None:
            zeff_comp = self.get_zeff(self.data_paths, zeff_rmin, zeff_rmax)
            self.zeff_in = zeff_comp

        self.zeff_in = float(self.zeff_in)

        # Write the paths to the correlation configs
        config["data sets"] = {}
        config["data sets"]["zeff"] = str(self.zeff_in)
        corr_paths = [str(path) for path in self.corr_paths]
        config["data sets"]["ini files"] = " ".join(corr_paths)
        if "global_cov_file" in fit_info:
            config["data sets"]["global-cov-file"] = fit_info.get("global_cov_file")

        # Write the scale parameters functions
        config["cosmo-fit type"] = {}
        config["cosmo-fit type"]["cosmo fit func"] = self.options["scale_params"]
        config["cosmo-fit type"]["full-shape"] = str(self.options["full_shape"])
        config["cosmo-fit type"]["full-shape-alpha"] = str(self.options["full_shape_alpha"])
        config["cosmo-fit type"]["smooth-scaling"] = str(self.options["smooth_scaling"])

        # Write the template info
        config["fiducial"] = {}
        config["fiducial"]["filename"] = self.options["template"]

        # Write the output path
        run_name = fit_type
        if self.name_extension is not None:
            run_name += "-{}".format(self.name_extension)
        config["output"] = {}
        config["output"]["filename"] = str(self.fitter_out_path / run_name)

        # Write the sampled parameters
        sample_params = fit_info["sample_params"]
        config["sample"] = {}
        if type(sample_params) is list:
            for param in sample_params:
                config["sample"][param] = "True"
        elif type(sample_params) is dict:
            for param, setup in sample_params.items():
                config["sample"][param] = setup
        else:
            raise TypeError("The sample_params object has to be either a list or a dict.")

        # Parameters marginalized analytically by Vega instead of being sampled
        if isinstance(fit_info.get("analytic_marginalization"), str):
            raise TypeError("fit_info['analytic_marginalization'] must be a list of names.")
        marginalized_params = list(fit_info.get("analytic_marginalization", []))

        # Write the priors (Gaussian priors of marginalized parameters also go here)
        if "priors" in fit_info:
            config["priors"] = {}
            for par, prior in fit_info["priors"].items():
                assert par in config["sample"] or par in marginalized_params, (
                    "Cannot add prior for parameter that is not sampled or analytically "
                    "marginalized"
                )
                config["priors"][par] = prior

        # Write the parameters
        self.parameters = parameters

        config["parameters"] = {}
        for name, value in self.parameters.items():
            config["parameters"][name] = str(value)

        # Check all sampled parameters are defined
        for param in sample_params:
            if param not in config["parameters"]:
                raise ValueError(
                    f'Asked for unknown parameter "{param}". This does not exist in '
                    "the current configuration. Please check the vega configuration "
                    "you requested is correct. If this is a new parameter that does "
                    "not have a default value yet, please add it to the parameters "
                    "dictionary when calling BuildConfig."
                )

        # Write the analytically marginalized parameters
        if marginalized_params:
            self._check_marginalized_params(marginalized_params, sample_params, config)
            config["analytic marginalization"] = {}
            for param in marginalized_params:
                config["analytic marginalization"][param] = "True"

        # Check if we need the sampler
        config["control"] = {"run_sampler": "False"}
        if "use_template_growth_rate" in fit_info:
            config["control"]["use_template_growth_rate"] = fit_info["use_template_growth_rate"]
        # Written only when given, so that configs built without it are unchanged
        if "marginalize_in_fit" in fit_info:
            config["control"]["marginalize-in-fit"] = self._format_config_value(
                "marginalize_in_fit", fit_info["marginalize_in_fit"], "bool"
            )
        if self.run_sampler:
            config["control"]["run_sampler"] = "True"
            config["control"]["sampler"] = self.sampler
            config["control"]["low_mem_mode"] = fit_info.get("low_mem_mode", "False")
            if self.sampler == "Polychord":
                config["Polychord"] = {}
                config["Polychord"]["path"] = str(self.sampler_out_path)
                config["Polychord"]["name"] = run_name

                config["Polychord"]["num_live"] = fit_info["Polychord"].get(
                    "num_live", str(25 * len(sample_params))
                )
                config["Polychord"]["num_repeats"] = fit_info["Polychord"].get(
                    "num_repeats", str(len(sample_params))
                )
                config["Polychord"]["do_clustering"] = fit_info["Polychord"].get(
                    "do_clustering", "True"
                )
                config["Polychord"]["boost_posterior"] = fit_info["Polychord"].get(
                    "boost_posterior", str(0)
                )

                # Optional settings, e.g. the evidence precision criterion and the seed
                for key, value_type in self.optional_polychord_settings.items():
                    if key in fit_info["Polychord"]:
                        config["Polychord"][key] = self._format_config_value(
                            f"Polychord[{key!r}]", fit_info["Polychord"][key], value_type
                        )
            elif self.sampler == "PocoMC":
                config["PocoMC"] = {}
                config["PocoMC"]["path"] = str(self.sampler_out_path)
                config["PocoMC"]["name"] = run_name

                config["PocoMC"]["precondition"] = fit_info["PocoMC"].get("precondition", "True")
                config["PocoMC"]["dynamic"] = fit_info["PocoMC"].get("dynamic", "False")
                config["PocoMC"]["n_effective"] = fit_info["PocoMC"].get("n_effective", "512")
                config["PocoMC"]["n_active"] = fit_info["PocoMC"].get("n_active", "256")
                config["PocoMC"]["n_total"] = fit_info["PocoMC"].get("n_total", "1024")
                config["PocoMC"]["n_evidence"] = fit_info["PocoMC"].get("n_evidence", "0")
                config["PocoMC"]["save_every"] = fit_info["PocoMC"].get("save_every", "3")
                config["PocoMC"]["use_mpi"] = fit_info["PocoMC"].get("use_mpi", "True")
                config["PocoMC"]["num_cpu"] = fit_info["PocoMC"].get("num_cpu", "64")
            else:
                raise ValueError(
                    f"Sampler {self.sampler} is not supported. "
                    'Please choose from ["Polychord", "PocoMC"].'
                )

        if "monte_carlo" in fit_info:
            config["mc parameters"] = {}
            for key, value in fit_info["monte_carlo"]["parameters"].items():
                config["mc parameters"][key] = str(value)

            config["control"]["run_montecarlo"] = "True"
            if "forecast" in fit_info["monte_carlo"]:
                config["control"]["forecast"] = str(fit_info["monte_carlo"]["forecast"])

            if "global_cov_rescale" in fit_info["monte_carlo"]:
                config["control"]["global_cov_rescale"] = str(
                    fit_info["monte_carlo"]["global_cov_rescale"]
                )

            if "mc_output" in fit_info["monte_carlo"]:
                config["output"]["mc_output"] = str(fit_info["monte_carlo"]["mc_output"])

            if "num_mc_mocks" in fit_info["monte_carlo"]:
                config["control"]["num_mc_mocks"] = str(fit_info["monte_carlo"]["num_mc_mocks"])

            if "mc_seed" in fit_info["monte_carlo"]:
                config["control"]["mc_seed"] = str(fit_info["monte_carlo"]["mc_seed"])

            if "run_mc_fits" in fit_info["monte_carlo"]:
                config["control"]["run_mc_fits"] = str(fit_info["monte_carlo"]["run_mc_fits"])

            config["monte carlo"] = copy.deepcopy(config["sample"])
            config["sample"] = {}

        # Write main config
        if self.name_extension is None:
            main_path = self.config_path / "main.ini"
        else:
            main_path = self.config_path / "main-{}.ini".format(self.name_extension)

        if main_path.is_file() and not self.overwrite:
            raise ValueError(f"File {main_path} already exists. Please change the name extension.")

        with open(main_path, "w") as configfile:
            configfile.write(f"# File written on {datetime.now()} \n")
            configfile.write(f"# Vega git hash: {git_hash} \n\n")
            config.write(configfile)

        return main_path

    @staticmethod
    def _format_config_value(name, value, value_type):
        """Format a config value and check that Vega can read it as the expected type.

        Parameters
        ----------
        name : str
            Name of the ``fit_info`` entry, used in the error message.
        value : bool, int, float or str
            Value to write.
        value_type : {"bool", "int", "float"}
            Type with which Vega reads the option (``getboolean``, ``getint`` or
            ``getfloat``).

        Returns
        -------
        str
            ``str(value)``; Python booleans become ``"True"`` or ``"False"``.

        Raises
        ------
        ValueError
            If ``value`` cannot be read as ``value_type`` by ``ConfigParser`` (a Python
            boolean is refused for a numerical option).
        """
        text = str(value)
        if value_type == "bool":
            is_valid = text.lower() in ConfigParser.BOOLEAN_STATES
        else:
            # getint and getfloat convert the string with int() and float()
            converter = int if value_type == "int" else float
            try:
                converter(text)
                is_valid = not isinstance(value, bool)
            except ValueError:
                is_valid = False

        if not is_valid:
            raise ValueError(f"fit_info[{name!r}] = {value!r} is not a valid {value_type}.")
        return text

    def _check_marginalized_params(self, marginalized_params, sample_params, config):
        """Validate the parameters of the ``[analytic marginalization]`` section.

        Vega marginalizes analytically only ``desi_inst_sys_amp``, ``qso_rad_strength``,
        the metal biases (``bias_<metal>`` or ``bias_eta_<metal>``) and the metal x metal
        amplitudes ``amp_<metal1>_<metal2>``. Vega sets missing values of marginalized
        parameters to zero, but it still needs a template for each of them. BuildConfig
        writes each of these parameters to ``[parameters]`` exactly when the model option
        that creates its template is on (``desi-instrumental-systematics``,
        ``radiation_effects``, ``metals`` with ``use_metal_bias_eta`` selecting
        ``bias_eta_<metal>``, and ``metal-auto-amplitudes``), so a supported name that is
        absent from the written ``[parameters]`` has no template in this configuration.

        Parameters
        ----------
        marginalized_params : list of str
            Parameters to marginalize analytically.
        sample_params : list or dict
            Sampled parameters (``fit_info["sample_params"]``).
        config : ConfigParser
            Main config, with the ``[parameters]`` section already written.

        Raises
        ------
        ValueError
            If a parameter is listed twice, is also sampled, is not a parameter that
            Vega can marginalize analytically, or is not part of this configuration.
        """
        duplicates = sorted(
            {name for name in marginalized_params if marginalized_params.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"Parameters listed twice for analytic marginalization: {duplicates}")

        sampled = sorted(set(marginalized_params) & set(sample_params))
        if sampled:
            raise ValueError(
                f"Parameters both sampled and analytically marginalized: {sampled}. Vega "
                "refuses this; remove them from sample_params."
            )

        metals = self.options["metals"] or []
        metal_bias_names = {f"bias_{metal}" for metal in metals}
        metal_bias_names |= {f"bias_eta_{metal}" for metal in metals}

        def is_supported(name):
            if name in ("desi_inst_sys_amp", "qso_rad_strength"):
                return True
            return name.startswith("amp_") or name in metal_bias_names

        unsupported = [name for name in marginalized_params if not is_supported(name)]
        if unsupported:
            raise ValueError(
                f"Vega cannot marginalize {unsupported} analytically. Supported parameters: "
                "desi_inst_sys_amp, qso_rad_strength, bias_<metal> or bias_eta_<metal> of the "
                "configured metals, and amp_<metal1>_<metal2>."
            )

        missing = [name for name in marginalized_params if name not in config["parameters"]]
        if missing:
            raise ValueError(
                f"The analytically marginalized parameters {missing} are not part of this "
                "configuration: they have no template. Check the model options "
                "(desi-instrumental-systematics, radiation_effects, metals, "
                "use_metal_bias_eta, metal-auto-amplitudes and use_metal_autos)."
            )

    def _metals_of_amplitude(self, amplitude_name):
        """Metals of a metal x metal amplitude parameter.

        Parameters
        ----------
        amplitude_name : str
            Amplitude name ``amp_<metal1>_<metal2>``, metal names sorted.

        Returns
        -------
        tuple of str
            The two configured metals whose amplitude this is.
        """
        # Matched against the configured metals rather than split on "_", so that the
        # metal names themselves are unconstrained.
        for metal1 in self.options["metals"]:
            for metal2 in self.options["metals"]:
                if Metals.amplitude_name(metal1, metal2) == amplitude_name:
                    return metal1, metal2
        raise ValueError(f"{amplitude_name} is not an amplitude of the configured metals.")

    @property
    def parameters(self):
        """Parameters property

        Returns
        -------
        dict
            Parameters
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """Setter for parameters property

        Parameters
        ----------
        parameters : dict
            Parameters dictionary
        """
        if self._params_template is None:
            # Read template
            config = ConfigParser()
            config.optionxform = lambda option: option
            template_path = find_file("vega/templates/parameters.ini")
            config.read(template_path)
            self._params_template = config["parameters"]

        def get_par(name):
            if name in parameters:
                return parameters[name]
            elif name not in self._params_template:
                raise ValueError("Unknown parameter: {}, please pass a default value.".format(name))
            else:
                return self._params_template[name]

        new_params = {}

        # Scale parameters
        if self.options["scale_params"] == "ap_at":
            new_params["ap"] = get_par("ap")
            new_params["at"] = get_par("at")
        elif self.options["scale_params"] == "phi_alpha":
            new_params["phi"] = get_par("phi")
            new_params["alpha"] = get_par("alpha")
            if self.options["full_shape"]:
                new_params["phi_full"] = get_par("phi_full")
            if self.options["full_shape_alpha"]:
                new_params["alpha_full"] = get_par("alpha_full")
            if self.options["smooth_scaling"]:
                new_params["phi_smooth"] = get_par("phi_smooth")
                new_params["alpha_smooth"] = get_par("alpha_smooth")
        elif self.options["scale_params"] == "aiso_epsilon":
            new_params["aiso"] = get_par("aiso")
            new_params["epsilon"] = get_par("epsilon")
        else:
            raise ValueError("Unknown scale parameters: {}".format(self.options["scale_params"]))

        # Peak parameters
        if self.options["bao_broadening"]:
            new_params["sigmaNL_per"] = get_par("sigmaNL_per")
            new_params["sigmaNL_par"] = get_par("sigmaNL_par")
        else:
            new_params["sigmaNL_per"] = 0.0
            new_params["sigmaNL_par"] = 0.0
        new_params["bao_amp"] = get_par("bao_amp")

        def add_bias_beta(new_params, tracer, bias_beta_config, bias, bias_eta, beta, growth_rate):
            if bias_beta_config == "bias_beta":
                new_params[f"bias_{tracer}"] = bias
                new_params[f"beta_{tracer}"] = beta
            elif bias_beta_config == "bias_bias_eta":
                new_params[f"bias_{tracer}"] = bias
                new_params[f"bias_eta_{tracer}"] = bias_eta
                new_params["growth_rate"] = growth_rate
            elif bias_beta_config == "bias_eta_beta":
                new_params[f"beta_{tracer}"] = beta
                new_params[f"bias_eta_{tracer}"] = bias_eta
                new_params["growth_rate"] = growth_rate
            else:
                raise ValueError(
                    f"Option {bias_beta_config} not a valid bias_beta_config. "
                    'Choose from ["bias_beta", "bias_eta_beta", "bias_bias_eta"].'
                )

        # bias beta model
        for name in self.corr_names:
            bias_beta_config = self.fit_info["bias_beta_config"].get(name, "bias_beta")

            growth_rate = parameters.get("growth_rate", None)
            if growth_rate is None:
                growth_rate = self.get_growth_rate(self.zeff_in)

            if (name == "LYA") or (name == "LYB") or (name == "CIV"):
                bias = parameters.get(f"bias_{name}", self.get_lya_bias(self.zeff_in))
                bias_eta = parameters.get(f"bias_eta_{name}", None)
                beta = float(get_par(f"beta_{name}"))

                if bias_eta is None:
                    bias_eta = bias * beta / growth_rate
            elif (name == "QSO") or (name == "DLA") or (name == "SBLA"):
                bias = parameters.get(f"bias_{name}", self.get_qso_bias(self.zeff_in))
                beta = parameters.get(f"beta_{name}", None)
                bias_eta = 1

                if beta is None:
                    beta = growth_rate / bias
            else:
                raise ValueError(f"Tracer {name} not supported yet. Please open an issue")

            add_bias_beta(new_params, name, bias_beta_config, bias, bias_eta, beta, growth_rate)

            new_params[f"alpha_{name}"] = get_par(f"alpha_{name}")

        # Small scale non-linear model
        if self.options["small_scale_nl"]:
            new_params["dnl_arinyo_q1"] = get_par("dnl_arinyo_q1")
            new_params["dnl_arinyo_q2"] = get_par("dnl_arinyo_q2")
            new_params["dnl_arinyo_kv"] = get_par("dnl_arinyo_kv")
            new_params["dnl_arinyo_av"] = get_par("dnl_arinyo_av")
            new_params["dnl_arinyo_bv"] = get_par("dnl_arinyo_bv")
            new_params["dnl_arinyo_kp"] = get_par("dnl_arinyo_kp")

        # HCDs
        if self.options["hcd_model"] is not None:
            new_params["bias_hcd"] = get_par("bias_hcd")
            new_params["beta_hcd"] = get_par("beta_hcd")
            new_params["L0_hcd"] = get_par("L0_hcd")

        # Delta_rp
        if "QSO" in self.corr_names:
            new_params["drp_QSO"] = get_par("drp_QSO")

        # Velocity dispersion parameters
        if self.options["velocity_dispersion"] is not None:
            if self.options["velocity_dispersion"] == "lorentz":
                for name in self.corr_names:
                    if name in ["QSO", "DLA", "SBLA"]:
                        key = f"sigma_velo_disp_lorentz_{name}"
                        new_params[key] = get_par(key)
            else:
                for name in self.corr_names:
                    if name in ["QSO", "DLA", "SBLA"]:
                        key = f"sigma_velo_disp_gauss_{name}"
                        new_params[key] = get_par(key)

        # QSO radiation effects
        if self.options["radiation_effects"]:
            new_params["qso_rad_strength"] = get_par("qso_rad_strength")
            new_params["qso_rad_asymmetry"] = get_par("qso_rad_asymmetry")
            new_params["qso_rad_lifetime"] = get_par("qso_rad_lifetime")
            new_params["qso_rad_decrease"] = get_par("qso_rad_decrease")

        # UV background parameters
        if self.options["UVB-fluctuations"]:
            new_params["bias_gamma"] = get_par("bias_gamma")
            new_params["bias_prim"] = get_par("bias_prim")
            new_params["lambda_uv"] = get_par("lambda_uv")
            new_params["uv_shotnoise_amp"] = get_par("uv_shotnoise_amp")

        if self.options["HeII-reionization"]:
            new_params["bias_gamma_e"] = get_par("bias_gamma_e")
            new_params["bias_prim"] = get_par("bias_prim")
            new_params["lambda_HeII"] = get_par("lambda_HeII")
            new_params["uv_shotnoise_amp"] = get_par("uv_shotnoise_amp")

        # Metals
        if self.options["metals"] is not None:
            for name in self.options["metals"]:
                if self.options["use_metal_bias_eta"]:
                    new_params["bias_eta_{}".format(name)] = get_par("bias_eta_{}".format(name))
                else:
                    new_params["bias_{}".format(name)] = get_par("bias_{}".format(name))

                new_params["beta_{}".format(name)] = get_par("beta_{}".format(name))
                new_params["alpha_{}".format(name)] = get_par("alpha_{}".format(name))

            if self.options["single-metal-beta"]:
                new_params["beta_metals"] = get_par("beta_metals")

            # Metal x metal amplitudes. The default bias product reproduces the model without
            # metal-auto-amplitudes, as the amplitude multiplies the metal x metal correlation
            # computed with unit biases.
            amplitude_names = getattr(self, "metal_auto_amplitude_names", [])
            marginalized_params = self.fit_info.get("analytic_marginalization", [])
            for amplitude_name in amplitude_names:
                if amplitude_name in parameters:
                    new_params[amplitude_name] = parameters[amplitude_name]
                    continue

                metal1, metal2 = self._metals_of_amplitude(amplitude_name)
                if not self.options["use_metal_bias_eta"]:
                    bias1 = float(new_params[f"bias_{metal1}"])
                    bias2 = float(new_params[f"bias_{metal2}"])
                    new_params[amplitude_name] = bias1 * bias2
                elif amplitude_name in marginalized_params:
                    # The value of a marginalized amplitude does not enter the likelihood;
                    # zero is the value Vega uses for its base model.
                    new_params[amplitude_name] = 0.0
                else:
                    # bias_<metal> = bias_eta_<metal> * f / beta_<metal> depends on which
                    # growth rate Vega uses at run time (fiducial or [parameters]), so no
                    # default is derived here.
                    raise ValueError(
                        f"metal-auto-amplitudes with use_metal_bias_eta: give the value of "
                        f"{amplitude_name} in the parameters dictionary (or marginalize it "
                        "analytically); its default bias product is not defined from "
                        "bias_eta_<metal>."
                    )

        # Full-shape smoothing
        if self.options["fullshape_smoothing"] is not None:
            if self.options["fullshape_smoothing"] == "exp":
                new_params["par_exp_smooth"] = get_par("par_exp_smooth")
                new_params["per_exp_smooth"] = get_par("per_exp_smooth")
                new_params["par_sigma_smooth"] = get_par("par_sigma_smooth")
                new_params["per_sigma_smooth"] = get_par("per_sigma_smooth")

            if self.options["fullshape_smoothing"] == "gauss_iso":
                new_params["par_sigma_smooth"] = get_par("par_sigma_smooth")

            if self.options["fullshape_smoothing"] == "gauss":
                if "par_sigma_smooth" in parameters:
                    new_params["par_sigma_smooth"] = get_par("par_sigma_smooth")
                    new_params["per_sigma_smooth"] = get_par("per_sigma_smooth")
                if "par_sigma_smooth_QSO" in parameters:
                    new_params["par_sigma_smooth_QSO"] = get_par("par_sigma_smooth_QSO")
                    new_params["per_sigma_smooth_QSO"] = get_par("per_sigma_smooth_QSO")
                if "par_sigma_smooth_LYA" in parameters:
                    new_params["par_sigma_smooth_LYA"] = get_par("par_sigma_smooth_LYA")
                    new_params["per_sigma_smooth_LYA"] = get_par("per_sigma_smooth_LYA")
                if "par_sigma_smooth_metals" in parameters:
                    new_params["par_sigma_smooth_metals"] = get_par("par_sigma_smooth_metals")
                    new_params["per_sigma_smooth_metals"] = get_par("per_sigma_smooth_metals")

        if self.options["mock-los-smoothing"] == "amplitude":
            new_params["los_smooth_amp"] = get_par("los_smooth_amp")

        # DESI instrumental systematics amplitude
        if self.options["desi-instrumental-systematics"]:
            new_params["desi_inst_sys_amp"] = get_par("desi_inst_sys_amp")

        # Check for broadband parameters
        for name, value in parameters.items():
            if "BB" in name and name not in new_params:
                new_params[name] = value

        self._parameters = new_params

    @staticmethod
    def get_lya_bias(z):
        """Compute default Lya bias at redshift z

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Default Lya bias
        """
        return -0.1167 * ((1 + z) / (1 + 2.334)) ** 2.9

    @staticmethod
    def get_qso_bias(z):
        """Compute default QSO bias at redshift z

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Default QSO bias
        """
        return 3.91 * ((1 + z) / (1 + 2.39)) ** 1.7133

    @staticmethod
    def get_growth_rate(z, Omega_m=0.3153):
        """Compute default growth rate at redshift z

        Parameters
        ----------
        z : float
            Redshift
        Omega_m : float, optional
            Matter fraction, by default 0.3153

        Returns
        -------
        float
            Default growth rate
        """
        Omega_m_z = Omega_m * ((1 + z) ** 3) / (Omega_m * ((1 + z) ** 3) + 1 - Omega_m)
        Omega_lambda_z = 1 - Omega_m_z
        growth_rate = (Omega_m_z**0.6) + (Omega_lambda_z / 70.0) * (1 + Omega_m_z / 2.0)
        return growth_rate
