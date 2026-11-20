"""Tests of the config generation by ``BuildConfig``.

The configs are written to a temporary directory and read back as text; no data files are
opened (the effective redshift is given explicitly).
"""

from configparser import ConfigParser
from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

from vega.build_config import BuildConfig
from vega.data import Data

# Metals of the DESI DR2 Lya analyses with metal x metal amplitudes
DR2_METALS = [
    "FeII(1145)",
    "SiII(1190)",
    "SiII(1193)",
    "SiIII(1207)",
    "NV(1239)",
    "SiII(1260)",
    "SiII(1304)",
    "CIV(eff)",
]

# Distinct metal biases, so that the default amplitudes identify the bias products
DR2_METAL_PARAMETERS = {}
for index, metal in enumerate(DR2_METALS):
    DR2_METAL_PARAMETERS[f"bias_{metal}"] = -1e-3 * (index + 1)
    DR2_METAL_PARAMETERS[f"beta_{metal}"] = 0.5
    DR2_METAL_PARAMETERS[f"alpha_{metal}"] = 1.0

# Main config and correlation configs written by BuildConfig before the analytic
# marginalization options were added (commit cd6cb40), without the header comments.
REFERENCE_LYAXLYA = """
[data]
name = lyaxlya
tracer1 = LYA
tracer2 = LYA
tracer1-type = continuous
tracer2-type = continuous
filename = lyaxlya.fits

[cuts]
rp-min = -300
rp-max = +300.
rt-min = 0
rt-max = 300.
r-min = 10
r-max = 180
mu-min = -1
mu-max = 1

[model]
z evol LYA = bias_vs_z_std
model binning = True
use_metal_autos = True
marginalize-all-rmin-cuts = False

[metals]
filename = lyaxlya-metals.fits
z evol = bias_vs_z_std
in tracer1 = SiII(1190) CIV(eff)
in tracer2 = SiII(1190) CIV(eff)

[broadband]
bb1 = add pre rp,rt 0:2:1 0:2:1
"""

REFERENCE_LYAXQSO = """
[data]
name = lyaxqso
tracer1 = LYA
tracer2 = QSO
tracer1-type = continuous
tracer2-type = discrete
filename = lyaxqso.fits

[cuts]
rp-min = -300
rp-max = +300.
rt-min = 0
rt-max = 300.
r-min = 10
r-max = 180
mu-min = -1
mu-max = 1

[model]
z evol LYA = bias_vs_z_std
z evol QSO = bias_vs_z_std
model binning = True
marginalize-all-rmin-cuts = False

[metals]
filename = lyaxqso-metals.fits
z evol = bias_vs_z_std
in tracer1 = SiII(1190) CIV(eff)
"""

REFERENCE_MAIN = """
[data sets]
zeff = 2.33
ini files = {root}/lyaxlya.ini {root}/lyaxqso.ini

[cosmo-fit type]
cosmo fit func = ap_at
full-shape = False
full-shape-alpha = False
smooth-scaling = False

[fiducial]
filename = PlanckDR16/PlanckDR16.fits

[output]
filename = {root}/output_fitter/lyaxlya_lyaxqso

[sample]
ap = True
at = True
bias_LYA = True

[priors]
bias_LYA = gaussian -0.1 0.05

[parameters]
ap = 1.0
at = 1.0
sigmaNL_per = 0.0
sigmaNL_par = 0.0
bao_amp = 1.
bias_LYA = -0.11629442782749021
beta_LYA = 1.67
alpha_LYA = 2.9
bias_QSO = 3.792183285869361
beta_QSO = 0.25512018199838027
alpha_QSO = 1.44
drp_QSO = 0.0
bias_SiII(1190) = -0.0052
beta_SiII(1190) = 0.5
alpha_SiII(1190) = 1.
bias_CIV(eff) = -0.01
beta_CIV(eff) = 0.5
alpha_CIV(eff) = 0.
BB-lyaxlya-0-broadband = 0.0

[control]
run_sampler = False
"""


def correlation_info(names, **extra_options):
    """Correlation dictionaries for ``BuildConfig.build`` with placeholder data paths.

    Parameters
    ----------
    names : list of str
        Correlation names, e.g. ``["lyaxlya", "lyaxqso"]``.
    **extra_options
        Entries added to every correlation dictionary.

    Returns
    -------
    dict
        ``{name: correlation dictionary}``.
    """
    return {
        name: {"corr_path": f"{name}.fits", "metal_path": f"{name}-metals.fits"} | extra_options
        for name in names
    }


def default_fit_info(**extra_info):
    """Minimal ``fit_info`` with a fixed effective redshift, so that no data are read.

    Parameters
    ----------
    **extra_info
        Entries added to or replacing the defaults.

    Returns
    -------
    dict
        Fit information for ``BuildConfig.build``.
    """
    fit_info = {
        "zeff": 2.33,
        "sample_params": ["ap", "at", "bias_LYA"],
        "bias_beta_config": {},
        "priors": {"bias_LYA": "gaussian -0.1 0.05"},
    }
    return fit_info | extra_info


def build_configs(out_path, options, correlations, fit_type, fit_info, parameters=None):
    """Build the configs and read them back.

    Parameters
    ----------
    out_path : pathlib.Path
        Output directory.
    options : dict
        ``BuildConfig`` options.
    correlations : dict
        Correlation dictionaries.
    fit_type : str
        Correlations joined by underscores.
    fit_info : dict
        Fit information.
    parameters : dict, optional
        Parameter values, by default None.

    Returns
    -------
    dict
        ``{file name: ConfigParser}`` of the written configs, keyed without extension.
    BuildConfig
        The builder.
    """
    builder = BuildConfig(options)
    builder.build(correlations, fit_type, fit_info, out_path, parameters=parameters)

    configs = {}
    for path in sorted(out_path.glob("*.ini")):
        config = ConfigParser()
        config.optionxform = lambda option: option
        config.read(path)
        configs[path.stem] = config
    return configs, builder


def strip_header(text, root):
    """Config text without the date and git hash comments, with the root as placeholder.

    Parameters
    ----------
    text : str
        Text of a written config.
    root : pathlib.Path
        Output directory, replaced by ``{root}``.

    Returns
    -------
    str
        Normalized text.
    """
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    return "\n".join(lines).replace(str(root), "{root}").strip()


def data_amplitude_names(metals, same_tracer, use_metal_autos):
    """Amplitude names of the metal x metal pairs selected by ``vega.data.Data``.

    Parameters
    ----------
    metals : list of str
        Metals in both tracers.
    same_tracer : bool
        Whether the two main tracer dictionaries are equal.
    use_metal_autos : bool
        Whether SiII x SiII correlations are used.

    Returns
    -------
    set of str
        Names ``amp_<m1>_<m2>`` of the metal x metal correlations.
    """
    tracer1 = {"name": "LYA", "type": "continuous"}
    tracer2 = dict(tracer1) if same_tracer else {"name": "LYA", "type": "continuous", "w": 1}
    data_stub = SimpleNamespace(tracer1=tracer1, tracer2=tracer2, use_metal_autos=use_metal_autos)
    data_stub._use_correlation = lambda name1, name2: Data._use_correlation(data_stub, name1, name2)

    metal_config = {"in tracer1": None, "in tracer2": None}
    pairs = Data._init_metal_correlations(data_stub, metal_config, metals, metals)
    return {
        BuildConfig.get_metal_auto_amplitude_names([name1], [name2], False)[0]
        for name1, name2 in pairs
        if name1 in metals and name2 in metals
    }


@pytest.mark.parametrize("same_tracer", [True, False])
@pytest.mark.parametrize("use_metal_autos", [True, False])
def test_amplitude_names_match_data_pair_selection(same_tracer, use_metal_autos):
    """The amplitudes are those of the metal x metal correlations built by Data."""
    expected = data_amplitude_names(DR2_METALS, same_tracer, use_metal_autos)
    names = BuildConfig.get_metal_auto_amplitude_names(
        DR2_METALS, DR2_METALS, same_tracer, use_metal_autos
    )

    assert len(names) == len(set(names))
    assert set(names) == expected


def test_metal_auto_amplitudes_auto_and_cross(tmp_path):
    """Lya auto, Lya x Lyb and QSO crosses: 28 pairs of 7 metals plus CIV(eff) x CIV(eff)."""
    correlations = correlation_info(
        ["lyaxlya", "lyaxlyb", "lyaxqso", "lybxqso"], fast_metals="True"
    )
    options = {"metals": DR2_METALS, "metal-auto-amplitudes": True, "use_metal_autos": True}
    configs, _ = build_configs(
        tmp_path,
        options,
        correlations,
        "lyaxlya_lyaxlyb_lyaxqso_lybxqso",
        default_fit_info(),
        parameters=dict(DR2_METAL_PARAMETERS),
    )
    parameters = configs["main"]["parameters"]

    non_civ_metals = DR2_METALS[:-1]
    expected_names = {
        f"amp_{min(pair)}_{max(pair)}" for pair in combinations_with_replacement(non_civ_metals, 2)
    }
    expected_names.add("amp_CIV(eff)_CIV(eff)")
    amplitude_names = {name for name in parameters if name.startswith("amp_")}
    assert len(expected_names) == 29
    assert amplitude_names == expected_names

    # Default values: product of the metal biases, which leaves the model unchanged
    for name in amplitude_names:
        metal1, metal2 = next(
            (m1, m2)
            for m1 in DR2_METALS
            for m2 in DR2_METALS
            if BuildConfig.get_metal_auto_amplitude_names([m1], [m2], False) == [name]
        )
        bias_product = float(parameters[f"bias_{metal1}"]) * float(parameters[f"bias_{metal2}"])
        assert float(parameters[name]) == pytest.approx(bias_product, rel=1e-12)

    for name in ["lyaxlya", "lyaxlyb", "lyaxqso", "lybxqso"]:
        assert configs[name]["model"]["metal-auto-amplitudes"] == "True"


def test_metal_auto_amplitudes_without_siii_autos(tmp_path):
    """Without use_metal_autos no SiII x SiII amplitude is written."""
    options = {"metals": DR2_METALS, "metal-auto-amplitudes": True, "use_metal_autos": False}
    configs, _ = build_configs(
        tmp_path,
        options,
        correlation_info(["lyaxlya"]),
        "lyaxlya",
        default_fit_info(),
        parameters=dict(DR2_METAL_PARAMETERS),
    )
    amplitude_names = [name for name in configs["main"]["parameters"] if name.startswith("amp_")]

    # 7 non-CIV metals give 28 pairs. Data._use_correlation drops every pair of names that
    # contain "SiII", which includes SiIII(1207): 15 pairs of the 5 Si lines are dropped.
    assert len(amplitude_names) == 28 - 15 + 1
    assert "amp_SiII(1190)_SiII(1193)" not in amplitude_names
    assert "amp_SiII(1190)_SiIII(1207)" not in amplitude_names
    assert "amp_FeII(1145)_SiII(1190)" in amplitude_names
    assert "amp_SiII(1260)_SiII(1260)" not in amplitude_names


def test_metal_auto_amplitudes_qso_cross_only(tmp_path):
    """A correlation with only one continuous tracer has no metal x metal amplitudes."""
    options = {"metals": DR2_METALS, "metal-auto-amplitudes": True}
    configs, _ = build_configs(
        tmp_path,
        options,
        correlation_info(["lyaxqso"]),
        "lyaxqso",
        default_fit_info(),
        parameters=dict(DR2_METAL_PARAMETERS),
    )

    assert not any(name.startswith("amp_") for name in configs["main"]["parameters"])
    assert configs["lyaxqso"]["model"]["metal-auto-amplitudes"] == "True"


def test_metal_auto_amplitudes_given_values_and_sampling(tmp_path):
    """Given values take precedence, unused amplitudes are dropped, and sampling works."""
    options = {"metals": ["SiII(1190)", "SiII(1193)", "CIV(eff)"], "metal-auto-amplitudes": True}
    given = {
        "amp_SiII(1190)_SiII(1193)": 2e-4,
        "amp_CIV(eff)_CIV(eff)": 3e-4,
        # CIV(eff) has no correlation with the other metals: not written
        "amp_CIV(eff)_SiII(1190)": 1.0,
    }
    sample_params = {
        "ap": "0.5 1.5 1.0 0.1",
        "bias_LYA": "-1 0 -0.1 0.01",
        "amp_SiII(1190)_SiII(1193)": "-1 1 0.0002 0.0001",
    }
    fit_info = default_fit_info(sample_params=sample_params)
    configs, _ = build_configs(
        tmp_path, options, correlation_info(["lyaxlya"]), "lyaxlya", fit_info, parameters=given
    )
    parameters = configs["main"]["parameters"]

    assert float(parameters["amp_SiII(1190)_SiII(1193)"]) == 2e-4
    assert float(parameters["amp_CIV(eff)_CIV(eff)"]) == 3e-4
    assert "amp_CIV(eff)_SiII(1190)" not in parameters
    bias_product = float(parameters["bias_SiII(1190)"]) ** 2
    assert float(parameters["amp_SiII(1190)_SiII(1190)"]) == pytest.approx(bias_product)
    assert configs["main"]["sample"]["amp_SiII(1190)_SiII(1193)"] == "-1 1 0.0002 0.0001"


def test_metal_auto_amplitudes_with_bias_eta(tmp_path):
    """With metal bias_eta, amplitudes need a value unless they are marginalized."""
    options = {
        "metals": ["SiII(1190)", "CIV(eff)"],
        "metal-auto-amplitudes": True,
        "use_metal_bias_eta": True,
    }
    correlations = correlation_info(["lyaxlya"])
    with pytest.raises(ValueError, match="use_metal_bias_eta"):
        build_configs(tmp_path, options, correlations, "lyaxlya", default_fit_info())

    fit_info = default_fit_info(analytic_marginalization=["amp_SiII(1190)_SiII(1190)"])
    given = {"amp_CIV(eff)_CIV(eff)": 1e-4}
    builder = BuildConfig(options, overwrite=True)
    builder.build(correlations, "lyaxlya", fit_info, tmp_path, parameters=given)
    config = ConfigParser()
    config.optionxform = lambda option: option
    config.read(tmp_path / "main.ini")

    assert float(config["parameters"]["amp_SiII(1190)_SiII(1190)"]) == 0.0
    assert float(config["parameters"]["amp_CIV(eff)_CIV(eff)"]) == 1e-4


def test_analytic_marginalization_section(tmp_path):
    """Listed parameters are written to the section and may have Gaussian priors."""
    options = {
        "metals": ["SiII(1190)", "CIV(eff)"],
        "metal-auto-amplitudes": True,
        "desi-instrumental-systematics": True,
    }
    marginalized = ["desi_inst_sys_amp", "bias_SiII(1190)", "amp_CIV(eff)_CIV(eff)"]
    priors = {"bias_LYA": "gaussian -0.1 0.05", "desi_inst_sys_amp": "gaussian 3e-4 1e-4"}
    fit_info = default_fit_info(analytic_marginalization=marginalized, priors=priors)
    configs, _ = build_configs(
        tmp_path, options, correlation_info(["lyaxlya"]), "lyaxlya", fit_info
    )
    main = configs["main"]

    assert dict(main["analytic marginalization"]) == dict.fromkeys(marginalized, "True")
    assert main["priors"]["desi_inst_sys_amp"] == "gaussian 3e-4 1e-4"
    assert main["analytic marginalization"].getboolean("desi_inst_sys_amp")


@pytest.mark.parametrize(
    "marginalized, options, match",
    [
        # Sampled and marginalized at the same time
        (["bias_LYA"], {}, "both sampled and analytically marginalized"),
        # Not a linear parameter that Vega can marginalize
        (["beta_SiII(1190)"], {}, "cannot marginalize"),
        # No template: the option creating it is off
        (["desi_inst_sys_amp"], {}, "not part of this configuration"),
        (["amp_SiII(1190)_SiII(1190)"], {}, "not part of this configuration"),
        (["bias_eta_SiII(1190)"], {}, "not part of this configuration"),
        # CIV(eff) x SiII is never built
        (["amp_CIV(eff)_SiII(1190)"], {"metal-auto-amplitudes": True}, "not part of this"),
        (["qso_rad_strength", "qso_rad_strength"], {}, "listed twice"),
    ],
)
def test_analytic_marginalization_validation(tmp_path, marginalized, options, match):
    """Invalid [analytic marginalization] lists are refused with a clear error."""
    options = {"metals": ["SiII(1190)", "CIV(eff)"]} | options
    fit_info = default_fit_info(analytic_marginalization=marginalized)
    with pytest.raises(ValueError, match=match):
        build_configs(tmp_path, options, correlation_info(["lyaxlya"]), "lyaxlya", fit_info)


def test_analytic_marginalization_prior_of_unlisted_parameter(tmp_path):
    """A prior on a parameter that is neither sampled nor marginalized is still refused."""
    fit_info = default_fit_info(priors={"bias_SiII(1190)": "gaussian 0 1"})
    with pytest.raises(AssertionError, match="Cannot add prior"):
        build_configs(
            tmp_path,
            {"metals": ["SiII(1190)"]},
            correlation_info(["lyaxlya"]),
            "lyaxlya",
            fit_info,
        )


def test_marginalized_broadband_options(tmp_path):
    """Per-correlation broadband marginalization options go to [model]."""
    broadband = {"bb1": "add pre rp,rt 0:2:1 0:2:1"}
    correlations = {
        "lyaxlya": {
            "corr_path": "lyaxlya.fits",
            "broadband": broadband,
            "marginalize-broadbands": True,
            "broadband-prior-sigma": 1e-4,
        },
        "lyaxqso": {
            "corr_path": "lyaxqso.fits",
            "broadband": broadband,
            "marginalize-broadbands": True,
        },
        "qsoxqso": {"corr_path": "qsoxqso.fits", "broadband": broadband},
    }
    configs, _ = build_configs(
        tmp_path, {}, correlations, "lyaxlya_lyaxqso_qsoxqso", default_fit_info()
    )

    assert configs["lyaxlya"]["model"]["marginalize-broadbands"] == "True"
    assert configs["lyaxlya"]["model"].getfloat("broadband-prior-sigma") == 1e-4
    assert configs["lyaxqso"]["model"]["marginalize-broadbands"] == "True"
    assert "broadband-prior-sigma" not in configs["lyaxqso"]["model"]
    assert "marginalize-broadbands" not in configs["qsoxqso"]["model"]


@pytest.mark.parametrize(
    "corr_options, match",
    [
        ({"marginalize-broadbands": True}, "no broadband"),
        ({"broadband": {"bb1": "add pre rp,rt 0:2:1 0:2:1"}, "broadband-prior-sigma": 1.0}, "Vega"),
        (
            {
                "broadband": {"bb1": "add pre rp,rt 0:2:1 0:2:1"},
                "marginalize-broadbands": True,
                "broadband-prior-sigma": 0.0,
            },
            "positive",
        ),
    ],
)
def test_marginalized_broadband_validation(tmp_path, corr_options, match):
    """Inconsistent broadband marginalization options are refused."""
    correlations = {"lyaxlya": {"corr_path": "lyaxlya.fits"} | corr_options}
    with pytest.raises(ValueError, match=match):
        build_configs(tmp_path, {}, correlations, "lyaxlya", default_fit_info())


def test_configs_without_new_options_unchanged(tmp_path):
    """Without the new options the configs equal those written before they were added."""
    correlations = correlation_info(["lyaxlya", "lyaxqso"])
    correlations["lyaxlya"]["broadband"] = {"bb1": "add pre rp,rt 0:2:1 0:2:1"}
    options = {"metals": ["SiII(1190)", "CIV(eff)"], "metal-auto-amplitudes": False}
    builder = BuildConfig(options)
    builder.build(
        correlations,
        "lyaxlya_lyaxqso",
        default_fit_info(),
        tmp_path,
        parameters={"BB-lyaxlya-0-broadband": 0.0},
    )

    references = {
        "lyaxlya.ini": REFERENCE_LYAXLYA,
        "lyaxqso.ini": REFERENCE_LYAXQSO,
        "main.ini": REFERENCE_MAIN,
    }
    assert sorted(path.name for path in tmp_path.glob("*.ini")) == sorted(references)
    for file_name, reference in references.items():
        text = (tmp_path / file_name).read_text()
        assert strip_header(text, tmp_path) == reference.strip()


# Main config with a PolyChord run written by BuildConfig before the optional sampler
# settings and marginalize-in-fit were added (commit c6ec253), without the header comments.
REFERENCE_MAIN_POLYCHORD = """
[data sets]
zeff = 2.33
ini files = {root}/lyaxlya.ini

[cosmo-fit type]
cosmo fit func = ap_at
full-shape = False
full-shape-alpha = False
smooth-scaling = False

[fiducial]
filename = PlanckDR16/PlanckDR16.fits

[output]
filename = {root}/output_fitter/lyaxlya

[sample]
ap = True
at = True
bias_LYA = True

[priors]
bias_LYA = gaussian -0.1 0.05

[parameters]
ap = 1.0
at = 1.0
sigmaNL_per = 0.0
sigmaNL_par = 0.0
bao_amp = 1.
bias_LYA = -0.11629442782749021
beta_LYA = 1.67
alpha_LYA = 2.9
bias_SiII(1190) = -0.0052
beta_SiII(1190) = 0.5
alpha_SiII(1190) = 1.

[control]
run_sampler = True
sampler = Polychord
low_mem_mode = False

[Polychord]
path = {root}/output_sampler
name = lyaxlya
num_live = 75
num_repeats = 3
do_clustering = True
boost_posterior = 0
"""


def test_polychord_config_without_new_options_unchanged(tmp_path):
    """Without the optional settings the PolyChord main config is unchanged."""
    fit_info = default_fit_info(run_sampler=True, Polychord={})
    BuildConfig({"metals": ["SiII(1190)"]}).build(
        correlation_info(["lyaxlya"]), "lyaxlya", fit_info, tmp_path
    )

    text = (tmp_path / "main.ini").read_text()
    assert strip_header(text, tmp_path) == REFERENCE_MAIN_POLYCHORD.strip()


@pytest.mark.parametrize("run_sampler", [False, True])
@pytest.mark.parametrize("marginalize_in_fit", [True, False, "True"])
def test_marginalize_in_fit_written(tmp_path, run_sampler, marginalize_in_fit):
    """marginalize-in-fit goes to [control] as a boolean, with or without a sampler."""
    fit_info = default_fit_info(
        run_sampler=run_sampler, Polychord={}, marginalize_in_fit=marginalize_in_fit
    )
    configs, _ = build_configs(
        tmp_path, {"metals": ["SiII(1190)"]}, correlation_info(["lyaxlya"]), "lyaxlya", fit_info
    )
    control = configs["main"]["control"]

    assert control.getboolean("marginalize-in-fit") is (str(marginalize_in_fit) == "True")
    assert control.getboolean("run_sampler") is run_sampler


def test_optional_polychord_settings_written(tmp_path):
    """Given PolyChord settings are written in a form the sampler reads back."""
    polychord_info = {
        "num_live": "400",
        "precision": 0.01,
        "seed": 42,
        "resume": False,
        "write_dead": "False",
        "cluster_posteriors": True,
        "maximise": True,
    }
    fit_info = default_fit_info(run_sampler=True, Polychord=polychord_info)
    configs, _ = build_configs(
        tmp_path, {"metals": ["SiII(1190)"]}, correlation_info(["lyaxlya"]), "lyaxlya", fit_info
    )
    polychord = configs["main"]["Polychord"]

    # Read as in vega.samplers.polychord.Polychord.get_sampler_settings
    assert polychord.getint("num_live") == 400
    assert polychord.getfloat("precision") == 0.01
    assert polychord.getint("seed") == 42
    assert polychord.getboolean("resume") is False
    assert polychord.getboolean("write_dead") is False
    assert polychord.getboolean("cluster_posteriors") is True
    assert polychord.getboolean("maximise") is True

    # Defaults of the existing settings are unchanged
    assert polychord.getint("num_repeats") == 3
    assert polychord.getboolean("do_clustering") is True
    assert polychord.getfloat("boost_posterior") == 0.0


def test_optional_polychord_settings_only_when_given(tmp_path):
    """Settings not given are absent, so that the sampler defaults apply."""
    fit_info = default_fit_info(run_sampler=True, Polychord={"seed": 7})
    configs, _ = build_configs(
        tmp_path, {"metals": ["SiII(1190)"]}, correlation_info(["lyaxlya"]), "lyaxlya", fit_info
    )
    polychord = configs["main"]["Polychord"]

    assert polychord["seed"] == "7"
    for key in BuildConfig.optional_polychord_settings:
        if key != "seed":
            assert key not in polychord
    assert "marginalize-in-fit" not in configs["main"]["control"]


@pytest.mark.parametrize(
    "extra_info",
    [
        {"marginalize_in_fit": "yes please"},
        {"run_sampler": True, "Polychord": {"resume": "maybe"}},
        {"run_sampler": True, "Polychord": {"seed": 1.5}},
        {"run_sampler": True, "Polychord": {"seed": True}},
        {"run_sampler": True, "Polychord": {"precision": "small"}},
    ],
)
def test_invalid_control_and_polychord_values(tmp_path, extra_info):
    """Values that Vega cannot read as the expected type are refused."""
    fit_info = default_fit_info(**extra_info)
    with pytest.raises(ValueError, match="is not a valid"):
        build_configs(
            tmp_path,
            {"metals": ["SiII(1190)"]},
            correlation_info(["lyaxlya"]),
            "lyaxlya",
            fit_info,
        )
