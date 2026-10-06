"""Tests of the new-metal matrices, the exponent handling and the fast-metals guards.

The new-metal model (``new_metals = True``) builds the metal distortion matrices at
initialization from stacked delta weights. The tests use a small synthetic setup (a stacked
delta-attributes file with 120 pixels, the upstream test correlation file with the cosmology
added to its header) written to a temporary directory, so no data files are added to
``tests/data``. They cover:

- the identity between the estimator-convention matrices and an explicit stacked-pair sum
  computed independently in the test;
- the ``legacy`` convention against numbers generated with the unmodified code;
- the deprecated and the new layout of ``[metal-matrix]``;
- the shared growth helper;
- the refusal of varied matrix exponents and of fast metals combined with varied metal shape
  parameters, ``no-metal-decomp = False`` and ``metal-scaling = True``;
- the Monte Carlo input model when ``[mc parameters]`` changes a matrix exponent.
"""

import configparser
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from picca import constants as picca_constants
from scipy.integrate import quad

from vega import VegaInterface, interface_utils, redshift_weights
from vega.utils import find_file, growth_function, normalized_growth_factor

TESTS_DIR = Path(__file__).resolve().parent

NUM_PIXELS = 120
WAVELENGTH_RANGE = (3600.0, 5500.0)  # Angstrom, observed frame

# Correlation -> (upstream correlation INI, name of the QSO tracer or None)
CORRELATION_SOURCES = {
    "lyaxlya": ("lyalya_lyalya.ini", None),
    "lyaxqso": ("lyalya_qso.ini", "tracer2"),
}

# Name of the correlation inside the INI files
CORRELATION_NAMES = {"lyaxlya": "lyalya_lyalya", "lyaxqso": "lyalya_qso"}

# [metal-matrix] in the deprecated layout (as of the reference numbers of the legacy test):
# the amplitude exponents of the metals and the weighting exponents are mixed.
# alpha_CIV(eff) is 0 here but 1 in [parameters]: the legacy convention uses the former,
# the estimator convention the latter.
DEPRECATED_METAL_MATRIX_OPTIONS = {
    "rebin_factor": "3",
    "alpha_LYA": "2.9",
    "alpha_SiII(1260)": "1.",
    "alpha_SiIII(1207)": "1.",
    "alpha_SiII(1193)": "1.",
    "alpha_SiII(1190)": "1.",
    "alpha_CIV(eff)": "0.",
    "z_ref_objects": "2.25",
    "z_evol_objects": "1.44",
    "z_bins_objects": "1000",
}

# Same estimator weighting, new layout: only weighting exponents, no metal amplitude exponents
WEIGHT_EVOL_FOREST = 2.9
WEIGHT_EVOL_QSO = 1.44
Z_REF_OBJECTS = 2.25
Z_BINS_OBJECTS = 1000
REBIN_FACTOR = 3
NEW_METAL_MATRIX_OPTIONS = {
    "rebin_factor": str(REBIN_FACTOR),
    "weight_evol_LYA": str(WEIGHT_EVOL_FOREST),
    "z_ref_objects": str(Z_REF_OBJECTS),
    "z_bins_objects": str(Z_BINS_OBJECTS),
}

# Cosmology written to the header of the data-file copy (the new-metal model needs it to convert
# redshifts to comoving distances; the upstream test data files carry no OMEGAM)
DATA_FILE_COSMOLOGY = {"OMEGAM": 0.3147, "OMEGAK": 0.0, "OMEGAR": 0.0, "WL": -1.0}

# Options holding file paths that are made absolute in the config copies
PATH_OPTIONS = {
    "data": [
        "filename",
        "distortion-file",
        "covariance-file",
        "weights-tracer1",
        "weights-tracer2",
    ],
    "metals": ["filename"],
}

# Summaries of the metal vector of the legacy convention on the synthetic setups, generated with
# the unmodified (upstream) code: metals.compute(params, pk_full, "full") with
# params = vega._get_lcl_prms(None) and params["peak"] = False. ``indices`` are size // 7 * k for
# k = 1..5.
LEGACY_REFERENCE = {
    "lyaxlya": {
        "size": 2500,
        "sum": 0.0069052157147724115,
        "sum_squares": 2.048969925100543e-06,
        "indices": [357, 714, 1071, 1428, 1785],
        "values": [
            2.7665315633315998e-06,
            1.5934059922486843e-07,
            -2.209109374413379e-06,
            -1.210640472007301e-06,
            -6.3898431184681e-07,
        ],
    },
    "lyaxqso": {
        "size": 5000,
        "sum": -0.22126971196720674,
        "sum_squares": 0.0004043450444882479,
        "indices": [714, 1428, 2142, 2856, 3570],
        "values": [
            5.6701361225828236e-06,
            -2.7916262316075963e-06,
            -2.317070484028768e-07,
            2.3989720102906517e-05,
            1.6226900884838584e-06,
        ],
    },
}

# Full specification "min max value error" of a sampled parameter (the default values file has
# no entry for most of the parameters used here)
SAMPLE_SPEC = "-5. 5. {value} 0.1"


# ----------------------------------------------------------------------------------------------
# Helpers that write the configurations
# ----------------------------------------------------------------------------------------------


def read_config(path):
    """Read an INI file keeping the case of the option names and the raw values.

    Parameters
    ----------
    path : str or pathlib.Path
        INI file.

    Returns
    -------
    configparser.ConfigParser
        Parsed configuration.
    """
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    config.read(path)
    return config


def apply_edits(config, edits):
    """Set or remove options of a configuration, creating missing sections.

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration, modified in place.
    edits : dict or None
        ``{section: {option: value}}``. A value of None removes the option; any other value is
        written with ``str``. An empty dictionary creates an empty section.
    """
    for section, options in (edits or {}).items():
        if not config.has_section(section):
            config.add_section(section)

        for option, value in options.items():
            if value is None:
                config.remove_option(section, option)
            else:
                config.set(section, option, str(value))


def write_config_copy(main_ini, directory, main_edits=None, corr_edits=None, ini_files=None):
    """Write a modified copy of a main INI and of the correlation INIs it links.

    The file paths of ``PATH_OPTIONS`` are made absolute with ``vega.utils.find_file``, so the
    copy can be initialized from any directory. The ``[fiducial]`` file is left as it is.

    Parameters
    ----------
    main_ini : str or pathlib.Path
        Main INI to copy.
    directory : str or pathlib.Path
        Output directory (created if missing).
    main_edits : dict, optional
        Edits of the main INI, see ``apply_edits``.
    corr_edits : dict, optional
        Edits applied to every copied correlation INI, see ``apply_edits``.
    ini_files : list of str, optional
        File names (without directory) of the correlation INIs to keep; by default all.

    Returns
    -------
    pathlib.Path
        Absolute path of the copied main INI.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    main_config = read_config(main_ini)

    corr_paths = []
    for linked_path in main_config["data sets"]["ini files"].split():
        corr_config_path = Path(find_file(linked_path))
        if ini_files is not None and corr_config_path.name not in ini_files:
            continue

        corr_config = read_config(corr_config_path)
        for section, options in PATH_OPTIONS.items():
            for option in options:
                if corr_config.has_option(section, option):
                    value = corr_config[section][option]
                    corr_config[section][option] = str(Path(find_file(value)).resolve())

        apply_edits(corr_config, corr_edits)

        new_corr_path = directory / corr_config_path.name
        with open(new_corr_path, "w") as file:
            corr_config.write(file)
        corr_paths.append(str(new_corr_path))

    main_config["data sets"]["ini files"] = " ".join(corr_paths)
    apply_edits(main_config, main_edits)

    new_main_path = directory / "main.ini"
    with open(new_main_path, "w") as file:
        main_config.write(file)

    return new_main_path


def write_synthetic_setup(directory, correlation, layout="new"):
    """Write a synthetic new-metals setup for one correlation.

    Files written to ``directory``: ``delta_attributes.fits`` (HDU 1 with float64 columns
    ``LOGLAM``, ``NUM_PIXELS`` pixels uniform in log wavelength over ``WAVELENGTH_RANGE``, and
    ``WEIGHT = 1 + 0.5 sin(2 pi i / NUM_PIXELS)``), ``data.fits`` (copy of the upstream
    correlation file with ``DATA_FILE_COSMOLOGY`` added to its header), ``<correlation>.ini``
    and ``main.ini``. The correlation INI is the upstream one with ``new_metals = True``, the
    stacked weights in ``[data]`` (forest: the delta file; QSO: ``tests/data/qsoauto_zcat.fits``)
    and a ``[metal-matrix]`` section; the picca metal-matrix file is dropped. The main INI is the
    upstream ``full_configs/main.ini`` with this single correlation.

    Parameters
    ----------
    directory : str or pathlib.Path
        Output directory (created if missing).
    correlation : {"lyaxlya", "lyaxqso"}
        Correlation to set up.
    layout : {"new", "deprecated"}, optional
        Layout of ``[metal-matrix]``: ``"new"`` has ``weight_evol_<tracer>`` weighting exponents
        and no metal amplitude exponents; ``"deprecated"`` has the old mixed keys
        (``alpha_LYA``, ``alpha_<metal>``, ``z_evol_objects``), as in the legacy reference.
        By default "new".

    Returns
    -------
    pathlib.Path
        Absolute path of the main INI.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    # Stacked delta weights
    pixel_index = np.arange(NUM_PIXELS)
    loglam = np.linspace(*np.log10(WAVELENGTH_RANGE), NUM_PIXELS)
    weight = 1.0 + 0.5 * np.sin(2.0 * np.pi * pixel_index / NUM_PIXELS)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="LOGLAM", format="D", array=loglam.astype(np.float64)),
            fits.Column(name="WEIGHT", format="D", array=weight.astype(np.float64)),
        ]
    )
    delta_path = directory / "delta_attributes.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(delta_path, overwrite=True)

    # Correlation INI
    source_name, qso_tracer = CORRELATION_SOURCES[correlation]
    corr_config = read_config(TESTS_DIR / "full_configs" / source_name)

    # Copy of the data file with the cosmology in the header
    data_path = directory / "data.fits"
    with fits.open(TESTS_DIR / corr_config["data"]["filename"]) as hdul:
        for keyword, value in DATA_FILE_COSMOLOGY.items():
            hdul[1].header[keyword] = value
        hdul.writeto(data_path, overwrite=True)
    corr_config["data"]["filename"] = str(data_path)

    for option in ["distortion-file", "covariance-file"]:
        if option in corr_config["data"]:
            corr_config["data"][option] = str(TESTS_DIR / corr_config["data"][option])

    # Forest tracer 1 is always Lya; in lyaxqso tracer 2 is the QSO catalogue
    corr_config["data"]["weights-tracer1"] = str(delta_path)
    if qso_tracer is None:
        corr_config["data"]["weights-tracer2"] = str(delta_path)
    else:
        corr_config["data"]["weights-tracer2"] = str(TESTS_DIR / "data" / "qsoauto_zcat.fits")

    corr_config["model"]["new_metals"] = "True"
    corr_config["metals"].pop("filename", None)

    if layout == "deprecated":
        corr_config["metal-matrix"] = dict(DEPRECATED_METAL_MATRIX_OPTIONS)
    else:
        corr_config["metal-matrix"] = dict(NEW_METAL_MATRIX_OPTIONS)
        if qso_tracer is not None:
            corr_config["metal-matrix"]["weight_evol_QSO"] = str(WEIGHT_EVOL_QSO)

    corr_path = directory / f"{correlation}.ini"
    with open(corr_path, "w") as file:
        corr_config.write(file)

    # Main INI: upstream main with only this correlation
    main_config = read_config(TESTS_DIR / "full_configs" / "main.ini")
    main_config["data sets"]["ini files"] = str(corr_path)

    main_path = directory / "main.ini"
    with open(main_path, "w") as file:
        main_config.write(file)

    return main_path


def build_interface(main_ini):
    """Initialize ``VegaInterface`` and capture what it prints.

    Parameters
    ----------
    main_ini : str or pathlib.Path
        Main INI.

    Returns
    -------
    vega.VegaInterface
        Initialized interface.
    str
        Text printed to stdout during the initialization.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        vega = VegaInterface(str(main_ini))

    return vega, buffer.getvalue()


def sample_edit(**parameters):
    """Build the entries of a [sample]-type section for the given parameters.

    Parameters
    ----------
    **parameters
        ``{name: start value}``. Names with parentheses are passed by dictionary unpacking,
        e.g. ``sample_edit(**{"beta_SiII(1190)": 0.5})``.

    Returns
    -------
    dict
        ``{name: "min max value error"}`` (the full specification, see ``SAMPLE_SPEC``).
    """
    return {name: SAMPLE_SPEC.format(value=value) for name, value in parameters.items()}


class SyntheticSetups:
    """Synthetic new-metals setups written to a temporary directory, with memoized interfaces.

    Parameters
    ----------
    root : pathlib.Path
        Directory receiving the setups and the variants of their configurations.
    """

    def __init__(self, root):
        self.root = root
        self._main_inis = {}
        self._interfaces = {}

    def main_ini(self, correlation="lyaxlya", layout="new"):
        """Main INI of a synthetic setup, written on first use.

        Parameters
        ----------
        correlation : {"lyaxlya", "lyaxqso"}, optional
            Correlation, by default "lyaxlya".
        layout : {"new", "deprecated"}, optional
            Layout of ``[metal-matrix]``, by default "new".

        Returns
        -------
        pathlib.Path
            Main INI.
        """
        if (correlation, layout) not in self._main_inis:
            self._main_inis[(correlation, layout)] = write_synthetic_setup(
                self.root / f"{correlation}_{layout}", correlation, layout
            )

        return self._main_inis[(correlation, layout)]

    def interface(self, correlation="lyaxlya", layout="new", model_options=None):
        """Initialized interface of a synthetic setup, memoized (one per distinct call).

        Parameters
        ----------
        correlation : {"lyaxlya", "lyaxqso"}, optional
            Correlation, by default "lyaxlya".
        layout : {"new", "deprecated"}, optional
            Layout of ``[metal-matrix]``, by default "new".
        model_options : dict, optional
            Extra ``[model]`` options of the correlation INI, e.g.
            ``{"metal_matrix_convention": "legacy"}``.

        Returns
        -------
        vega.VegaInterface
            Initialized interface.
        str
            Text printed during the initialization.
        """
        key = (correlation, layout, tuple(sorted((model_options or {}).items())))
        if key not in self._interfaces:
            main_ini = self.main_ini(correlation, layout)
            if model_options:
                main_ini = write_config_copy(
                    main_ini,
                    self.root / f"variant_{len(self._interfaces)}",
                    corr_edits={"model": model_options},
                )
            self._interfaces[key] = build_interface(main_ini)

        return self._interfaces[key]


@pytest.fixture(scope="module")
def setups(tmp_path_factory):
    """Synthetic setups shared by the tests of this module.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest factory of temporary directories.

    Returns
    -------
    SyntheticSetups
        Setups; the interfaces are initialized on demand and kept for the module.
    """
    return SyntheticSetups(tmp_path_factory.mktemp("synthetic_metals"))


# ----------------------------------------------------------------------------------------------
# T1: estimator matrices against the explicit stacked-pair sum
# ----------------------------------------------------------------------------------------------


def explicit_stacked_pair_expectation(vega, correlation_name, corr_hash):
    """Expectation of the estimator for a constant undistorted correlation, pair by pair.

    For every pair of stacked-weight elements (forest pixels, or redshift bins of the QSO
    catalogue) it computes the weight ``W`` (product of the two estimator weights) and the
    amplitude evolution ``E`` (product of the two per-leg evolutions
    ``((1 + z) / (1 + z0))**alpha D(z) / D(z0)`` of the true absorbers), the assumed and the true
    line-of-sight separation, and returns, for each observed r_par bin A,
    ``sum_{pairs in A, true r_par in the grid} W E / sum_{pairs in A} W``.

    The calculation uses only the inputs (weights files, [parameters], fiducial cosmology and
    the correlation configuration), not the helpers of ``vega.metals``. The estimator weight
    of a forest pixel is ``w (1 + z_assumed)**(gamma - 1)`` with the redshift of the *assumed*
    absorber, as in the estimator; ``vega.metals`` evaluates the factor at the true redshift,
    which changes it by a constant per leg that cancels in the ratio.

    Parameters
    ----------
    vega : vega.VegaInterface
        Interface of a synthetic estimator-convention setup.
    correlation_name : str
        Name of the correlation.
    corr_hash : tuple of str
        Names of the true absorbers of tracers 1 and 2 of the metal correlation.

    Returns
    -------
    numpy.ndarray, shape (N_rp,)
        Expectation for each observed r_par bin (zero where the bin has no pair).
    numpy.ndarray, shape (N_rp,)
        Denominator ``sum_{pairs in A} W`` of each bin.
    """
    corr_item = vega.corr_items[correlation_name]
    coordinates = corr_item.model_coordinates
    z_ref = corr_item.z_eff
    omega_m = vega.fiducial["Omega_m"]
    omega_de = vega.fiducial["Omega_de"]

    zmin = corr_item.config["data"].getfloat("zmin", 0.0)
    zmax = corr_item.config["data"].getfloat("zmax", 10.0)

    true_redshifts = []
    assumed_redshifts = []
    estimator_weights = []
    amplitude_evolutions = []
    for tracer, true_absorber in zip((corr_item.tracer1, corr_item.tracer2), corr_hash):
        alpha_true = vega.params[f"alpha_{true_absorber}"]

        if tracer["type"] == "continuous":
            wave, stacked_weights = redshift_weights.get_forest_weights(
                tracer["weights-path"], rebin_factor=REBIN_FACTOR
            )
            true_z = wave / picca_constants.ABSORBER_IGM[true_absorber] - 1.0
            assumed_z = wave / picca_constants.ABSORBER_IGM[tracer["name"]] - 1.0
            weight = stacked_weights * (1.0 + assumed_z) ** (WEIGHT_EVOL_FOREST - 1.0)
        else:
            true_z, weight = redshift_weights.get_qso_weights(
                tracer["weights-path"],
                z_ref=Z_REF_OBJECTS,
                z_evol=WEIGHT_EVOL_QSO,
                z_bins=Z_BINS_OBJECTS,
            )
            assumed_z = true_z

        bias_evolution = ((1.0 + true_z) / (1.0 + z_ref)) ** alpha_true
        growth = normalized_growth_factor(true_z, z_ref, omega_m, omega_de)

        true_redshifts.append(true_z)
        assumed_redshifts.append(assumed_z)
        estimator_weights.append(weight)
        amplitude_evolutions.append(bias_evolution * growth)

    def separations(redshifts):
        """Line-of-sight separation of all pairs (absolute value unless a leg is discrete)."""
        distance_1 = corr_item.cosmo.get_r_comov(redshifts[0])
        distance_2 = corr_item.cosmo.get_r_comov(redshifts[1])
        rp_pairs = (distance_1[:, None] - distance_2[None, :]).ravel()
        if corr_item.tracer1["type"] == "continuous" and corr_item.tracer2["type"] == "continuous":
            rp_pairs = np.abs(rp_pairs)
        return rp_pairs

    true_rp_pairs = separations(true_redshifts)
    assumed_rp_pairs = separations(assumed_redshifts)

    pair_weight = (estimator_weights[0][:, None] * estimator_weights[1][None, :]).ravel()
    pair_evolution = (amplitude_evolutions[0][:, None] * amplitude_evolutions[1][None, :]).ravel()

    # Only pairs whose mean assumed redshift is inside the redshift range of the data count
    mean_assumed_z = ((assumed_redshifts[0][:, None] + assumed_redshifts[1][None, :]) / 2).ravel()
    pair_weight = pair_weight * ((mean_assumed_z >= zmin) & (mean_assumed_z <= zmax))

    rp_bin_edges = np.linspace(coordinates.rp_min, coordinates.rp_max, coordinates.rp_nbins + 1)
    true_in_grid = (true_rp_pairs >= rp_bin_edges[0]) & (true_rp_pairs <= rp_bin_edges[-1])

    numerator, _ = np.histogram(
        assumed_rp_pairs, bins=rp_bin_edges, weights=pair_weight * pair_evolution * true_in_grid
    )
    denominator, _ = np.histogram(assumed_rp_pairs, bins=rp_bin_edges, weights=pair_weight)

    expectation = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )

    return expectation, denominator


@pytest.mark.parametrize("correlation", ["lyaxlya", "lyaxqso"])
def test_estimator_matrix_identity(setups, correlation):
    """Row sums of the 2D and r_par-only estimator matrices equal the stacked-pair expectation."""
    vega_2d, _ = setups.interface(correlation)
    vega_rp, _ = setups.interface(correlation, model_options={"rp_only_metal_mats": "True"})

    name = CORRELATION_NAMES[correlation]
    metals_2d = vega_2d.models[name].metals
    metals_rp = vega_rp.models[name].metals
    num_rp = metals_2d.rp_nbins
    num_rt = metals_2d.rt_nbins

    assert metals_2d.metal_matrix_convention == "estimator"
    assert len(metals_2d.rp_metal_dmats) > 0

    for corr_hash in vega_2d.corr_items[name].metal_correlations:
        expectation, denominator = explicit_stacked_pair_expectation(vega_2d, name, corr_hash)
        populated = denominator > 0

        # The expectation is not trivially 1: the evolution and the grid edges matter
        assert populated.sum() > num_rp // 2
        assert np.ptp(expectation[populated]) > 1e-3

        # (a) r_par-only matrix
        row_sums_rp = metals_rp.rp_metal_dmats[corr_hash].sum(axis=1)
        np.testing.assert_allclose(row_sums_rp, expectation, rtol=1e-11, atol=1e-14)

        # (b) 2D matrix = kron(M_par, M_perp): the row sum of (rp, rt) is the product of the row
        # sums. Rows of M_perp sum to 1 where non-empty, and are empty at most in the last bins.
        row_sums_2d = np.asarray(metals_2d.rp_metal_dmats[corr_hash].sum(axis=1)).reshape(
            num_rp, num_rt
        )
        perp_row_sum = np.max(row_sums_2d[populated], axis=0) / np.max(expectation[populated])
        np.testing.assert_allclose(perp_row_sum[: num_rt // 2], 1.0, rtol=1e-11)
        assert np.all(
            np.isclose(perp_row_sum, 1.0, rtol=1e-11) | np.isclose(perp_row_sum, 0.0, atol=1e-12)
        )

        for rt_index in np.flatnonzero(np.isclose(perp_row_sum, 1.0, rtol=1e-11)):
            np.testing.assert_allclose(
                row_sums_2d[:, rt_index], expectation, rtol=1e-11, atol=1e-14
            )

        # Effective redshift of the undistorted metal correlation is z0 = zeff everywhere
        metal_z_grid = metals_2d.Xi_metal[corr_hash]._z
        assert np.all(metal_z_grid == vega_2d.corr_items[name].z_eff)


# ----------------------------------------------------------------------------------------------
# Legacy convention
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("correlation", ["lyaxlya", "lyaxqso"])
def test_legacy_convention_regression(setups, correlation):
    """The legacy convention reproduces the metal vector of the unmodified code."""
    vega, _ = setups.interface(
        correlation, layout="deprecated", model_options={"metal_matrix_convention": "legacy"}
    )
    name = CORRELATION_NAMES[correlation]

    # Parameters as passed by Model.compute when the metals are not decomposed
    local_params = vega._get_lcl_prms(None)
    local_params["peak"] = False
    xi_metals = vega.models[name].metals.compute(local_params, vega.fiducial["pk_full"], "full")

    reference = LEGACY_REFERENCE[correlation]
    assert xi_metals.size == reference["size"]
    assert np.sum(xi_metals) == pytest.approx(reference["sum"], rel=1e-10)
    assert np.sum(xi_metals**2) == pytest.approx(reference["sum_squares"], rel=1e-10)
    for index, value in zip(reference["indices"], reference["values"]):
        assert xi_metals[index] == pytest.approx(value, rel=1e-10)


def test_estimator_differs_from_legacy(setups):
    """The estimator matrices are not the legacy ones (the convention flag has an effect)."""
    vega_est, _ = setups.interface("lyaxlya", layout="deprecated")
    vega_leg, _ = setups.interface(
        "lyaxlya", layout="deprecated", model_options={"metal_matrix_convention": "legacy"}
    )

    metals_est = vega_est.models["lyalya_lyalya"].metals
    metals_leg = vega_leg.models["lyalya_lyalya"].metals
    corr_hash = ("SiII(1260)", "LYA")

    assert metals_est.metal_matrix_convention == "estimator"
    assert metals_leg.metal_matrix_convention == "legacy"
    assert metals_est.rp_metal_dmats[corr_hash].shape == metals_leg.rp_metal_dmats[corr_hash].shape
    assert (
        abs(metals_est.rp_metal_dmats[corr_hash] - metals_leg.rp_metal_dmats[corr_hash]).max() > 0
    )


# ----------------------------------------------------------------------------------------------
# [metal-matrix] layout and convention flag
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("correlation", ["lyaxlya", "lyaxqso"])
def test_deprecated_and_new_layout_give_identical_matrices(setups, correlation):
    """The deprecated [metal-matrix] keys map onto the new ones (estimator convention)."""
    name = CORRELATION_NAMES[correlation]
    vega_new, output_new = setups.interface(correlation, layout="new")
    vega_old, output_old = setups.interface(correlation, layout="deprecated")

    dmats_new = vega_new.models[name].metals.rp_metal_dmats
    dmats_old = vega_old.models[name].metals.rp_metal_dmats
    assert list(dmats_new) == list(dmats_old)

    for corr_hash, dmat_new in dmats_new.items():
        dmat_old = dmats_old[corr_hash]
        assert np.array_equal(dmat_new.indptr, dmat_old.indptr)
        assert np.array_equal(dmat_new.indices, dmat_old.indices)
        assert np.array_equal(dmat_new.data, dmat_old.data)

    # The deprecated layout prints warnings, the new one is silent
    assert "[metal-matrix]" not in output_new
    assert "alpha_LYA is deprecated" in output_old
    assert "[metal-matrix]" in output_old
    assert "ignored in the estimator convention" in output_old
    assert "alpha_SiII(1260)" in output_old
    if correlation == "lyaxlya":
        # alpha_CIV(eff) is 0 in [metal-matrix] but 1 in [parameters]: the latter is used
        assert "alpha_CIV(eff) ([metal-matrix]: 0.0, [parameters]: 1.0)" in output_old
    if correlation == "lyaxqso":
        assert "z_evol_objects is deprecated" in output_old


@pytest.mark.parametrize(
    "correlation, conflicting_edit, message",
    [
        ("lyaxlya", {"weight_evol_LYA": "2.5"}, "weight_evol_LYA"),
        ("lyaxqso", {"weight_evol_QSO": "1.0"}, "weight_evol_QSO"),
    ],
)
def test_conflicting_old_and_new_weighting_keys_raise(
    setups, tmp_path, correlation, conflicting_edit, message
):
    """A new weighting key that differs from the deprecated one is an error."""
    # Deprecated layout (alpha_LYA = 2.9, z_evol_objects = 1.44) plus a different new key
    main_ini = write_config_copy(
        setups.main_ini(correlation, "deprecated"),
        tmp_path,
        corr_edits={"metal-matrix": conflicting_edit},
    )

    with pytest.raises(ValueError, match=message):
        build_interface(main_ini)


def test_invalid_metal_matrix_convention_raises(setups, tmp_path):
    """An unknown metal_matrix_convention is refused."""
    main_ini = write_config_copy(
        setups.main_ini(),
        tmp_path,
        corr_edits={"model": {"metal_matrix_convention": "picca"}},
    )

    with pytest.raises(ValueError, match="metal_matrix_convention"):
        build_interface(main_ini)


# ----------------------------------------------------------------------------------------------
# Growth helper
# ----------------------------------------------------------------------------------------------

OMEGA_M = 0.3153
OMEGA_DE = 1.0 - OMEGA_M


def test_normalized_growth_factor_einstein_de_sitter():
    """Without dark energy, D(z) / D(z_ref) = (1 + z_ref) / (1 + z), whatever Omega_m is."""
    z_grid = np.linspace(1.5, 4.0, 11)

    ratio = normalized_growth_factor(z_grid, 2.3, OMEGA_M, None)

    np.testing.assert_allclose(ratio, (1 + 2.3) / (1 + z_grid), rtol=1e-14)
    np.testing.assert_array_equal(ratio, normalized_growth_factor(z_grid, 2.3, 0.9, None))
    assert normalized_growth_factor(2.3, 2.3, OMEGA_M, None) == 1.0


def test_normalized_growth_factor_lcdm():
    """With dark energy it is the ratio of growth_function values and the flat LCDM integral."""
    z_grid = np.linspace(1.5, 4.0, 11)
    z_ref = 2.33

    ratio = normalized_growth_factor(z_grid, z_ref, OMEGA_M, OMEGA_DE)

    expected = growth_function(z_grid, OMEGA_M, OMEGA_DE) / growth_function(
        z_ref, OMEGA_M, OMEGA_DE
    )
    np.testing.assert_array_equal(ratio, expected)
    assert normalized_growth_factor(z_ref, z_ref, OMEGA_M, OMEGA_DE) == pytest.approx(1.0)

    # Independent check with the flat-LCDM solution D(a) ~ E(a) int_0^a da' / (a' E(a'))^3
    def unnormalized_growth(z):
        scale_factor = 1 / (1 + z)

        def hubble(a):
            return np.sqrt(OMEGA_M / a**3 + OMEGA_DE)

        integral = quad(lambda a: 1 / (a * hubble(a)) ** 3, 0, scale_factor)[0]
        return hubble(scale_factor) * integral

    direct = np.array([unnormalized_growth(z) / unnormalized_growth(z_ref) for z in z_grid])
    np.testing.assert_allclose(ratio, direct, rtol=1e-6)

    # Growth is lower at higher redshift, and LCDM differs from EdS by about 1 per cent
    assert np.all(np.diff(ratio) < 0)
    eds = normalized_growth_factor(z_grid, z_ref, OMEGA_M, None)
    assert 1e-3 < np.max(np.abs(ratio / eds - 1)) < 2e-2


def test_old_growth_func_option_raises(tmp_path):
    """The removed old_growth_func option is an error."""
    main_ini = write_config_copy(
        TESTS_DIR / "full_configs" / "main.ini",
        tmp_path,
        corr_edits={"model": {"old_growth_func": "True"}},
        ini_files=["lyalya_lyalya.ini"],
    )

    with pytest.raises(ValueError, match="old_growth_func"):
        build_interface(main_ini)


# ----------------------------------------------------------------------------------------------
# Matrix exponents cannot be varied (estimator convention)
# ----------------------------------------------------------------------------------------------

METAL_ALPHA = "alpha_SiII(1260)"


@pytest.mark.parametrize(
    "main_edits",
    [
        pytest.param({"sample": sample_edit(**{METAL_ALPHA: 1.0})}, id="sample-metal"),
        pytest.param({"sample": sample_edit(alpha_LYA=2.9)}, id="sample-main-tracer"),
        pytest.param(
            {
                "monte carlo": sample_edit(**{METAL_ALPHA: 1.0}),
                "mc parameters": {METAL_ALPHA: "1.0"},
            },
            id="monte-carlo-metal",
        ),
        pytest.param({"chi2 scan": {METAL_ALPHA: "0.5 1.5 3"}}, id="chi2-scan-metal"),
        pytest.param({"chi2 scan": {"alpha_LYA": "2.5 3.0 3"}}, id="chi2-scan-main-tracer"),
    ],
)
def test_varied_matrix_exponent_raises_under_estimator(setups, tmp_path, main_edits):
    """Varying alpha_<metal> or alpha_<tracer> is refused under the estimator convention."""
    main_ini = write_config_copy(setups.main_ini(), tmp_path, main_edits)

    with pytest.raises(ValueError, match="metal_matrix_convention = estimator") as error:
        build_interface(main_ini)

    varied_name = METAL_ALPHA if "alpha_LYA" not in str(main_edits) else "alpha_LYA"
    assert varied_name in str(error.value)


@pytest.mark.parametrize("scale_name", ["alpha_smooth", "alpha"])
def test_other_alpha_parameters_are_not_matrix_exponents(setups, tmp_path, scale_name):
    """Sampling alpha_smooth or the full-shape alpha is allowed: only exact names are refused."""
    main_edits = {
        "parameters": {scale_name: "1.0"},
        "sample": sample_edit(**{scale_name: 1.0}),
    }
    main_ini = write_config_copy(setups.main_ini(), tmp_path, main_edits)

    vega, _ = build_interface(main_ini)

    assert scale_name in vega.sample_params["limits"]


def test_varied_metal_exponent_allowed_for_legacy_with_slow_metals(setups, tmp_path):
    """Under the legacy convention (slow metals) a sampled alpha_<metal> initializes."""
    main_edits = {"sample": sample_edit(**{METAL_ALPHA: 1.0})}
    main_ini = write_config_copy(
        setups.main_ini(),
        tmp_path,
        main_edits,
        corr_edits={"model": {"metal_matrix_convention": "legacy"}},
    )

    vega, _ = build_interface(main_ini)

    assert METAL_ALPHA in vega.sample_params["limits"]
    assert vega.minimizer is not None


# ----------------------------------------------------------------------------------------------
# Monte Carlo input model with exponents different from [parameters]
# ----------------------------------------------------------------------------------------------


def test_monte_carlo_input_model_uses_mc_exponent(setups, tmp_path):
    """The MC input model has the metal matrices of the MC exponent; the models are restored."""
    mc_alpha = 1.5
    no_sample = {"bias_eta_LYA": None, "beta_LYA": None}
    main_edits_mc = {
        "sample": no_sample,
        "monte carlo": {},
        "mc parameters": {METAL_ALPHA: str(mc_alpha)},
    }
    main_edits_direct = {"sample": no_sample, "parameters": {METAL_ALPHA: str(mc_alpha)}}

    base_main = setups.main_ini()
    vega_mc, _ = build_interface(write_config_copy(base_main, tmp_path / "mc", main_edits_mc))
    vega_direct, _ = build_interface(
        write_config_copy(base_main, tmp_path / "direct", main_edits_direct)
    )

    name = CORRELATION_NAMES["lyaxlya"]
    assert vega_mc.params[METAL_ALPHA] == 1.0
    assert vega_direct.params[METAL_ALPHA] == mc_alpha

    models_before = vega_mc.models
    metals_before = vega_mc.models[name].metals
    model_mask = vega_mc.data[name].model_mask
    plain_model = vega_mc.compute_model(run_init=False)[name][model_mask]

    # No initial fit: [sample] is empty and mc_start_from_fit is not set
    assert vega_mc.mc_config["sample"]["limits"] == {}
    assert vega_mc.sample_params["limits"] == {}
    assert "mc_start_from_fit" not in vega_mc.main_config["control"]

    # An MC value equal to [parameters] needs no new models
    assert (
        interface_utils.build_models_for_mc_exponents(vega_mc, {METAL_ALPHA: 1.0}, print_func=print)
        is None
    )
    assert (
        interface_utils.build_models_for_mc_exponents(vega_mc, {"beta_LYA": 0.3}, print_func=print)
        is None
    )

    messages = []
    mc_input_model = vega_mc.get_fiducial_for_monte_carlo(print_func=messages.append)[name]
    direct_model = vega_direct.compute_model(run_init=False)[name][
        vega_direct.data[name].model_mask
    ]

    # Equal to the model of an interface initialized directly with the MC exponent, and different
    # from the model with the [parameters] exponent
    assert np.array_equal(mc_input_model, direct_model)
    assert not np.allclose(mc_input_model, plain_model, rtol=1e-8, atol=0)
    assert any(METAL_ALPHA in message for message in messages)

    # The models of the interface (and the fits to the mocks) are unchanged
    assert vega_mc.models is models_before
    assert vega_mc.models[name].metals is metals_before
    for corr_item in vega_mc.corr_items.values():
        assert corr_item.parameters is vega_mc.params
    assert np.array_equal(vega_mc.compute_model(run_init=False)[name][model_mask], plain_model)


# ----------------------------------------------------------------------------------------------
# Fast-metals guards (picca-matrix path of the upstream test configuration)
# ----------------------------------------------------------------------------------------------

FAST_METALS_MESSAGE = "fast_metals = True in [model] of"
FAST = {"model": {"fast_metals": "True"}}


def full_configs_main(directory, main_edits=None, corr_edits=None):
    """Write a single-correlation (lyalya_lyalya) copy of ``tests/full_configs``.

    Parameters
    ----------
    directory : pathlib.Path
        Output directory.
    main_edits : dict, optional
        Edits of the main INI, see ``apply_edits``.
    corr_edits : dict, optional
        Edits of the correlation INI, see ``apply_edits``.

    Returns
    -------
    pathlib.Path
        Main INI of the copy.
    """
    return write_config_copy(
        TESTS_DIR / "full_configs" / "main.ini",
        directory,
        main_edits,
        corr_edits,
        ini_files=["lyalya_lyalya.ini"],
    )


def varied_names_edits(section, **parameters):
    """Edits that make parameters vary through one of the sections of the main INI.

    Parameters
    ----------
    section : {"sample", "monte carlo", "chi2 scan"}
        Section receiving the varied parameters: the full sampling specification around the
        value for ``[sample]`` and ``[monte carlo]`` (which also gets the matching
        ``[mc parameters]`` entries), and a range of +-0.1 around it for ``[chi2 scan]``.
    **parameters
        ``{name: value}``, passed by dictionary unpacking for names with parentheses.

    Returns
    -------
    dict
        Main-INI edits.
    """
    if section == "chi2 scan":
        return {
            "chi2 scan": {
                name: f"{value - 0.1} {value + 0.1} 3" for name, value in parameters.items()
            }
        }

    edits = {section: sample_edit(**parameters)}
    if section == "monte carlo":
        edits["mc parameters"] = {name: str(value) for name, value in parameters.items()}

    return edits


# Cases that raise: (id, main edits, correlation edits, substrings of the error message)
FAST_METALS_RAISING_CASES = [
    pytest.param(
        {"sample": sample_edit(**{"beta_SiII(1190)": 0.5})},
        {},
        ["beta_SiII(1190)"],
        id="sampled-metal-beta",
    ),
    pytest.param(
        {"parameters": {"beta_metals": "0.5"}, "sample": sample_edit(beta_metals=0.5)},
        {"model": {"single-metal-beta": "True"}},
        ["beta_metals"],
        id="sampled-beta-metals",
    ),
    pytest.param(
        {"sample": sample_edit(**{"alpha_SiII(1190)": 1.0})},
        {},
        ["alpha_SiII(1190)"],
        id="sampled-metal-alpha",
    ),
    pytest.param(
        {},
        {"model": {"no-metal-decomp": "False"}},
        ["no-metal-decomp = False"],
        id="no-metal-decomp-false",
    ),
    pytest.param(
        {"cosmo-fit type": {"metal-scaling": "True"}},
        {},
        ["metal-scaling = True"],
        id="metal-scaling",
    ),
    pytest.param(
        varied_names_edits("monte carlo", **{"beta_SiII(1190)": 0.5}),
        {},
        ["beta_SiII(1190)"],
        id="metal-beta-in-monte-carlo",
    ),
    pytest.param(
        varied_names_edits("chi2 scan", **{"beta_SiII(1190)": 0.5}),
        {},
        ["beta_SiII(1190)"],
        id="metal-beta-in-chi2-scan",
    ),
    pytest.param(
        {
            "sample": sample_edit(**{"beta_SiII(1260)": 0.5, "alpha_CIV(eff)": 1.0}),
            "cosmo-fit type": {"metal-scaling": "True"},
        },
        {"model": {"no-metal-decomp": "False"}},
        [
            "beta_SiII(1260)",
            "alpha_CIV(eff)",
            "no-metal-decomp = False",
            "metal-scaling = True",
        ],
        id="all-offending-items-reported",
    ),
]


@pytest.mark.parametrize("main_edits, corr_edits, substrings", FAST_METALS_RAISING_CASES)
def test_fast_metals_guard_raises(tmp_path, main_edits, corr_edits, substrings):
    """fast_metals = True with an incompatible parameter or option is refused at init."""
    corr_edits = {section: dict(options) for section, options in corr_edits.items()}
    corr_edits.setdefault("model", {})["fast_metals"] = "True"
    main_ini = full_configs_main(tmp_path, main_edits, corr_edits)

    with pytest.raises(ValueError) as error:
        build_interface(main_ini)

    message = str(error.value)
    assert FAST_METALS_MESSAGE in message
    assert "'lyalya_lyalya'" in message
    for substring in substrings:
        assert substring in message

    # Not the matrix-exponent error: this configuration uses picca's matrices
    assert "metal_matrix_convention" not in message


# Cases that must not raise with fast metals
FAST_METALS_ALLOWED_CASES = [
    pytest.param({}, id="baseline-sampled-bias-eta-and-beta-LYA"),
    pytest.param(
        {
            "parameters": {"bias_SiII(1190)": "0.01"},
            "sample": sample_edit(**{"bias_SiII(1190)": 0.01}),
        },
        id="bias-metal",
    ),
    pytest.param({"sample": sample_edit(**{"bias_eta_SiII(1190)": -0.0026})}, id="bias-eta-metal"),
    pytest.param(
        {"parameters": {"alpha_smooth": "1.0"}, "sample": sample_edit(alpha_smooth=1.0)},
        id="alpha-smooth",
    ),
    pytest.param({"sample": sample_edit(alpha_LYA=2.9)}, id="main-tracer-alpha"),
]


@pytest.mark.parametrize("main_edits", FAST_METALS_ALLOWED_CASES)
def test_fast_metals_allowed_parameters(tmp_path, main_edits):
    """Varying biases and parameters of the main tracers is compatible with fast metals."""
    main_ini = full_configs_main(tmp_path, main_edits, FAST)

    vega, _ = build_interface(main_ini)

    # The varied parameters were recognized (not skipped), and fast metals are on
    sampled = {"bias_eta_LYA", "beta_LYA"} | set(main_edits.get("sample", {}))
    assert sampled <= set(vega.sample_params["limits"])
    assert vega.models["lyalya_lyalya"].metals.fast_metals


def test_fast_metals_options_allowed_with_slow_metals(tmp_path):
    """The settings refused with fast metals initialize when fast_metals is off."""
    main_edits = {
        "sample": sample_edit(**{"beta_SiII(1190)": 0.5, "alpha_SiII(1190)": 1.0}),
        "cosmo-fit type": {"metal-scaling": "True"},
        **varied_names_edits("chi2 scan", **{"beta_SiII(1260)": 0.5}),
    }
    corr_edits = {"model": {"no-metal-decomp": "False"}}
    main_ini = full_configs_main(tmp_path, main_edits, corr_edits)

    vega, _ = build_interface(main_ini)

    metals = vega.models["lyalya_lyalya"].metals
    assert not metals.fast_metals
    assert not vega.models["lyalya_lyalya"].no_metal_decomp
    assert vega.scale_params.metal_scaling


# ----------------------------------------------------------------------------------------------
# Matrix-exponent error versus fast-metals error
# ----------------------------------------------------------------------------------------------


def test_matrix_exponent_error_is_not_the_fast_metals_error(setups, tmp_path):
    """Under the estimator convention a sampled alpha_<metal> gets the exponent error only."""
    main_edits = {"sample": sample_edit(**{METAL_ALPHA: 1.0})}
    main_ini = write_config_copy(
        setups.main_ini(),
        tmp_path,
        main_edits,
        corr_edits={"model": {"fast_metals": "True"}},
    )

    with pytest.raises(ValueError) as error:
        build_interface(main_ini)

    assert "metal_matrix_convention" in str(error.value)
    assert METAL_ALPHA in str(error.value)
    assert FAST_METALS_MESSAGE not in str(error.value)


def test_fast_metals_error_for_metal_beta_on_new_metals(setups, tmp_path):
    """On the new-metals estimator setup a sampled metal beta gets the fast-metals error only."""
    main_edits = {"sample": sample_edit(**{"beta_SiII(1260)": 0.5})}
    main_ini = write_config_copy(
        setups.main_ini(),
        tmp_path,
        main_edits,
        corr_edits={"model": {"fast_metals": "True"}},
    )

    with pytest.raises(ValueError) as error:
        build_interface(main_ini)

    assert FAST_METALS_MESSAGE in str(error.value)
    assert "beta_SiII(1260)" in str(error.value)
    assert "metal_matrix_convention" not in str(error.value)


def test_fast_metals_error_for_metal_alpha_under_legacy(setups, tmp_path):
    """Under the legacy convention a sampled alpha_<metal> is refused by the fast-metals guard."""
    main_edits = {"sample": sample_edit(**{METAL_ALPHA: 1.0})}
    main_ini = write_config_copy(
        setups.main_ini(),
        tmp_path,
        main_edits,
        corr_edits={"model": {"fast_metals": "True", "metal_matrix_convention": "legacy"}},
    )

    with pytest.raises(ValueError) as error:
        build_interface(main_ini)

    assert FAST_METALS_MESSAGE in str(error.value)
    assert METAL_ALPHA in str(error.value)
    assert "metal_matrix_convention = estimator" not in str(error.value)


# ----------------------------------------------------------------------------------------------
# Smoke test of the sensitivity computation (moved to vega.interface_utils with the refactoring)
# ----------------------------------------------------------------------------------------------


def test_compute_sensitivity_smoke(tmp_path):
    """compute_sensitivity() fills ``vega.sensitivity`` for the sampled parameters.

    Uses the single-correlation copy of ``tests/full_configs`` (sampled: bias_eta_LYA, beta_LYA).
    Fast metals and the fast metal bias are switched off because the metal model refuses to save
    the model components (which ``compute_sensitivity`` switches on) with them.
    """
    corr_edits = {"model": {"fast_metals": "False", "fast_metal_bias": "False"}}
    main_ini = full_configs_main(tmp_path, corr_edits=corr_edits)
    vega, _ = build_interface(main_ini)
    sampled_names = ["bias_eta_LYA", "beta_LYA"]
    assert set(vega.sample_params["limits"]) == set(sampled_names)

    nominal = {
        "bias_eta_LYA": (vega.params["bias_eta_LYA"], 0.01),
        "beta_LYA": (vega.params["beta_LYA"], 0.05),
    }
    vega.compute_sensitivity(nominal=nominal, verbose=False)
    sensitivity = vega.sensitivity

    assert set(sensitivity) == {"nominal", "partials", "fisher"}
    assert sensitivity["nominal"] == nominal
    assert vega.fiducial["save-components"]

    # Partial derivatives: (distorted/undistorted, peak/smooth, model bin) per parameter
    parameter_pairs = [
        ("bias_eta_LYA", "bias_eta_LYA"),
        ("bias_eta_LYA", "beta_LYA"),
        ("beta_LYA", "beta_LYA"),
    ]
    for name, corr_item in vega.corr_items.items():
        num_model_bins = len(corr_item.model_coordinates.rp_grid)

        assert set(sensitivity["partials"][name]) == set(sampled_names)
        for pname in sampled_names:
            partials = sensitivity["partials"][name][pname]
            assert partials.shape == (2, 2, num_model_bins)
            assert np.all(np.isfinite(partials))

        # Fisher information: (distorted/undistorted, model bin) per parameter pair, NaN where
        # the bin is masked
        assert set(sensitivity["fisher"][name]) == set(parameter_pairs)
        data_mask = vega.data[name].data_mask
        for pair in parameter_pairs:
            fisher = sensitivity["fisher"][name][pair]
            assert fisher.shape == (2, num_model_bins)
            assert np.all(np.isfinite(fisher[:, data_mask]))
