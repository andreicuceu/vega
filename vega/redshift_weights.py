"""Shared helpers for reading tracer redshift / weight catalogs.

Used by metal-matrix construction and by catalog-based bias evolution.
"""
import numpy as np
from astropy.io import fits


def rebin(vector, rebin_factor):
    """Rebin a vector by averaging contiguous blocks of length rebin_factor."""
    size = vector.size
    return vector[:(size // rebin_factor) * rebin_factor].reshape(
        (size // rebin_factor), rebin_factor).mean(-1)


def get_forest_weights(weights_path, rebin_factor=None):
    """Read forest stacked weights from a delta_attributes FITS file.

    Parameters
    ----------
    weights_path : str or Path
        Path to delta_attributes FITS (columns LOGLAM, WEIGHT).
    rebin_factor : int or None
        Optional rebinning factor applied to wavelength and weights.

    Returns
    -------
    wave : 1D array
        Wavelengths (Angstrom).
    weights : 1D array
        Forest weights.
    """
    with fits.open(weights_path) as hdul:
        stack_table = hdul[1].data

    wave = 10**stack_table["LOGLAM"]
    weights = stack_table["WEIGHT"]

    if rebin_factor is not None:
        wave = rebin(wave, rebin_factor)
        weights = rebin(weights, rebin_factor)

    return wave, weights


def get_qso_weights(weights_path, z_ref=2.25, z_evol=1.44, z_bins=1000):
    """Read QSO catalog redshifts and build a weighted redshift histogram.

    Parameters
    ----------
    weights_path : str or Path
        Path to QSO catalog FITS with column Z.
    z_ref : float
        Reference redshift for catalog weight evolution.
    z_evol : float
        Evolution index used as ((1+z)/(1+z_ref))**(z_evol - 1).
    z_bins : int
        Number of histogram bins over the catalog redshift range.

    Returns
    -------
    z_qso : 1D array
        Weighted-mean redshift in each occupied bin.
    weights_qso : 1D array
        Total weight in each occupied bin.
    """
    with fits.open(weights_path) as hdul:
        z_qso_cat = hdul[1].data['Z']

    weights_qso_cat = ((1. + z_qso_cat) / (1. + z_ref))**(z_evol - 1.)

    histo_w, zbins = np.histogram(z_qso_cat, bins=z_bins, weights=weights_qso_cat)
    histo_wz, _ = np.histogram(z_qso_cat, bins=zbins, weights=weights_qso_cat * z_qso_cat)
    selection = histo_w > 0
    z_qso = histo_wz[selection] / histo_w[selection]
    weights_qso = histo_w[selection]

    return z_qso, weights_qso


def weighted_mean_z(z, weights):
    """Return the weight-averaged redshift."""
    z = np.asarray(z, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wsum = np.sum(weights)
    if wsum <= 0:
        raise ValueError("Cannot compute weighted mean redshift: total weight is zero.")
    return float(np.sum(weights * z) / wsum)


def catalog_bias_evolution_factor(z, weights, alpha, z_eff):
    """Average ((1+z)/(1+z_eff))**alpha over the catalog weights.

    Parameters
    ----------
    z : array
        Redshift samples (e.g. histogram bin means).
    weights : array
        Weights for each sample.
    alpha : float
        Bias evolution index (alpha_LYA or alpha_QSO).
    z_eff : float
        Pivot redshift for the evolution factor.

    Returns
    -------
    float
        Catalog-averaged bias evolution factor F(alpha).
    """
    z = np.asarray(z, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wsum = np.sum(weights)
    if wsum <= 0:
        raise ValueError("Cannot compute catalog bias evolution: total weight is zero.")
    rel = (1. + z) / (1. + z_eff)
    return float(np.sum(weights * rel**alpha) / wsum)


def forest_wave_to_z(wave, absorber_name='LYA'):
    """Convert forest wavelengths to redshift for a given absorber."""
    from picca import constants as picca_constants
    return wave / picca_constants.ABSORBER_IGM[absorber_name] - 1.


def load_tracer_redshift_weights(tracer, config=None, absorber_name='LYA'):
    """Load (z, weights) for a tracer from its weights-path.

    Parameters
    ----------
    tracer : dict
        Tracer config with 'type' and 'weights-path'.
    config : ConfigParser section or dict-like, optional
        Source for z_ref_objects / z_evol_objects / z_bins_objects / rebin_factor.
        May be a full ConfigParser (uses [metal-matrix] if present) or a section.
    absorber_name : str
        Absorber used to convert forest wavelengths to redshift.

    Returns
    -------
    z : 1D array
    weights : 1D array
    """
    path = tracer.get('weights-path')
    if path is None:
        raise ValueError(
            f"Tracer {tracer.get('name', '?')} has no weights-path; "
            "set weights-tracer1/2 in [data]."
        )

    z_ref, z_evol, z_bins, rebin_factor = _parse_weight_config(config)

    if tracer['type'] == 'discrete':
        return get_qso_weights(path, z_ref=z_ref, z_evol=z_evol, z_bins=z_bins)

    if tracer['type'] == 'continuous':
        wave, weights = get_forest_weights(path, rebin_factor=rebin_factor)
        z = forest_wave_to_z(wave, absorber_name=absorber_name)
        return z, weights

    raise ValueError(f"Unknown tracer type for redshift weights: {tracer['type']}")


def _parse_weight_config(config):
    """Extract weight-histogram options with the same defaults as metals."""
    z_ref = 2.25
    z_evol = 1.44
    z_bins = 1000
    rebin_factor = None

    if config is None:
        return z_ref, z_evol, z_bins, rebin_factor

    section = config
    # Full ConfigParser: prefer [metal-matrix], else [model]
    if hasattr(config, 'has_section'):
        if config.has_section('metal-matrix'):
            section = config['metal-matrix']
        elif config.has_section('model'):
            section = config['model']
        else:
            return z_ref, z_evol, z_bins, rebin_factor

    if hasattr(section, 'getfloat'):
        z_ref = section.getfloat('z_ref_objects', z_ref)
        z_evol = section.getfloat('z_evol_objects', z_evol)
        z_bins = section.getint('z_bins_objects', z_bins)
        if section.get('rebin_factor', None) is not None:
            rebin_factor = section.getint('rebin_factor')
    elif isinstance(section, dict):
        z_ref = float(section.get('z_ref_objects', z_ref))
        z_evol = float(section.get('z_evol_objects', z_evol))
        z_bins = int(section.get('z_bins_objects', z_bins))
        rb = section.get('rebin_factor', None)
        rebin_factor = int(rb) if rb is not None else None

    return z_ref, z_evol, z_bins, rebin_factor
