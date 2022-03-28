import numpy as np
import matplotlib.pyplot as plt

from .wedges import Wedge


def array_or_dict(input_obj, corr_name='lyalya_lyalya'):
    if type(input_obj) == dict:
        return input_obj[corr_name]

    return input_obj


def plot_wedges(models, covariance, multi_model=False, labels=None, data=None, cross=False):
    """Compute and plot wedges from vega model

    Parameters
    ----------
    models : np.array or list
        If multi_model is True, this is a list of correlation function models
        If multi_model is False, this is just a single model
    covariance : np.array
        The covariance matrix
    multi_model : bool, optional
        Whether there is only one or multiple models
    labels : list[string], optional
        List of labels for each model
    data : dict, optional
        Data vector
    cross : bool, optional
        Whether the input model/data are cross-correlations
    """
    plt.rcParams['font.size'] = 14
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    axs = np.array(axs).reshape(-1)
    mus = np.array([0., 0.5, 0.8, 0.95, 1.])
    mu_zip = zip(mus[:-1],mus[1:])

    for i, mu in enumerate(mu_zip):
        if not cross:
            wedge_obj = Wedge(mu=mu, rp=(0., 200., 50), rt=(0., 200., 50),
                                r=(0., 200., 50), abs_mu=True)
        else:
            wedge_obj = Wedge(mu=mu, rp=(-200., 200., 100), rt=(0., 200., 50),
                                r=(0., 200., 50), abs_mu=True)

        if data is not None:
            rd, dd, cd = wedge_obj(data, covariance=covariance)
            axs[i].errorbar(rd, dd * rd**2, yerr=np.sqrt(cd.diagonal()) * rd**2, fmt="o")

        if multi_model:
            for model, label in zip(models, labels):
                r, d, _ = wedge_obj(model, covariance=covariance)
                axs[i].plot(r, d * r**2, '-', label=label)
        else:
            r, d, _ = wedge_obj(models, covariance=covariance)
            axs[i].plot(r, d * r**2, '-', label='Model')

        axs[i].set_ylabel(r"$r^2\xi(r)$")
        axs[i].set_xlabel(r"$r~[\mathrm{Mpc/h}]$")
        axs[i].set_title(r"${}<\mu<{}$".format(mu[0], mu[1]), fontsize=16)
        axs[i].set_xlim(0, 180)
        axs[i].legend()
        axs[i].grid()
