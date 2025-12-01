import numpy as np
import matplotlib.pyplot as plt

from .wedges import Wedge
from .shell import Shell


def array_or_dict(input_obj, corr_name='lyalya_lyalya'):
    if isinstance(input_obj, dict):
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


def plot_shells(vega, model, angle_var='theta', rs=(30, 40, 50, 60, 70), corr='lyaxlya'):
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True, height_ratios=(4,1))

    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    r_zip = zip(rs[:-1], rs[1:])

    cmap = plt.get_cmap('seismic')
    colors = cmap((0.25, 0.75, 0.03, 1.0))
    fmts = ['d', '.', 'd', '.']

    cross = 'qso' in corr
    if angle_var == 'theta':
        angle_range = (0, np.pi) if cross else (0, np.pi/2)
    else:
        angle_range = (-1, 1) if cross else (0, 1)

    mask = vega.corr_items[corr].dist_model_coordinates.get_mask_to_other(
        vega.corr_items[corr].data_coordinates)

    for i, r in enumerate(r_zip):
        factor = np.mean(r)*np.sqrt(r[1]-r[0])*3
        if cross:
            wedge_obj = Shell(
                r=r, rp=(-200, 200, 100), rt=(0, 200, 50), num_bins_fraction=factor,
                abs_mu=False, angle_var=angle_var, angle_range=angle_range
            )
        else:
            wedge_obj = Shell(
                r=r, rp=(0, 200, 50), rt=(0, 200, 50), num_bins_fraction=factor,
                abs_mu=True, angle_var=angle_var, angle_range=angle_range
            )

        mud, dd, cd = wedge_obj(vega.data[corr].data_vec, covariance=vega.data[corr].cov_mat)
        axs[0, i//2].errorbar(
            mud, dd*1e3, yerr=np.sqrt(cd.diagonal())*1e3, fmt=fmts[i], c=colors[i],
            capsize=2,  label=r"$r \in [{}, {}]$ Mpc/h".format(r[0], r[1])
        )

        mu, d, _ = wedge_obj(model[corr][mask], covariance=vega.data[corr].cov_mat)
        axs[1, i//2].errorbar(
            mud, (dd-d)/np.sqrt(cd.diagonal()), yerr=np.ones_like(d), fmt=fmts[i], c=colors[i],
            capsize=2,  label=r"$r \in [{}, {}]$ Mpc/h".format(r[0], r[1])
        )

        axs[0, i//2].plot(mu, d*1e3, '-', c=colors[i])

        var_latex = r"\mu" if angle_var == 'mu' else r"\mu^2" if angle_var == 'mu2' else r"\theta"
        axs[0, i//2].set_ylabel(r"$10^3\xi(" + var_latex + r")$")
        axs[1, i//2].set_ylabel(r"$\Delta\xi(" + var_latex + r")/\sigma_{\xi}$")
        axs[1, i//2].set_xlabel(f"${var_latex}$")
        if cross:
            axs[0, i//2].legend(loc='upper center')
        else:
            axs[0, i//2].legend(loc='lower left')

        axs[1, i//2].axhline(0, c='k')
        axs[1, i//2].set_ylim(-4, 4)

        if angle_var == 'theta':
            axs[0, i//2].xaxis.set_inverted(True)
            axs[1, i//2].xaxis.set_inverted(True)
    for ax in axs.flatten():
        ax.grid()
    plt.tight_layout()
