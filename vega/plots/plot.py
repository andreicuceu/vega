import numpy as np
import matplotlib.pyplot as plt

from .wedges import Wedge
from .utils import array_or_dict


class VegaPlots:
    def __init__(self, vega_data=None, models=None):
        """Initialize plotting module with the vega internal info

        Parameters
        ----------
        vega_data : vega.Data, optional
            Vega data object, by default None
        models : List[np.array] or List[dict], optional
            List of models, by default None
        """
        self.cross_flag = {}
        self.data = {}
        self.cov_mat = {}
        if vega_data is not None:
            for name in vega_data.keys():
                cross_flag = vega_data[name].tracer1['type'] != vega_data[name].tracer2['type']
                self.cross_flag[name] = cross_flag
                self.data[name] = vega_data[name].data_vec
                if vega_data[name].has_cov_mat():
                    self.cov_mat[name] = vega_data[name].cov_mat

        self.models = None
        if models is not None:
            self.models = models

    def initialize_wedge(self, mu_bin, cross_flag=False):
        """Initialize wedge object

        Parameters
        ----------
        mu_bin : array or tuple
            Array or tuple containing mu_min and mu_max of the wedge
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False

        Returns
        -------
        vega.Wedge
            Vega wedge object
        """
        if not cross_flag:
            wedge_obj = Wedge(mu=mu_bin, rp=(0., 200., 50), rt=(0., 200., 50),
                              r=(0., 200., 50), abs_mu=True)
        else:
            wedge_obj = Wedge(mu=mu_bin, rp=(-200., 200., 100), rt=(0., 200., 50),
                              r=(0., 200., 50), abs_mu=True)

        return wedge_obj

    def plot_data(self, ax, wedge_obj, data=None, cov_mat=None, label=None,
                  corr_name='lyalya_lyalya', data_fmt='o', data_color=None, **kwargs):
        """Plot the data in the input ax object using the input wedge object

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the data in
        wedge_obj : vega.Wedge
            Vega wedge object for computing the wedge
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        label : str, optional
            Label for the data points, by default None
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        data_fmt : str, optional
            Data formatting, by default 'o'
        data_color : str, optional
            Color for the data points, by default None
        """
        if data is None:
            if corr_name not in self.data:
                raise ValueError('Correlation {} not found in input data'.format(corr_name))

            data = self.data[corr_name]

        data_vec = array_or_dict(data, corr_name)

        if cov_mat is None:
            if corr_name not in self.cov_mat:
                raise ValueError('Correlation {} not found in input data'.format(corr_name))
            cov_mat = self.cov_mat[corr_name]

        covariance = array_or_dict(cov_mat, corr_name)

        rd, dd, cd = wedge_obj(data_vec, covariance=covariance)
        ax.errorbar(rd, dd * rd**2, yerr=np.sqrt(cd.diagonal()) * rd**2, fmt=data_fmt,
                    color=data_color, label=label)

    def plot_model(self, ax, wedge_obj, model=None, cov_mat=None, label=None,
                   corr_name='lyalya_lyalya', model_ls='-', model_color=None, **kwargs):
        """Plot the model in the input ax object using the input wedge object

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the model in
        wedge_obj : vega.Wedge
            Vega wedge object for computing the wedge
        model : array or dict, optional
            Model vector as an array or a dictionary of components, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        label : str, optional
            Label for the model, by default None
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        model_ls : str, optional
            Model line style, by default '-'
        model_color : str, optional
            Color for the model line, by default None
        """
        if cov_mat is None:
            if corr_name in self.cov_mat:
                cov_mat = self.cov_mat[corr_name]
                if label is None:
                    label = corr_name

        model_vec = array_or_dict(model, corr_name)

        if cov_mat is None:
            r, d = wedge_obj(model_vec)
            ax.plot(r, d * r**2, ls=model_ls, color=model_color, label=label)
        else:
            covariance = array_or_dict(cov_mat, corr_name)
            r, d, _ = wedge_obj(model_vec, covariance=covariance)
            ax.plot(r, d * r**2, ls=model_ls, color=model_color, label=label)

    def postprocess_plot(self, ax, mu_bin, xlim=(0, 180), ylim=None, **kwargs):
        """Add postprocessing to the plot on input axes

        Parameters
        ----------
        ax : plt.axes
            Axes object to postprocess
        mu_bin : array or tuple
            Array or tuple containing mu_min and mu_max of the wedge
        xlim : tuple, optional
            Limits of the x axis, by default (0, 180)
        """
        ax.set_ylabel(r"$r^2\xi(r)$")
        ax.set_xlabel(r"$r~[\mathrm{Mpc/h}]$")
        if 'title' in kwargs:
            ax.set_title(kwargs['title'], fontsize=16)
        else:
            ax.set_title(r"${}<\mu<{}$".format(mu_bin[0], mu_bin[1]), fontsize=16)
        ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
        ax.grid()

    def plot_wedge(self, ax, mu_bin, models=None, cov_mat=None, labels=None, data=None,
                   cross_flag=False, corr_name='lyalya_lyalya', models_only=False,
                   data_only=False, data_label=None, **kwargs):
        """Plot a wedge into the input axes using the input mu_bin

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the wedge in
        wedge_obj : vega.Wedge
            Vega wedge object for computing the wedge
        models : List[array] or List[dict], optional
            List of models to plot, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        labels : List[str], optional
            List of labels for the models, by default None
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        """
        wedge_obj = self.initialize_wedge(mu_bin, cross_flag)

        if not models_only:
            self.plot_data(ax, wedge_obj, data, cov_mat, data_label, corr_name, **kwargs)

        if not data_only:
            models_colors = None
            if 'model_colors' in kwargs:
                models_colors = kwargs['model_colors']

            models_ls = None
            if 'models_ls' in kwargs:
                models_ls = kwargs['models_ls']

            for i, model in enumerate(models):
                label = None
                if labels is not None and i < len(labels):
                    label = labels[i]

                model_color = None
                if models_colors is not None:
                    model_color = models_colors[i]

                model_ls = None
                if models_ls is not None:
                    model_ls = models_ls[i]

                self.plot_model(ax, wedge_obj, model, cov_mat, label, corr_name,
                                model_ls=model_ls, model_color=model_color, **kwargs)

        self.postprocess_plot(ax, mu_bin, **kwargs)

    def plot_1wedge(self, models=None, cov_mat=None, labels=None, data=None, cross_flag=False,
                    corr_name='lyalya_lyalya', models_only=False, data_only=False, data_label=None,
                    **kwargs):
        """Plot the correlations into one wedge from mu=0 to mu=1

        Parameters
        ----------
        models : List[array] or List[dict], optional
            List of models to plot, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        labels : List[str], optional
            List of labels for the models, by default None
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        """
        plt.rcParams['font.size'] = 14
        fig, axs = plt.subplots(1, figsize=(10, 6))

        self.plot_wedge(axs, (0, 1), models=models, cov_mat=cov_mat, labels=labels, data=data,
                        cross_flag=cross_flag, corr_name=corr_name, models_only=models_only,
                        data_only=data_only, data_label=data_label, **kwargs)

    def plot_2wedges(self, mu_bins=(0, 0.5, 1), models=None, cov_mat=None, labels=None,
                     data=None, cross_flag=False, corr_name='lyalya_lyalya', models_only=False,
                     data_only=False, data_label=None, vertical_plots=False, **kwargs):
        """Plot the correlations into two wedges defined by the limits in mu_bins

        Parameters
        ----------
        mu_bins : tuple, optional
            Limits of mu bins that define the two wedges, by default (0, 0.5, 1)
        models : List[array] or List[dict], optional
            List of models to plot, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        labels : List[str], optional
            List of labels for the models, by default None
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        vertical_plots : bool, optional
            Whether to plot the two wedges vertically, by default False
        """
        assert len(mu_bins) == 3
        plt.rcParams['font.size'] = 14
        if not vertical_plots:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        axs = axs.flatten()
        mu_bins = np.flip(np.array(mu_bins))
        mu_limits = zip(mu_bins[1:], mu_bins[:-1])

        for ax, mu_bin in zip(axs, mu_limits):
            self.plot_wedge(ax, mu_bin, models=models, cov_mat=cov_mat, labels=labels,
                            data=data, cross_flag=cross_flag, corr_name=corr_name,
                            models_only=models_only, data_only=data_only, data_label=data_label,
                            **kwargs)

    def plot_4wedges(self, mu_bins=(0, 0.5, 0.8, 0.95, 1), models=None, cov_mat=None,
                     labels=None, data=None, cross_flag=False, corr_name='lyalya_lyalya',
                     models_only=False, data_only=False, data_label=None, **kwargs):
        """Plot the correlations into four wedges defined by the limits in mu_bins

        Parameters
        ----------
        mu_bins : tuple, optional
            Limits of mu bins that define the two wedges, by default (0, 0.5, 1)
        models : List[array] or List[dict], optional
            List of models to plot, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        labels : List[str], optional
            List of labels for the models, by default None
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        """
        assert len(mu_bins) == 5
        plt.rcParams['font.size'] = 14
        fig, axs = plt.subplots(2, 2, figsize=(20, 14))

        axs = axs.flatten()
        mu_bins = np.flip(np.array(mu_bins))
        mu_limits = zip(mu_bins[1:], mu_bins[:-1])

        for ax, mu_bin in zip(axs, mu_limits):
            self.plot_wedge(ax, mu_bin, models=models, cov_mat=cov_mat, labels=labels,
                            data=data, cross_flag=cross_flag, corr_name=corr_name,
                            models_only=models_only, data_only=data_only, data_label=data_label,
                            **kwargs)
