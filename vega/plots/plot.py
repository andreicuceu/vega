import numpy as np
import matplotlib.pyplot as plt

from .wedges import Wedge
from .utils import array_or_dict


class VegaPlots:
    def __init__(self, vega_data=None):
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
        self.rp_setup_model = {}
        self.rt_setup_model = {}
        self.r_setup_model = {}
        self.rp_setup_data = {}
        self.rt_setup_data = {}
        self.r_setup_data = {}
        self.has_data = False
        self.cuts = {}

        if vega_data is not None:
            for name in vega_data.keys():
                cross_flag = vega_data[name].tracer1['type'] != vega_data[name].tracer2['type']
                self.cross_flag[name] = cross_flag
                self.data[name] = vega_data[name].data_vec
                if vega_data[name].has_cov_mat:
                    self.cov_mat[name] = vega_data[name].cov_mat

                # Initialize model coordinates
                self.rp_setup_model[name] = (vega_data[name].rp_min_model,
                                             vega_data[name].rp_max_model,
                                             vega_data[name].num_bins_rp_model)
                self.rt_setup_model[name] = (0., vega_data[name].rt_max_model,
                                             vega_data[name].num_bins_rt_model)
                self.r_setup_model[name] = self.rp_setup_model[name]

                # Initialize data coordinates
                self.rp_setup_data[name] = (vega_data[name].rp_min_data,
                                            vega_data[name].rp_max_data,
                                            vega_data[name].num_bins_rp_data)
                self.rt_setup_data[name] = (0., vega_data[name].rt_max_data,
                                            vega_data[name].num_bins_rt_data)
                self.r_setup_data[name] = self.rp_setup_data[name]

                self.cuts[name] = {'r_min': vega_data[name].r_min_cut,
                                   'r_max': vega_data[name].r_max_cut}

            self.has_data = True

    def initialize_wedge(self, mu_bin, corr_name=None, is_data=False, cross_flag=False,
                         rp_setup=None, rt_setup=None, r_setup=None, abs_mu=True, **kwargs):
        """Initialize wedge object

        Parameters
        ----------
        mu_bin : (float, float)
            Min and max mu value defining the wedge
        corr_name : str, optional
            Name of the correlation component, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        rp_setup : (float, float, int), optional
            (min, max, size) specification for input r_parallel, by default None
        rt_setup : (float, float, int), optional
            (min, max, size) specification for input r_transverse, by default None
        r_setup : (float, float, int), optional
            (min, max, size) specification for output isotropic r, by default None
        abs_mu : bool, optional
            Whether to compute wedges in abs(mu), by default True

        Returns
        -------
        vega.Wedge
            Vega wedge object
        """
        if corr_name is not None:
            if is_data:
                rp = self.rp_setup_data[corr_name]
                rt = self.rt_setup_data[corr_name]
                r = self.r_setup_data[corr_name]
            else:
                rp = self.rp_setup_model[corr_name]
                rt = self.rt_setup_model[corr_name]
                r = self.r_setup_model[corr_name]
            if self.cross_flag[corr_name] and abs_mu:
                r = (0, rp[1], rp[2]//2)
        else:
            if rp_setup is not None:
                rp = rp_setup
            elif cross_flag:
                rp = (-200., 200., 100)
            else:
                rp = (0., 200., 50)

            rt = rt_setup if rt_setup is not None else (0., 200., 50)
            r = r_setup if r_setup is not None else (0., 200., 50)

        return Wedge(mu=mu_bin, rp=rp, rt=rt, r=r, abs_mu=abs_mu)

    def plot_data(self, ax, mu_bin, data=None, cov_mat=None, cross_flag=False, label=None,
                  corr_name='lyalya_lyalya', data_fmt='o', data_color=None,
                  scaling_power=2, use_local_coordinates=True, **kwargs):
        """Plot the data in the input ax object using the input wedge object

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the data in
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
        scaling_power : float, optional
            The power of r that multiples the plotted correlation (xi * r^scaling_power),
            by default None
        use_local_coordinates : bool, optional
            Whether to use the stored coordinate settings or defaul/input values, by default True
        """
        if use_local_coordinates and self.has_data:
            wedge_obj = self.initialize_wedge(mu_bin, corr_name, True, cross_flag, **kwargs)
        else:
            wedge_obj = self.initialize_wedge(mu_bin, cross_flag=cross_flag, **kwargs)

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
        ax.errorbar(rd, dd * rd**scaling_power, yerr=np.sqrt(cd.diagonal()) * rd**scaling_power,
                    fmt=data_fmt, color=data_color, label=label)

        return rd, dd, cd

    def plot_model(self, ax, mu_bin, model=None, cov_mat=None, cross_flag=False,
                   label=None, corr_name='lyalya_lyalya', model_ls='-', model_color=None,
                   scaling_power=2, use_local_coordinates=True, **kwargs):
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
        scaling_power : float, optional
            The power of r that multiples the plotted correlation (xi * r^scaling_power),
            by default None
        use_local_coordinates : bool, optional
            Whether to use the stored coordinate settings or defaul/input values, by default True
        """
        if use_local_coordinates and self.has_data:
            wedge_obj = self.initialize_wedge(mu_bin, corr_name, False, cross_flag, **kwargs)
        else:
            wedge_obj = self.initialize_wedge(mu_bin, cross_flag=cross_flag, **kwargs)

        if cov_mat is None:
            if corr_name in self.cov_mat:
                cov_mat = self.cov_mat[corr_name]

        model_vec = array_or_dict(model, corr_name)

        if cov_mat is None or wedge_obj.weights.shape[1] != len(model_vec):
            r, d = wedge_obj(model_vec)
            ax.plot(r, d * r**scaling_power, ls=model_ls, color=model_color, label=label)
        else:
            covariance = array_or_dict(cov_mat, corr_name)
            r, d, _ = wedge_obj(model_vec, covariance=covariance)
            ax.plot(r, d * r**scaling_power, ls=model_ls, color=model_color, label=label)

        return r, d

    def postprocess_plot(self, ax, mu_bin=None, xlim=(0, 180), ylim=None, no_legend=False,
                         title='mu_bin', legend_loc='best', legend_ncol=1, **kwargs):
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
        if not kwargs.get('no_ylabel', False):
            ax.set_ylabel(r"$r^2\xi(r)$")
        if not kwargs.get('no_xlabel', False):
            ax.set_xlabel(r"$r~[\mathrm{Mpc/h}]$")

        if title == 'mu_bin' and mu_bin is not None:
            ax.set_title(r"${}<\mu<{}$".format(mu_bin[0], mu_bin[1]))
        elif title is not None:
            ax.set_title(title)

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        if not no_legend:
            ax.legend(loc=legend_loc, ncol=legend_ncol)
        ax.grid()

    def plot_wedge(self, ax, mu_bin, models=None, cov_mat=None, labels=None, data=None,
                   cross_flag=False, corr_name='lyalya_lyalya', models_only=False,
                   data_only=False, data_label=None, no_postprocess=False, **kwargs):
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
        # if use_local_coordinates and self.has_data:
        #     wedge_obj = self.initialize_wedge(mu_bin, corr_name, cross_flag, **kwargs)
        # else:
        #     wedge_obj = self.initialize_wedge(mu_bin, cross_flag=cross_flag, **kwargs)

        data_wedge = None
        if not models_only:
            data_wedge = self.plot_data(ax, mu_bin, data, cov_mat, cross_flag, data_label,
                                        corr_name, **kwargs)

        model_wedge = None
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

                model_ls = '-'
                if models_ls is not None:
                    model_ls = models_ls[i]

                model_wedge = self.plot_model(ax, mu_bin, model, cov_mat, cross_flag, label,
                                              corr_name, model_ls=model_ls,
                                              model_color=model_color, **kwargs)

        if not no_postprocess:
            self.postprocess_plot(ax, mu_bin, **kwargs)

        return data_wedge, model_wedge

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
        if not kwargs.get('no_font', False):
            plt.rcParams['font.size'] = 14
        fig, axs = plt.subplots(1, figsize=(10, 6))

        _ = self.plot_wedge(axs, (0, 1), models=models, cov_mat=cov_mat, labels=labels, data=data,
                            cross_flag=cross_flag, corr_name=corr_name, models_only=models_only,
                            data_only=data_only, data_label=data_label, **kwargs)

        self.fig = fig

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
        if not kwargs.get('no_font', False):
            plt.rcParams['font.size'] = 14
        if not vertical_plots:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        axs = axs.flatten()
        mu_bins = np.flip(np.array(mu_bins))
        mu_limits = zip(mu_bins[1:], mu_bins[:-1])

        for ax, mu_bin in zip(axs, mu_limits):
            _ = self.plot_wedge(ax, mu_bin, models=models, cov_mat=cov_mat, labels=labels,
                                data=data, cross_flag=cross_flag, corr_name=corr_name,
                                models_only=models_only, data_only=data_only,
                                data_label=data_label, **kwargs)

        self.fig = fig

    def plot_4wedges(self, mu_bins=(0, 0.5, 0.8, 0.95, 1), models=None, cov_mat=None,
                     labels=None, data=None, cross_flag=False, corr_name='lyalya_lyalya',
                     models_only=False, data_only=False, data_label=None, figsize=(14, 8),
                     mu_bin_labels=False, **kwargs):
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
        if not kwargs.get('no_font', False):
            plt.rcParams['font.size'] = 14
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()

        mu_bins = np.flip(np.array(mu_bins))
        mu_limits = zip(mu_bins[1:], mu_bins[:-1])

        no_xlabel = [True, True, False, False]
        no_ylabel = [False, True, False, True]

        for ax, mu_bin, no_xl, no_yl in zip(axs, mu_limits, no_xlabel, no_ylabel):
            if mu_bin_labels:
                data_label = r"${}<|\mu|<{}$".format(mu_bin[0], mu_bin[1])
            _ = self.plot_wedge(ax, mu_bin, models=models, cov_mat=cov_mat, labels=labels,
                                data=data, cross_flag=cross_flag, corr_name=corr_name,
                                models_only=models_only, data_only=data_only,
                                data_label=data_label, no_xlabel=no_xl, no_ylabel=no_yl, **kwargs)

            if self.has_data:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.fill_betweenx((ymin, ymax), xmin, self.cuts[corr_name]['r_min'],
                                 color='gray', alpha=0.7)
                ax.fill_betweenx((ymin, ymax), self.cuts[corr_name]['r_max'], xmax,
                                 color='gray', alpha=0.7)
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(xmin, xmax)

        plt.tight_layout()
        self.fig = fig

    def plot_4wedge_panel(self, mu_bins=(0, 0.5, 0.8, 0.95, 1), model=None, cov_mat=None,
                          data=None, cross_flag=False, corr_name='lyalya_lyalya', colors=None,
                          data_only=False, title=None, figsize=(8, 6), **kwargs):
        """Plot the correlations into four wedges on one panel

        Parameters
        ----------
        mu_bins : tuple, optional
            Limits of mu bins that define the two wedges, by default (0, 0.5, 1)
        model : array or dict, optional
            Model to plot, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        corr_name : str, optional
            Name of the correlation component, by default 'lyalya_lyalya'
        colors : List[string], optional
            List of colors for the wedges, by default None
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        title : string, optional
            Title for plot, by default None
        figsize : (float, float), optional
            figsize object passed to plt.subplots, by default (10, 6)
        """
        assert len(mu_bins) == 5
        if not kwargs.get('no_font', False):
            plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots(1, figsize=figsize)

        mu_bins = np.flip(np.array(mu_bins))
        mu_limits = zip(mu_bins[1:], mu_bins[:-1])

        if colors is None:
            cmap = plt.get_cmap('seismic')
            colors = cmap((0.03, 0.25, 0.75, 1))

        for mu_bin, color in zip(mu_limits, colors):
            label = f'{mu_bin[0]:.2f} < ' + r'$|\mu|$' + f' < {mu_bin[1]:.2f}'
            data_label = label if data_only else None

            _ = self.plot_wedge(ax, mu_bin, models=[model], cov_mat=cov_mat, labels=[label],
                                model_colors=[color], data_color=color, data=data,
                                cross_flag=cross_flag, corr_name=corr_name, models_only=False,
                                data_only=data_only, data_label=data_label,
                                no_postprocess=True, **kwargs)

        xmin, xmax = ax.get_xlim()
        self.postprocess_plot(ax, title=title, **kwargs)
        if self.has_data:
            ymin, ymax = ax.get_ylim()
            ax.fill_betweenx((ymin, ymax), xmin, self.cuts[corr_name]['r_min'],
                             color='gray', alpha=0.7)
            ax.fill_betweenx((ymin, ymax), self.cuts[corr_name]['r_max'], xmax,
                             color='gray', alpha=0.7)
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

        self.fig = fig
