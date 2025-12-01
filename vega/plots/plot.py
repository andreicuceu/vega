import numpy as np
import matplotlib.pyplot as plt

from .wedges import Wedge
from .shell import Shell
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
        self.mask = {}

        if vega_data is not None:
            for name, data in vega_data.items():
                cross_flag = data.tracer1['type'] != data.tracer2['type']
                self.cross_flag[name] = cross_flag
                self.data[name] = data.data_vec
                if data.has_cov_mat:
                    self.cov_mat[name] = data.cov_mat

                # Initialize data coordinates
                self.rp_setup_data[name], self.rt_setup_data[name], self.r_setup_data[name] = \
                    self.initialize_coordinates(data.data_coordinates)

                self.cuts[name] = {'r_min': data.r_min_cut,
                                   'r_max': data.r_max_cut}

                self.mask[name] = data.dist_model_coordinates.get_mask_to_other(
                    data.data_coordinates)

                # Initialize model coordinates
                self.rp_setup_model[name], self.rt_setup_model[name], self.r_setup_model[name] = \
                    self.initialize_coordinates(data.model_coordinates)

            self.has_data = True

    def initialize_coordinates(self, coordinates):
        rp_setup = (coordinates.rp_min, coordinates.rp_max, coordinates.rp_nbins)
        rt_setup = (0., coordinates.rt_max, coordinates.rt_nbins)
        r_setup = rt_setup
        return rp_setup, rt_setup, r_setup

    def initialize_wedge(
        self, mu_bin, corr_name=None, is_data=False, cross_flag=False,
        rp_setup=None, rt_setup=None, r_setup=None, abs_mu=True, **kwargs
    ):
        """Initialize wedge object

        Parameters
        ----------
        mu_bin : (float, float)
            Min and max mu value defining the wedge
        corr_name : str, optional
            Name of the correlation component, by default None
        is_data : bool, optional
            Whether the wedge is for data, by default False
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

    def initialize_shell(
        self, r_bin, corr_name=None, is_data=False, cross_flag=False,
        rp_setup=None, rt_setup=None, angle_var='theta', **kwargs
    ):
        """Initialize shell object

        Parameters
        ----------
        r_bin : (float, float)
            Min and max r value defining the shell
        corr_name : str, optional
            Name of the correlation component, by default None
        is_data : bool, optional
            Whether the shell is for data, by default False
        cross_flag : bool, optional
            Whether the shell is for the cross-correlation, by default False
        rp_setup : (float, float, int), optional
            (min, max, size) specification for input r_parallel, by default None
        rt_setup : (float, float, int), optional
            (min, max, size) specification for input r_transverse, by default None
        angle_var : str, optional
            Name of the angle variable, by default 'theta'

        Returns
        -------
        vega.Shell
            Vega shell object
        """
        if corr_name is not None:
            if is_data:
                rp = self.rp_setup_data[corr_name]
                rt = self.rt_setup_data[corr_name]
            else:
                rp = self.rp_setup_model[corr_name]
                rt = self.rt_setup_model[corr_name]
        else:
            if rp_setup is not None:
                rp = rp_setup
            elif cross_flag:
                rp = (-200., 200., 100)
            else:
                rp = (0., 200., 50)

            rt = rt_setup if rt_setup is not None else (0., 200., 50)

        # Compute range of angles/mu for the shell
        if angle_var == 'theta':
            angle_range = (0, np.pi) if cross_flag else (0, np.pi/2)
        else:
            angle_range = (-1, 1) if cross_flag else (0, 1)

        # Compute binning factor for the shell
        # This was manually tuned to get a reasonable number of bins
        binning_factor = np.mean(r_bin) * np.sqrt(r_bin[1] - r_bin[0]) * 3

        return Shell(
            r=r_bin, rp=rp, rt=rt, angle_var=angle_var, angle_range=angle_range,
            num_bins_fraction=binning_factor
        )

    def plot_data(
        self, ax, x_bin, is_shell=False, data=None, cov_mat=None, cross_flag=False, label=None,
        corr_name='lyaxlya', data_fmt='o', data_color=None, scaling_power=2,
        use_local_coordinates=True, alpha=1.0, **kwargs
    ):
        """Plot the data in the input ax object

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the data in
        x_bin : (float, float)
            Bin edges for the x-axis, mu for wedges and r for shells
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the plot is for a cross-correlation, by default False
        label : str, optional
            Label for the data points, by default None
        corr_name : str, optional
            Name of the correlation component, by default 'lyaxlya'
        data_fmt : str, optional
            Data formatting, by default 'o'
        data_color : str, optional
            Color for the data points, by default None
        scaling_power : float, optional
            The power of r that multiples the plotted correlation (xi * r^scaling_power),
            by default None
        use_local_coordinates : bool, optional
            Whether to use the stored coordinate settings or defaul/input values, by default True
        alpha: float, optional
            The transparency level of the plotted data points, by default 1.0
        """
        init_func = self.initialize_shell if is_shell else self.initialize_wedge
        if use_local_coordinates and self.has_data:
            compression_obj = init_func(x_bin, corr_name, True, cross_flag, **kwargs)
        else:
            compression_obj = init_func(x_bin, cross_flag=cross_flag, **kwargs)

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

        x_grid, x_data, x_cov = compression_obj(data_vec, covariance=covariance)

        if is_shell:
            ax.errorbar(
                x_grid, x_data * 1e3, yerr=np.sqrt(x_cov.diagonal()) * 1e3,
                fmt=data_fmt, color=data_color, label=label, alpha=alpha, capsize=2
            )
        else:
            ax.errorbar(
                x_grid, x_data * x_grid**scaling_power,
                yerr=np.sqrt(x_cov.diagonal()) * x_grid**scaling_power,
                fmt=data_fmt, color=data_color, label=label, alpha=alpha
            )

        return x_grid, x_data, x_cov

    def plot_model(
        self, ax, x_bin, is_shell=False, model=None, cov_mat=None, cross_flag=False,
        label=None, corr_name='lyaxlya', model_ls='-', model_color=None,
        scaling_power=2, use_local_coordinates=True, **kwargs
    ):
        """Plot the model in the input ax object using the input wedge object

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the model in
        x_bin : (float, float)
            Bin edges for the x-axis, mu for wedges and r for shells
        is_shell : bool, optional
            Whether the plot is for a shell, by default False
        model : array or dict, optional
            Model vector as an array or a dictionary of components, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the plot is for a cross-correlation, by default False
        label : str, optional
            Label for the model, by default None
        corr_name : str, optional
            Name of the correlation component, by default 'lyaxlya'
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
        if cov_mat is None:
            if corr_name in self.cov_mat:
                cov_mat = self.cov_mat[corr_name]

        covariance = None
        masked_model = None
        model_vec = np.array(array_or_dict(model, corr_name))
        if cov_mat is not None and corr_name in self.mask:
            covariance = array_or_dict(cov_mat, corr_name)

            if len(self.mask[corr_name]) == len(model_vec):
                masked_model = model_vec[self.mask[corr_name]]
                if len(masked_model) != len(self.data[corr_name]):
                    raise ValueError('Masked model array does not match data array.')

        init_func = self.initialize_shell if is_shell else self.initialize_wedge
        if masked_model is not None:
            wedge_obj = init_func(x_bin, corr_name, True, cross_flag, **kwargs)
        elif use_local_coordinates and self.has_data:
            wedge_obj = init_func(x_bin, corr_name, False, cross_flag, **kwargs)
        else:
            wedge_obj = init_func(x_bin, cross_flag=cross_flag, **kwargs)

        if cov_mat is None or wedge_obj.weights.shape[1] != cov_mat.shape[0]:
            x_grid, x_model = wedge_obj(model_vec)
        else:
            covariance = array_or_dict(cov_mat, corr_name)
            model_to_compress = masked_model if masked_model is not None else model_vec
            x_grid, x_model, _ = wedge_obj(model_to_compress, covariance=covariance)

        if is_shell:
            ax.plot(
                x_grid, x_model * 1e3, ls=model_ls, color=model_color, label=label)
        else:
            ax.plot(
                x_grid, x_model * x_grid**scaling_power,
                ls=model_ls, color=model_color, label=label
            )

        return x_grid, x_model

    def postprocess_wedge_plot(
        self, ax, mu_bin=None, xlim=(0, 180), ylim=None, no_legend=False,
        title='mu_bin', legend_loc='best', legend_ncol=1, **kwargs
    ):
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

    @staticmethod
    def postprocess_fig(fig, xlim=(0, 180), ylim=None):
        for ax in fig.axes:
            ax.grid()
            ax.set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ylim = np.array(ylim)
            if ylim.ndim == 1:
                for ax in fig.axes:
                    ax.set_ylim(ylim[0], ylim[1])
            elif ylim.ndim == 2:
                for ax, (ymin, ymax) in zip(fig.axes, ylim):
                    ax.set_ylim(ymin, ymax)
            else:
                raise ValueError(f'ylim variable has unsupported ndim {ylim.ndim}, '
                                 'only 1D and 2D arrays/lists/tuples allowed')

    def plot_wedge(
        self, ax, mu_bin, models=None, cov_mat=None, labels=None, data=None,
        cross_flag=False, corr_name='lyaxlya', models_only=False,
        data_only=False, data_label=None, no_postprocess=False, **kwargs
    ):
        """Plot a wedge into the input axes using the input mu_bin

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the wedge in
        mu_bin : (float, float)
            Min and max mu value defining the wedge
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
            Name of the correlation component, by default 'lyaxlya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        no_postprocess : bool, optional
            Whether to skip the postprocessing of the plot, by default False
        """
        data_wedge = None
        if not models_only:
            data_wedge = self.plot_data(
                ax, mu_bin, data=data, cov_mat=cov_mat, cross_flag=cross_flag,
                data_label=data_label, corr_name=corr_name, **kwargs
            )

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

                model_wedge = self.plot_model(
                    ax, mu_bin, model=model, cov_mat=cov_mat, cross_flag=cross_flag,
                    label=label, corr_name=corr_name, model_ls=model_ls,
                    model_color=model_color, **kwargs
                )

        if not no_postprocess:
            self.postprocess_wedge_plot(ax, mu_bin, **kwargs)

        return data_wedge, model_wedge

    def plot_shells_panel(
        self, ax, r_bins, model=None, cov_mat=None, labels=None, data=None, cross_flag=False,
        corr_name='lyaxlya', models_only=False, data_fmts=None, colors=None,
        data_only=False, no_postprocess=False, **kwargs
    ):
        """Plot a wedge into the input axes using the input mu_bin

        Parameters
        ----------
        ax : plt.axes
            Axes object to plot the wedge in
        r_bins : List[(float, float)]
            List of min and max r values defining the shells
        model : array or dict, optional
            Model to plot, by default None
        cov_mat : array or dict, optional
            Covariance matrix as an array or a dictionary of components, by default None
        label : str, optional
            Label for the model, by default None
        data : array or dict, optional
            Data vector as an array or a dictionary of components, by default None
        cross_flag : bool, optional
            Whether the wedge is for the cross-correlation, by default False
        corr_name : str, optional
            Name of the correlation component, by default 'lyaxlya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_fmts : List[str], optional
            List of formats for the data points, by default None
        colors : List[str], optional
            List of colors for the models and data, by default None
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        no_postprocess : bool, optional
            Whether to skip the postprocessing of the plot, by default False
        """
        data_shells = []
        model_shells = []

        for i, r_bin in enumerate(r_bins):
            fmt = '.' if data_fmts is None else data_fmts[i]
            color = None if colors is None else colors[i]
            if labels is None:
                label = r"$r \in [{:.0f}, {:.0f}]$ Mpc/h".format(r_bin[0], r_bin[1])
            else:
                label = labels[i] if i < len(labels) else None

            if not models_only:
                data_shells.append(self.plot_data(
                    ax, r_bin, is_shell=True, data=data, cov_mat=cov_mat,
                    cross_flag=cross_flag, label=label, corr_name=corr_name,
                    data_fmt=fmt, data_color=color, **kwargs
                ))

            if not data_only:
                model_shells.append(self.plot_model(
                    ax, r_bin, is_shell=True, model=model, cov_mat=cov_mat,
                    cross_flag=cross_flag, corr_name=corr_name, model_color=color,
                    **kwargs
                ))

        # if not no_postprocess:
        #     self.postprocess_plot(ax, mu_bin, **kwargs)

        return data_shells, model_shells

    def plot_shells_residuals(
        self, ax, data_shells, model_shells, data_fmts=None, colors=None, alpha=1.0,
        var_latex=r"\theta", set_ylabel=True, **kwargs
    ):
        assert len(data_shells) == len(model_shells)

        max_residual = 0
        for i, (data_shell, model_shell) in enumerate(zip(data_shells, model_shells)):
            x_grid, x_data, x_cov = data_shell
            x_model = model_shell[1]

            # Compute residuals
            x_residuals = (x_data - x_model) / np.sqrt(x_cov.diagonal())
            max_residual = max(max_residual, np.max(np.abs(x_residuals)))

            # Plot residuals
            fmt = '.' if data_fmts is None else data_fmts[i]
            color = None if colors is None else colors[i]

            ax.errorbar(
                x_grid, x_residuals, yerr=np.ones_like(x_residuals),
                fmt=fmt, color=color, alpha=alpha, capsize=2
            )

        if set_ylabel:
            ax.set_ylabel(r"$\Delta\xi(" + var_latex + r")/\sigma_{\xi}$")

        if 'theta' in var_latex:
            ax.set_xlabel(r"$\theta$ [deg]")
        else:
            ax.set_xlabel(f"${var_latex}$")
        ax.axhline(0, c='k')

        if max_residual < 3:
            ax.set_ylim(-4, 4)
        else:
            ax.set_ylim(-max_residual - 1, max_residual + 1)

    def plot_1wedge(self, models=None, cov_mat=None, labels=None, data=None, cross_flag=False,
                    corr_name='lyaxlya', models_only=False, data_only=False, data_label=None,
                    fig=None, **kwargs):
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
            Name of the correlation component, by default 'lyaxlya'
        models_only : bool, optional
            Whether to only plot models and ignore the data, by default False
        data_only : bool, optional
            Whether to only plot data and ignore the models, by default False
        data_label : str, optional
            Label for the data, by default None
        """
        if not kwargs.get('no_font', False):
            plt.rcParams['font.size'] = 14

        if fig is None:
            fig, axs = plt.subplots(1, figsize=(10, 6))
        else:
            axs = fig.axes[0]

        _ = self.plot_wedge(axs, (0, 1), models=models, cov_mat=cov_mat, labels=labels, data=data,
                            cross_flag=cross_flag, corr_name=corr_name, models_only=models_only,
                            data_only=data_only, data_label=data_label, **kwargs)

        self.fig = fig

    def plot_2wedges(self, mu_bins=(0, 0.5, 1), models=None, cov_mat=None, labels=None,
                     data=None, cross_flag=False, corr_name='lyaxlya', models_only=False,
                     data_only=False, data_label=None, vertical_plots=False, fig=None, **kwargs):
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
            Name of the correlation component, by default 'lyaxlya'
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

        if fig is None:
            if not vertical_plots:
                fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            else:
                fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        else:
            axs = np.array(fig.axes)

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
                     labels=None, data=None, cross_flag=False, corr_name='lyaxlya',
                     models_only=False, data_only=False, data_label=None, figsize=(14, 8),
                     mu_bin_labels=False, fig=None, **kwargs):
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
            Name of the correlation component, by default 'lyaxlya'
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

        if fig is None:
            fig, axs = plt.subplots(2, 2, figsize=figsize)
        else:
            axs = np.array(fig.axes)

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
                ax.fill_betweenx((-100, 100), xmin, self.cuts[corr_name]['r_min'],
                                 color='gray', alpha=0.7)
                ax.fill_betweenx((-100, 100), self.cuts[corr_name]['r_max'], xmax,
                                 color='gray', alpha=0.7)
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(xmin, xmax)

        plt.tight_layout()
        self.fig = fig

    def plot_4wedge_panel(self, mu_bins=(0, 0.5, 0.8, 0.95, 1), model=None, cov_mat=None,
                          data=None, cross_flag=False, corr_name='lyaxlya', colors=None,
                          data_only=False, title=None, figsize=(8, 6), fig=None, **kwargs):
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
            Name of the correlation component, by default 'lyaxlya'
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

        if fig is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            ax = fig.axes[0]

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
        self.postprocess_wedge_plot(ax, title=title, **kwargs)
        if self.has_data:
            ymin, ymax = ax.get_ylim()
            ax.fill_betweenx((ymin, ymax), xmin, self.cuts[corr_name]['r_min'],
                             color='gray', alpha=0.7)
            ax.fill_betweenx((ymin, ymax), self.cuts[corr_name]['r_max'], xmax,
                             color='gray', alpha=0.7)
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

        self.fig = fig

    def plot_4shells(
        self, model, angle_var='theta', r_bins=None, corr_name='lyaxlya',
        var_latex=r'\theta'
    ):
        if r_bins is None:
            rmin = self.cuts[corr_name]['r_min']
            rmax = self.cuts[corr_name]['r_max']
            r_bins = np.logspace(np.log10(rmin), np.log10(rmax), 5)
            r_bins[1:-1] = np.round(r_bins[1:-1], -1)
        else:
            assert len(r_bins) == 5, 'plot_4shels works with exactly 4 shells (5 bin edges)'

        plt.rcParams['font.size'] = 16
        plt.rc('axes', labelsize=22)
        plt.rc('axes', titlesize=18)
        plt.rc('legend', fontsize=20)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        fig, axs = plt.subplots(
            2, 2, figsize=(16, 8), sharex=True, height_ratios=(4, 1), gridspec_kw={'hspace': 0})

        r_zip = list(zip(r_bins[:-1], r_bins[1:]))

        cmap = plt.get_cmap('seismic')
        colors = cmap((0.25, 0.75, 0.03, 1.0))
        fmts = ['d', '.', 'd', '.']

        cross = 'qso' in corr_name
        # return r_zip
        data_shells, model_shells = self.plot_shells_panel(
            axs[0, 0], r_zip[:2], model=model, cross_flag=cross, corr_name=corr_name,
            data_fmts=fmts[:2], colors=colors[:2], angle_var=angle_var
        )

        self.plot_shells_residuals(
            axs[1, 0], data_shells, model_shells, data_fmts=fmts[:2],
            colors=colors[:2], angle_var=angle_var
        )

        data_shells, model_shells = self.plot_shells_panel(
            axs[0, 1], r_zip[2:], model=model, cross_flag=cross, corr_name=corr_name,
            data_fmts=fmts[2:], colors=colors[2:], angle_var=angle_var
        )

        self.plot_shells_residuals(
            axs[1, 1], data_shells, model_shells, data_fmts=fmts[2:],
            colors=colors[2:], set_ylabel=False, angle_var=angle_var
        )

        axs[0, 0].set_ylabel(r"$10^3\xi(" + var_latex + r")$")
        axs[0, 0].legend()
        axs[0, 1].legend()

        if cross:
            axs[1, 0].set_xticks([0, 30, 60, 90, 120, 150, 180])
            axs[1, 1].set_xticks([0, 30, 60, 90, 120, 150, 180])
        else:
            axs[1, 0].set_xticks([0, 30, 60, 90])
            axs[1, 1].set_xticks([0, 30, 60, 90])

        self.fig = fig

    def plot_sensitivity(self, sensitivity, pname='ap', pname2=None, pct=95,
                         distorted=True, comp='both', rpow=0, save=None):
        """Plot parameter sensitivities.

        Plot the sensitivity to one parameter or the joint sensitivity to a pair of parameters.
        The resulting plot shows the partial derivatives of `pname` on the left-hand side and
        the distribution of the Fisher information for pname or, if pname2 is specified,
        (pname, pname2) on the right-hand side.

        Parameters
        ----------
        sensitivity - dict
            Dictionary with keys `nominal`, `partials` and `fisher`, normally obtained by calling
            `compute_sensitivity()` on a VegaInterface object, then passing its `sensitivity`
            attribute here.
        pname - str
            Name of the first parameter to use. Partial derivatives are only displayed for this
            parameter, even when pname2 is specified.
        pname2 - str or None
            Name of the second parameter to use. Displays the Fisher information associated with
            the covariance of (pname,pname2) when specified. If None, then use (pname,pname).
        pct - float
            Clip the color map for values above this percentile value.
        distorted - bool
            Plot the sensitivity of the predicted correlation including the distortion matrix
            when True.  Otherwise, use the undistorted correlation function model.
        comp - str
            Which component of the signal model to display. Select either `peak`, `smooth`
            or `both`.
        rpow - float
            The power of the radial weight to use for plotting the partial derivatives of pname.
        save - str or None
            Save the produced plot a file with this name. When None, do not save the plot.
        """
        # Get the indices of the requested parameters.
        pnames = list(sensitivity['nominal'].keys())
        if pname not in pnames:
            raise RuntimeError(f'Unknown floating parameter "{pname}".')
        if not pname2:
            pname2 = pname
        elif pname2 not in pnames:
            raise RuntimeError(f'Unknown floating parameter "{pname2}".')

        cname = list(sensitivity['fisher'].keys())[0]
        ppair = (
            (pname, pname2) if (pname, pname2) in sensitivity['fisher'][cname] else (pname2, pname)
        )

        if comp not in ('peak', 'smooth', 'both'):
            raise ValueError(f'Invalid comp "{comp}" (expected peak/smooth/both)')

        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        pvalue, perror = sensitivity['nominal'][pname]
        title = f'{pname} = {pvalue:.4f} ± {perror:.3f}'
        if pname2:
            pvalue2, perror2 = sensitivity['nominal'][pname2]
            title += f', {pname2} = {pvalue2:.4f} ± {perror2:.3f}'
        fig.suptitle(title)
        gs = fig.add_gridspec(3, 4)

        # Lookup the max value of the Fisher info over all datasets,
        # to normalize the Fisher info plots.
        max_info = np.max([
            np.nanpercentile(sensitivity['fisher'][cname][ppair], pct)
            for cname in sensitivity['fisher']
        ])

        rtxt = '' if rpow == 0 else ('r ' if rpow == 1 else f'r**{rpow} ')
        dist = 0 if distorted else 1

        for cname in self.data:

            rtgrid = np.linspace(*self.rt_setup_data[cname])
            rpgrid = np.linspace(*self.rp_setup_data[cname])
            rt, rp = np.meshgrid(rpgrid, rtgrid)
            r = np.hypot(rp, rt).reshape(-1)
            nrt = len(rtgrid)
            nrp = len(rpgrid)
            bbox = tuple(np.percentile(rp, (0, 100))) + tuple(np.percentile(rt, (0, 100)))

            row = 0 if cname.startswith('lya') else slice(1, None)
            col = 0 if cname.endswith('lya') else 1
            y1, y2 = (0.92, 0.84) if cname.startswith('lya') else (0.96, 0.92)

            P = r**rpow * sensitivity['partials'][cname][pname][dist]
            if comp == 'both':
                P = P.sum(axis=0)
            elif comp == 'peak':
                P = P[0]
            elif comp == 'smooth':
                P = P[1]
            if np.all(P == 0):
                continue

            ax = fig.add_subplot(gs[row, col])
            vlim = np.percentile(np.abs(P), pct)
            ax.imshow(
                P.reshape(nrp, nrt), origin='lower', interpolation='none', cmap='seismic',
                vmin=-vlim, vmax=+vlim, extent=bbox, aspect='auto'
            )
            ax.text(0.95, y1, cname + ':', ha='right', transform=ax.transAxes)
            ax.text(0.95, y2, f'{rtxt}∂M(rp,rt)/∂p', ha='right', transform=ax.transAxes)

            cmap = plt.get_cmap('afmhot_r').copy()
            cmap.set_bad('lightgray')

            # Lookup the Fisher distribution for this sample,
            # the specified params, and distortion option.
            F = sensitivity['fisher'][cname][ppair][dist]
            ax = fig.add_subplot(gs[row, col + 2])
            ax.imshow(
                F.reshape(nrp, nrt), origin='lower', interpolation='none', cmap=cmap,
                vmin=0, vmax=max_info, extent=bbox, aspect='auto'
            )
            ax.text(0.95, y1, cname + ':', ha='right', transform=ax.transAxes)
            ax.text(0.95, y2, '∂$^2$F$_{pq}$(rt,rp)/∂rt∂rp', ha='right', transform=ax.transAxes)

        if save:
            plt.savefig(save)
