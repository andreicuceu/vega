from pathlib import Path
import os.path
from astropy.io import fits
import numpy as np
import h5py


class Output:
    """Class for handling the Vega output,
    and reading/writing output files.
    """
    def __init__(self, config, data, corr_items, analysis=None):
        """

        Parameters
        ----------
        config : ConfigParser
            Output section of main config file
        data : dict
            Vega Data objects
        corr_items : dict
            Vega correlation_item objects
        analysis : Analysis, optional
            Analysis object, by default None
        """
        self.data = data
        self.analysis = analysis
        self.corr_items = corr_items
        self.type = config.get('type', 'fits')
        self.overwrite = config.get('overwrite', False)
        self.outfile = os.path.expandvars(config['filename'])
        self.output_cf = config.getboolean('write_cf', False)
        self.output_pk = config.getboolean('write_pk', False)

    def write_results(self, corr_funcs, params, minimizer=None, scan_results=None, models=None):
        """Write results in the fits or hdf format

        Parameters
        ----------
        corr_funcs : dict
            Model correlation functions to write to file.
            This should be the output of vega.compute_model()
        params : dict
            Parameters to write to file. These should be the
            parameters vega.compute_model() was called with.
        minimizer : Minimizer, optional
            Minimizer object after minimization was done, by default None
        scan_results : list, optional
            List of scan results, by default None
        models : dict, optional
            Dictionary with the Vega Model objects, by default None
        """
        if self.type == 'fits':
            self.write_results_fits(corr_funcs, params, minimizer, scan_results, models)
        elif self.type == 'hdf' or self.type == 'h5':
            self.write_results_hdf(minimizer, scan_results)
        else:
            raise ValueError('Unknown output type. Set type = fits'
                             ' or type = hdf')

    def write_results_fits(self, corr_funcs, params, minimizer=None, scan_results=None,
                           models=None):
        """Write output in the fits format

        Parameters
        ----------
        corr_funcs : dict
            Model correlation functions to write to file.
            This should be the output of vega.compute_model()
        params : dict
            Parameters to write to file. These should be the
            parameters vega.compute_model() was called with.
        minimizer : Minimizer, optional
            Minimizer object after minimization was done, by default None
        scan_results : list, optional
            List of scan results, by default None
        models : dict, optional
            Dictionary with the Vega Model objects, by default None
        """
        if self.data is None:
            raise ValueError('Output object was initialized with an invalid data object.'
                             ' Reinitialize with a valid vega.data object.')

        primary_hdu = fits.PrimaryHDU()
        model_hdu = self._model_hdu(corr_funcs, params)
        hdu_list = [primary_hdu, model_hdu]

        if minimizer is not None:
            bestfit_hdu = self._bestfit_hdu(minimizer)
            hdu_list.append(bestfit_hdu)

        if self.output_pk:
            assert models is not None
            for key, model in models.items():
                pk_hdu = self._pk_hdu(key, model)
                hdu_list.append(pk_hdu)

        if self.output_cf:
            assert models is not None
            for key, model in models.items():
                cf_hdu = self._cf_hdu(key, model)
                hdu_list.append(cf_hdu)

        if scan_results is not None:
            assert minimizer is not None
            scan_hdu = self._scan_hdu(scan_results, minimizer)
            hdu_list.append(scan_hdu)

        hdul = fits.HDUList(hdu_list)

        if self.outfile[-5:] != '.fits':
            self.outfile += '.fits'

        hdul.writeto(Path(self.outfile), overwrite=self.overwrite)

    def _model_hdu(self, corr_funcs, params):
        """Create HDU with the computed model correlations,
        and the parameters used to compute them

        Parameters
        ----------
        corr_funcs : dict
            Output correlations given compute_model
        params : dict
            Computation parameters

        Returns
        -------
        astropy.io.fits.hdu.table.BinTableHDU
            HDU with the model correlation
        """
        sizes = {name: len(cf) for name, cf in corr_funcs.items()}
        num_rows = np.max(list(sizes.values()))
        print(sizes)
        print(num_rows)
        columns = []
        for name, cf in corr_funcs.items():
            pad = (0, num_rows - sizes[name])
            padded_cf = np.pad(cf, pad, constant_values=0.)
            padded_mask = np.pad(self.data[name].mask, pad, constant_values=False)
            padded_data = np.pad(self.data[name].data_vec, pad, constant_values=0.)
            padded_variance = np.pad(self.data[name].cov_mat.diagonal(), pad, constant_values=0.)
            padded_rp = np.pad(self.corr_items[name].rp_rt_grid[0], pad, constant_values=0.)
            padded_rt = np.pad(self.corr_items[name].rp_rt_grid[1], pad, constant_values=0.)
            padded_z = np.pad(self.corr_items[name].z_grid, pad, constant_values=0.)

            columns.append(fits.Column(name=name+'_MODEL', format='D', array=padded_cf))
            columns.append(fits.Column(name=name+'_MASK', format='L', array=padded_mask))
            columns.append(fits.Column(name=name+'_DATA', format='D', array=padded_data))
            columns.append(fits.Column(name=name+'_VAR', format='D', array=padded_variance))
            columns.append(fits.Column(name=name+'_RP', format='D', array=padded_rp))
            columns.append(fits.Column(name=name+'_RT', format='D', array=padded_rt))
            columns.append(fits.Column(name=name+'_Z', format='D', array=padded_z))

            if self.data[name].nb is not None:
                padded_nb = np.pad(self.data[name].nb, pad, constant_values=0)
                columns.append(fits.Column(name=name+'_NB', format='K', array=padded_nb))

        model_hdu = fits.BinTableHDU.from_columns(columns)
        model_hdu.name = 'Model'

        for name, size in sizes.items():
            card_name = 'hierarch ' + name + '_size'
            model_hdu.header[card_name] = size

        for par, val in params.items():
            card_name = 'hierarch ' + par
            model_hdu.header[card_name] = val

        return model_hdu

    def _bestfit_hdu(self, minimizer):
        """Create HDU with the bestfit info

        Parameters
        ----------
        minimizer : Minimizer
            Minimizer object after minimization was done

        Returns
        -------
        astropy.io.fits.hdu.table.BinTableHDU
            HDU with the bestfit data
        """
        # Get parameter names
        names = np.array(list(minimizer.values.keys()))

        # Check if any parameter name is too long
        max_length = np.max([len(name) for name in names])
        name_format = str(max_length) + 'A'

        # Get parameter values and errors
        values = np.array([minimizer.values[name] for name in names])
        errors = np.array([minimizer.errors[name] for name in names])
        num_pars = len(names)

        # Build the covariance matrix
        cov_mat = np.array(minimizer.covariance)

        cov_format = str(num_pars) + 'D'
        # Create the columns with the bestfit data
        col1 = fits.Column(name='names', format=name_format, array=names)
        col2 = fits.Column(name='values', format='D', array=values)
        col3 = fits.Column(name='errors', format='D', array=errors)
        col4 = fits.Column(name='covariance', format=cov_format, array=cov_mat)

        # Create the Table HDU from the columns
        bestfit_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
        bestfit_hdu.name = 'Bestfit'

        # Add all the attributes of the minimum to the header
        # for item, value in minimizer.fmin.items():
        #     name = item
        #     if len(item) > 8:
        #         name = 'hierarch ' + item
        bestfit_hdu.header['FVAL'] = minimizer.fmin.fval

        bestfit_hdu.header.comments['TTYPE1'] = 'Names of sampled parameters'
        bestfit_hdu.header.comments['TTYPE2'] = 'Bestfit values of sampled parameters'
        bestfit_hdu.header.comments['TTYPE3'] = 'Errors around the bestfit'
        bestfit_hdu.header.comments['TTYPE4'] = 'Covariance matrix around the bestfit'
        bestfit_hdu.header.comments['FVAL'] = 'Bestfit chi^2 value'

        return bestfit_hdu

    def _scan_hdu(self, scan_results, minimizer):
        """Create HDU with the scan info

        Parameters
        ----------
        scan_results : list, optional
            List of scan results, by default None
        names : list
            Parameter names
        name_format : string
            Format for writing parameter names to a fits file

        Returns
        -------
        astropy.io.fits.hdu.table.BinTableHDU
            HDU with the scan data
        """
        # Get parameter names
        names = np.array(list(minimizer.values.keys()))

        # Check if any parameter name is too long
        max_length = np.max([len(name) for name in names])
        name_format = str(max_length) + 'A'

        # Get list of parameters and results
        results = []
        for res in scan_results:
            results.append([res[par] for par in names])
        results = np.array(results)

        # Create the columns
        name_col = fits.Column(name='names', format=name_format, array=names)
        columns = [name_col]
        comms = []
        for col, name in zip(results.T, names):
            columns.append(fits.Column(name=name, format='D', array=col))
            comms.append('Bestfit grid values for ' + name)

        # Create the Table HDU from the columns
        scan_hdu = fits.BinTableHDU.from_columns(columns)
        scan_hdu.name = 'SCAN'

        # Add extra info to the header if we have the analysis object
        if self.analysis is not None:
            params = self.analysis.grids.keys()
            for par in params:
                grid = self.analysis.grids[par]
                scan_hdu.header[par + '_min'] = grid[0]
                scan_hdu.header[par + '_max'] = grid[-1]
                scan_hdu.header[par + '_num_bins'] = len(grid)
                scan_hdu.header.comments[par + '_min'] = 'Grid start for '+par
                scan_hdu.header.comments[par + '_max'] = 'Grid end for '+par
                scan_hdu.header.comments[par + '_num_bins'] = 'Grid size for '\
                                                              + par

        scan_hdu.header.comments['TTYPE1'] = 'Names of sampled parameters'
        for i, comm in enumerate(comms):
            scan_hdu.header.comments['TTYPE' + str(i+2)] = comm

        return scan_hdu

    def _pk_hdu(self, component, model):
        """Create HDU with Pk data for a component

        Parameters
        ----------
        component : string
            Name of component
        model : vega.Model
            Model object for the component

        Returns
        -------
        astropy.io.fits.hdu.table.BinTableHDU
            HDU with the Pk data for the component
        """
        # Get the Pk components and create the columns
        columns = self._get_components(model.pk)

        # Create the Table HDU from the columns
        pk_hdu = fits.BinTableHDU.from_columns(columns)
        pk_hdu.name = 'PK_' + component

        return pk_hdu

    def _cf_hdu(self, component, model):
        """Create HDU with correlation function data for a component

        Parameters
        ----------
        component : string
            Name of component
        model : vega.Model
            Model object for the component

        Returns
        -------
        astropy.io.fits.hdu.table.BinTableHDU
            HDU with the Xi data for the component
        """
        # Get the Xi components, before and after distortion
        columns = self._get_components(model.xi, name_prefix='raw_')
        columns += self._get_components(model.xi_distorted,
                                        name_prefix='distorted_')

        # Create the Table HDU from the columns
        cf_hdu = fits.BinTableHDU.from_columns(columns)
        cf_hdu.name = 'Xi_' + component

        return cf_hdu

    @staticmethod
    def _get_components(model_components, name_prefix=''):
        """Get the saved model components and create astropy Columns

        Parameters
        ----------
        model_components : dict
            Dictionary with saved Xi/Pk data
        name_prefix : str, optional
            Prefix for column names, by default ''

        Returns
        -------
        list
            List of astropy Columns for HDU creation
        """
        columns = []

        # Parts are smooth and/or peak
        for part, data in model_components.items():
            shape = np.shape(data['core'])
            if len(shape) == 1:
                form = 'D'
            else:
                size = shape[1]
                form = str(size) + 'D'

            for key, item in data.items():
                if key == 'core':
                    name = name_prefix + part + '_core'
                    columns.append(fits.Column(name=name, format=form,
                                               array=item))
                else:
                    name = name_prefix + part + '_' + key[0] + '_' + key[1]
                    columns.append(fits.Column(name=name, format=form,
                                               array=item))

        return columns

    def write_results_hdf(self, minimizer, scan_results=None):
        """Write Vega analysis results, including the bestfit
        and chi2 scan results if they exist.

        Parameters
        ----------
        minimizer : Minimizer
            Minimizer object after minimization was done
        scan_results : list, optional
            List of scan results, by default None
        """
        if minimizer is None:
            raise ValueError("The hdf output format is outdated and"
                             " does not work without minimization")
        h5_file = h5py.File(Path(self.outfile), 'w')

        # Write bestfit
        bf_group = h5_file.create_group("best fit")
        bf_group = self._write_bestfit_hdf(bf_group, minimizer)

        # Write scan results
        if scan_results is not None:
            scan_group = h5_file.create_group("chi2 scan")
            scan_group = self._write_scan_hdf(scan_group, scan_results)

        h5_file.close()

    @staticmethod
    def _write_bestfit_hdf(bf_group, minimizer):
        # Write the parameters values
        for param, value in minimizer.values.items():
            error = minimizer.errors[param]
            bf_group.attrs[param] = (value, error)

        # Write the covariance
        for (par1, par2), cov in minimizer.covariance.items():
            bf_group.attrs["cov[{}, {}]".format(par1, par2)] = cov

        # Write down all attributes of the minimum
        for item, value in minimizer.fmin.items():
            bf_group.attrs[item] = value

        return bf_group

    @staticmethod
    def _write_scan_hdf(scan_group, scan_results):
        # Get list of parameters and results
        params = list(scan_results[0].keys())
        results = []
        for res in scan_results:
            results.append([res[par] for par in params])
        results = np.array(results)

        # Write parameter indeces
        for i, par in enumerate(params):
            scan_group.attrs[par] = i

        # Write results
        values = scan_group.create_dataset("values", np.shape(results),
                                           dtype="f")
        values[...] = results
        return scan_group
