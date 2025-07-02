from pathlib import Path
import os.path
from astropy.io import fits
import numpy as np
import h5py


class Output:
    """Class for handling the Vega output,
    and reading/writing output files.
    """
    def __init__(self, config, data, corr_items, analysis=None, percival=1):
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
        self.mc_output = config.get('mc_output', None)
        self.percival = percival

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

    @staticmethod
    def pad_array(array, size_to_match, pad_value=np.nan):
        return np.pad(array, (0, size_to_match - len(array)), constant_values=pad_value)

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
        columns = []
        for name, cf in corr_funcs.items():
            if len(self.data[name].data_vec) > num_rows:
                raise ValueError('Data coordinate grid is larger than the model grid.')

            columns.append(fits.Column(
                name=name+'_MODEL', format='D', array=self.pad_array(cf, num_rows)))
            columns.append(fits.Column(
                name=name+'_MODEL_MASK', format='L',
                array=self.pad_array(self.data[name].model_mask, num_rows, False)
            ))
            columns.append(fits.Column(
                name=name+'_MASK', format='L',
                array=self.pad_array(self.data[name].data_mask, num_rows, False)
            ))
            columns.append(fits.Column(
                name=name+'_DATA', format='D',
                array=self.pad_array(self.data[name].data_vec, num_rows)
            ))
            columns.append(fits.Column(
                name=name+'_VAR', format='D',
                array=self.pad_array(self.data[name].cov_mat.diagonal(), num_rows)
            ))

            if not self.corr_items[name].use_multipoles:
                columns.append(fits.Column(
                    name=name+'_RP', format='D',
                    array=self.pad_array(self.corr_items[name].dist_model_coordinates.rp_grid, num_rows)
                ))
                columns.append(fits.Column(
                    name=name+'_RT', format='D',
                    array=self.pad_array(self.corr_items[name].dist_model_coordinates.rt_grid, num_rows)
                ))
            else:
                nmu = self.corr_items[name].dist_model_coordinates.mu_nbins
                nr = self.corr_items[name].dist_model_coordinates.r_nbins
                ells = np.repeat(self.corr_items[name].ells_to_model, nr)
                rmodel = self.corr_items[name].dist_model_coordinates.r_grid.reshape(
                    nmu, nr).mean(0)
                rmodel = np.tile(rmodel, len(self.corr_items[name].ells_to_model))

                columns.append(fits.Column(
                    name=name+'_RP', format='K',
                    array=self.pad_array(ells, num_rows)
                ))
                columns.append(fits.Column(
                    name=name+'_RT', format='D',
                    array=self.pad_array(rmodel, num_rows)
                ))

            if num_rows < self.corr_items[name].model_coordinates.z_grid.size:
                columns.append(fits.Column(name=name+'_Z', format='D', array=np.zeros(num_rows)))
            else:
                columns.append(fits.Column(
                    name=name+'_Z', format='D',
                    array=self.pad_array(self.corr_items[name].model_coordinates.z_grid, num_rows)
                ))

            if self.data[name].nb is not None:
                columns.append(fits.Column(name=name+'_NB', format='K',
                                           array=self.pad_array(self.data[name].nb, num_rows)))

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
        errors = np.sqrt(self.percival) * np.array([minimizer.errors[name] for name in names])
        num_pars = len(names)

        # Build the covariance matrix
        cov_mat = self.percival * np.array(minimizer.covariance)

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
        bestfit_hdu.header['FVAL'] = minimizer.fmin.fval
        bestfit_hdu.header['VALID'] = minimizer.minuit.valid
        bestfit_hdu.header['ACCURATE'] = minimizer.minuit.accurate
        bestfit_hdu.header['PERCIVAL'] = self.percival

        bestfit_hdu.header.comments['TTYPE1'] = 'Names of sampled parameters'
        bestfit_hdu.header.comments['TTYPE2'] = 'Bestfit values of sampled parameters'
        bestfit_hdu.header.comments['TTYPE3'] = 'Errors around the bestfit'
        bestfit_hdu.header.comments['TTYPE4'] = 'Covariance matrix around the bestfit'
        bestfit_hdu.header.comments['FVAL'] = 'Bestfit chi^2 value'
        bestfit_hdu.header.comments['VALID'] = 'Flag for valid fit'
        bestfit_hdu.header.comments['ACCURATE'] = 'Flag for accurate fit'
        bestfit_hdu.header.comments['PERCIVAL'] = 'Applied to errors and covariance'

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
        names = np.array(list(scan_results[0].keys()))

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
                scan_hdu.header.comments[par + '_min'] = 'Grid start for ' + par
                scan_hdu.header.comments[par + '_max'] = 'Grid end for ' + par
                scan_hdu.header.comments[par + '_num_bins'] = 'Grid size for ' + par

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
            if not data:
                continue
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

    def write_monte_carlo(self, cpu_id=None):
        assert self.analysis is not None
        assert self.analysis.has_monte_carlo

        primary_hdu = fits.PrimaryHDU()
        hdu_list = [primary_hdu]

        bestfits = self.analysis.mc_bestfits
        covariances = np.array(self.analysis.mc_covariances)

        if not bestfits:
            print('No MC bestfit data to write.')
        else:
            names = np.array(list(bestfits.keys()))
            bestfit_table = np.array([bestfits[name][:, 0] for name in names])
            errors_table = np.array([bestfits[name][:, 1] for name in names])
            covariances = covariances.reshape(bestfit_table.shape[1]*len(names), len(names)).T

            # Get the data types for the columns
            max_length = np.max([len(name) for name in names])
            name_format = str(max_length) + 'A'
            fit_format = f'{bestfit_table.shape[1]}D'
            cov_format = f'{covariances.shape[1]}D'

            # Create the columns with the bestfit data
            col1 = fits.Column(name='names', format=name_format, array=names)
            col2 = fits.Column(name='values', format=fit_format, array=bestfit_table)
            col3 = fits.Column(name='errors', format=fit_format, array=errors_table)
            col4 = fits.Column(name='covariance', format=cov_format, array=covariances)

            # Create the Table HDU from the columns
            bestfit_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
            bestfit_hdu.name = 'Bestfit'
            hdu_list += [bestfit_hdu]

            # Create the columns with the fit information
            col1 = fits.Column(name='chisq', format='D', array=self.analysis.mc_chisq)
            col2 = fits.Column(name='valid_minima', format='L', array=self.analysis.mc_valid_minima)
            col3 = fits.Column(name='valid_hesse', format='L', array=self.analysis.mc_valid_hesse)
            col4 = fits.Column(name='failed_mask', format='L', array=self.analysis.mc_failed_mask)

            # Create the Table HDU from the columns
            fitinfo_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
            fitinfo_hdu.name = 'FitInfo'
            hdu_list += [fitinfo_hdu]

        mocks = self.analysis.mc_mocks
        columns = []
        for name in mocks.keys():
            table = np.array(mocks[name])
            columns.append(fits.Column(name=name, format=f'{table.shape[1]}D', array=table))

        mocks_hdu = fits.BinTableHDU.from_columns(columns)
        mocks_hdu.name = 'Mocks'
        hdu_list += [mocks_hdu]

        hdul = fits.HDUList(hdu_list)
        if self.mc_output is None:
            dir_path = Path(self.outfile).parent / 'monte_carlo'
        else:
            dir_path = Path(self.mc_output)
        dir_path.mkdir(parents=True, exist_ok=True)
        if cpu_id is None:
            filepath = dir_path / 'monte_carlo.fits'
        else:
            filepath = dir_path / f'monte_carlo_{cpu_id}.fits'

        hdul.writeto(filepath, overwrite=self.overwrite)

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
