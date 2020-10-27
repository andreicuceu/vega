from pathlib import Path
# from vega.minimizer import Minimizer
from astropy.io import fits
import numpy as np
import h5py


class Output:
    """Class for handling the Vega output,
    and reading/writing output files.
    """
    def __init__(self, config):
        """

        Parameters
        ----------
        config : ConfigParser
            Output section of main config file
        """
        self.type = config.get('type', 'fits')
        self.outfile = config['filename']

    def write_results(self, minimizer, scan_results=None):
        """Write results in the fits or hdf format

        Parameters
        ----------
        minimizer : Minimizer
            Minimizer object after minimization was done
        scan_results : list, optional
            List of scan results, by default None
        """
        if self.type == 'fits':
            self.write_results_fits(minimizer, scan_results)
        elif self.type == 'hdf' or self.type == 'h5':
            self.write_results_hdf(minimizer, scan_results)
        else:
            raise ValueError('Unknown output type. Set type = fits \
                             or type = hdf')

    def write_results_fits(self, minimizer, scan_results=None):
        primary_hdu = fits.PrimaryHDU()

        bestfit_hdu = self._bestfit_hdu(minimizer)

        hdul = fits.HDUList([primary_hdu, bestfit_hdu])

        if self.outfile[-5:] != '.fits':
            self.outfile += '.fits'

        hdul.writeto(Path(self.outfile))

    def _bestfit_hdu(self, minimizer):
        # Get parameter names, values and errors
        names = np.array(list(minimizer.values.keys()))
        values = np.array([minimizer.values[name] for name in names])
        errors = np.array([minimizer.errors[name] for name in names])
        num_pars = len(names)

        # Build the covariance matrix
        cov_mat = np.zeros((num_pars, num_pars))
        for i, par1 in enumerate(names):
            for j, par2 in enumerate(names):
                cov_mat[i, j] = minimizer.covariance[(par1, par2)]

        # Check if any parameter name is too long
        max_length = 20  # Increase this if you have names longer than 20 chars
        lengths = np.array([len(name) for name in names])
        if (lengths > max_length).any():
            raise ValueError('The current maximum allowed number of chars in \
                a parameter name is 10. Change the output module if you \
                need more. [_bestfit_hdu function in the Output class]')
        name_format = str(max_length) + 'A'
        cov_format = str(num_pars) + 'D'

        # Create the columns with the bestfit data
        col1 = fits.Column(name='names', format=name_format, array=names)
        col2 = fits.Column(name='values', format='D', array=values)
        col3 = fits.Column(name='errors', format='D', array=errors)
        col4 = fits.Column(name='covariance', format=cov_format, array=cov_mat)

        # Create the Table HDU from the columns
        bestfit_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])

        # Add all the attributes of the minimum to the header
        for item, value in minimizer.fmin.items():
            print(item)
            bestfit_hdu.header[item] = value

        return bestfit_hdu

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
