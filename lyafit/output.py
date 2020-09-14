from pathlib import Path
import numpy as np
import h5py


class Output:
    """Class for handling the lyafit output,
    and reading/writing output files
    """
    def __init__(self, config):
        self.outfile = Path(config['filename'])

    def write_results(self, minimizer, scan_results=None):
        h5_file = h5py.File(self.outfile, 'w')

        # Write bestfit
        bf_group = h5_file.create_group("best fit")
        bf_group = self.write_bestfit(bf_group, minimizer)

        # Write scan results
        if scan_results is not None:
            scan_group = h5_file.create_group("chi2 scan")
            scan_group = self.write_scan(scan_group, scan_results)

        h5_file.close()

    @staticmethod
    def write_bestfit(bf_group, minimizer):
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
    def write_scan(scan_group, scan_results):
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
