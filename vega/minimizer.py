import iminuit
import time
import copy
from sys import stdout


class Minimizer:
    """Class for handling the interface to the minimizer.
    """
    def __init__(self, chi2_func, sample_params):
        """

        Parameters
        ----------
        chi2_func : function
            Function that takes dictionary of params and returns a chi^2 value
        sample_params : dict
            Dictionary with the sample params config
        """
        self.chi2_func = chi2_func
        self._names = sample_params['limits'].keys()
        self._sample_params = sample_params
        self._config = {}

        self._run_flag = False

    def chi2(self, *pars):
        """Wrapper of chi2 function for iminuit.

        Returns
        -------
        float
            chi^2
        """
        sample_params = {par: pars[i] for i, par in enumerate(self._names)}
        return self.chi2_func(sample_params)

    def minimize(self, params=None):
        """Minimize the chi2.

        Parameters
        ----------
        params : dict, optional
            Dictionary of sample parameters, used to change starting value
            and/or fix parameters, by default None
        """
        t0 = time.time()

        params_init = copy.deepcopy(self._sample_params['values'])
        errors = copy.deepcopy(self._sample_params['errors'])
        limits = copy.deepcopy(self._sample_params['limits'])
        fixed = copy.deepcopy(self._sample_params['fix'])

        def write_settings(params, name, out_container):
            if name in params:
                for par, val in params[name].items():
                    out_container[par] = val

        if params is not None:
            write_settings(params, 'values', params_init)
            write_settings(params, 'errors', errors)
            write_settings(params, 'limits', limits)
            write_settings(params, 'fix', fixed)

        # Do an initial "fast" minimization over biases
        bias_flag = bool(len([par for par in self._names if 'bias' in par]))
        if bias_flag:
            mig_init = iminuit.Minuit(self.chi2, name=self._names, **params_init)
            for name in self._names:
                mig_init.errors[name] = errors[name]
                mig_init.limits[name] = limits[name]
                mig_init.fixed[name] = fixed[name]

            for name in self._names:
                if 'bias' not in name:
                    mig_init.fixed[name] = True

            mig_init.errordef = 1
            mig_init.print_level = 1
            mig_init.migrad()
            print(mig_init.fmin)
            print(mig_init.params)

            for param, value in mig_init.values.to_dict().items():
                params_init[param] = value

        # Do the actual minimization
        self._minuit = iminuit.Minuit(self.chi2, name=self._names, **params_init)
        for name in self._names:
            self._minuit.errors[name] = errors[name]
            self._minuit.limits[name] = limits[name]
            self._minuit.fixed[name] = fixed[name]

        self._minuit.errordef = 1
        self._minuit.print_level = 1
        self._minuit.migrad()
        print(self._minuit.fmin)
        print(self._minuit.params)

        print("INFO: minimized in {}".format(time.time()-t0))
        stdout.flush()
        self._run_flag = True

    @property
    def params(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results before minimization.')
        return self._minuit.params

    @property
    def values(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results before minimization.')
        return dict(self._minuit.values.to_dict())

    @property
    def errors(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results before minimization.')
        return dict(self._minuit.errors.to_dict())

    @property
    def covariance(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results before minimization.')
        return self._minuit.covariance

    @property
    def fmin(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results before minimization.')
        return self._minuit.fmin

    @property
    def minuit(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results before minimization.')
        return self._minuit
