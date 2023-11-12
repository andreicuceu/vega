import iminuit
import time
import copy
from sys import stdout


class Minimizer:
    """Class for handling the interface to the minimizer.
    """
    def __init__(self, chi2_func, sample_params, cf_names):
        """

        Parameters
        ----------
        chi2_func : function
            Function that takes dictionary of params and returns a chi^2 value
        sample_params : dict
            Dictionary with the sample params config
        """
        self.chi2_func = chi2_func
        self._cf_names = cf_names
        self._names = sample_params['limits'].keys()
        self._sample_params = sample_params
        self._config = {}
        self.params_init = copy.deepcopy(self._sample_params['values'])
        self.run_errors = copy.deepcopy(self._sample_params['errors'])
        self.limits = copy.deepcopy(self._sample_params['limits'])
        self.fixed = copy.deepcopy(self._sample_params['fix'])

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

    def run_iminuit(self, params_to_sample):
        minuit = iminuit.Minuit(self.chi2, name=self._names, **self.params_init)
        for name in self._names:
            minuit.errors[name] = self.run_errors[name]
            minuit.limits[name] = self.limits[name]
            minuit.fixed[name] = self.fixed[name]

        for name in self._names:
            if name not in params_to_sample:
                minuit.fixed[name] = True

        minuit.errordef = 1
        minuit.print_level = 1
        minuit.migrad()
        print(minuit.fmin)
        print(minuit.params)

        return minuit

    def minimize(self, params=None):
        """Minimize the chi2.

        Parameters
        ----------
        params : dict, optional
            Dictionary of sample parameters, used to change starting value
            and/or fix parameters, by default None
        """
        t0 = time.time()

        if params is not None:
            if 'values' in params:
                self.params_init |= params['values']
            if 'errors' in params:
                self.run_errors |= params['errors']
            if 'limits' in params:
                self.limits |= params['limits']
            if 'fix' in params:
                self.fixed |= params['fix']

        # Do an initial "fast" minimization over biases
        bias_params = [par for par in self._names if 'bias' in par]
        if bool(len(bias_params)):
            minuit_biases = self.run_iminuit(bias_params)

            for param, value in minuit_biases.values.to_dict().items():
                self.params_init[param] = value

        # If we have broadband polynomials, we first maximize one correlation at a time
        bb_params = [par for par in self._names if 'BB-' in par]
        if bool(len(bb_params)):
            for cf_name in self._cf_names:
                cf_bb_params = [par for par in bb_params if cf_name in par]
                minuit_bb = self.run_iminuit(cf_bb_params)

                for param, value in minuit_bb.values.to_dict().items():
                    self.params_init[param] = value

        # Do the actual minimization
        self._minuit = self.run_iminuit(self._names)

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
