import iminuit
import time
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
        for param in self._names:
            self._config[param] = sample_params['values'][param]
            self._config['error_' + param] = sample_params['errors'][param]
            self._config['limit_' + param] = sample_params['limits'][param]
            self._config['fix_' + param] = sample_params['fix'][param]

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
        kwargs = self._config.copy()
        if params is not None:
            for param, val in params['values'].items():
                kwargs[param] = val
                kwargs['fix_' + param] = params['fix'][param]

        # Do an initial "fast" minimization over biases
        bias_flag = bool(len([par for par in self._names if 'bias' in par]))
        if bias_flag:
            kwargs_init = kwargs.copy()
            for param in self._names:
                if 'bias' not in param:
                    kwargs_init['fix_' + param] = True

            minuit_init = iminuit.Minuit(self.chi2,
                                         forced_parameters=self._names,
                                         errordef=1, print_level=1,
                                         **kwargs_init)
            minuit_init.migrad()
            minuit_init.print_param()

            for param, value in minuit_init.values.items():
                kwargs[param] = value

        # Do the actual minimization
        self._minuit = iminuit.Minuit(self.chi2, forced_parameters=self._names,
                                      errordef=1, print_level=1, **kwargs)
        self._minuit.migrad()
        self._minuit.print_param()

        print("INFO: minimized in {}".format(time.time()-t0))
        stdout.flush()
        self._run_flag = True

    @property
    def params(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results'
                               ' before minimization.')
        return self._minuit.params

    @property
    def values(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results'
                               ' before minimization.')
        return dict(self._minuit.values)

    @property
    def errors(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results'
                               ' before minimization.')
        return dict(self._minuit.errors)

    @property
    def covariance(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results'
                               ' before minimization.')
        return dict(self._minuit.covariance)

    @property
    def fmin(self):
        if not self._run_flag:
            print('Run Minimizer.minimize() before asking for results')
            raise RuntimeError('Tried to access minimization results'
                               ' before minimization.')
        return self._minuit.fmin
