import numpy as np
from scipy.special import legendre


class XiMultipoles():
    @classmethod
    def getOverfit(
            cls, rdata, mudata, xi_data, cov_data, mumax=0.999, nlmin=6, nlmax=20, chi_c=0.8
    ):
        """ Find the nl that overfits the data such that chi2 < chi_c.
        """
        for nl in range(nlmin, nlmax):
            xi_multipoles = cls(nl=nl)
            chi2 = xi_multipoles.fit_WLS(rdata, mudata, xi_data, cov_data, mumax=mumax)
            if chi2 < chi_c:
                print("Converged for nl=", nl)
                break

        return xi_multipoles

    def __init__(self, r1=42.0, r2=178.0, nr=35, nl=6):
        self.rgrid, dr = np.linspace(r1, r2, nr, retstep=True)
        self.rmin, self.rmax = r1 - dr / 2, r2 + dr / 2
        self.dr = dr
        self.nr = nr
        self.nl = nl
        self.leg_polys = [legendre(2 * ell) for ell in range(self.nl)]
        self.xi_knots_fit = np.zeros((nl, nr))
        self.std_xi_mc = np.zeros((nl, nr))

    def __call__(self, r, mu, xi_knots=None):
        # assert r.size == mu.size
        if xi_knots is None:
            xi_knots = self.xi_knots_fit
        # assert xi_knots.shape == (self.nl, self.nr)

        results = np.zeros_like(r)
        for ell in range(self.nl):
            results += np.interp(
                r, self.rgrid, xi_knots[ell]
            ) * self.leg_polys[ell](mu)

        return results

    def fit(self, rdata, mudata, xi_data, cov_data, mumax=0.999, nmc=0, seed=0, return_mcs=False):
        w = (self.rmin < rdata) & (rdata < self.rmax)
        w &= mudata < mumax
        rr = rdata[w]
        mumu = mudata[w]
        xi_in = xi_data[w].copy()
        cov_in = cov_data[w, :][:, w]
        invcov = np.linalg.inv(cov_in)

        Nell = self.nr * self.nl
        Nr = rr.size
        matrix_A = np.zeros((Nr, Nell))
        idx = ((rr - self.rmin) / self.dr).astype(int)

        for i in range(Nr):
            d = (rr[i] - self.rgrid[0]) / self.dr
            n1 = min(self.nr - 1, int(d))
            d -= n1
            if d > 0:
                n2 = min(self.nr - 1, n1 + 1)
            else:
                n2 = max(0, n1 - 1)

            for ell in range(self.nl):
                matrix_A[i, n1 + ell * self.nr] += (1.0 - d) * self.leg_polys[ell](mumu[i])
                matrix_A[i, n2 + ell * self.nr] += d * self.leg_polys[ell](mumu[i])

        AtW = matrix_A.T.dot(invcov)
        AtWA = AtW.dot(matrix_A)
        solver = np.linalg.inv(AtWA).dot(AtW)
        self.xi_knots_fit = solver.dot(xi_in).reshape(self.nl, self.nr)
        self.xi_knots_cov = solver.dot(cov_in.dot(solver.T))
        self.std_xi_mc = np.sqrt(
            self.xi_knots_cov.diagonal()).reshape(self.nl, self.nr)

        d = xi_in - self(rr, mumu)
        chi2 = d.dot(invcov.dot(d)) / d.size
        if nmc <= 0:
            return chi2

        rng = np.random.default_rng(seed)
        randoms = rng.multivariate_normal(
            np.zeros(rdata.size), cov=cov_data, size=nmc
        )[:, w]
        xi_ins = xi_in + randoms
        xi_knots_mc = solver.dot(xi_ins.T).T

        self.std_xi_mc = np.std(xi_knots_mc, axis=0).reshape(
            self.nl, self.nr)

        if return_mcs:
            return chi2, xi_knots_mc

        return chi2
        
    def fit_mini(self, rdata, mudata, xi_data, cov_data, nmc=20, seed=1, return_mcs=False):
        """ Uses the minimizer. Slow """
        from scipy.optimize import minimize
        w = (self.rmin < rdata) & (rdata < self.rmax)
        rr = rdata[w]
        mumu = mudata[w]
        xi_in = xi_data[w].copy()
        invcov = np.linalg.inv(cov_data[w, :][:, w])

        def _cost(xi_knots):
            xi = self(rr, mumu, xi_knots.reshape(self.nl, self.nr))
            diff = xi - xi_in
            return diff.dot(invcov.dot(diff))

        answer = minimize(_cost, x0=self.xi_knots_fit.ravel(),
                          method='L-BFGS-B')
        self.xi_knots_fit = answer.x.reshape(self.nl, self.nr)

        if nmc <= 0:
            return self.xi_knots_fit, np.zeros((self.nl, self.nr))

        xi_knots_mc = np.zeros((nmc, self.xi_knots_fit.size))
        xi_data_org = xi_in.copy()
        rng = np.random.default_rng(seed)
        randoms = rng.multivariate_normal(
            np.zeros(rdata.size), cov=cov_data, size=nmc
        )[:, w]

        for _ in range(nmc):
            xi_in = xi_data_org + randoms[_]
            answer = minimize(_cost, x0=self.xi_knots_fit.ravel(),
                              method='L-BFGS-B')
            xi_knots_mc[_] = answer.x

        self.std_xi_mc = np.std(xi_knots_mc, axis=0).reshape(
            self.nl, self.nr)

        if return_mcs:
            return self.xi_knots_fit, self.std_xi_mc, xi_knots_mc

        return self.xi_knots_fit, self.std_xi_mc
