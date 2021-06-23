import numpy as np
from numpy import fft
from scipy import special
from scipy import interpolate
from mcfit import P2xi


class PktoXi:
    """Transform a 2D power spectrum to a correlation function
    """
    def __init__(self, k_grid, muk_grid, ell_max=6, old_fftlog=False):
        """Initialize the FFTLog and the Legendre polynomials

        Parameters
        ----------
        k : 1D Array
            Wavenumber grid of power spectrum
        muk : ND Array
            k_parallel / k grid for input power spectrum
        ell_max : int, optional
            Maximum multipole to sum over, by default 6
        """
        self.k_grid = k_grid
        self.muk_grid = muk_grid
        self.dmuk = 1 / len(muk_grid)
        self.ell_max = ell_max
        self._old_fftlog = old_fftlog

        # Initialize the multipole values we will need (only even ells)
        self.ell_vals = np.arange(0, ell_max + 1, 2)

        # Initialize FFTLog objects and Legendre polynomials for each multipole
        self.fftlog_objects = {}
        self.legendre_pk = {}
        self.legendre_xi = {}
        for ell in self.ell_vals:
            if not self._old_fftlog:
                self.fftlog_objects[ell] = P2xi(k_grid, l=ell, lowring=True)
            # Precompute the Legendre polynomials used to decompose Pk into Pk_ell
            self.legendre_pk[ell] = special.legendre(ell)(self.muk_grid)
            # We don't know the mu grid for Xi in advance, so just initialize
            self.legendre_xi[ell] = special.legendre(ell)

    def compute(self, r_grid, mu_grid, pk, single_ell=-1):
        """Compute the correlation function from an input Pk
        over the r/mu coordinate grid.
        If valid single_ell is passed, returns the relevant multipole (Xi_ell) only

        Parameters
        ----------
        r_grid : 1D Array
            Grid of r coordinates for the output correlation
        mu_grid : 1D Array
            Grid of mu coordinates for the output correlation
        pk : ND Array
            Input power spectrum
        single_ell : int, optional
            If set to an integer >= 0, returns the relevant multipole, by default -1

        Returns
        -------
        1D Array
            Output correlation function
        """
        if self._old_fftlog:
            return self.pk_to_xi(r_grid, mu_grid, pk, single_ell)

        # Check if we need only one multipole
        ell_vals = self.ell_vals
        if not single_ell < 0:
            assert type(single_ell) is int, "You need to pass an integer"
            ell_vals = [single_ell]

        # Initialize the Xi_ell array
        xi_ell_arr = np.zeros([len(ell_vals), len(r_grid)])
        for ell in ell_vals:
            # Compute the Pk_ell multipole
            pk_ell = np.sum(self.dmuk * self.legendre_pk[ell] * pk, axis=0) * (2 * ell + 1)

            # Compute the FFTLog to transform Pk_ell to Xi_ell
            r_fft, xi_fft = self.fftlog_objects[ell](pk_ell)

            # Interpolate to r grid
            xi_interp = interpolate.interp1d(np.log(r_fft), xi_fft, kind='cubic')

            # Check for nans and get the model correlation
            mask = r_grid != 0
            xi_ell = np.zeros(len(r_grid))
            xi_ell[mask] = xi_interp(np.log(r_grid[mask]))

            # If only one multipole was required we are done
            if not single_ell < 0:
                return xi_ell

            # Add the Legendre polynomials
            xi_ell_arr[ell//2, :] = xi_ell * self.legendre_xi[ell](mu_grid)

        # Sum over the multipoles
        full_xi = np.sum(xi_ell_arr, axis=0)
        return full_xi

    @staticmethod
    def Pk2Mp(ar, k, pk, ell_vals, muk, dmuk, tform=None):
        """
        This function is outdated and will be removed
        Implementation of FFTLog from A.J.S. Hamilton (2000)
        assumes log(k) are equally spaced
        """

        k0 = k[0]
        l = np.log(k.max()/k0)
        r0 = 1.

        N = len(k)
        emm = N * fft.fftfreq(N)
        r = r0*np.exp(-emm*l/N)
        dr = abs(np.log(r[1]/r[0]))
        s = np.argsort(r)
        r = r[s]

        xi = np.zeros([len(ell_vals), len(ar)])

        for ell in ell_vals:
            if tform == "rel":
                pk_ell = pk
                n = 1.
            elif tform == "asy":
                pk_ell = pk
                n = 2.
            else:
                pk_ell = np.sum(dmuk*special.legendre(ell)(muk)*pk, axis=0)*(2*ell+1)
                pk_ell *= (-1)**(ell//2)/2/np.pi**2
                n = 2.
            mu = ell+0.5
            q = 2-n-0.5
            x = q+2*np.pi*1j*emm/l
            lg1 = special.loggamma((mu+1+x)/2)
            lg2 = special.loggamma((mu+1-x)/2)

            um = (k0*r0)**(-2*np.pi*1j*emm/l)*2**x*np.exp(lg1-lg2)
            um[0] = np.real(um[0])
            an = fft.fft(pk_ell*k**n*np.sqrt(np.pi/2))
            an *= um
            xi_loc = fft.ifft(an)
            xi_loc = xi_loc[s]
            xi_loc /= r**(3-n)
            xi_loc[-1] = 0
            spline = interpolate.splrep(np.log(r)-dr/2, np.real(xi_loc), k=3, s=0)
            xi[ell//2, :] = interpolate.splev(np.log(ar), spline)

        return xi

    def pk_to_xi(self, r_grid, mu_grid, pk, multipole=-1):
        """This function is outdated and will be removed
        Compute the correlation function from an input power spectrum

        Parameters
        ----------
        r_grid : 1D Array
            Grid of r coordinates for the output correlation
        mu_grid : 1D Array
            Grid of mu coordinates for the output correlation
        pk : ND Array
            Input power spectrum
        ell_max : int
            Maximum multipole to sum over
        multipole : int
            If set, returns the single multipole

        Returns
        -------
        1D Array
            Output correlation function
        """
        ell_vals = self.ell_vals
        # Check what multipoles we need and compute them
        if not multipole < 0:
            assert type(multipole) is int
            ell_vals = [multipole]

        xi = self.Pk2Mp(r_grid, self.k_grid, pk, ell_vals, self.muk_grid, self.dmuk)

        # Add the Legendre polynomials and sum over the multipoles
        if multipole < 0:
            for ell in ell_vals:
                xi[ell//2, :] *= self.legendre_xi[ell](mu_grid)
            full_xi = np.sum(xi, axis=0)
        else:
            full_xi = xi[multipole//2]

        return full_xi

    def pk_to_xi_relativistic(self, r_grid, mu_grid, pk, params):
        """Calculate the cross-correlation contribution from
        relativistic effects (Bonvin et al. 2014).

        Parameters
        ----------
        r_grid : 1D Array
            Grid of r coordinates for the output correlation
        mu_grid : 1D Array
            Grid of mu coordinates for the output correlation
        pk : ND Array
            Input power spectrum
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output xi relativistic
        """
        # Compute the dipole and octupole terms
        ell_vals = [1, 3]
        xi = self.Pk2Mp(r_grid, self.k_grid, pk, ell_vals, self.muk_grid, self.dmuk, tform='rel')

        # Get the relativistic parameters and sum over the monopoles
        A_rel_1 = params['Arel1']
        A_rel_3 = params['Arel3']
        xi_rel = A_rel_1 * xi[1//2, :] * special.legendre(1)(mu_grid)
        xi_rel += A_rel_3 * xi[3//2, :] * special.legendre(3)(mu_grid)
        return xi_rel

    def pk_to_xi_asymmetry(self, r_grid, mu_grid, pk, params):
        """Calculate the cross-correlation contribution from
        standard asymmetry (Bonvin et al. 2014).

        Parameters
        ----------
        r_grid : 1D Array
            Grid of r coordinates for the output correlation
        mu_grid : 1D Array
            Grid of mu coordinates for the output correlation
        pk : ND Array
            Input power spectrum
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output xi asymmetry
        """
        # Compute the monopole and quadrupole terms
        ell_vals = [0, 2]
        xi = self.Pk2Mp(r_grid, self.k_grid, pk, ell_vals, self.muk_grid, self.dmuk, tform='asy')

        # Get the asymmetry parameters and sum over the monopoles
        A_asy_0 = params['Aasy0']
        A_asy_2 = params['Aasy2']
        A_asy_3 = params['Aasy3']
        xi_asy = (A_asy_0 * xi[0, :] - A_asy_2 * xi[1, :]) * r_grid * special.legendre(1)(mu_grid)
        xi_asy += A_asy_3 * xi[1, :] * r_grid * special.legendre(3)(mu_grid)
        return xi_asy
