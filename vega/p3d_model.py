import numpy as np
import scipy as sp
from scipy import special


class P3DModel:
    """Model for 3D Power Spectrum multipoles as computed by https://arxiv.org/pdf/2403.08241

    # ! Slow operations should be kept in init as that is only called once

    # ! Compute is called many times and should be fast

    Extensions should have their separate method of the form
    'compute_extension' that can be called from outside
    """
    def __init__(self, config, pk_coordinates, window_path):
        self.config = config

        window_matrices = np.loadtxt(window_path).T
        # TODO Check that the window matrices make sense
        self.r_window_grid = window_matrices[0]
        window_multipoles = np.arange(0, 2*window_matrices[1:].shape[0], 2)
        self.window_matrices = {
            ell: window_matrices[i+1] for i, ell in enumerate(window_multipoles)}

        num_mu_bins = config.getint('p3d_num_mu_bins', 100)
        self.dmu = 1 / num_mu_bins
        dr = config.getfloat('p3d_dr', 2)
        r_min = config.getfloat('p3d_r_min', 0)
        r_max = config.getfloat('p3d_r_max', 200)
        r_max_window = config.getfloat('p3d_r_max_window', r_max)

        self.r_regular_grid = np.arange(r_min + dr/2, r_max, dr)
        self.mu_regular_grid = np.arange(self.dmu / 2, 1, self.dmu)
        r_grid, mu_grid = np.meshgrid(self.r_regular_grid, self.mu_regular_grid)

        self.rp_interp = r_grid * mu_grid
        self.rt_interp = r_grid * np.sqrt(1 - mu_grid**2)

        self.multipoles = pk_coordinates.model_multipoles
        self.legendre_xi = {}
        for ell in self.multipoles:
            self.legendre_xi[ell] = special.legendre(ell)

        self.window_rfunc = np.vectorize(self.scalar_window_rfunc)
        self.window_r = self.window_rfunc(self.r_regular_grid / r_max_window)

        # TODO add more options here
        self.sph_kernels = self.compute_sph_kernels(
            pk_coordinates.k_edges, pk_coordinates.k_centers, self.r_window_grid)

    def compute(self, xi2d_model, xi2d_coords):
        # Decompose the 2D correlation function into multipoles
        xiell = self.xi2d_to_xiell(xi2d_model, xi2d_coords)

        # Interpolate onto the window r grid
        xiell = np.array([np.interp(self.r_window_grid, self.r_regular_grid, xi) for xi in xiell])

        # Add window matrix and pair separation window function
        r2xiell_windowed = self.add_window(xiell)

        # FHT to get the 3D power spectrum multipoles
        # TODO Implement alternative with mcfit or hankl
        pkell = []
        for ell in self.multipoles:
            pkell_integrand = np.outer(r2xiell_windowed[ell], self.sph_kernels[ell])
            pkell.append(np.sum(pkell_integrand, axis=0))

        return np.array(pkell)

    def xi2d_to_xiell(self, xi2d_model, xi2d_coords):
        # TODO: xi2d_model needs to be reshaped

        # TODO: This interpolation needs proper testing and validation
        interp_xi2d = sp.interpolate.RegularGridInterpolator(
            (xi2d_coords.rp_regular_grid, xi2d_coords.rt_regular_grid), xi2d_model,
            method='quintic', bounds_error=False, fill_value=None)

        xi2d_rmu = interp_xi2d((self.rp_interp, self.rt_interp))
        xiell = np.zeros((self.multipoles.size, self.r_regular_grid.size))
        for i, ell in enumerate(self.multipoles):
            xiell[i] = np.sum(
                self.dmu * self.legendre_xi[ell](self.mu_regular_grid) * xi2d_rmu.T, axis=1) \
                * (2 * ell + 1)

        return xiell

    def add_window(self, xiell):
        # r^2 Xi_ell * Phi_ell products. First index is ell Xi and second is ell phi.
        xi_r2phi = {}
        for i, ell1 in enumerate(self.multipoles):
            wxi = xiell[i] * self.window_r
            xi_r2phi[ell1] = {ell2: wxi * phi for ell2, phi in self.window_matrices.items()}

        r2xiell_windowed = {}
        for ell in self.multipoles:
            if ell == 0:
                r2xiell_windowed[ell] = xi_r2phi[0][0] + xi_r2phi[2][2] / 5 + xi_r2phi[4][4] / 9
            elif ell == 2:
                r2xiell_windowed[ell] = (
                    xi_r2phi[0][2] + xi_r2phi[2][0] + 2 * xi_r2phi[2][2] / 7
                    + 2 * xi_r2phi[2][4] / 7 + 2 * xi_r2phi[4][2] / 7
                    + 100 * xi_r2phi[4][4] / 693 + 25 * xi_r2phi[4][6] / 143
                )
            elif ell == 4:
                r2xiell_windowed[ell] = (
                    xi_r2phi[0][4] + xi_r2phi[4][0] + 18 * xi_r2phi[2][2] / 35
                    + 20 * xi_r2phi[2][4] / 77 + 20 * xi_r2phi[4][2] / 77 + 45 * xi_r2phi[2][6] / 143
                    + 162 * xi_r2phi[4][4] / 1001 + 20 * xi_r2phi[4][6] / 143
                )

        return r2xiell_windowed

    @staticmethod
    def scalar_window_rfunc(x):
        """Pair separation window function"""
        if x < 1./2.:
            return 1.
        elif x < 3./4.:
            return 1.-8.*pow(2.*x-1., 3.)+8.*pow(2.*x-1., 4.)
        elif x < 1:
            return -64.*pow(x-1., 3.)-128.*pow(x-1., 4.)
        else:
            return 0.

    @staticmethod
    def D_func(ell, kr):
        if ell == 0:
            return kr**2 * special.spherical_jn(1, kr)
        elif ell == 2:
            return kr * np.cos(kr) - 4 * np.sin(kr) + 3 * special.sici(kr)[0]
        elif ell == 4:
            return 0.5 * ((105 / kr - 2 * kr) * np.cos(kr)
                          + (22 - 105 / kr**2) * np.sin(kr)
                          + 15 * special.sici(kr)[0])
        else:
            raise ValueError(f"Ell={ell} not supported.")

    def compute_sph_kernels(self, k_edges, k_centers, r_grid, k_integration=True):
        """Compute the Bessel integration kernels.
        These are i^ell j_ell(kr) optionally (analytically) integrated over each k-bin"""
        sph_kernels = {}
        if not k_integration:
            for ell in self.multipoles:
                sph_kernels[ell] = (1.0j)**ell * special.spherical_jn(
                    ell, np.outer(k_centers, r_grid))
        else:
            for ell in sph_kernels:
                this_kernel = []
                for k_low, k_high in k_edges:
                    kr_low = k_low * r_grid
                    kr_high = k_high * r_grid

                    this_term = 3 * (-1)**(ell / 2)
                    this_term *= (self.D_func(ell, kr_high) - self.D_func(ell, kr_low))
                    this_term /= (kr_high**3 - kr_low**3)

                    this_kernel.append(this_term)
                sph_kernels[ell] = np.asarray(this_kernel).T

        return sph_kernels
