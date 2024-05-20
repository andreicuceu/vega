import numpy as np


class Coordinates:
    """Class to handle Vega coordinate grids
    """

    def __init__(self, rp_min, rp_max, rt_max, rp_nbins, rt_nbins,
                 rp_grid=None, rt_grid=None, z_grid=None, z_eff=None):
        """Initialize the coordinate grids.

        Parameters
        ----------
        rp_min : float
            Minimum rp
        rp_max : float
            Maximum rp
        rt_max : float
            Maximum rt
        rp_nbins : float
            Number of rp bins
        rt_nbins : float
            Number of rt bins
        rp_grid : Array , optional
            rp grid, by default None
        rt_grid : Array, optional
            rt grid, by default None
        z_grid : Array, optional
            z grid, by default None
        """
        self.rp_min = rp_min
        self.rp_max = rp_max
        self.rt_max = rt_max
        self.rp_nbins = rp_nbins
        self.rt_nbins = rt_nbins

        self.rp_binsize = (rp_max - rp_min) / rp_nbins
        self.rt_binsize = rt_max / rt_nbins

        self.rp_regular_grid = np.arange(rp_min + self.rp_binsize / 2, rp_max, self.rp_binsize)
        self.rt_regular_grid = np.arange(self.rt_binsize / 2, rt_max, self.rt_binsize)

        rt_regular_meshgrid, rp_regular_meshgrid = np.meshgrid(
            self.rt_regular_grid, self.rp_regular_grid)
        self.rp_regular_meshgrid = rp_regular_meshgrid.flatten()
        self.rt_regular_meshgrid = rt_regular_meshgrid.flatten()

        self.rp_grid = self.rp_regular_meshgrid if rp_grid is None else rp_grid
        self.rt_grid = self.rt_regular_meshgrid if rt_grid is None else rt_grid

        self.r_grid = np.sqrt(self.rp_grid**2 + self.rt_grid**2)
        self.r_regular_grid = np.sqrt(self.rp_regular_meshgrid**2 + self.rt_regular_meshgrid**2)

        self.mu_grid = np.zeros_like(self.r_grid)
        w = self.r_grid > 0.
        self.mu_grid[w] = self.rp_grid[w] / self.r_grid[w]

        self.mu_regular_grid = np.zeros_like(self.r_regular_grid)
        w = self.r_regular_grid > 0.
        self.mu_regular_grid[w] = self.rp_regular_meshgrid[w] / self.r_regular_grid[w]

        if z_grid is None and z_eff is None:
            self.z_grid = None
        else:
            self.z_grid = z_eff if z_grid is None else z_grid

    @classmethod
    def init_from_grids(cls, other, rp_grid, rt_grid, z_grid):
        """Initialize from other coordinates and new grids

        Parameters
        ----------
        other : Coordinates
            Other coordinates
        rp_grid : Array
            rp grid
        rt_grid : Array
            rt grid
        z_grid : Array
            z grid

        Returns
        -------
        Coordinates
            New coordinates
        """
        return cls(
            other.rp_min, other.rp_max, other.rt_max, other.rp_nbins, other.rt_nbins,
            rp_grid=rp_grid, rt_grid=rt_grid, z_grid=z_grid
        )

    def get_mask_to_other(self, other):
        """Build mask from the current coordinates to the other coordinates.

        Parameters
        ----------
        other : Coordinates
            Other coordinates

        Returns
        -------
        Array
            Mask
        """
        assert self.rp_binsize == other.rp_binsize
        assert self.rt_binsize == other.rt_binsize
        mask = (self.rp_grid >= other.rp_min) & (self.rp_grid <= other.rp_max)
        mask &= (self.rt_grid <= other.rt_max)
        return mask

    def get_mask_scale_cuts(self, cuts_config):
        """Build mask to apply scale cuts

        Parameters
        ----------
        cuts_config : ConfigParser
            Cuts section from config

        Returns
        -------
        Array
            Mask
        """
        # Read the cuts
        rp_min_cut = cuts_config.getfloat('rp-min', 0.)
        rp_max_cut = cuts_config.getfloat('rp-max', 200.)

        rt_min_cut = cuts_config.getfloat('rt-min', 0.)
        rt_max_cut = cuts_config.getfloat('rt-max', 200.)

        r_min_cut = cuts_config.getfloat('r-min', 10.)
        r_max_cut = cuts_config.getfloat('r-max', 180.)

        mu_min_cut = cuts_config.getfloat('mu-min', -1.)
        mu_max_cut = cuts_config.getfloat('mu-max', +1.)

        mask = (self.rp_regular_meshgrid > rp_min_cut) & (self.rp_regular_meshgrid < rp_max_cut)
        mask &= (self.rt_regular_meshgrid > rt_min_cut) & (self.rt_regular_meshgrid < rt_max_cut)
        mask &= (self.r_regular_grid > r_min_cut) & (self.r_regular_grid < r_max_cut)
        mask &= (self.mu_regular_grid > mu_min_cut) & (self.mu_regular_grid < mu_max_cut)

        return mask


class PkCoordinates:
    '''Class to handle P(k) coordinate grids
    '''
    def __init__(self, k_edges, num_ells=3):
        '''Initialize the coordinate grids.

        Parameters
        ----------
        k_edges : Array
            k edges
        num_ells : int, optional
            Number of multipoles, by default 3
        '''
        self.k_edges = k_edges
        self.k_centers = self.k_edges.mean(1)
        self.k_binsizes = k_edges[:, 1] - k_edges[:, 0]
        self.num_bins = len(self.k_centers)
        self.num_ells = num_ells
        self.k_min = self.k_edges.min()
        self.k_max = self.k_edges.max()
        self.data_ells = np.arange(0, 2*num_ells, 2)

    def get_mask_scale_cuts(self, cuts_config):
        '''Build mask to apply scale cuts

        Parameters
        ----------
        cuts_config : ConfigParser
            Cuts section from config

        Returns
        -------
        Array
            Mask
        '''
        # Read the cuts
        k_min_cut = cuts_config.getfloat('k-min', 0.)
        k_max_cut = cuts_config.getfloat('k-max', 0.2)
        self.model_ells = np.array(cuts_config.get('use_multipoles', '0,2,4').split(',')).astype(int)

        for ell in self.model_ells:
            if ell not in self.data_ells:
                raise ValueError(
                    f'Invalid multipole in cuts. Valid multipoles are {self.data_ells}')

        mask = np.full((self.num_bins, self.num_ells), True, dtype=bool)
        for ell in self.data_ells:
            mask[:, ell] = (ell in self.model_ells) \
                            & (self.k_centers > k_min_cut) \
                            & (self.k_centers < k_max_cut)

        return mask
