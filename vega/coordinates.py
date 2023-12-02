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

        rp_regular_grid = np.arange(rp_min + self.rp_binsize / 2, rp_max, self.rp_binsize)
        rt_regular_grid = np.arange(self.rt_binsize / 2, rt_max, self.rt_binsize)

        rt_regular_grid, rp_regular_grid = np.meshgrid(rt_regular_grid, rp_regular_grid)
        self.rp_regular_grid = rp_regular_grid.flatten()
        self.rt_regular_grid = rt_regular_grid.flatten()

        self.rp_grid = self.rp_regular_grid if rp_grid is None else rp_grid
        self.rt_grid = self.rt_regular_grid if rt_grid is None else rt_grid

        self.r_grid = np.sqrt(self.rp_grid**2 + self.rt_grid**2)
        self.r_regular_grid = np.sqrt(self.rp_regular_grid**2 + self.rt_regular_grid**2)

        self.mu_grid = np.zeros_like(self.r_grid)
        w = self.r_grid > 0.
        self.mu_grid[w] = self.rp_grid[w] / self.r_grid[w]

        self.mu_regular_grid = np.zeros_like(self.r_regular_grid)
        w = self.r_regular_grid > 0.
        self.mu_regular_grid[w] = self.rp_regular_grid[w] / self.r_regular_grid[w]

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

        mask = (self.rp_regular_grid > rp_min_cut) & (self.rp_regular_grid < rp_max_cut)
        mask &= (self.rt_regular_grid > rt_min_cut) & (self.rt_regular_grid < rt_max_cut)
        mask &= (self.r_regular_grid > r_min_cut) & (self.r_regular_grid < r_max_cut)
        mask &= (self.mu_regular_grid > mu_min_cut) & (self.mu_regular_grid < mu_max_cut)

        return mask
