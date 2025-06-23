import numpy as np


class Coordinates:
    def get_mask_to_other(self, other):
        raise NotImplementedError

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


class RtRpCoordinates(Coordinates):
    """Class to handle Vega coordinate grids
    """

    def __init__(
        self, rp_min, rp_max, rt_max, rp_nbins, rt_nbins,
        rp_grid=None, rt_grid=None, z_grid=None, z_eff=None,
        r_grid=None, mu_grid=None
    ):
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

        if r_grid is None:
            self.r_grid = np.sqrt(self.rp_grid**2 + self.rt_grid**2)
        else:
            self.r_grid = r_grid
        self.r_regular_grid = np.sqrt(self.rp_regular_grid**2 + self.rt_regular_grid**2)

        if mu_grid is None:
            self.mu_grid = np.zeros_like(self.r_grid)
            w = self.r_grid > 0.
            self.mu_grid[w] = self.rp_grid[w] / self.r_grid[w]
        else:
            self.mu_grid = mu_grid

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

    @classmethod
    def init_from_r_mu_grids(cls, r_grid, mu_grid, z_eff=None):
        """Initialize from r and mu grids

        Parameters
        ----------
        r_grid : Array
            r grid
        mu_grid : Array
            mu grid

        Returns
        -------
        Coordinates
            New coordinates
        """
        if len(r_grid) != len(mu_grid):
            raise ValueError(
                'r_grid and mu_grid must either be on a meshgrid or have the same size')
        rp_grid = r_grid * mu_grid
        rt_grid = r_grid * np.sqrt(1 - mu_grid**2)
        return cls(
            rp_min=rp_grid.min(), rp_max=rp_grid.max(), rt_max=rt_grid.max(),
            rp_nbins=len(r_grid), rt_nbins=len(r_grid), rp_grid=rp_grid, rt_grid=rt_grid,
            r_grid=r_grid, mu_grid=mu_grid, z_eff=z_eff,
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


class RMuCoordinates(Coordinates):
    """Class to handle Vega coordinate grids
    """

    def __init__(
        self, mu_min, mu_max, r_max, mu_nbins, r_nbins,
        mu_grid=None, r_grid=None, z_grid=None, z_eff=None,
        rp_grid=None, rt_grid=None
    ):
        """Initialize the coordinate grids.

        Parameters
        ----------
        mu_min : float
            Minimum mu
        mu_max : float
            Maximum mu
        r_max : float
            Maximum r
        mu_nbins : int
            Number of mu bins
        r_nbins : int
            Number of r bins
        rp_grid : Array , optional
            rp grid, by default None
        rt_grid : Array, optional
            rt grid, by default None
        z_grid : Array, optional
            z grid, by default None
        """
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.r_max = r_max
        self.mu_nbins = mu_nbins
        self.r_nbins = r_nbins

        self.mu_binsize = (mu_max - mu_min) / mu_nbins
        self.r_binsize = r_max / r_nbins

        mu_regular_grid = (0.5 + np.arange(mu_nbins)) * self.mu_binsize
        r_regular_grid = (0.5 + np.arange(r_nbins)) * self.r_binsize

        r_regular_grid, mu_regular_grid = np.meshgrid(r_regular_grid, mu_regular_grid)
        self.mu_regular_grid = mu_regular_grid.flatten()
        self.r_regular_grid = r_regular_grid.flatten()

        self.mu_grid = self.mu_regular_grid if mu_grid is None else mu_grid
        self.r_grid = self.r_regular_grid if r_grid is None else r_grid

        if rp_grid is None:
            self.rp_grid = self.r_grid * self.mu_grid
        else:
            self.rp_grid = rp_grid

        if rt_grid is None:
            self.rt_grid = self.r_grid * np.sqrt(1.0 - self.mu_grid**2)
        else:
            self.rt_grid = rt_grid

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
        raise NotImplementedError

    @classmethod
    def init_from_r_mu_grids(cls, r_grid, mu_grid, z_eff=None):
        """Initialize from r and mu grids

        Parameters
        ----------
        r_grid : Array
            r grid
        mu_grid : Array
            mu grid

        Returns
        -------
        Coordinates
            New coordinates
        """
        raise NotImplementedError

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
        assert self.mu_binsize == other.mu_binsize
        assert self.r_binsize == other.r_binsize
        mask = (self.mu_grid >= other.mu_min) & (self.mu_grid <= other.mu_max)
        mask &= (self.r_grid <= other.r_max)
        return mask
