import numpy as np

from .wedges import Wedge


class RtWedge(Wedge):
    """
    Computes a perpendicualr distance bin for a 2D function
    """

    def __init__(self, rp=(0., 200., 50),
                 rt=(0., 200., 50),
                 rt_cut=(0., 4.0)):
        """Initialize computation of a wedge
        Parameters
        ----------
        rp : tuple, optional
            (Min, Max, Size) for r_parallel, by default (0., 200., 50)
        rt : tuple, optional
            (Min, Max, Size) for r_transverse, by default (0., 200., 50)
        r : tuple, optional
            (Min, Max, Size) for radius, by default (0., 200., 50)
        rt_cut: tuple, optional
            (Min, Max) for r_transverse cuts, by default (0., 4.0)
        scaling : int, optional
            Scaling for grid computation, by default 10
        """
        # Init bin limits on the fine scaled grid and get centers
        rp_scaled_bins = np.linspace(rp[0], rp[1], rp[2] + 1)
        rt_scaled_bins = np.linspace(rt[0], rt[1], rt[2] + 1)
        rp_centers = self.get_bin_centers(rp_scaled_bins)
        rt_centers = self.get_bin_centers(rt_scaled_bins)

        # Create meshes on the finer grid for all elements
        rt_mesh, rp_mesh = np.meshgrid(rt_centers, rp_centers)

        # Init the right bin limits
        rp_bins = np.linspace(rp[0], rp[1], rp[2] + 1)
        rt_bins = np.linspace(rt[0], rt[1], rt[2] + 1)

        # Compute the normal bin indices on the fine meshes
        rt_idx = np.digitize(rt_mesh, rt_bins) - 1
        rp_idx = np.digitize(rp_mesh, rp_bins) - 1

        # Compute bins on mesh. The numbers are positions in weights array
        bins = rt_idx + rt[2] * rp_idx + rt[2] * rp[2] * rp_idx

        # Compute the mask for the wedge
        mask = (rt_mesh > rt_cut[0]) & (rt_mesh < rt_cut[1])

        # Compute the right counts and their index
        wedge_bins = bins[mask]
        counts = np.bincount(wedge_bins.flatten())
        positive_idx = np.where(counts != 0)

        # Initialize the weights and insert the right counts
        self.weights = np.zeros((rp[2], rt[2] * rp[2]))
        weights_idx = np.unravel_index(positive_idx, np.shape(self.weights))
        self.weights[weights_idx] = counts[positive_idx]
        self.r = self.get_bin_centers(rp_bins)
