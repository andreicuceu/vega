import numpy as np


class Shell:
    """
    Compress 2D correlation functions defined on an r_parallel/r_transverse grid
    into shells as a function mu
    """
    def __init__(
        self, rp=(0, 200, 50), rt=(0, 200, 50), angle_var='theta',
        angle_range=(0, np.pi/2), num_bins_fraction=50,
        r=(30, 45), scaling=10, abs_mu=False
    ):
        """Initialize computation of a shell

        Parameters
        ----------
        rp : tuple, optional
            (Min, Max, Size) for r_parallel, by default (0, 200, 50)
        rt : tuple, optional
            (Min, Max, Size) for r_transverse, by default (0, 200, 50)
        angle_var : str, optional
            Variable to use for the angle from ['theta', 'mu', 'mu2'], by default 'theta'
        angle_range : tuple, optional
            (Min, Max) for angle variable defined above, by default (0, np.pi/2)
        num_bins_fraction : int, optional
            _description_, by default 50
        r : tuple, optional
            (Min, Max) for isotropic separation bin, by default (30, 45)
        scaling : int, optional
            Scaling for grid computation, by default 10
        abs_mu : bool, optional
            Flag for working with absolute values of mu, by default False
        """
        assert angle_var in ['theta', 'mu', 'mu2'], "angle_var must be from ['theta', 'mu', 'mu2']"
        if angle_var != 'theta':
            angle_range = (angle_range[0], min(angle_range[1], 1))

        # Init bin limits on the fine scaled grid and get centers
        rp_scaled_bins = np.linspace(rp[0], rp[1], (scaling * rp[2]) + 1)
        rt_scaled_bins = np.linspace(rt[0], rt[1], (scaling * rt[2]) + 1)
        rp_centers = self.get_bin_centers(rp_scaled_bins)
        rt_centers = self.get_bin_centers(rt_scaled_bins)

        # Create meshes on the finer grid for all elements
        rt_mesh, rp_mesh = np.meshgrid(rt_centers, rp_centers)
        r_mesh = np.sqrt(rp_mesh**2 + rt_mesh**2)
        mu_mesh = (rp_mesh/r_mesh)

        # Check if we need the absolute value of mu
        if abs_mu:
            mu_mesh = np.absolute(mu_mesh)
            mu2_mesh = mu_mesh**2
            theta_mesh = np.arccos(mu_mesh)
        else:
            mu2_mesh = mu_mesh**2
            mu2_mesh[mu_mesh < 0] *= -1
            theta_mesh = np.arccos(mu_mesh)

        # Initialize the right bin limits
        rp_bins = np.linspace(rp[0], rp[1], rp[2] + 1)
        rt_bins = np.linspace(rt[0], rt[1], rt[2] + 1)

        # Compute the normal bin indices on the fine meshes
        rt_idx = np.digitize(rt_mesh, rt_bins) - 1
        rp_idx = np.digitize(rp_mesh, rp_bins) - 1

        # Compute the r bin centers on the mesh array so we can check the cuts
        # ? Is there a more elegant way to do this?
        rp_centers = rp[0] + (rp_idx + 0.5) * (rp[1] - rp[0]) / rp[2]
        rt_centers = rt[0] + (rt_idx + 0.5) * (rt[1] - rt[0]) / rt[2]
        r_centers = np.sqrt(rp_centers**2 + rt_centers**2)
        mu_centers = (rp_centers / r_centers)
        mu2_centers = mu_centers**2
        theta_centers = np.arccos(mu_centers)

        mesh = mu_mesh if angle_var == 'mu' else mu2_mesh if angle_var == 'mu2' else theta_mesh
        angle_centers = (
            mu_centers if angle_var == 'mu'
            else mu2_centers if angle_var == 'mu2'
            else theta_centers
        )

        # Compute the mask for the wedge
        mask = (r_mesh >= r[0]) & (r_mesh <= r[1])
        mask &= (angle_centers > angle_range[0]) & (angle_centers < angle_range[1])

        num_bins_angle = int(np.ceil(np.sum(mask) / num_bins_fraction))
        angle_idx = (
            (mesh - angle_range[0]) / (angle_range[1] - angle_range[0]) * num_bins_angle
        ).astype(int)

        # Compute bins on mesh. The numbers are positions in weights array
        bins = rt_idx + rt[2] * rp_idx + rt[2] * rp[2] * angle_idx

        # Compute the right counts and their index
        wedge_bins = bins[mask]
        counts = np.bincount(wedge_bins.flatten())
        positive_idx = np.where(counts != 0)

        # Initialize the weights and insert the right counts
        self.weights = np.zeros((num_bins_angle, rt[2] * rp[2]))
        weights_idx = np.unravel_index(positive_idx, np.shape(self.weights))
        self.weights[weights_idx] = counts[positive_idx]

        angle_bins = np.linspace(angle_range[0], angle_range[1], num_bins_angle + 1)
        self.angle = self.get_bin_centers(angle_bins) 
        if angle_var == 'theta':
            self.angle *= 180 / np.pi  # Convert to degrees

    def __call__(self, data, covariance=None):
        """Computes the shell for the input data and optional covariance

        Parameters
        ----------
        data : 1D array
            Data vector
        covariance : 2D array, optional
            Covariance Matrix, by default None

        Returns
        -------
        tuple
            mu coordinates, shell compression, shell covariance (optional)
        """
        # Init covariance
        if covariance is None:
            cov_weight = np.ones(len(data))
        else:
            cov_weight = 1 / np.diagonal(covariance)

        # Transform weights using the covariance and norm
        norm = self.weights.dot(cov_weight)
        data_weights = self.weights * cov_weight
        mask = norm > 0
        data_weights[mask, :] /= norm[mask, None]

        # Compute wedge and return simple
        shell = data_weights.dot(data)
        if covariance is None:
            return self.angle, shell

        # Compute wedge covariance and return full
        shell_cov = data_weights.dot(covariance).dot(data_weights.T)
        return self.angle, shell, shell_cov

    @staticmethod
    def get_bin_centers(bin_limits):
        """Computes array of bin centers given an array of bin limits
        Parameters
        ----------
        bin_limits : 1D array
            Array with the limits of the bins. Size = Num_Bins + 1
        """
        return (bin_limits[1:] + bin_limits[:-1]) / 2
