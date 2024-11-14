import numpy as np


class BroadbandPolynomials:
    """ Class for computing broadband polynomials. """
    def __init__(self, bb_input, cf_name, model_coordinates, dist_model_coordinates):
        self.model_coordinates = model_coordinates
        self.dist_model_coordinates = dist_model_coordinates

        self.bb_terms = {
            'pre-add': [],
            'pre-mul': [],
            'post-add': [],
            'post-mul': []
        }

        for i, bb in enumerate(bb_input.values()):
            bb = bb.split()

            # Check if the bb setup is valid
            if len(bb) not in [5, 6]:
                raise ValueError(
                    f'Broadband setup must have 5 or 6 elements. Got {len(bb)} elements')
            if bb[0] not in ['add', 'mul']:
                raise ValueError(f'Broadband type must be either "add" or "mul". Got {bb[0]}')
            if bb[1] not in ['pre', 'post']:
                raise ValueError(f'Broadband position must be either "pre" or "post". Got {bb[1]}')
            if bb[2] not in ['rp,rt', 'r,mu']:
                raise ValueError(
                    f'Broadband coordinates must be either "rp,rt" or "r,mu". Got {bb[2]}')
            if len(bb[3].split(':')) != 3:
                raise ValueError(
                    f'Broadband coordinates must be in the format "min:max:step". Got {bb[3]}')
            if len(bb[4].split(':')) != 3:
                raise ValueError(
                    f'Broadband coordinates must be in the format "min:max:step". Got {bb[4]}')
            if len(bb) > 5 and bb[5] != 'broadband_sky':
                raise ValueError(
                    'If passing six elements in the broadband config, '
                    f'the sixth element must be "broadband_sky". Got {bb[5]}'
                )

            # Initialize the broadband config
            r1_min, r1_max, dr1 = bb[3].split(':')
            r2_min, r2_max, dr2 = bb[4].split(':')
            if len(bb) > 5:
                name = f'BB-{cf_name}-{i}-{bb[5]}'
            else:
                name = f'BB-{cf_name}-{i} {bb[0]} {bb[1]} {bb[2]}'

            # Create the broadband term dictionary
            bb_term = {
                'name': name,
                'func': 'broadband' if len(bb) == 5 else bb[5],
                'coordinates': bb[2],
                'r1_config': (int(r1_min), int(r1_max), int(dr1)),
                'r2_config': (int(r2_min), int(r2_max), int(dr2))
            }
            self.bb_terms[f'{bb[1]}-{bb[0]}'] += [bb_term]

    def compute(self, params, pos_type):
        assert pos_type in list(self.bb_terms.keys())

        if 'pre' in pos_type:
            coordinates = self.model_coordinates
        else:
            coordinates = self.dist_model_coordinates

        bb_poly_total = None
        for bb_term in self.bb_terms[pos_type]:
            if bb_term['func'] == 'broadband':
                bb_poly = self._compute_broadband(bb_term, params, coordinates)
            elif bb_term['func'] == 'broadband_sky':
                bb_poly = self._compute_broadband_sky(bb_term['name'], params, coordinates)
            else:
                raise ValueError(f'Broadband function {bb_term["func"]} not supported')

            if bb_poly_total is None:
                bb_poly_total = 1 + np.exp(bb_poly) if 'mul' in pos_type else bb_poly
            elif 'mul' in pos_type:
                bb_poly_total *= (1 + np.exp(bb_poly))
            else:
                bb_poly_total += bb_poly

        if bb_poly_total is None:
            bb_poly_total = 1 if 'mul' in pos_type else 0

        return bb_poly_total

    @staticmethod
    def _compute_broadband_sky(bb_term_name, params, coordinates):
        """Compute sky broadband term.

        Calculates a Gaussian broadband in rp,rt for the sky residuals.

        Parameters
        ----------
        bb_term : dict
            broadband term config
        params : dict
            Computation parameters

        Returns
        -------
        1d Array
            Output broadband
        """
        scale = params[bb_term_name + '-scale-sky']
        sigma = params[bb_term_name + '-sigma-sky']

        corr = scale / (sigma * np.sqrt(2. * np.pi))
        corr *= np.exp(-0.5 * (coordinates.rt_grid / sigma)**2)
        w = (coordinates.rp_grid >= 0.) & (coordinates.rp_grid < coordinates.rp_binsize)
        corr[~w] = 0.

        return corr

    @staticmethod
    def _compute_broadband(bb_term, params, coordinates):
        """Compute broadband term.

        Calculates a power-law broadband in r and mu or rp,rt.

        Parameters
        ----------
        bb_term : dict
            broadband term config
        params : dict
            Computation parameters

        Returns
        -------
        1d Array
            Output broadband
        """
        if bb_term['coordinates'] == 'r,mu':
            r1 = coordinates.r_grid / 100.
            r2 = coordinates.mu_grid
        elif bb_term['coordinates'] == 'rp,rt':
            r1 = coordinates.r_grid / 100. * coordinates.mu_grid
            r2 = coordinates.r_grid / 100. * np.sqrt(1 - coordinates.mu_grid**2)
        else:
            raise ValueError(f'Coordinates {bb_term["coordinates"]} not supported')

        r1_min, r1_max, dr1 = bb_term['r1_config']
        r2_min, r2_max, dr2 = bb_term['r2_config']
        r1_powers = np.arange(r1_min, r1_max + 1, dr1)
        r2_powers = np.arange(r2_min, r2_max + 1, dr2)

        bb_params = []
        for i in r1_powers:
            for j in r2_powers:
                bb_params.append(params[f'{bb_term["name"]} ({i},{j})'])

        # the first dimension of bb_params is that of r1 power indices
        # the second dimension of bb_params is that of r2 power indices
        bb_params = np.array(bb_params).reshape(r1_max - r1_min + 1, -1)

        # we are summing 3D array along 2 dimensions.
        # first dimension is the data array (2500 for the standard rp rt grid of 50x50)
        # second dimension is the first dimension of bb_params  = r1 power indices
        # third dimension is the sectond dimension of bb_params = r2 power indices
        # dimensions addressed in an array get the indices ':'
        # dimensions not addressed in an array get the indices 'None'
        # we sum over the second and third dimensions which are powers of r
        corr = (bb_params[None, :, :] * r1[:, None, None]**r1_powers[None, :, None]
                * r2[:, None, None]**r2_powers[None, None, :]).sum(axis=(1, 2))

        return corr
