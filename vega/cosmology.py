import numpy as np
import astropy.units as units
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM


class Cosmology:
    """Class for cosmological computations based on astropy cosmology.
    """
    def __init__(self, Omega_m: float, H0: float, Omega_de: float = None, w0: float = None) -> None:
        # Setup some stuff we need
        m_nu = np.array([0.06, 0.0, 0.0]) * units.electronvolt
        Neff = 3.046
        Tcmb = 2.72548

        # Initialize the right cosmology object
        if Omega_de is None and w0 is None:
            self._cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
        elif w0 is None:
            self._cosmo = LambdaCDM(H0=H0, Om0=Omega_m, Ode0=Omega_de, Tcmb0=Tcmb, Neff=Neff,
                                    m_nu=m_nu)
        elif Omega_de is None:
            self._cosmo = FlatwCDM(H0=H0, Om0=Omega_m, w0=w0, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
        else:
            self._cosmo = wCDM(H0=H0, Om0=Omega_m, Ode0=Omega_de, w0=w0, Tcmb0=Tcmb, Neff=Neff,
                               m_nu=m_nu)

    def comoving_distance_mpc(self, z):
        return self._cosmo.comoving_distance(z).value

    def comoving_distance_hinv_mpc(self, z):
        return self._cosmo.comoving_distance(z).value * (self._cosmo.H0.value / 100)

    def comoving_transverse_distance_mpc(self, z):
        return self._cosmo.comoving_transverse_distance(z).value

    def comoving_transverse_distance_hinv_mpc(self, z):
        return self._cosmo.comoving_transverse_distance(z).value * (self._cosmo.H0.value / 100)
