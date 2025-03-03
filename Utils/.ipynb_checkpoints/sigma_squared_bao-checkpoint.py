import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from Utils.no_wiggle_power import no_wiggle_power
from Utils.extrapolate_pk import extrapolate_pk
from scipy.special import spherical_jn

def sigma_squared_bao(k_pk:np.ndarray):
    
    """ 
    Computes displacement dispersion using brute-force integration. 
    high k cutoff is 0.2 in the provided units (so not necessarily h_fid). Note disp is not dimensionless but has units (Mpc/h)^2.
    Input k_pk array, output float.
    """

    k_pk = extrapolate_pk(k_pk)
    k_pk_nw = no_wiggle_power(k_pk)
    k_nw = k_pk_nw[0]
    pk_nw = k_pk_nw[1]

    kosc = 1/105
    
    pk_nw_log_spl = CubicSpline(np.log10(k_nw), pk_nw)
    disp_integrand = lambda q: 1 / (6 * np.pi ** 2) * pk_nw_log_spl(np.log10(q)) * (1 - spherical_jn(0, q / kosc) + 2 * spherical_jn(2, q / kosc))
    disp = quad(disp_integrand, 0.0001, 0.2)[0]
    
    return disp