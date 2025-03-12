import numpy as np
from cobra.Utils.no_wiggle_power import no_wiggle_power
from cobra.Utils.sigma_squared_bao import sigma_squared_bao
from cobra.Utils.extrapolate_pk import extrapolate_pk
from scipy.interpolate import CubicSpline

def ir_resummed_power(k_pk:np.ndarray):

    """
    Compute ir resummed power using wiggle-no-wiggle split. Same input range restrictions apply as to no wiggle power.
    Input and output are k_pk arrays. Output range is same as input range.
    
    """
    if np.max(k_pk[0]) < 1.5 or np.min(k_pk[0]) > 8e-3:
        print("warning - please provide wavenumbers on at least 0.008 - 1.5 h/Mpc for reliable results")
        
    k_pk_extrap = extrapolate_pk(k_pk)
    k_in = k_pk_extrap[0]
    pk_in = k_pk_extrap[1]
    pk_nw = no_wiggle_power(k_pk_extrap)[1]
    disp = sigma_squared_bao(k_pk_extrap)
    pk_ir = pk_nw + np.exp(- k_in ** 2 * disp) * (pk_in - pk_nw)
    pk_ir_spl = CubicSpline(np.log10(k_in),pk_ir)(np.log10(k_pk[0]))
    
    return np.array([k_pk[0], pk_ir_spl])