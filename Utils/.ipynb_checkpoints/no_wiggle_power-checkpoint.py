import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad_vec
from Utils.extrapolate_pk import extrapolate_pk

def no_wiggle_power(k_pk:np.ndarray): 
    
    """ 
    Computes no wiggle power spectrum by using the procedure described in 2407.04660.
    Requires broad enough input (0.008 - 1.5 h/Mpc) to be reliable. 
    Input k_pk array, output k_pk array on the same wavenumbers. 
    
    """

    k_in = k_pk[0]
    if np.max(k_in) < 1.5 or np.min(k_in) > 8e-3:
        print("warning - please provide wavenumbers on at least 0.008 - 1.5 h/Mpc for reliable results")
    k_pk_extrap = extrapolate_pk(k_pk)    
    k_extrap = k_pk_extrap[0]
    pk_extrap = k_pk_extrap[1]

    pk_log_spl = CubicSpline(np.log10(k_extrap), k_pk_extrap[1])

    k_pk_eh = np.loadtxt("Utils/eh_spec.dat", unpack=True).T ## normalize by eisenstein/hu

    k_eh = k_pk_eh[0]
    pk_eh = k_pk_eh[1]

    pk_eh_log_spl = CubicSpline(np.log10(k_eh), pk_eh)

    k_eh_cut = k_eh[277:477] ## compute p_nowiggle between roughly 1e-2 and 1e0. 
    lk_eh_cut = np.log10(k_eh_cut)
    pk_eh_cut = pk_eh[277:477]
    sigma = 0.005 + 0.45 * np.exp(- 0.9 * (lk_eh_cut + 0.15) ** 2) * (1 / (1 + np.exp(8 * (lk_eh_cut + 0.5))))

    pk_nw_integrand = lambda lq: 1 / (abs(sigma) * np.sqrt(2 * np.pi)) * pk_log_spl(lq) / pk_eh_log_spl(lq) * \
    np.exp(- 1 / (2 * sigma ** 2) * (lq - lk_eh_cut) ** 2)
    
    pk_nw = pk_eh_cut * quad_vec(pk_nw_integrand, np.log10(8e-4), np.log10(4), norm='max', points=[-2,0]) [0] 
 
    ## attach pk_extrap at high and low ends again 
    idx_low = np.min(np.where(k_extrap > k_eh_cut[0])[0])

    idx_khigh = np.max(np.where(k_extrap < k_eh_cut[-1])[0])

    k_tot = np.concatenate([k_extrap[:idx_low-1], k_eh_cut, k_extrap[idx_khigh+1:]])
    pk_nw_extrap = np.concatenate([pk_extrap[:idx_low-1], pk_nw, pk_extrap[idx_khigh+1:]])
    
    pk_nw_out = CubicSpline(np.log10(k_tot), pk_nw_extrap)(np.log10(k_in))
    return np.array([k_in, pk_nw_out])
    