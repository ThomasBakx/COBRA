import numpy as np

def extrapolate_pk(k_pk:np.ndarray):
    """ 
    Extrapolate k_pk array using a power law at low k and gaussian decay at high k.
    Output range is from 1e-5 to 1e2.
    """
    
    k_in = k_pk[0]
    pk_in = k_pk[1]
    
    kmin = k_in[0]
    kmax = k_in[-1]

    if kmin < 1e-5 and kmax > 1e2:
        return k_pk
    else:
        ns_fid = 0.96
        decay = 50
        n_extrap_low = np.ceil(50*np.log10(kmin/1e-5)).astype(int)
        n_extrap_high = np.ceil(50*np.log10(1e2/kmax)).astype(int)
        k_extrap_low = np.logspace(-5, np.log10(kmin), n_extrap_low)
        pk_extrap_low = pk_in[0] * (k_extrap_low / kmin) ** ns_fid
        k_extrap_high = np.logspace(np.log10(kmax), 2, n_extrap_high)
        pk_extrap_high = pk_in[-1] * np.exp(- ((k_extrap_high - kmax) / decay) ** 2)
               
        k_extrap = np.concatenate([k_extrap_low[:-1], k_in, k_extrap_high[1:]])
        pk_extrap = np.concatenate([pk_extrap_low[:-1], pk_in, pk_extrap_high[1:]])
        
        return np.array([k_extrap, pk_extrap])
                          
        