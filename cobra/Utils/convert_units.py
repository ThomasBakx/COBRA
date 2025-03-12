import numpy as np

def convert_units_k_pk_from_hfid(k_pk:np.ndarray, h_out:float):
    
    """
    Converts a power spectrum in h_fid = 0.7 units to other units given by h_out
    Takes 2d array k_pk of size 2 x n_bins (first index is k, second index is pk)
    Outputs array of the same dimension with the spectrum in h_out units
    Not vectorized - provide only one h_out.
    """   

    k_in = k_pk[0]
    pk_in = k_pk[1]

    k_out = (0.7 / h_out) * k_in
    pk_out = (h_out / 0.7) ** 3 * pk_in
        
    return np.array([k_out,pk_out])

def convert_units_k_pk_to_hfid(k_pk:np.ndarray, h_in:float):
    
    """
    Converts a power spectrum in h_in units to other units given by h_fid = 0.7
    Takes 2d array k_pk of size 2 x n_bins (first index is k, second index is pk)
    Outputs array of the same dimension with the spectrum in h_fid units
    Not vectorized - provide only one h_in.
    """ 

    k_in = k_pk[0]
    pk_in = k_pk[1]

    k_out = (h_in / 0.7) * k_in
    pk_out = (0.7 / h_in) ** 3 * pk_in
        
    return np.array([k_out,pk_out])

def convert_units_disp_from_hfid(disp:np.ndarray, h_out:np.ndarray):
    """
    Converts displacement in h_fid = 0.7 units to other units h_out. 
    Vectorized.
    """   

    disp_out = (h_out / 0.7) ** 2 * disp
        
    return disp_out

def convert_units_disp_to_hfid(disp:np.ndarray, h_in:np.ndarray):
    """
    Converts displacement in h_in units to h_fid = 0.7. 
    Vectorized.
    """  

    disp_out = (0.7 / h_in) ** 2 * disp
        
    return disp_out