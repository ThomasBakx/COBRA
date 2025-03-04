import numpy as np
from scipy.special import hyp2f1, poch, factorial, eval_hermite
from numpy.polynomial import legendre 

def get_svd_results(space:str, param_range:str):
    
    """
    Load results from svd into memory. All tables also contain the (same) k-range themselves in the 1st column.
    Returns tuple containing mean power spectrum template, orthonormal basis vj_hat, no-wiggle basis and v_j basis = vj_hat * pbar
    """
    
    k_pk_bar = np.loadtxt("./" + space + "/Stables/" + param_range + "/pbar_" + space + param_range + ".dat", unpack=True).T
    k_vj_hat = np.loadtxt("./" + space + "/Stables/" + param_range + "/vjhat_" + space + param_range + ".dat", unpack=True).T
    k_vj_nw = np.loadtxt("./" + space + "/Stables/" + param_range + "/vj_nw_" + space + param_range + ".dat", unpack=True).T
    k_vj = np.loadtxt("./" + space + "/Stables/" + param_range + "/vj_" + space + param_range + ".dat", unpack=True).T
    
    return k_pk_bar, k_vj_hat, k_vj_nw, k_vj

def get_rbf_tables(space:str, param_range:str):

    """
    Load rbf tables for weight emulation into memory. For LCDM there is the displacements, growth rate, growth factor 
    and weights at fixed evolution parameters. For GEN there is only the growth factor and the weights normalized by D^2\alpha.
    Returns tuple containing all the relevant arrays.
    """

    if space == 'LCDM':
        
        bmat_disp = np.array([np.loadtxt("./" + space + "/RBFtables/" + param_range + "/bmat_disp_" + space + param_range + ".dat", unpack=True)])
        bmat_gr = np.array([np.loadtxt("./" + space + "/RBFtables/" + param_range + "/bmat_gr_" + space + param_range + ".dat", unpack=True)])
        bmat_d = np.array([np.loadtxt("./" + space + "/RBFtables/" + param_range + "/bmat_d_" + space + param_range + ".dat", unpack=True)])
        bmat_wts = np.loadtxt("./" + space + "/RBFtables/" + param_range + "/bmat_wts_" + space + param_range + ".dat", unpack=True).T

        return bmat_disp, bmat_gr, bmat_d, bmat_wts

    if space == 'GEN':

        bmat_d = np.array([np.loadtxt("./" + space + "/RBFtables/" + param_range + "/bmat_d_" + space + param_range + ".dat", unpack=True)])
        bmat_wts = np.loadtxt("./" + space + "/RBFtables/" + param_range + "/bmat_wts_" + space + param_range + ".dat", unpack=True)

        return bmat_d, bmat_wts 
    
        
def get_loop_table(space:str, param_range:str, w_nw:str): 

    """
    Load no-wiggle and wiggle loop tables into memory. Requires also space as input, even though tables are currently only available for LCDM. 
    Specify w_nw = 'w' or 'nw' to get wiggle or no-wiggle tables
    Returns tuple containing the wavenumbers (nk) and tensor of size nterms x n_basis_max x n_basis_max x nk.
    """
    
    nterms = 31    
    if param_range == 'def':
        n_basis_max = 12        
    elif param_range == 'ext':
        n_basis_max = 16
    else:
        raise ValueError(f'Unknown param_range {param_range}')

    k_s_table_loop = np.loadtxt("./" + space + "/Stables/" + param_range + "/gg1loop_" + w_nw + "_" + space + param_range + ".dat", unpack=True).T
    k_loop = k_s_table_loop[0]
    nk = len(k_loop)
    nc = round(n_basis_max * (n_basis_max + 1) / 2)
    s_table_reshape = np.reshape(k_s_table_loop[1:], (nterms, nc, nk))
    s_table_loop = np.zeros((nterms, n_basis_max, n_basis_max, nk))
    
    for m in range(n_basis_max):
        for j in range(m + 1):
            ind = round(m * (m + 1) / 2 + j)
            s_table_loop[:, m, j, :] = s_table_reshape[:, ind, :]
            s_table_loop[:, j, m, :] = s_table_reshape[:, ind, :]

    return k_loop, s_table_loop
    
def get_angles_and_GL_weights(): 
    
    """
    Do angle integration as in velocileptors: see e.g. 
    https://github.com/sfschen/velocileptors/blob/master/velocileptors/EPT/ept_fullresum_fftw.py
    and https://arxiv.org/pdf/2005.00523,  https://arxiv.org/abs/2012.04636. 
    """

    ngauss = 4
    nus, ws = legendre.leggauss(2 * ngauss)
    mu = nus[0:ngauss] 
    
    ## legendre polynomials computed at appropriate angles - include factor of 2 and (2l+1)/2, as well as angular weights #
    
    L0 = np.array([2 * 1/2 * legendre.Legendre((1))(mu)])
    L2 = np.array([2 * 5/2 * legendre.Legendre((0, 0, 1))(mu)])
    L4 = np.array([2 * 9/2 * legendre.Legendre((0, 0, 0, 0, 1))(mu)])

    ang = (np.concatenate([L0, L2, L4]).T) * ws[:ngauss, None]
    return mu, ang

def get_rbf_param_config(dim):

    """ 
    rbf hyperparameters, as described in 2407.04660 and Fasshauer / McCourt: https://epubs.siam.org/doi/10.1137/110824784
    used for dimension 3 (in LCDM),6 (for the growth factor in generalized cosmologies) or 9 (for weights in generalized cosmologies).
    Returns tuple containing dimension, epsilon, alpha plus derived parameters beta and delta squared (all float), 
    and lastly the maximum degree max_deg of the hermite polynomials to be evaluated in the rbf interpolator (float) 
    and the corresponding multi-indices (see eqs. A16,A17 in 2407.04660). 
    """
    if dim == 3:
        eps = 0.1
        alpha = 1.8
        max_deg = 11
    elif dim == 6:
        eps = 0.1
        alpha = 1.8
        max_deg = 15
    elif dim == 9:
        eps = 0.1
        alpha = 2.5 ##important!! Inverse global length scale associated with the problem, take alpha = 2 or 3?
        max_deg = 16
    else:
        raise ValueError(f'Incorrect value of dim={dim}')
        
    beta = (1 + (2 * eps/alpha) ** 2) ** (1 / 4)
    deltasq = (alpha ** 2 / 2) * (beta ** 2 - 1)
    
    ind_array = np.ones((1, dim), dtype=int)
    for m in range(1, max_deg):
        ind_array = np.concatenate([ind_array, index_array(dim,m)])
    ind_array_m = ind_array[1:] - 1
        
    return dim, eps, alpha, beta, deltasq, max_deg, ind_array_m

def get_bounds(space:str, param_range:str): 
    
    """
    cosmo bounds and growth factor bounds for all cosmological parameter spaces involved. 
    """ 
    
    if space == 'LCDM':
        if param_range == 'def':
            bounds_cbns = np.array([[0.095, 0.0202, 0.91], [0.145, 0.0238, 1.01]])
            bounds_mha = np.array([[0.11, 0.53, 0.25], [0.175, 0.82, 1]])
            return bounds_cbns, bounds_mha
        if param_range == 'ext':
            bounds_cbns = np.array([[0.08, 0.020, 0.8], [0.175, 0.025, 1.2]])
            bounds_mha = np.array([[0.095, 0.48, 0.25], [0.205, 0.92, 1]])
            return bounds_cbns, bounds_mha

    if space == 'GEN':
        if param_range == 'def':
            bounds_9D = np.array([[0.095, 0.0202, -0.12, 0.55, 0.9, 0, -1.25, -0.3, 0], \
                                  [0.145, 0.0238, 0.12, 0.8, 1.02, 0.6, -0.75, 0.3, 3]])
            bounds_6D = np.array([[0.11, -0.13, 0.53, -1.28, -0.33, 0.25], \
                                  [0.175, 0.13, 0.82, -0.72, 0.33, 1]])
            return bounds_9D, bounds_6D
        if param_range == 'ext':
            bounds_9D = np.array([[0.08, 0.020, -0.2, 0.5, 0.8, 0, -1.4, -1.8, 0], \
                                  [0.155, 0.025, 0.2, 0.9, 1.1, 1, -0.6, -0.4, 3.1]])
            bounds_6D = np.array([[0.095, -0.21, 0.49, -1.42, -1.85, 0.25], \
                                  [0.185, 0.21, 0.92, -0.58, -0.35, 1]])
            return bounds_9D, bounds_6D

def growth_rate(cosm:np.ndarray):  # growth rate in LCDM
    omega_m, h, a = cosm
    omega_l = h ** 2 - omega_m  # little omega_l
    q = (omega_l / omega_m) * a ** 3
    return 1 - 6 / 11 * q * (hyp2f1(2, 4 / 3, 17 / 6, -q)) / (hyp2f1(1, 1 / 3, 11 / 6, -q))


def growth_factor_fid(omega_m:np.ndarray):  
    
    """
    (unnormalized) growth factor D(z) in LCDM at redshift 1 and h = hfid = 0.7. Vectorized.
    """
    omega_l = 0.7 ** 2 - omega_m
    q = (omega_l / omega_m) * (1 / 2) ** 3
    return (1 / 2) * hyp2f1(1, 1 / 3, 11 / 6, -q)


def growth_factor(cosm:np.ndarray): 
    
    """
    (unnormalized) growth factor D(z) in LCDM. Vectorized.
    """
    
    omega_m, h, a = cosm
    omega_l = h ** 2 - omega_m  # little omega_l
    q = (omega_l / omega_m) * a ** 3
    return a * hyp2f1(1, 1 / 3, 11 / 6, -q)


def growth_factor_bar(cosm:np.ndarray, param_range:str):  # D_bar 

    """
    D_bar from Eq. C5. Vectorized. Note that the second to last entry of the input cosmology array is w+ or w_a depending on param_range.
    """

    if param_range == 'def':
        omega_m, ok, h, w0, wa, a = cosm
    elif param_range == 'ext':
        omega_m, ok, h, w0, wp, a = cosm
        wa = wp - w0
        
    omega_k = ok * h ** 2
    omega_l = h ** 2 - omega_m - omega_k
    q = (omega_l / omega_m) * a ** (-3 * (w0 + wa)) * np.exp(3 * wa * (a - 1))
    x = omega_k / (np.abs(omega_l) ** (1 / 3) * omega_m ** (2 / 3))

    exp_fac = np.sqrt(omega_m * a ** -3 + omega_k * a ** -2 + omega_l * a ** (-3 * (1 + wa + w0)) * np.exp(
        3 * wa * (a - 1)))  # expansion factor H(a)/H0
    exp_fac_k0 = np.sqrt(omega_m * a ** -3 + omega_l * a ** (-3 * (1 + wa + w0)) * np.exp(
        3 * wa * (a - 1)))  # expansion factor H(a)/H0 at k=0 (zero curvature)

    n = np.arange(15)
    t1 = ((5 / (5 + 2 * n[:, None])) * poch(3 / 2, n)[:, None] * (((-1) ** n) / (factorial(n)))[:, None] * (
        q[None, :]) ** (n[:, None] / 3) * ((1 + q)[None, :]) ** (-n[:, None]) * ((x[None, :]) ** (n[:, None])))
    t2 = (hyp2f1(((w0 - 1) / (2 * w0))[None, :], (2 * n[:, None] - 1) / (3 * w0[None, :]),
                 1 - (5 + 2 * n[:, None]) / (6 * w0[None, :]), -q[None, :]))

    terms = t1 * t2

    ret = np.sum(terms, axis=0) * a * exp_fac / exp_fac_k0

    return ret


def index_array(dim:int, tot:int):  # has dimension dim along axis 1
    
    """
    list of indices for computing Hermite polynomials from Eq. A2 given dimension and total degree. 
    this function gives all 'dim'-element integer lists that sum to 'tot', where 'dim' is the dimension for interpolation.
    """
    if dim == 1:
        return np.array([[tot]], dtype = int)
    ar = np.ones((1, dim), dtype = int)  # initiate array, this element is chopped off at the end
    for k in np.arange(1, tot):
        prev = index_array(dim - 1, tot - k)
        size = np.shape(prev)[0]
        ks = k * np.ones((size, 1), dtype = int)
        ark = np.concatenate([ks, prev], axis = 1)
        ar = np.concatenate([ar, ark], axis = 0)
    return ar[1:, :]  # chop off first element

def rbf_interpolator(unit_cube_points, param_config, bmat): 
    
    """
    Computes eq. A14 for given set of unit cube points and parameter configuration, where bmat = Phi * g. 
    """
    dim, eps, alpha, beta, deltasq, max_deg, ind_array_m = param_config
    md = max_deg - dim 
    exp_fac = np.exp(- deltasq * np.linalg.norm(unit_cube_points, axis=1) ** 2)
    herm = eval_hermite(np.arange(md)[:, None], alpha * beta * unit_cube_points[:, None, :])
    ev = np.prod(herm[:, ind_array_m, np.arange(dim)], axis = 2)
    psi = (exp_fac[:, None] * ev)  

    return np.dot(psi, bmat.T) 
