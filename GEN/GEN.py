from scipy.special import expit 
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import CubicSpline
from Utils.cobra_helper import get_svd_results, get_rbf_tables, get_angles_and_gl_weights, get_rbf_param_config, get_bounds, \
    growth_factor_bar, rbf_interpolator
from Utils.cobra_error import ParamRangeError, NBasisError, KRangeError, DimensionError, ConfigError
from Utils.convert_units import convert_units_k_pk_to_hfid

class CobraGEN:
    def __init__(self, param_range:str):
        """
        Class for computations with generalized cosmologies including curvature, dynamical dark energy and massive neutrinos.
        Parameter range can either be 'def' or 'ext' as in the LCDM class.
        Can compute linear matter power spectrum and linear galaxy power spectrum including optional resummation with wiggle-no-wiggle split.
        For now no one-loop features available. 
        
        """
        
        if not (param_range == 'def' or param_range == 'ext'):
            raise ParamRangeError("param_range must be either 'def' or 'ext'")
            
        self.param_range = param_range 
        self.space = 'GEN'
        
        # result from the svd

        k_pk_bar, k_vj_hat, k_vj_nw, k_vj = get_svd_results(space = self.space, param_range = self.param_range)
        k_lin, vj_hat = k_vj_hat[0], k_vj_hat[1:]
        vj_nw = k_vj_nw[1:]
        vj = k_vj[1:]
        vj_w = vj - vj_nw

        self.k_pk_bar = k_pk_bar

        self.s_tables_lin = {"k_lin": k_lin, "vj_hat": vj_hat, "vj": vj, "vj_nw": vj_nw, "vj_w": vj_w}

        # multipole calculation: Gauss-Legendre quadrature as in velocileptors
        mu, ang = get_angles_and_gl_weights()
        self.mu = mu
        self.ang = ang
        
        # rbf tables

        bmat_d, bmat_wts = get_rbf_tables(space = self.space, param_range = self.param_range)
        
        self.bmat_wts = bmat_wts
        self.bmat_d = bmat_d
        
        # rbf parameter configs
        
        self.param_config_9d = get_rbf_param_config(dim = 9)
        self.param_config_6d = get_rbf_param_config(dim = 6)
        
        # bounds for parameter spaces

        bounds_9d, bounds_6d = get_bounds(space = self.space, param_range = self.param_range)
        
        self.bounds_9d = bounds_9d
        self.bounds_6d = bounds_6d

        if param_range == 'def':
            self.cosmo_keys = ['omch2', 'ombh2', 'Omk', 'h', 'ns', 'Mnu', 'w0', 'wa', 'z', 'As']

        if param_range == 'ext':
            self.cosmo_keys = ['omch2', 'ombh2', 'Omk', 'h', 'ns', 'Mnu', 'w0', 'wp', 'z', 'As']

        self.bias_keys_lin = ['b1']
        
    # methods ##################

    def rbf_weights(self, cosmo:dict[str,list], n_basis_list:list[int]):
        """
        Compute weights (coefficients of scale functions) given an input cosmology dictionary. 
        Uses rbf interpolation as described in 2407.04660. 
        Accepts vector of cosmologies (i.e. every dict value is a 1d array of the same length).
        Outputs 2d array of n_cosmo x n_basis. 
        
        """

        if len(n_basis_list) != 1:
            raise NBasisError("Need one number of basis functions for rbf weights")
            
        n_basis = n_basis_list[0]
        if n_basis > 16:
            raise NBasisError("Can use at most 16 basis functions for this range")

        cosmo_keys_list = [key for key in cosmo.keys()]
        if cosmo_keys_list != self.cosmo_keys:
            raise DimensionError(f"Please specify 10 parameters for every cosmology: {', '.join(self.cosmo_keys)}")
        if self.param_range == 'def':
            alpha = 1
        elif self.param_range == 'ext':
            alpha = 0.35
        else:
            raise ValueError(f'Unknown param_range {self.param_range}')
                
        cosm_tot = np.array([val for val in cosmo.values()]).T

        as_over_asfid = cosm_tot[:, -1] / 2
        cosm_tot = cosm_tot[:, :-1]
        cosm_6d = np.delete(cosm_tot, [4, 5], 1)
        cosm_6d[:, 1] = cosm_6d[:, 0] + cosm_6d[:, 1]
        cosm_6d = np.delete(cosm_6d, 0, 1)
        cosm_6d[:, -1] = 1 / (1 + cosm_6d[:, -1])  # c+b,k,h,w0,wa or wp,a

        unit_cube_9d = qmc.scale(cosm_tot, self.bounds_9d[0], self.bounds_9d[1], reverse = True)
        unit_cube_6d = qmc.scale(cosm_6d, self.bounds_6d[0], self.bounds_6d[1], reverse = True)
    
        wts_rbf = rbf_interpolator(unit_cube_9d, self.param_config_9d, self.bmat_wts[:n_basis + 1]) # n_cosmo x (1 + n_basis)
        growth_fac_rbf = rbf_interpolator(unit_cube_6d, self.param_config_6d, self.bmat_d)[:,0] # n_cosmo

        p_over_growth_sq_rbf = wts_rbf[:,0]
        if self.param_range == 'ext':
            p_over_growth_sq_rbf = 1 / (expit(p_over_growth_sq_rbf) * 100)  # undo logit transformation (see App. C)

        # rescale disps and wts by A_s and growth factor squared
        growth_over_growth_fid = growth_fac_rbf * (growth_factor_bar(cosm_6d.T, self.param_range)) ** alpha

        wts = wts_rbf[:,1:] * (as_over_asfid * p_over_growth_sq_rbf * growth_over_growth_fid ** 2)[:, None]

        return wts 

    def project_pk_into_basis(self, k_pk:np.ndarray, h:float, n_basis_list:list[int]): ## not vectorized
        
        """
        Project a given input power spectrum (2d array of size 2 x n_bins) into the cobra basis.
        Input units need to be specified via float h.
        The power spectrum provided must contain the cobra range. 
        Returns tuple containing weights and the resulting projected power spectrum over the cobra range.
        Not vectorized (not possible to pass multiple input power spectra)
        """

        if len(n_basis_list) != 1:
            raise NBasisError("Need one number of basis functions for projection")
        
        n_basis = n_basis_list[0]
        
        k_lin_internal = self.s_tables_lin["k_lin"]

        k_pk_hfid = convert_units_k_pk_to_hfid(k_pk, h)
        k_hfid = k_pk_hfid[0]
        
        if k_hfid[-1] < k_lin_internal[-1] or k_hfid[0] > k_lin_internal[0]:
            raise KRangeError("Wavenumber range not broad enough: must contain k_lin") 

        else:
            pk_hfid = CubicSpline(np.log10(k_hfid),k_pk_hfid[1])(np.log10(k_lin_internal))
            
        pk_bar = self.k_pk_bar[1]
        vj_hat = self.s_tables_lin["vj_hat"]
        vj = self.s_tables_lin["vj"]
        wts = np.dot(vj_hat[:n_basis], pk_hfid / pk_bar)
        pk_cobra = wts @ vj[:n_basis]
        k_pk_cobra = np.array([k_lin_internal, pk_cobra])

        return wts, k_pk_cobra

    def linear_matter_power(self, cosmo:dict[str,list], k_out_hfid:np.ndarray, n_basis_list:list[int], 
                            weights:np.ndarray = None, resum:bool = False,
                            disps_hfid:np.ndarray = None):
        
        """
        Compute linear matter power spectrum given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. 
        Returns output power spectrum at bins k_out_hfid (in cosmology-independent units). 
        """
        
        k_lin_internal = self.s_tables_lin["k_lin"]
        
        if len(n_basis_list) != 1:
            raise NBasisError("Need one number of basis functions for linear theory")

        n_basis = n_basis_list[0]

        if self.param_range == 'def':
            alpha = 1
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")
        elif self.param_range == 'ext':
            alpha = 0.35
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")
        else:
            raise ValueError(f'Unknown param_range {self.param_range}')

        if k_out_hfid[-1] > k_lin_internal[-1] or k_out_hfid[0] < k_lin_internal[0]:
            raise KRangeError("Wavenumber out of bounds")

        if weights is None:
            cosmo_keys_list = [key for key in cosmo.keys()]
            if self.param_range == 'def':
                if cosmo_keys_list != self.cosmo_keys:
                    raise DimensionError("Please specify 10 parameters for every cosmology: omch2, ombh2, Omk, h, ns, Mnu, w0, wa, z, As")
            elif self.param_range == 'ext':
                if cosmo_keys_list != self.cosmo_keys:
                    raise DimensionError("Please specify 10 parameters for every cosmology: omch2, ombh2, Omk, h, ns, Mnu, w0, wp, z, As")
                
            cosm_tot = np.array([val for val in cosmo.values()]).T
            
            as_over_asfid = cosm_tot[:, -1] / 2
            cosm_tot = cosm_tot[:, :-1]
            cosm_6d = np.delete(cosm_tot, [4, 5], 1)
            cosm_6d[:, 1] = cosm_6d[:, 0] + cosm_6d[:, 1]
            cosm_6d = np.delete(cosm_6d, 0, 1)
            cosm_6d[:, -1] = 1 / (1 + cosm_6d[:, -1])  # c+b,k,h,w0,wa or wp,a

            unit_cube_9d = qmc.scale(cosm_tot, self.bounds_9d[0], self.bounds_9d[1], reverse = True)
            unit_cube_6d = qmc.scale(cosm_6d, self.bounds_6d[0], self.bounds_6d[1], reverse = True)
    
            wts_rbf = rbf_interpolator(unit_cube_9d, self.param_config_9d, self.bmat_wts[:n_basis + 1]) # n_cosmo x (1 + n_basis)
            growth_fac_rbf = rbf_interpolator(unit_cube_6d, self.param_config_6d, self.bmat_d)[:,0] # n_cosmo

            p_over_growth_sq_rbf = wts_rbf[:,0]
            if self.param_range == 'ext':
                p_over_growth_sq_rbf = 1 / (expit(p_over_growth_sq_rbf) * 100)  # undo logit transformation (see App. C)

            # rescale disps and wts by A_s and growth factor squared
            growth_over_growth_fid = growth_fac_rbf * (growth_factor_bar(cosm_6d.T, self.param_range)) ** alpha

            wts = wts_rbf[:,1:] * (as_over_asfid * p_over_growth_sq_rbf * growth_over_growth_fid ** 2)[:, None]
        else:
            if cosmo is not None:
                raise ConfigError("If you provide weights, put cosmo = None")
            wts = weights  ## wts is n_cosmo x n_basis

        # assemble all the pieces
        if resum:
            if disps_hfid is None:
                raise ConfigError("If resum = True, displacements must also be provided")
            else:
                disps = disps_hfid
            vj_nw, vj_w = self.s_tables_lin["vj_nw"], self.s_tables_lin["vj_w"]
            vj_ir = vj_nw[None, :n_basis, :] + vj_w[None, :n_basis, :] * np.exp(- (k_lin_internal ** 2)[None, None, :] * disps[:, None, None])
            pk = np.einsum('ab, abk -> ak', wts, vj_ir, optimize = 'greedy')
        else:
            vj = self.s_tables_lin["vj"]
            pk = wts @ vj[:n_basis]

        pk_out_hfid = CubicSpline(np.log10(k_lin_internal), pk, axis = 1)(np.log10(k_out_hfid))
        
        return pk_out_hfid

    def linear_galaxy_power_at_mu(self, cosmo:dict[str,list], bias:dict[str,list], k_out_hfid:np.ndarray, mu_out:np.ndarray, 
                                  n_basis_list:list[int], growth_rates:np.ndarray, weights:np.ndarray = None, resum:bool = False,
                                  disps_hfid:np.ndarray = None):

        """
        Compute linear galaxy power spectrum in redshift space given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. Note default resum = False here, since displacement cannot be emulated yet. Growth rates must also be provided -
        cannot be emulated yet.
        Returns output power spectrum at bins k_out_hfid (in cosmology-independent units) and mu_out, i.e. n_cosmo x n_mu x n_k.
        """
        
        k_lin_internal = self.s_tables_lin["k_lin"]

        bias_keys_list = [key for key in bias.keys()]
        if bias_keys_list != self.bias_keys_lin:
            raise DimensionError("Please specify one bias: b1")
            
        bias_arr = np.array([val for val in bias.values()]) ## convert to array
        
        bias_shape = np.shape(bias_arr)

        # bias vector
        if bias_shape[1] != 1:
            raise DimensionError("Please specify one bias: b1")

        if len(n_basis_list) != 1:
            raise NBasisError("Need one number of basis functions for linear theory")

        n_basis = n_basis_list[0]

        if self.param_range == 'def':
            alpha = 1
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")
        elif self.param_range == 'ext':
            alpha = 0.35
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")
        else:
            raise ValueError(f'Unknown param_range {self.param_range}')

        if k_out_hfid[-1] > k_lin_internal[-1] or k_out_hfid[0] < k_lin_internal[0]:
            raise KRangeError("Wavenumber out of bounds")

        if weights is not None:
            if cosmo is not None:
                raise ConfigError("If you provide weights, put cosmo = None")
            n_cosmo = np.shape(weights)[0]
            if bias_shape[1] != n_cosmo:
                raise DimensionError("Number of bias parameter combinations must match number of cosmologies provided")
            if np.shape(weights)[1] < n_basis:
                raise ConfigError("Not enough weights provided: number of requested basis functions exceeds number of weights") 

            else: 
                wts = weights[:, :n_basis] ## wts is n_cosmo x n_basis_max

        else:
            cosmo_keys_list = [key for key in cosmo.keys()]
            if self.param_range == 'def':
                if cosmo_keys_list != self.cosmo_keys:
                    raise DimensionError("Please specify 10 parameters for every cosmology: omch2, ombh2, Omk, h, ns, Mnu, w0, wa, z, As")
            elif self.param_range == 'ext':
                if cosmo_keys_list != self.cosmo_keys:
                    raise DimensionError("Please specify 10 parameters for every cosmology: omch2, ombh2, Omk, h, ns, Mnu, w0, wp, z, As")
                
            cosm_tot = np.array([val for val in cosmo.values()]).T
            n_cosmo = np.shape(cosm_tot)[0] ## added n_cosmo definition when weights = None
            
            as_over_asfid = cosm_tot[:, -1] / 2
            cosm_tot = cosm_tot[:, :-1]
            cosm_6d = np.delete(cosm_tot, [4, 5], 1)
            cosm_6d[:, 1] = cosm_6d[:, 0] + cosm_6d[:, 1]
            cosm_6d = np.delete(cosm_6d, 0, 1)
            cosm_6d[:, -1] = 1 / (1 + cosm_6d[:, -1])  # c+b,k,h,w0,wa or wp,a

            unit_cube_9d = qmc.scale(cosm_tot, self.bounds_9d[0], self.bounds_9d[1], reverse = True)
            unit_cube_6d = qmc.scale(cosm_6d, self.bounds_6d[0], self.bounds_6d[1], reverse = True)
    
            wts_rbf = rbf_interpolator(unit_cube_9d, self.param_config_9d, self.bmat_wts[:n_basis + 1]) # n_cosmo x (1 + n_basis)
            growth_fac_rbf = rbf_interpolator(unit_cube_6d, self.param_config_6d, self.bmat_d)[:,0] # n_cosmo

            p_over_growth_sq_rbf = wts_rbf[:,0]
            if self.param_range == 'ext':
                p_over_growth_sq_rbf = 1 / (expit(p_over_growth_sq_rbf) * 100)  # undo logit transformation (see App. C)

            # rescale disps and wts by A_s and growth factor squared
            growth_over_growth_fid = growth_fac_rbf * (growth_factor_bar(cosm_6d.T, self.param_range)) ** alpha

            wts = wts_rbf[:,1:] * (as_over_asfid * p_over_growth_sq_rbf * growth_over_growth_fid ** 2)[:, None]
            
        ## moved resum block below weights calculation 
        if resum:
            if disps_hfid is None:
                raise ConfigError("If resum = True, displacements must also be provided")
            else:
                disps = disps_hfid
        else:
            disps = np.zeros(n_cosmo)

        ## moved out of if-statement 
        mu2 = (mu_out ** 2)[None, None, :]
        disps_tens = disps[None, :, None]
        f_tens = growth_rates[None, :, None]
        
        k_lin_internal, vj_nw, vj_w = self.s_tables_lin["k_lin"], self.s_tables_lin["vj_nw"], self.s_tables_lin["vj_w"]

        k2 = (k_lin_internal ** 2)[:, None, None]
        damp_e = -disps_tens * k2 * (1 + (f_tens * (2 + f_tens)) * mu2)
        damp_f = np.exp(damp_e) #n_k_lin x n_cosmo x n_mu
    
        plin_nw = (wts @ vj_nw[:n_basis]).T  #  n_k_lin x n_cosmo
        plin_w = (wts @ vj_w[:n_basis]).T  # n_k_lin x n_cosmo

        plin_nw_tens = plin_nw[:, :, None]
        plin_w_tens = plin_w[:, :, None]

        b1 = bias_arr[:,0]

        b1_tens = b1[None, :, None]
 
        lin0 = (b1_tens + f_tens * mu2) ** 2 * (plin_nw_tens + damp_f * plin_w_tens)

        # lin_tot = np.einsum('rkm,ml->klr', lin0 + lin1 + lin2, self.ang, optimize='greedy')
        lin_tot = np.moveaxis(lin0, 0, -1)

        pmuk_out_hfid = CubicSpline(np.log10(k_lin_internal), lin_tot, axis=2)(np.log10(k_out_hfid))

        return pmuk_out_hfid

    def linear_galaxy_power_multipoles(self, cosmo:dict[str,list], bias:dict[str,list], k_out_hfid:np.ndarray, n_basis_list:list[int], 
                                       growth_rates:np.ndarray, weights:np.ndarray = None, resum:bool = False, disps_hfid:np.ndarray = None):

        """
        Compute linear galaxy power spectrum multipoles given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. 
        Integration via Gauss-Legendre scheme as in velocileptors.
        Returns output monopole, quadrupole and hexadecapole power spectrum at bins k_out_hfid (in h_fid units), i.e. n_cosmo x 3 x n_k.
        """
        
        mu_out = self.mu
        
        pmuk_out_hfid = self.linear_galaxy_power_at_mu(cosmo, bias, k_out_hfid, mu_out, n_basis_list, growth_rates, 
                                                       weights, resum, disps_hfid)
        
        pellk_out_hfid = np.einsum('kmr,ml->klr', pmuk_out_hfid, self.ang, optimize='greedy')

        return pellk_out_hfid
        