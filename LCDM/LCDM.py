#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:13:59 2024

@author: thomasbakx
"""

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import CubicSpline
from Utils.cobra_helper import get_svd_results, get_loop_table, get_rbf_tables, get_angles_and_gl_weights, get_rbf_param_config, get_bounds, \
growth_factor_fid, growth_factor, growth_rate, rbf_interpolator
from Utils.cobra_error import ParamRangeError, NBasisError, KRangeError, DimensionError, ConfigError 
from Utils.convert_units import convert_units_k_pk_to_hfid

class CobraLCDM:
    def __init__(self, param_range:str):

        """
        Class for computations with LCDM cosmologies.
        Parameter range can either be 'def' or 'ext'.
        Can compute one-loop and linear matter power spectrum and one-loop and linear galaxy power spectrum.
        Includes optional resummation with wiggle-no-wiggle split.
        
        """
        
        if not (param_range == 'def' or param_range == 'ext'):
            raise ParamRangeError("param_range must be either 'def' or 'ext'")
            
        self.param_range = param_range 
        self.space = 'LCDM'
        
        ## result from the svd

        k_pk_bar, k_vj_hat, k_vj_nw, k_vj = get_svd_results(space = self.space, param_range = self.param_range)
        k_lin, vj_hat = k_vj_hat[0], k_vj_hat[1:]
        vj_nw = k_vj_nw[1:]
        vj = k_vj[1:]
        vj_w = vj - vj_nw

        self.k_pk_bar = k_pk_bar

        self.s_tables_lin = {"k_lin": k_lin, "vj_hat": vj_hat, "vj": vj, "vj_nw": vj_nw, "vj_w": vj_w}

        k_loop, gg1loop_nw = get_loop_table(space = self.space, param_range = self.param_range, w_nw = "nw")
        k_loop, gg1loop_w = get_loop_table(space = self.space, param_range = self.param_range, w_nw = "w")
        
        self.s_tables_loop = {"k_loop": k_loop, "gg1loop_nw": gg1loop_nw, "gg1loop_w": gg1loop_w, 
                              "mm1loop_nw": gg1loop_nw[0], "mm1loop_w": gg1loop_w[0]}

        ## multipole calculation: Gauss-Legendre quadrature as in velocileptors 
        mu, ang = get_angles_and_gl_weights()
        self.mu = mu
        self.ang = ang
        
        ## rbf tables
        bmat_disp, bmat_gr, bmat_d, bmat_wts = get_rbf_tables(space = self.space, param_range = self.param_range)
        
        self.bmat_disp = bmat_disp
        self.bmat_wts = bmat_wts
        self.bmat_gr = bmat_gr
        self.bmat_d = bmat_d
        
        ## rbf parameter configs
        
        self.param_config_3d = get_rbf_param_config(dim = 3)
        
        ## bounds for parameter spaces

        bounds_cbns, bounds_mha = get_bounds(space = self.space, param_range = self.param_range)
        
        self.bounds_cbns = bounds_cbns
        self.bounds_mha = bounds_mha

        self.param_keys = ['omch2', 'ombh2', 'ns', 'As', 'h', 'z']
        
        ## bias ordering as in velocileptors: see e.g. 
        #https://github.com/sfschen/velocileptors/blob/master/velocileptors/EPT/ept_fullresum_fftw.py
        #and https://arxiv.org/pdf/2005.00523,  https://arxiv.org/abs/2012.04636. 
        
        self.bias_keys = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4', 'bfog']
        
        self.bias_keys_lin = ['b1']
        self.bias_keys_ctr = ['csq']
        
    ################ methods ##################
    
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

        if self.param_range == 'def':
            if n_basis > 12:
                raise NBasisError("Can use at most 12 basis functions for this range")
        elif self.param_range == 'ext':
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")
        else:
            raise ValueError(f'Unknown param_range {self.param_range}') ## added same error as in GEN.py
        
        cosmo_keys_list = [key for key in cosmo.keys()]
        if cosmo_keys_list != self.param_keys:
            raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, 10^9*As, h, z.")
            
        cosm_tot = np.array([val for val in cosmo.values()]).T

        cbns = cosm_tot[:, :3] #c,b,ns

        As_over_Asfid = cosm_tot[:, 3] / 2
    
        mha = np.delete(cosm_tot, [2, 3], 1)
        mha[:, 1] = mha[:, 0] + mha[:, 1]
        mha = np.delete(mha, 0, 1)
        mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

        cbns_unit_cube = qmc.scale(cbns, self.bounds_cbns[0], self.bounds_cbns[1], reverse = True)
        mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)
    
        wts_rbf = rbf_interpolator(cbns_unit_cube, self.param_config_3d, self.bmat_wts[:n_basis]) # n_cosmo x (n_basis)
        growth_fac_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, self.bmat_d)[:, 0] # n_cosmo

            ## rescale disps and wts by A_s and growth factor squared
        growth_over_growth_fid = growth_fac_rbf * growth_factor(mha.T) / growth_factor_fid(mha[:, 0])
            
        wts = wts_rbf * (As_over_Asfid * growth_over_growth_fid ** 2)[:, None] 

        return wts

    
    def rbf_growth_rate(self, cosmo:dict[str,list]):

        """
        Compute growth rate f given an input cosmology dictionary. 
        Uses rbf interpolation as described in 2407.04660. 
        Accepts vector of cosmologies (i.e. every dict value is a 1d array of the same length).
        Outputs 1d array of size n_cosmo. 
        
        """
        
        cosmo_keys_list = [key for key in cosmo.keys()]
        if cosmo_keys_list != self.param_keys:
            raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, As, h, z.")
            
        cosm_tot = np.array([val for val in cosmo.values()]).T

        mha = np.delete(cosm_tot, [2, 3], 1)
        mha[:, 1] = mha[:, 0] + mha[:, 1]
        mha = np.delete(mha, 0, 1)
        mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

        mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)

        growth_rate_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, self.bmat_gr)[:,0] # n_cosmo
        return growth_rate_rbf * growth_rate(mha.T) 

    def rbf_sigma_squared_bao(self, cosmo:dict[str,list]):

        """
        Compute displacement dispersion Sigma^2 given an input cosmology dictionary. 
        Uses rbf interpolation as described in 2407.04660. 
        Accepts vector of cosmologies (i.e. every dict value is a 1d array of the same length).
        Outputs 1d array of size n_cosmo. 
        
        """
        
        cosmo_keys_list = [key for key in cosmo.keys()]
        if cosmo_keys_list != self.param_keys:
            raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, As, h, z.")
            
        cosm_tot = np.array([val for val in cosmo.values()]).T

        cbns = cosm_tot[:, :3] #c,b,ns

        As_over_Asfid = cosm_tot[:,3] / 2
    
        mha = np.delete(cosm_tot, [2, 3], 1)
        mha[:, 1] = mha[:, 0] + mha[:, 1]
        mha = np.delete(mha, 0, 1)
        mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

        cbns_unit_cube = qmc.scale(cbns, self.bounds_cbns[0], self.bounds_cbns[1], reverse = True)
        mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)

        disps_rbf = rbf_interpolator(cbns_unit_cube, self.param_config_3d, self.bmat_disp)[:,0] # n_cosmo 
        growth_fac_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, self.bmat_d)[:,0] # n_cosmo

        growth_over_growth_fid = growth_fac_rbf * growth_factor(mha.T) / growth_factor_fid(mha[:, 0])
            ## rescale disps by A_s and growth factor squared
        disps = disps_rbf * As_over_Asfid * growth_over_growth_fid ** 2

        return disps

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
        
        if self.param_range == 'def':
            if n_basis > 12:
                raise NBasisError("Can use at most 12 basis functions for this range")
        elif self.param_range == 'ext':
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")

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
                      weights:np.ndarray = None, resum:bool = False, disps_hfid:np.ndarray = None):

        """
        Compute linear matter power spectrum given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. If disps_hfid (1d array) is provided (in units (Mpc/hfid)^2) this is used for displacement, otherwise emulated via rbf.
        Returns output power spectrum at bins k_out_hfid (in cosmology-independent units). 
        """
        
        k_lin_internal = self.s_tables_lin["k_lin"]
        
        if len(n_basis_list) != 1:
            raise NBasisError("Need one number of basis functions for linear theory")

        n_basis = n_basis_list[0]

        if self.param_range == 'def':
            if n_basis > 12:
                raise NBasisError("Can use at most 12 basis functions for this range")
        elif self.param_range == 'ext':
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")

        if k_out_hfid[-1] > k_lin_internal[-1] or k_out_hfid[0] < k_lin_internal[0]:
            raise KRangeError("Wavenumber out of bounds")

        if weights is not None:
            if cosmo is not None:
                raise ConfigError("If you provide weights, put cosmo = None")
            wts = weights ## wts is n_cosmo x n_basis 
            if resum:
                if disps_hfid is None:
                    raise ConfigError("If weights are provided and resum = True, displacements must also be provided")
                else:
                    disps = disps_hfid

        else:
            cosmo_keys_list = [key for key in cosmo.keys()]
            if cosmo_keys_list != self.param_keys:
                raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, As, h, z.")
            
            cosm_tot = np.array([val for val in cosmo.values()]).T

            cbns = cosm_tot[:, :3] #c,b,ns

            As_over_Asfid = cosm_tot[:, 3] / 2
    
            mha = np.delete(cosm_tot, [2, 3], 1)
            mha[:, 1] = mha[:, 0] + mha[:, 1]
            mha = np.delete(mha, 0, 1)
            mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

            cbns_unit_cube = qmc.scale(cbns, self.bounds_cbns[0], self.bounds_cbns[1], reverse = True)
            mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)

            bmat_disps_wts = np.concatenate([self.bmat_disp, self.bmat_wts[:n_basis]])
    
            disps_wts_rbf = rbf_interpolator(cbns_unit_cube, self.param_config_3d, bmat_disps_wts) # n_cosmo x (1 + n_basis)
            growth_fac_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, self.bmat_d)[:, 0] # n_cosmo

            ## rescale disps and wts by A_s and growth factor squared
            growth_over_growth_fid = growth_fac_rbf * growth_factor(mha.T) / growth_factor_fid(mha[:, 0])
            
            disps_wts = disps_wts_rbf * (As_over_Asfid * growth_over_growth_fid ** 2)[:, None] 

            wts = disps_wts[:,1:] # wts is n_cosmo x n_basis 
            
            if disps_hfid is None: #displacement is set and computed even if resum = False (negligible overhead)
                disps = disps_wts[:, 0]  
            else:
                disps = disps_hfid 

        ## assemble all the pieces
        if resum:
            vj_nw, vj_w = self.s_tables_lin["vj_nw"], self.s_tables_lin["vj_w"]
            vj_ir = vj_nw[None, :n_basis, :] + vj_w[None, :n_basis, :] * np.exp(- (k_lin_internal ** 2)[None, None, :] * disps[:, None, None])
            pk = np.einsum('ab, abk -> ak', wts, vj_ir, optimize = 'greedy')

        else:
            vj = self.s_tables_lin["vj"]
            pk = wts @ vj[:n_basis]

        pk_out_hfid = CubicSpline(np.log10(k_lin_internal), pk, axis = 1)(np.log10(k_out_hfid))
        
        return pk_out_hfid 


    def oneloop_galaxy_power_at_mu(self, cosmo:dict[str,list], bias:dict[str,list], k_out_hfid:np.ndarray, mu_out:np.ndarray, 
                                   n_basis_list:list[int], growth_rates:np.ndarray = None, weights:np.ndarray = None, resum:bool = True,  
                                   disps_hfid:np.ndarray = None, has_linear:bool = True):

        """
        Compute one-loop galaxy power spectrum in redshift space given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. If disps_hfid (1d array) is provided (in units (Mpc/hfid)^2) this is used for displacement, otherwise emulated via rbf.
        Returns output power spectrum at bins k_out_hfid (in cosmology-independent units) and mu_out, i.e. n_cosmo x n_mu x n_k.
        Needs number of basis functions for both linear and one-loop calculation.
        This calculation used the EPT resummed model of velocileptors, but subtracts the k->0 contribution from the loop integrals.
        """

        k_loop_internal =  self.s_tables_loop["k_loop"]

        bias_keys_list = [key for key in bias.keys()]
        if bias_keys_list != self.bias_keys:
            raise DimensionError("Please specify 12 biases: b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4, bFoG")
            
        bias_arr = np.array([val for val in bias.values()]) ## convert to array
        
        bias_shape = np.shape(bias_arr)
            
        # bias vector
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4, bfog = bias_arr

        if len(n_basis_list) != 2:
            raise NBasisError("Need two numbers of basis functions for one loop: linear and one loop")
        
        n_basis_lin = n_basis_list[0]
        n_basis_loop = n_basis_list[1]
        n_basis_max = np.max(n_basis_list)

        if self.param_range == 'def':
            if n_basis_max > 12:
                raise NBasisError("Can use at most 12 basis functions for this range")
        elif self.param_range == 'ext':
            if n_basis_max > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")

        if k_out_hfid[-1] > k_loop_internal[-1] or k_out_hfid[0] < k_loop_internal[0]:
            raise KRangeError("Wavenumber out of bounds")

        lin_comp = 0 ## linear theory is not subtracted unless has_linear = False
        
        if not has_linear:
            b1_dict = {'b1':b1}
            lin_comp = self.linear_galaxy_power_at_mu(cosmo, b1_dict, k_out_hfid, mu_out, [n_basis_lin], growth_rates = growth_rates,
                                                        weights = weights, resum = resum, disps_hfid = disps_hfid)
            
        ### compute weights, growth rate and disps - either passed as args or computed from cosmo 
    
        if weights is not None:
            if cosmo is not None:
                raise ConfigError("If you provide weights, put cosmo = None")
            n_cosmo = np.shape(weights)[0]
            if bias_shape[1] != n_cosmo:
                raise DimensionError("Number of bias parameter combinations must match number of cosmologies provided")
            if np.shape(weights)[1] < n_basis_max:
                raise ConfigError("Not enough weights provided: number of requested basis functions exceeds number of weights") 

            else: 
                wts = weights[:, :n_basis_max] ## wts is n_cosmo x n_basis_max 
                wts_lin = wts[:, :n_basis_lin]
                wts_loop = wts[:, :n_basis_loop]
                
            if resum:
                if disps_hfid is None:
                    raise ConfigError("If weights are provided and resum = True, displacements must also be provided")
                else:
                    disps = disps_hfid

            if growth_rates is None:
                raise ConfigError("If weights are provided, growth rate must also be provided")
            else:
                f = growth_rates 

        else:
            cosmo_keys_list = [key for key in cosmo.keys()]
            if cosmo_keys_list != self.param_keys:
                raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, As, h, z.")
            
            cosm_tot = np.array([val for val in cosmo.values()]).T
            n_cosmo = np.shape(cosm_tot)[0]
            if bias_shape[1] != n_cosmo:
                raise DimensionError("Number of bias parameter combinations must match number of cosmologies provided")

            cbns = cosm_tot[:, :3] #c,b,ns

            As_over_Asfid = cosm_tot[:, 3] / 2
    
            mha = np.delete(cosm_tot, [2, 3], 1)
            mha[:, 1] = mha[:, 0] + mha[:, 1]
            mha = np.delete(mha, 0, 1)
            mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

            cbns_unit_cube = qmc.scale(cbns, self.bounds_cbns[0], self.bounds_cbns[1], reverse = True)
            mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)

            bmat_disps_wts = np.concatenate([self.bmat_disp, self.bmat_wts[:n_basis_max]])
            bmat_d_f = np.concatenate([self.bmat_d, self.bmat_gr])
    
            disps_wts_rbf = rbf_interpolator(cbns_unit_cube, self.param_config_3d, bmat_disps_wts) # n_cosmo x (1 + n_basis_max)
            d_f_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, bmat_d_f) # n_cosmo x 2

            ## rescale disps and wts by A_s and growth factor squared
            growth_over_growth_fid = d_f_rbf[:, 0] * growth_factor(mha.T) / growth_factor_fid(mha[:, 0])
            
            disps_wts = disps_wts_rbf * (As_over_Asfid * growth_over_growth_fid ** 2)[:, None] 

            wts = disps_wts[:,1:] # wts is n_cosmo x n_basis_max
            wts_lin = wts[:, :n_basis_lin]
            wts_loop = wts[:, :n_basis_loop]
            
            if disps_hfid is None:
                disps = disps_wts[:, 0]  
            else:
                disps = disps_hfid 

            if growth_rates is None:
                f = d_f_rbf[:, 1] * growth_rate(mha.T)
            else:
                f = growth_rates 
                
        if not resum:
            disps = np.zeros(n_cosmo)

        ### weights, disps and growth rate are now set - move on to loop calculation

        wts_tens = np.einsum("ki,kj->ijk", wts_loop, wts_loop)  # n_basis_loop x n_basis_loop x n_cosmo
        gg1loop_nw, gg1loop_w = self.s_tables_loop["gg1loop_nw"], self.s_tables_loop["gg1loop_w"]
    
        mu2 = (mu_out ** 2)[None, None, :]
        mu4 = (mu_out ** 4)[None, None, :]
        mu6 = (mu_out ** 6)[None, None, :]
        k2 = (k_loop_internal ** 2)[:, None, None]
        k4 = (k_loop_internal ** 4)[:, None, None]
        disps_tens = disps[None, :, None]
        f_tens = f[None, :, None]
    
        # damping factor = n_k_loop x n_cosmo x n_mu
    
        damp_e = -disps_tens * k2 * (1 + (f_tens * (2 + f_tens)) * mu2)
        damp_f = np.exp(damp_e)
    
        # use tensordot to make an angle-dependent bias vector
    
        pk_bias = np.tensordot(np.array([b1 ** 2, b1 * b2, b1 * bs, b2 ** 2, b2 * bs, bs ** 2, b1 * b3]),
                               np.ones(len(mu_out)), axes=0)
        vk_bias = np.tensordot(np.array([b1 * f, b1 ** 2 * f, b2 * f, b1 * b2 * f, bs * f, b1 * bs * f, b3 * f]),
                               - mu_out ** 2, axes=0)
    
        s0k_bias = np.tensordot(np.array([1 * f ** 2, b1 * f ** 2, b1 ** 2 * f ** 2, b2 * f ** 2, bs * f ** 2]),
                                (- 0.5 * mu_out ** 2), axes = 0)
        s2k_bias = np.tensordot(np.array([1 * f ** 2, b1 * f ** 2, b1 ** 2 * f ** 2, b2 * f ** 2, bs * f ** 2]),
                                (- 0.5 * mu_out ** 2) * (0.5 * (3 * mu_out ** 2 - 1)), axes=0)
    
        g1k_bias = np.tensordot(np.array([1 * f ** 3, b1 * f ** 3]),
                                (1 / 6 * mu_out ** 3) * mu_out, axes=0)
        g3k_bias = np.tensordot(np.array([f ** 3, b1 * f ** 3]),
                                (1 / 6 * mu_out ** 3) * mu_out ** 3, axes=0)
    
        k0k_bias = np.tensordot(np.array([f ** 4]), 1 / 24 * mu_out ** 4, axes=0)
        k2k_bias = np.tensordot(np.array([f ** 4]), 1 / 24 * mu_out ** 4 * mu_out ** 2, axes=0)
        k4k_bias = np.tensordot(np.array([f ** 4]), 1 / 24 * mu_out ** 4 * mu_out ** 4, axes=0)
    
        biases = [pk_bias, vk_bias, s0k_bias, s2k_bias, g1k_bias, g3k_bias, k0k_bias, k2k_bias, k4k_bias]
        bias_vec_quad = np.concatenate(biases, axis=0)
    
        # tensor contractions for the quadratic part
        
        td1_nw = np.einsum('sijr,ijk,skm->rkm', gg1loop_nw[:, :n_basis_loop, :n_basis_loop, :], wts_tens, bias_vec_quad,
                           optimize=['einsum_path', (0, 1), (0, 1)])
        td1_w = np.einsum('sijr,ijk,skm->rkm', gg1loop_w[:, :n_basis_loop, :n_basis_loop, :], wts_tens, bias_vec_quad,
                          optimize=['einsum_path', (0, 1), (0, 1)])
    
        quad_tot = np.moveaxis(td1_nw + td1_w * damp_f, 0, -1)
        
        #td2_nw = np.einsum('rkm,ml->klr', td1_nw, self.ang, optimize='greedy')  # no-wiggle loop, n_cosmo x n_ell x n_k_loop
        #td2_w = np.einsum('rkm,rkm,ml->klr', td1_w, damp_f, self.ang, optimize='greedy')  # wiggle loop, n_cosmo x n_ell x n_k_loop
    
        # linear part:
        k_lin_internal, vj_nw, vj_w = self.s_tables_lin["k_lin"], self.s_tables_lin["vj_nw"], self.s_tables_lin["vj_w"]
    
        plin_nw = (wts_lin @ vj_nw[:n_basis_lin]).T  #  n_k_lin x n_cosmo
        plin_w = (wts_lin @ vj_w[:n_basis_lin]).T  # n_k_lin x n_cosmo

        plin_nw_kloop = CubicSpline(np.log10(k_lin_internal), plin_nw)(np.log10(k_loop_internal))
        plin_w_kloop = CubicSpline(np.log10(k_lin_internal), plin_w)(np.log10(k_loop_internal))

        plin_nw_tens = plin_nw_kloop[:, :, None]
        plin_w_tens = plin_w_kloop[:, :, None]

        alpha0_tens = alpha0[None, :, None]
        alpha2_tens = alpha2[None, :, None]
        alpha4_tens = alpha4[None, :, None]
        alpha6_tens = alpha6[None, :, None]
        b1_tens = b1[None, :, None]
        bfog_tens = bfog[None, :, None]
    
        lin0 = (b1_tens + f_tens * mu2) ** 2 * (plin_nw_tens + damp_f * plin_w_tens)
    
        # compensation for IR resummation
        lin1 = - damp_e * damp_f * (b1_tens + f_tens * mu2) ** 2 * plin_w_tens
    
        # counterterms
        lin2 = ((alpha0_tens + alpha2_tens * mu2 + alpha4_tens * mu4 + alpha6_tens * mu6) * k2 \
                - bfog_tens * mu4 * k4 * (b1_tens + f_tens * mu2) ** 2) * (plin_nw_tens + damp_f * plin_w_tens)
    
        #lin_tot = np.einsum('rkm,ml->klr', lin0 + lin1 + lin2, self.ang, optimize='greedy')
        lin_tot = np.moveaxis(lin0 + lin1 + lin2, 0, -1)

        # constant part:

        const = sn[None, :, None] + sn2[None, :, None] * k2 * mu2 + sn4[None, :, None] * k4 * mu4
        #const_tot = np.einsum('rkm,ml->klr', const, self.ang, optimize='greedy')
        
        const_tot = np.moveaxis(const, 0, -1)
    
        ###############################################
    
        pmuk_out_hfid = CubicSpline(np.log10(k_loop_internal), quad_tot + lin_tot + const_tot, axis=2)(np.log10(k_out_hfid))

        pmuk_out_hfid = pmuk_out_hfid - lin_comp ## subtract linear theory if has_linear = False
    
        return pmuk_out_hfid # n_cosmo x n_mu x n_k_out
    
    def oneloop_galaxy_power_multipoles(self, cosmo:dict[str,list], bias:dict[str:list], k_out_hfid:np.ndarray, n_basis_list:list[int],
                            growth_rates:np.ndarray = None, weights:np.ndarray = None, resum:bool = True, disps_hfid:np.ndarray = None, 
                            has_linear:bool = True):

        """
        Compute one-loop galaxy power spectrum in redshift space given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. If disps_hfid (1d array) is provided (in units (Mpc/hfid)^2) this is used for displacement, otherwise emulated via rbf.
        Returns power spectrum monopole, quadrupole, hexadecapole at bins k_out_hfid (in cosmology-independent units), i.e. n_cosmo x 3 x n_k.
        Needs number of basis functions for both linear and one-loop calculation.
        This calculation used the EPT resummed model of velocileptors, but subtracts the k->0 contribution from the loop integrals.
        """

        mu_out = self.mu
        
        pmuk_out_hfid = self.oneloop_galaxy_power_at_mu(cosmo, bias, k_out_hfid, mu_out, n_basis_list, 
                                                 growth_rates, weights, resum, disps_hfid, has_linear)
        
        pellk_out_hfid = np.einsum('kmr,ml->klr', pmuk_out_hfid, self.ang, optimize='greedy')

        return pellk_out_hfid

    
    def linear_galaxy_power_at_mu(self, cosmo:dict[str,list], bias:dict[str,list], k_out_hfid:np.ndarray, mu_out:np.ndarray, 
                                  n_basis_list:list[int], growth_rates:np.ndarray = None, weights:np.ndarray = None, resum:bool = True, 
                                  disps_hfid:np.ndarray = None):
        """
        Compute linear galaxy power spectrum in redshift space given input cosmology dict or weights (2d array of size n_cosmo x n_weights). 
        Optional resum. If disps_hfid (1d array) is provided (in units (Mpc/hfid)^2) this is used for displacement, otherwise emulated via rbf. 
        If growth_rates is provided this is used for displacement, otherwise emulated via rbf. 
        Returns output power spectrum at bins k_out_hfid (in cosmology-independent units) and mu_out, i.e. n_cosmo x n_mu x n_k.
        """
        
        k_lin_internal = self.s_tables_lin["k_lin"]

        bias_keys_list = [key for key in bias.keys()]
        if bias_keys_list != self.bias_keys_lin:
            raise DimensionError("Please specify one bias: b1")
            
        bias_arr = np.array([val for val in bias.values()]) ## convert to array
        
        bias_shape = np.shape(bias_arr)

        # bias vector
        b1 = bias_arr       
        if bias_shape[1] != 1:
            raise DimensionError("Please specify one bias: b1")

        if len(n_basis_list) != 1:
            raise NBasisError("Need one number of basis functions for linear theory")

        n_basis = n_basis_list[0]

        if self.param_range == 'def':
            if n_basis > 12:
                raise NBasisError("Can use at most 12 basis functions for this range")
        elif self.param_range == 'ext':
            if n_basis > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")

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
            if resum:
                if disps_hfid is None:
                    raise ConfigError("If weights are provided and resum = True, displacements must also be provided")
                else:
                    disps = disps_hfid

            if growth_rates is None:
                raise ConfigError("If weights are provided, growth rates must also be provided")
            else:
                f = growth_rates

        else:
            cosmo_keys_list = [key for key in cosmo.keys()]
            if cosmo_keys_list != self.param_keys:
                raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, As, h, z.")
            
            cosm_tot = np.array([val for val in cosmo.values()]).T
            n_cosmo = np.shape(cosm_tot)[0]
            if bias_shape[1] != n_cosmo:
                raise DimensionError("Number of bias parameter combinations must match number of cosmologies provided")

            cbns = cosm_tot[:, :3] #c,b,ns

            As_over_Asfid = cosm_tot[:, 3] / 2
    
            mha = np.delete(cosm_tot, [2, 3], 1)
            mha[:, 1] = mha[:, 0] + mha[:, 1]
            mha = np.delete(mha, 0, 1)
            mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

            cbns_unit_cube = qmc.scale(cbns, self.bounds_cbns[0], self.bounds_cbns[1], reverse = True)
            mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)

            bmat_disps_wts = np.concatenate([self.bmat_disp, self.bmat_wts[:n_basis]])
            bmat_d_f = np.concatenate([self.bmat_d, self.bmat_gr])
    
            disps_wts_rbf = rbf_interpolator(cbns_unit_cube, self.param_config_3d, bmat_disps_wts) # n_cosmo x (1 + n_basis_max)
            d_f_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, bmat_d_f) # n_cosmo x 2

            ## rescale disps and wts by A_s and growth factor squared
            growth_over_growth_fid = d_f_rbf[:, 0] * growth_factor(mha.T) / growth_factor_fid(mha[:, 0])
            
            disps_wts = disps_wts_rbf * (As_over_Asfid * growth_over_growth_fid ** 2)[:, None] 

            wts = disps_wts[:,1:] # wts is n_cosmo x n_basis_max
            
            if disps_hfid is None:
                disps = disps_wts[:, 0]  
            else:
                disps = disps_hfid 

            if growth_rates is None:
                f = d_f_rbf[:, 1] * growth_rate(mha.T)
            else:
                f = growth_rates 
                
        if not resum:
            disps = np.zeros(n_cosmo)

        mu2 = (mu_out ** 2)[None, None, :]
        disps_tens = disps[None, :, None]
        f_tens = f[None, :, None]
        
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


        #lin_tot = np.einsum('rkm,ml->klr', lin0 + lin1 + lin2, self.ang, optimize='greedy')
        lin_tot = np.moveaxis(lin0, 0, -1)

        pmuk_out_hfid = CubicSpline(np.log10(k_lin_internal), lin_tot, axis=2)(np.log10(k_out_hfid))

        return pmuk_out_hfid
        

    def linear_galaxy_power_multipoles(self, cosmo:dict[str,list], bias:dict[str,list], k_out_hfid:np.ndarray, n_basis_list:list[int],
                          growth_rates:np.ndarray = None, weights:np.ndarray = None, resum:bool = True, disps_hfid:np.ndarray = None):
        """
        Compute linear galaxy power spectrum multipoles given cosmology dict or weights (2d array of size n_cosmo x n_weights) and bias dict. 
        Optional resum. If disps_hfid (1d array) is provided (in units (Mpc/hfid)^2) this is used for displacement, otherwise emulated via rbf.
        Integration via Gauss-Legendre scheme as in velocileptors.
        Returns output monopole, quadrupole and hexadecapole power spectrum at bins k_out_hfid (in h_fid units), i.e. n_cosmo x 3 x n_k.
        """
        
        mu_out = self.mu

        pmuk_out_hfid = self.linear_galaxy_power_at_mu(cosmo, bias, k_out_hfid, mu_out, n_basis_list,
                                                 growth_rates, weights, resum, disps_hfid)
        
        pellk_out_hfid = np.einsum('kmr,ml->klr', pmuk_out_hfid, self.ang, optimize='greedy')

        return pellk_out_hfid

    def oneloop_matter_power(self, cosmo:dict[str,list], bias:dict[str,list], k_out_hfid:np.ndarray, n_basis_list:list[int],
                        weights:np.ndarray = None, resum:bool = True, disps_hfid:np.ndarray = None, has_linear:bool = True):
        # TODO: turn this into PyDoc format
        """
        Compute one-loop matter power spectrum given cosmology dict or weights (2d array of n_cosmo x n_weights) and one counterterm. 
        Optional resum. If disps_hfid (1d array) is provided (in units (Mpc/hfid)^2) this is used for displacement, otherwise emulated via rbf.
        Returns output power spectrum at bins k_out_hfid (in h_fid units), i.e. n_cosmo x n_k.
        """

        k_loop_internal =  self.s_tables_loop["k_loop"]

        ctr_keys_list = [key for key in bias.keys()]
        if ctr_keys_list != self.bias_keys_ctr:
            raise DimensionError("Please specify one counterterm: csq")
            
        ctr_arr = np.array([val for val in bias.values()]) ## convert to array
        
        bias_shape = np.shape(ctr_arr)

        if len(n_basis_list) != 2:
            raise NBasisError("Need two numbers of basis functions for one loop: linear and one loop")

        csq = ctr_arr[0]
        
        n_basis_lin = n_basis_list[0]
        n_basis_loop = n_basis_list[1]
        n_basis_max = np.max(n_basis_list)

        if self.param_range == 'def':
            if np.max(n_basis_list) > 12:
                raise NBasisError("Can use at most 12 basis functions for this range")
        elif self.param_range == 'ext':
            if np.max(n_basis_list) > 16:
                raise NBasisError("Can use at most 16 basis functions for this range")

        if k_out_hfid[-1] > k_loop_internal[-1] or k_out_hfid[0] < k_loop_internal[0]:
            raise KRangeError("Wavenumber out of bounds")

        lin_comp = 0 ## linear theory is not subtracted unless has_linear = False
        
        if not has_linear:
            lin_comp = self.linear_matter_power(cosmo, k_out_hfid, [n_basis_lin], weights = weights, resum = resum, 
                                          disps_hfid = disps_hfid)
            
        ### compute weights, growth rate and disps - either passed as args or computed from cosmo 
    
        if weights is not None:
            if cosmo is not None:
                raise ConfigError("If you provide weights, put cosmo = None")
            n_cosmo = np.shape(weights)[0]
            if bias_shape[1] != n_cosmo:
                raise DimensionError("Number of bias parameter combinations must match number of cosmologies provided")
            if np.shape(weights)[1] < n_basis_max:
                raise ConfigError("Not enough weights provided: number of requested basis functions exceeds number of weights")  

            else: 
                wts = weights[:, :n_basis_max] ## wts is n_cosmo x n_basis_max 
                wts_lin = wts[:, :n_basis_lin]
                wts_loop = wts[:, :n_basis_loop]
                
            if resum:
                if disps_hfid is None:
                    raise ConfigError("If weights are provided and resum = True, displacements must also be provided")
                else:
                    disps = disps_hfid
                    
        else:
            cosmo_keys_list = [key for key in cosmo.keys()]
            if cosmo_keys_list != self.param_keys:
                raise DimensionError("Please specify 6 parameters for every cosmology: the order is omch2, ombh2, ns, As, h, z.")
            
            cosm_tot = np.array([val for val in cosmo.values()]).T
            n_cosmo = np.shape(cosm_tot)[0]
            if bias_shape[1] != n_cosmo:
                raise DimensionError("Number of bias parameter combinations must match number of cosmologies provided")


            cbns = cosm_tot[:, :3] #c,b,ns

            As_over_Asfid = cosm_tot[:, 3] / 2
    
            mha = np.delete(cosm_tot, [2, 3], 1)
            mha[:, 1] = mha[:, 0] + mha[:, 1]
            mha = np.delete(mha, 0, 1)
            mha[:, 2] = 1 / (1 + mha[:, 2])  # c+b,h,a

            cbns_unit_cube = qmc.scale(cbns, self.bounds_cbns[0], self.bounds_cbns[1], reverse = True)
            mha_unit_cube = qmc.scale(mha, self.bounds_mha[0], self.bounds_mha[1], reverse = True)

            bmat_disps_wts = np.concatenate([self.bmat_disp, self.bmat_wts[:n_basis_max]])
    
            disps_wts_rbf = rbf_interpolator(cbns_unit_cube, self.param_config_3d, bmat_disps_wts) # n_cosmo x (1 + n_basis_max)
            growth_fac_rbf = rbf_interpolator(mha_unit_cube, self.param_config_3d, self.bmat_d)[:, 0] # n_cosmo

            ## rescale disps and wts by A_s and growth factor squared
            growth_over_growth_fid = growth_fac_rbf * growth_factor(mha.T) / growth_factor_fid(mha[:, 0])
            
            disps_wts = disps_wts_rbf * (As_over_Asfid * growth_over_growth_fid ** 2)[:, None] 

            wts = disps_wts[:,1:] # wts is n_cosmo x n_basis_max
            wts_lin = wts[:, :n_basis_lin]
            wts_loop = wts[:, :n_basis_loop]
            
            if disps_hfid is None:
                disps = disps_wts[:, 0]  
            else:
                disps = disps_hfid 

        if not resum:
            disps = np.zeros(n_cosmo)

        ## onto the loop: 

        wts_tens = np.einsum("ki,kj->ijk", wts_loop, wts_loop)  # n_basis_loop x n_basis_loop x n_cosmo
        mm1loop_nw, mm1loop_w = self.s_tables_loop["mm1loop_nw"], self.s_tables_loop["mm1loop_w"]
  
        k2 = (k_loop_internal ** 2)[:, None]
        disps_tens = disps[None, :]
        csq_tens = csq[None, :]
    
        # damping factor = n_k_loop x n_cosmo
    
        damp_e = -disps_tens * k2
        damp_f = np.exp(damp_e)

        quad_part_nw = np.einsum("ijk,ijr -> kr", wts_tens, mm1loop_nw[:n_basis_loop, :n_basis_loop, :])
        quad_part_w = np.einsum("ijk,rk,ijr -> kr", wts_tens, damp_f, mm1loop_w[:n_basis_loop, :n_basis_loop, :])

        # linear part

        k_lin_internal, vj_nw, vj_w = self.s_tables_lin["k_lin"], self.s_tables_lin["vj_nw"], self.s_tables_lin["vj_w"]
    
        plin_nw = (wts_lin @ vj_nw[:n_basis_lin]).T  #  n_k_lin x n_cosmo
        plin_w = (wts_lin @ vj_w[:n_basis_lin]).T  # n_k_lin x n_cosmo

        plin_nw_kloop = CubicSpline(np.log10(k_lin_internal), plin_nw)(np.log10(k_loop_internal)) ## spline onto loop wavenumbers 
        plin_w_kloop = CubicSpline(np.log10(k_lin_internal), plin_w)(np.log10(k_loop_internal))

        ## counterterm with default knl = 1 hfid / Mpc
        lin0 = (plin_nw_kloop + damp_f * plin_w_kloop) 
        - 2 * (2 * np.pi) * csq_tens * k_loop_internal ** 2 * (plin_nw_kloop + damp_f * plin_w_kloop)
    
        # compensation for IR resummation
        lin1 = - damp_e * damp_f * plin_w_kloop

        ploopk_out_hfid = CubicSpline(np.log10(k_loop_internal), (lin0 + lin1).T + quad_part_nw + quad_part_w, axis=1)(np.log10(k_out_hfid))

        ploopk_out_hfid = ploopk_out_hfid - lin_comp

        return ploopk_out_hfid

        

        

        



        

        
    
    
    
    