#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:14:52 2025

@author: thomasbakx
"""

import numpy as np
from cobra.GEN.GEN import CobraGEN
from cobra.LCDM.LCDM import CobraLCDM

from cobra.Utils.ir_resummed_power import ir_resummed_power
from cobra.Utils.extrapolate_pk import extrapolate_pk

## check: using weights and disps or using cosmo within cobra range gives the same answer
def test_weights_or_cosmo():
    
    # LCDM
    
    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    bias = {'b1':[1.5]}
    bias_loop = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[50], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[3000], 'sn2':[3000], 'sn4':[3000], 'bfog':[2]} 
    ctr = {'csq':[2]}

    # calculate weights and disps
    weights_rbf = cobra_lcdm.rbf_weights(params, n_basis_list=[12])
    disps_rbf = cobra_lcdm.rbf_sigma_squared_bao(params)
    growth_rbf = cobra_lcdm.rbf_growth_rate(params)

    linpow1 = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True)
    linpow2 = cobra_lcdm.linear_matter_power(cosmo = None, k_out_hfid = k_out_hfid, n_basis_list = [12], 
                                             weights = weights_rbf, resum = True, disps_hfid = disps_rbf)

    gglin1 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12], resum = True)
    gglin2 = cobra_lcdm.linear_galaxy_power_multipoles(cosmo = None, bias = bias, k_out_hfid = k_out_hfid, n_basis_list = [12], 
                                                       growth_rates = growth_rbf, weights = weights_rbf, 
                                                       resum = True, disps_hfid = disps_rbf)
    
    ggloop1 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias_loop, k_out_hfid, n_basis_list = [12,12], resum = True)
    ggloop2 = cobra_lcdm.oneloop_galaxy_power_multipoles(cosmo = None, bias = bias_loop, k_out_hfid = k_out_hfid, n_basis_list = [12,12], 
                                                       growth_rates = growth_rbf, weights = weights_rbf, 
                                                       resum = True, disps_hfid = disps_rbf)
    
    mmloop1 = cobra_lcdm.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = True)
    mmloop2 = cobra_lcdm.oneloop_matter_power(cosmo = None, ctr = ctr, k_out_hfid = k_out_hfid, n_basis_list = [12,12], weights = weights_rbf, 
                                                       resum = True, disps_hfid = disps_rbf)
    

    # GEN
    cobra_gen = CobraGEN(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params2 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0], 'h': [0.68], 'ns':[0.97], 
               'Mnu':[0.4], 'w0':[-1.1], 'wa':[-0.2], 'z':[0.5], 'As': [3]} 

    # calculate weights and disps
    weights_rbf2 = cobra_gen.rbf_weights(params2, n_basis_list=[16])

    linpow3 = cobra_gen.linear_matter_power(params2, k_out_hfid, n_basis_list = [16], resum = False)
    linpow4 = cobra_gen.linear_matter_power(cosmo = None, k_out_hfid = k_out_hfid, n_basis_list = [16], 
                                             weights = weights_rbf2, resum = False)

    gglin3 = cobra_gen.linear_galaxy_power_multipoles(params2, bias, k_out_hfid, n_basis_list = [16], 
                                                      growth_rates = growth_rbf, resum = False)
    gglin4 = cobra_gen.linear_galaxy_power_multipoles(cosmo = None, bias = bias, k_out_hfid = k_out_hfid, 
                                                      n_basis_list = [16], growth_rates = growth_rbf, 
                                             weights = weights_rbf2, resum = False)
    
    assert np.allclose(linpow1,linpow2, rtol = 1e-5) 
    assert np.allclose(linpow3,linpow4, rtol = 1e-5)
    assert np.allclose(gglin1,gglin2, rtol = 1e-5)
    assert np.allclose(gglin3, gglin4, rtol = 1e-5)
    assert np.allclose(ggloop1,ggloop2, rtol = 1e-5)
    assert np.allclose(mmloop1,mmloop2, rtol = 1e-5)
    
## check: using disps directly or using cosmo within cobra range gives the same answer
def test_disps_or_cosmo():
    # LCDM
    
    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 

    # calculate weights and disps
    disps_rbf = cobra_lcdm.rbf_sigma_squared_bao(params)

    linpow1 = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True)
    linpow2 = cobra_lcdm.linear_matter_power(params, k_out_hfid = k_out_hfid, n_basis_list = [12], 
                                            resum = True, disps_hfid = disps_rbf)
    
    assert np.allclose(linpow1,linpow2, rtol = 1e-5)
    
## check: using growth rate directly or using cosmo within cobra range gives the same answer
def test_growth_rate_or_cosmo():

    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    bias_lin = {'b1':[1]}
    bias = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[50], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[3000], 'sn2':[3000], 'sn4':[3000], 'bfog':[2]}

    growth_rbf = cobra_lcdm.rbf_growth_rate(params)

    ## linear
    
    gglin1 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12], resum = True)
    gglin2 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12], 
                                                       resum = True, growth_rates = growth_rbf)

    ggloop1 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], resum = True)
    ggloop2 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], 
                                                       resum = True, growth_rates = growth_rbf)


    assert np.allclose(gglin1, gglin2, rtol = 1e-5)
    assert np.allclose(ggloop1, ggloop2, rtol = 1e-5)
    
## check that growth rates in LCDM are overwritten when they are specified by user
def test_growth_rate_override():
    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    bias_lin = {'b1':[1]}

    bias = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[50], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[3000], 'sn2':[3000], 'sn4':[3000], 'bfog':[2]}

    growth_rbf = cobra_lcdm.rbf_growth_rate(params)

    gglin1 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12])
    gglin2 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12], growth_rates = 0.5*growth_rbf)

    ggloop1 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], resum = True)
    ggloop2 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], 
                                                       resum = True, growth_rates = 0.5*growth_rbf)

    assert not np.allclose(gglin1, gglin2, rtol = 1e-5)
    assert not np.allclose(ggloop1, ggloop2, rtol = 1e-5)
    

## check that disps are overwritten when they are specified by user
def test_disps_override():
    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    bias = {'b1':[1]}

    disps = cobra_lcdm.rbf_sigma_squared_bao(params)

    gglin1 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12])
    gglin2 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12], disps_hfid = 0.5 * disps)

    assert not np.allclose(gglin1, gglin2, rtol = 1e-5)
    
## check: using GEN or using LCDM gives the same answer for LCDM cosmologies for linear matter and galaxy power
def test_gen_or_lcdm():

    cobra_lcdm = CobraLCDM(param_range = 'def')
    cobra_gen = CobraGEN(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params1 = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [3], 'h': [0.68], 'z':[1]} 
    params2 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0], 'h': [0.68], 'ns':[0.97], 
               'Mnu':[0], 'w0':[-1], 'wa':[0], 'z':[1], 'As': [3]}
    bias = {'b1':[0.5]}

    linpow1 = cobra_lcdm.linear_matter_power(params1, k_out_hfid, n_basis_list = [12], resum = False)
    linpow2 = cobra_gen.linear_matter_power(params2, k_out_hfid, n_basis_list = [16], resum = False)

    gglin1 = cobra_lcdm.linear_galaxy_power_multipoles(params1, bias, k_out_hfid, n_basis_list = [12], 
                                                       growth_rates = np.array([1]), resum = False)
    gglin2 = cobra_gen.linear_galaxy_power_multipoles(params2, bias, k_out_hfid, n_basis_list = [16], 
                                                      growth_rates = np.array([1]), resum = False)

    assert np.allclose(linpow1, linpow2, rtol = 5e-3)
    assert np.allclose(gglin1, gglin2, rtol = 5e-3) ## a little more cushion for Mnu = 0
    
## check: default and extended cosmologies give the same answer when overlapping for lcdm
def test_def_or_ext_lcdm():

    cobra_def = CobraLCDM(param_range = 'def')
    cobra_ext = CobraLCDM(param_range = 'ext')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    ctr = {'csq':[-0.5]}
    bias_lin = {'b1':[0.5]}
    bias_nl = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[50], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[3000], 'sn2':[3000], 'sn4':[3000], 'bfog':[0]}

    linpow1 = cobra_def.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True)
    linpow2 = cobra_ext.linear_matter_power(params, k_out_hfid, n_basis_list = [16], resum = True)

    gglin1 = cobra_def.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12], 
                                                       growth_rates = np.array([1]), resum = False)
    gglin2 = cobra_ext.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [16], 
                                                      growth_rates = np.array([1]), resum = False)

    mmloop1 = cobra_def.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = True)
    mmloop2 = cobra_ext.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [16,16], resum = True)

    ggloop1 = cobra_def.oneloop_galaxy_power_multipoles(params, bias_nl, k_out_hfid, n_basis_list = [12,12], resum = True)
    ggloop2 = cobra_ext.oneloop_galaxy_power_multipoles(params, bias_nl, k_out_hfid, n_basis_list = [16,16], resum = True)    

    assert np.allclose(linpow1,linpow2, rtol = 1e-4)
    assert np.allclose(gglin1,gglin2, rtol = 1e-4)
    assert np.allclose(mmloop1,mmloop2, rtol = 1e-4)
    assert np.allclose(ggloop1,ggloop2, rtol = 1e-4)
    
## check: default and extended cosmologies give the same answer when overlapping for gen
def test_def_or_ext_gen():

    cobra_def = CobraGEN(param_range = 'def')
    cobra_ext = CobraGEN(param_range = 'ext')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params1 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0], 'h': [0.68], 'ns':[0.97], 
               'Mnu':[0.4], 'w0':[-1.1], 'wa':[0.2], 'z':[0.5], 'As': [3]}
    params2 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0], 'h': [0.68], 'ns':[0.97], 
               'Mnu':[0.4], 'w0':[-1.1], 'wp':[-0.9], 'z':[0.5], 'As': [3]}
    bias = {'b1':[0.5]}

    linpow1 = cobra_def.linear_matter_power(params1, k_out_hfid, n_basis_list = [16])
    linpow2 = cobra_ext.linear_matter_power(params2, k_out_hfid, n_basis_list = [16])

    gglin1 = cobra_def.linear_galaxy_power_multipoles(params1, bias, k_out_hfid, n_basis_list = [16], 
                                                       growth_rates = np.array([1]), resum = False)
    gglin2 = cobra_ext.linear_galaxy_power_multipoles(params2, bias, k_out_hfid, n_basis_list = [16], 
                                                      growth_rates = np.array([1]), resum = False)

    assert np.allclose(linpow1, linpow2, rtol = 1e-3)
    assert np.allclose(gglin1, gglin2, rtol = 1e-3)
    
## check: using resum or resumming with ir_resummed_power gives the same answer
def test_direct_or_indirect_resum():

    ## LCDM 

    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.7], 'z':[1]} ## need h = 0.7 for this test

    linpow1 = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True)
    linpow2 = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12])
    k_pk2 = np.array([k_out_hfid, linpow2[0]])
    linpow2_res = ir_resummed_power(k_pk2)[1]

    assert np.allclose(linpow1[0], linpow2_res, rtol = 1e-3)
    
    
## check: projecting a power spectrum yields the same weights as computing them directly
def test_projection_weights():

    ## LCDM

    cobra_lcdm = CobraLCDM(param_range = 'ext')
    k_min_hfid = 0.0008
    k_max_hfid = 4
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.7], 'z':[1]} ## need h = 0.7 for this test

    linpow = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12])
    linpow_extrap = extrapolate_pk(np.array([k_out_hfid, linpow[0]]))
    wts_project = cobra_lcdm.project_pk_into_basis(linpow_extrap, h = 0.7, n_basis_list = [12])[0]

    wts_true = cobra_lcdm.rbf_weights(params, n_basis_list = [12])[0]

    ## GEN, def 

    cobra_gen = CobraGEN(param_range = 'def')
    k_min_hfid = 0.001
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params2 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0.1], 'h': [0.7], 'ns':[0.97], 
               'Mnu':[0.4], 'w0':[-1.1], 'wa':[0.2], 'z':[0.5], 'As': [3]} ## need h = 0.7 for this test

    linpow = cobra_gen.linear_matter_power(params2, k_out_hfid, n_basis_list = [16])
    linpow_extrap = extrapolate_pk(np.array([k_out_hfid, linpow[0]]))
    wts_project2 = cobra_gen.project_pk_into_basis(linpow_extrap, h = 0.7, n_basis_list = [16])[0]

    wts_true2 = cobra_gen.rbf_weights(params2, n_basis_list = [12])[0]
    
    ##  GEN, ext
    
    cobra_gen = CobraGEN(param_range = 'ext')
    k_min_hfid = 0.001
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params3 = {'omch2':[0.10], 'ombh2':[0.021], 'Omk':[0.1], 'h': [0.7], 'ns':[0.97], 
               'Mnu':[0.6], 'w0':[-1.1], 'wp':[-0.7], 'z':[0.5], 'As': [3]} ## need h = 0.7 for this test

    linpow = cobra_gen.linear_matter_power(params3, k_out_hfid, n_basis_list = [16])
    linpow_extrap = extrapolate_pk(np.array([k_out_hfid, linpow[0]]))
    wts_project3 = cobra_gen.project_pk_into_basis(linpow_extrap, h = 0.7, n_basis_list = [16])[0]

    wts_true3 = cobra_gen.rbf_weights(params3, n_basis_list = [12])[0]
    
    assert np.allclose(wts_true[:4], wts_project[:4],rtol = 1e-2)
    assert np.allclose(wts_true2[:4], wts_project2[:4],rtol = 1e-2)
    assert np.allclose(wts_true3[:4], wts_project3[:4],rtol = 1e-2)
    
    
    

## check: resum = False gives same answer as putting disps = 0
def test_resum_or_noresum():

    ## LCDM 

    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    bias = {'b1': [1]}
    bias_nl = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[50], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[3000], 'sn2':[3000], 'sn4':[3000], 'bfog':[0]}
    ctr = {'csq': [2]}

    linpow1 = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True, disps_hfid = np.array([0]))
    linpow2 = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = False)

    galpow1 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12], 
                                                        resum = True, disps_hfid = np.array([0]))
    galpow2 = cobra_lcdm.linear_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12], resum = False)

    mmloop1 = cobra_lcdm.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = True, disps_hfid = np.array([0]))
    mmloop2 = cobra_lcdm.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = False)

    ggloop1 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias_nl, k_out_hfid, n_basis_list = [12,12], 
                                                        resum = True, disps_hfid = np.array([0]))
    ggloop2 = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias_nl, k_out_hfid, n_basis_list = [12,12], resum = False)


    ## GEN 

    cobra_gen = CobraGEN(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params1 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0], 'h': [0.68], 'ns':[0.97], 
               'Mnu':[0.4], 'w0':[-1.1], 'wa':[0.2], 'z':[0.5], 'As': [3]}
    bias = {'b1': [1]}
    ctr = {'csq': [2]}

    linpow3 = cobra_gen.linear_matter_power(params1, k_out_hfid, n_basis_list = [16], resum = True, disps_hfid = np.array([0]))
    linpow4 = cobra_gen.linear_matter_power(params1, k_out_hfid, n_basis_list = [16], resum = False)

    galpow3 = cobra_gen.linear_galaxy_power_multipoles(params1, bias, k_out_hfid, n_basis_list = [16], growth_rates = np.array([1]),
                                                        resum = True, disps_hfid = np.array([0]))
    galpow4 = cobra_gen.linear_galaxy_power_multipoles(params1, bias, k_out_hfid, n_basis_list = [16], 
                                                       growth_rates = np.array([1]), resum = False)

    assert np.allclose(linpow1,linpow2, rtol = 1e-5)
    assert np.allclose(mmloop1,mmloop2, rtol = 1e-5)
    assert np.allclose(linpow3,linpow4, rtol = 1e-5)
    assert np.allclose(galpow1,galpow2, rtol = 1e-5)
    assert np.allclose(galpow3,galpow4, rtol = 1e-5)
    assert np.allclose(ggloop1,ggloop2, rtol = 1e-5)

    
## check: linear galaxy power spectrum reduces to linear matter power spectrum when growth = 0 and b1 = 1
def test_galaxy_or_matter_linear():

    # LCDM
    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    bias = {'b1': [1]}

    linpow = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True)
    galpow = cobra_lcdm.linear_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12], 
                                                       growth_rates = np.array([0]), resum = True)
    ## GEN
    cobra_gen = CobraGEN(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 1.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params2 = {'omch2':[0.13], 'ombh2':[0.021], 'Omk':[0], 'h': [0.68], 'ns':[0.97], 
               'Mnu':[0.4], 'w0':[-1.1], 'wa':[0.2], 'z':[0.5], 'As': [3]}
    bias = {'b1': [1]}

    linpow2 = cobra_gen.linear_matter_power(params2, k_out_hfid, n_basis_list = [16], resum = False)
    galpow2 = cobra_gen.linear_galaxy_power_multipoles(params2, bias, k_out_hfid, n_basis_list = [16], 
                                                       growth_rates = np.array([0]), resum = False)

    assert np.allclose(linpow[0],galpow[0,0], rtol = 1e-4)
    assert np.allclose(linpow2[0],galpow2[0,0], rtol = 1e-4)
    
## check: one loop galaxy power spectrum reduces to one loop matter power spectrum when growth = 0 and b1 = 1
def test_galaxy_or_matter_oneloop():
    # cobra object
    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    ctr = {'csq': [0]}
    bias = {'b1':[1], 'b2':[0], 'bs':[0], 'b3':[0], \
               'alpha0':[0], 'alpha2':[0], 'alpha4':[0], 'alpha6':[0], \
               'sn':[0], 'sn2':[0], 'sn4':[0], 'bfog':[0]}

    matpow = cobra_lcdm.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = True)
    galpow = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], 
                                                       growth_rates = np.array([0]), resum = True)

    assert np.allclose(matpow[0],galpow[0,0], rtol = 1e-4)
    
## check: difference between has_linear = True and has_linear = False is exactly plinear for matter
def test_has_linear_difference_matter():

    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    ctr = {'csq': [10]}
    

    matpow_nolin = cobra_lcdm.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = True, has_linear = False)
    matpow_oneloop = cobra_lcdm.oneloop_matter_power(params, ctr, k_out_hfid, n_basis_list = [12,12], resum = True)
    matpow_lin = cobra_lcdm.linear_matter_power(params, k_out_hfid, n_basis_list = [12], resum = True)

    assert np.allclose(matpow_lin + matpow_nolin, matpow_oneloop, rtol = 1e-5) 
    
def test_has_linear_difference_galaxy():

    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
    
    bias = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[50], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[3000], 'sn2':[3000], 'sn4':[3000], 'bfog':[0]}
    bias_lin = {'b1':[1.68593608]}

    galpow_nolin = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], 
                                                              resum = True, has_linear = False)
    galpow_oneloop = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], resum = True)
    galpow_lin = cobra_lcdm.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12], resum = True)

    assert np.allclose(galpow_lin + galpow_nolin, galpow_oneloop, rtol = 1e-5)
    
## check that the large-scale limit of one-loop galaxy recovers linear theory
def test_large_scale_limit():

    cobra_lcdm = CobraLCDM(param_range = 'def')
    k_min_hfid = 0.002
    k_max_hfid = 0.5
    n_bins = 300
    k_out_hfid = np.logspace(np.log10(k_min_hfid), np.log10(k_max_hfid), n_bins)
    params = {'omch2':[0.13], 'ombh2':[0.021], 'ns':[0.97], 'As': [2], 'h': [0.68], 'z':[1]} 
   
    bias = {'b1':[1.68593608], 'b2':[-1.17], 'bs':[-0.715], 'b3':[-0.479], \
               'alpha0':[0], 'alpha2':[50], 'alpha4':[50], 'alpha6':[50], \
               'sn':[0], 'sn2':[0], 'sn4':[0], 'bfog':[0]}
    bias_lin = {'b1':[1.68593608]}

    galpow_oneloop = cobra_lcdm.oneloop_galaxy_power_multipoles(params, bias, k_out_hfid, n_basis_list = [12,12], resum = True)
    galpow_lin = cobra_lcdm.linear_galaxy_power_multipoles(params, bias_lin, k_out_hfid, n_basis_list = [12], resum = True)

    assert np.allclose(galpow_lin[:,:,:20], galpow_oneloop[:,:,:20], rtol = 1e-3)
    