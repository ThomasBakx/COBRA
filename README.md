# COBRA
COBRA Repository for computing cosmological observables

Currently supports: (i) linear power spectrum in LambdaCDM; (ii) linear power spectrum in extended parameter space including curvature, neutrinos, w0, wa; (iii) one-loop EPT resummed power spectrum of biased tracers in redshift space for LambdaCDM based on Chen, Vlah, White: https://arxiv.org/pdf/2005.00523. 

By default, the code uses units of hfid/Mpc for wavenumbers, where hfid = 0.7 throughout. This means that you also get the power spectra in units of (Mpc/hfid)**3. If you want to change this to h/Mpc or 1/Mpc, set the 'units' keyword as desired when calling COBRA. The allowed ranges are given in units of hfid/Mpc and are [0.001,4] for (i), [0.001,1.5] for (ii) and [0.001,0.5] for (iii).   

All modules are vectorized, so you can feed L cosmologies at once by using L by D arrays, where D is the number of parameters per cosmology (D=3 for (i), D=9 for (ii), D=16 for (iii)). Due to memory constraints it is not advised to use L of more than a few hundred for cases (ii) and (iii), after which the speedup saturates anyways. You can only pass one value of comp_max at a time, and only one grid of output k-values. 

For questions / issues please contact me at t.j.m.bakx@uu.nl.
