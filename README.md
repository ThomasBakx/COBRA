# COBRA
COBRA Repository for computing cosmological observables

Currently supports: (i) linear matter and galaxy power spectrum in LambdaCDM; (ii) linear matter and galaxy power spectrum in extended parameter space including curvature, neutrinos, $w_0$, $w_a$; (iii) one-loop EPT resummed power spectrum of biased tracers in redshift space for LambdaCDM based on Chen, Vlah, White: https://arxiv.org/pdf/2005.00523, (iv) one-loop matter power spectrum in LambdaCDM.

By default, the code uses units of $(h*)/\text{Mpc}$ for wavenumbers, where $h* = 0.7$ throughout. This means that you also get the power spectra in units of $(\text{Mpc}/h*)^3$. The allowed ranges are given in units of $(h*)/\text{Mpc}$ and are $[0.0008,4]$ for (i), $[0.001,1.5]$ for (ii) and $[0.001,0.5]$ for (iii).   

All main modules are vectorized, so you can feed multiple cosmologies at once by using lists as values for the cosmology dict keys. Due to memory constraints it is not advised to use L of more than a few hundred, after which the speedup saturates anyways. 

For questions / issues please contact me at t.j.m.bakx@uu.nl.

## Credit

These demonstration notebooks are released under a CC-BY 4.0 license: https://creativecommons.org/licenses/by/4.0/
If you use this code, please cite the corresponding paper (arXiv number TBD)
