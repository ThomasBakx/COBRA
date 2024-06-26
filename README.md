# COBRA
COBRA Repository for computing cosmological observables

Currently supports: (i) linear power spectrum in $\Lambda$CDM; (ii) linear power spectrum in extended parameter space including curvature, neutrinos, $w_0$, $w_a$; (iii) one-loop EPT resummed power spectrum of biased tracers in redshift space for LambdaCDM based on Chen, Vlah, White: https://arxiv.org/pdf/2005.00523. 

By default, the code uses units of $(h*)/\text{Mpc}$ for wavenumbers, where $h* = 0.7$ throughout. This means that you also get the power spectra in units of $(\text{Mpc}/h*)^3$. If you want to change this to or $1/\text{Mpc}$, set the 'units' keyword as desired when calling COBRA. The allowed ranges are given in units of $(h*)/\text{Mpc}$ and are $[0.0008,4]$ for (i), $[0.001,1.5]$ for (ii) and $[0.001,0.5]$ for (iii).   

All modules are vectorized, so you can feed L cosmologies at once by using L by D arrays, where D is the number of parameters per cosmology (D=6 for (i), D=10 for (ii), D=18 for (iii)). Due to memory constraints it is not advised to use L of more than a few hundred, after which the speedup saturates anyways. You can only pass one value of comp_max at a time, and only one grid of output k-values. 

For questions / issues please contact me at t.j.m.bakx@uu.nl.

## Credit

These demonstration notebooks are released under a CC-BY 4.0 license: https://creativecommons.org/licenses/by/4.0/
If you use this code, please cite the corresponding paper (arXiv number TBD)
