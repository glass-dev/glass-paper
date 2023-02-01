import numpy as np

from _config import *


def camb_galaxies_spectra(cosmo, z, nz, bias, lmax, *, limber=False):
    pars = cosmo._p.copy()
    pars.WantCls = True
    pars.Want_CMB = False
    pars.min_l = 1
    pars.set_for_lmax(lmax)
    pars.SourceTerms.limber_windows = limber
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_redshift = False
    pars.SourceTerms.counts_lensing = False
    pars.SourceTerms.counts_velocity = False
    pars.SourceTerms.counts_radial = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_ISW = False
    pars.SourceTerms.counts_potential = False
    pars.SourceTerms.counts_evolve = False

    sources = []
    for w in nz:
        sources += [
            camb.sources.SplinedSourceWindow(z=z, W=w, bias=bias, source_type='counts'),
            camb.sources.SplinedSourceWindow(z=z, W=w, bias=bias, source_type='lensing'),
        ]
    pars.SourceWindows = sources

    results = camb.get_results(pars)
    cls_dict = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

    n = len(sources)
    cls = [cls_dict[f'W{i+1}xW{i+j+1}'] for j in range(n) for i in range(n-j)]
    return cls


cls = camb_galaxies_spectra(cosmo, zz, dndz, gal_bias, lmax_gal, limber=False)

np.save(spec_path/'galaxies.npy', cls)
