import numpy as np
from glass.matter import effective_redshifts

from _config import *


def camb_lensing_spectra(cosmo, zsrcs, lmax, *, limber=False):
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

    norms = []
    sources = []
    for zsrc in zsrcs:
        z = np.linspace(0, zsrc, 1000)
        w = 3*cosmo.omega_m/2*cosmo.xm(z)/cosmo.xm(zsrc)*cosmo.xm(z, zsrc)*(1 + z)/cosmo.ef(z)
        norms += [np.trapz(w, z)]
        sources += [camb.sources.SplinedSourceWindow(z=z, W=w)]
    pars.SourceWindows = sources

    results = camb.get_results(pars)
    cls_dict = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

    n = len(sources)
    cls = [norms[i]*norms[j]*cls_dict[f'W{i+1}xW{j+1}']
           for i in range(n) for j in range(i, -1, -1)]
    return cls


zsrc = effective_redshifts(weights)

logger.info('source redshifts:\n%s', zsrc)

cls = camb_lensing_spectra(cosmo, zsrc, lmax, limber=False)

np.save(spec_path/'lensing.npy', cls)
