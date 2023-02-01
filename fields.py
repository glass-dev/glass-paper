import numpy as np
import healpy as hp

import glass.all

from _config import *


cls = np.load(spec_path/'matter.npy')

assert len(cls) == len(shells)*(len(shells)-1)//2, 'cls and shells mismatch'

zkap = glass.matter.effective_redshifts(weights)
wkap = glass.lensing.multi_plane_weights(zkap, weights)

delta_alms = []
delta_cls = []
kappa_alms = []
kappa_cls = []

ell = np.arange(lmax+1)

# use this to compute the cls as observed
cls_obs = glass.fields.gaussian_gls(cls, lmax=lmax, nside=nside)

with glass.user.profile('fields', logger) as prof:

    prof.log('lognormal gls')

    logger.info('nside: %d', nside)
    logger.info('lmax: %d', lmax)
    logger.info('ncorr: %d', ncorr)

    gls = glass.fields.lognormal_gls(cls, lmax=lmax, ncorr=ncorr, nside=nside)

    matter = glass.fields.generate_lognormal(gls, nside, ncorr=ncorr, rng=rng)

    convergence = glass.lensing.MultiPlaneConvergence(cosmo)

    prof.log('shells')

    for i, delta in enumerate(matter):

        prof.loop(f'shell {i}')

        logger.info('delta')
        logger.info('  min: %g', np.min(delta))
        logger.info('  max: %g', np.max(delta))
        logger.info('  mean: %g', np.mean(delta))
        logger.info('  var: %g [%g]', np.var(delta), np.sum((2*ell+1)/(4*np.pi)*cls_obs[i*(i+1)//2]))

        prof.log('delta')

        alm = hp.map2alm(delta, lmax=lmax, use_pixel_weights=True)
        delta_alms.append(alm)
        delta_cls += [hp.alm2cl(alm, alm_) for alm_ in delta_alms[::-1]]

        prof.log('kappa')

        convergence.add_plane(delta, zkap[i], wkap[i])
        alm = hp.map2alm(convergence.kappa, lmax=lmax, use_pixel_weights=True)
        kappa_alms.append(alm)
        kappa_cls += [hp.alm2cl(alm, alm_) for alm_ in kappa_alms[::-1]]

filename = data_path/f'fields.{job_id}.npz'
logger.info('saving %s', filename)
np.savez(filename, delta_cls=delta_cls, kappa_cls=kappa_cls)
