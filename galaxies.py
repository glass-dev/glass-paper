import gc
import numpy as np
import healpy as hp

import glass.all

from _config import *


cls = np.load(spec_path/'matter.npy')

assert len(cls) == len(shells)*(len(shells)-1)//2, 'cls and shells mismatch'

zkap = glass.matter.effective_redshifts(weights)
wkap = glass.lensing.multi_plane_weights(zkap, weights)

ngal = glass.galaxies.density_from_dndz(zz, dndz, bins=shells)

logger.info('measuring modes up to %d using NSIDE=%d', lmax_gal, nside_gal)

npix = 12*nside_gal**2
nbar = (glass.math.ARCMIN2_SPHERE/npix) * np.trapz(dndz, zz, axis=-1)

ntot = np.zeros_like(nbar, dtype=int)

maps = np.zeros((nbin, 5, npix))

with glass.user.profile('galaxies', logger) as prof:

    prof.log('gaussian cls')

    gal_gls = glass.fields.lognormal_gls(cls, lmax=lmax_gal, ncorr=ncorr_gal, nside=nside_gal)

    matter = glass.fields.generate_lognormal(gal_gls, nside_gal, ncorr=ncorr_gal, rng=rng)

    convergence = glass.lensing.MultiPlaneConvergence(cosmo)

    prof.log('shells')

    for i, delta_i in enumerate(matter):

        prof.loop(f'shell {i}')

        zmin, zmax = shells[i], shells[i+1]

        prof.log('convergence')

        convergence.add_plane(delta_i, zkap[i], wkap[i])
        kappa_i = convergence.kappa

        prof.log('shear')

        gamm1_i, gamm2_i = glass.lensing.shear_from_convergence(kappa_i, lmax_gal)

        prof.log('reduced shear')

        g1_i = gamm1_i/(1 - kappa_i)
        g2_i = gamm2_i/(1 - kappa_i)

        prof.log('galaxy positions')

        gal_lon, gal_lat = glass.points.positions_from_delta(ngal[i], delta_i, gal_bias, rng=rng)
        gal_size = len(gal_lon)
        logger.info('number of galaxies: %s', f'{gal_size:,}')

        prof.log('galaxy redshifts')

        gal_z, gal_pop = glass.galaxies.redshifts_from_nz(gal_size, zz, dndz, zmin=zmin, zmax=zmax, rng=rng)

        prof.log('galaxy position and shear maps')

        for k in range(nbin):
            sel = (gal_pop == k)
            ipix = hp.ang2pix(nside_gal, gal_lon[sel], gal_lat[sel], lonlat=True)
            n = len(ipix)
            if n > 0:
                ntot[k] += n
                ipix, count = np.unique(ipix, return_counts=True)
                maps[k, 0, ipix] += count
                maps[k, 1] += (n/ntot[k]) * (g1_i - maps[k, 1])
                maps[k, 2] += (n/ntot[k]) * (g2_i - maps[k, 2])
                maps[k, 3] += (n/ntot[k]) * (gamm1_i - maps[k, 3])
                maps[k, 4] += (n/ntot[k]) * (gamm2_i - maps[k, 4])

        del kappa_i, gamm1_i, gamm2_i, g1_i, g2_i, gal_lon, gal_lat, gal_z, gal_pop
        gc.collect()

assert np.all(np.sum(maps[:, 0], axis=-1) == ntot)

logger.info('total galaxies counted: %s', ntot)

maps[:, 0] /= np.reshape(nbar, (-1, 1))
maps[:, 0] -= 1

alms = np.array([hp.map2alm([m[0], m[1], m[2]], lmax=lmax_gal, pol=True, use_pixel_weights=True)
                 for m in maps])
cls = [hp.alm2cl(alms[divmod(i, 2)], alms[divmod(i+k, 2)])
       for k in range(2*nbin) for i in range(2*nbin-k)]

alms = np.array([hp.map2alm([m[0], m[3], m[4]], lmax=lmax_gal, pol=True, use_pixel_weights=True)
                 for m in maps])
shear_cls = [hp.alm2cl(alms[divmod(i, 2)], alms[divmod(i+k, 2)])
             for k in range(2*nbin) for i in range(2*nbin-k)]

filename = data_path/f'galaxies.{job_id}.npz'
print('saving', filename)
np.savez(filename, cls=cls, shear_cls=shear_cls)
