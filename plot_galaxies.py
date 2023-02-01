import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from online_mean import add_sample
from glass.matter import effective_redshifts
from glass.lensing import multi_plane_weights, multi_plane_matrix
from glass.math import ARCMIN2_SPHERE, restrict_interval
from _plotting import getcl, bin_in_ell, multi_row_label, multi_col_label, symlog_no_zero
from _config import *


filenames = sorted(data_path.glob('galaxies.*.npz'))
logger.info('found %d data files', len(filenames))

matter_cls = np.load(spec_path/'matter.npy')

cls = np.zeros(((2*nbin)*(2*nbin+1)//2, lmax_gal+1))
shear_cls = np.zeros_like(cls)
sigma_cls = np.zeros_like(cls)
for i, filename in enumerate(filenames):
    with np.load(filename) as npz:
        add_sample(i, npz['cls'], cls, var=sigma_cls)
        add_sample(i, npz['shear_cls'], shear_cls)
np.sqrt(sigma_cls, out=sigma_cls)

npix = 12*nside_gal**2
nbar = np.reshape(np.trapz(dndz, zz, axis=-1), -1)
nbar *= ARCMIN2_SPHERE/npix

zkap = effective_redshifts(weights)
wkap = multi_plane_weights(zkap, weights)
lm = multi_plane_matrix(zkap, wkap, cosmo)

nw = []
for z1, z2 in zip(shells, shells[1:]):
    nz, z = restrict_interval(dndz, zz, z1, z2)
    nw.append(np.trapz(nz, z, axis=-1))
nw = np.transpose(nw)
nw /= np.sum(nw, axis=-1, keepdims=True)

dndz_eff = np.dot(nw, [np.interp(zz, z, w/np.trapz(w, z), left=0., right=0.)
                       for z, w in zip(weights.z, weights.w)])

lw = np.dot(nw, lm)

nw *= gal_bias

ns = len(shells) - 1

position_cls = sum(nw[:, np.newaxis, [i]]*nw[np.newaxis, :, [j]]*getcl(matter_cls, i, j)
                   for i in range(ns) for j in range(ns))

lensing_cls = sum(lw[:, np.newaxis, [i]]*lw[np.newaxis, :, [j]]*getcl(matter_cls, i, j)
                  for i in range(ns) for j in range(ns))

cross_cls = sum(nw[:, np.newaxis, [i]]*lw[np.newaxis, :, [j]]*getcl(matter_cls, i, j)
                for i in range(ns) for j in range(ns))

pw = hp.pixwin(nside_gal, lmax=lmax_gal, pol=True)

l = np.arange(lmax_gal+1)
fl = -np.sqrt((l+2)*(l+1)*l*(l-1))/np.clip(l*(l+1), 1, None)

nl = []
for i in range(nbin):
    nl += [
        4*np.pi/(npix*nbar[i]) * (l >= 1),
        0.
    ]

for n in range(3):

    fig, ax = plt.subplots(2*nbin, 2*nbin, figsize=(4*nbin, 4*nbin), sharex=True, sharey=True)

    for i, j in zip(*np.triu_indices(2*nbin, 1)):
        ax[i, j].axis('off')

    for i, j in zip(*np.tril_indices(2*nbin)):
        ib, ip = divmod(i, 2)
        jb, jp = divmod(j, 2)

        lmin = 1 if ip == jp == 0 else 2

        cl = cls[i - (i-j+1)*(i-j-4*nbin)//2 - 2*nbin]
        ql = shear_cls[i - (i-j+1)*(i-j-4*nbin)//2 - 2*nbin]
        if i == j:
            cl = cl - nl[i]
            ql = ql - nl[i]

        if ip == jp == 0:
            al = position_cls[ib, jb][:lmax_gal+1]
        elif ip == jp == 1:
            al = lensing_cls[ib, jb][:lmax_gal+1]
        elif ip == 0 and jp == 1:
            al = cross_cls[ib, jb][:lmax_gal+1]
        else:
            al = cross_cls[jb, ib][:lmax_gal+1]
        al = al * fl**(ip+jp) * (pw[ip]*pw[jp])

        sl = sigma_cls[i - (i-j+1)*(i-j-4*nbin)//2 - 2*nbin]

        if gal_lbin is not None:
            l_ = bin_in_ell(l, l, gal_lbin)
            cl_ = bin_in_ell(l, cl, gal_lbin)
            ql_ = bin_in_ell(l, ql, gal_lbin)
            al_ = bin_in_ell(l, al, gal_lbin)
            sl_ = bin_in_ell(l, sl, gal_lbin)
        else:
            l_ = l[lmin:]
            cl_ = cl[lmin:]
            ql_ = ql[lmin:]
            al_ = al[lmin:]
            sl_ = sl[lmin:]

        if n == 0:
            ax[i, j].plot(l_, (2*l_+1)*cl_)
            ax[i, j].plot(l_, (2*l_+1)*al_, ls='-', c='k', lw=0.5)
            ax[i, j].fill_between(l_, (2*l_+1)*(cl_-sl_), (2*l_+1)*(cl_+sl_),
                                  fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)
        elif n == 1:
            ax[i, j].plot(l_, (cl_ - al_)/np.fabs(al_), label='reduced shear', zorder=3)
            ax[i, j].plot(l_, (ql_ - al_)/np.fabs(al_), label='shear', zorder=2)
            ax[i, j].fill_between(l_, -sl_/np.fabs(al_), +sl_/np.fabs(al_),
                                  fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)
            ax[i, j].grid(True, which='major', axis='y')
        elif n == 2:
            ax[i, j].plot(l_, (cl_ - al_)/sl_, label='reduced shear', zorder=3)
            ax[i, j].plot(l_, (ql_ - al_)/sl_, label='shear', zorder=2)
            ax[i, j].fill_between(l_, -np.ones_like(l_), +np.ones_like(l_),
                                  fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)
            ax[i, j].grid(True, which='major', axis='y')

    if n > 0:
        ax[1, 1].legend(loc='upper right')

    ax[-1, 0].set_xlabel('angular mode number $l$')

    ax[0, 0].set_xscale('log')
    ax[0, 0].xaxis.get_major_locator().set_params(numticks=99)
    ax[0, 0].xaxis.get_minor_locator().set_params(numticks=99, subs=[.1, .2, .3, .4, .5, .6, .7, .8, .9])

    if n == 0:
        ax[0, 0].set_ylim(-8e-5, 8e-4)
        ax[0, 0].set_yscale('symlog', linthresh=1e-7, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        ax[-1, 0].set_ylabel('mean angular power spectrum $(2l + 1) \\, \\langle C_l \\rangle$')
    elif n == 1:
        ax[0, 0].set_ylim(-9, 9)
        ax[0, 0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        ax[-1, 0].set_ylabel('mean relative error $\\langle \\Delta C_l \\rangle/|C_l|$')
    elif n == 2:
        ax[0, 0].set_ylim(-9, 9)
        ax[0, 0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        ax[-1, 0].set_ylabel('mean residuals $\\langle \\Delta C_l \\rangle/\\sigma_l$')

    fig.tight_layout()
    multi_col_label(fig, ax[-1, 0])
    multi_row_label(fig, ax[-1, 0])
    symlog_no_zero(ax)

    for i in range(2*nbin):
        ib, ip = i//2, i % 2
        ax[i, i].set_title(f'positions bin {ib+1}' if ip == 0 else f'shear bin {ib+1}', in_layout=False)

    (x0, y0) = fig.transFigure.inverted().transform(ax[0, -2].transAxes.transform((0., 0.)))
    (x1, y1) = fig.transFigure.inverted().transform(ax[0, -2].transAxes.transform((1., 1.)))
    (x2, y2) = fig.transFigure.inverted().transform(ax[0, -1].transAxes.transform((1., 1.)))

    if nbin > 1:
        ax_ = fig.add_axes((0.5*(x0+x1), y0, x2 - 0.5*(x0+x1), y1-y0))
        ax_.plot(zz, dndz_eff.T, c='C0', lw=0.5)
        for nz in dndz_eff:
            ax_.fill_between(zz, nz, fc='C0', ec='none', alpha=0.5)
        for z in shells:
            ax_.axvline(z, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
                        lw=plt.rcParams['grid.linewidth'], zorder=-2)
        ax_.margins(0.1, 0.2)
        ax_.set_title('source distribution')
        ax_.set_xlabel('redshift $z$')
        ax_.set_ylabel('galaxy density $dn/dz$')

    if n == 0:
        filename_tag = ''
    elif n == 1:
        filename_tag = '_err'
    elif n == 2:
        filename_tag = '_res'

    fig.savefig(plot_path/f'galaxies{filename_tag}.pdf', dpi=300, bbox_inches='tight')

    plt.close()
