import numpy as np
import matplotlib.pyplot as plt
from glass.matter import effective_redshifts
from glass.lensing import multi_plane_weights, multi_plane_matrix
from glass.math import restrict_interval
from _plotting import getcl, multi_row_label, symlog_no_zero
from _config import *


cls = np.load(spec_path/'matter.npy')
theory_cls = np.load(spec_path/'galaxies.npy')

zkap = effective_redshifts(weights)
wkap = multi_plane_weights(zkap, weights)
lm = multi_plane_matrix(zkap, wkap, cosmo)

wgrid = np.array([np.interp(zz, z, w/np.trapz(w, z), left=0., right=0.)
                  for z, w in zip(weights.z, weights.w)])

fact = []
for za, zb in zip(shells, shells[1:]):
    nz, z = restrict_interval(dndz, zz, za, zb)
    fact.append(np.trapz(nz, z, axis=-1))
fact = np.reshape(fact, (len(fact), -1)).T

approx_dndz = np.dot(fact, wgrid)

fig, ax = plt.subplots(nbin, 1, figsize=(4, 1.5*nbin), sharex=True, sharey=True)
ax = np.atleast_1d(ax)

for i, zm in enumerate(gal_mean_z):

    ax[i].annotate(f'$\\langle z_g \\rangle = {zm:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                   xycoords='axes fraction', textcoords='offset points',
                   ha='right', va='top', backgroundcolor=(1., 1., 1., 0.8))

    ax[i].plot(zz, dndz[i], '-', c='k', lw=0.5, zorder=2)
    ax[i].plot(zz, approx_dndz[i], zorder=1)

    for z in shells:
        ax[i].axvline(z, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
                      lw=plt.rcParams['grid.linewidth'], zorder=-2)

ax[0].margins(y=0.2)

ax[-1].set_xlabel('redshift $z$')
ax[-1].set_ylabel('effective galaxy distribution')

fig.tight_layout()
multi_row_label(fig, ax[-1])

fig.savefig(plot_path/'galaxies_kernel.pdf', dpi=300, bbox_inches='tight')

plt.close()

###

fact /= np.sum(fact, axis=-1, keepdims=True)

n = len(shells)-1
b = [gal_bias]*n
position_cls = sum(b[i]*b[j]*fact[:, [i]]*fact[:, [j]]*getcl(cls, i, j, lmax=lmax_gal)
                   for i in range(n) for j in range(n))

fact = np.dot(fact, lm)

lensing_cls = sum(fact[:, [i]]*fact[:, [j]]*getcl(cls, i, j, lmax=lmax_gal)
                  for i in range(n) for j in range(n))

l = np.arange(lmax_gal+1)

fig, ax = plt.subplots(nbin, 1, figsize=(4, 1.5*nbin), sharex=True, sharey=True)
ax = np.atleast_1d(ax)

for i, zm in enumerate(gal_mean_z):
    ax[i].annotate(f'$\\langle z_g \\rangle = {zm:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                   xycoords='axes fraction', textcoords='offset points',
                   ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    l_ = l[1:]
    tl_ = theory_cls[2*i, 1:lmax_gal+1]
    cl_ = position_cls[i, 1:]

    ax[i].plot(l_, (cl_ - tl_)/np.fabs(tl_), label='positions')

    tl_ = theory_cls[2*i+1, 1:lmax_gal+1]
    cl_ = lensing_cls[i, 1:]

    ax[i].plot(l_, (cl_ - tl_)/np.fabs(tl_), label='lensing')

    ax[i].fill_between(l_, -1/(l_+0.5)**0.5, +1/(l_+0.5)**0.5,
                       fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

    ax[i].grid(True, which='major', axis='y')

ax[0].legend(loc='lower right', fontsize=8)

ax[-1].set_xlabel('angular mode number $l$')
ax[0].set_ylabel('relative error $\\Delta C_l/C_l$')

ax[0].set_ylim(-0.9, 0.9)
ax[0].set_xscale('log')
ax[0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])

ax[0].xaxis.get_major_locator().set_params(numticks=99)
ax[0].xaxis.get_minor_locator().set_params(numticks=99, subs=[.1, .2, .3, .4, .5, .6, .7, .8, .9])

fig.tight_layout()
multi_row_label(fig, ax[0])
symlog_no_zero(ax)

fig.savefig(plot_path/'galaxies_approx.pdf', dpi=300, bbox_inches='tight')

plt.close()
