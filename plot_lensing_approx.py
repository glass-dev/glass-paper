import numpy as np
import matplotlib.pyplot as plt
from glass.matter import effective_redshifts
from glass.lensing import multi_plane_weights, multi_plane_matrix
from _plotting import getcl, split_bins, multi_row_label, symlog_no_zero
from _config import *


cls = np.load(spec_path/'matter.npy')

theory_cls = np.load(spec_path/'lensing.npy')
theory_cls = split_bins(theory_cls)

n = len(weights.z)

zkap = effective_redshifts(weights)
wkap = multi_plane_weights(zkap, weights)
lmat = multi_plane_matrix(zkap, wkap, cosmo)
approx_cls = sum(lmat[:, i, np.newaxis]*lmat[:, j, np.newaxis]*getcl(cls, i, j, lmax=lmax)
                 for i in range(n) for j in range(n))

plot_bins = np.searchsorted(shells[1:], showz)

fig, axes = plt.subplots(len(showz), 1, figsize=(4, 4), sharex=True, sharey=True)
axes = np.atleast_1d(axes)

for i, ax in zip(plot_bins, axes):

    zsrc = zkap[i]

    ax.annotate(f'$z_s = {zsrc:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    z = np.arange(0, np.nextafter(zsrc+0.01, zsrc), 0.01)
    w = 3*cosmo.omega_m/2*cosmo.xm(z)/cosmo.xm(zsrc)*cosmo.xm(z, zsrc)*(1 + z)/cosmo.ef(z)

    ax.plot(z, w, '-', c='k', lw=0.5, zorder=1)

    for k in range(i+1):
        z = weights.z[k]
        w = lmat[i, k]*weights.w[k]/np.trapz(weights.w[k], weights.z[k])
        ax.plot(z, w, c='C0', zorder=0)

    for z in shells:
        ax.axvline(z, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
                   lw=plt.rcParams['grid.linewidth'], zorder=-2)
        if z > zsrc:
            break

axes[0].margins(y=0.2)

axes[-1].set_xlabel('redshift $z$')
axes[-1].set_ylabel('effective lensing kernel')

fig.tight_layout()
multi_row_label(fig, axes[-1])

fig.savefig(plot_path/'lensing_kernel.pdf', dpi=300, bbox_inches='tight')

plt.close()

###

l = np.arange(1, lmax+1)

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)
axes = np.atleast_1d(axes)

for i, ax in zip(plot_bins, axes):

    zsrc = zkap[i]

    ax.annotate(f'$z_s = {zsrc:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.8))

    tl = theory_cls[i][0][1:lmax+1]
    al = approx_cls[i][1:]
    sl = 1/(l + 0.5)**0.5

    ax.plot(l, (al - tl)/np.fabs(tl))
    ax.fill_between(l, +sl, -sl,
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

    ax.grid(True, which='major', axis='y')

axes[0].set_ylim(-0.9, 0.9)
axes[0].set_xscale('log')
axes[0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])
axes[-1].set_xlabel('angular mode number $l$')
axes[0].set_ylabel('relative error $\\Delta C_l^{\\kappa\\kappa}/C_l^{\\kappa\\kappa}$')

fig.tight_layout()
multi_row_label(fig, axes[0])
symlog_no_zero(axes)

fig.savefig(plot_path/'lensing_approx.pdf', dpi=300, bbox_inches='tight')

plt.close()
