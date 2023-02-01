import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from online_mean import add_sample
from glass.matter import effective_redshifts
from glass.lensing import multi_plane_weights, multi_plane_matrix
from _plotting import getcl, split_bins, multi_row_label, symlog_no_zero
from _config import *


filenames = list(data_path.glob('fields.*.npz'))
logger.info('found %d data files', len(filenames))

n = len(shells)-1

theory_cls = np.load(spec_path/'lensing.npy')[..., :lmax+1]
kappa_cls = np.zeros((n*(n+1)//2, lmax+1))
sigma_cls = np.zeros_like(kappa_cls)
for i, filename in enumerate(filenames):
    with np.load(filename) as npz:
        cls_ = npz['kappa_cls']
        add_sample(i, cls_[..., :lmax+1], kappa_cls, var=sigma_cls)
np.sqrt(sigma_cls, out=sigma_cls)

zkap = effective_redshifts(weights)
wkap = multi_plane_weights(zkap, weights)
lm = multi_plane_matrix(zkap, wkap, cosmo)
matter_cls = np.load(spec_path/'matter.npy')
approx_cls = sum(lm[:, i, np.newaxis]*lm[:, j, np.newaxis]*getcl(matter_cls, i, j, lmax=lmax)
                 for i in range(n) for j in range(n))

theory_cls = split_bins(theory_cls)
kappa_cls = split_bins(kappa_cls)
sigma_cls = split_bins(sigma_cls)

pw = hp.pixwin(nside, lmax=lmax)

l = np.arange(1, lmax+1)

plot_bins = np.searchsorted(shells[1:], showz)

####
if False:
    kappa3_cls = []
    filenames = glob.glob('data/ncorr_3.*.npz')
    logger.info('found %d data files', len(filenames))
    for filename in filenames:
        logger.debug('loading %s', filename)
        with np.load(filename) as npz:
            kappa3_cls += [npz['kappa_cls']]
    kappa3_cls = np.asanyarray(kappa3_cls)
    rel3_err = (kappa3_cls[..., 1:] - theory_cls[..., 1:])/theory_cls[..., 1:]
    rel3_err = np.mean(rel3_err, axis=0)
####

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)

for i, ax in zip(plot_bins, axes):
    zsrc = zkap[i]

    ax.annotate(f'$z_s = {zsrc:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    cl = kappa_cls[i][0][1:]
    tl = theory_cls[i][0][1:] * pw[1:]**2
    al = approx_cls[i][1:] * pw[1:]**2
    sl = sigma_cls[i][0][1:]

    ax.plot(l, (2*l+1)*cl)
    ax.plot(l, (2*l+1)*al)
    ax.plot(l, (2*l+1)*tl, c='k', lw=1.)
    # ax.plot(l, rel3_err[i], label='$N_{\\rm corr} = 3$')
    # ax.plot(l, rel_err[i][0], c='C0', alpha=0.5)
    ax.fill_between(l, (2*l+1)*(cl+sl), (2*l+1)*(cl-sl),
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

# axes[0].legend()

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[-1].set_xlabel('angular mode number $l$')
axes[-1].set_ylabel('mean angular power spectrum $(2l + 1) \\, \\langle\\Delta C_l^{\\kappa\\kappa}\\rangle$')

fig.tight_layout()
multi_row_label(fig, axes[-1])
symlog_no_zero(axes)

fig.savefig(plot_path/'kappa.pdf', dpi=300, bbox_inches='tight')

plt.close()

###

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)

for i, ax in zip(plot_bins, axes):
    zsrc = zkap[i]

    ax.annotate(f'$z_s = {zsrc:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    cl = kappa_cls[i][0][1:]
    tl = theory_cls[i][0][1:] * pw[1:]**2
    al = approx_cls[i][1:] * pw[1:]**2
    sl = sigma_cls[i][0][1:]

    ax.plot(l, (cl - tl)/np.fabs(tl), label='$N_{\\rm corr} = 1$')
    # ax.plot(l, rel3_err[i], label='$N_{\\rm corr} = 3$')
    # ax.plot(l, rel_err[i][0], c='C0', alpha=0.5)
    ax.plot(l, (al - tl)/np.fabs(tl), ls='-', c='k', lw=1)

    ax.fill_between(l, +sl/np.fabs(tl), -sl/np.fabs(tl),
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

    ax.grid(True, which='major', axis='y')

# axes[0].legend()

axes[0].set_ylim(-0.9, 0.9)
axes[0].set_xscale('log')
axes[0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])
axes[-1].set_xlabel('angular mode number $l$')
axes[-1].set_ylabel('mean relative error $\\langle\\Delta C_l^{\\kappa\\kappa}/C_l^{\\kappa\\kappa}\\rangle$')

fig.tight_layout()
multi_row_label(fig, axes[-1])
symlog_no_zero(axes)

fig.savefig(plot_path/'kappa_err.pdf', dpi=300, bbox_inches='tight')

plt.close()

###

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)

for i, ax in zip(plot_bins, axes):
    zsrc = zkap[i]

    ax.annotate(f'$z_s = {zsrc:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    cl = kappa_cls[i][0][1:]
    tl = theory_cls[i][0][1:] * pw[1:]**2
    al = approx_cls[i][1:] * pw[1:]**2
    sl = sigma_cls[i][0][1:]

    ax.plot(l, (cl - tl)/sl)
    ax.plot(l, (al - tl)/sl, ls='-', c='k', lw=1)
    ax.fill_between(l, +np.ones_like(l), -np.ones_like(l),
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1, alpha=0.2)

axes[0].set_ylim(-1.5, 1.5)
axes[-1].set_xlabel('angular mode number $l$')
axes[0].set_ylabel('mean residuals $\\langle\\Delta C_l^{\\kappa\\kappa}/\\sigma_l^{\\kappa\\kappa}\\rangle$')

fig.tight_layout()
multi_row_label(fig, axes[0])

fig.savefig(plot_path/'kappa_res.pdf', dpi=300, bbox_inches='tight')

plt.close()
