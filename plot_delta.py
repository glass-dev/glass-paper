import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from online_mean import add_sample
from _plotting import split_bins, multi_row_label, symlog_no_zero
from _config import *


filenames = list(data_path.glob('fields.*.npz'))
logger.info('found %d data files', len(filenames))

n = len(shells)-1

input_cls = np.load(spec_path/'matter.npy')[..., :lmax+1]
delta_cls = np.zeros((n*(n+1)//2, lmax+1))
sigma_cls = np.zeros_like(delta_cls)
for i, filename in enumerate(filenames):
    with np.load(filename) as npz:
        cls_ = npz['delta_cls'][..., :lmax+1]
        add_sample(i, cls_, delta_cls, var=sigma_cls)
np.sqrt(sigma_cls, out=sigma_cls)

input_cls = split_bins(input_cls)
delta_cls = split_bins(delta_cls)
sigma_cls = split_bins(sigma_cls)

pw = hp.pixwin(nside, lmax=lmax)

l = np.arange(1, lmax+1)

plot_bins = np.searchsorted(shells[1:], showz)

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)

for i, ax in zip(plot_bins, axes):
    zmin, zmax = shells[i], shells[i+1]

    ax.annotate(f'${zmin:.2f} \\leq z \\leq {zmax:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    cl = delta_cls[i][0][1:]
    tl = input_cls[i][0][1:] * pw[1:]**2
    sl = sigma_cls[i][0][1:]

    ax.plot(l, (2*l+1)*cl)
    ax.plot(l, (2*l+1)*tl, c='k', lw=1.)
    ax.fill_between(l, (2*l+1)*(cl+sl), (2*l+1)*(cl-sl),
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

    ax.axvline(nside, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
               lw=plt.rcParams['grid.linewidth'], zorder=-1)

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[-1].set_xlabel('angular mode number $l$')
axes[0].set_ylabel('mean angular power spectrum $(2l + 1) \\, \\langle\\Delta C_l^{\\delta\\delta}\\rangle$')

fig.tight_layout()
multi_row_label(fig, axes[0])

fig.savefig(plot_path/'delta.pdf', dpi=300, bbox_inches='tight')

plt.close()

###

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)

for i, ax in zip(plot_bins, axes):
    zmin, zmax = shells[i], shells[i+1]

    ax.annotate(f'${zmin:.2f} \\leq z \\leq {zmax:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    cl = delta_cls[i][0][1:]
    tl = input_cls[i][0][1:] * pw[1:]**2
    sl = sigma_cls[i][0][1:]

    ax.plot(l, (cl - tl)/np.fabs(tl))
    ax.plot(l, np.zeros_like(l), c='k', lw=1.)
    ax.fill_between(l, +sl/np.fabs(tl), -sl/np.fabs(tl),
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

    ax.grid(True, which='major', axis='y')
    ax.axvline(nside, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
               lw=plt.rcParams['grid.linewidth'], zorder=-1)

axes[0].set_ylim(-0.9, 0.9)
axes[0].set_xscale('log')
axes[0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=[2, 3, 4, 5, 6, 7, 8, 9])
axes[-1].set_xlabel('angular mode number $l$')
axes[0].set_ylabel('mean relative error $\\langle\\Delta C_l^{\\delta\\delta}/\\Delta_l^{\\delta\\delta}\\rangle$')

fig.tight_layout()
multi_row_label(fig, axes[0])
symlog_no_zero(axes)

fig.savefig(plot_path/'delta_err.pdf', dpi=300, bbox_inches='tight')

plt.close()

###

fig, axes = plt.subplots(len(plot_bins), 1, figsize=(4, 4), sharex=True, sharey=True)

for i, ax in zip(plot_bins, axes):
    zmin, zmax = shells[i], shells[i+1]

    ax.annotate(f'${zmin:.2f} \\leq z \\leq {zmax:.2f}$', xy=(1., 1.), xytext=(-5, -8),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', backgroundcolor=(1., 1., 1., 0.5))

    cl = delta_cls[i][0][1:]
    tl = input_cls[i][0][1:] * pw[1:]**2
    sl = sigma_cls[i][0][1:]

    ax.plot(l, (cl - tl)/sl)
    ax.plot(l, np.zeros_like(l), c='k', lw=1.)
    ax.fill_between(l, +np.ones_like(l), -np.ones_like(l),
                    fc=plt.rcParams['hatch.color'], ec='none', zorder=-1)

    ax.axvline(nside, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
               lw=plt.rcParams['grid.linewidth'], zorder=-1)

axes[0].set_ylim(-1.5, 1.5)
axes[-1].set_xlabel('angular mode number $l$')
axes[0].set_ylabel('mean residuals $\\langle\\Delta C_l^{\\delta\\delta}/\\sigma_l^{\\delta\\delta}\\rangle$')

fig.tight_layout()
multi_row_label(fig, axes[0])

fig.savefig(plot_path/'delta_res.pdf', dpi=300, bbox_inches='tight')

plt.close()
