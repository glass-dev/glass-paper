import numpy as np
import matplotlib.pyplot as plt

from gaussiancl import gcllim


lmax = 5000
lmax2 = 500
llim = 30_000

l = np.arange(0, lmax+1)
l2 = np.arange(0, lmax2+1)
gl = 5e-4*(1 + l)/(1 + l/20)**2.8
gl2 = 3e-4*(1 + l2)/(1 + l2/20)**2.8

l_ = np.arange(0, 3*lmax+1)
cl = gcllim(np.pad(gl, (0, llim-lmax)), 'lognormal', (1.,))[:l_.size]
cl2 = gcllim(np.pad(gl2, (0, llim-lmax2)), 'lognormal', (1.,))[:l_.size]

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

ax[0].plot(l[1:], gl[1:], c='C0', ls='-')
ax[0].plot(l2[1:], gl2[1:], c='C1', ls='--')
ax[1].plot(l_[1:], cl[1:], c='C0', ls='-')
ax[1].plot(l_[1:], cl2[1:], c='C1', ls='--')

for ax_ in ax:
    ax_.axvline(lmax, c=plt.rcParams['grid.color'], ls='-',
                lw=plt.rcParams['grid.linewidth'], zorder=-2)
    ax_.axvline(lmax2, c=plt.rcParams['grid.color'], ls='--',
                lw=plt.rcParams['grid.linewidth'], zorder=-2)

ax[0].set_xscale('log')
ax[0].set_yscale('log')
# ax[0].set_yscale('symlog', linthresh=1e-7, linscale=0.9, subs=[2, 3, 4, 5, 6, 7, 8, 9])
ax[-1].set_xlabel('angular mode number $l$')
ax[0].set_ylabel('Gaussian $G_l$')
ax[1].set_ylabel('lognormal $C_l$')

fig.tight_layout()

fig.savefig('plot/gaussian_cl.pdf', dpi=300, bbox_inches='tight')

plt.close()
