import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from glass.matter import effective_redshifts
from _plotting import split_bins, symlog_no_zero
from _config import *


cls = np.load(spec_path/'matter.npy')

nshell = len(shells)-1
zeff = effective_redshifts(weights)

l = np.arange(cls.shape[-1])

cls = split_bins(cls)

cmap = LinearSegmentedColormap.from_list('C0-C6-C2', ['C0', 'C6', 'C2'])

fig, axes = plt.subplots(4, 2, figsize=(8, 4), sharex=True, sharey=True)

for k, ax in enumerate(axes.T.flat):
    for i in range(nshell):
        if len(cls[i]) > k+1:
            r = cls[i][k+1, 1:]/(cls[i][0, 1:]*cls[i-k-1][0, 1:])**0.5
            ax.plot(l[1:], r, c=cmap(zeff[i]/zend), ls='-', lw=0.5)

    ax.grid(which='major', axis='y')
    ax.set_ylabel(f'$R_l^{{i,i-{k+1}}}$')

axes[0, 0].set_ylim(-0.5, 0.5)
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('symlog', linthresh=1e-2, linscale=0.45, subs=None)
axes[-1, 0].set_xlabel('angular mode number $l$')
axes[-1, 1].set_xlabel('angular mode number $l$')

symlog_no_zero(axes)

fig.tight_layout()

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=zend), cmap=cmap),
                    ax=axes, orientation='vertical',
                    label='effective redshift $\\bar{z}_i$',
                    fraction=0.04, shrink=0.5, pad=0.03)
cbar.ax.tick_params(rotation=90)

fig.savefig(plot_path/'correlations.pdf', dpi=300, bbox_inches='tight')

plt.close()
