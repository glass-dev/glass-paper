import numpy as np
import matplotlib.pyplot as plt

from _config import *


fig, ax = plt.subplots(1, 1, figsize=(4, 3))

for z in shells:
    ax.axvline(z, c=plt.rcParams['grid.color'], ls=plt.rcParams['grid.linestyle'],
               lw=plt.rcParams['grid.linewidth'], zorder=-2)

for z, w in zip(weights.z, weights.w):
    ax.plot(z, w, c='C0', alpha=0.5, zorder=2)

z = np.arange(shells[0], np.nextafter(shells[-1]+0.01, shells[-1]), 0.01)
w = np.zeros_like(z)
for z_, w_ in zip(weights.z, weights.w):
    w += np.interp(z, z_, w_, left=0., right=0.)
ax.plot(z, w, c='C1', zorder=1)

ax.set_xlabel('redshift $z$')
ax.set_ylabel('matter weight function $W_i(z)$')

fig.tight_layout()

fig.savefig(plot_path/'shells.pdf', dpi=300, bbox_inches='tight')

plt.close()
