import numpy as np
import matplotlib.pyplot as plt

from gaussiancl import gaussiancl, gcllim


lmax = 5000

l = np.arange(0, lmax+1)
cl_ = 5e-4*(1 + l)/(1 + l/10)**2

settings = [(-4, 2), (-4, 3), (-5, 3), (-5, 4)]

gls = []
cls = []
for tol, nfac in settings:

    gl, info, err, niter = gaussiancl(cl_, 'lognormal', (1.,), n=nfac*len(cl_),
                                      cltol=10.**tol, gltol=10.**tol)

    print(info, err, niter)

    cl = gcllim(np.pad(gl, (0, 1_000_000 - len(gl))), 'lognormal', (1.,))

    gls.append(gl)
    cls.append(cl)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

for (tol, nfac), cl in zip(settings, cls):
    rel = np.fabs(cl[:len(cl_)] - cl_)/cl_

    ax.plot(l[1:], rel[1:], label=f'tolerance$= 10^{{{tol}}}$, $n = {nfac}\\,N$')

ax.legend(fontsize=7)
# ax.set_ylim(2e-7, 2e-4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('angular mode number $l$')
ax.set_ylabel('relative error $\\Delta C_l/C_l$')

fig.tight_layout()

fig.savefig('plot/gaussian_cl_err.pdf', dpi=300, bbox_inches='tight')

plt.show()
