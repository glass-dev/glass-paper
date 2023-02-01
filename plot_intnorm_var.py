import numpy as np
import matplotlib.pyplot as plt


sig_eta = np.linspace(0.001, 1.5, 100)

e = np.linspace(0, 1, 1000, endpoint=False)
h = np.arctanh(e)

p_intr = 4*h*np.exp(-2*h**2/sig_eta[:, np.newaxis]**2)/(1 - e**2)/sig_eta[:, np.newaxis]**2

sig_eps = np.sqrt(np.trapz(p_intr*e**2, e)/2)

sig_fit = sig_eps*np.sqrt((5*sig_eps**2 + 8)/(2 - 4*sig_eps**2))

plt.plot(sig_eps, sig_eta)
plt.plot(sig_eps, sig_fit, ls=(0, (2, 2)))

plt.xlabel('extrinsic normal $\\sigma_\\epsilon$')
plt.ylabel('intrinsic normal $\\sigma_\\eta$')

plt.xlim(-0.025, 0.525)
plt.ylim(-0.15, 1.65)

plt.grid()

plt.tight_layout()

plt.savefig('plot/intnorm_var.pdf', bbox_inches='tight')

plt.close()
