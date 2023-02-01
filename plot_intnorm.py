import numpy as np
import matplotlib.pyplot as plt

e = np.linspace(0, 1, 1000, endpoint=False)

var_eps = 0.256**2
var_eta = 0.315

h = np.arctanh(e)

p_extr = e*np.exp(-e**2/(2*var_eps))/var_eps

p_intr = 4*h*np.exp(-2*h**2/var_eta)/(1 - e**2)/var_eta

var_extr = np.trapz(p_extr*e**2, e)/2
var_intr = np.trapz(p_intr*e**2, e)/2

print('sig_extr =', var_extr**0.5)
print('sig_intr =', var_intr**0.5)

plt.plot(e, p_intr, '-', c='C0', label='intrinsic normal')
plt.plot(e, p_extr, '-', c='C1', label='extrinsic normal')
plt.plot(e, p_intr, '-', alpha=0.5)

plt.legend()

plt.xlabel('ellipticity magnitude $|\\epsilon|$')
plt.ylabel('probability density $p(|\\epsilon|)$')

plt.ylim(-0.25, 2.75)

plt.tight_layout()

plt.savefig('plot/intnorm.pdf', bbox_inches='tight')

plt.close()
