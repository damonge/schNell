import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

det = snl.LISADetector(0)
freqs = np.geomspace(5E-5, 2, 2048)

fstar = det.clight / (2*np.pi*det.L)
plt.plot(freqs, det.psd_A(freqs), 'k-', label=r'$N^{11}_f$')
plt.plot(freqs, det.psd_X(freqs), 'r--', label=r'$N^{12}_f$')
plt.plot(freqs, -det.psd_X(freqs), 'r:', label=r'$-N^{12}_f$')
plt.plot([fstar, fstar], [1E-43, 1E-32], 'b--', lw=1)
plt.text(fstar*0.6, 1E-33, r'$f_*$', fontsize=16, color='b')
plt.loglog()
plt.xlim([9E-5, 1.2])
plt.ylim([6E-43, 6E-33])
plt.xlabel(r'$f\,\,[{\rm Hz}]$', fontsize=16)
plt.ylabel(r'$N_f\,\,[{\rm Hz}^{-1}]$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.legend(loc='upper right', fontsize=14, frameon=False)
plt.savefig("psd_LISA.pdf", bbox_inches='tight')
plt.show()
