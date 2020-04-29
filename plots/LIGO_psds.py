import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

freqs = np.geomspace(8., 1010., 2048)

dets = [snl.GroundDetector('Hanford',     46.4, -119.4, 171.8,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Virgo',       43.6,   10.5, 116.5,
                           'data/Virgo.txt'),
        snl.GroundDetector('KAGRA',       36.3,  137.2, 225.0,
                           'data/KAGRA.txt')]

plt.plot(freqs, dets[0].psd(freqs), 'k-', label='LIGO')
plt.plot(freqs, dets[2].psd(freqs), 'k--', label='Virgo')
plt.plot(freqs, dets[3].psd(freqs), 'k:', label='KAGRA')
plt.loglog()
plt.xlim([10, 1000])
plt.ylim([2E-48, 2E-43])
plt.xlabel(r'$f\,\,[{\rm Hz}]$', fontsize=16)
plt.ylabel(r'$N_f\,\,[{\rm Hz}^{-1}]$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.legend(loc='upper right', fontsize=14, frameon=False)
plt.savefig("psd_LIGO.pdf", bbox_inches='tight')
plt.show()
