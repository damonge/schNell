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
                           'data/KAGRA.txt'),
        snl.GroundDetector('Cosmic Explorer', 37.24804, -115.800155, 0.,
                           'data/CE1_strain.txt')]
et = snl.GroundDetectorTriangle(name='ET0', lat=40.1, lon=9.0,
                                fname_psd='data/ET.txt', detector_id=0)


plt.figure()
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


freqsa = np.geomspace(6, 5000., 3072)
freqsb = np.geomspace(1., 10010., 3072)
plt.figure()
plt.plot(freqsb, et.psd(freqsb), 'k-', label='ET-D')
plt.plot(freqsb, dets[4].psd(freqsb), 'k--', label='CE-S1')
plt.plot(freqsa, dets[0].psd(freqsa), 'k:', label='LIGO A+')
plt.xlim([1.5, 1E4])
plt.ylim([5E-50, 9E-42])
plt.loglog()
plt.xlabel(r'$f\,\,[{\rm Hz}]$', fontsize=16)
plt.ylabel(r'$N_f\,\,[{\rm Hz}^{-1}]$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.gca().set_yticks([1E-48, 1E-46, 1E-44, 1E-42])
plt.legend(loc='upper right', fontsize=14, frameon=False)
plt.savefig("psd_ET.pdf", bbox_inches='tight')
plt.show()
