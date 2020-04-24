import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import schnell as snl
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


dets = [snl.LISADetector(i) for i in range(3)]
# Correlation between detectors
rho = snl.NoiseCorrelationLISA(dets[0])
mc = snl.MapCalculator(dets, f_pivot=1E-2,
                       corr_matrix=rho)

nside = 32
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))

fig = plt.figure(figsize=(7, 8))
freqs = [0.001, 0.01, 0.2, 0.5]
facs = [1., 1., 20., 100.]
sfac = ['', '', '20\\times', '100\\times']
for i, (f, fc, sfc) in enumerate(zip(freqs, facs, sfac)):
    resp_11 = mc.get_antenna(0, 0, 0., f, theta, phi,
                             inc_baseline=True)*facs[i]
    resp_12 = mc.get_antenna(0, 1, 0., f, theta, phi,
                             inc_baseline=True)*facs[i]
    fs = ('%f' % f).rstrip('0').rstrip('.')
    if i != 3:
        margins = [0.017] * 4
    else:
        margins = None
    hp.mollview(np.abs(resp_11), sub=420+2*i+1,
                notext=True, cbar=i == 3, max=0.14,
                min=0, margins=margins,
                title='$%s |{\\cal A}_{11}(f,\\hat{\\bf n})|,\\,\\,f=%s\\,{\\rm Hz}$' % (sfc, fs))
    hp.mollview(np.abs(resp_12), sub=420+2*i+2,
                notext=True, cbar=i == 3, max=0.07,
                min=0, margins=margins,
                title='$%s |{\\cal A}_{12}(f,\\hat{\\bf n})|,\\,\\,f=%s\\,{\\rm Hz}$' % (sfc, fs))
plt.savefig("antenna_LISA.pdf", bbox_inches='tight')
plt.show()
