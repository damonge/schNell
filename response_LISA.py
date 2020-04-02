import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from detector import LISADetector
from mapping import MapCalculator
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


det_0 = LISADetector(0, map_transfer=True)
det_1 = LISADetector(1, map_transfer=True)
mc = MapCalculator(det_0, det_1, f_pivot=1E-2)

nside = 32
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))

plt.figure(figsize=(7, 8))
for i, f in enumerate([0.001, 0.01, 0.1, 0.2]):
    resp_pp = mc.get_gamma(0., f, theta, phi, inc_baseline=True, typ='+,+')
    resp_mm = mc.get_gamma(0., f, theta, phi, inc_baseline=True, typ='-,-')
    fs = ('%f' % f).rstrip('0').rstrip('.')
    hp.mollview(np.abs(resp_pp), sub=420+2*i+1, coord=['E', 'G'],
                notext=True, cbar=False,
                title=r'$|{\cal R}_{+\,+}(\hat{\bf n})|,\,\,f=%s\,{\rm Hz}$' % fs)
    hp.mollview(np.abs(resp_mm), sub=420+2*i+2, coord=['E', 'G'],
                notext=True, cbar=False,
                title=r'$|{\cal R}_{-\,-}(\hat{\bf n})|,\,\,f=%s\,{\rm Hz}$' % fs)
plt.savefig("plots/antenna_LISA.pdf", bbox_inches='tight')
plt.show()
