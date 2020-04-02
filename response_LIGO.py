import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from detector import GroundDetector
from mapping import MapCalculator
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


f_ref = 63.
nside = 64

dets = {'Hanford':     GroundDetector('Hanford',     46.4, -119.4, 171.8,
                                      'data/curves_May_2019/aligo_design.txt'),
        'Livingstone': GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                                      'data/curves_May_2019/aligo_design.txt'),
        'VIRGO':       GroundDetector('VIRGO',       43.6,   10.5, 116.5,
                                      'data/curves_May_2019/advirgo_sqz.txt'),
        'Kagra':       GroundDetector('Kagra',       36.3,  137.2, 225.0,
                                      'data/curves_May_2019/kagra_sqz.txt'),
        'GEO600':      GroundDetector('GEO600',      48.0,    9.8,  68.8,
                                      'data/curves_May_2019/o1.txt')}
# Initialize the map calculator
mcals = {s1: {s2: MapCalculator(d1, d2, f_pivot=f_ref)
              for s2, d2 in dets.items()}
         for s1, d1 in dets.items()}

mc_HL = mcals['Hanford']['Livingstone']
mc_HV = mcals['Hanford']['VIRGO']
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))
plt.figure(figsize=(7, 8))
for i, lf in enumerate([1, 2, 3]):
    f = 10.**lf
    resp_HL = mc_HL.get_gamma(0., f, theta, phi)
    resp_HV = mc_HV.get_gamma(0., f, theta, phi)
    hp.mollview(np.real(resp_HL), sub=420+2*i+1, coord=['C', 'G'],
                notext=True, cbar=False,
                title=r'${\rm Re}({\cal R}_{\rm HL}(\hat{\bf n})),\,\,f=10^{%d}\,{\rm Hz}$' % lf)
    hp.mollview(np.real(resp_HV), sub=420+2*i+2, coord=['C', 'G'],
                notext=True, cbar=False,
                title=r'${\rm Re}({\cal R}_{\rm HV}(\hat{\bf n})),\,\,f=10^{%d}\,{\rm Hz}$' % lf)
gamma_HL = mc_HL.get_gamma(0., f, theta, phi, inc_baseline=False)
gamma_HV = mc_HV.get_gamma(0., f, theta, phi, inc_baseline=False)
hp.mollview(gamma_HL, sub=427, coord=['C', 'G'],
            notext=True, cbar=False,
            title=r'$\gamma_{\rm HL}(\hat{\bf n})$')
hp.mollview(gamma_HV, sub=428, coord=['C', 'G'],
            notext=True, cbar=False,
            title=r'$\gamma_{\rm HV}(\hat{\bf n})$')
plt.savefig("plots/antenna_LIGO.pdf", bbox_inches='tight')
plt.show()
