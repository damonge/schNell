import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from detector import GroundDetector
from mapping import MapCalculator


dum_det = GroundDetector('Dummy', 0, 0, 0,
                         'data/curves_May_2019/aligo_design.txt')
# are these per-detector PSDs?

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

mcal_dd = MapCalculator(dum_det, dum_det)
mcals = {s1: {s2: MapCalculator(d1, d2)
              for s2, d2 in dets.items()}
         for s1, d1 in dets.items()}

mcal_dd.plot_gamma(0, 0)

nside = 64
npix = hp.nside2npix(nside)
pix_area = 4*np.pi/npix
theta, phi = hp.pix2ang(nside, np.arange(npix))

names = list(dets.keys())
ind1, ind2 = np.triu_indices(len(names), k=0)
for i1, i2 in zip(ind1, ind2):
    s1 = names[i1]
    s2 = names[i2]
    print(s1, s2,
          np.sum(np.real(mcals[s1][s2].get_gamma(0, 0, theta, phi,
                                                 inc_baseline=False)*pix_area)))

obs_time = 365*24*3600.
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))

ls = np.arange(3*nside)
nlt = np.zeros(3*nside)
ind1, ind2 = np.triu_indices(len(names), k=1)
plt.figure(figsize=(12, 6))
for i1, i2 in zip(ind1, ind2):
    s1 = names[i1]
    s2 = names[i2]
    nl = np.sum(mcals[s1][s2].get_G_ell(0, freqs, nside),
                axis=0) * obs_time * dfreq
    nlt += nl
    print(s1, s2)
    plt.plot(ls, ls*(ls+1.)/(2*np.pi*nl), '--',
             label='%s-%s' % (s1, s2))
nlt = 1./nlt
np.savetxt("nls.txt", np.transpose([ls, nl]))
plt.plot(ls, ls * (ls + 1.) * nlt / (2 * np.pi), 'k-', label='Total')
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell(\ell+1)N_\ell/2\pi$', fontsize=16)
plt.loglog()
plt.ylim([3E-21, 1E-6])
plt.legend(loc='upper left', ncol=2)
plt.savefig("nls_ligo.pdf", bbox_inches='tight')
plt.show()
