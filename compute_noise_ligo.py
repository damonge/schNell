import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from detector import Detector, GroundDetector
from mapping import MapCalculator


dets = {'Dummy':       GroundDetector('Dummy',         0.,     0.,    0.,
                                      'data/curves_May_2019/aligo_design.txt'),  # are these per-detector PSDs?
        'Hanford':     GroundDetector('Hanford',     46.4, -119.4, 171.8,
                                      'data/curves_May_2019/aligo_design.txt'),
        'Livingstone': GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                                      'data/curves_May_2019/aligo_design.txt'),
        'VIRGO':       GroundDetector('VIRGO',       43.6,   10.5, 116.5,
                                      'data/curves_May_2019/advirgo_sqz.txt'),
        'Kagra':       GroundDetector('Kagra',       36.3,  137.2, 225.0,
                                      'data/curves_May_2019/kagra_sqz.txt'),
        'GEO600':      GroundDetector('GEO600',      48.0,    9.8,  68.8,
                                      'data/curves_May_2019/o1.txt')}

mcals = {s1: {s2: MapCalculator(d1, d2)
              for s2, d2 in dets.items()}
         for s1, d1 in dets.items()}

mcals['Dummy']['Dummy'].plot_gamma()
plt.show()
exit(1)
    
nside=64
theta, phi = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)))
hp.mollview(mcals['Dummy']['Dummy'].get_gamma(0, theta, phi),
            title=r"$\gamma^I(\theta,\varphi),\,\,{\rm %s}$",
            notext=True)
plt.show(); exit(1)
names = list(dets.keys())
for i1, s1 in enumerate(names):
    for s2 in names[i1:]:
        print(s1, s2, np.sum(mcals[s1][s2].get_gamma(0, theta, phi)*4*np.pi/hp.nside2npix(nside)))
        hp.mollview(mcals[s1][s2].get_gamma(0, theta, phi), coord=['C','G'],
                    title=r"$\gamma^I(\theta,\varphi),\,\,{\rm %s}-{\rm %s}$" % (s1, s2),
                    notext=True)
plt.show()
exit(1)
plt.savefig("overlap_HH.pdf", bbox_inches='tight')
hp.mollview(mcals['Hanford']['Livingstone'].get_gamma(0, theta, phi), coord=['C','G'],
            title=r"$\gamma^I(\theta,\varphi)$", notext=True)
plt.savefig("overlap_HL.pdf", bbox_inches='tight')
hp.mollview(mcals['Livingstone']['Livingstone'].get_gamma(0, theta, phi), coord=['C','G'],
            title=r"$\gamma^I(\theta,\varphi)$", notext=True)
plt.savefig("overlap_LL.pdf", bbox_inches='tight')

plt.figure()
obs_time = 1000*365*24*3600.
nl = np.zeros(3*nside)
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))
for f in freqs:
    print(f)
    n = mcals['Hanford']['Livingstone'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Hanford']['VIRGO'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Hanford']['Kagra'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Hanford']['GEO600'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Livingstone']['VIRGO'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Livingstone']['Kagra'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Livingstone']['GEO600'].get_G_ell(0, f, nside) * dfreq
    n += mcals['VIRGO']['Kagra'].get_G_ell(0, f, nside) * dfreq
    n += mcals['VIRGO']['GEO600'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Kagra']['GEO600'].get_G_ell(0, f, nside) * dfreq
    nl += n
nl *= obs_time
ls = np.arange(len(nl))
nl = 1./nl
plt.plot(ls, ls * (ls + 1.) * nl / (2 * np.pi), 'k--')
plt.plot(ls, 4E-26 * ls ** (5. / 6.), 'r-')
plt.xlabel('$\\ell$', fontsize=16)
plt.ylabel('$N_\\ell$', fontsize=16)
plt.loglog()
plt.show()
