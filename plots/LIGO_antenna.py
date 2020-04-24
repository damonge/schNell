import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import schnell as snl
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


f_ref = 63.
nside = 64

dets = {'Hanford':     snl.GroundDetector('Hanford',     46.4, -119.4, 171.8,
                                          'data/aLIGO.txt'),
        'Livingstone': snl.GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                                          'data/aLIGO.txt'),
        'VIRGO':       snl.GroundDetector('Virgo',       43.6,   10.5, 116.5,
                                          'data/Virgo.txt'),
        'Kagra':       snl.GroundDetector('KAGRA',       36.3,  137.2, 225.0,
                                          'data/KAGRA.txt')}


def compute_nice_minmax(mp):
    mn = np.amin(mp)
    mx = np.amax(mp)
    emn = np.floor(np.log10(np.fabs(mn)))
    emx = np.ceil(np.log10(np.fabs(mx)))
    imn = int(10.**(-emn+1)*mn)
    imx = int(10.**(-emx+2)*mx)
    mno = imn*10.**(emn-1)
    mxo = imx*10.**(emx-2)
    return mno, mxo


# Initialize the map calculators
mc_HL = snl.MapCalculator([dets['Hanford'], dets['Livingstone']],
                          f_pivot=f_ref)
mc_HV = snl.MapCalculator([dets['Hanford'], dets['VIRGO']],
                          f_pivot=f_ref)
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))
plt.figure(figsize=(7, 9))
for i, lf in enumerate([1, 2, 3]):
    f = 10.**lf
    resp_HL = mc_HL.get_antenna(0, 1, 0., f, theta, phi)
    resp_HV = mc_HV.get_antenna(0, 1, 0., f, theta, phi)
    mn, mx = compute_nice_minmax(np.real(resp_HL))
    hp.mollview(np.real(resp_HL), sub=420+2*i+1,
                notext=True, cbar=True, min=mn, max=mx,
                title=r'${\rm Re}({\cal A}_{\rm HL}(f, \hat{\bf n})),\,\,f=10^{%d}\,{\rm Hz}$' % lf)
    mn, mx = compute_nice_minmax(np.real(resp_HV))
    hp.mollview(np.real(resp_HV), sub=420+2*i+2,
                notext=True, cbar=True, min=mn, max=mx,
                title=r'${\rm Re}({\cal A}_{\rm HV}(f, \hat{\bf n})),\,\,f=10^{%d}\,{\rm Hz}$' % lf)
antenna_HL = np.real(mc_HL.get_antenna(0, 1, 0., f, theta, phi,
                                       inc_baseline=False))
antenna_HV = np.real(mc_HV.get_antenna(0, 1, 0., f, theta, phi,
                                       inc_baseline=False))
mn, mx = compute_nice_minmax(antenna_HL)
hp.mollview(antenna_HL, sub=427,
            notext=True, cbar=True, min=mn, max=mx,
            title=r'$\gamma_{\rm HL}(f, \hat{\bf n})$')
mn, mx = compute_nice_minmax(antenna_HV)
hp.mollview(antenna_HV, sub=428,
            notext=True, cbar=True, min=mn, max=mx,
            title=r'$\gamma_{\rm HV}(f, \hat{\bf n})$')
plt.savefig("antenna_LIGO.pdf", bbox_inches='tight')
plt.show()
