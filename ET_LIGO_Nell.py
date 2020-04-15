import numpy as np
from detector import GroundDetector, GroundDetectorTriangle
from mapping import MapCalculatorFromArray
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# First set these parameters:
#  - Observing time (in years)
t_obs = 1
#  - Reference frequency (in Hz)
f_ref = 63
#  - HEALPix resolution parameter
#    N_ell will be provided up to ell=3*nside
nside = 32
#  - Correlation coefficient between the different ET detectors
rE = -0.2

obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))


# This is the ET detector
et = [GroundDetectorTriangle(name='ET%d' % i, lat=41.9, lon=12.5,
                             fname_psd='data/curves_May_2019/et_d.txt',
                             detector_id=i)
      for i in range(3)]

# This are the other non-ET detectors
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


def compute_N_ell(mc, no_autos):
    nl = 1./(np.sum(mc.get_G_ell(0., freqs, nside,
                                 no_autos=no_autos),
                    axis=0) * obs_time * dfreq)
    return nl


# OK, so say you want to calculate stuff for Hanford-Livingstone
print("HL")
# You need to declare the map calculator as follows:
detectors = [dets['Hanford'], dets['Livingstone']]
mc_HL = MapCalculatorFromArray(detectors, f_pivot=f_ref)
# And this is how you calculate the corresponding noise power spectrum
# Note that you need to specify the detectors whose auto-correlations
# you want to ignore. In this case it's both of them:
no_autos = [True, True]
nl_HL = compute_N_ell(mc_HL, no_autos)

# OK, same as above, but now we add Virgo, and we want to
# check what happens if we include the Hanford auto-correlation
# (but not the other ones)
print("HLV")
detectors = [dets['Hanford'], dets['Livingstone'], dets['VIRGO']]
no_autos = [False, True, True]
mc_HLV = MapCalculatorFromArray(detectors, f_pivot=f_ref)
nl_HLV = compute_N_ell(mc_HLV, no_autos)

# OK, now let's ET. In this case, we need to specify a correlation
# matrix for ET. This should be 1 in the diagonal and the
# correlation coefficient in the off-diagonal:
print("E")
corr = np.array([[1., rE, rE],
                 [rE, 1., rE],
                 [rE, rE, 1.]])
detectors = et
# Also, let's include all autos
no_autos = [False, False, False]
mc_E = MapCalculatorFromArray(detectors, f_pivot=f_ref,
                              corr_matrix=corr)
nl_E = compute_N_ell(mc_E, no_autos)

# Now let's mix H, L, V (no autos) and ET. In total that's 6
# detectors. First we declare the full correlation matrix
# and then the auto-correlations to remove.
print("HLVE")
detectors = [dets['Hanford'], dets['Livingstone'], dets['VIRGO']] + et
corr = np.array([[1., 0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0., 0.],
                 [0., 0., 1., 0., 0., 0.],
                 [1., 0., 0., 1., rE, rE],
                 [1., 0., 0., rE, 1., rE],
                 [1., 0., 0., rE, rE, 1.]])
no_autos = [True, True, True, False, False, False]
mc_HLVE = MapCalculatorFromArray(detectors, f_pivot=f_ref,
                                 corr_matrix=corr)
nl_HLVE = compute_N_ell(mc_HLVE, no_autos)

ls = np.arange(3*nside)
plt.figure()
plt.plot(ls, (ls * nl_HL), 'r-', label='H+L')
plt.plot(ls, (ls * nl_HLV), 'g-', label='H(auto)+L+V')
plt.plot(ls[::2], (ls * nl_E)[::2], 'bo', label='ET')
plt.plot(ls[1::2], (ls * nl_E)[1::2], 'bx')
plt.plot(ls, (ls * nl_HLVE), 'b-', label='H+L+V+ET')
plt.ylim([1.5E-24, 1E-10])
plt.legend(loc='upper left')
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell\,N_\ell$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.loglog()
plt.savefig("plots/nl_ET_LIGO.pdf", bbox_inches='tight')
plt.show()
