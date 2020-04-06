import numpy as np
import matplotlib.pyplot as plt
from detector import GroundDetector
from mapping import MapCalculator, MapCalculatorFromArray


# First set these parameters:
#  - Observing time (in years)
t_obs = 1
#  - Reference frequency (in Hz)
f_ref = 63.
#  - HEALPix resolution parameter
#    N_ell will be provided up to ell=3*nside
nside = 64


########################################
# PROBABLY NO NEED TO TOUCH ANY OF THIS
#
# Initialize all detectors
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
mcal_all = MapCalculatorFromArray([dets['Hanford'],
                                   dets['Livingstone'],
                                   dets['VIRGO']], f_pivot=f_ref)
# Some internal arrays
obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))
#
########################################


# This function will give you the inverse noise power spectrum
# for two detectors with names name_A and name_B
def get_inverse_nell(name_A, name_B):
    mc = mcals[name_A][name_B]
    gls = mc.get_G_ell(0, freqs, nside)
    i_nl = np.sum(gls, axis=0) * obs_time * dfreq
    return i_nl


# If you want to get the total power spectrum for a set of
# pair combinations, first get the inverse power spectrum
# for each pair combination, then add them up, then invert
# the result. For example:
inl_HL = get_inverse_nell('Hanford', 'Livingstone')
inl_HV = get_inverse_nell('Hanford', 'VIRGO')
inl_LV = get_inverse_nell('Livingstone', 'VIRGO')
nl_total = 1./(inl_HL+inl_HV+inl_LV)
nl_totalb = 1./(np.sum(mcal_all.get_G_ell(0., freqs, nside, no_autos=True),
                       axis=0) * obs_time * dfreq)

# Then plot
ls = np.arange(3*nside)
plt.figure()
plt.plot(ls, ls/inl_HL, label='HL')
plt.plot(ls, ls/inl_HV, label='HV')
plt.plot(ls, ls/inl_LV, label='LV')
plt.plot(ls, ls*nl_total, label='Total')
plt.plot(ls, ls*nl_total, '--', label='Total (new)')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell\,N_\ell$', fontsize=16)
plt.ylim([3E-21, 1E-6])
plt.legend(loc='upper left', ncol=2)
plt.show()
