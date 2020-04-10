import numpy as np
import matplotlib.pyplot as plt
from detector import GroundDetector
from mapping import MapCalculatorFromArray
from scipy.interpolate import interp1d


# First set these parameters:
#  - Observing time (in years)
t_obs = 1
#  - Reference frequency (in Hz)
f_ref = 63.
#  - HEALPix resolution parameter
#    N_ell will be provided up to ell=3*nside
nside = 16


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
mcal = MapCalculatorFromArray([dets['Hanford'],
                               dets['Livingstone'],
                               dets['VIRGO']],
                              f_pivot=f_ref)

# Some internal arrays
obs_time = t_obs*365*24*3600.
# Calculate 1/sigma^2 for logarithmic spacing
freqs = np.geomspace(10., 1010., 1010)
inv_dsig2_dnu_dt = mcal.get_dsigm2_dnu_t(0, freqs, nside,
                                         no_autos=True)
# Interpolate to linear spacing
sens_i = interp1d(np.log10(freqs), inv_dsig2_dnu_dt)
freqs = np.linspace(10., 1010., 1010)
inv_dsig2_dnu_dt = sens_i(np.log10(freqs))
dfreq = np.mean(np.diff(freqs))


# Get PI curve
def Ob(beta, rho=1):
    snm2 = np.sum(inv_dsig2_dnu_dt *
                  (freqs/f_ref)**(2*beta))*dfreq
    return rho * (freqs/f_ref)**beta / np.sqrt(obs_time*snm2)


betas = np.linspace(-10, 10, 100)
obs = np.array([Ob(b, rho=2) for b in betas])
pi_2sigma = np.max(obs, axis=0)

# Plotting
plt.plot(freqs, pi_2sigma)
plt.loglog()
plt.xlim([10, 1000])
plt.ylim([1E-10, 1E-5])
plt.xlabel(r'$f\,[{\rm Hz}]$', fontsize=15)
plt.ylabel(r'$\Omega_{\rm GW}(f)$', fontsize=15)
plt.show()
