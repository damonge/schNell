import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


t_obs = 1
f_ref = 63.
nside = 16
obs_time = t_obs*365*24*3600.
freqs = np.geomspace(10., 1010., 1010)


dets = [snl.GroundDetector('Hanford',     46.4, -119.4, 171.8,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Virgo',       43.6,   10.5, 116.5,
                           'data/Virgo.txt'),
        snl.GroundDetector('KAGRA',       36.3,  137.2, 225.0,
                           'data/KAGRA.txt')]
mc = snl.MapCalculator(dets[:2], f_pivot=f_ref)
pi2_HL = mc.get_pi_curve(obs_time, freqs, nside,
                         no_autos=True, nsigma=2)
mc = snl.MapCalculator(dets[:3], f_pivot=f_ref)
pi2_HLV = mc.get_pi_curve(obs_time, freqs, nside,
                          no_autos=True, nsigma=2)
mc = snl.MapCalculator(dets[:4], f_pivot=f_ref)
pi2_HLVK = mc.get_pi_curve(obs_time, freqs, nside,
                           no_autos=True, nsigma=2)
pi2_HLVKa = mc.get_pi_curve(obs_time, freqs, nside,
                            no_autos=False, nsigma=2)

# Plotting
plt.plot(freqs, pi2_HL, 'k--', label='LIGO')
plt.plot(freqs, pi2_HLV, 'k:', label='+ Virgo')
plt.plot(freqs, pi2_HLVK, 'k-', label='+ KAGRA')
plt.plot(freqs, pi2_HLVKa, 'k-.', label='+ auto correlations')
plt.loglog()
plt.xlim([10, 1000])
plt.ylim([1E-10, 1E-5])
plt.xlabel(r'$f\,[{\rm Hz}]$', fontsize=16)
plt.ylabel(r'$\Omega_{\rm GW}(f)$', fontsize=16)
plt.legend(loc='upper left', fontsize='x-large', frameon=False)
plt.gca().tick_params(labelsize="large")
plt.savefig("pi_LIGO.pdf", bbox_inches='tight')
plt.show()
