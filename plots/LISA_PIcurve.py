import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


t_obs = 4
f_ref = 1E-2
nside = 16

dets = [snl.LISADetector(i) for i in range(3)]
rho = snl.NoiseCorrelationLISA(dets[0])
mc = snl.MapCalculator(dets, f_pivot=f_ref,
                       corr_matrix=rho)
freqs = np.geomspace(1E-4, 0.2, 1001)
obs_time = t_obs * 365 * 24 * 3600

pi_LISA = mc.get_pi_curve(obs_time, freqs, nside,
                          is_fspacing_log=True,
                          nsigma=5, proj={'vectors': np.ones(3),
                                          'deproject': True})

# Plotting
plt.plot(freqs, pi_LISA, 'k--', label='LISA')
plt.loglog()
plt.xlabel(r'$f\,[{\rm Hz}]$', fontsize=16)
plt.ylabel(r'$\Omega_{\rm GW}(f)$', fontsize=16)
plt.legend(loc='upper left', fontsize='x-large', frameon=False)
plt.gca().tick_params(labelsize="large")
plt.show()
