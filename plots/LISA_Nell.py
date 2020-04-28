import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

t_obs = 4
f_ref = 1E-2
nside = 32

dets = [snl.LISADetector(i) for i in range(3)]
rho = snl.NoiseCorrelationLISA(dets[0])
mc_auto = snl.MapCalculator([dets[0]], f_pivot=f_ref)
mca = snl.MapCalculator(dets, f_pivot=f_ref,
                        corr_matrix=rho)
freqs = np.geomspace(1E-4, 1, 201)
obs_time = t_obs * 365 * 24 * 3600

nl_auto = mc_auto.get_N_ell(obs_time, freqs, nside,
                            is_fspacing_log=True)
nl_AE = mca.get_N_ell(obs_time, freqs, nside,
                      is_fspacing_log=True,
                      proj={'vectors': np.ones(3),
                            'deproject': True})
nl_all = mca.get_N_ell(obs_time, freqs, nside,
                       is_fspacing_log=True)
ls = np.arange(3*nside)

plt.figure()
plt.plot(ls[::2], ((ls+0.5)*nl_auto)[::2], 'ko-.',
         label='Single-detector (even $\\ell$)')
plt.plot(ls[::2], ((ls+0.5)*nl_AE)[::2], 'ko--',
         label='A and E channels (even $\\ell$)')
plt.plot(ls[::2], ((ls+0.5)*nl_all)[::2], 'ko-',
         label='All data (even $\\ell$)')
plt.plot(ls[1::2], ((ls+0.5)*nl_all)[1::2], 'kx:',
         label='All data (odd $\\ell$)')
plt.xlim([-0.5, 10.9])
plt.ylim([5E-28, 5E-14])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$(\ell+1/2)\,N_\ell$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.yscale('log')
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("nl_LISA.pdf", bbox_inches='tight')
plt.show()
