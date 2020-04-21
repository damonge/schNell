import numpy as np
import snell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

t_obs = 4
f_ref = 1E-2
nside = 32
r = -0.2

dets = [snl.LISADetector(i, map_transfer=True)
        for i in range(3)]
rho = np.array([[1, r, r],
                [r, 1, r],
                [r, r, 1]])
mca = snl.MapCalculator(dets, f_pivot=f_ref,
                        corr_matrix=rho)
mcb = snl.MapCalculator([dets[0]], f_pivot=f_ref)
freqs = np.geomspace(1E-4, 1, 101)
obs_time = t_obs * 365 * 24 * 3600

nl_total_a = mca.get_N_ell(obs_time, freqs, nside,
                           is_fspacing_log=True)
nl_total_b = mcb.get_N_ell(obs_time, freqs, nside,
                           is_fspacing_log=True)
ls = np.arange(3*nside)

plt.figure()
plt.plot(ls[::2], ((ls+0.5)*nl_total_b)[::2], 'ko--',
         label='Single-detector (even $\\ell$)')
plt.plot(ls[::2], ((ls+0.5)*nl_total_a)[::2], 'ko-',
         label='Array (even $\\ell$)')
plt.plot(ls[1::2], ((ls+0.5)*nl_total_a)[1::2], 'kx',
         label='Array (odd $\\ell$)')
plt.xlim([1, 20])
plt.ylim([5E-28, 9E-16])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$(\ell+1/2)\,N_\ell$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.loglog()
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("nl_LISA.pdf", bbox_inches='tight')
plt.show()
