import numpy as np
import schnell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

t_obs = 1
f_ref = 63
nside = 64
obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)

dets = [snl.GroundDetector('Hanford',     46.4, -119.4, 171.8,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                           'data/aLIGO.txt'),
        snl.GroundDetector('Virgo',       43.6,   10.5, 116.5,
                           'data/Virgo.txt'),
        snl.GroundDetector('KAGRA',       36.3,  137.2, 225.0,
                           'data/KAGRA.txt')]
print("0")
mc = snl.MapCalculator(dets, f_pivot=f_ref,
                       spectral_index=0.)
nl_a0 = mc.get_N_ell(obs_time, freqs, nside, no_autos=True)
print("2/3")
mc = snl.MapCalculator(dets, f_pivot=f_ref,
                       spectral_index=2./3.)
nl_a2o3 = mc.get_N_ell(obs_time, freqs, nside, no_autos=True)
print("3")
mc = snl.MapCalculator(dets, f_pivot=f_ref,
                       spectral_index=3.)
nl_a3 = mc.get_N_ell(obs_time, freqs, nside, no_autos=True)
ls = np.arange(3*nside)

plt.figure()
plt.plot(ls, (ls+0.5)*nl_a3, 'k--', label=r'$\alpha=3$')
plt.plot(ls, (ls+0.5)*nl_a2o3, 'k-', label=r'$\alpha=2/3$')
plt.plot(ls, (ls+0.5)*nl_a0, 'k:', label=r'$\alpha=0$')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$(\ell+1/2)\,N_\ell$', fontsize=16)
plt.ylim([3E-20, 1E-10])
plt.xlim([1, 100])
plt.legend(loc='upper left', fontsize='x-large', frameon=False)
plt.gca().tick_params(labelsize="large")
plt.savefig("Nell_alphas.pdf", bbox_inches='tight')
plt.show()
