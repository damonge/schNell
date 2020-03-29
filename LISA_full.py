import numpy as np
from detector import LISADetector
from mapping import MapCalculator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

det_0 = LISADetector(0, map_transfer=True)
det_1 = LISADetector(1, map_transfer=True)
mc = MapCalculator(det_0, det_1, f_pivot=1E-2)

nside = 32
lfreqs = np.linspace(-4, 0, 101)
dlfreq = np.mean(np.diff(lfreqs))
freqs = 10.**lfreqs
dfreqs = dlfreq * np.log(10.) * freqs
obs_time = 365 * 24 * 3600
ls = np.arange(3*nside)
nl_I_I = 1./np.sum(mc.get_G_ell(0, freqs, nside, typ='I,I') *
                   dfreqs[:, None] * obs_time, axis=0)
nl_I_II = 1./np.sum(mc.get_G_ell(0, freqs, nside, typ='I,II') *
                    dfreqs[:, None] * obs_time, axis=0)
nl_II_II = 1./np.sum(mc.get_G_ell(0, freqs, nside, typ='II,II') *
                     dfreqs[:, None] * obs_time, axis=0)
nl_p_p = 1./np.sum(mc.get_G_ell(0, freqs, nside, typ='+,+') *
                   dfreqs[:, None] * obs_time, axis=0)
nl_m_m = 1./np.sum(mc.get_G_ell(0, freqs, nside, typ='-,-') *
                   dfreqs[:, None] * obs_time, axis=0)
nl_p_m = 1./np.sum(mc.get_G_ell(0, freqs, nside, typ='+,-') *
                   dfreqs[:, None] * obs_time, axis=0)
nl_I_II_sum = 1./(1./nl_I_I + 1./nl_I_II + 1./nl_II_II)
nl_p_m_sum = 1./(1./nl_p_p + 1./nl_m_m + 1./nl_p_m)

plt.figure()
plt.plot(ls[::2], (ls*(ls+1)*nl_I_I/(2*np.pi))[::2], 'ro--',
         label='I,I')
plt.plot(ls[::2], (ls*(ls+1)*nl_II_II/(2*np.pi))[::2], 'rv-.',
         label='II,II')
plt.plot(ls[::2], (ls*(ls+1)*nl_I_II/(2*np.pi))[::2], 'rd:',
         label='I,II')
plt.plot(ls[::2], (ls*(ls+1)*nl_I_II_sum/(2*np.pi))[::2], 'rx-',
         label='I,II - comb')
plt.xlim([1, 60])
plt.ylim([2E-27, 6E-11])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell(\ell+1)N_\ell/2\pi$', fontsize=16)
plt.loglog()
plt.legend(loc='upper left')
plt.savefig("LISA_I_II.pdf", bbox_inches='tight')

plt.figure()
plt.plot(ls[::2], (ls*(ls+1)*nl_p_p/(2*np.pi))[::2], 'bo--',
         label='+,+')
plt.plot(ls[::2], (ls*(ls+1)*nl_m_m/(2*np.pi))[::2], 'bv-.',
         label='-,-')
plt.plot(ls[::2], (ls*(ls+1)*nl_p_m/(2*np.pi))[::2], 'bd:',
         label='+,-')
plt.plot(ls[::2], (ls*(ls+1)*nl_p_m_sum/(2*np.pi))[::2], 'bx-',
         label='+,- - comb')
plt.xlim([1, 60])
plt.ylim([2E-27, 6E-11])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell(\ell+1)N_\ell/2\pi$', fontsize=16)
plt.loglog()
plt.legend(loc='upper left')
plt.savefig("LISA_p_m.pdf", bbox_inches='tight')
plt.show()
