import numpy as np
import healpy as hp
from detector import LISADetector
from mapping import MapCalculator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

det_b = LISADetector(0, map_transfer=True, is_L5Gm=True)
det_0 = LISADetector(0, map_transfer=True)
det_0b = LISADetector(0, map_transfer=False)
det_1 = LISADetector(1, map_transfer=True)
fs = np.geomspace(1E-5, 1, 1024)

# Sensitivity curves
plt.figure()
plt.plot(fs, np.sqrt(det_0.psd(fs)), 'k-')
plt.plot(fs, np.sqrt(det_0b.psd(fs)), 'r-')
plt.loglog()
plt.xlabel(r'$f\,[{\rm Hz}]$', fontsize=14)
plt.ylabel(r'$\sqrt{S_n(f)}\,[{\rm Hz}^{-1/2}]$', fontsize=14)
plt.savefig("psd.png", bbox_inches='tight')

# Arm lengths
t_arr = np.linspace(0, 365*24*3600, 1024)
pos = det_0.pos_all(t_arr)
dx_01 = np.sqrt(np.sum((pos[0] - pos[1])**2, axis=0))
dx_12 = np.sqrt(np.sum((pos[1] - pos[2])**2, axis=0))
dx_20 = np.sqrt(np.sum((pos[2] - pos[0])**2, axis=0))
plt.figure()
plt.plot(t_arr / (365 * 24 * 3600), dx_01, label=r'$L_{12}$')
plt.plot(t_arr / (365 * 24 * 3600), dx_12, label=r'$L_{23}$')
plt.plot(t_arr / (365 * 24 * 3600), dx_20, label=r'$L_{31}$')
plt.xlabel(r'$t\,\,[{\rm yr}]$', fontsize=14)
plt.ylabel(r'${\rm Arm\,\,length\,\,[{\rm m}]}$', fontsize=14)
plt.legend(loc='center left', fontsize=13)
plt.savefig("arm_length.png", bbox_inches='tight')

# Spacecraft trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')
for ip, p in enumerate(pos):
    ax.plot(p[0] / det_0.R_AU, p[1] / det_0.R_AU, p[2] / det_0.R_AU,
            label=r'${\rm %d-th\,\,spacecraft}$' % (ip+1))
ax.set_zlim([-0.1, 0.1])
ax.legend()
plt.savefig("trajectories.png", bbox_inches='tight')

# Antenna patterns (matching Fig. 3 of astro-ph/0105374)
mc_b = MapCalculator(det_b, det_b, f_pivot=1E-2)
nside = 64
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))
for f in [1E-3, 1E-2, 5E-2, 0.1]:
    gamma = np.abs(mc_b.get_gamma(0, f, theta, phi, inc_baseline=False))
    hp.mollview(gamma, coord=['E', 'G'],
                title=r'$\nu = %lf\,{\rm Hz}$' % f)

# Power spectra
mc_00 = MapCalculator(det_0, det_0, f_pivot=1E-2)
mc_01 = MapCalculator(det_0, det_1, f_pivot=1E-2)

nside = 32
lfreqs = np.linspace(-4, 0, 101)
dlfreq = np.mean(np.diff(lfreqs))
freqs = 10.**lfreqs
dfreqs = dlfreq * np.log(10.) * freqs
obs_time = 365 * 24 * 3600
ls = np.arange(3*nside)
nl_00 = 1./(np.sum(mc_00.get_G_ell(0, freqs, nside) * dfreqs[:, None],
                   axis=0) * obs_time)
nl_01 = 1./(np.sum(mc_01.get_G_ell(0, freqs, nside) * dfreqs[:, None],
                   axis=0) * obs_time)

plt.figure()
plt.plot(ls[::2], (ls*(ls+1)*nl_00/(2*np.pi))[::2], 'ro-', label='Auto')
plt.plot(ls[::2], (ls*(ls+1)*nl_01/(2*np.pi))[::2], 'ko-', label='Cross')
plt.xlim([1, 60])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell(\ell+1)N_\ell/2\pi$', fontsize=16)
plt.loglog()
plt.legend(loc='upper left')
plt.savefig("nls_lisa.pdf", bbox_inches='tight')
plt.show()
