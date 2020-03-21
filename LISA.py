import numpy as np
from detector import LISADetector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
    
det = LISADetector(0)
det_b = LISADetector(0, map_transfer=True)
fs = np.geomspace(1E-5, 1, 1024)

plt.figure()
plt.plot(fs, np.sqrt(det.psd(fs)), 'k-')
plt.plot(fs, np.sqrt(det_b.psd(fs)), 'r-')
plt.loglog()
plt.xlabel(r'$f\,[{\rm Hz}]$', fontsize=14)
plt.ylabel(r'$\sqrt{S_n(f)}\,[{\rm Hz}^{-1/2}]$', fontsize=14)
plt.savefig("psd.png", bbox_inches='tight')

t_arr = np.linspace(0, 365*24*3600, 1024)
pos = det.pos_all(t_arr)
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

fig = plt.figure()
ax = fig.gca(projection='3d')
for ip, p in enumerate(pos):
    ax.plot(p[0] / det.R_AU, p[1] / det.R_AU, p[2] / det.R_AU,
            label=r'${\rm %d-th\,\,spacecraft}$' % (ip+1))
ax.set_zlim([-0.1, 0.1])
ax.legend()
plt.savefig("trajectories.png", bbox_inches='tight')
plt.show()
