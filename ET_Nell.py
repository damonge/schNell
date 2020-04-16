import numpy as np
from detector import GroundDetectorTriangle
from mapping import MapCalculatorFromArray
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


t_obs = 1
f_ref = 63
nside = 64

dets = [GroundDetectorTriangle(name='ET%d' % i, lat=40.1, lon=9.0,
                               fname_psd='data/curves_May_2019/et_d.txt',
                               detector_id=i)
        for i in range(3)]
r = -0.2
rho = np.array([[1, r, r],
                [r, 1, r],
                [r, r, 1]])
mc = MapCalculatorFromArray(dets, f_pivot=f_ref,
                            corr_matrix=rho)

obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))

gls = mc.get_G_ell(0, freqs, nside)
nl = 1./(np.sum(gls, axis=0) * obs_time * dfreq)

ls = np.arange(3*nside)

plt.figure()
plt.plot(ls[::2], (ls * nl)[::2], 'ro-')
plt.plot(ls[1::2], (ls * nl)[1::2], 'rx')
plt.xlim([1, 30])
plt.ylim([1.5E-24, 9E-10])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell\,N_\ell$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.loglog()
plt.savefig("plots/nl_ET.pdf", bbox_inches='tight')
plt.show()
