import numpy as np
from detector import LISADetector
from mapping import MapCalculator, MapCalculatorFromArray
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# First set these parameters:
#  - Observing time (in years)
t_obs = 4
#  - Reference frequency (in Hz)
f_ref = 1E-2
#  - HEALPix resolution parameter
#    N_ell will be provided up to ell=3*nside
nside = 32

########################################
# PROBABLY NO NEED TO TOUCH ANY OF THIS
#
# Initialize detectors
det_0 = LISADetector(0, map_transfer=True)
det_1 = LISADetector(1, map_transfer=True)
det_2 = LISADetector(2, map_transfer=True)
# Initialize the map calculator
mc = MapCalculator(det_0, det_1, f_pivot=f_ref)
# Correlation between detectors
r = -0.2
rho = np.array([[1, r, r],
                [r, 1, r],
                [r, r, 1]])
mca = MapCalculatorFromArray([det_0, det_1, det_2], f_pivot=f_ref,
                             corr_matrix=rho)
mcb = MapCalculatorFromArray([det_0], f_pivot=f_ref)
# Some internal arrays
lfreqs = np.linspace(-4, 0, 101)
dlfreq = np.mean(np.diff(lfreqs))
freqs = 10.**lfreqs
dfreqs = dlfreq * np.log(10.) * freqs
obs_time = t_obs * 365 * 24 * 3600
#
########################################


# This function will give you the inverse noise power spectrum
# for two effective detector combinations. The detector
# combinations you can pass are ('+,+', '+,-', '-,-') or
# ('I,I', 'I,II', 'II,II'), corresponding to the two
# possible decompositions of LISA we currently have in the
# draft.
def get_inverse_nell(comb):
    gls = mc.get_G_ell(0, freqs, nside, typ=comb)
    i_nl = np.sum(gls * dfreqs[:, None], axis=0) * obs_time
    return i_nl


# For instance this will give you the spectra for all the
# +/- combinations and the total spectrum (by inverse-variance
# summing the individual combinations):
inl_pp = get_inverse_nell('+,+')
inl_pm = get_inverse_nell('+,-')
inl_mm = get_inverse_nell('-,-')
nl_total = 1./(inl_pp+inl_pm+inl_mm)

gla = mca.get_G_ell(0, freqs, nside) * dfreqs[:, None]
nl_total_a = 1./(np.sum(gla, axis=0) * obs_time)
glb = mcb.get_G_ell(0, freqs, nside) * dfreqs[:, None]
nl_total_b = 1./(np.sum(glb, axis=0) * obs_time)

# Then you can plot. Note that, as we said, all the odd
# ells have a horrible noise, so you can skip plotting them
# by evaluating the arrays at [::2] (i.e. skipping every second
# element).
# If you wanted to look at the odd multipoles, you could do
# [1::2]

ls = np.arange(3*nside)

plt.figure()
plt.plot(ls[::2], ((ls+0.5)*nl_total_b)[::2], 'ko--', label='Single-detector (even $\\ell$)')
plt.plot(ls[::2], ((ls+0.5)*nl_total_a)[::2], 'ko-', label='Array (even $\\ell$)')
plt.plot(ls[1::2], ((ls+0.5)*nl_total_a)[1::2], 'kx', label='Array (odd $\\ell$)')
plt.xlim([1, 20])
plt.ylim([5E-28, 9E-16])
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$(\ell+1/2)\,N_\ell$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.loglog()
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("plots/nl_LISA.pdf", bbox_inches='tight')
plt.show()
