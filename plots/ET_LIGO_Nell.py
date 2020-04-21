import numpy as np
import snell as snl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

t_obs = 1
f_ref = 63
nside = 64
rE = -0.2

obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))

et = [snl.GroundDetectorTriangle(name='ET%d' % i, lat=40.1, lon=9.0,
                                 fname_psd='../data/curves_May_2019/et_d.txt',
                                 detector_id=i)
      for i in range(3)]
dets = {'Hanford':     snl.GroundDetector('Hanford',     46.4, -119.4, 171.8,
                                          '../data/curves_May_2019/NewaPlusLIGO.dat'),
        'Livingstone': snl.GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                                          '../data/curves_May_2019/NewaPlusLIGO.dat'),
        'VIRGO':       snl.GroundDetector('VIRGO',       43.6,   10.5, 116.5,
                                          '../data/curves_May_2019/NewVirgoO5High.dat'),
        'Kagra':       snl.GroundDetector('Kagra',       36.3,  137.2, 225.0,
                                          '../data/curves_May_2019/NewKAGRA128Mpc.dat'),
        'GEO600':      snl.GroundDetector('GEO600',      48.0,    9.8,  68.8,
                                          '../data/curves_May_2019/o1.txt')}

print("HLVK")
detectors = [dets['Hanford'], dets['Livingstone'], dets['VIRGO'], dets['Kagra']]
no_autos = [True, True, True, True]
mc_HLVK = snl.MapCalculator(detectors, f_pivot=f_ref)
nl_HLVK = mc_HLVK.get_N_ell(obs_time, freqs, nside,
                            no_autos=no_autos)
detectors = [dets['Hanford'], dets['Livingstone'],
             dets['VIRGO'], dets['Kagra']] + et
corr = np.array([[1., 0., 0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 1., 0., 0., 0., 0.],
                 [0., 0., 0., 1., 0., 0., 0.],
                 [0., 1., 0., 0., 1., rE, rE],
                 [0., 1., 0., 0., rE, 1., rE],
                 [0., 1., 0., 0., rE, rE, 1.]])
mc_HLVKE = snl.MapCalculator(detectors, f_pivot=f_ref,
                             corr_matrix=corr)
print("HLVKEx")
no_autos = [[True,  False, False, False, False, False, False],
            [False, True,  False, False, False, False, False],
            [False, False, True,  False, False, False, False],
            [False, False, False, True,  False, False, False],
            [False, False, False, False, True,  True,  True],
            [False, False, False, False, True,  True,  True],
            [False, False, False, False, True,  True,  True]]
nl_HLVKEx = mc_HLVKE.get_N_ell(obs_time, freqs, nside,
                               no_autos=no_autos)
print("HLVKE")
no_autos = [True, True, True, True, False, False, False]
nl_HLVKE = mc_HLVKE.get_N_ell(obs_time, freqs, nside,
                              no_autos=no_autos)


ls = np.arange(3*nside)
plt.figure()
plt.plot(ls, (ls+0.5)*nl_HLVK, 'k:',label='LIGO + Virgo + KAGRA')
plt.plot(ls, (ls+0.5)*nl_HLVKEx, 'k-',label=' + ET (cross-only)')
plt.plot(ls, (ls+0.5)*nl_HLVKE, 'k-.',label=' + ET (all)')
plt.loglog()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$(\ell+1/2)\,N_\ell$', fontsize=16)
plt.ylim([3E-24, 1E-10])
plt.xlim([1,100])
plt.legend(loc='upper left', fontsize='x-large', frameon=False)
plt.gca().tick_params(labelsize="large")
plt.savefig("ET_LIGO_Nell.pdf", bbox_inches='tight')
plt.show()
