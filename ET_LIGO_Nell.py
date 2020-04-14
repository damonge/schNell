import numpy as np
from detector import GroundDetector, GroundDetectorTriangle
from mapping import MapCalculatorFromArray
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica']})
rc('text', usetex=True)


t_obs = 1
f_ref = 63
nside = 32
rET = -0.2

et = [GroundDetectorTriangle(name='ET%d' % i, lat=41.9, lon=12.5,
                             fname_psd='data/curves_May_2019/et_d.txt',
                             detector_id=i)
      for i in range(3)]
dets = {'Hanford':     GroundDetector('Hanford',     46.4, -119.4, 171.8,
                                      'data/curves_May_2019/aligo_design.txt'),
        'Livingstone': GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                                      'data/curves_May_2019/aligo_design.txt'),
        'VIRGO':       GroundDetector('VIRGO',       43.6,   10.5, 116.5,
                                      'data/curves_May_2019/advirgo_sqz.txt'),
        'Kagra':       GroundDetector('Kagra',       36.3,  137.2, 225.0,
                                      'data/curves_May_2019/kagra_sqz.txt'),
        'GEO600':      GroundDetector('GEO600',      48.0,    9.8,  68.8,
                                      'data/curves_May_2019/o1.txt')}
obs_time = t_obs*365*24*3600.
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))

print("HL")
no_autos_HL = np.ones(2, dtype=bool)
mc_HL = MapCalculatorFromArray([dets['Hanford'], dets['Livingstone']],
                               f_pivot=f_ref)
nl_HL = 1./(np.sum(mc_HL.get_G_ell(0., freqs, nside,
                                   no_autos=no_autos_HL),
                   axis=0) * obs_time * dfreq)

print("HLV")
no_autos_HLV = np.ones(3, dtype=bool)
mc_HLV = MapCalculatorFromArray([dets['Hanford'], dets['Livingstone'],
                                 dets['VIRGO']], f_pivot=f_ref)
nl_HLV = 1./(np.sum(mc_HLV.get_G_ell(0., freqs, nside,
                                     no_autos=no_autos_HLV),
                    axis=0) * obs_time * dfreq)

print("ET")
no_autos_ET = np.zeros(3, dtype=bool)
rho_ET = np.identity(3)
for i in range(3):
    for j in range(i+1, 3):
        rho_ET[i, j] = rET
        rho_ET[j, i] = rET
print(rho_ET)
print(no_autos_ET)
mc_ET = MapCalculatorFromArray(et, f_pivot=f_ref, corr_matrix=rho_ET)
nl_ET = 1./(np.sum(mc_ET.get_G_ell(0., freqs, nside,
                                   no_autos=no_autos_ET),
                   axis=0) * obs_time * dfreq)

print("HLE")
no_autos_HLE = np.ones(5, dtype=bool)
rho_HLE = np.identity(5)
for i in range(3):
    no_autos_HLE[2+i] = False
    for j in range(i+1, 3):
        rho_HLE[2+i, 2+j] = rET
        rho_HLE[2+j, 2+i] = rET
print(rho_HLE)
print(no_autos_HLE)
mc_HLE = MapCalculatorFromArray([dets['Hanford'], dets['Livingstone']] +
                                et, f_pivot=f_ref, corr_matrix=rho_HLE)
nl_HLE = 1./(np.sum(mc_HLE.get_G_ell(0., freqs, nside,
                                     no_autos=no_autos_HLE),
                    axis=0) * obs_time * dfreq)

print("all")
no_autos_all = np.ones(8, dtype=bool)
rho_all = np.identity(8)
for i in range(3):
    no_autos_all[5+i] = False
    for j in range(i+1, 3):
        rho_all[5+i, 5+j] = rET
        rho_all[5+j, 5+i] = rET
print(rho_all)
print(no_autos_all)
mc_all = MapCalculatorFromArray([dets['Hanford'], dets['Livingstone'],
                                 dets['VIRGO'], dets['Kagra'],
                                 dets['GEO600']] + et,
                                f_pivot=f_ref, corr_matrix=rho_all)
nl_all = 1./(np.sum(mc_all.get_G_ell(0., freqs, nside,
                                     no_autos=no_autos_all),
                    axis=0) * obs_time * dfreq)

ls = np.arange(3*nside)
plt.figure()
plt.plot(ls, (ls * nl_HL), 'r-', label='HL')
plt.plot(ls, (ls * nl_HLV), 'g-', label='HLV')
plt.plot(ls, (ls * nl_HLE), 'b-', label='HLE')
plt.plot(ls[::2], (ls * nl_ET)[::2], 'yo', label='ET')
plt.plot(ls[1::2], (ls * nl_ET)[1::2], 'yx')
plt.plot(ls, (ls * nl_all), 'k--', label='all')
plt.ylim([1.5E-24, 1E-10])
plt.legend(loc='upper left')
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$\ell\,N_\ell$', fontsize=16)
plt.gca().tick_params(labelsize="large")
plt.loglog()
plt.savefig("plots/nl_ET_LIGO.pdf", bbox_inches='tight')
plt.show()
