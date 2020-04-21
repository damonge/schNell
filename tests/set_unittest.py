import healpy as hp
import numpy as np
from detector import GroundDetector, LISADetector
from mapping import MapCalculator
import os


os.system('mkdir -p test_data')

# Detectors
det1 = GroundDetector('Hanford',     46.4, -119.4, 171.8,
                      'data/curves_May_2019/aligo_design.txt')
det2 = GroundDetector('Livingstone', 30.7,  -90.8, 243.0,
                      'data/curves_May_2019/aligo_design.txt')
detl = LISADetector(0, map_transfer=True, is_L5Gm=False)


# Calculators
mc11 = MapCalculator(det1, det1)
mc12 = MapCalculator(det1, det2)
mcLL = MapCalculator(detl, detl, f_pivot=1E-2)


# Gamma maps
nside = 64
theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
g11 = mc11.get_gamma(0, 0, theta, phi)
g12 = mc12.get_gamma(0, 0, theta, phi)
gLL = mcLL.get_gamma(0, 1E-2, theta, phi)
hp.write_map("test_data/gamma_test.fits", [g11, g12, gLL], overwrite=True)


# G_ells
ls = np.arange(3*nside)
gl11 = mc11.get_G_ell(0, 100., nside)
gl12 = mc12.get_G_ell(0, 100., nside)
glLL = mcLL.get_G_ell(0, 1E-2, nside)
np.savetxt("test_data/gls_test.txt",
           np.transpose([ls, gl11, gl12, glLL]))
