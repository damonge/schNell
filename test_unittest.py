import healpy as hp
import numpy as np
# import matplotlib.pyplot as plt
from detector import GroundDetector, LISADetector
from mapping import MapCalculator


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

# Angles
nside = 64
theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))


def test_gamma():
    g11 = np.real(mc11.get_gamma(0, 0, theta, phi, inc_baseline=False))
    g12 = np.real(mc12.get_gamma(0, 0, theta, phi, inc_baseline=False))
    gLL = np.abs(mcLL.get_gamma(0, 1E-2, theta, phi, inc_baseline=False))
    g11_test, g12_test, gLL_test = hp.read_map("test_data/gamma_test.fits",
                                               field=None)
    assert np.all(np.fabs(g11-g11_test) < 1E-5)
    assert np.all(np.fabs(g12-g12_test) < 1E-5)
    assert np.all(np.fabs(gLL-gLL_test) < 1E-5)


def test_Gell():
    # The factor 2 here corrects for a pevious missing factor
    # for auto-correlations.
    gl11 = mc11.get_G_ell(0, 100., nside) * 2
    gl12 = mc12.get_G_ell(0, 100., nside)
    glLL = mcLL.get_G_ell(0, 1E-2, nside) * 2
    ls, gl11_test, gl12_test, glLL_test = np.loadtxt("test_data/gls_test.txt",
                                                     unpack=True)
    assert np.all(np.fabs(gl11-gl11_test) < 1E-8)
    assert np.all(np.fabs(gl12-gl12_test) < 1E-8)
    assert np.all(np.fabs(glLL-glLL_test) < 1E-8)
