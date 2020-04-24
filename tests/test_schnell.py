import healpy as hp
import numpy as np
from schnell import MapCalculator, GroundDetector, LISADetector


# Detectors
det1 = GroundDetector('Hanford',     46.4, -119.4, 90-171.8,
                      'plots/data/aLIGO_design.txt')
det2 = GroundDetector('Livingstone', 30.7,  -90.8, 90-243.0,
                      'plots/data/aLIGO_design.txt')
detl = LISADetector(0, is_L5Gm=False)

# Calculators
mc11_n = MapCalculator([det1])
mc12_n = MapCalculator([det1, det2])
mcLL_n = MapCalculator([detl], f_pivot=1E-2)

# Angles
nside = 64
theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))


def test_antenna_new():
    a11 = np.real(mc11_n.get_antenna(0, 0, 0, 0, theta, phi,
                                     inc_baseline=False))
    a12 = np.real(mc12_n.get_antenna(0, 1, 0, 0, theta, phi,
                                     inc_baseline=False))
    aLL = np.abs(mcLL_n.get_antenna(0, 0, 0, 1E-2, theta, phi,
                                    inc_baseline=False))
    a11_test, a12_test, aLL_test = hp.read_map(
        "tests/test_data/antenna_test.fits",
        field=None)
    assert np.all(np.fabs(a11-a11_test) < 1E-5)
    assert np.all(np.fabs(a12-a12_test) < 1E-5)
    assert np.all(np.fabs(aLL-aLL_test) < 1E-5)


def test_Gell_new():
    # The factor 2 here corrects for a pevious missing factor
    # for auto-correlations.
    gl11 = mc11_n.get_G_ell(0, 100., nside) * 2 * 4
    gl12 = mc12_n.get_G_ell(0, 100., nside, no_autos=True) * 4
    glLL = mcLL_n.get_G_ell(0, 1E-2, nside) * 2 * 4
    ls, gl11_test, gl12_test, glLL_test = np.loadtxt(
        "tests/test_data/gls_test.txt",
        unpack=True)
    assert np.all(np.fabs(gl11/gl11_test-1)[::2] < 1E-8)
    assert np.all(np.fabs(gl12/gl12_test-1)[::2] < 1E-8)
    # We changed the noise model, so this doesn't agree anymore
    assert np.all(np.fabs(16*glLL/glLL_test-1)[::2] < 0.05)
