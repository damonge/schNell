import numpy as np
import scipy.integrate as integr
from .detector import Detector, LISAlikeDetector

class LISADetector2(LISAlikeDetector):
    """ :class:`LISADetector` objects can be used to describe
    the properties of the LISA network.

    Args:
        detector_id: detector number (0, 1 or 2).
        is_L5Gm (bool): whether the arm length should be
            5E9 meters (otherwise 2.5E9 meters will be assumed)
            (default `False`).
        static (bool): if `True`, a static configuration corresponding
            to a perfect equilateral triangle in the x-y plane will
            be assumed (default False).
        include_GCN (bool): if `True`, include galactic confusion
            noise in PSD computation (default False).
        mission_duration (float): mission duration in years (default 4.).
    """
    def __init__(self, detector_id, is_L5Gm=False,
                 static=False, include_GCN=False,
                 mission_duration=4.):
        self.i_d = detector_id % 3
        self.name = 'LISA_%d' % self.i_d
        self.get_transfer = self._get_transfer_LISA
        if is_L5Gm:  # 5 Gm arm length
            self.L = 5E9
            self.e = 0.00965
        else:  # 2.5 Gm arm length
            self.L = 2.5E9
            self.e = 0.00482419
        self.static = static
        self.include_GCN = include_GCN
        self.mission_duration = mission_duration
        self.path_fluctuation = 1.5E-11
        self.acc_noise = 3E-15

    def _get_transfer_LISA(self, u, f, nv):        
        # Eq. 48 in astro-ph/0105374
        # xf = f/(2*fstar)/pi, fstar = c/(2*pi*L)
        xf = self.L * f / self.clight

        # u,nv is [nt, npix]
        u_dot_n = np.sum(u[:, :, None] * nv[:, None, :],
                         axis=0)

        # For some reason Numpy's sinc(x) is sin(pi*x) / (pi*x)
        sinc1 = np.sinc(xf[None, :, None]*(1-u_dot_n[:, None, :]))
        sinc2 = np.sinc(xf[None, :, None]*(1+u_dot_n[:, None, :]))
        phase1 = -np.pi*xf[None, :, None]*(3+u_dot_n[:, None, :])
        phase2 = -np.pi*xf[None, :, None]*(1+u_dot_n[:, None, :])
        tr = 0.5*(np.exp(1j*phase1)*sinc1 +
                  np.exp(1j*phase2)*sinc2)
        return tr


class ALIADetector2(LISAlikeDetector):
    """ :class:`ALIADetector` objects can be used to describe
    the properties of the ALIA network.

    Args:
        detector_id: detector number (0, 1 or 2).
        static (bool): if `True`, a static configuration corresponding
            to a perfect equilateral triangle in the x-y plane will
            be assumed (default False).
        include_GCN (bool): if `True`, include galactic confusion
            noise in PSD computation (default False).
        mission_duration (float): mission duration in years (default 4.).
    """
    def __init__(self, detector_id,
                 static=False, include_GCN=False,
                 mission_duration=4.):
        self.i_d = detector_id % 3
        self.name = 'ALIA_%d' % self.i_d
        self.get_transfer = self._get_transfer_ALIA
        self.L = 5E8
        self.e = 0.000965
        self.static = static
        self.include_GCN = include_GCN
        self.mission_duration = mission_duration
        self.path_fluctuation = 1.5E-14
        self.acc_noise = 3E-16

    def _get_transfer_ALIA(self, u, f, nv):
        # Eq. 48 in astro-ph/0105374
        # xf = f/(2*fstar)/pi, fstar = c/(2*pi*L)
        xf = self.L * f / self.clight

        # u,nv is [nt, npix]
        u_dot_n = np.sum(u[:, :, None] * nv[:, None, :],
                         axis=0)

        # For some reason Numpy's sinc(x) is sin(pi*x) / (pi*x)
        sinc1 = np.sinc(xf[None, :, None]*(1-u_dot_n[:, None, :]))
        sinc2 = np.sinc(xf[None, :, None]*(1+u_dot_n[:, None, :]))
        phase1 = -np.pi*xf[None, :, None]*(3+u_dot_n[:, None, :])
        phase2 = -np.pi*xf[None, :, None]*(1+u_dot_n[:, None, :])
        tr = 0.5*(np.exp(1j*phase1)*sinc1 +
                  np.exp(1j*phase2)*sinc2)
        return tr

class LISAandALIADetector(Detector):
    """ :class:`LISAandALIADetector objects can be used to describe 
    the combination of a LISA and an ALIA network. id 0 to 2 is LISA;
    id 3 to 5 is ALIA.
    """
    trans_freq_earth = 2 * np.pi / (365 * 24 * 3600)
    R_AU = 1.496E11
    kap = 0  # initial longitude
    lam = 0  # initial orientation
    clight = 299792458.

    def __init__(self, detector_id, is_L5Gm=False,
                 static=False, include_GCN=False,
                 mission_duration=4.):
        if detector_id in [0, 1, 2]:
            self.detector = LISADetector2(detector_id, is_L5Gm, static,
                                          include_GCN, mission_duration)
            self.i_d = detector_id
            self.name = 'LISA_%d' % self.i_d
            self.get_transfer = self.detector._get_transfer_LISA
        elif detector_id in [3, 4, 5]:
            self.detector = ALIADetector2(detector_id-3, static, include_GCN, mission_duration)
            self.name = 'ALIA_%d' % self.i_d
            self.get_transfer = self.detector._get_transfer_ALIA
        else:
            raise RuntimeError('Detector i_d must be between 0 and 5')
        self.i_d = detector_id
        self.L = self.detector.L
        self.e = self.detector.e
    
    def psd_A(self, f):
        return self.detector.psd_A(f)
    
    def psd_X(self, f):
        return self.detector.psd_X(f)
    
    def GCN(self, f):
        return self.detector.GCN(f)
    
    def psd(self, f):
        return self.detector.psd(f)
    
    def response(self, f):
        return self.detector.response(f)
    
    def sensitivity(self, f):
        return self.detector.sensitivity(f)
    
    def _pos_all(self, t):
        return
    
    def _pos_single(self, t, n):
        if n in [0, 1, 2]:
            return LISADetector2(0)._pos_single(t, n)
        elif n in [3, 4, 5]:
            return ALIADetector2(0)._pos_single(t, n)
        else:
            raise RuntimeError('Detector id must be between 0 and 5')
    
    def get_u_v(self, t):
        return self.detector.get_u_v(t)