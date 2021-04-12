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
    kap = 0  # initial longitude of LISA ; ALIA is behind
    lam = 0  # initial orientation
    clight = 299792458.

    def __init__(self, detector_id, is_L5Gm=False,
                 static=False, include_GCN=False,
                 mission_duration=4., separation=0.7):
        self.separation = separation # distance between LISA and ALIA, in AU
        # We suppose that the Earth's motion is circular (?)
        self.ang_separation = 2 * np.arcsin(self.separation / 2)

        if detector_id in [0, 1, 2]:
            self.detector = LISADetector2(detector_id, is_L5Gm, static,
                                          include_GCN, mission_duration)
            self.name = 'LISA_%d' % detector_id
            self.get_transfer = self.detector._get_transfer_LISA
            self.detector.kap = self.kap
        elif detector_id in [3, 4, 5]:
            self.detector = ALIADetector2(detector_id-3, static, include_GCN, mission_duration)
            self.name = 'ALIA_%d' % detector_id
            self.get_transfer = self.detector._get_transfer_ALIA
            self.detector.kap = self.kap - self.ang_separation
        else:
            raise RuntimeError('Detector i_d must be between 0 and 5')
        self.is_L5Gm = is_L5Gm
        self.detector.lam = self.lam
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
    
    def get_position(self, t):
        """ Returns a 2D array containing the 3D position of
        the detector at a series of times. The output array
        has shape [3, N_t], where N_t is the size of `t`.

        .. note:: The spacecraft orbits are calculated using Eq. 1
                  of gr-qc/0311069.

        Args:
            t: time of observation (in seconds).

        Returns:
            array_like: detector position (in m) as a function \
                of time.
        """
        return self._pos_single(t, self.i_d)
    
    def _pos_all(self, t):
        return np.array([self._pos_single(t, i)
                         for i in range(6)])
    
    def _pos_single(self, t, n):
        if n in [0, 1, 2]:
            detect = LISADetector2(0, is_L5Gm=self.is_L5Gm,
                                     static=self.detector.static,
                                     include_GCN=self.detector.include_GCN,
                                     mission_duration=self.detector.mission_duration)
            detect.kap = self.kap
            detect.lam = self.lam
        elif n in [3, 4, 5]:
            detect = ALIADetector2(0, static=self.detector.static,
                                   include_GCN=self.detector.include_GCN,
                                   mission_duration=self.detector.mission_duration)
            detect.kap = self.kap - self.ang_separation
            detect.lam = self.lam
        else:
            raise RuntimeError('Detector id must be between 0 and 5')
        return detect._pos_single(t, n)

    def get_u_v(self, t):
        return self.detector.get_u_v(t)


class TwoLISADetector(Detector):
    """ :class:`TwoLISADetector objects can be used to describe 
    the combination of two LISA networks. id 0 to 2 is LISA 1;
    id 3 to 5 is LISA 2.
    """
    trans_freq_earth = 2 * np.pi / (365 * 24 * 3600)
    R_AU = 1.496E11
    kap = 0  # initial longitude of LISA1 ; LISA2 is behind
    lam = 0  # initial orientation
    clight = 299792458.

    def __init__(self, detector_id, is_L5Gm=False,
                 static=False, include_GCN=False,
                 mission_duration=4., separation=0.7):
        self.separation = separation # distance between the LISAs, in AU
        # We suppose that the Earth's motion is circular (?)
        self.ang_separation = 2 * np.arcsin(self.separation / 2)
        self.detector = LISADetector2(detector_id, is_L5Gm, static,
                                    include_GCN, mission_duration)
        self.name = 'LISA_%d' % detector_id
        self.get_transfer = self.detector._get_transfer_LISA

        if detector_id in [0, 1, 2]:
            self.detector.kap = self.kap
        elif detector_id in [3, 4, 5]:
            self.detector.kap = self.kap - self.ang_separation
        else:
            raise RuntimeError('Detector i_d must be between 0 and 5')

        self.is_L5Gm = is_L5Gm
        self.detector.lam = self.lam
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
    
    def get_position(self, t):
        """ Returns a 2D array containing the 3D position of
        the detector at a series of times. The output array
        has shape [3, N_t], where N_t is the size of `t`.

        .. note:: The spacecraft orbits are calculated using Eq. 1
                  of gr-qc/0311069.

        Args:
            t: time of observation (in seconds).

        Returns:
            array_like: detector position (in m) as a function \
                of time.
        """
        return self._pos_single(t, self.i_d)
    
    def _pos_all(self, t):
        return np.array([self._pos_single(t, i)
                         for i in range(6)])
    
    def _pos_single(self, t, n):
        detect = LISADetector2(0, is_L5Gm=self.is_L5Gm,
                                    static=self.detector.static,
                                    include_GCN=self.detector.include_GCN,
                                    mission_duration=self.detector.mission_duration)
        if n in [0, 1, 2]:
            detect.kap = self.kap
        elif n in [3, 4, 5]:
            detect.kap = self.kap - self.ang_separation
        else:
            raise RuntimeError('Detector id must be between 0 and 5')
        detect.lam = self.lam
        return detect._pos_single(t, n)

    def get_u_v(self, t):
        return self.detector.get_u_v(t)