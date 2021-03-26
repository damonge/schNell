import numpy as np
from .detector import LISADetector, LISAlikeDetector
from .space_detector import LISADetector2, ALIADetector2, LISAandALIADetector


class NoiseCorrelationBase(object):
    """ Noise correlation objects have methods to compute
    noise PSD correlation matrices.

    Do not use the bare class.
    """
    def __init__(self, ndet):
        self.ndet

    def _get_corrmat(self, f):
        raise NotImplementedError("Don't use the NoiseCorrelationBase class")

    def get_corrmat(self, f):
        """ Return covariance matrix as a function
        of frequency.

        Args:
            f: array of `N_f` frequencies.

        Returns:
            array_like: array of shape `[N_f, N_d, N_d]`, \
                where `N_d` is the number of detectors in \
                the network, containing the correlation \
                matrix for each input frequency.
        """
        return self._get_corrmat(f)


class NoiseCorrelationConstant(NoiseCorrelationBase):
    """ This describes constant correlation matrices.

    Args:
        corrmat: 2D array providing the constant covariance
            matrix.
    """
    def __init__(self, corrmat):
        if not np.all(np.fabs(corrmat) <= 1):
            raise ValueError("The input correlation matrix "
                             "has elements larger than 1")
        if not np.ndim(corrmat) == 2:
            raise ValueError("Correlation matrices should be 2D")
        self.ndet = len(corrmat)
        self.mat = corrmat

    def _get_corrmat(self, f):
        f_use = np.atleast_1d(f)
        nf = len(f_use)
        return np.tile(self.mat, (nf, 1)).reshape([nf,
                                                   self.ndet,
                                                   self.ndet])


class NoiseCorrelationConstantIdentity(NoiseCorrelationConstant):
    """ This describes diagonal correlation matrices.

    Args:
        ndet: number of detectors in the network.
    """
    def __init__(self, ndet):
        self.ndet = ndet
        self.mat = np.eye(self.ndet)


class NoiseCorrelationConstantR(NoiseCorrelationConstant):
    """ This class implements correlation matrices that
    have the same cross-correlation coefficient for all
    pairs of different detector, which is also constant
    in frequency.

    Args:
        ndet: number of detectors in the network.
        r: pairwise correlation coefficient.
    """
    def __init__(self, ndet, r):
        self.ndet = ndet
        self.mat = ((1-r)*np.eye(self.ndet) +
                    np.full([self.ndet, self.ndet], r))


class NoiseCorrelationFromFunctions(NoiseCorrelationBase):
    """ This implements a correlation matrix that has
    the same auto-correlation PSD for all detectors and
    the same cross-correlation PSD for all pairs of
    different detectors.

    Args:
        ndet: number of detectors in the network.
        psd_auto: function of frequency returning the
            detector noise auto-correlation.
        psd_cross: function of frequency returning the
            detector noise cross-correlation.
    """
    def __init__(self, ndet, psd_auto, psd_cross):
        self.ndet = ndet
        self.psda = psd_auto
        self.psdx = psd_cross

    def _rho(self, f):
        a = self.psda(f)
        x = self.psdx(f)
        return x/a

    def _get_corrmat(self, f):
        f_use = np.atleast_1d(f)
        r = self._rho(f_use)
        mat = np.zeros([len(f_use), self.ndet, self.ndet])
        for i in range(self.ndet):
            mat[:, i, i] = 1
            for j in range(i+1, self.ndet):
                mat[:, i, j] = r
                mat[:, j, i] = r
        return mat


class NoiseCorrelationLISA(NoiseCorrelationFromFunctions):
    """ This implements the LISA noise correlation
    matrix.

    Args:
        det: :class:`~schnell.LISADetector` object.
    """
    def __init__(self, det):
        self.ndet = 3
        if not (isinstance(det, LISADetector) or isinstance(det, LISADetector2)):
            raise ValueError("`det` must be of type LISADetector")
        self.psda = det.psd_A
        self.psdx = det.psd_X


class NoiseCorrelationLISAlike(NoiseCorrelationFromFunctions):
    """ This implements the noise correlation
    matrix for LISA-like detectors.

    Args:
        det: :class:`~schnell.LISAlikeDetector` object.
    """
    def __init__(self, det):
        self.ndet = 3
        if not (isinstance(det, LISAlikeDetector)):
            raise ValueError("`det` must be of type LISAlikeDetector")
        self.psda = det.psd_A
        self.psdx = det.psd_X

class NoiseCorrelationLISALIA(NoiseCorrelationBase):
    """ This implements the correlation matrix for LISA and ALIA combined.

    Args:
        det: :class:`~schnell.LISAandALIADetector` object.
    """
    def __init__(self, det):
        self.ndet = 6
        LISAdet = LISADetector2(0, is_L5Gm=det.is_L5Gm,
                                static=det.detector.static,
                                include_GCN=det.detector.include_GCN,
                                mission_duration=det.detector.mission_duration)
        ALIAdet = ALIADetector2(0, static=det.detector.static,
                                include_GCN=det.detector.include_GCN,
                                mission_duration=det.detector.mission_duration)
        self.psdaL = LISAdet.psd_A
        self.psdxL = LISAdet.psd_X
        self.psdaA = ALIAdet.psd_A
        self.psdxA = ALIAdet.psd_X

    def _rhoL(self, f):
        a = self.psdaL(f)
        x = self.psdxL(f)
        return x/a

    def _rhoA(self, f):
        a = self.psdaL(f)
        x = self.psdxL(f)
        return x/a

    def _get_corrmat(self, f):
        f_use = np.atleast_1d(f)
        rL = self._rhoL(f_use)
        rA = self._rhoA(f_use)
        mat = np.zeros([len(f_use), 6, 6])
        for i in range(3):
            mat[:, i, i] = 1
            for j in range(i+1, 3):
                mat[:, i, j] = rL
                mat[:, j, i] = rL
        for i in range(3, 6):
            mat[:, i, i] = 1
            for j in range(i+1, 6):
                mat[:, i, j] = rA
                mat[:, j, i] = rA
        return mat