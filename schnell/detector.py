import numpy as np


class Detector(object):
    """ :class:`Detector` objects encode information about individual
    GW detectors. The most relevant quantities are:

    * Detector position.

    * Detector transfer function.

    * Unit vectors in arm directions.

    * Detector response tensor.

    * Noise PSDs.

    Baseline :class:`Detector` objects serve only as a superclass for
    all other detector types. Do not use them.
    """
    def __init__(self, name):
        self.name = name
        raise NotImplementedError("Don't use the bare class")

    def get_position(self, t):
        """ Returns a 2D array containing the 3D position of
        the detector at a series of times. The output array
        has shape [3, N_t], where N_t is the size of `t`.

        Args:
            t: time of observation (in seconds).

        Returns:
            array_like: detector position (in m) as a function \
                of time.
        """
        raise NotImplementedError("get_position not implemented")

    def _get_uu_vv_from_u_v(self, u, v):
        # [3, 3, nt]
        uu = u[:, None, ...]*u[None, :, ...]
        vv = v[:, None, ...]*v[None, :, ...]
        return uu, vv

    def get_transfer(self, u, f, nv):
        """ Returns the detector transfer function as a
        funciton of position, frequency and sky coordinates.

        Args:
            u: 2D array of size `[3, N_t]` containing the unit
                vector pointing along one of the detector arms
                at `N_t` different time intervals.
            f: 1D array of frequencies (in Hz).
            nv: 2D array of shape `[3, N_pix]` containining the
                normalized coordinates of `N_pix` points in the
                celestial sphere.

        Returns:
            array_like: array of shape `[N_t, N_f, N_pix]` \
                containing the transfer function as a function \
                of time, frequency and sky position.
        """
        # u is [3, nt]
        # f is [nf]
        # nv is [3, npix]
        # output is [nt, nf, npix]
        tr = np.ones([len(u[0]), len(f), len(nv[0])]) + 0j
        return tr

    def get_uu_vv(self, t):
        """ Returns the outer product of the detector arm
        unit vectors as a function of time.

        Args:
            t: array of size `N_t` containing different
                observing times (in s).

        Returns:
            array_like: 2 arrays of shape `[3, 3, N_t]` \
                containing the outer products of the unit \
                vectors pointing in the directions of the \
                two detector arms.
        """
        u, v = self.get_u_v(t)
        return self._get_uu_vv_from_u_v(u, v)

    def get_Fp(self, t, f, e_p, e_x, nv):
        """ Compute the quantity:

        .. math::
            F^p(f,\\hat{n}) = a^{ij}e^p_{ij}\\exp(i2\\pi f\\hat{n} {\\bf x})

        (i.e. Eq. 12 of the companion paper).

        Args:
            t: array of size `N_t` containing observing
                times (in s).
            f: array of size `N_f` containing frequencies
                (in Hz).
            ep: array of shape `[3, 3, N_pix]` containing the
                "+" polarization tensor at `N_pix` different
                sky positions.
            ex: same as `ep` for the "x" polarization tensor.
            nv: array of shape `[3, N_pix]` containing the
                unit vector pointing in the direction of
                `N_pix` sky positions.

        Returns:
            array_like: 2 arrays of shape `[N_t, N_f, N_pix]` \
                containing :math:`F^+` and :math:`F^\\times` \
                as a function of time, frequency and sky position.
        """
        # e_p/e_x is [3, 3, npix]

        # u/v are [3, nt]
        u, v = self.get_u_v(t)

        # uu/vv/a are [3, 3, nt]
        uu, vv = self._get_uu_vv_from_u_v(u, v)

        # Transfer function
        # [nt, nf, npix]
        tf_u = self.get_transfer(u, f, nv)
        tf_v = self.get_transfer(v, f, nv)

        # Tr[uu * e_p] etc.
        # [nt, npix]
        tr_uu_p = np.sum(uu[:, :, :, None] * e_p[:, :, None, :],
                         axis=(0, 1))
        tr_vv_p = np.sum(vv[:, :, :, None] * e_p[:, :, None, :],
                         axis=(0, 1))
        tr_uu_x = np.sum(uu[:, :, :, None] * e_x[:, :, None, :],
                         axis=(0, 1))
        tr_vv_x = np.sum(vv[:, :, :, None] * e_x[:, :, None, :],
                         axis=(0, 1))
        # Output is [nt, nf, npix]
        Fp = 0.5*(tr_uu_p[:, None, :]*tf_u -
                  tr_vv_p[:, None, :]*tf_v)
        Fx = 0.5*(tr_uu_x[:, None, :]*tf_u -
                  tr_vv_x[:, None, :]*tf_v)
        return Fp, Fx

    def _read_psd(self, fname):
        from scipy.interpolate import interp1d
        nu, fnu = np.loadtxt(fname, unpack=True)
        self.lpsdf = interp1d(np.log(nu), np.log(fnu),
                              bounds_error=False,
                              fill_value=15)

    def psd(self, f):
        """ Returns noise PSD as a function of frequency.

        Args:
            f: array of frequencies (in Hz).

        Returns:
            array_like: array of PSD values in units of \
                1/Hz.
        """
        return np.exp(2*self.lpsdf(np.log(f)))


class GroundDetectorTriangle(Detector):
    """ :class:`GroundDetectorTriangle` objects represent detectors
    in a triangular configureation located at fixed position on
    Earth (e.g. the Einstein Telescope).

    Args:
        name: detector name.
        lat: latitude of the triangle's barycenter in degrees.
        lon: longitude in degrees.
        fname_psd: path to file containing the detector's noise
            curve. The file should contain two columns,
            corresponding to the frequency (in Hz) and the
            corresponding value of the strain-level noise
            (in units of Hz:math:`^{-1/2}`).
        detector_id: detector number (0, 1 or 2).
        beta0_deg: orientation angle, defined as the angle between
            the vertex with id 0 and the local meridian.
        arm_length_km: arm length in km.
    """
    rot_freq_earth = 2*np.pi/(24*3600)
    earth_radius = 6.371E6  # in meters

    def __init__(self, name, lat, lon, fname_psd, detector_id,
                 beta0_deg=0, arm_length_km=10.):
        self.name = name
        self.i_d = detector_id
        self.phi_e = np.radians(lon)
        self.theta_e = np.radians(90-lat)
        self.ct = np.cos(self.theta_e)
        self.st = np.sin(self.theta_e)
        self.L = arm_length_km*1000
        self.alpha = self.L/(np.sqrt(3.) * self.earth_radius)
        self.betas = np.radians(beta0_deg) + 2 * np.arange(3) * np.pi/3.
        self.ca = np.cos(self.alpha)
        self.sa = np.sin(self.alpha)
        self.cbs = np.cos(self.betas)
        self.sbs = np.sin(self.betas)
        self._read_psd(fname_psd)

    def _pos_single(self, t, n):
        # Returns [3, nt]
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        cb = self.cbs[n]
        sb = self.sbs[n]
        pos = np.array([self.ct * self.sa * cp * cb -
                        self.sa * sp * sb +
                        self.st * self.ca * cp,
                        self.ct * self.sa * sp * cb +
                        self.sa * cp * sb +
                        self.st * self.ca * sp,
                        o*(-self.st * self.sa * cb +
                           self.ct * self.ca)])
        pos *= self.earth_radius
        return pos

    def _pos_all(self, t):
        return np.array([self._pos_single(t, i)
                         for i in range(3)])

    def get_position(self, t):
        """ Returns a 2D array containing the 3D position of
        the detector at a series of times. The output array
        has shape [3, N_t], where N_t is the size of `t`.

        .. note:: We assume the Earth is a sphere of radius
                  6371 km that performs a full rotation every
                  24 h exactly.

        Args:
            t: time of observation (in seconds).

        Returns:
            array_like: detector position (in m) as a function \
                of time.
        """
        return self._pos_single(t, self.i_d)

    def get_u_v(self, t):
        """ Returns unit vectors in the directions of the
        detector arms as a function of time.

        Args:
            t: array of size `N_t` containing different
                observing times (in s).

        Returns:
            array_like: 2 arrays of shape `[3, N_t]` \
                containing the arm unit vectors.
        """
        t_use = np.atleast_1d(t)
        pos = self._pos_all(t_use)
        np0 = (self.i_d + 0) % 3
        np1 = (self.i_d + 1) % 3
        np2 = (self.i_d + 2) % 3
        uv = pos[np1] - pos[np0]
        ul = np.sqrt(np.sum(uv**2, axis=0))
        u = uv[:, :] / ul[None, :]
        vv = pos[np2] - pos[np0]
        vl = np.sqrt(np.sum(vv**2, axis=0))
        v = vv[:, :] / vl[None, :]
        return u, v


class GroundDetector(Detector):
    """ :class:`GroundDetector` objects represent detectors
    located at fixed position on Earth.

    Args:
        name: detector name.
        lat: latitude in degrees.
        lon: longitude in degrees.
        alpha: orientation angle, defined as the angle between
            the vertex bisector and the local parallel.
            In degrees.
        fname_psd: path to file containing the detector's noise
            curve. The file should contain two columns,
            corresponding to the frequency (in Hz) and the
            corresponding value of the strain-level noise
            (in units of Hz:math:`^{-1/2}`).
        aperture: arm aperture angle (in degrees).
    """
    rot_freq_earth = 2*np.pi/(24*3600)
    earth_radius = 6.371E6  # in meters

    def __init__(self, name, lat, lon, alpha, fname_psd, aperture=90):
        self.name = name
        self.alpha = np.radians(90-alpha)
        self.beta = np.radians(aperture / 2)
        self.phi_e = np.radians(lon)
        self.theta_e = np.radians(90-lat)
        self.ct = np.cos(self.theta_e)
        self.st = np.sin(self.theta_e)
        self.cabp = np.cos(self.alpha+self.beta)
        self.cabm = np.cos(self.alpha-self.beta)
        self.sabp = np.sin(self.alpha+self.beta)
        self.sabm = np.sin(self.alpha-self.beta)
        self._read_psd(fname_psd)

    def get_position(self, t):
        """ Returns a 2D array containing the 3D position of
        the detector at a series of times. The output array
        has shape [3, N_t], where N_t is the size of `t`.

        .. note:: We assume the Earth is a sphere of radius
                  6371 km that performs a full rotation every
                  24 h exactly.

        Args:
            t: time of observation (in seconds).

        Returns:
            array_like: detector position (in m) as a function \
                of time.
        """
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        return self.earth_radius * np.array([self.st*cp,
                                             self.st*sp,
                                             self.ct*o])

    def get_u_v(self, t):
        """ Returns unit vectors in the directions of the
        detector arms as a function of time.

        Args:
            t: array of size `N_t` containing different
                observing times (in s).

        Returns:
            array_like: 2 arrays of shape `[3, N_t]` \
                containing the arm unit vectors.
        """
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        # [3, nt]
        u = np.array([-self.cabm*cp*self.ct-self.sabm*sp,
                      self.sabm*cp-self.cabm*sp*self.ct,
                      self.cabm*self.st*o])

        # [3, nt]
        v = np.array([-self.cabp*cp*self.ct-self.sabp*sp,
                      self.sabp*cp-self.cabp*sp*self.ct,
                      self.cabp*self.st*o])
        return u, v


class LISADetector(Detector):
    """ :class:`LISADetector` objects can be used to describe
    the properties of the LISA network.

    Args:
        detector_id: detector number (0, 1 or 2).
        is_L5Gm (bool): whether the arm length should be
            5E9 meters (otherwise 2.5E9 meters will be assumed)
            (default `False`).
        static (bool): if `True`, a static configuration corresponding
            to a perfect equilateral triangle in the x-y plane will
            be assumed.
    """
    trans_freq_earth = 2 * np.pi / (365 * 24 * 3600)
    R_AU = 1.496E11
    kap = 0  # initial longitude
    lam = 0  # initial orientation
    clight = 299792458.

    def __init__(self, detector_id, is_L5Gm=False,
                 static=False):
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

    def psd_A(self, f):
        """ Returns auto-noise PSD as a function of frequency.
        Uses Eq. 55 from arXiv:1908.00546.

        Args:
            f: array of frequencies (in Hz).

        Returns:
            array_like: array of PSD values in units of \
                1/Hz.
        """
        # Equation 55 from 1908.00546
        fstar = self.clight / (2 * np.pi * self.L)
        Poms = (1.5E-11)**2
        Pacc = (3E-15)**2 * (1 + (4E-4/f)**2)/(2*np.pi*f)**4
        Pn = (4*Poms+8*(1+np.cos(f/fstar)**2)*Pacc) / self.L**2

        return Pn

    def psd_X(self, f):
        """ Returns cross-noise PSD as a function of frequency.
        Uses Eq. 56 from arXiv:1908.00546.

        Args:
            f: array of frequencies (in Hz).

        Returns:
            array_like: array of PSD values in units of \
                1/Hz.
        """
        # Equation 56 from 1908.00546
        fstar = self.clight / (2 * np.pi * self.L)
        Poms = (1.5E-11)**2
        Pacc = (3E-15)**2 * (1 + (4E-4/f)**2)/(2*np.pi*f)**4
        Pn = - (2*Poms+8*Pacc) * np.cos(f/fstar) / self.L**2

        return Pn

    def psd(self, f):
        """ Returns noise PSD as a function of frequency.
        Uses Eq. 55 from arXiv:1908.00546.

        Args:
            f: array of frequencies (in Hz).

        Returns:
            array_like: array of PSD values in units of \
                1/Hz.
        """
        return self.psd_A(f)

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
                         for i in range(3)])

    def _pos_single(self, t, n):
        if self.static:
            if np.ndim(t) == 0:
                ll = self.L
                z = 0
            else:
                ll = self.L * np.ones_like(t)
                z = np.zeros_like(t)
            if n % 3 == 0:
                return np.array([z, z, z])
            elif n % 3 == 1:
                return np.array([ll*1/2,
                                 ll*np.sqrt(3)/2,
                                 z])
            elif n % 3 == 2:
                return np.array([-ll*1/2,
                                 ll*np.sqrt(3)/2,
                                 z])
        else:
            # Equation 1 from gr-qc/0311069
            a = self.trans_freq_earth * t + self.kap
            b = 2 * np.pi * n / 3. + self.lam
            e = self.e
            e2 = e*e
            x = np.cos(a) + \
                0.5 * e * (np.cos(2*a-b) - 3*np.cos(b)) + \
                0.125 * e2 * (3*np.cos(3*a-2*b) -
                              10*np.cos(a) -
                              5*np.cos(a-2*b))
            y = np.sin(a) + \
                0.5 * e * (np.sin(2*a-b) - 3*np.sin(b)) + \
                0.125 * e2 * (3*np.sin(3*a-2*b) -
                              10*np.sin(a) +
                              5*np.sin(a-2*b))
            z = np.sqrt(3.) * (-e * np.cos(a-b) +
                               e2 * (1 + 2*np.sin(a-b)**2))
            return self.R_AU * np.array([x, y, z])

    def get_u_v(self, t):
        """ Returns unit vectors in the directions of the
        detector arms as a function of time.

        Args:
            t: array of size `N_t` containing different
                observing times (in s).

        Returns:
            array_like: 2 arrays of shape `[3, N_t]` \
                containing the arm unit vectors.
        """
        t_use = np.atleast_1d(t)
        pos = self._pos_all(t_use)
        np0 = (self.i_d + 0) % 3
        np1 = (self.i_d + 1) % 3
        np2 = (self.i_d + 2) % 3
        uv = pos[np1] - pos[np0]
        ul = np.sqrt(np.sum(uv**2, axis=0))
        u = uv[:, :] / ul[None, :]
        vv = pos[np2] - pos[np0]
        vl = np.sqrt(np.sum(vv**2, axis=0))
        v = vv[:, :] / vl[None, :]

        return u, v
