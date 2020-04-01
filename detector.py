import numpy as np


class Detector(object):
    def __init__(self, name):
        self.name = name
        raise NotImplementedError("Don't use the bare class")

    def get_position(self, t):
        raise NotImplementedError("get_position not implemented")

    def _get_xx_yy_from_x_y(self, x, y):
        # [3, 3, nt]
        xx = x[:, None, ...]*x[None, :, ...]
        yy = y[:, None, ...]*y[None, :, ...]
        return xx, yy

    def get_transfer(self, x, f, nv):
        # x is [3, nt]
        # f is [nf]
        # nv is [3, npix]
        # output is [nt, nf, npix]
        tr = np.ones([len(x[0]), len(f), len(nv[0])]) + 0j
        return tr

    def get_xx_yy(self, t):
        x, y = self.get_x_y(t)
        return self._get_xx_yy_from_x_y(x, y)

    def get_Fp(self, t, f, e_p, e_x, nv):
        # e_p/e_x is [3, 3, npix]

        # x/y are [3, nt]
        x, y = self.get_x_y(t)

        # xx/yy/a are [3, 3, nt]
        xx, yy = self._get_xx_yy_from_x_y(x, y)

        # Transfer function
        # [nt, nf, npix]
        tf_x = self.get_transfer(x, f, nv)
        tf_y = self.get_transfer(y, f, nv)

        # Tr[xx * e_p] etc.
        # [nt, npix]
        tr_xx_p = np.sum(xx[:, :, :, None] * e_p[:, :, None, :],
                         axis=(0, 1))
        tr_yy_p = np.sum(yy[:, :, :, None] * e_p[:, :, None, :],
                         axis=(0, 1))
        tr_xx_x = np.sum(xx[:, :, :, None] * e_x[:, :, None, :],
                         axis=(0, 1))
        tr_yy_x = np.sum(yy[:, :, :, None] * e_x[:, :, None, :],
                         axis=(0, 1))
        # Output is [nt, nf, npix]
        Fp = 0.5*(tr_xx_p[:, None, :]*tf_x -
                  tr_yy_p[:, None, :]*tf_y)
        Fx = 0.5*(tr_xx_x[:, None, :]*tf_x -
                  tr_yy_x[:, None, :]*tf_y)
        return Fp, Fx

    def read_psd(self, fname):
        from scipy.interpolate import interp1d
        nu, fnu = np.loadtxt(fname, unpack=True)
        self.lpsdf = interp1d(np.log(nu), np.log(fnu),
                              bounds_error=False,
                              fill_value=15)

    def psd(self, nu):
        return np.exp(2*self.lpsdf(np.log(nu)))


class GroundDetector(Detector):
    rot_freq_earth = 2*np.pi/(24*3600)
    earth_radius = 6.371E6  # in meters

    def __init__(self, name, lat, lon, alpha, fname_psd, aperture=90):
        # Translate between Renzini's alpha and mine
        self.name = name
        self.alpha = np.radians(alpha)
        self.beta = np.radians(aperture / 2)
        self.phi_e = np.radians(lon)
        self.theta_e = np.radians(90-lat)
        self.ct = np.cos(self.theta_e)
        self.st = np.sin(self.theta_e)
        self.cabp = np.cos(self.alpha+self.beta)
        self.cabm = np.cos(self.alpha-self.beta)
        self.sabp = np.sin(self.alpha+self.beta)
        self.sabm = np.sin(self.alpha-self.beta)
        self.read_psd(fname_psd)

    def get_position(self, t):
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        return self.earth_radius * np.array([self.st*cp,
                                             self.st*sp,
                                             self.ct*o])

    def get_x_y(self, t):
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        # [3, nt]
        x = np.array([-self.cabm*cp*self.ct-self.sabm*sp,
                      self.sabm*cp-self.cabm*sp*self.ct,
                      self.cabm*self.st*o])

        # [3, nt]
        y = np.array([-self.cabp*cp*self.ct-self.sabp*sp,
                      self.sabp*cp-self.cabp*sp*self.ct,
                      self.cabp*self.st*o])
        return x, y


class LISADetector(Detector):
    trans_freq_earth = 2 * np.pi / (365 * 24 * 3600)
    R_AU = 1.496E11
    kap = 0  # initial longitude
    lam = 0  # initial orientation
    clight = 299792458.

    def __init__(self, detector_id, is_L5Gm=False, map_transfer=False):
        self.i_d = detector_id % 3
        self.name = 'LISA_%d' % self.i_d
        self.map_transfer = map_transfer
        if self.map_transfer:
            self.get_transfer = self._get_transfer_LISA
        if is_L5Gm:  # 5 Gm arm length
            self.L = 5E9
            self.e = 0.00965
        else:  # 2.5 Gm arm length
            self.L = 2.5E9
            self.e = 0.00482419

    def _get_transfer_LISA(self, x, f, nv):
        def sinc(x):
            x_np = x / np.pi
            return np.sinc(x_np)

        # Eq. 48 in astro-ph/0105374
        # xf = f/(2*fstar)/pi, fstar = c/(2*pi*L)
        xf = self.L * f / self.clight

        # x.nv is [nt, npix]
        x_dot_n = np.sum(x[:, :, None] * nv[:, None, :],
                         axis=0)
        # For some reason Numpy's sinc is sin(pi*x) / (pi*x)
        sinc1 = np.sinc(xf[None, :, None]*(1-x_dot_n[:, None, :]))
        sinc2 = np.sinc(xf[None, :, None]*(1+x_dot_n[:, None, :]))
        phase1 = -(np.pi*xf)[None, :, None]*(3+x_dot_n[:, None, :])
        phase2 = -(np.pi*xf)[None, :, None]*(1+x_dot_n[:, None, :])
        tr = 0.5*(np.exp(1j*phase1)*sinc1 +
                  np.exp(1j*phase2)*sinc2)
        return tr

    def psd(self, nu):
        # Equation 1 from 1803.01944 (without background)
        # TODO: do these curves assume that you already have
        # 2 independent detectors?
        fstar = self.clight / (2 * np.pi * self.L)
        Poms = (1.5E-11)**2*(1+(2E-3/nu)**4)
        Pacc = (3E-15)**2 * (1 + (4E-4/nu)**2)*(1+(nu/8E-3)**4)
        Pn = (Poms+2*(1+np.cos(nu/fstar)**2)*Pacc/(2*np.pi*nu)**4) / self.L**2

        if self.map_transfer:
            return Pn
        else:
            # TODO: check prefactor. I think it already accounts
            # for the 60deg aperture.
            Rinv = 10 * (1 + 0.6 * (nu / fstar)**2) / 3
            return Pn * Rinv

    def get_position(self, t):
        return self.pos_single(t, self.i_d)

    def pos_all(self, t):
        return np.array([self.pos_single(t, i)
                         for i in range(3)])

    def pos_single(self, t, n):
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

    def get_x_y(self, t):
        t_use = np.atleast_1d(t)
        pos = self.pos_all(t_use)
        np0 = (self.i_d + 0) % 3
        np1 = (self.i_d + 1) % 3
        np2 = (self.i_d + 2) % 3
        xv = pos[np1] - pos[np0]
        xl = np.sqrt(np.sum(xv**2, axis=0))
        x = xv[:, :] / xl[None, :]
        yv = pos[np2] - pos[np0]
        yl = np.sqrt(np.sum(yv**2, axis=0))
        y = yv[:, :] / yl[None, :]

        return x, y