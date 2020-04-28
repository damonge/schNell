import healpy as hp
import numpy as np
from .correlation import (
    NoiseCorrelationBase,
    NoiseCorrelationConstant,
    NoiseCorrelationConstantIdentity,
    NoiseCorrelationConstantR)


class MapCalculator(object):
    """ Map calculators compute map-level quantities for a given
    network of detectors.

    Args:
        det_array: list of `Detector` objects.
        f_pivot: pivot frequency in Hz (default: 63 Hz)
        spectral_index: power-law spectral index. This should correspond
            to the index in units of Omega_GW, not intensity.
        corr_matrix: noise correlation matrix for the array. If `None`
            the identity is assumed. If a constant, this will be assumed
            to be the correlation coefficient between pairs of different
            detectors. If a 2D array, it will be the frequency-independent
            correlation matrix. Otherwise, pass a
            :class:`~schnell.NoiseCorrelationBase` object.
        h: value of the Hubble constant in units of 100 km/s/Mpc
            (default: 0.67).
    """
    clight = 299792458.

    def __init__(self, det_array, f_pivot=63., spectral_index=2./3.,
                 corr_matrix=None, h=0.67):
        self.dets = det_array
        self.ndet = len(det_array)
        self.f_pivot = f_pivot
        self.rcond = 1E-10
        self.specin_omega = spectral_index - 3
        self.h = h
        if not isinstance(corr_matrix, NoiseCorrelationBase):
            if corr_matrix is None:
                self.rho = NoiseCorrelationConstantIdentity(self.ndet)
            elif np.ndim(corr_matrix) == 0:
                self.rho = NoiseCorrelationConstantR(self.ndet, corr_matrix)
            elif np.ndim(corr_matrix) == 2:
                self.rho = NoiseCorrelationConstant(corr_matrix)
            else:
                raise ValueError("`corr_matrix` must be `None`, a number, "
                                 " a 2D array or a `NoiseCorrelationBase` "
                                 " object.")
        else:
            self.rho = corr_matrix

        if self.rho.ndet != self.rho.ndet:
            raise ValueError("Your correlation matrix has the wrong "
                             "dimensions.")

        # H0 in km/s/Mpc in Hz
        H0 = self.h * 3.24077929E-18
        # 2 pi^2 f^3 / 3 H0^2
        # Normalization for I->Omega translation
        self.norm_pivot = 4 * np.pi**2 * self.f_pivot**3 / (3 * H0**2)

    def _get_iS_f(self, f, pmat):
        # Get S matrix:
        # [n_f, n_det]
        S_f_diag = np.array([d.psd(f) for d in self.dets]).T
        # [n_f, n_det, n_det]
        S_f = np.sqrt(S_f_diag[:, :, None] * S_f_diag[:, None, :])
        S_f *= self.rho.get_corrmat(f)
        if pmat is not None:
            S_f = np.einsum('ijk,lj,mk', S_f, pmat, pmat)
        # Invert
        iS_f = np.linalg.pinv(S_f, rcond=self.rcond)
        return iS_f

    def _compute_projector(self, proj):
        if proj is not None:
            vecs = proj.get('vectors').copy()
            if np.ndim(vecs) == 1:
                vecs = np.array([vecs])
            if np.ndim(vecs) != 2:
                raise ValueError("Projection vectors have wrong shape")

            # vecs is [nv, nd]
            nv, nd = vecs.shape
            if nd != self.ndet:
                raise ValueError("Projection vectors have wrong shape")

            # Compute covariance and pseudo-inverse
            # C = v vT -> [nv, nd] [nd, nv] -> [nv, nv]
            cov = np.einsum('ij,kj', vecs, vecs)
            icov = np.linalg.pinv(cov)

            # P = vT IC v -> [nd, nv] [nv, nv] [nv, nd] -> [nd, nd]
            pmat = np.einsum('ik,ij,jl',
                             vecs, icov, vecs)
            if proj.get('deproject'):
                pmat = np.eye(nd) - pmat
        else:
            pmat = None
        return pmat

    def _precompute_skyvec(self, theta, phi):
        # Computes cos and sin for theta and phi.
        theta_use = np.atleast_1d(theta)
        phi_use = np.atleast_1d(phi)
        ct = np.cos(theta_use)
        st = np.sin(theta_use)
        cp = np.cos(phi_use)
        sp = np.sin(phi_use)
        return ct, st, cp, sp

    def _get_baseline_product(self, t, ct, st, cp, sp,
                              dA, dB):
        # Computes (x_A - x_B) . \hat{n} as a function
        # of time.
        t_use = np.atleast_1d(t)

        # [3, nt]
        x_A = dA.get_position(t_use)
        x_B = dB.get_position(t_use)

        # [3, npix]
        nv = np.array([st*cp, st*sp, ct])

        # [nt, npix]
        bprod = np.einsum('ik,il', x_A-x_B, nv)
        return bprod

    def _get_antenna_ij(self, i, j, t, f, ct, st, cp, sp,
                        pol=False, inc_baseline=True):
        # Returns antenna pattern for detector pair (i, j).
        dA = self.dets[i]
        dB = self.dets[j]
        t_use = np.atleast_1d(t)
        f_use = np.atleast_1d(f)

        # [3, npix]
        ll = np.array([sp, -cp, np.zeros_like(sp)])
        # [3, npix]
        mm = np.array([cp*ct, sp*ct, -st])
        # [3, npix]
        nn = np.array([st*cp, st*sp, ct])
        # e_+ [3, 3, npix]
        e_p = (ll[:, None, ...]*ll[None, :, ...] -
               mm[:, None, ...]*mm[None, :, ...])
        # e_x [3, 3, npix]
        e_x = (ll[:, None, ...]*mm[None, :, ...] +
               mm[:, None, ...]*ll[None, :, ...])

        # [nt, nf, npix]
        tr_Ap, tr_Ax = dA.get_Fp(t_use, f_use, e_p, e_x, nn)
        tr_Bp, tr_Bx = dB.get_Fp(t_use, f_use, e_p, e_x, nn)

        def tr_prod(tr1, tr2):
            return tr1 * np.conj(tr2)

        # Antenna patterns
        prefac = 5/(8*np.pi)
        if pol:
            g = prefac*np.array([tr_prod(tr_Ap, tr_Bp) +
                                 tr_prod(tr_Ax, tr_Bx),  # I
                                 tr_prod(tr_Ap, tr_Bp) -
                                 tr_prod(tr_Ax, tr_Bx),  # Q
                                 tr_prod(tr_Ap, tr_Bx) +
                                 tr_prod(tr_Ax, tr_Bp),  # U
                                 1j*(tr_prod(tr_Ap, tr_Bx) -
                                     tr_prod(tr_Ax, tr_Bp))])  # V
        else:
            g = prefac*(tr_prod(tr_Ap, tr_Bp) +
                        tr_prod(tr_Ax, tr_Bx))

        if inc_baseline:
            # [nt, npix]
            bn = self._get_baseline_product(t_use, ct, st, cp, sp,
                                            dA, dB)
            # [nt, nf, npix]
            phase = np.exp(-1j*2*np.pi *
                           f_use[None, :, None] *
                           bn[:, None, :] / self.clight)
            if pol:
                g = g * phase[None, ...]
            else:
                g = g * phase
        return g

    def get_antenna(self, i, j, t, f, theta, phi,
                    pol=False, inc_baseline=True):
        """ Returns antenna pattern for a detector pair
        as a function of time, frequency and sky position.

        Args:
            i: index of first detector
            j: index of second detector
            t: array of `N_t` times (in s).
            f: array of `N_f` times (in Hz).
            theta: array of `N_pix` colatitude values (in radians).
            phi: array of `N_pix` azimuth values (in radians).
            pol (bool): compute all polarized components?
                (default: False).
            inc_baseline: include baseline-related phase. Otherwise
                only the :math:`\\gamma` overlap function in Eq.
                22 of the companion paper will be returned.
                (default: True).
        """
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_antenna_ij(i, j, t, f,
                                               ct, st, cp, sp,
                                               pol=pol,
                                               inc_baseline=inc_baseline))

    def _plot_antenna(self, t, f, n_theta=100, n_phi=100, i=0, j=0):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        phi = np.linspace(0, np.pi, n_phi)
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi, theta = np.meshgrid(phi, theta)
        antenna = self.get_antenna(i, j, t, f,
                                   theta.flatten(),
                                   phi.flatten(),
                                   inc_baseline=False)
        antenna = np.abs(antenna.reshape([n_theta, n_phi]))
        x = antenna * np.sin(phi) * np.cos(theta)
        y = antenna * np.sin(phi) * np.sin(theta)
        z = antenna * np.cos(phi)
        gmax, gmin = antenna.max(), antenna.min()
        fcolors = (antenna - gmin)/(gmax - gmin)
        fig = plt.figure(figsize=plt.figaspect(1.))  # noqa
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.det_A.name+" "+self.det_B.name)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1,
                        facecolors=cm.seismic(fcolors))  # noqa
        ax.set_axis_off()

    def get_Ninv_t(self, t, f, nside, is_fspacing_log=False,
                   no_autos=False, deltaOmega_norm=True,
                   proj=None):
        """ Computes inverse noise variance map for a set of
        timeframes integrated over frequency.

        Args:
            t: array of `N_t` time values (in s).
            f: array of frequency values that will be
                integrated over.
            nside: HEALPix resolution parameter.
            is_fspacing_log: if `True`, `f` is log-spaced
                (linearly-spaced otherwise).
                (Default: `False`).
            no_autos (bool, or array_like): if a single `True`
                value, all detector auto-correlations will be
                removed. If a 1D array, only the auto-correlations
                for which the array element is `True` will be
                removed. If a 2D array, all autos and cross-
                correlations for which the array element is `True`
                will be removed.
            deltaOmega_norm: if `True`, the quantity being mapped is
                :math:`\\delta\\Omega = (\\Omega/\\bar{\\Omega}-1)/4\\pi`.
                Otherwise the :math:`4\\pi` factor is omitted.
                (Default: `True`).
            proj (dictionary or `None`): if you want to project the data
                onto a set of linear combinations of the detectors, pass
                the linear coefficients of those here. `proj` should be
                a dictionary with two items: 'vectors' containing a 2D
                array (or a single vector) with the linear coefficients
                as rows and 'deproj'. If 'deproj' is `True`, then those
                linear combinations will actually be projeted out. If
                `proj` is `None`, then no projection or de-projection
                will happen.

        Returns:
            array_like: array of shape `[N_t, N_pix]` containing
            the inverse noise variance map sampled at the
            `N_pix` pixel positions corresponding to the input
            HEALPix resolution parameter (in RING ordering).
        """
        if np.ndim(no_autos) == 0:
            no_autos = np.array([no_autos] * self.ndet)
        else:
            if len(no_autos) != self.ndet:
                raise ValueError("No autos should have %d elements" %
                                 self.ndet)
        if np.ndim(no_autos) == 1:
            no_autos = np.diag(no_autos)
        no_autos = np.array(no_autos)

        pmat = self._compute_projector(proj)

        t_use = np.atleast_1d(t)
        f_use = f
        if is_fspacing_log:
            dlf = np.mean(np.diff(np.log(f)))
            df = f * dlf
        else:
            df = np.mean(np.diff(f)) * np.ones(len(f))

        npix = hp.nside2npix(nside)
        th, ph = hp.pix2ang(nside, np.arange(npix))
        ct, st, cp, sp = self._precompute_skyvec(th, ph)

        # Get S matrix:
        iS_f = self._get_iS_f(f_use, pmat)

        # Get all maps
        antennas = np.zeros([self.ndet, self.ndet,
                             len(t_use), len(f_use), npix],
                            dtype=np.cdouble)
        for i1 in range(self.ndet):
            for i2 in range(i1, self.ndet):
                a12 = self._get_antenna_ij(i1, i2, t_use, f_use,
                                           ct, st, cp, sp,
                                           inc_baseline=True)
                antennas[i1, i2, :, :, :] = a12
                if i2 != i1:
                    antennas[i2, i1, :, :, :] = np.conj(a12)
        if pmat is not None:
            antennas = np.einsum('ijmno,ki,lj', antennas, pmat, pmat)

        # Prefactors
        e_f = (f_use / self.f_pivot)**self.specin_omega / self.norm_pivot
        prefac = 0.5 * (2 * e_f / 5)**2
        if deltaOmega_norm:
            prefac *= (4*np.pi)**2

        # Loop over detectors
        inoivar = np.zeros([len(t_use), npix])
        for iA in range(self.ndet):
            for iB in range(self.ndet):
                iS_AB = iS_f[:, iA, iB]
                for iC in range(self.ndet):
                    gBC = antennas[iB, iC, :, :, :]
                    if iB == iC and no_autos[iB, iC]:
                        continue
                    for iD in range(self.ndet):
                        iS_CD = iS_f[:, iC, iD]
                        gDA = antennas[iD, iA, :, :, :]
                        if iA == iD and no_autos[iA, iD]:
                            continue
                        ff = df*prefac*iS_AB*iS_CD
                        inoivar += np.sum(ff[None, :, None] *
                                          np.real(gBC * gDA), axis=1)
        return np.squeeze(inoivar)

    def get_pi_curve(self, t, f, nside, is_fspacing_log=False,
                     no_autos=False, beta_range=[-10, 10],
                     nsigma=1, proj=None):
        """ Computes the power-law-integrated (PI) sensitivity curve
        for this network (see arXiv:1310.5300).

        Args:
            t (float or array_like): `N_t` time values (in s). If a single
                number is passed, then the "rigid network" approximation
                is used, and this time is interpreted as the total
                observing time. Otherwise, an integral over time is
                performed.
            f: array of `N_f` frequency values (in Hz). This will be the
                frequencies at which the PI curve will be sampled, and also
                the frequencies used for numerical integration.
            nside: HEALPix resolution parameter. Used to create
                maps of the antenna pattern and computes its sky
                average.
            no_autos (bool, or array_like): if a single `True`
                value, all detector auto-correlations will be
                removed. If a 1D array, only the auto-correlations
                for which the array element is `True` will be
                removed. If a 2D array, all autos and cross-
                correlations for which the array element is `True`
                will be removed.
            beta_range: a list containing the range of power law indices
                for which the PI curve will be computed.
            nsigma: S/N of the PI curve (default: 1-sigma).
            proj (dictionary or `None`): if you want to project the data
                onto a set of linear combinations of the detectors, pass
                the linear coefficients of those here. `proj` should be
                a dictionary with two items: 'vectors' containing a 2D
                array (or a single vector) with the linear coefficients
                as rows and 'deproj'. If 'deproj' is `True`, then those
                linear combinations will actually be projeted out. If
                `proj` is `None`, then no projection or de-projection
                will happen.

        Returns:
            array_like: array of size `N_f`.
        """
        t_use = np.atleast_1d(t)
        f_use = f
        if is_fspacing_log:
            dlf = np.mean(np.diff(np.log(f)))
            df = f * dlf
        else:
            df = np.mean(np.diff(f)) * np.ones(len(f))
        inv_dsig2_dnu_dt = self.get_dsigm2_dnu_t(t_use, f_use, nside,
                                                 no_autos=no_autos,
                                                 proj=proj)
        # Sum over time
        if len(t_use) == 1:
            inv_dsig2_dnu = np.squeeze(inv_dsig2_dnu_dt * t)
        else:
            dt = np.mean(np.diff(t_use))
            inv_dsig2_dnu = np.sum(inv_dsig2_dnu_dt, axis=0) * dt

        def _om(beta):
            # Sum over frequencies
            plaw = (f_use/self.f_pivot)**beta
            snm2 = np.sum(inv_dsig2_dnu * plaw**2 * df)
            return nsigma * plaw / np.sqrt(snm2)

        betas = np.linspace(beta_range[0], beta_range[1], 100)
        oms = np.array([_om(b) for b in betas])
        pi = np.max(oms, axis=0)
        return pi

    def get_dsigm2_dnu_t(self, t, f, nside, no_autos=False,
                         proj=None):
        """ Computes :math:`d\\sigma^{-2}/df\\,dt` for a set
        of frequencies and times.

        Args:
            t: array of `N_t` time values (in s).
            f: array of `N_f` frequency values (in Hz).
            nside: HEALPix resolution parameter. Used to create
                maps of the antenna pattern and computes its sky
                average.
            no_autos (bool, or array_like): if a single `True`
                value, all detector auto-correlations will be
                removed. If a 1D array, only the auto-correlations
                for which the array element is `True` will be
                removed. If a 2D array, all autos and cross-
                correlations for which the array element is `True`
                will be removed.
            proj (dictionary or `None`): if you want to project the data
                onto a set of linear combinations of the detectors, pass
                the linear coefficients of those here. `proj` should be
                a dictionary with two items: 'vectors' containing a 2D
                array (or a single vector) with the linear coefficients
                as rows and 'deproj'. If 'deproj' is `True`, then those
                linear combinations will actually be projeted out. If
                `proj` is `None`, then no projection or de-projection
                will happen.

        Returns:
            array_like: array of shape `[N_t, N_f]`.
        """
        if np.ndim(no_autos) == 0:
            no_autos = np.array([no_autos] * self.ndet)
        else:
            if len(no_autos) != self.ndet:
                raise ValueError("No autos should have %d elements" %
                                 self.ndet)
        if np.ndim(no_autos) == 1:
            no_autos = np.diag(no_autos)
        no_autos = np.array(no_autos)

        pmat = self._compute_projector(proj)

        t_use = np.atleast_1d(t)
        f_use = f

        npix = hp.nside2npix(nside)
        pix_area = 4*np.pi/npix
        th, ph = hp.pix2ang(nside, np.arange(npix))
        ct, st, cp, sp = self._precompute_skyvec(th, ph)

        # Get S matrix:
        iS_f = self._get_iS_f(f_use, pmat)

        # Get all maps
        gammas = np.zeros([self.ndet, self.ndet,
                           len(t_use), len(f_use)],
                          dtype=np.cdouble)
        for i1 in range(self.ndet):
            for i2 in range(i1, self.ndet):
                a12 = self._get_antenna_ij(i1, i2, t_use, f_use,
                                           ct, st, cp, sp,
                                           inc_baseline=True)
                # Sky integral
                ia12 = np.sum(a12, axis=-1) * pix_area
                gammas[i1, i2, :, :] = ia12
                if i2 != i1:
                    gammas[i2, i1, :, :] = np.conj(ia12)
        if pmat is not None:
            gammas = np.einsum('ijmn,ki,lj', gammas, pmat, pmat)

        # Translation between Omega and I
        e_f = (self.f_pivot / f_use)**3 / self.norm_pivot
        # Prefactors
        prefac = 0.5 * (2 * e_f / 5)**2

        # Loop over detectors
        inoivar = np.zeros([len(t_use), len(f_use)])
        for iA in range(self.ndet):
            for iB in range(self.ndet):
                iS_AB = iS_f[:, iA, iB]
                for iC in range(self.ndet):
                    rBC = gammas[iB, iC, :, :]
                    if iB == iC and no_autos[iB, iC]:
                        continue
                    for iD in range(self.ndet):
                        iS_CD = iS_f[:, iC, iD]
                        rDA = gammas[iD, iA, :, :]
                        if iA == iD and no_autos[iA, iD]:
                            continue
                        ff = prefac*iS_AB*iS_CD
                        inoivar += ff[None, :] * np.real(rBC * rDA)
        return np.squeeze(inoivar)

    def get_N_ell(self, t, f, nside, is_fspacing_log=False,
                  no_autos=False, deltaOmega_norm=True, proj=None):
        """ Computes :math:`N_\\ell` for this network.

        Args:
            t (float or array_like): `N_t` time values (in s). If a single
                number is passed, then the "rigid network" approximation
                is used, and this time is interpreted as the total
                observing time. Otherwise, an integral over time is
                performed.
            f: array of `N_f` frequency values (in Hz).
            nside: HEALPix resolution parameter used to compute spherical
                harmonic transforms.
            is_fspacing_log: if `True`, `f` is log-spaced
                (linearly-spaced otherwise).
                (Default: `False`).
            no_autos (bool, or array_like): if a single `True`
                value, all detector auto-correlations will be
                removed. If a 1D array, only the auto-correlations
                for which the array element is `True` will be
                removed. If a 2D array, all autos and cross-
                correlations for which the array element is `True`
                will be removed.
            deltaOmega_norm: if `True`, the quantity being mapped is
                :math:`\\delta\\Omega = (\\Omega/\\bar{\\Omega}-1)/4\\pi`.
                Otherwise the :math:`4\\pi` factor is omitted.
                (Default: `True`).
            proj (dictionary or `None`): if you want to project the data
                onto a set of linear combinations of the detectors, pass
                the linear coefficients of those here. `proj` should be
                a dictionary with two items: 'vectors' containing a 2D
                array (or a single vector) with the linear coefficients
                as rows and 'deproj'. If 'deproj' is `True`, then those
                linear combinations will actually be projeted out. If
                `proj` is `None`, then no projection or de-projection
                will happen.

        Returns:
            array_like: array of size `N_l = 3 * nside` containing the noise
                power spectrum.
        """
        t_use = np.atleast_1d(t)
        f_use = np.atleast_1d(f)
        if is_fspacing_log:
            dlf = np.mean(np.diff(np.log(f)))
            df = f * dlf
        else:
            df = np.mean(np.diff(f)) * np.ones(len(f))
        gls = self.get_G_ell(t_use, f_use, nside, no_autos=no_autos,
                             deltaOmega_norm=deltaOmega_norm, proj=proj)
        # Sum over frequencies
        gls = np.sum(gls * df[:, None, None], axis=0)
        # Sum over times
        if len(t_use) == 1:
            gls = np.squeeze(gls * t)
        else:
            dt = np.mean(np.diff(t_use))
            gls = np.sum(gls, axis=0) * dt
        return 1/gls

    def get_G_ell(self, t, f, nside, no_autos=False, deltaOmega_norm=True,
                  proj=None):
        """ Computes :math:`G_\\ell` in Eq. 37 of the companion paper.

        Args:
            t: array of `N_t` time values (in s).
            f: array of `N_f` frequency values (in Hz).
            nside: HEALPix resolution parameter used to compute spherical
                harmonic transforms.
            no_autos (bool, or array_like): if a single `True`
                value, all detector auto-correlations will be
                removed. If a 1D array, only the auto-correlations
                for which the array element is `True` will be
                removed. If a 2D array, all autos and cross-
                correlations for which the array element is `True`
                will be removed.
            deltaOmega_norm: if `True`, the quantity being mapped is
                :math:`\\delta\\Omega = (\\Omega/\\bar{\\Omega}-1)/4\\pi`.
                Otherwise the :math:`4\\pi` factor is omitted.
                (Default: `True`).
            proj (dictionary or `None`): if you want to project the data
                onto a set of linear combinations of the detectors, pass
                the linear coefficients of those here. `proj` should be
                a dictionary with two items: 'vectors' containing a 2D
                array (or a single vector) with the linear coefficients
                as rows and 'deproj'. If 'deproj' is `True`, then those
                linear combinations will actually be projeted out. If
                `proj` is `None`, then no projection or de-projection
                will happen.

        Returns:
            array_like: array of shape `[N_f, N_t, N_l]`, where
                `N_l = 3 * nside`, containing :math:`G_\\ell` at each
                frequency and time.
        """
        if np.ndim(no_autos) == 0:
            no_autos = np.array([no_autos] * self.ndet)
        else:
            if len(no_autos) != self.ndet:
                raise ValueError("No autos should have %d elements" %
                                 self.ndet)
        if np.ndim(no_autos) == 1:
            no_autos = np.diag(no_autos)
        no_autos = np.array(no_autos)

        pmat = self._compute_projector(proj)

        t_use = np.atleast_1d(t)
        f_use = np.atleast_1d(f)

        nf = len(f_use)
        nt = len(t_use)
        npix = hp.nside2npix(nside)
        nalm = (3*nside*(3*nside+1))//2
        nell = 3*nside
        th, ph = hp.pix2ang(nside, np.arange(npix))
        ct, st, cp, sp = self._precompute_skyvec(th, ph)

        # Get S matrix:
        iS_f = self._get_iS_f(f_use, pmat)

        # Get antenna alms
        aalms_r = np.zeros([self.ndet, self.ndet,
                            nt, nf, nalm],
                           dtype=np.cdouble)
        aalms_i = np.zeros([self.ndet, self.ndet,
                            nt, nf, nalm],
                           dtype=np.cdouble)
        for i1 in range(self.ndet):
            for i2 in range(i1, self.ndet):
                if no_autos[i1, i2]:
                    continue
                antenna = self._get_antenna_ij(i1, i2, t_use, f_use,
                                               ct, st, cp, sp,
                                               inc_baseline=True)
                for i_t in range(nt):
                    for i_f in range(nf):
                        a = antenna[i_t, i_f, :]
                        alm_r = hp.map2alm(np.real(a))
                        alm_i = hp.map2alm(np.imag(a))
                        aalms_r[i1, i2, i_t, i_f, :] = alm_r
                        aalms_i[i1, i2, i_t, i_f, :] = alm_i
                        if i2 != i1:
                            aalms_r[i2, i1, i_t, i_f, :] = alm_r
                            aalms_i[i2, i1, i_t, i_f, :] = -alm_i

        if pmat is not None:
            aalms_r = np.einsum('ijmno,ki,lj', aalms_r, pmat, pmat)
            aalms_i = np.einsum('ijmno,ki,lj', aalms_i, pmat, pmat)

        # Prefactors
        e_f = (f_use / self.f_pivot)**self.specin_omega / self.norm_pivot
        prefac = 0.5 * (2 * e_f / 5)**2
        if deltaOmega_norm:
            prefac *= (4*np.pi)**2

        gls = np.zeros([nf, nt, nell])
        for i_t in range(nt):
            for i_f in range(nf):
                for iA in range(self.ndet):
                    for iB in range(self.ndet):
                        iS_AB = iS_f[i_f, iA, iB]
                        for iC in range(self.ndet):
                            gBCr = aalms_r[iB, iC, i_t, i_f, :]
                            gBCi = aalms_i[iB, iC, i_t, i_f, :]
                            if no_autos[iB, iC]:
                                continue
                            for iD in range(self.ndet):
                                iS_CD = iS_f[i_f, iC, iD]
                                gADr = aalms_r[iA, iD, i_t, i_f, :]
                                gADi = aalms_i[iA, iD, i_t, i_f, :]
                                if no_autos[iA, iD]:
                                    continue
                                clr = hp.alm2cl(gBCr, gADr)
                                cli = hp.alm2cl(gBCi, gADi)
                                gls[i_f, i_t, :] += iS_AB * iS_CD * (clr + cli)

        gls = gls * prefac[:, None, None]
        if np.ndim(f) == 0:
            gls = np.squeeze(gls, axis=0)
            if np.ndim(t) == 0:
                gls = np.squeeze(gls, axis=0)
        else:
            if np.ndim(t) == 0:
                gls = np.squeeze(gls, axis=1)
        return gls
