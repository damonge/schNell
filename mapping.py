import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class MapCalculator(object):
    clight = 299792458.

    def __init__(self, det_A, det_B, f_pivot=63., spectral_index=2./3.):
        self.det_A = det_A
        self.det_B = det_B
        self.f_pivot = f_pivot
        self.specin_omega = spectral_index - 3

    def norm_pivot(self, h=0.67):
        # H0 in km/s/Mpc in Hz
        H0 = h * 3.24077929E-18
        # 2 pi^2 f^3 / 3 H0^2
        return 4 * np.pi**2 * self.f_pivot**3 / (3 * H0**2)

    def _precompute_skyvec(self, theta, phi):
        theta_use = np.atleast_1d(theta)
        phi_use = np.atleast_1d(phi)
        ct = np.cos(theta_use)
        st = np.sin(theta_use)
        cp = np.cos(phi_use)
        sp = np.sin(phi_use)
        return ct, st, cp, sp

    def _get_baseline_product(self, t, ct, st, cp, sp,
                              dA=None, dB=None):
        if dA is None:
            dA = self.det_A
        if dB is None:
            dB = self.det_B
        t_use = np.atleast_1d(t)

        # [3, nt]
        x_A = dA.get_position(t_use)
        x_B = dB.get_position(t_use)

        # [3, npix]
        nv = np.array([st*cp, st*sp, ct])

        # [nt, npix]
        bprod = np.einsum('ik,il', x_A-x_B, nv)
        return bprod

    def get_baseline_product(self, t, theta, phi, dA=None, dB=None):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_baseline_product(t, ct, st, cp, sp,
                                                     dA=dA, dB=dB))

    def _get_gamma(self, t, f, ct, st, cp, sp, pol=False,
                   inc_baseline=True, typ='A,B'):
        if typ == 'A,B':
            g = self._get_gamma_single(t, f, ct, st, cp, sp,
                                       pol=pol,
                                       inc_baseline=inc_baseline,
                                       dA=None, dB=None)
        elif typ == 'I,I':
            g = self._get_gamma_single(t, f, ct, st, cp, sp,
                                       pol=pol,
                                       inc_baseline=inc_baseline,
                                       dA=self.det_A, dB=self.det_A)
        else:
            gAA = self._get_gamma_single(t, f, ct, st, cp, sp,
                                         pol=pol,
                                         inc_baseline=inc_baseline,
                                         dA=self.det_A, dB=self.det_A)
            gAB = self._get_gamma_single(t, f, ct, st, cp, sp,
                                         pol=pol,
                                         inc_baseline=inc_baseline,
                                         dA=self.det_A, dB=self.det_B)
            if typ == 'I,II':
                g = (gAA + 2*gAB) / np.sqrt(3)
            else:
                gBB = self._get_gamma_single(t, f, ct, st, cp, sp,
                                             pol=pol,
                                             inc_baseline=inc_baseline,
                                             dA=self.det_B, dB=self.det_B)
                if typ == 'II,II':
                    g = (gAA + 4*gBB + 4*np.real(gAB)) / 3.
                elif typ == '+,+':
                    g = 0.5*(gAA + gBB + 2*np.real(gAB))
                elif typ == '-,-':
                    g = 0.5*(gAA + gBB - 2*np.real(gAB))
                elif typ == '+,-':
                    g = 0.5*(gAA - gBB - 2*1j*np.imag(gAB))
                else:
                    raise ValueError("Unknown gamma type " + typ)
        return g

    def _get_gamma_single(self, t, f, ct, st, cp, sp, pol=False,
                          inc_baseline=True, dA=None, dB=None):
        if dA is None:
            dA = self.det_A
        if dB is None:
            dB = self.det_B
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

        # Gammas
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
                                            dA=dA, dB=dB)
            # [nt, nf, npix]
            phase = np.exp(1j*2*np.pi *
                           f_use[None, :, None] *
                           bn[:, None, :] / self.clight)
            if pol:
                g = g * phase[None, ...]
            else:
                g = g * phase
        return g

    def get_gamma(self, t, f, theta, phi, pol=False,
                  inc_baseline=True, typ='A,B'):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_gamma(t, f, ct, st, cp, sp,
                                          pol=pol, typ=typ,
                                          inc_baseline=inc_baseline))

    def plot_gamma(self, t, f, n_theta=100, n_phi=100, typ='A,B'):
        from mpl_toolkits.mplot3d import Axes3D
        phi = np.linspace(0, np.pi, n_phi)
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi, theta = np.meshgrid(phi, theta)
        gamma = self.get_gamma(t, f,
                               theta.flatten(),
                               phi.flatten(),
                               inc_baseline=False,
                               typ=typ)
        gamma = np.abs(gamma.reshape([n_theta, n_phi]))
        x = gamma * np.sin(phi) * np.cos(theta)
        y = gamma * np.sin(phi) * np.sin(theta)
        z = gamma * np.cos(phi)
        gmax, gmin = gamma.max(), gamma.min()
        fcolors = (gamma - gmin)/(gmax - gmin)
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.det_A.name+" "+self.det_B.name)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1,
                        facecolors=cm.seismic(fcolors))
        ax.set_axis_off()

    def get_Ninv_t(self, t, f, nside, typ='A,B', is_fspacing_log=False, deltaOmega_norm=True):
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

        # [nt, nf, npix]
        gamma = self._get_gamma(t_use, f_use, ct, st, cp, sp,
                                inc_baseline=True, typ=typ)

        s_A = self.det_A.psd(f_use)
        s_B = self.det_B.psd(f_use)
        e_f = (f_use / self.f_pivot)**self.specin_omega / self.norm_pivot()
        pre_A = 2 * e_f / (5 * s_A)
        pre_B = 2 * e_f / (5 * s_B)
        if deltaOmega_norm:
            pre_A *= 4*np.pi
            pre_B *= 4*np.pi

        # Noise prefactor for special detector combinations
        if typ == 'A,B':
            if self.det_A.name == self.det_B.name:
                # Extra factor 2 if auto-correlation
                nprefac = 0.5
            else:
                nprefac = 1
        elif (typ == 'I,I') or (typ == 'II,II'):
            nprefac = 0.5
        elif typ == 'I,II':
            nprefac = 1
        elif typ == '+,+':
            # 0.5 for auto-correlation, 1/0.5 for each + combination
            nprefac = 0.5 * 2**2
        elif typ == '-,-':
            # 0.5 for auto-correlation, 1/1.5 for each - combination
            nprefac = 0.5 * (2./3.)**2
        elif typ == '+,-':
            # 1/0.5 for +, 1/1.5 for -
            nprefac = 4./3.
        else:
            raise ValueError("Unknown gamma type " + typ)

        # Integrate over frequencies
        inoivar = np.sum(((nprefac * pre_A * pre_B * df)[None, :, None] *
                          np.abs(gamma)**2), axis=1)
        return np.squeeze(inoivar)

    def get_G_ell(self, t, f, nside, typ='A,B', deltaOmega_norm=True):
        t_use = np.atleast_1d(t)
        f_use = np.atleast_1d(f)

        nf = len(f_use)
        nt = len(t_use)
        npix = hp.nside2npix(nside)
        nell = 3*nside

        th, ph = hp.pix2ang(nside, np.arange(npix))
        ct, st, cp, sp = self._precompute_skyvec(th, ph)

        # [nt, nf, npix]
        gamma = self._get_gamma(t_use, f_use, ct, st, cp, sp,
                                inc_baseline=True, typ=typ)

        s_A = self.det_A.psd(f_use)
        s_B = self.det_B.psd(f_use)
        e_f = (f_use / self.f_pivot)**self.specin_omega / self.norm_pivot()
        pre_A = 2 * e_f / (5 * s_A)
        pre_B = 2 * e_f / (5 * s_B)
        if deltaOmega_norm:
            pre_A *= 4*np.pi
            pre_B *= 4*np.pi

        # Noise prefactor for special detector combinations
        if typ == 'A,B':
            if self.det_A.name == self.det_B.name:
                # Extra factor 2 if auto-correlation
                nprefac = 0.5
            else:
                nprefac = 1
        elif (typ == 'I,I') or (typ == 'II,II'):
            nprefac = 0.5
        elif typ == 'I,II':
            nprefac = 1
        elif typ == '+,+':
            # 0.5 for auto-correlation, 1/0.5 for each + combination
            nprefac = 0.5 * 2**2
        elif typ == '-,-':
            # 0.5 for auto-correlation, 1/1.5 for each - combination
            nprefac = 0.5 * (2./3.)**2
        elif typ == '+,-':
            # 1/0.5 for +, 1/1.5 for -
            nprefac = 4./3.
        else:
            raise ValueError("Unknown gamma type " + typ)

        gls = np.zeros([nf, nt, nell])
        for i_t, time in enumerate(t_use):
            for i_f, freq in enumerate(f_use):
                g = gamma[i_t, i_f, :]
                # Power spectrum of the real part
                g_r = hp.anafast(np.real(g))
                # Power spectrum of the imaginary part
                g_i = hp.anafast(np.imag(g))
                gls[i_f, i_t, :] = (g_r + g_i) * pre_A[i_f] * pre_B[i_f]

        return nprefac * np.squeeze(gls)


class MapCalculatorFromArray(MapCalculator):
    clight = 299792458.

    def __init__(self, det_array, f_pivot=63., spectral_index=2./3.,
                 corr_matrix=None):
        self.dets = det_array
        self.ndet = len(det_array)
        self.f_pivot = f_pivot
        self.specin_omega = spectral_index - 3
        if corr_matrix is None:
            self.rho = np.eye(self.ndet)
        else:
            if corr_matrix.shape != (self.ndet, self.ndet):
                raise ValueError("Wrong correlation matrix shape")
            self.rho = corr_matrix

    def _get_gamma_ij(self, i, j, t, f, ct, st, cp, sp, pol=False,
                      inc_baseline=True):
        return self._get_gamma_single(t, f, ct, st, cp, sp,
                                      pol=pol, inc_baseline=inc_baseline,
                                      dA=self.dets[i], dB=self.dets[j])

    def get_gamma(self, i, j, t, f, theta, phi, pol=False,
                  inc_baseline=True):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_gamma_ij(i, j, t, f,
                                             ct, st, cp, sp,
                                             pol=pol,
                                             inc_baseline=inc_baseline))

    def plot_gamma(self, t, f, n_theta=100, n_phi=100, i=0, j=0):
        from mpl_toolkits.mplot3d import Axes3D
        phi = np.linspace(0, np.pi, n_phi)
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi, theta = np.meshgrid(phi, theta)
        gamma = self.get_gamma(i, j, t, f,
                               theta.flatten(),
                               phi.flatten(),
                               inc_baseline=False)
        gamma = np.abs(gamma.reshape([n_theta, n_phi]))
        x = gamma * np.sin(phi) * np.cos(theta)
        y = gamma * np.sin(phi) * np.sin(theta)
        z = gamma * np.cos(phi)
        gmax, gmin = gamma.max(), gamma.min()
        fcolors = (gamma - gmin)/(gmax - gmin)
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.det_A.name+" "+self.det_B.name)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1,
                        facecolors=cm.seismic(fcolors))
        ax.set_axis_off()

    def get_Ninv_t(self, t, f, nside, is_fspacing_log=False,
                   no_autos=False, deltaOmega_norm=True):
        if np.ndim(no_autos) == 0:
            no_autos = np.array([no_autos] * self.ndet)
        else:
            if len(no_autos) != self.ndet:
                raise ValueError("No autos should have %d elements" %
                                 self.ndet)
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
        # [n_f, n_det]
        S_f_diag = np.array([d.psd(f_use) for d in self.dets]).T
        # [n_f, n_det, n_det]
        S_f = np.sqrt(S_f_diag[:, :, None] * S_f_diag[:, None, :])
        S_f *= self.rho[None, :, :]
        # Invert
        iS_f = np.linalg.inv(S_f)

        # Get all maps
        gammas = np.zeros([self.ndet, self.ndet,
                           len(t_use), len(f_use), npix],
                          dtype=np.cdouble)
        for i1 in range(self.ndet):
            for i2 in range(i1, self.ndet):
                g12 = self._get_gamma_ij(i1, i2, t_use, f_use,
                                         ct, st, cp, sp,
                                         inc_baseline=True)
                gammas[i1, i2, :, :, :] = g12
                if i2 != i1:
                    gammas[i2, i1, :, :, :] = np.conj(g12)

        # Prefactors
        e_f = (f_use / self.f_pivot)**self.specin_omega / self.norm_pivot()
        prefac = 0.5 * (2 * e_f / 5)**2
        if deltaOmega_norm:
            prefac *= (4*np.pi)**2

        # Loop over detectors
        inoivar = np.zeros([len(t_use), npix])
        for iA in range(self.ndet):
            for iB in range(self.ndet):
                iS_AB = iS_f[:, iA, iB]
                for iC in range(self.ndet):
                    gBC = gammas[iB, iC, :, :, :]
                    if iB == iC and no_autos[iB]:
                        continue
                    for iD in range(self.ndet):
                        iS_CD = iS_f[:, iC, iD]
                        gDA = gammas[iD, iA, :, :, :]
                        if iA == iD and no_autos[iA]:
                            continue
                        ff = df*prefac*iS_AB*iS_CD
                        inoivar += np.sum(ff[None, :, None] *
                                          np.real(gBC * gDA), axis=1)
        return np.squeeze(inoivar)

    def get_dsigm2_dnu_t(self, t, f, nside, no_autos=False):
        if np.ndim(no_autos) == 0:
            no_autos = np.array([no_autos] * self.ndet)
        else:
            if len(no_autos) != self.ndet:
                raise ValueError("No autos should have %d elements" %
                                 self.ndet)
        t_use = np.atleast_1d(t)
        f_use = f

        npix = hp.nside2npix(nside)
        pix_area = 4*np.pi/npix
        th, ph = hp.pix2ang(nside, np.arange(npix))
        ct, st, cp, sp = self._precompute_skyvec(th, ph)

        # Get S matrix:
        # [n_f, n_det]
        S_f_diag = np.array([d.psd(f_use) for d in self.dets]).T
        # [n_f, n_det, n_det]
        S_f = np.sqrt(S_f_diag[:, :, None] * S_f_diag[:, None, :])
        S_f *= self.rho[None, :, :]
        # Invert
        iS_f = np.linalg.inv(S_f)

        # Get all maps
        rhos = np.zeros([self.ndet, self.ndet,
                         len(t_use), len(f_use)],
                        dtype=np.cdouble)
        for i1 in range(self.ndet):
            for i2 in range(i1, self.ndet):
                g12 = self._get_gamma_ij(i1, i2, t_use, f_use,
                                         ct, st, cp, sp,
                                         inc_baseline=True)
                # Sky integral
                ig12 = np.sum(g12, axis=-1) * pix_area
                rhos[i1, i2, :, :] = ig12
                if i2 != i1:
                    rhos[i2, i1, :, :] = np.conj(ig12)

        # Translation between Omega and I
        e_f = (self.f_pivot / f_use)**3 / self.norm_pivot()
        # Prefactors
        prefac = 0.5 * (2 * e_f / 5)**2

        # Loop over detectors
        inoivar = np.zeros([len(t_use), len(f_use)])
        for iA in range(self.ndet):
            for iB in range(self.ndet):
                iS_AB = iS_f[:, iA, iB]
                for iC in range(self.ndet):
                    rBC = rhos[iB, iC, :, :]
                    if iB == iC and no_autos[iB]:
                        continue
                    for iD in range(self.ndet):
                        iS_CD = iS_f[:, iC, iD]
                        rDA = rhos[iD, iA, :, :]
                        if iA == iD and no_autos[iA]:
                            continue
                        ff = prefac*iS_AB*iS_CD
                        inoivar += ff[None, :] * np.real(rBC * rDA)
        return np.squeeze(inoivar)

    def get_G_ell(self, t, f, nside, no_autos=False, deltaOmega_norm=True):
        if np.ndim(no_autos) == 0:
            no_autos = np.array([no_autos] * self.ndet)
        else:
            if len(no_autos) != self.ndet:
                raise ValueError("No autos should have %d elements" %
                                 self.ndet)
        if np.ndim(no_autos) == 1:
            no_autos = np.diag(no_autos)
        no_autos = np.array(no_autos)

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
        # [n_f, n_det]
        S_f_diag = np.array([d.psd(f_use) for d in self.dets]).T
        # [n_f, n_det, n_det]
        S_f = np.sqrt(S_f_diag[:, :, None] * S_f_diag[:, None, :])
        S_f *= self.rho[None, :, :]
        # Invert
        iS_f = np.linalg.inv(S_f)

        # Get gamma alms
        galms_r = np.zeros([self.ndet, self.ndet,
                            nt, nf, nalm],
                           dtype=np.cdouble)
        galms_i = np.zeros([self.ndet, self.ndet,
                            nt, nf, nalm],
                           dtype=np.cdouble)
        for i1 in range(self.ndet):
            for i2 in range(i1, self.ndet):
                if no_autos[i1, i2]:
                    continue
                gamma = self._get_gamma_ij(i1, i2, t_use, f_use,
                                           ct, st, cp, sp,
                                           inc_baseline=True)
                for i_t in range(nt):
                    for i_f in range(nf):
                        g = gamma[i_t, i_f, :]
                        alm_r = hp.map2alm(np.real(g))
                        alm_i = hp.map2alm(np.imag(g))
                        galms_r[i1, i2, i_t, i_f, :] = alm_r
                        galms_i[i1, i2, i_t, i_f, :] = alm_i
                        if i2 != i1:
                            galms_r[i2, i1, i_t, i_f, :] = np.conj(alm_r)
                            galms_i[i2, i1, i_t, i_f, :] = np.conj(alm_i)

        # Prefactors
        e_f = (f_use / self.f_pivot)**self.specin_omega / self.norm_pivot()
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
                            gBCr = galms_r[iB, iC, i_t, i_f, :]
                            gBCi = galms_i[iB, iC, i_t, i_f, :]
                            if no_autos[iB, iC]:
                                continue
                            for iD in range(self.ndet):
                                iS_CD = iS_f[i_f, iC, iD]
                                gADr = galms_r[iA, iD, i_t, i_f, :]
                                gADi = galms_i[iA, iD, i_t, i_f, :]
                                if no_autos[iA, iD]:
                                    continue
                                clr = hp.alm2cl(gBCr, gADr)
                                cli = hp.alm2cl(gBCi, gADi)
                                gls[i_f, i_t, :] += iS_AB * iS_CD * (clr + cli)

        return np.squeeze(gls * prefac[:, None, None])
