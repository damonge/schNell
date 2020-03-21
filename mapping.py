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
        self.spectral_index_omega = spectral_index - 3

    def norm_pivot(self, h=0.67):
        # H0 in km/s/Mpc in Hz
        H0 = h * 3.24077929E-18
        # 2 pi^2 f^3 / 3 H0^2
        return 2 * np.pi**2 * self.f_pivot**3 / (3 * H0**2)

    def _precompute_skyvec(self, theta, phi):
        theta_use = np.atleast_1d(theta)
        phi_use = np.atleast_1d(phi)
        ct = np.cos(theta_use)
        st = np.sin(theta_use)
        cp = np.cos(phi_use)
        sp = np.sin(phi_use)
        return ct, st, cp, sp

    def _get_baseline_product(self, t, ct, st, cp, sp):
        t_use = np.atleast_1d(t)
        
        # [3, nt]
        x_A = self.det_A.get_position(t_use)
        x_B = self.det_B.get_position(t_use)
        
        # [3, npix]
        nv = np.array([st*cp,st*sp,ct])

        # [nt, npix]
        bprod = np.einsum('ik,il',x_A-x_B,nv)
        return bprod

    def get_baseline_product(self, t, theta, phi):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_baseline_product(t, ct, st, cp, sp))

    def _get_gamma(self, t, f, ct, st, cp, sp, pol=False):
        t_use = np.atleast_1d(t)

        # [3, 3, nt]
        xx_A, yy_A = self.det_A.get_xx_yy(t_use)
        a_A = 0.5 * (xx_A - yy_A)

        # [3, 3, nt]
        xx_B, yy_B = self.det_B.get_xx_yy(t_use)
        a_B = 0.5 * (xx_B - yy_B)

        # [3, npix]
        l = np.array([sp,-cp,np.zeros_like(sp)])
        # [3, npix]
        m = np.array([cp*ct,sp*ct,-st])
        # e_+ [3, 3, npix]
        e_p = l[:,None,...]*l[None,:,...]-m[:,None,...]*m[None,:,...]
        # e_x [3, 3, npix]
        e_x = l[:,None,...]*m[None,:,...]+m[:,None,...]*l[None,:,...]

        # Tr[a_A*e_+]
        # [nt, npix]
        tr_Ap = np.einsum('ijk,jil',a_A, e_p)
        tr_Bp = np.einsum('ijk,jil',a_B, e_p)
        tr_Ax = np.einsum('ijk,jil',a_A, e_x)
        tr_Bx = np.einsum('ijk,jil',a_B, e_x)

        # Gammas
        prefac = 5/(8*np.pi)
        if pol:
            g = prefac*np.array([tr_Ap*tr_Bp+tr_Ax*tr_Bx, # I
                                 tr_Ap*tr_Bp-tr_Ax*tr_Bx, # Q
                                 tr_Ap*tr_Bx+tr_Ax*tr_Bp, # U
                                 1j*(tr_Ap*tr_Bx-tr_Ax*tr_Bp)]) # V
        else:
            g = prefac*(tr_Ap*tr_Bp+tr_Ax*tr_Bx)

        return g

    def get_gamma(self, t, f, theta, phi, pol=False):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_gamma(t, f, ct, st, cp, sp, pol=pol))

    def plot_gamma(self, t, f, n_theta=100, n_phi=100):
        from mpl_toolkits.mplot3d import Axes3D
        phi = np.linspace(0, np.pi, n_phi)
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi, theta = np.meshgrid(phi, theta)
        gamma = np.fabs(self.get_gamma(t, f,
                                       theta.flatten(),
                                       phi.flatten()).reshape([n_theta, n_phi]))
        x = gamma * np.sin(phi) * np.cos(theta)
        y = gamma * np.sin(phi) * np.sin(theta)
        z = gamma * np.cos(phi)
        gmax, gmin = gamma.max(), gamma.min()
        fcolors = (gamma - gmin)/(gmax - gmin)
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.det_A.name+" "+self.det_B.name)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
        ax.set_axis_off()

    def get_G_ell(self, t, f, nside):
        t_use = np.atleast_1d(t)
        f_use = np.atleast_1d(f)

        nf = len(f_use)
        nt = len(t_use)
        npix = hp.nside2npix(nside)
        nell = 3*nside

        th,ph=hp.pix2ang(nside,np.arange(npix))
        ct, st, cp, sp = self._precompute_skyvec(th,ph)

        # [nt, npix]
        gamma = self._get_gamma(t, f, ct, st, cp, sp)
        # [nt, npix]
        bn = self._get_baseline_product(t,ct,st,cp,sp)

        s_A = self.det_A.psd(f_use)
        s_B = self.det_B.psd(f_use)
        e_f = (f_use / self.f_pivot)**self.spectral_index_omega / self.norm_pivot()
        pre_A = 8 * np.pi * e_f / (5 * s_A)
        pre_B = 8 * np.pi * e_f / (5 * s_B)

        gls = np.zeros([nf, nt, nell])
        for i_t, time in enumerate(t_use):
            b = bn[i_t, :]
            g = gamma[i_t, :]
            for i_f, freq in enumerate(f_use):
                phase = 2 * np.pi * freq * b / self.clight
                # Power spectrum of the real part
                g_r = hp.anafast(g * np.cos(phase))
                # Power spectrum of the imaginary part
                g_i = hp.anafast(g * np.sin(phase))
                gls[i_f, i_t, :] = (g_r + g_i) * pre_A[i_f] * pre_B[i_f]

        return np.squeeze(gls)
