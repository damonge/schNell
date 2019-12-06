import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Detector(object):
    rot_freq_earth = 2*np.pi/(24*3600)
    tran_freq_earth = 2*np.pi/(24*3600*365) # approximating circular orbit
    cos_ecliptic = np.cos(np.radians(23.))
    sin_ecliptic = np.sin(np.radians(23.))
    earth_radius = 6.371E6 # in meters

    def __init__(self, lat, lon, alpha):
        # Translate between Renzini's alpha and mine
        self.alpha = alpha - 45.
        self.phi_e = np.radians(lon)
        self.theta_e = np.radians(90-lat)
        self.ct = np.cos(self.theta_e)
        self.st = np.sin(self.theta_e)
        self.ca = np.cos(np.radians(self.alpha))
        self.sa = np.sin(np.radians(self.alpha))
        self.nv_e0 = np.array([self.st*np.cos(self.phi_e),
                               self.st*np.sin(self.phi_e),
                               self.ct])

    def get_position(self, t):
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        return self.earth_radius * np.array([self.st*cp,
                                             self.st*sp,
                                             self.ct*o])

    def get_a(self, t):
        phi = self.phi_e + self.rot_freq_earth * t
        cp = np.cos(phi)
        sp = np.sin(phi)
        o = np.ones_like(t)
        # [3, nt]
        x = np.array([-self.ca*cp*self.ct-self.sa*sp,
                      self.sa*cp-self.ca*sp*self.ct,
                      self.ca*self.st*o])
        # [3, nt]
        y = np.array([self.sa*cp*self.ct-self.ca*sp,
                      self.ca*cp+self.sa*sp*self.ct,
                      -self.sa*self.st*o])
        # [3, 3, nt]
        a = (x[:,None,...]*x[None,:,...]-y[:,None,...]*y[None,:,...])*0.5
        return a

class MapCalculator(object):
    clight = 299792458.

    def __init__(self, det_A, det_B):
        self.det_A = det_A
        self.det_B = det_B

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
        
        bprod = np.einsum('ik,il',x_A-x_B,nv)
        return bprod

    def get_baseline_product(self, t, theta, phi):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_baseline_product(t, ct, st, cp, sp))

    def _get_gamma(self, t, ct, st, cp, sp, pol=False):
        t_use = np.atleast_1d(t)

        # [3, 3, nt]
        a_A = self.det_A.get_a(t_use)
        # [3, 3, nt]
        a_B = self.det_B.get_a(t_use)

        # [3, npix]
        l = np.array([sp,-cp,np.zeros_like(sp)])
        # [3, npix]
        m = np.array([cp*ct,sp*ct,-st])
        # e_+ [3, 3, npix]
        e_p = l[:,None,...]*l[None,:,...]-m[:,None,...]*m[None,:,...]
        # e_x [3, 3, npix]
        e_x = l[:,None,...]*m[None,:,...]+m[:,None,...]*l[None,:,...]

        # Tr[a_A*e_+]
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

    def get_gamma(self, t, theta, phi, pol=False):
        ct, st, cp, sp = self._precompute_skyvec(theta, phi)
        return np.squeeze(self._get_gamma(t, ct, st, cp, sp, pol=pol))

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
        gamma = self._get_gamma(t,ct,st,cp,sp)
        # [nt, npix]
        bn = self._get_baseline_product(t,ct,st,cp,sp)

        # TODO: get power spectral densities
        s_A = np.ones_like(f_use)
        s_B = np.ones_like(f_use)
        # TODO: get frequency dependence
        e_f = np.ones_like(f_use)
        pre_A = 16 * np.pi * e_f / (5 * s_A)
        pre_B = 16 * np.pi * e_f / (5 * s_B)

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
                

dets = {'Hanford':     Detector(46.4, -119.4, 171.8),
        'Livingstone': Detector(30.7,  -90.8, 243.0),
        'VIRGO':       Detector(43.6,   10.5, 116.5),
        'Kagra':       Detector(36.3,  137.2, 225.0),
        'GEO600':      Detector(48.0,    9.8,  68.8)}

mcals = {s1: {s2: MapCalculator(d1, d2) for s2, d2 in dets.items()} for s1, d1 in dets.items()}

nside=64
theta, phi = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)))

plt.figure()
nls = []
for i_f, f in enumerate(np.geomspace(10., 1000., 20)):
    nl = mcals['Hanford']['Livingstone'].get_G_ell(0.,f, nside)
    nls.append(nl)
    plt.plot(1./nl,'-',c=cm.bone((i_f+0.5)/20))
plt.plot(1./np.mean(np.array(nls),axis=0),'k-')
plt.loglog()

hp.mollview(mcals['Hanford']['Livingstone'].get_baseline_product(0.,theta, phi),coord=['C','G'])
hp.mollview(mcals['Hanford']['Livingstone'].get_gamma(0.,theta, phi),coord=['C','G'])
plt.show()
