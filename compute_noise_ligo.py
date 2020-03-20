import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Detector(object):
    def __init__(self):
        raise NotImplementedError("Don't use the bare class")

    def get_position(self, t):
        raise NotImplementedError("get_position not implemented")

    def get_a(self, t):
        raise NotImplementedError("get_a not implemented")

    def read_psd(self, fname):
        from scipy.interpolate import interp1d
        nu, fnu = np.loadtxt(fname, unpack=True)
        self.lpsdf = interp1d(np.log(nu),np.log(fnu),
                              bounds_error=False,
                              fill_value=15)

    def psd(self, nu):
        return np.exp(2*self.lpsdf(np.log(nu)))

    
class GroundDetector(Detector):
    rot_freq_earth = 2*np.pi/(24*3600)
    earth_radius = 6.371E6 # in meters

    def __init__(self, lat, lon, alpha, fname_psd, aperture=90):
        # Translate between Renzini's alpha and mine
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

    def get_a(self, t):
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
        # [3, 3, nt]
        a = (x[:,None,...]*x[None,:,...]-y[:,None,...]*y[None,:,...])*0.5
        return a


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
        # [nt, npix]
        tr_Ap = np.einsum('ijk,jil',a_A, e_p)
        tr_Bp = np.einsum('ijk,jil',a_B, e_p)
        tr_Ax = np.einsum('ijk,jil',a_A, e_x)
        tr_Bx = np.einsum('ijk,jil',a_B, e_x)

        # Gammas
        prefac = 5/(16*np.pi)
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

        s_A = self.det_A.psd(f_use)
        s_B = self.det_B.psd(f_use)
        e_f = (f_use / self.f_pivot)**self.spectral_index_omega / self.norm_pivot()
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


dets = {'Hanford':     GroundDetector(46.4, -119.4, 171.8, 'data/curves_May_2019/aligo_design.txt'),  # are these per-detector PSDs?
        'Livingstone': GroundDetector(30.7,  -90.8, 243.0, 'data/curves_May_2019/aligo_design.txt'),  # are these per-detector PSDs?
        'VIRGO':       GroundDetector(43.6,   10.5, 116.5, 'data/curves_May_2019/advirgo_sqz.txt'),
        'Kagra':       GroundDetector(36.3,  137.2, 225.0, 'data/curves_May_2019/kagra_sqz.txt'),
        'GEO600':      GroundDetector(48.0,    9.8,  68.8, 'data/curves_May_2019/o1.txt')}

mcals = {s1: {s2: MapCalculator(d1, d2)
              for s2, d2 in dets.items()}
         for s1, d1 in dets.items()}

nside=64
theta, phi = hp.pix2ang(nside,np.arange(hp.nside2npix(nside)))

for s,d in dets.items():
    print(s, np.sum(mcals[s][s].get_gamma(0, theta, phi)*4*np.pi/hp.nside2npix(nside)))

hp.mollview(mcals['Hanford']['Hanford'].get_gamma(0, theta, phi), coord=['C','G'],
            title=r"$\gamma^I(\theta,\varphi)$", notext=True)
plt.savefig("overlap_HH.pdf", bbox_inches='tight')
hp.mollview(mcals['Hanford']['Livingstone'].get_gamma(0, theta, phi), coord=['C','G'],
            title=r"$\gamma^I(\theta,\varphi)$", notext=True)
plt.savefig("overlap_HL.pdf", bbox_inches='tight')
hp.mollview(mcals['Livingstone']['Livingstone'].get_gamma(0, theta, phi), coord=['C','G'],
            title=r"$\gamma^I(\theta,\varphi)$", notext=True)
plt.savefig("overlap_LL.pdf", bbox_inches='tight')

plt.figure()
obs_time = 1000*365*24*3600.
nl = np.zeros(3*nside)
freqs = np.linspace(10., 1010., 101)
dfreq = np.mean(np.diff(freqs))
for f in freqs:
    print(f)
    n = mcals['Hanford']['Livingstone'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Hanford']['VIRGO'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Hanford']['Kagra'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Hanford']['GEO600'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Livingstone']['VIRGO'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Livingstone']['Kagra'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Livingstone']['GEO600'].get_G_ell(0, f, nside) * dfreq
    n += mcals['VIRGO']['Kagra'].get_G_ell(0, f, nside) * dfreq
    n += mcals['VIRGO']['GEO600'].get_G_ell(0, f, nside) * dfreq
    n += mcals['Kagra']['GEO600'].get_G_ell(0, f, nside) * dfreq
    nl += n
nl *= obs_time
ls = np.arange(len(nl))
nl = 1./nl
plt.plot(ls, ls * (ls + 1.) * nl / (2 * np.pi), 'k--')
plt.plot(ls, 4E-26 * ls ** (5. / 6.), 'r-')
plt.xlabel('$\\ell$', fontsize=16)
plt.ylabel('$N_\\ell$', fontsize=16)
plt.loglog()
plt.show()
