import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Detector(object):
    def __init__(self, name):
        self.name = name
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
