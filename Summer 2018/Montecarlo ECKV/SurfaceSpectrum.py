# -*- coding: utf-8 -*-
"""------------------------------------------------------###
#   SurfaceSpectrum.py
#
#   Main file for Montecarlo lidar TPU simulation using the ECKV spectrum
#
#   Authors:
#   Firat Eren, Phd.
#   Brian Calder, Phd.
#   Timothy Kammerer
###------------------------------------------------------"""

from Utility import ECKV_spectrum, Cos2S_spread, faceNormal, frange
from math import pi
import numpy as np
from scipy.spatial import Delaunay as delaunayTriangulation

class SurfaceSpectrum:
    Lx = 20.0   # Length of spatial domain in x (m)
    Ly = 20.0   # Length of spatial domain in y (m)
    Nx = 64     # Number of samples in spatial domain in x
    Ny = 64     # Number of samples in spatial domain in y
    g = 9.82    # Gravitational constant (?)
    
    def __init__(self, wave_age):
        # The wave age is only used when the 1D spectrum is generated
        # (in SetWind()), so we just cache this for now.
        self.wave_age = wave_age    # Non-dimensional wave age parameter for spectrum
        
        
        dx = self.Lx/self.Nx        # Spatial resolution in x and y-axisw
        dy = self.Ly/self.Ny
        
        vfx = 1/self.Lx             # fundamental frequencies of x-axis
        self.kfx = 2*pi*vfx         # spatial frequency in x axis
        
        vfy = 1/self.Ly             # fundamental frequencies of y-axis
        self.kfy = 2*pi*vfy         # spatial frequency in y axis
        
        NyxL = 2*dx                 # minimum wavelength
        
        VNyx = 1/NyxL               # Nyquist frequency in x-axis
        kNyx = 2*pi*VNyx            # angular Nyquist frwquency in x-axis
        
        r = np.array(range(self.Nx))        # index number for x-axis
        s = np.array(range(self.Ny))        # index number for y-axis
        
        RES = np.array([0, 0])
        xr = r*dx + RES[0] - self.Lx/2      # Spatial index - length and width
        ys = s*dy - RES[1] - self.Ly/2      # Depends on the location of the footprint
        
        u = np.array(range(-(self.Nx/2-1), self.Nx/2+1))
        v = np.array(range(-(self.Ny/2-1), self.Ny/2+1))
        
        # Spatial variables in Math order
        kxu = u*self.kfx
        kyv = v*self.kfy
        
        self.k = frange(0, kNyx + self.kfx/2, self.kfx)
        ang = np.linspace(-pi/2, pi/2, self.Nx)
        
        s = 6                               # MAGIC NUMBER - REASON UNKNOWN    #TODO: find out why
        self.SF = Cos2S_spread(ang, s)      # Spreading function
        
        KX = np.meshgrid(kxu, kyv)[0]
        
        self.scaled_frequency_bins = np.lib.scimath.sqrt(np.array(self.g*KX))
        [self.X, self.Y] = np.meshgrid(xr, ys)
        
        self.domain_triangulation = delaunayTriangulation(np.transpose([np.reshape(self.X, np.size(self.X), 1),
                                                          np.reshape(self.Y, np.size(self.Y), 1)]))
                                            # Mark that the spectrum has yet to be generated
        
        self.wave_spectrum = np.array([])             # Mark the the spectrum has yet to be generated
    
    def SetWind(self, wind):
        # Set the current wind conditions (in knots)
        
        S = ECKV_spectrum (self.k, wind, self.wave_age)[0]
        
        # Omnidirectional 1D Pierson-Moskowitz wave spectrum
        S[0] = 0        # Reset mean value (D.C. term) to zero
        self.wave_spectrum = np.sqrt(np.matmul(np.transpose(np.array([self.SF])),
                                               np.array([S]))*self.kfx*self.kfy/2)
    
    def Sample(self, t):
        # Sample the surface, at time t, returning triangulated face
        # normal vectors for the surface
        if(np.size(self.wave_spectrum) == 0):
            print "That object does not have a wave spectrum defined."
            return
        
        # The phase information is generated as a Argand pair of normal
        # deviates, rather than doing a uniform phase and trying to
        # remap to Argand outputs.
        rho = np.random.randn(np.size(self.wave_spectrum, 0), np.size(self.wave_spectrum, 1))
        sig = np.random.randn(np.size(self.wave_spectrum, 0), np.size(self.wave_spectrum, 1))
        
        zhat = np.zeros(np.shape(self.wave_spectrum))
        
        t_c = np.sqrt(t)
        cos_f = np.cos(t_c*self.scaled_frequency_bins)
        sin_f = np.sin(t_c*self.scaled_frequency_bins)
        
        for i in range(0, np.size(self.wave_spectrum, 0)):
            for j in range(0, np.size(self.wave_spectrum, 1)):
                zhat[i][j] = ((rho[i][j]*self.wave_spectrum[i][j]*cos_f[i][j] - sig[i][j]*self.wave_spectrum[i][j]*sin_f[i][j]) +
                    1j*(rho[i][j]*self.wave_spectrum[i][j]*sin_f[i][j] + sig[i][j]*self.wave_spectrum[i][j]*cos_f[i][j]))
        
        # zhat[0][0] = complex(0, 0)
        
        zhat2 = np.fliplr(np.conj(zhat))
        zhat2 = zhat2[:,1:-1]
        zhat2 = np.flipud(zhat2)
        zhat = np.concatenate([zhat2, zhat], 1)
        
        zhat_ifft = np.fft.ifftshift(zhat)
        Z = np.real(np.fft.ifft2(zhat_ifft))    #FIXME: check if this is the same as matlab w/ "symmetric"
        
        surf_el = self.Nx*self.Ny*Z
        global tri_connect, tri_points
        tri_points = np.concatenate((self.domain_triangulation.points, np.reshape(surf_el, (np.size(surf_el), 1))), 1)
        tri_connect = self.domain_triangulation.simplices
        
        face_normals = faceNormal(tri_points, tri_connect)
        return face_normals