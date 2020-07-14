# -*- coding: utf-8 -*-
"""------------------------------------------------------#
#   Utility.py
#
#   Utility functions for lidar TPU Montecarlo simulations
#
#   Authors:
#   Firat Eren, Phd.
#   Brian Calder, Phd.
#   Timothy Kammerer
#------------------------------------------------------"""

import numpy as np
from math import pi
from scipy.special import gamma

"""
 * ECKV_spectrum(k, U_10, Omega_c)
 * 
 * Pierson-Moskowitz omnidirectional wave spectrum (continuous)
 * Function uses angular spatial frequency, k.
"""
def ECKV_spectrum(k, U_10, Omega_c):
    alpha = 0.0081
    beta = 1.25
    g = 9.82
    
    Cd = 0.00144 # drag coefficient
    u_star = np.sqrt(Cd)*U_10
    
    km = 370
    cm = 0.23
    am = 0.13*u_star/cm
    if Omega_c <= 1:
        gamm = 1.7
    else:
        gamm = 1.7 + 6*np.log10(Omega_c)
    
    sigma = 0.08*(1+4*(pow(Omega_c, -3)))
    alpha_p = 0.006*pow(Omega_c, 0.55)
    if u_star <= cm:
        alpha_m = 0.01*(1+np.log(u_star/cm))
    else:
        alpha_m = 0.01*(1+3*np.log(u_star/cm))
    
    ko = g/(U_10*U_10)
    kp = ko*Omega_c*Omega_c
    cp = np.sqrt(g/kp)
    
    c = np.sqrt((g/k)*(1+np.power((k/km), 2)))
    
    L_PM = np.exp(-1.25*np.power((kp/k), 2))
    Gam = np.exp((-1/(2*sigma*sigma))*np.power((np.sqrt(np.abs(k)/kp)-1), 2))
    Jp = np.power(gamm, Gam)
    Fm = L_PM*Jp*np.exp(-0.25*np.power(k/km-1, 2))
    Fp = L_PM*Jp*np.exp(-0.3162*Omega_c*(np.sqrt(np.abs(k)/kp)-1))
    Bl=0.5*alpha_p*(cp/c)*Fp
    Bh=0.5*alpha_m*(cm/c)*Fm
    S = (Bl+Bh)/np.power(k, 3)
    
    return np.array([S, cp, c, cm, am])

"""
 * faceNormal(vertices, connectivity)
 * 
 * Finds the vertical normal of the triangles determined by the vertices and connectivity arrays
"""
def faceNormal(vertices, connectivity):
    output = np.zeros(np.shape(connectivity))
    for n in range(np.size(connectivity, 0)):
        output[n] = tri_faceNormal(vertices[connectivity[n]])
    return output

"""
 * tri_faceNormal(vertices)
 * 
 * Finds the vertical normal of the given triangle
"""
def tri_faceNormal(vertices):
    edge0 = vertices[0] - vertices[1]
    edge1 = vertices[1] - vertices[2]
    output = np.cross(edge0, edge1)
    if(output[2] < 0):
        output = output * -1
    return output

"""
 * Cos2S_spread(angle, spread_param)
 * 
 * Cosine2S spreading function for wave generation.
 * Input to the system is the angular distributions
 * The angle range could be between -pi/2 to pi/2 or
 * -pi to pi. If -pi to pi is chosen, make sure to change the formula to
 * angle to angle/2 
"""
def Cos2S_spread(angle, spread_param):
    Cs = 1/(2*np.sqrt(pi))*gamma(spread_param+1)/gamma(spread_param+0.5)
    return Cs*np.power(np.cos(angle), 2*spread_param)

"""
 * float[3] GenerateLidarRays(float* ray_x_pos, float* ray_y_pos)
 * 
 * This computes, from the algorithm constants in _algConst_, the location
 * of the lidar shot point in 3D, and the locations of the simulation rays
 * within the lidar beam on the surface of the water: represented by ray_x_pos and ray_y_pos.
"""
def generateLidarRays(AlgConst):
    Rypr = TateBryan(AlgConst["roll"], AlgConst["pitch"], AlgConst["yaw"])
    AlgConst["POS_LAS"] = AlgConst["laser_location"] + np.matmul(Rypr, AlgConst["PG"]);
    radius = AlgConst["laser_location"][2]*(np.tan((AlgConst["scan_angle"] + AlgConst["beam_div"] / 2) * pi / 180) -
                  np.tan((AlgConst["scan_angle"] - AlgConst["beam_div"] / 2) * pi / 180))/2
    x0 = AlgConst["laser_location"][2]*np.tan(AlgConst["scan_angle"] * pi / 180)
    y0 = 0
    t = 2*pi*np.random.rand(AlgConst["Nrays"])
    rr = radius*np.sqrt(np.random.rand(AlgConst["Nrays"]))
    AlgConst["x_rays"] = x0 + rr*np.cos(t)
    AlgConst["y_rays"] = y0 + rr*np.sin(t)
    
"""
 * TateBryan(float roll, float pitch, float yaw)
 * 
 * @returns float[3][3]
 * 
 * Compose the total rotation matrix, in Tate-Bryan order, for the given
 * orientation angles
"""
def TateBryan(roll, pitch, yaw):
    cr = np.cos(roll);   sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw);   sy = np.sin(yaw)
    return np.array([[cp*cy, -cr*sy+sr*sp*cy, sr*sy+cr*sp*cy], [cp*sy, cr*cy+sr*sp*sy, -cy*sr+sp*sy], [-sp, sr*cp, cr*cp]])

"""
 * TateBryanPY(float roll, float pitch, float yaw)
 * 
 * @returns float[3][3]
 * 
 * Compose the reduced rotation matrix, in Tate-Bryan order, for the given
 * orientation angles, assuming zero roll
"""
def TateBryanPY(pitch, yaw):
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw);   sy = np.sin(yaw)
    return np.array([[cp*cy, -sy, sp*cy], [cp*sy, cy, sp*sy], [-sp, 0, cp]])

"""
 * frange(float start, float end, float delta)
 * 
 * Create a range of floats starting at start and ending at end stepping by delta
"""
def frange(start, end, delta = 1.0):
    r = np.array(range(int(start/delta+0.1), int(end/delta+1.1)))
    r = r * delta
    return r

"""
 * FormatTimeString(double time)
 * 
 * @returns string
 * 
 * Convert time _t_ into a printable string, taking into account the
 * magnitude (i.e., so < 60 s is printed in seconds, < 1 hr in minutes,
 * etc.).
"""
def FormatTimeString(time):
    if time < 60:
        return str(time) + "s"
    if time < 3600:
        return str(round(time / 60, 2)) + "min"
    if time < 86400:
        return str(round(time / 3600, 2)) + "hours"
    return str(round(time / 86400, 2)) + "days"