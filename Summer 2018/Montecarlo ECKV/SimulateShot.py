# -*- coding: utf-8 -*-
"""------------------------------------------------------###
#   SimulateShot.py
#
#   Main file for Montecarlo lidar TPU simulation using the ECKV spectrum
#
#   Authors:
#   Firat Eren, Phd.
#   Brian Calder, Phd.
#   Timothy Kammerer
###------------------------------------------------------"""

from Utility import TateBryanPY
import numpy as np
from numpy import matlib

"""
 * SimulateShot(depth, attenuation_coeff, surface_spectrum, AlgConst)
 * 
 * Simulate a single shot of the lidar, generating and then tracing the rays
 * through the water to the specified _depth_.  This returns the mean position
 * of the simulated rays as a 3x1 (x, y, z)' in _mean_pos_.
"""
def SimulateShot(depth, attenuation_coeff, surface_spectrum, AlgConst):
    scattered_ray_length = -(1/attenuation_coeff*np.log(np.random.rand(AlgConst["Nrays"], AlgConst["MaxScatterEvents"])))   # path length for each ray
    
    rays_albedo = np.random.rand(AlgConst["Nrays"], 1)      # random number for single scattering albedo
    
    is_scattered = np.transpose(rays_albedo <= AlgConst["wo"])[0]          # if the random number is lower than wo, it is a scattering event.
    
    n_rays_scattered = np.sum(is_scattered)
    
    # This is the azimuth angle that determines the plane of scattering.
    scat_anglez = 360*np.random.rand(n_rays_scattered, AlgConst["MaxScatterEvents"])
    scat_angley = np.rad2deg(np.arccos(AlgConst["hg_const_1"] + AlgConst["hg_const_2"]/(1 + AlgConst["g_pf"]*(1.0-2.0*np.random.rand(n_rays_scattered, AlgConst["MaxScatterEvents"])))))
    
    # Sub-set out the position and scattered ray lengths for
    # only those rays that actually undergo scattering
    
    x = AlgConst["x_rays"][is_scattered]
    y = AlgConst["y_rays"][is_scattered]
    scattered_ray_length = scattered_ray_length[is_scattered]
    
    # Face normal vectors from the modeled water surface
    fn1 = surface_spectrum.Sample(0)
    rand_select1 = np.random.randint(0, np.size(fn1, 0)-1)
    face_norm = fn1[rand_select1, :]        # we use one water surface
    
    RES = np.array([x, y, AlgConst["water_elevation"]*np.ones(np.size(x))])     # This is the on-water vector.
    
    VECT = np.transpose([AlgConst["laser_location"][0] - x,
                     AlgConst["laser_location"][1] - y,
                     AlgConst["laser_location"][2] - RES[2]])
    
    uv = VECT / np.transpose(matlib.repmat(np.sqrt(np.sum(VECT*VECT, 1)), 3, 1))
    face_norm1 = np.matlib.repmat(face_norm, np.size(uv, 0), 1)
    
    c1 = np.cross(uv, face_norm1, 1)
    b1 = np.sqrt(np.sum(np.power(np.abs(c1), 2), 1))
    dd = np.dot(uv, np.transpose(face_norm1))
    dd = np.transpose(dd)[0]
    
    # The following equations are derived from the Stanford paper
    # Refraction
    Theta_degrees = np.rad2deg(np.arctan2(b1, dd))
    Refract_ang = np.rad2deg(np.arcsin(np.sqrt(np.power(AlgConst["air_refraction_index"]/AlgConst["water_refraction_index"], 2)*(1-np.power(np.cos(np.deg2rad(Theta_degrees)), 2)))))
    refr_vec = (AlgConst["air_refraction_index"]/AlgConst["water_refraction_index"])*uv - np.matmul(np.transpose([(AlgConst["air_refraction_index"]/AlgConst["water_refraction_index"])*np.cos(np.deg2rad(Theta_degrees)) - np.sqrt(1 - np.power(np.sin(np.deg2rad(Refract_ang)), 2))]), np.array([face_norm]))
    refr_norm = np.sqrt(np.sum(np.power(np.abs(refr_vec), 2), 1))
    refr_vec = refr_vec/np.transpose(np.matlib.repmat(refr_norm, 3, 1))
    
    num_rays = np.size(x, 0)
    
    upd_pos = np.zeros([AlgConst["MaxScatterEvents"], 3])
    xcalc = np.zeros(num_rays)
    ycalc = np.zeros(num_rays)
    zcalc = np.zeros(num_rays)
    
    ray_scalar = np.cos(np.deg2rad(AlgConst["scan_angle"]/AlgConst["water_refraction_index"]))
    for ray in range(num_rays):         # For each ray
        tk = refr_vec[ray]
        posf = np.array([x[ray], 0, AlgConst["water_elevation"]])       # position on the water surface
        posf_in = posf
        for scatter_event in range(AlgConst["MaxScatterEvents"]):       # For each scattering event
            ff = TateBryanPY(scat_angley[ray][scatter_event], scat_anglez[ray][scatter_event])
            tn = np.matmul(tk, ff)      # direction vector
            upd_pos[scatter_event] = posf - scattered_ray_length[ray][scatter_event]*tn     # position of the laser after each scattering event
            posf = upd_pos[scatter_event]       # store the updated laser position as posf and use it in the next iteration
            tk = tn         # update the direction vector
            # Geometrical mean solution
            if posf[2] <= depth:            # When the last scattered location is deeper than the set depth. use the geometrical mean to calculate the laser position.
                dz1 = posf[2] - depth       # difference between actual depth and the last scattered position within the set depth
                if scatter_event > 1:
                    dz2 = depth - upd_pos[scatter_event - 1][2]     # difference between actual depth and the scattered position that exceeds the set depth
                    upd_pos[scatter_event] = (posf*dz2 + upd_pos[scatter_event-1]*dz1)/(dz1 + dz2)
                else:       # in the case of no scattering and the refraction results in deeper than actual depth.
                    dz2 = depth - posf_in[2]
                    upd_pos[scatter_event] = (posf*dz2 + posf_in*dz1)/(dz1 + dz2)
                
                # The following lists the laser locations of each scattered
                # event in descending depth values.
                if scatter_event != 0:
                    r1 = np.sum(np.sqrt(np.sum(np.power(upd_pos[1:scatter_event] - upd_pos[0:scatter_event-1], 2), 1)))
                    + np.sqrt(np.sum(np.power(upd_pos[0]-posf_in, 2)))
                else:
                    r1 = np.sqrt(np.sum(np.power(upd_pos[0]-posf_in, 2)))
                xcalc[ray] = upd_pos[scatter_event][0]
                ycalc[ray] = upd_pos[scatter_event][1]
                zcalc[ray] = r1*ray_scalar
                break
    
    xcalc = xcalc[xcalc != 0]
    ycalc = ycalc[ycalc != 0]
    zcalc = zcalc[zcalc != 0]
    
    mean_pos = np.array([np.mean(xcalc), np.mean(ycalc), np.mean(zcalc)])
    
    return mean_pos