# -*- coding: utf-8 -*-
"""------------------------------------------------------###
#   Lidar_TPU_MC_subaqueous.py
#
#   Main file for Montecarlo lidar TPU simulation using the ECKV spectrum
#
#   Authors:
#   Firat Eren, Phd.
#   Brian Calder, Phd.
#   Timothy Kammerer
###------------------------------------------------------"""

import numpy as np
from Utility import generateLidarRays, FormatTimeString, frange
from SurfaceSpectrum import SurfaceSpectrum
from SimulateShot import SimulateShot
import time

#-------------------------------------------------------------------------
# User modifiable paramters
#
# Some of these parameters are passed to the core simulation function, and
# therefore need to be gathered together into a structure that can be
# passed more readily.  In order to avoid special cases, they are all
# collected in sub-structures.
#-------------------------------------------------------------------------
def userParameters():
    
    AlgConst = dict()
    
    # Properties of the simulation
    AlgConst["Nrays"] = 1000                            # number of rays used in the MC simulations
    AlgConst["MaxScatterEvents"] = 20                   # Maximum number of refraction layers to consider
    AlgConst["Nsim"] = 2000                             # Number of MC simulations
    AlgConst["wind_spread"] = frange(1,10);              # Wind speeds to simulate, in knots
    AlgConst["Kd_spread"] = frange(0.06, 0.36, 0.01)     # Absorption coefficient range to simulate (unitless)
    
    # Scatter parameters: probability (albedo), and phase function
    AlgConst["wo"] = 0.80               # single scattering albedo
    AlgConst["g_pf"] = 0.995            # Henyey-Greenstein phase function forward scattering parameter
    
    # Properties of the environment: refractions, wave height, area depth
    AlgConst["air_refraction_index"] = 1            # index of refraction in air
    AlgConst["water_refraction_index"] = 1.33       # index of refraction in water 
    AlgConst["wave_age"] = 3.5          # wave age used in the ECKV modeled water surface
    AlgConst["dshallow"] = -1           # Shallowest depth
    AlgConst["ddeep"] = -10             # Deepest depth
    AlgConst["water_elevation"] = 0     # Water surface elevation (metres)
    
    # Properties of the Lidar doing the survey
    AlgConst["scan_angle"] = 20         # scanning angle of Riegl VQ-880-G (degrees)
    AlgConst["beam_div"] = 1            # half beam divergence angle in mrad. Between 0.7-2 mrad as defined in Riegl Vq-880G document
    AlgConst["PG"] = np.array([0, 0, 0])          # This is the distance vector from the airplane rotation sensor, i.e. IMU. to the laser unit (level arm offset vector).
    AlgConst["dkap"] = 0                # Sensor boresight angle (degrees)
    AlgConst["dpsi"] = 0                # Sensor boresight angle (degrees)
    AlgConst["domeg"] = 0               # Sensor boresight angle (degrees)
    
    # Current attitude and position of sensor during simulation
    AlgConst["laser_location"] = np.array([0, 0, 600])    # Laser location (x, y, z) (metres)
    AlgConst["roll"] = 0                        # Current sensor roll during simulation (degrees)
    AlgConst["pitch"] = 0                       # Current sensor pitch during simulation (degrees)
    AlgConst["yaw"] = 0                         # Current sensor yaw during simulation (degrees)
    
    #-------------- Preliminary Computations --------------#
    # Intermediate computations on the constants to get them into the right
    # form for the rest of the computation
    
    AlgConst["hg_const_1"] = (1 + AlgConst["g_pf"]*AlgConst["g_pf"])/(2*AlgConst["g_pf"])       # Pre-computes part of the scattering computation
    AlgConst["hg_const_2"] = (-1 + AlgConst["g_pf"]*AlgConst["g_pf"])/(2*AlgConst["g_pf"])      # Pre-computes part of the scattering computation
    
    AlgConst["drange"] = frange(AlgConst["ddeep"]-0.1, AlgConst["dshallow"]-0.1, 0.1)        # Depth range to be simulated
    AlgConst["beam_div"] = AlgConst["beam_div"] * 0.0572958                         # conversion from mrad to degrees
    
    # We pre-generate a set of AlgConst.Sim.Nrays pseudo-rays within the beam
    # divergence in order to Monte Carlo trace them into the water, and work
    # out how much they're being shifted by the surface interaction and
    # scattering.  The rays are generated in a circle on the water surface
    # according to the divergence.
    generateLidarRays(AlgConst)
    
    return AlgConst

def main():
    AlgConst = userParameters()
    
    #-------------- Output Space pre-allocation --------------#
    # The mean/std. dev. values being generated in the depth loop (i.e., the
    # outputs that we're going to memoise in the LUT) are cached temporarily as
    # the depth loop is computed.  The same number of depth points are done on
    # each pass, so we can do a one-time allocation here
    
    num_depths = len(AlgConst["drange"])
    mean_depth = np.zeros([num_depths])
    std_depth = np.zeros([num_depths])
    mean_x = np.zeros([num_depths])
    std_x = np.zeros([num_depths])
    mean_y = np.zeros([num_depths])
    std_y = np.zeros([num_depths])
    sim_result = np.zeros([AlgConst["Nsim"], 3])
    
    #-------------- Main Simulation Loop --------------#
    # The surface wave spectrum depends only on the wave age and wind for the
    # most part (i.e., to the scale of the 2D spatial spectrum), so we set-up
    # for wave age first, reset to the right wind in the appropriate loop, and
    # sample inside the inner-most loop.  This avoids a lot of re-computation
    # (and particularly triangulation) in the inner loop.
    surface_spectrum = SurfaceSpectrum(AlgConst["wave_age"])
    
    # For effort tracking, we need to know the total number of inner loops that
    # we're going to run, and track the number of depth loops complete (so we
    # can estimate the time to finish the total computation).
    n_depth_loops = len(AlgConst["wind_spread"])*len(AlgConst["Kd_spread"])
    depth_loops_complete = 0
    
    for wind in AlgConst["wind_spread"]:
        # The surface spectrum magnitude is only dependent on the wind speed,
        # so we compute here
        surface_spectrum.SetWind(wind)
        
        wind_prefix = "Wind {} kt".format(wind)
        # For display during the inner (depth) loop
        
        for Kd in AlgConst["Kd_spread"]:
            cb = (Kd - 0.04)/0.2        # Conversion from Kd to beam attenuation coefficient
            
            Kd_prefix = "Kd {}".format(Kd)
            # For display during the inner (depth) loop
            
            di = 0      # index variable to output into mean_{depth,x,y} and std_{depth,x,y} as a function of depth loop
            depth_cumulative_time = 0
            print "Generating LUT for {}, {}.".format(wind_prefix, Kd_prefix)
            for depth in AlgConst["drange"]:
                start_time = time.time()
                for sim in range(AlgConst["Nsim"]):             #TODO: Run for loop in parallel
                    sim_result[sim, :] = SimulateShot(depth, cb, surface_spectrum, AlgConst)
                
                mean_position = np.mean(sim_result, 0)
                std_position = np.std(sim_result, 0)
                
                mean_depth[di] = mean_position[2]
                std_depth[di] = std_position[2]
                
                mean_x[di] = mean_position[1]
                std_x[di] = std_position[1]
                
                mean_y[di] = mean_position[0]
                std_y[di] = std_position[0]
                
                elapsed = time.time() - start_time
                depth_cumulative_time += elapsed
                depth_mean_time = depth_cumulative_time / (di+1)
                
                di+=1
                
                print " ... Computation time for {}, {}, Depth {}m: {}s, mean {}s/depth sample.".format(
                        wind_prefix, Kd_prefix, depth, round(elapsed, 2), round(depth_mean_time, 2))
            depth_loops_complete += 1
            remaining_time = (n_depth_loops - depth_loops_complete)*depth_cumulative_time
            time_string = FormatTimeString(remaining_time)
            print " ... Total time for depth loop {}s (estimate {} to complete remaining simulations).".format(
                        round(depth_cumulative_time, 2), time_string)
            print " ... Saving LUT for {}, {}.".format(wind_prefix, Kd_prefix)
            
            outputFile = open("table_wind_{}_Kd{}_sig_xyz_PF{}_TB_right.csv".format(int(wind), int(Kd*100), AlgConst["g_pf"]), 'w')
            for n in range(num_depths):
                outputFile.write("{}, {}, {}, {}, {}, {}, {}\n".format(AlgConst["drange"][n], mean_x[n], std_x[n], mean_y[n], std_y[n], -mean_depth[n], std_depth))

if __name__ == "__main__":
    main()