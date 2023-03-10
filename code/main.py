'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

# # typical mistakes?

#   --> raycastings
#   -- > pointing in which direction counterclockwise clockwise -- angle of laser pointing in the right direction
#   --> how do you check if you've walked outside the map area

#   --> sensor model --- visualize pdf --
#   if zhit is in certain order of magnitude, orders should Be lesser 

#   zhit 4 --> decrease
#   zshort should be lower than OM 25 --- it should be 2.5 ish<<<
#   zmax 0.5 is approp
#   increase zrand above 50  >>> 500 works
#   explain params in write up 


#     muchhhhh smalllller --2-3  OM smalller karo --> relatively correct
#   a1 0.17
#   a2 0.17
#   a3 10
#   a4 10 

# in sensor model
# value of min probability = 0.35 --same 
# max range offset resolution 

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_fixed_location(num_particles):
    """
    Initialize all particles at the same location to check motion model 
    """
     # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = 3500*np.ones((num_particles, 1))
    x0_vals = 3500*np.ones((num_particles, 1))
    theta0_vals = np.zeros((num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init

def count_lines(path):
    with open(path, 'r') as fp:
        x = len(fp.readlines())
    return x

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    """
    # initialize [x, y, theta] positions in world_frame for all particles
    # use knowledge from occupany grid 
    This version converges faster than init_particles_random
    """

    y0_vals = np.zeros((num_particles, 1))
    x0_vals = np.zeros((num_particles, 1))

    particle = 0 
    while particle < num_particles:
        x0_vals[particle,0] = np.random.uniform(3000, 7000)
        y0_vals[particle,0] = np.random.uniform(0, 7500)
        p_occ = occupancy_map[int(y0_vals[particle,0]/10) , int(x0_vals[particle,0]/10)]

        if p_occ<0.15 and p_occ !=-1: #0.2 is safer
            #accept initialization for this particle
            particle+=1
         
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    # map_obj.visualize_map(X_bar_init[:,:-1])
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='/home/shreyas/Desktop/SLAM/hw1/data/map/wean.dat')
    parser.add_argument('--path_to_log', default='/home/shreyas/Desktop/SLAM/hw1/data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() # np array with shape 800,800
    #Each grid has a resolution of 10cm, so we cover 2D area of 80m x 80m 

    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    #motion model tuneable params alpha 1->4 (i.e. common for all particles)
    #update method returns new belief of state (x,y,theta) for each particle

    sensor_model = SensorModel(occupancy_map)
    ##occupany map gives probability of a 10cm x 10cm grid being occupied (contains 800x800 grids)

    resampler = Resampling()

    num_particles = args.num_particles


    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        # print("Processing time step {} at time {}s".format(time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        
        X_bar[:,:-1] = motion_model.vecUpdate(u_t0,u_t1,X_bar[:,:-1]) # Vectorized implementation

        for m in range(0, num_particles):
        
            x_t1 = X_bar[m,:-1]
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
              X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)

