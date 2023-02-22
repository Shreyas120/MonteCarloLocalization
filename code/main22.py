
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
import math

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='/home/shreyas/Desktop/SLAM/hw1/data/map/wean.dat')
    parser.add_argument('--data_log', default='1', type=str)
    parser.add_argument('--dead_reck', default=False, type=bool)
    parser.add_argument('--no_noise', default=False, type=bool)
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--step_viz', default=True, type=bool)
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log =  '/home/shreyas/Desktop/SLAM/hw1/data/log/robotdata' + args.data_log + '.log'
    

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() # np array with shape 800,800
    #Each grid has a resolution of 10cm, so we cover 2D area of 80m x 80m 

    logfile = open(src_path_log, 'r')

    motion_model = MotionModel(args.no_noise)
    #motion model tuneable params alpha 1->4 (i.e. common for all particles)
    #update method returns new belief of state (x,y,theta) for each particle
    sensor_model = SensorModel(occupancy_map,map_obj._resolution,map_obj._size_x,map_obj._size_y)
    resampler = Resampling()
    
    num_particles = args.num_particles

    # X_bar = init_particles_fixed_location(num_particles)
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    # map_obj.visualize_map(X_bar, "Initial particle locations")
    dead_reckon = np.zeros((count_lines(src_path_log),3))
    
    if args.step_viz:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(occupancy_map, cmap='Greys')
        x = (X_bar[:, 0] / 10.0).tolist()
        y = (X_bar[:, 1] / 10.0).tolist()
        sp, = ax.plot(x,y,label='toto',ms=1,color='r',marker='o',ls='')  
    
    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        meas_type = line[0]
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]
        
        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        ################################ MOTION MODEL ###################################
        X_bar[:,:-1] = motion_model.vecUpdate(u_t0,u_t1,X_bar[:,:-1]) # Vectorized implementation
        # for m in range(0, num_particles):
        #     x_t0 = X_bar[m, 0:3]
        #     x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        #     X_bar[m,:-1] = x_t1
        ##########################################################################
        
        for m in range(0, num_particles):
        
            x_t1 = X_bar[m,0:3]

            """
            SENSOR MODEL
            """
            # if (meas_type == "L"):
            #     z_t = ranges
            #     w_t,_ = sensor_model.beam_range_finder_model(z_t, x_t1)
            #     X_bar[m, -1] = w_t
        
        print("Processing time step {} at time {}s".format(time_idx, time_stamp))
        
        u_t0 = u_t1
        X_bar = resampler.low_variance_sampler(X_bar)

        if  args.dead_reck:
            dead_reckon[time_idx,:] = [np.mean(X_bar[:, 0] / 10.0), np.mean(X_bar[:, 1] / 10.0), time_idx]
            
        if args.step_viz and time_idx%10==0:
            x_locs = X_bar[:, 0] / 10.0
            y_locs = X_bar[:, 1] / 10.0
            sp.set_data(x_locs,y_locs)
            plt.title("Processed {:.2f}%, Time {:.2f}s , change in x {}, change in y {}".format(time_idx*100/2218.0, time_stamp, u_t1[0] - u_t0[0],  u_t0[1] - u_t1[1]))
            fig.canvas.draw()
            fig.canvas.flush_events()
            
    

    if args.dead_reck:
        # plt.imshow(occupancy_map, cmap='Greys')
        plt.scatter(dead_reckon[1:,1], dead_reckon[1:,0], s=0.2, c= dead_reckon[1:,2], cmap='Reds')
        plt.title('Dead reckon signal ')
        # Add a colorbar to show the intensity scale
        cbar = plt.colorbar()
        cbar.set_label('Intensity')
        plt.show()


    if args.no_noise:
        print('End pos ', dead_reckon[-1,1] , " \t ", dead_reckon[-1,0])

    map_obj.visualize_map(X_bar, "Final particle locations")
