'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
### Written by Indraneel ###

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
import copy


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map, reso, size_x, size_y):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        self.occ_map = occupancy_map
        self.occ_res = reso
        self.occ_size_x = size_x
        self.occ_size_y = size_y
    
    def check_if_free(self,x, y):
        # Check if in free space
        map_obj = self.occ_map
        x_ind = int(x/self.occ_res)
        y_ind = int(y/self.occ_res)

        if(x_ind<0 or y_ind<0 or x_ind >=self.occ_size_x or y_ind>=self.occ_size_y):
            return False
        # assert(y_ind>=0)
        # assert(x_ind<self.occ_size_x)
        # assert(y_ind<self.occ_size_y)

        if(map_obj[y_ind,x_ind]<self._min_probability and map_obj[y_ind,x_ind]>-0.1):
            return True
        else:
            return False

    def raycasting(self,x_t1):
        true_dist_array = []
        # robot to laser frame
        laser_x_t1 = [x_t1[0]+25*math.cos(x_t1[2]), x_t1[1]+25*math.sin(x_t1[2]), x_t1[2]]

#       *STARTING FROM THE RIGHT AND GOING LEFT*  Just like angles,
#        the laser readings are in counterclockwise order.
        ninety_radians = 3.14/2
        one_degree_radian = 0.0174

        for i in range(180):

            ## Use raycasting and find z_t^k*
            laser_ray_t1 = copy.deepcopy(laser_x_t1)
            delta_theta = (ninety_radians-one_degree_radian*i)
            laser_ray_t1[2] += delta_theta
            true_dist = 0
            while(self.check_if_free(laser_ray_t1[0],laser_ray_t1[1])):
                true_dist += 10

                # Update coordinates
                laser_ray_t1[0] += 10*math.cos(laser_ray_t1[2]) 
                laser_ray_t1[1] += 10*math.sin(laser_ray_t1[2]) 

                if(true_dist>self._max_range):
                    break
            true_dist_array.append([delta_theta,true_dist])
        return true_dist_array

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 1.0
        

        # Raycasting
        true_dist_array = self.raycasting(x_t1)

        # print(len(z_t1_arr))
        for i in range(len(z_t1_arr)):

            # Subsample sensor measurement
            if i%self._subsampling!=0:
                continue

            true_dist = true_dist_array[i][1]

            # P(correct range)
            p_hit = 0
            if(z_t1_arr[i]>=0 and z_t1_arr[i]<=self._max_range):
                p_hit = np.random.normal(true_dist, self._sigma_hit)

            # P(unexpected)
            p_short = 0
            if(z_t1_arr[i]>=0 and z_t1_arr[i]<=true_dist):
                p_short = self._lambda_short*np.exp(-self._lambda_short*z_t1_arr[i])

            # P(failures)
            p_max = 0
            if(z_t1_arr[i]==self._max_range):
                p_max = 1

            # P(random)
            p_rand = 0
            if(z_t1_arr[i]>=0 and z_t1_arr[i]<self._max_range):
                p_rand = 1/self._max_range

            p_ray =  self._z_hit*p_hit + self._z_short*p_short + self._z_max*p_max + self._z_rand*p_rand
            prob_zt1 = prob_zt1*p_ray


        return prob_zt1, true_dist_array