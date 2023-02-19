'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math
import copy

class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self, dead_reck_no_noise = False):
        """
        The original numbers are for reference but HAVE TO be tuned.
        """ 
        #increases noise in relative rotation only 
        self._alpha1 = 0 #scales relative_rot_1 and relative_rot_2
        self._alpha2 = 0 #scales relative _translation

        #increases noise in relative translation only 
        self._alpha3 = 10 #scales relative _translation
        self._alpha4 = 10 #scales relative_rot_1 and relative_rot_2

        if dead_reck_no_noise:
            self._alpha1, self._alpha2, self._alpha3, self._alpha4 = [0,0,0,0]
        

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]

        """
        #odometry motion model 
        
        ### Get relative change in robot position
        ### x,y,theta in odometry frame 
        ### (bar used to indicate odometry frame measurements)
        x_bar, y_bar, theta_bar = u_t0 
        x_bar_dash, y_bar_dash, theta_bar_dash = u_t1

        del_rot_1 = math.atan2((y_bar_dash - y_bar),(x_bar_dash - x_bar)) - theta_bar
        del_trans = math.sqrt((x_bar - x_bar_dash)**2 + (y_bar -y_bar_dash)**2)
        del_rot_2 = theta_bar_dash - theta_bar - del_rot_1 
        
        # print(del_rot_1, del_trans, del_rot_2)

        del_rot_2 = self.limit_angle(del_rot_2) #restrict rotation between -pi and pi

        del_rot_1_hat = del_rot_1 - self.sample(self._alpha1 * (del_rot_1**2) + self._alpha2 * (del_trans**2))
        del_trans_hat = del_trans - self.sample(self._alpha3 * (del_trans**2) + self._alpha4 * (del_rot_1**2) + self._alpha4 * (del_rot_2**2))
        del_rot_2_hat = del_rot_2 - self.sample(self._alpha1 * (del_rot_2**2) + self._alpha2 * (del_trans**2))

        ### updating robot pose in world frame based on odometry motion model 
        x, y, theta = x_t0 

        x_dash     = x     + del_trans_hat * math.cos(theta + del_rot_1_hat)
        y_dash     = y     + del_trans_hat * math.sin(theta + del_rot_1_hat)
        theta_dash = theta + del_rot_1_hat + del_rot_2_hat
        
        theta_dash = self.limit_angle(theta_dash) #restrict rotation between -pi and pi
        
        x_t1 = [x_dash, y_dash, theta_dash]
        return x_t1

    def sample(self,var):
        """
        Given a variance, sample zero-mean noise 
        Bounded between -1 and 1 ?
        """
        return var* np.sum( np.random.uniform(-1.0,1.0,(12,1)) ) / 12.0
        # return np.random.normal(0.0, math.sqrt(var))

    def limit_angle(self, angle_):
        """
        Truncate angles outside [-pi,pi] 
        Correct wrap around/spill over to avoid potential divergence of PF 
        """
        angle = copy.deepcopy(angle_)
        
        while angle > math.pi:
            angle -= 2*math.pi
        
        while angle < -math.pi:
            angle += 2*math.pi
        
        # if angle!=angle_:
        #     #some correction was made
        #     print("Limiting angle between -pi and pi")
        #     print("{} radians corrected to {} radians".format(angle_, angle))

        return angle 
