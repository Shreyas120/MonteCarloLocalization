'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import math
import copy

class MapReader:
    def __init__(self, src_path_map):

        self._occupancy_map = np.genfromtxt(src_path_map, skip_header=7)
        self._occupancy_map[self._occupancy_map < 0] = -1
        # The raw data stores P(free) the probability a cell is freespace
        # Convert to P(occupancy) by 1-P(free)
        self._occupancy_map[self._occupancy_map > 0] = 1 - self._occupancy_map[
            self._occupancy_map > 0]
        self._occupancy_map = np.flipud(self._occupancy_map)

        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        self._size_x = self._occupancy_map.shape[0] * self._resolution
        self._size_y = self._occupancy_map.shape[1] * self._resolution

        print('Finished reading 2D map of size: ({}, {})'.format(self._size_x, self._size_y))

    def visualize_map(self, X_bar, rays=None):
        fig = plt.figure()
        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        x_locs = X_bar[:, 0] / 10.0
        y_locs = X_bar[:, 1] / 10.0
        scat = plt.scatter(x_locs, y_locs, c='r', marker='.')

        if rays!=None:
            # Draw rays
            x_pos = X_bar[0]
            # robot to laser frame
            x_laser = x_pos[0]+25.0*math.cos(x_pos[2])
            y_laser = x_pos[1]+25.0*math.sin(x_pos[2])
            theta_laser = x_pos[2]
            laser_x = [x_laser, y_laser, theta_laser]
            
            x_values =[]
            y_values =[]
            for r in range(len(rays)):
                r = rays[r]
                #print(' {} {}'.format(r[0],r[1]))
                ray_x = copy.deepcopy(laser_x)
                ray_x[2] = ray_x[2] + r[0]

                # get end point
                ray_end_x = ray_x[0] + r[1]*math.cos(ray_x[2])
                ray_end_y = ray_x[1] + r[1]*math.sin(ray_x[2])

                x_values.append([ray_x[0]/10.0,ray_end_x/10.0])
                y_values.append([ray_x[1]/10.0,ray_end_y/10.0])
            
            #print(x_values)
            plt.plot(np.array(x_values).T,np.array(y_values).T,'red',linewidth=0.5)


        plt.ion()
        plt.imshow(self._occupancy_map, cmap='Greys')
        plt.axis([0, 800, 0, 800])
        plt.draw()
        plt.pause(0)
        input("Hit enter to close")

    def get_map(self):
        return self._occupancy_map

    def get_map_size_x(self):  # in cm
        return self._size_x

    def get_map_size_y(self):  # in cm
        return self._size_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    map1 = MapReader(src_path_map)
    map1.visualize_map()
