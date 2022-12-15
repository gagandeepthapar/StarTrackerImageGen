from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .star import GenericStar

class CelestialSphere:

    pi = np.pi

    def __init__(self):
        self.set_plot_properties()
        self.define_earth_surface()

    def set_plot_properties(self):
        self.plt = plt
        self.fig = self.plt.figure()
        self.ax = self.plt.axes(projection='3d')

        # set window boundaries         
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])

        # plot cleanliness (less clutter with label ticks)
        self.ax.set_box_aspect((1, 1, 1))
        self.plt.locator_params(axis='x', nbins=5)
        self.plt.locator_params(axis='y', nbins=5)
        self.plt.locator_params(axis='z', nbins=5)

        return

    def define_earth_surface(self):

        phi, theta = np.mgrid[0:self.pi:100j, 0:2*self.pi:100j]

        x_r = np.sin(phi)*np.cos(theta)
        y_r = np.sin(phi)*np.sin(theta)
        z_r = np.cos(phi)

        self.ax.plot_surface(x_r, y_r, z_r, alpha=0.1)

        return 

    def plot_multiple_stars(self, starlist:pd.DataFrame=None):
        if starlist is not None:
            for index, star in starlist.iterrows():
                self.plot_single_star(starlist_entry=star)

        return

    def plot_single_star(self, star:GenericStar=None, vector:np.ndarray=None, starlist_entry:pd.Series=None):

        if star:
            ra, d = star.get_right_ascension_declination()
            x,y,z = self.angle_to_xyz(ra, d)

        elif vector:
            x,y,z = vector

        elif starlist_entry is not None:
            ra, d = starlist_entry['right_ascension'], starlist_entry['declination']    
            x,y,z = self.angle_to_xyz(ra, d)

        else:
            x,y,z = np.array([0,1,0])

        self.ax.scatter(x, y, z, marker='^', color='g', alpha=1.0)
        
        return


    def angle_to_xyz(self, ra: float, d: float):

        x = np.sin(ra)*np.cos(d)
        y = np.sin(ra)*np.sin(d)
        z = np.cos(ra)

        return x,y,z

    def plot_boresight(self, vector:np.ndarray):
        self.ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], color='r', label='Boresight')
        self.ax.scatter(vector[0], vector[1], vector[2], marker='^', color='r', label='True Attitude')

    def save_image(self, name:str='test.png'):
        self.ax.legend()
        self.plt.savefig(name)

if __name__ == '__main__':
    plot = CelestialSphere()
    plot.plot_boresight(np.array([1,0,0]))
    plot.plot_single_star()
    plot.save_image()