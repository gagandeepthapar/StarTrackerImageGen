import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import j1

import generateProjection as gP
from python_scripts import constants

util_img = 'UTIL'
sim_img = 'SIM'

def gen_airy(overExpose:int=5, rad:float=10):

    # create meshgrid of spaces 
    r = lambda x,y: np.sqrt(x**2 + y**2)
    x = np.linspace(-1*rad, rad, 1000)
    y = np.linspace(-1*rad, rad, 1000)

    xM, yM = np.meshgrid(x, y)
    d = r(xM, yM)

    # calculate brightness based on Airy Disk
    z_raw = (overExpose+1) * 4 * (j1(d)/d)**2

    # raw intensity values (need to cut around [0, 1])
    for i in range(len(z_raw)):
        for j in range(len(z_raw[i])):
            z_raw[i][j] = max(0, z_raw[i][j])
            z_raw[i][j] = min(1, z_raw[i][j])

    zM = z_raw

    return xM, yM, zM

def plot_star(star:pd.Series, axes, xM, yM, zM):

    axes.pcolormesh(xM + star['IMG_X'], yM + star['IMG_Y'], zM, cmap='gray')

    return

def gen_bkgd(axes, img_wd:int, img_ht:int):

    # create meshgrid
    xB = np.linspace(0, img_wd-1, 10)
    yB = np.linspace(0, img_ht-1, 10)

    xB, yB = np.meshgrid(xB, yB)

    # set brightness to 0 (black)
    zB = np.zeros((np.size(xB[0]), np.size(yB[0])))

    # plot on axes
    axes.pcolormesh(xB, yB, zB, cmap='gray')

    return

def setup_util_image(img_wd:int, img_ht:int):

    fig = plt.figure()
    ax = plt.axes()
    
    fig.add_axes(ax)

    ax.set_aspect('equal')
    ax.set_xlim(0, img_wd)
    ax.set_ylim(0, img_ht)

    return fig, ax

def setup_sim_image(img_wd:int, img_ht:int):

    fig = plt.figure(frameon=False)
    iw = img_wd/100
    ih = img_ht/100

    fig.set_size_inches(iw, ih, forward=False)
    ax = plt.Axes(fig, [0.,0.,1.,1.])

    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.add_axes(ax)

    gen_bkgd(ax, img_wd, img_ht)

    return fig, ax

def create_image(starlist:pd.DataFrame, img_wd:int, img_ht:int, outfp:str, ra:float, dec:float, roll:float, showPlot:bool=False):

    # setup figure    
    framed_fig, framed_ax = setup_util_image(img_wd=img_wd, img_ht=img_ht)
    sim_fig, sim_ax = setup_sim_image(img_wd=img_wd, img_ht=img_ht)

    ttl = '({}, {}, {})'.format(ra, dec, roll)
    framed_ax.set_title(ttl)

    # set figure name
    fname = '_{}_{}_{}.png'.format(ra, dec, roll)

    # store airy distribution
    xM, yM, zM = gen_airy()

    # iterate over stars
    for index, row in starlist.iterrows():
        print(f"{row['catalog_number']}: {row['IMG_X']} x {row['IMG_Y']}")

        plot_star(row, framed_ax, xM, yM, zM)
        plot_star(row, sim_ax, xM, yM, zM)
    
    mname = outfp + util_img + fname
    framed_fig.savefig(mname)

    pname = outfp + sim_img + fname
    sim_fig.savefig(pname)

    if showPlot:
        plt.close(sim_fig)
        plt.show()        

    return

def parse_arguments():
    parser = argparse.ArgumentParser(description='generates images of stars given coordinates and image dimensions')
    
    parser.add_argument('-n', type=int, help='Number of pictures to create; Default: 1', default=1)
    parser.add_argument('-fp', type=str, help='dataframe pickle filepath (Single Image); Default: Random', default=None)
    parser.add_argument('-cam', type=str, help='Set camera config filepath; Default: Alvium', default=constants.DEFAULT_ALVIUM)
    parser.add_argument('-dname', type=str, help='folder where images are going; Default: _StarTrackerTestImages/simImages/', default=constants.SIM_IMAGES)
    
    return parser.parse_args()

def generate_image(data=pd.DataFrame, camera:str=constants.DEFAULT_ALVIUM, direc:str=constants.SIM_IMAGES, ra:float=None,dec:float=None,roll:float=None):

    cfg = json.load(open(camera))

    img_wd = cfg['IMAGE_X']
    img_ht = cfg['IMAGE_Y']

    create_image(data, img_wd, img_ht, direc, ra, dec, roll, showPlot=True)

if __name__ == '__main__':
    args = parse_arguments()

    if args.fp is not None:
        data = pd.read_pickle(args.fp)
        generate_image(data=data, camera=args.cam, direc=args.dname)
    
    else:
        for i in range(args.n):
            ra = random.uniform(-180, 180)
            dec = random.uniform(-180, 180)
            roll = random.uniform(-180, 180)
            
            random_data = gP.generate_projection(ra=ra, dec=dec, roll=roll, cfg_fp=args.cam, plot=False)
            generate_image(random_data, args.cam, args.dname)           