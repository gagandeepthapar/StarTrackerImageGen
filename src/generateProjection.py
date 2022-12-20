import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from catalogReader.catalogReader import CatalogReader
from catalogReader.headerFormats import YaleHeader
from catalogReader.star import YaleStar
from python_scripts import constants

PI = np.pi
MAXMAG = 5

def set_ECI_vector_x(row:pd.Series)->pd.Series:

    x = np.cos(row['right_ascension'])*np.cos(row['declination'])

    return x

def set_ECI_vector_y(row:pd.Series)->pd.Series:
    
    y = np.sin(row['right_ascension'])*np.cos(row['declination'])

    return y

def set_ECI_vector_z(row:pd.Series)->pd.Series:

    z = np.sin(row['declination'])

    return z

def remove_out_of_scope_stars(starlist:pd.DataFrame, ra:float, dec:float, fov:float)->pd.DataFrame:

    bs_x = np.cos(ra)*np.cos(dec)
    bs_y = np.sin(ra)*np.cos(dec)
    bs_z = np.sin(dec)

    dot_prod = np.dot(np.array([bs_x,
                                bs_y,
                                bs_z]),
                      np.array([starlist['ECI_X'],
                                starlist['ECI_Y'],
                                starlist['ECI_Z']]))

    star_fov = np.arccos(dot_prod)
    starlist['FOV'] = star_fov

    in_scope = starlist['FOV'] <= fov/2
    starlist = starlist[in_scope]
    
    return starlist

def get_rotation_accuracy(starlist:pd.DataFrame, bs:np.array, bsRot:np.array)->pd.DataFrame:

    bs_diff = np.zeros(len(starlist.index))
    star_diff = np.zeros(len(starlist.index))

    for i in range(len(starlist.index)):
        row = starlist.iloc[[i]]
        tilt_vec = np.array([row['ECI_X'],
                             row['ECI_Y'],
                             row['ECI_Z']]).flatten()
        flat_vec = np.array([row['CV_X'],
                             row['CV_Y'],
                             row['CV_Z']]).flatten()

        tilt = np.dot(bs, tilt_vec)/(np.linalg.norm(bs)*np.linalg.norm(tilt_vec))
        flat = np.dot(bsRot, flat_vec)/(np.linalg.norm(bsRot)*np.linalg.norm(flat_vec))

        bs_diff[i] = np.abs(flat-tilt)

        cur_max = 0
        for j in range(i+1,len(starlist.index)):
            new_row = starlist.iloc[[j]]
            
            tilt_new = np.array([new_row['ECI_X'],
                                new_row['ECI_Y'],
                                new_row['ECI_Z']]).flatten()

            flat_new = np.array([new_row['CV_X'],
                                new_row['CV_Y'],
                                new_row['CV_Z']]).flatten()

            tilt_diff = np.dot(tilt_vec, tilt_new)/(np.linalg.norm(tilt_vec) * np.linalg.norm(tilt_new))
            flat_diff = np.dot(flat_vec, flat_new)/(np.linalg.norm(flat_vec) * np.linalg.norm(flat_new))
            diff = np.abs(tilt_diff - flat_diff)
            
            if cur_max < diff:
                cur_max = diff
            
        star_diff[i] = cur_max

    starlist['Boresight_Error'] = bs_diff
    starlist['Max_Interstar_Error'] = star_diff

    return starlist

def set_camera_vectors(starlist:pd.DataFrame, ra:float, dec:float)->pd.DataFrame:

    bs_x = np.cos(ra)*np.cos(dec)
    bs_y = np.sin(ra)*np.cos(dec)
    bs_z = np.sin(dec)
    bs = np.array([bs_x, bs_y, bs_z])

    target = np.array([1, 0, 0])

    # quaternion rotation representation
    e = np.cross(target, bs)
    d = np.dot(target, bs)
    n = d + np.sqrt(d*d + np.dot(e,e))

    # normalize quaternion
    q = np.array([e[0], e[1], e[2], n])
    q = q/np.linalg.norm(q)

    skew = lambda q: np.array([0, -1*q[2], q[1], q[2], 0, -1*q[0], -1*q[1], q[0], 0]).reshape((3,3))
    rot_matr = (2*q[3]**2 - 1)*np.eye(3) + 2*np.matmul(np.array(q[0:3]).reshape((3,1)), np.array(q[0:3]).reshape((1,3))) - 2*q[3]*skew(q[0:3])
    
    col_x = np.zeros([len(starlist.index)])
    col_y = np.zeros([len(starlist.index)])
    col_z = np.zeros([len(starlist.index)])

    for i in range(len(starlist.index)):
        row = starlist.iloc[[i]]
        mult = np.matmul(rot_matr, np.array([row['ECI_X'],
                                                  row['ECI_Y'],
                                                  row['ECI_Z']]).reshape((3,1)))

        col_x[i] = mult[0]
        col_y[i] = mult[1]
        col_z[i] = mult[2]

    starlist['CV_X'] = col_x
    starlist['CV_Y'] = col_y
    starlist['CV_Z'] = col_z

    starlist = get_rotation_accuracy(starlist, bs, target)

    return starlist

def get_roll_accuracy(starlist:pd.DataFrame, bs:np.array, bsRot:np.array)->pd.DataFrame:

    bs_diff = np.zeros(len(starlist.index))

    for i in range(len(starlist.index)):
        row = starlist.iloc[[i]]
        tilt_star_vec = np.array([row['ECI_X'],
                                  row['ECI_Y'],
                                  row['ECI_Z']])
        
        flat_star_vec = np.array([row['CV_X'],
                                  row['CV_Y'],
                                  row['CV_Z']])

        

    return starlist

def update_roll_coordinates(starlist:pd.DataFrame, roll:float)->pd.DataFrame:

    cvx = np.zeros(len(starlist.index))
    cvy = np.zeros(len(starlist.index))

    rot = np.array([np.cos(roll), -1*np.sin(roll), np.sin(roll), np.cos(roll)]).reshape((2,2))

    for i in range(len(starlist.index)):
        row = starlist.iloc[[i]]
        x = row['CV_Y']
        y = row['CV_Z']

        cvx[i] = x*np.cos(roll) - y*np.sin(roll)
        cvy[i] = x*np.sin(roll) + y*np.cos(roll)

    starlist['CV_X_ROLL'] = cvx
    starlist['CV_Y_ROLL'] = cvy

    return starlist

def set_image_coordinates(starlist:pd.DataFrame, img_wd:int, img_ht:int, fov:float)->pd.DataFrame:

    center_wd = img_wd/2
    center_ht = img_ht/2

    imgx = np.zeros(len(starlist.index))
    imgy = np.zeros(len(starlist.index))

    for i in range(len(starlist.index)):
        row = starlist.iloc[[i]]
        # vec = np.array([-1*row['CV_X_ROLL'],
        #                 row['CV_Y_ROLL']])
        # normvec = vec/np.linalg.norm(vec)

        rad = np.sqrt(center_ht**2 + center_wd**2)
        
        imgx[i] = -1*row['CV_X_ROLL']*rad/(0.5*fov) + center_wd
        imgy[i] = row['CV_Y_ROLL']*rad/(0.5*fov) + center_ht

    starlist['IMG_X'] = imgx
    starlist['IMG_Y'] = imgy

    return starlist

def remove_out_of_image_stars(starlist:pd.DataFrame, img_wd:int, img_ht:int)->pd.DataFrame:

    check_x = starlist[(0 < starlist['IMG_X']) & (starlist['IMG_X'] < img_wd)]
    in_image = check_x[(0 < check_x['IMG_Y']) & (check_x['IMG_Y'] < img_ht)]

    return in_image

def driver(starlist:pd.DataFrame, ra:float=0, dec:float=0, roll:float=0, cfg:dict=None, showPlot:bool=False, saveFrame:str=None)->pd.DataFrame:
    """main code to generate dataframe with star image coords

    Args:
        starlist (pd.DataFrame): dataframe of all stars in catalog
        ra (float, optional): right ascension angle in degrees. Defaults to 0
        dec (float, optional): declination angle in degrees. Defaults to 0.
        roll (float, optional): roll angle in degrees. Defaults to 0.
        cfg (dict, optional): camera configuration file containing parameters. Defaults to default alvium properties.
        showPlot (bool, optional): bool to show analysis plot. Defaults to False.

    Returns:
        pd.DataFrame: updated starlist with image coordinates
    """

    # copy list and update parameters
    fstars = starlist.copy()
    
    ra = ra * PI/180
    dec = dec * PI/180
    roll = roll * PI/180

    # extract config file information and convert to rad where applicable
    camera_fov = cfg['FOV'] * PI/180
    img_wd = cfg['IMAGE_X']
    img_ht = cfg['IMAGE_Y']

    print(f'\tImage Size: {img_wd}x{img_ht}')
    print(f'\tFOV: {camera_fov*180/PI}')
    print(f'\tInitial Starlist Length: {len(fstars.index)}\n')

    # set ECI vectors
    fstars['ECI_X'] = fstars.apply(set_ECI_vector_x, axis=1)
    fstars['ECI_Y'] = fstars.apply(set_ECI_vector_y, axis=1)
    fstars['ECI_Z'] = fstars.apply(set_ECI_vector_z, axis=1)

    # remove out of FOV stars
    fstars = remove_out_of_scope_stars(fstars, ra, dec, camera_fov)
    
    # set CV vectors
    fstars = set_camera_vectors(fstars, ra, dec)

    # update camera coordinates with roll
    fstars = update_roll_coordinates(fstars, roll)

    # determine image coordinates
    fstars = set_image_coordinates(fstars, img_wd, img_ht, camera_fov)
    
    if showPlot:
        plt = plot_sphere(fstars, ra, dec, roll, camera_fov, img_wd, img_ht)

    # remove out of image stars
    fstars = remove_out_of_image_stars(fstars, img_wd, img_ht)

    # print starlist information; store information to csv
    print(f'\tFinal Starlist Length: {len(fstars.index)}\n')
    print(fstars.to_string())

    
    if showPlot:
        plt.show()
    
    if saveFrame is not None:
        if saveFrame[-4:] == '.pkl':
            saveFrame = saveFrame[:-4]
        saveFrame = saveFrame + '_{}_{}_{}.pkl'.format(ra, dec, roll)
        fstars.to_pickle(saveFrame)

        fname = saveFrame + '_{}_{}_{}.csv'.format(ra, dec, roll)
        fstars.to_csv(fname, sep='\t', encoding='utf-8', header='true')

    return fstars

def plot_sphere(starlist:pd.DataFrame, ra:float, dec:float, roll:float, fov:float, img_wd:int, img_ht:int):

    def get_cone(start:np.array, end:np.array, rad:float, size:int)->np.array:

        def rotm(vec1:np.array, vec2:np.array)->np.array:
                a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
                v = np.cross(a, b)
                c = np.dot(a, b)
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
                return rotation_matrix

        h = np.linalg.norm(end-start)
        t = np.linspace(0, 2*np.pi, num=size)
        x = rad*np.cos(t)
        y = rad*np.sin(t)
        z = h*np.ones(size)

        R = rotm(np.array([0, 0, 1]), end-start)

        xr = np.zeros(size)
        yr = np.zeros(size)
        zr = np.zeros(size)

        for i in range(size):
            r = np.matmul(R, np.array([x[i], y[i], z[i]]))
            xr[i] = r[0]
            yr[i] = r[1]
            zr[i] = r[2]
        
        filler = np.zeros(size)
        X = np.array([filler, xr]).reshape((2,size))
        Y = np.array([filler, yr]).reshape((2, size))
        Z = np.array([filler, zr]).reshape((2, size))
        
        return np.array([X, Y, Z])

    radec2vec = lambda ra, dec: np.array([np.cos(ra)*np.cos(dec),
                                            np.sin(ra)*np.cos(dec),
                                            np.sin(dec)])

    bs = radec2vec(ra, dec)
    bs_x = bs[0]
    bs_y = bs[1]
    bs_z = bs[2]

    abs_cone = get_cone(np.array([0,0,0]), bs, np.tan(fov/2), 25)
    rel_cone = get_cone(np.array([0, 0, 0]), np.array([1,0,0]), np.tan(fov/2), 25)

    fig = plt.figure()

    # 3D PLOT
    ax = fig.add_subplot(2, 2, (1,3), projection='3d')

    r = 1
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.3)   # sphere

    # absolute/ECI representation
    ax.scatter3D(bs_x, bs_y, bs_z, color='red', marker='x') # absolute boresight
    ab = ax.plot3D(np.array([0, bs_x]), np.array([0, bs_y]), np.array([0, bs_z]), color='red') # absolute boresight
    ac = ax.plot_surface(abs_cone[0], abs_cone[1], abs_cone[2], color='red', alpha=0.5)  # absolute cone

    # relative representation
    ax.scatter3D(1, 0, 0, color='blue', marker='x')
    cb = ax.plot3D(np.array([0, 1]), np.array([0, 0]), np.array([0, 0]), color='blue')
    ax.plot_surface(rel_cone[0], rel_cone[1], rel_cone[2], color='blue', alpha=0.5) 

    title = 'Celestial Sphere\nPointing at ({:.2f}\u00b0, {:.2f}\u00b0, {:.2f}\u00b0)'.format(ra*180/PI, dec*180/PI, roll*180/PI)

    ax.axis('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i, row in starlist.iterrows():
        ax.scatter3D(row['ECI_X'], row['ECI_Y'], row['ECI_Z'], color='red', marker='*') # absolute star
        ax.scatter3D(row['CV_X'], row['CV_Y'], row['CV_Z'], color='blue', marker='*') # absolute star

    # 2D Plot
    ax = fig.add_subplot(2,2,2)
    
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = fov/2
    x = radius * np.cos( theta )
    y = radius * np.sin( theta )

    c = img_wd/img_ht
    rectY = -1*np.sqrt(fov**2 / (4*(c**2 +1)))
    rectX = c*rectY
    rect = Rectangle((rectX,rectY), 2*np.abs(rectX), 2*np.abs(rectY), color='black', fill=False)

    ax.plot(x, y, '--b')
    ax.scatter(0, 0, color='blue', marker='x')
    ax.add_patch(rect)

    ct = 0
    for i, row in starlist.iterrows():
        ax.scatter(-1*row['CV_Y'],row['CV_Z'], color='red', marker='*', alpha=0.1)
        ax.scatter(-1*row['CV_X_ROLL'], row['CV_Y_ROLL'], color='black', marker='*',)
        
        label = 'ID {:d}'.format(int(row['catalog_number']))
        ax.annotate(label, (-1*row['CV_X_ROLL'], row['CV_Y_ROLL']+0.01), fontsize=7)

        ax.plot([-1*row['CV_Y'], -1*row['CV_X_ROLL']], [row['CV_Z'], row['CV_Y_ROLL']], alpha=0.1, color='black')

        if rectX <= row['CV_X_ROLL'] and row['CV_X_ROLL'] <= rectX+(2*np.abs(rectX)):
            if rectY <= row['CV_Y_ROLL'] and row['CV_Y_ROLL'] <= rectY+(2*np.abs(rectY)):
                ct += 1

    ax.axis('equal')
    ax.set_xlabel('Relative Right Ascension, deg')
    ax.set_ylabel('Relative Declination, deg')
    # ax.legend()

    title = '{} of {} Stars Captured in Image'.format(ct, len(starlist.index))
    ax.set_title(title)

    # simulated image test
    ax = fig.add_subplot(2,2,4)

    rect = Rectangle((0,0), img_wd, img_ht, color='black')
   
    ax.add_patch(rect)
    ax.scatter(img_wd/2, img_ht/2, color='red', marker='x')

    check_x = lambda x: 0 <= x and x <= img_wd
    check_y = lambda y: 0 <= y and y <= img_ht

    for i, row in starlist.iterrows():
        x = row['IMG_X']
        y = row['IMG_Y']
        if check_x(x) and check_y(y):
            ax.scatter(x, y, color='white', marker='.')
            label = 'ID {:d}'.format(int(row['catalog_number']))

            ax.annotate(label, (x, y+40), fontsize=7, color='red') 

    ax.axis('equal')
    # ax.legend()
    ax.set_xlabel('IMAGE X [Pixels]')
    ax.set_ylabel('IMAGE Y [Pixels]')
    delta = 10
    ax.set_xlim(0 - delta, img_wd + delta)
    ax.set_ylim(0 - delta, img_ht + delta)
    return plt

def parse_arguments()->argparse.Namespace:

    parser = argparse.ArgumentParser(description="Set camera and simulation properties")

    parser.add_argument('-fp', help='Set camera config filepath; Default: Alvium', type=str, default=constants.DEFAULT_ALVIUM)
    parser.add_argument('-ra', help='Set Right Ascension [deg]; Default: Random [-180, 180]', type=float, default=random.uniform(-180, 180))
    parser.add_argument('-dec', help='Set Declination [dec]; Default: Random [-180, 180]', type=float, default=random.uniform(-180, 180))
    parser.add_argument('-roll', help='Set Roll Angle [deg]; Default: Random [-180, 180]', type=float, default=random.uniform(-180, 180))    
    parser.add_argument('-m', help='Set Min Magnitude; Default: Camera Specific', type=float, default=None)
    parser.add_argument('-p', help='Show/Hide Plot (0/1)', type=int, default=1)
    parser.add_argument('-s', help='Filepath to save dataframe (opt); Default: N/A', type=str, default=None)

    args = parser.parse_args()

    return args

def generate_projection(ra:float=random.uniform(-180,180),
                        dec:float=random.uniform(-180, 180),
                        roll:float=random.uniform(-180, 180),
                        cfg_fp:str=constants.DEFAULT_ALVIUM,
                        catpkl_fp:str=constants.YBSC_PATH,
                        camera_mag:float=None,
                        plot:bool=False,
                        save:str=None)->pd.DataFrame:

    # prepare camera config file
    cfg = json.load(open(cfg_fp))    
    star_frame = pd.read_pickle(catpkl_fp)

    # remove dim stars (to be moved up the process later)
    if camera_mag is None:
        camera_mag = cfg['MAX_MAG']
    star_frame = star_frame[star_frame['v_magnitude'] <= camera_mag]

    # start projection process
    print(f"Generating Projection Dataframe with parameters:\n\tRight Ascension = {ra}\n\tDeclination = {dec}\n\tRoll = {roll}\n\tPlotting: {plot}")
    fstars = driver(star_frame, ra, dec, roll, cfg, showPlot=plot, saveFrame=save)

    return fstars

if __name__ == '__main__':
    args = parse_arguments()

    generate_projection(ra=args.ra,
         dec=args.dec,
         roll=args.roll,
         cfg_fp=args.fp,
         camera_mag= args.m,
         plot=bool(args.p),
         save=args.s)

