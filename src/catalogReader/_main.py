import sys
from imageGenerator import CelestialSphere
from catalogReader import CatalogReader
from star import YaleStar
from starlist import StarList
import pandas as pd
import numpy as np

from headerFormats import YaleHeader
sys.path.append('../../')

def main():
    b = '../../BSC5'
    reader = CatalogReader(b, YaleHeader(), YaleStar())
    reader.read_file()
    starlist = reader.starlist

    print(starlist.star_frame)

    # celestial_sphere = CelestialSphere()
    # celestial_sphere.plot_boresight(np.array([1,0,0]))

    # # print(type(starlist.star_frame.iloc[456]))

    # celestial_sphere.plot_multiple_stars(starlist=starlist.star_frame)

    # celestial_sphere.save_image('entry.png')


if __name__ == '__main__':
    main()