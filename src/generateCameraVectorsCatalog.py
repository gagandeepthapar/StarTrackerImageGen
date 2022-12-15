import csv
import shutil
import json
import random

import argparse
import pandas as pd

import generateProjection
from catalogReader.catalogReader import CatalogReader
from catalogReader.headerFormats import YaleHeader
from catalogReader.star import YaleStar

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generates csv\'s of star ids and camera vectors for a projection of a given, or random pointing direction')
    parser.add_argument('--directory', type=str, help='CSV file directory out, used when generating multiple persepctives', default='simulated_projection_vectors')
    parser.add_argument('--cfg', type=str, help='cfg file path for simulated camera projection', default='camera_configs/simulated_alvium.json')
    parser.add_argument('--randomset', type=int, 
                        help='if greater than 0, produces a random set of N vector csv\'s with random pointing directions and puts them in --directory',
                        default=0)
    return parser.parse_args()


def write_projection_vector_csv(fname:str, starlist:pd.DataFrame, ra:float, dec:float, roll:float, cfg:dict):
    projection = generateProjection.project_onto_plane(starlist, ra, dec, 0, cfg)
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for indx, row in projection.iterrows(): 
            writer.writerow([row['catalog_number'], row['image_vector'][0], row['image_vector'][1], row['image_vector'][2]])

def main():
    args = parse_arguments()
    cfg = json.load(open(args.cfg))
    reader = CatalogReader(cfg['CATALOG_FILE'], YaleHeader(), YaleStar())
    reader.read_file()
    starlist = reader.starlist.star_frame
    if(args.randomset):
        for _ in range(args.randomset):
            ra = random.randint(0,360)
            dec = random.randint(0,360)
            roll = 0
            write_projection_vector_csv(f'{args.directory}/ra_dec_roll_{ra}_{dec}_{roll}.csv', starlist,ra, dec, roll, cfg)
        shutil.copy(args.cfg, f'{args.directory}/camera_config_used.json' )

if __name__ == '__main__':
    main()
